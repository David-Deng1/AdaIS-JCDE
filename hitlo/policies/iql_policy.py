"""
hitlo.policies.iql_policy
============================
Implicit Q-Learning (IQL) policy with structured OSV action space.
Extracted from app_v11_iqlfix.py (RunningNorm, IQLPolicy classes).
"""

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


class RunningNorm:
    def __init__(self, eps: float = 1e-6):
        self.mean = None
        self.var = None
        self.count = 0
        self.eps = float(eps)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # 防止 NaN/Inf 污染归一化统计（否则 IQL actor logits 可能变 NaN）
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e6, 1e6)
        n = int(x.shape[0])
        if n <= 0:
            return
        if self.mean is None:
            self.mean = np.nan_to_num(np.mean(x, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
            self.var = np.nan_to_num(np.var(x, axis=0), nan=1.0, posinf=1.0, neginf=1.0)
            self.var = np.maximum(self.var, self.eps)
            self.count = n
            return

        mean = np.nan_to_num(self.mean, nan=0.0, posinf=0.0, neginf=0.0)
        var = np.nan_to_num(self.var, nan=1.0, posinf=1.0, neginf=1.0)
        var = np.maximum(var, self.eps)
        count = max(1, int(self.count))

        new_mean = np.nan_to_num(np.mean(x, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        new_var = np.nan_to_num(np.var(x, axis=0), nan=1.0, posinf=1.0, neginf=1.0)
        new_var = np.maximum(new_var, self.eps)
        total = count + n
        delta = new_mean - mean
        mean2 = mean + delta * (n / max(total, 1))
        M2 = var * count + new_var * n + (delta ** 2) * count * n / max(total, 1)
        var2 = M2 / max(total, 1)

        self.mean = np.nan_to_num(mean2, nan=0.0, posinf=0.0, neginf=0.0)
        self.var = np.nan_to_num(var2, nan=1.0, posinf=1.0, neginf=1.0)
        self.var = np.maximum(self.var, self.eps)
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1e6, 1e6)
        if self.mean is None or self.var is None:
            return x
        mean = np.nan_to_num(self.mean, nan=0.0, posinf=0.0, neginf=0.0)
        var = np.nan_to_num(self.var, nan=1.0, posinf=1.0, neginf=1.0)
        var = np.maximum(var, self.eps)
        out = (x - mean) / np.sqrt(var + self.eps)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class IQLPolicy:
    """IQL implementation (Eq.(13)~(16), Appendix A.0.10).

    Notes (v11+):
      - The OSV action vector is *structured*:
          [bounded_cont(≈12 dims), alg_logits(N dims), op_logits(OP_DIM dims)]
        where bounded_cont is in [-1,1] (tanh-squashed), alg_logits is in [-L,L]
        (logits for softmax scheduling), and op is categorical (argmax / Categorical).
      - Earlier versions treated *all* continuous dims as [-1,1], which
        unintentionally compressed algorithm-logit magnitudes and weakened
        portfolio scheduling. Here we model alg_logits with a separate tanh-scale.

    This is a legitimate bug-fix / modeling correction (not a "result tweak"):
    it makes the learned controller's action parametrization match the runtime OSV semantics.
    """
    def __init__(
        self,
        action_dim: int,
        op_dim: int,
        alg_dim: Optional[int] = None,
        seed: int = 0,
        expectile: float = 0.7,
        beta: float = 3.0,
        gamma: float = 0.98,
        *,
        fixed_bound_dim: int = 12,
        alg_logit_scale: float = 5.0,
        lr: float = 3e-4,
    ):
        self.action_dim = int(action_dim)
        self.op_dim = int(op_dim)

        # Infer alg_dim if not provided (OSV structure is fixed: pref(2)+rob(2)+param(4)+model(4)=12)
        if alg_dim is None:
            inferred = int(self.action_dim - self.op_dim - int(fixed_bound_dim))
            alg_dim = max(0, inferred)
        self.alg_dim = int(alg_dim)

        self.cont_dim = int(self.action_dim - self.op_dim)
        self.bound_dim = int(self.cont_dim - self.alg_dim)
        if self.bound_dim < 0:
            raise ValueError(f"[IQLPolicy] Invalid dims: action_dim={self.action_dim}, op_dim={self.op_dim}, alg_dim={self.alg_dim}")

        self.alg_logit_scale = float(alg_logit_scale)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.expectile = float(expectile)
        self.beta = float(beta)
        self.gamma = float(gamma)
        # Optimizer learning rate (configurable; default keeps backward-compat).
        self.lr = float(lr)
        self.device = torch.device("cpu") if HAS_TORCH else None
        self.torch_gen = None
        if HAS_TORCH:
            try:
                self.torch_gen = torch.Generator(device="cpu")
                self.torch_gen.manual_seed(self.seed)
            except Exception:
                self.torch_gen = None

        self.state_norm = RunningNorm()

        # Elementwise scales for tanh-squashed continuous part:
        # - bounded channels: scale 1.0
        # - algorithm logits: scale alg_logit_scale (default 5.0, matching OSV.sample clip)
        self._cont_scale = np.ones(self.cont_dim, dtype=np.float32)
        if self.alg_dim > 0:
            self._cont_scale[self.bound_dim:] = float(self.alg_logit_scale)

        self._state_dim: Optional[int] = None
        self.q1 = None
        self.q2 = None
        self.v = None
        self.actor_base = None
        self.actor_mean = None
        self.actor_logstd = None
        self.actor_op = None
        self.q1_tgt = None
        self.q2_tgt = None
        self.v_tgt = None
        self.opt_q = None
        self.opt_v = None
        self.opt_actor = None

        self.buffer: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
        self.min_batch = 256
        self.train_every = 1
        self.polyak = 0.995

        self.max_grad_norm = 10.0
        self.use_tanh_correction = True
        self._ckpt_loaded = False

    @property
    def state_dim(self) -> int:
        return int(self._state_dim or 0)

    @property
    def cont_scale(self) -> np.ndarray:
        return np.asarray(self._cont_scale, dtype=np.float32)

    def _ensure_models(self, state_dim: int):
        if self.q1 is not None:
            return
        self._state_dim = int(state_dim)
        # Q takes: [s, a_cont_scaled, a_op_onehot]
        in_q = self._state_dim + self.cont_dim + self.op_dim
        hid = 256

        # Critics (Double-Q)
        self.q1 = nn.Sequential(
            nn.Linear(in_q, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_q, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )

        # Value
        self.v = nn.Sequential(
            nn.Linear(self._state_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )

        # Actor (Gaussian for cont, categorical for op)
        self.actor_base = nn.Sequential(
            nn.Linear(self._state_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hid, self.cont_dim)
        self.actor_logstd = nn.Linear(hid, self.cont_dim)
        self.actor_op = nn.Linear(hid, self.op_dim)

        # Targets
        self.q1_tgt = copy.deepcopy(self.q1)
        self.q2_tgt = copy.deepcopy(self.q2)
        self.v_tgt = copy.deepcopy(self.v)

        lr = float(getattr(self, "lr", 3e-4))
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=lr)
        self.opt_actor = torch.optim.Adam(
            list(self.actor_base.parameters()) + list(self.actor_mean.parameters()) +
            list(self.actor_logstd.parameters()) + list(self.actor_op.parameters()),
            lr=lr
        )

    def push_transition(self, s: np.ndarray, u: np.ndarray, r: float, s_next: np.ndarray, extra_info=None, **kwargs):
        s = np.asarray(s, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        s_next = np.asarray(s_next, dtype=float).reshape(-1)
        if s.size == 0 or u.size < self.action_dim or s_next.size == 0:
            return
        # sanitize to prevent NaN/Inf poisoning IQL (causes NaN logits -> Categorical crash)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        s_next = np.nan_to_num(s_next, nan=0.0, posinf=0.0, neginf=0.0)
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        s = np.clip(s, -1e6, 1e6)
        s_next = np.clip(s_next, -1e6, 1e6)
        try:
            r_f = float(r)
        except Exception:
            r_f = 0.0
        if not np.isfinite(r_f):
            r_f = 0.0
        self.state_norm.update(s)
        # Store only the first action_dim dims (defensive)
        self.buffer.append((s, u[: self.action_dim].copy(), r_f, s_next))
        if len(self.buffer) > 200000:
            self.buffer = self.buffer[-200000:]

    def _cont_to_scaled(self, u_cont_raw: np.ndarray) -> np.ndarray:
        """Map raw cont action to tanh-domain (scaled) in (-1,1) for training."""
        u = np.asarray(u_cont_raw, dtype=np.float32).reshape(-1)
        if u.size != self.cont_dim:
            u = u[: self.cont_dim]
        sc = self.cont_scale
        sc = np.maximum(sc, 1e-6)
        x = u / sc
        x = np.clip(x, -0.999, 0.999)
        return x

    def act(self, s: np.ndarray, *, deterministic: bool = False) -> Tuple[np.ndarray, int, np.ndarray]:
        """Return (u_cont_raw, op_idx, op_logits_raw).

        Hotfix: 防止 actor 输出 NaN/Inf 导致 torch.distributions.Categorical 直接报错。
        根因通常是：状态/奖励中出现 NaN -> state_norm/网络参数被污染。
        这里做两层保护：输入/输出 nan_to_num + 失效时回退 NOOP。
        """
        if not HAS_TORCH:
            u_cont = np.zeros(self.cont_dim, dtype=float)
            op_logits = np.zeros(self.op_dim, dtype=float)
            return u_cont, 0, op_logits

        s = np.asarray(s, dtype=float).reshape(1, -1)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        s = np.clip(s, -1e6, 1e6)
        self._ensure_models(s.shape[1])

        s_n = self.state_norm.normalize(s).astype(np.float32)
        s_n = np.nan_to_num(s_n, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        S = torch.from_numpy(s_n).float()
        S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            h = self.actor_base(S)
            h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

            mu = self.actor_mean(h)
            mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
            log_std = self.actor_logstd(h)
            log_std = torch.nan_to_num(log_std, nan=-3.0, posinf=2.0, neginf=-5.0)
            log_std = torch.clamp(log_std, -5.0, 2.0)
            std = torch.exp(log_std)
            std = torch.nan_to_num(std, nan=1.0, posinf=10.0, neginf=1.0)

            if deterministic:
                z = mu
            else:
                try:
                    eps = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=self.torch_gen)
                except Exception:
                    eps = torch.randn_like(mu)
                z = mu + std * eps
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            a_scaled = torch.tanh(z)
            sc = torch.from_numpy(self.cont_scale.reshape(1, -1)).float()
            sc = torch.nan_to_num(sc, nan=1.0, posinf=10.0, neginf=1.0)
            a_raw = a_scaled * sc

            op_logits = self.actor_op(h)
            op_logits = torch.nan_to_num(op_logits, nan=0.0, posinf=0.0, neginf=0.0)
            op_logits = torch.clamp(op_logits, -5.0, 5.0)
            if not torch.isfinite(op_logits).all():
                op_logits = torch.zeros_like(op_logits)
            if deterministic:
                op_idx = int(torch.argmax(op_logits, dim=-1).item())
            else:
                try:
                    probs = torch.softmax(op_logits, dim=-1)
                    op_idx = int(torch.multinomial(probs, num_samples=1, generator=self.torch_gen).item())
                except Exception:
                    dist_op = torch.distributions.Categorical(logits=op_logits)
                    op_idx = int(dist_op.sample().item())

        u_cont = a_raw.cpu().numpy().reshape(-1).astype(float)
        u_cont = np.nan_to_num(u_cont, nan=0.0, posinf=0.0, neginf=0.0)
        op_logits_np = op_logits.cpu().numpy().reshape(-1).astype(float)
        op_logits_np = np.nan_to_num(op_logits_np, nan=0.0, posinf=0.0, neginf=0.0)
        op_logits_np = np.clip(op_logits_np, -5.0, 5.0)
        return u_cont, op_idx, op_logits_np

    def q_min(self, s: np.ndarray, u_exec: np.ndarray) -> float:
        """Compute min(Q1,Q2) for a given executed OSV action vector.

        Notes:
        - u_exec is expected to be the *executed* OSV vector after Π_safe (same convention as offline dataset).
        - Continuous part is scaled back into (-1,1) domain using cont_scale; op part uses argmax -> onehot.
        """
        if not HAS_TORCH:
            return float("-inf")
        s = np.asarray(s, dtype=float).reshape(1, -1)
        u = np.asarray(u_exec, dtype=float).reshape(1, -1)
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        self._ensure_models(s.shape[1])
        # normalize state
        s_n = self.state_norm.normalize(s).astype(np.float32)
        S = torch.from_numpy(s_n).float()
        # split action
        cont_dim = int(self.cont_dim)
        Uc_raw = u[:, :cont_dim].astype(np.float32)
        Uop_logits = u[:, cont_dim:cont_dim + int(self.op_dim)].astype(np.float32) if self.op_dim > 0 else np.zeros((1, 0), dtype=np.float32)

        sc = self.cont_scale.reshape(1, -1).astype(np.float32)
        Uc_scaled = np.clip(Uc_raw / np.maximum(1e-6, sc), -0.999, 0.999)
        Uc_t = torch.from_numpy(Uc_scaled).float()

        if self.op_dim > 0:
            op_idx = int(np.argmax(Uop_logits, axis=1)[0])
            Uo = np.zeros((1, int(self.op_dim)), dtype=np.float32)
            Uo[0, op_idx] = 1.0
            Uo_t = torch.from_numpy(Uo).float()
            q_in = torch.cat([S, Uc_t, Uo_t], dim=1)
        else:
            q_in = torch.cat([S, Uc_t], dim=1)

        with torch.no_grad():
            q1 = self.q1(q_in)
            q2 = self.q2(q_in)
            q = torch.minimum(q1, q2)
        return float(q.item())

    def maybe_train(self, step: int, force: bool = False, **kwargs) -> Optional[Dict[str, float]]:
        if (len(self.buffer) < self.min_batch) or ((not force) and (step % self.train_every != 0)) or (not HAS_TORCH):
            return None

        # Sample minibatch
        idx = self.rng.integers(0, len(self.buffer), size=self.min_batch)
        batch = [self.buffer[i] for i in idx]
        S_np = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
        U_np = np.stack([b[1] for b in batch], axis=0).astype(np.float32)
        R_np = np.array([b[2] for b in batch], dtype=np.float32).reshape(-1, 1)
        S2_np = np.stack([b[3] for b in batch], axis=0).astype(np.float32)

        # Drop / sanitize non-finite samples (defensive, avoids NaN losses/logits)
        mask = (np.isfinite(S_np).all(axis=1) & np.isfinite(U_np).all(axis=1) &
                np.isfinite(R_np).all(axis=1) & np.isfinite(S2_np).all(axis=1))
        if not np.all(mask):
            S_np = S_np[mask]
            U_np = U_np[mask]
            R_np = R_np[mask]
            S2_np = S2_np[mask]
        if S_np.shape[0] < 8:
            return None


        self._ensure_models(S_np.shape[1])

        # Normalize states
        S_n = self.state_norm.normalize(S_np).astype(np.float32)
        S2_n = self.state_norm.normalize(S2_np).astype(np.float32)

        # Split actions
        Uc_raw = U_np[:, : self.cont_dim]
        Uo_raw = U_np[:, self.cont_dim : self.cont_dim + self.op_dim]

        # Cont: map to tanh-domain (scaled) in (-1,1)
        sc = self.cont_scale.reshape(1, -1).astype(np.float32)
        Uc_scaled = Uc_raw / np.maximum(sc, 1e-6)
        Uc_scaled = np.clip(Uc_scaled, -0.999, 0.999).astype(np.float32)

        # Op: categorical -> one-hot for critic input
        op_idx_np = np.argmax(Uo_raw, axis=1).astype(np.int64)
        Uo_oh = np.zeros((Uo_raw.shape[0], self.op_dim), dtype=np.float32)
        Uo_oh[np.arange(Uo_raw.shape[0]), op_idx_np] = 1.0

        # Torch tensors
        S_t = torch.from_numpy(S_n).float()
        S2_t = torch.from_numpy(S2_n).float()
        Uc_t = torch.from_numpy(Uc_scaled).float()
        Uo_t = torch.from_numpy(Uo_oh).float()
        R_t = torch.from_numpy(R_np).float()

        S_t = torch.nan_to_num(S_t, nan=0.0, posinf=0.0, neginf=0.0)
        S2_t = torch.nan_to_num(S2_t, nan=0.0, posinf=0.0, neginf=0.0)
        Uc_t = torch.nan_to_num(Uc_t, nan=0.0, posinf=0.0, neginf=0.0)
        Uo_t = torch.nan_to_num(Uo_t, nan=0.0, posinf=0.0, neginf=0.0)
        R_t = torch.nan_to_num(R_t, nan=0.0, posinf=0.0, neginf=0.0)


        # ---------- Critic update ----------
        with torch.no_grad():
            v2 = self.v_tgt(S2_t)
            y = R_t + self.gamma * v2

        q_in = torch.cat([S_t, Uc_t, Uo_t], dim=1)
        q1 = self.q1(q_in)
        q2 = self.q2(q_in)
        q_loss = ((q1 - y).pow(2).mean() + (q2 - y).pow(2).mean()) * 0.5

        if not torch.isfinite(q_loss):
            return None

        self.opt_q.zero_grad()
        q_loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), self.max_grad_norm)
        self.opt_q.step()

        # ---------- Value update (expectile regression) ----------
        with torch.no_grad():
            q_min = torch.min(self.q1(q_in), self.q2(q_in))
        v = self.v(S_t)
        diff = q_min - v
        w = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        v_loss = (w * diff.pow(2)).mean()

        if not torch.isfinite(v_loss):
            return None

        self.opt_v.zero_grad()
        v_loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.v.parameters(), self.max_grad_norm)
        self.opt_v.step()

        # ---------- Actor update (advantage-weighted regression) ----------
        with torch.no_grad():
            q_min_det = torch.min(self.q1(q_in), self.q2(q_in))
            v_det = self.v(S_t)
            adv = (q_min_det - v_det).clamp(-10.0, 10.0)
            weights = torch.exp(self.beta * adv).clamp(1e-4, 100.0)

        h = self.actor_base(S_t)
        mu = self.actor_mean(h)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        log_std = self.actor_logstd(h)
        log_std = torch.nan_to_num(log_std, nan=-3.0, posinf=2.0, neginf=-5.0)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        std = torch.nan_to_num(std, nan=1.0, posinf=10.0, neginf=1.0)
        dist = torch.distributions.Normal(mu, std)

        # Inverse tanh for cont in tanh-domain
        u_raw = torch.nan_to_num(Uc_t, nan=0.0, posinf=0.0, neginf=0.0)
        u_raw = torch.clamp(u_raw, -0.999, 0.999)
        u_atanh = 0.5 * (torch.log1p(u_raw) - torch.log1p(-u_raw))  # atanh
        logp_cont = dist.log_prob(u_atanh).sum(dim=1, keepdim=True)

        if self.use_tanh_correction:
            # Correction for a_raw = scale * tanh(z):
            # log|da/dz| = log(scale) + log(1 - tanh(z)^2) = log(scale) + log(1 - u^2)
            sc_t = torch.from_numpy(self.cont_scale.reshape(1, -1)).float()
            jac = torch.log(torch.clamp(sc_t, min=1e-6)) + torch.log(torch.clamp(1.0 - u_raw.pow(2), min=1e-6))
            logp_cont = logp_cont - jac.sum(dim=1, keepdim=True)

        # Op categorical log-prob
        op_logits = self.actor_op(h)
        op_logits = torch.nan_to_num(op_logits, nan=0.0, posinf=0.0, neginf=0.0)
        op_logits = torch.clamp(op_logits, -5.0, 5.0)
        dist_op = torch.distributions.Categorical(logits=op_logits)
        op_idx_t = torch.from_numpy(op_idx_np).long()
        logp_op = dist_op.log_prob(op_idx_t).reshape(-1, 1)

        logp = logp_cont + logp_op
        actor_loss = -(weights * logp).mean()

        if not torch.isfinite(actor_loss):
            return None

        self.opt_actor.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.actor_base.parameters()) + list(self.actor_mean.parameters()) +
                list(self.actor_logstd.parameters()) + list(self.actor_op.parameters()),
                self.max_grad_norm
            )
        self.opt_actor.step()

        # ---------- Polyak update targets ----------
        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_tgt.parameters()):
                pt.data.mul_(self.polyak).add_((1.0 - self.polyak) * p.data)
            for p, pt in zip(self.q2.parameters(), self.q2_tgt.parameters()):
                pt.data.mul_(self.polyak).add_((1.0 - self.polyak) * p.data)
            for p, pt in zip(self.v.parameters(), self.v_tgt.parameters()):
                pt.data.mul_(self.polyak).add_((1.0 - self.polyak) * p.data)

        return {
            "q_loss": float(q_loss.item()),
            "v_loss": float(v_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "adv_mean": float(torch.mean(adv).item()),
            "w_mean": float(torch.mean(weights).item()),
            "batch_n": int(self.min_batch),
        }

    def save(self, path: str):
        if not HAS_TORCH:
            return
        payload = {
            "action_dim": self.action_dim,
            "op_dim": self.op_dim,
            "alg_dim": self.alg_dim,
            "bound_dim": self.bound_dim,
            "alg_logit_scale": self.alg_logit_scale,
            "expectile": self.expectile,
            "beta": self.beta,
            "gamma": self.gamma,
            "cont_scale": self.cont_scale.tolist(),
            "state_norm": {"mean": None if self.state_norm.mean is None else self.state_norm.mean.tolist(),
                           "var": None if self.state_norm.var is None else self.state_norm.var.tolist(),
                           "count": int(self.state_norm.count)},
            "q1": None if self.q1 is None else self.q1.state_dict(),
            "q2": None if self.q2 is None else self.q2.state_dict(),
            "q1_tgt": None if self.q1_tgt is None else self.q1_tgt.state_dict(),
            "q2_tgt": None if self.q2_tgt is None else self.q2_tgt.state_dict(),
            "v": None if self.v is None else self.v.state_dict(),
            "v_tgt": None if self.v_tgt is None else self.v_tgt.state_dict(),
            "actor_base": None if self.actor_base is None else self.actor_base.state_dict(),
            "actor_mean": None if self.actor_mean is None else self.actor_mean.state_dict(),
            "actor_logstd": None if self.actor_logstd is None else self.actor_logstd.state_dict(),
            "actor_op": None if self.actor_op is None else self.actor_op.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: str):
        if not HAS_TORCH:
            return
        payload = torch.load(path, map_location="cpu")

        # Restore dims / hparams
        self.action_dim = int(payload.get("action_dim", self.action_dim))
        self.op_dim = int(payload.get("op_dim", self.op_dim))
        self.alg_dim = int(payload.get("alg_dim", getattr(self, "alg_dim", 0)))
        self.bound_dim = int(payload.get("bound_dim", getattr(self, "bound_dim", max(0, self.action_dim - self.op_dim - self.alg_dim))))
        self.cont_dim = int(self.action_dim - self.op_dim)
        self.alg_logit_scale = float(payload.get("alg_logit_scale", getattr(self, "alg_logit_scale", 5.0)))
        self.expectile = float(payload.get("expectile", self.expectile))
        self.beta = float(payload.get("beta", self.beta))
        self.gamma = float(payload.get("gamma", self.gamma))

        # Restore cont scale (fallback to recompute)
        cs = payload.get("cont_scale", None)
        if cs is not None:
            cs = np.asarray(cs, dtype=np.float32).reshape(-1)
            if cs.size == self.cont_dim:
                self._cont_scale = cs.copy()
            else:
                self._cont_scale = np.ones(self.cont_dim, dtype=np.float32)
                if self.alg_dim > 0:
                    self._cont_scale[self.bound_dim:] = float(self.alg_logit_scale)
        else:
            self._cont_scale = np.ones(self.cont_dim, dtype=np.float32)
            if self.alg_dim > 0:
                self._cont_scale[self.bound_dim:] = float(self.alg_logit_scale)

        # Ensure models and load weights
        # We need state_dim to build nets; infer from saved state_norm mean length if possible.
        stn = payload.get("state_norm", {}) or {}
        mean = stn.get("mean", None)
        if mean is not None:
            state_dim = int(len(mean))
        else:
            # Fallback: keep existing (will be ensured on first act/train call)
            state_dim = int(getattr(self, "_state_dim", 0) or 0)

        if state_dim > 0:
            self._ensure_models(state_dim)

        # Load norms
        self.state_norm.mean = None if stn.get("mean") is None else np.asarray(stn["mean"], dtype=float)
        self.state_norm.var = None if stn.get("var") is None else np.asarray(stn["var"], dtype=float)
        self.state_norm.count = int(stn.get("count", getattr(self.state_norm, "count", 0)))

        # Load parameters if nets exist
        if self.q1 is not None and payload.get("q1") is not None:
            self.q1.load_state_dict(payload["q1"])
        if self.q2 is not None and payload.get("q2") is not None:
            self.q2.load_state_dict(payload["q2"])
        if self.q1_tgt is not None and payload.get("q1_tgt") is not None:
            self.q1_tgt.load_state_dict(payload["q1_tgt"])
        if self.q2_tgt is not None and payload.get("q2_tgt") is not None:
            self.q2_tgt.load_state_dict(payload["q2_tgt"])
        if self.v is not None and payload.get("v") is not None:
            self.v.load_state_dict(payload["v"])
        if self.v_tgt is not None and payload.get("v_tgt") is not None:
            self.v_tgt.load_state_dict(payload["v_tgt"])

        if self.actor_base is not None and payload.get("actor_base") is not None:
            self.actor_base.load_state_dict(payload["actor_base"])
        if self.actor_mean is not None and payload.get("actor_mean") is not None:
            self.actor_mean.load_state_dict(payload["actor_mean"])
        if self.actor_logstd is not None and payload.get("actor_logstd") is not None:
            self.actor_logstd.load_state_dict(payload["actor_logstd"])
        if self.actor_op is not None and payload.get("actor_op") is not None:
            self.actor_op.load_state_dict(payload["actor_op"])

        self._ckpt_loaded = True
