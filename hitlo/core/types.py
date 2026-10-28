"""
hitlo.core.types
==================
Core data classes extracted from the monolithic app.

Contains: ConstraintAtom, UPV, OSVSample, OSV, ProcessFeatures, ModelIR, Transition
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "ConstraintAtom",
    "UPV",
    "OSVSample",
    "OSV",
    "ProcessFeatures",
    "ModelIR",
    "Transition",
]


# ------------------ 内部 UPV / OSV 结构 ------------------

@dataclass
class ConstraintAtom:
    kind: str
    idx: int
    sign: str
    value: float
    hard: bool = True
    penalty: float = 10.0


@dataclass
class UPV:
    """
    内部使用的 UPV 结构：
    - w, r, b, C_h, C_s, D, tau, rho, pi
    - kappa：硬/软/鲁棒罚权初值
    - sigma：用于 OSV 中 w,r,b,D 的指数平滑系数
    """
    w: np.ndarray
    r: np.ndarray
    b: List[Tuple[float, float]]
    C_h: List[ConstraintAtom] = field(default_factory=list)
    C_s: List[ConstraintAtom] = field(default_factory=list)
    D: List[Tuple[float, float]] = field(default_factory=list)
    tau: float = 0.2
    rho: Dict[str, Any] = field(default_factory=lambda: {"cvar_alpha": 0.9,
                                                         "chance": 0.9,
                                                         "tail_q": 0.9,
                                                         "gamma": 0.0,
                                                         "kappa0": 1.0,
                                                         "time_decay": 0.05,
                                                         "noise_std": 0.0})
    pi: Dict[str, Any] = field(default_factory=lambda: {"explore_bias": 0.0,
                                                        "priority": "balanced"})
    kappa: Dict[str, float] = field(default_factory=lambda: {"hard": 1e-6, "var": 0.2, "robust": 0.2})
    sigma: Dict[str, float] = field(default_factory=lambda: {"w": 0.5, "r": 0.5, "b": 0.5, "D": 0.5})
    provenance: List[str] = field(default_factory=list)

    def log(self, msg: str):
        stamp = time.strftime("%H:%M:%S")
        self.provenance.append(f"[{stamp}] {msg}")


@dataclass
class OSVSample:
    """One OSV action sample with gating and safety (Eq.(9)(18))."""
    u_tilde: np.ndarray            # raw policy output (logits for alg/op)
    g_vec: np.ndarray              # per-dimension gating vector in [0,1]
    u_gate: np.ndarray             # gated action before safety
    applied: np.ndarray            # u_exec after Pi_safe
    u_default: np.ndarray          # fixed conservative baseline
    channel_slices: Dict[str, slice]
    algo_names: List[str]
    op_index: int = 0
    behavior_id: str = ""
    perturb_mask: Optional[np.ndarray] = None
    noise_std: float = 0.0
    profile_id: int = 0
    regime_id: int = 0

    def channel(self, name: str) -> np.ndarray:
        sl = self.channel_slices.get(name)
        if sl is None:
            return np.zeros(0)
        return np.asarray(self.applied[sl], dtype=float)

    @property
    def channels(self) -> Dict[str, np.ndarray]:
        return {k: self.channel(k) for k in self.channel_slices.keys()}

    @property
    def algo_logits(self) -> np.ndarray:
        return self.channel("alg")

    @property
    def algo_probs(self) -> Dict[str, float]:
        z = np.asarray(self.algo_logits, dtype=float)
        if z.size == 0:
            return {}
        z = z - np.max(z)
        e = np.exp(np.clip(z, -20.0, 20.0))
        p = e / (np.sum(e) + 1e-12)
        return {name: float(p[i]) for i, name in enumerate(self.algo_names)}


@dataclass
class OSV:
    """唯一的控制向量（channel-wise 执行 + 门控）。"""
    algo_names: List[str]
    g_mean: np.ndarray
    g_std: np.ndarray
    gate_p: np.ndarray
    w: np.ndarray
    r: np.ndarray
    g_exec: np.ndarray = field(default_factory=lambda: np.zeros(0))
    t: int = 0

    OP_NAMES = ["NOOP", "R1", "R2", "R3", "R4", "R5"]
    OP_DIM = len(OP_NAMES)
    CHANNEL_ORDER = ["pref", "rob", "alg", "param", "model", "op"]
    CHANNEL_DIM = {"pref": 2, "rob": 2, "alg": None, "param": 4, "model": 4, "op": OP_DIM}

    @property
    def n_channel(self) -> int:
        return len(self.CHANNEL_ORDER)

    def _channel_slices(self) -> Dict[str, slice]:
        sizes = {**self.CHANNEL_DIM, "alg": len(self.algo_names)}
        start = 0
        slices = {}
        for name in self.CHANNEL_ORDER:
            k = sizes[name]
            slices[name] = slice(start, start + k)
            start += k
        return slices

    @property
    def total_dim(self) -> int:
        return sum((len(self.algo_names) if k == "alg" else v) for k, v in self.CHANNEL_DIM.items())

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(np.clip(x, -20.0, 20.0))

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z)
        e = np.exp(np.clip(z, -20.0, 20.0))
        s = np.sum(e)
        return e / (s + 1e-12)

    def _default_exec(self) -> np.ndarray:
        slices = self._channel_slices()
        total = self.total_dim
        g0 = np.zeros(total, dtype=float)
        for name, sl in slices.items():
            if name == "alg":
                g0[sl] = 0.0
            elif name == "op":
                g0[sl] = 0.0
                if sl.stop - sl.start > 0:
                    g0[sl.start] = 1.0
            else:
                g0[sl] = 0.0
        return g0

    @classmethod
    def from_upv(cls, upv: "UPV", algo_names: List[str]) -> "OSV":
        k = 2 + 2 + len(algo_names) + 4 + 4 + cls.OP_DIM
        g0 = np.zeros(k, dtype=float)
        gate_p = np.ones(len(cls.CHANNEL_ORDER), dtype=float) * 0.6
        return cls(
            algo_names=list(algo_names),
            g_mean=g0.copy(),
            g_std=np.ones(k, dtype=float) * 0.25,
            gate_p=gate_p,
            w=np.asarray(upv.w, dtype=float).copy(),
            r=np.asarray(upv.r, dtype=float).copy(),
            g_exec=None
        )

    # -------- backward-compatible scalar views (for UI / logs) --------
    @property
    def eta(self) -> float:
        exec_vec = self.g_exec if self.g_exec is not None else self._default_exec()
        sl = self._channel_slices()["param"]
        explore = float(np.asarray(exec_vec[sl])[0]) if sl.stop - sl.start > 0 else 0.0
        explore = 0.5 * explore + 0.5
        return float(0.2 + 1.5 * explore)

    @property
    def trust_region(self) -> float:
        exec_vec = self.g_exec if self.g_exec is not None else self._default_exec()
        sl = self._channel_slices()["param"]
        trust = float(np.asarray(exec_vec[sl])[1]) if sl.stop - sl.start > 1 else 0.0
        trust = 0.5 * trust + 0.5
        return float(0.05 + 0.45 * trust)

    @property
    def lambda_proxy(self) -> float:
        exec_vec = self.g_exec if self.g_exec is not None else self._default_exec()
        sl_alg = self._channel_slices()["alg"]
        probs = np.asarray(exec_vec[sl_alg], dtype=float)
        lam = 0.0
        for i, name in enumerate(self.algo_names):
            if name in ("BO", "CP/MILP", "RL"):
                lam += float(probs[i])
        return float(np.clip(lam, 0.0, 1.0))

    def sample(self, rng: np.random.Generator, use_gate: bool = True, gate_eff: Optional[np.ndarray] = None) -> OSVSample:
        """Sample u_tilde and apply continuous gating (Eq.(18))."""
        self.t += 1
        slices = self._channel_slices()
        total = self.total_dim
        if self.g_exec is None or len(np.asarray(self.g_exec)) != total:
            self.g_exec = self._default_exec()
        if not hasattr(self, "_u_default"):
            self._u_default = np.asarray(self._default_exec(), dtype=float).copy()

        z = rng.normal(loc=self.g_mean, scale=self.g_std, size=total)
        u_tilde = np.zeros_like(z)
        for name, sl in slices.items():
            if name in ("alg", "op"):
                u_tilde[sl] = np.clip(z[sl], -5.0, 5.0)
            else:
                u_tilde[sl] = self._tanh(z[sl])

        if gate_eff is None:
            gate_eff = self.gate_p if use_gate else np.ones(self.n_channel, dtype=float)
        gate_eff = np.clip(np.asarray(gate_eff, dtype=float), 0.0, 1.0)
        g_vec = np.zeros(total, dtype=float)
        for idx, name in enumerate(self.CHANNEL_ORDER):
            sl = slices[name]
            g_vec[sl] = gate_eff[idx]

        u_default = np.asarray(self._u_default, dtype=float)
        u_gate = g_vec * u_tilde + (1.0 - g_vec) * u_default
        op_sl = slices["op"]
        op_logits = u_gate[op_sl] if op_sl.stop > op_sl.start else np.zeros(0)
        op_index = int(np.argmax(op_logits)) if op_logits.size else 0

        return OSVSample(
            u_tilde=u_tilde,
            g_vec=g_vec,
            u_gate=u_gate,
            applied=u_gate.copy(),
            u_default=u_default.copy(),
            channel_slices=slices,
            algo_names=list(self.algo_names),
            op_index=op_index
        )

    def sample_from_action(self, u_tilde: np.ndarray, gate_eff: np.ndarray) -> OSVSample:
        """Build OSVSample from u_tilde and continuous gating."""
        slices = self._channel_slices()
        total = self.total_dim
        if self.g_exec is None or len(np.asarray(self.g_exec)) != total:
            self.g_exec = self._default_exec()
        if not hasattr(self, "_u_default"):
            self._u_default = np.asarray(self._default_exec(), dtype=float).copy()
        u_tilde = np.asarray(u_tilde, dtype=float).reshape(-1)
        if u_tilde.size < total:
            u_full = np.zeros(total, dtype=float)
            u_full[:u_tilde.size] = u_tilde
            u_tilde = u_full
        gate_eff = np.clip(np.asarray(gate_eff, dtype=float), 0.0, 1.0)
        g_vec = np.zeros(total, dtype=float)
        for idx, name in enumerate(self.CHANNEL_ORDER):
            sl = slices[name]
            g_vec[sl] = gate_eff[idx]
        u_default = np.asarray(self._u_default, dtype=float)
        u_gate = g_vec * u_tilde + (1.0 - g_vec) * u_default
        op_sl = slices["op"]
        op_logits = u_gate[op_sl] if op_sl.stop > op_sl.start else np.zeros(0)
        op_index = int(np.argmax(op_logits)) if op_logits.size else 0
        return OSVSample(
            u_tilde=u_tilde,
            g_vec=g_vec,
            u_gate=u_gate,
            applied=u_gate.copy(),
            u_default=u_default.copy(),
            channel_slices=slices,
            algo_names=list(self.algo_names),
            op_index=op_index
        )


# ------------------ 过程特征 ------------------

@dataclass
class ProcessFeatures:
    """Process feature window summary F(t) (Appendix A.0.7)."""
    d_hv_smooth: float = 0.0
    stall_flag: float = 0.0
    feas_ratio: float = 0.0
    mean_violation: float = 0.0
    diversity: float = 0.0
    sat: float = 0.0
    conf: float = 0.0
    risk_proxy: float = 0.0
    d_feas_ratio: float = 0.0
    d_mean_violation: float = 0.0
    d_diversity: float = 0.0
    d_sat: float = 0.0
    d_conf: float = 0.0
    d_risk_proxy: float = 0.0

    def as_vector(self) -> np.ndarray:
        return np.asarray([
            self.d_hv_smooth, self.stall_flag,
            self.feas_ratio, self.mean_violation,
            self.diversity, self.sat, self.conf,
            self.risk_proxy, self.d_feas_ratio,
            self.d_mean_violation, self.d_diversity,
            self.d_sat, self.d_conf, self.d_risk_proxy
        ], dtype=float)


# ------------------ 模型中间表示 ------------------

@dataclass
class ModelIR:
    """优化模型中间表示（Model Intermediate Representation, IR）
    由 UPV × OSV 编译得到，供各算法一致使用，实现"模型重构（目标/约束/域/风险）"。
    """
    bounds: List[Tuple[float, float]]          # 有效变量域（用于采样/候选生成）
    penalty_scale: float = 1.0                # 罚项缩放
    bias_lambda: float = 0.05                 # 偏好 shaping 强度
    scalar_mode: str = "ASF"                  # 'ASF' | 'TCH' | 'WSUM'
    obj_band: Optional[List[Tuple[float, float]]] = None  # 动态目标带（可选）
    band_penalty: float = 0.0                 # 目标带违背罚系数
    risk_alpha: float = 0.9                   # 鲁棒/风险参数（示意）
    trust_radius_scale: float = 1.0           # 信任域缩放（记录用）

    def clamp_x(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).copy()
        for i, (lo, hi) in enumerate(self.bounds):
            X[:, i] = np.clip(X[:, i], lo, hi)
        return X


# ------------------ 离线 RL 转换记录 ------------------

@dataclass
class Transition:
    """Eq.(12) transition record for offline RL."""
    s: np.ndarray
    u: np.ndarray
    r: np.float32
    s_next: np.ndarray
    t: np.int32
    seed: np.int32
    profile_id: np.int32
    regime_id: np.int32
    behavior_id: str
    g_vec: np.ndarray
    perturb_mask: Optional[np.ndarray]
    noise_std: np.float32
    dhv_raw: np.float32
    dhv_smooth: np.float32
    dhv_norm: np.float32
    viol_raw: np.float32
    viol_norm: np.float32
    sat: np.float32
    pref_gap: np.float32
    cost: np.int32
    cost_norm: np.float32
    query_cost: np.float32
    query_norm: np.float32
    alpha_decoded: np.ndarray
    op_idx: np.int16
    xi: np.ndarray
    hv_feas: np.float32
    feas_ratio: np.float32
    mean_violation: np.float32
    diversity: np.float32
