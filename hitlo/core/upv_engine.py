"""
hitlo.core.upv_engine
=======================
UPV-related functions: normalization, constraint/penalty evaluation,
conflict resolution layers (L1--L4), state-vector construction,
and model-IR compilation.

Extracted from the monolithic ``app_v11_iqlfix.py``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from hitlo.core.objective_scale import scaled
from hitlo.core.types import (
    ConstraintAtom,
    UPV,
    OSV,
    OSVSample,
    ProcessFeatures,
    ModelIR,
)

if TYPE_CHECKING:
    pass  # forward references for OptState handled via string annotations

__all__ = [
    "normalize_w",
    "apply_patch_to_upv",
    "make_neutral_upv",
    "pyd_upv_to_internal",
    "check_var_D",
    "check_constraint",
    "soft_penalty",
    "compute_penalty_and_feas",
    "constraint_violation",
    "compute_violation",
    "phi_objective",
    "compute_sat_prefgap",
    "asf_distance",
    "asf_distance_batch",
    "l1_project_aspiration",
    "l2_reconcile_D_Ch",
    "l3_adapt_soft_penalties",
    "l4_goal_consistency",
    "build_state_vector",
    "compile_model_ir",
]


# ------------------------------------------------------------------ #
#  weight normalisation
# ------------------------------------------------------------------ #

def normalize_w(w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 1e-12)
    return w / np.sum(w)


# ------------------------------------------------------------------ #
#  apply_patch_to_upv  (LLM patch -> UPV in-place update)
# ------------------------------------------------------------------ #

def apply_patch_to_upv(current_upv, patch: Dict[str, Any], M: int):
    """
    将 LLM 产生的 patch 增量应用到 **当前** UPV。
    支持操作：
      - set_weight(index, value)
      - set_cvar(value)
      - set_chance(value)
      - set_gamma(value)
      - set_decay(value)         -> 写入 rho['time_decay']
      - set_init_penalty(value)  -> 写入 kappa['var']/kappa['robust']
      - set_trust(value)
      - set_priority(value)      -> 写入 pi['priority']
      - set_reference(values[ ])
      - set_band(obj_index, lo, hi)
      - add_constraint(kind, idx, sign, value, hard, penalty)
    """
    if current_upv is None:
        raise ValueError("current_upv is None")

    # 取可编辑副本
    w = [float(x) for x in (list(current_upv.w) if hasattr(current_upv.w, "__iter__") else [current_upv.w])]
    r = [float(x) for x in (list(current_upv.r) if getattr(current_upv, "r", None) is not None else [0.0] * M)]
    b = [tuple(map(float, bi)) for bi in current_upv.b] if getattr(current_upv, "b", None) else [(-float("inf"), float("inf")) for _ in range(M)]

    tau   = float(getattr(current_upv, "tau", 0.2))
    rho   = dict(getattr(current_upv, "rho", {"cvar_alpha": 0.9, "chance": 0.9}))
    kappa = dict(getattr(current_upv, "kappa", {"hard": 1e-6, "var": 0.2, "robust": 0.2}))
    pi    = dict(getattr(current_upv, "pi", {"explore_bias": 0.0, "priority": "balanced"}))

    C_h = list(current_upv.C_h)
    C_s = list(current_upv.C_s)

    # 尺寸对齐
    if len(w) != M: w = (w + [1.0 / M] * M)[:M]
    if len(r) != M: r = (r + [0.0] * M)[:M]
    if len(b) != M: b = (b + [(-float("inf"), float("inf"))] * M)[:M]

    def _norm(arr: List[float]) -> List[float]:
        arr = [max(0.0, float(x)) for x in arr]
        s = sum(arr)
        return [x / s for x in arr] if s > 0 else [1.0 / M] * M

    # 应用补丁
    for op in patch.get("ops", []):
        t = op.get("op")
        if t == "set_weight":
            idx = int(op.get("index", -1))
            val = float(op.get("value", 0.0))
            if 0 <= idx < M:
                w[idx] = max(0.0, val)

        elif t == "set_cvar":
            rho["cvar_alpha"] = float(op.get("value", rho.get("cvar_alpha", 0.9)))

        elif t == "set_chance":
            rho["chance"] = float(op.get("value", rho.get("chance", 0.9)))

        elif t == "set_gamma":
            rho["gamma"] = float(op.get("value", rho.get("gamma", 0.0)))

        elif t == "set_decay":
            rho["time_decay"] = float(op.get("value", rho.get("time_decay", 0.05)))

        elif t == "set_init_penalty":
            v = float(op.get("value", kappa.get("var", 0.2)))
            kappa["var"] = v
            kappa["robust"] = v

        elif t == "set_trust":
            tau = float(op.get("value", tau))

        elif t == "set_priority":
            pi["priority"] = str(op.get("value", pi.get("priority", "balanced")))

        elif t == "set_reference":
            vals = op.get("values", [])
            if isinstance(vals, list) and len(vals) > 0:
                r = (list(map(float, vals)) + [0.0] * M)[:M]

        elif t == "set_band":
            j = int(op.get("obj_index", -1))
            lo = op.get("lo", -float("inf"))
            hi = op.get("hi",  float("inf"))
            try:
                lo = -float("inf") if str(lo).lower() in ("-inf", "-infinity") else float(lo)
                hi =  float("inf") if str(hi).lower() in ("inf", "infinity") else float(hi)
            except Exception:
                lo, hi = -float("inf"), float("inf")
            if 0 <= j < M:
                b[j] = (lo, hi)

        elif t == "add_constraint":
            kind = op.get("kind")
            idx  = int(op.get("idx", -1))
            sign = op.get("sign")
            val  = float(op.get("value", 0.0))
            hard = bool(op.get("hard", True))
            pen  = float(op.get("penalty", 10.0))
            if kind in ("obj", "var") and sign in (">=", "<=", "==") and idx >= 0:
                ca = ConstraintAtom(kind=kind, idx=idx, sign=sign, value=val, hard=hard, penalty=pen)
                (C_h if hard else C_s).append(ca)

    # 归一化 w
    w = _norm(w)

    # 写回 current_upv
    current_upv.w     = np.asarray(w, dtype=float)
    current_upv.r     = np.asarray(r, dtype=float)
    current_upv.b     = [(float(lo), float(hi)) for (lo, hi) in b]
    current_upv.tau   = float(tau)
    current_upv.rho   = dict(rho)
    current_upv.kappa = dict(kappa)
    current_upv.pi    = dict(pi)
    current_upv.C_h   = C_h
    current_upv.C_s   = C_s
    return current_upv


# ------------------------------------------------------------------ #
#  make_neutral_upv
# ------------------------------------------------------------------ #

def make_neutral_upv(upv: UPV) -> UPV:
    """返回一个"中性"UPV：保留约束/风险/软约束，但把 (w,r) 设为无偏好。
    主要用于论文对比组 A1/A2（不允许用 UPV-conditioned 控制）。
    """
    upv2 = copy.deepcopy(upv)
    try:
        if getattr(upv2, 'w', None) is not None:
            upv2.w = normalize_w(np.ones_like(upv2.w, dtype=float))
        if getattr(upv2, 'r', None) is not None:
            upv2.r = np.zeros_like(upv2.r, dtype=float)
    except Exception:
        # 极端情况下保持原值（不会影响程序运行）
        pass
    return upv2


def make_single_slot_upv(upv: UPV) -> UPV:
    """§4.5 ABL_SINGLE_SLOT: 仅保留 S1 槽位 (w), 其他六个槽位中性化.

    用于验证论文 §4.2.1 七槽位结构相对"仅目标权重"基线的增益.
    保留 w (S1) 与 D (变量域,问题级约束); 中性化 r/b/C_h/C_s/pi/rho.
    """
    upv2 = copy.deepcopy(upv)
    try:
        if getattr(upv2, 'r', None) is not None:
            upv2.r = np.zeros_like(upv2.r, dtype=float)            # S2 中性
        if getattr(upv2, 'b', None) is not None:
            n = len(upv2.b)
            upv2.b = [(-float('inf'), float('inf')) for _ in range(n)]   # S3 中性
        upv2.C_h = []                                              # S4 硬约束清空
        upv2.C_s = []                                              # S4 软约束清空
        upv2.pi = {"explore_bias": 0.0, "priority": "balanced"}    # S5 中性
        upv2.rho = {"cvar_alpha": 0.5, "chance": 0.5, "tail_q": 0.5,
                    "gamma": 0.0, "kappa0": 1.0, "time_decay": 0.0,
                    "noise_std": 0.0}                              # S6 中性
        # S7 sigma 保留默认 (置信度结构与 w 槽位本身耦合, 不动)
    except Exception:
        pass
    return upv2


def _ablation_active(state, flag: str) -> bool:
    """读取 state.experiment_config.ablations 判定某 ABL_* 标签是否启用."""
    cfg = getattr(state, "experiment_config", None)
    if cfg is None:
        return False
    abl = getattr(cfg, "ablations", None)
    if not abl:
        return False
    return flag in abl


# ------------------------------------------------------------------ #
#  pyd_upv_to_internal  (Pydantic UPV -> internal UPV)
# ------------------------------------------------------------------ #

def pyd_upv_to_internal(pu, n_var: int) -> UPV:
    """
    将 Pydantic 版 UPV（来自 NL2UPV）转成内部 UPV。
    同步 rho / pi / kappa / sigma 四块信息。
    """
    w = np.array(pu.w, dtype=float)
    r = np.array(pu.r, dtype=float)
    b = [(float(lo), float(hi)) for (lo, hi) in pu.b]
    D = [tuple(x) for x in pu.D] if len(pu.D) > 0 else []
    C_h, C_s = [], []
    for c in pu.C_h:
        C_h.append(ConstraintAtom(c.kind, int(c.idx), c.sign, float(c.value), True, float(c.penalty)))
    for c in pu.C_s:
        C_s.append(ConstraintAtom(c.kind, int(c.idx), c.sign, float(c.value), False, float(c.penalty)))

    # rho / pi
    rho = dict(pu.rho.model_dump())
    pi  = dict(pu.pi.model_dump())

    # kappa / sigma（如果 Pydantic 中没有，就用默认）
    default_kappa = {"hard": 1e-6, "var": 0.2, "robust": 0.2}
    default_sigma = {"w": 0.5, "r": 0.5, "b": 0.5, "D": 0.5}
    kappa = getattr(pu, "kappa", None)
    sigma = getattr(pu, "sigma", None)
    if isinstance(kappa, dict):
        kappa = {**default_kappa, **{k: float(v) for k, v in kappa.items()}}
    else:
        kappa = default_kappa
    if isinstance(sigma, dict):
        sigma = {**default_sigma, **{k: float(v) for k, v in sigma.items()}}
    else:
        sigma = default_sigma

    return UPV(
        w=w, r=r, b=b, C_h=C_h, C_s=C_s, D=D,
        tau=float(pu.tau),
        rho=rho, pi=pi,
        kappa=kappa, sigma=sigma
    )


# ------------------------------------------------------------------ #
#  constraint / penalty / metric helpers
# ------------------------------------------------------------------ #

def check_var_D(x: np.ndarray, D: List[Tuple[float, float]]) -> bool:
    if not D:
        return True
    for i, (lo, hi) in enumerate(D):
        if x[i] < lo - 1e-12 or x[i] > hi + 1e-12:
            return False
    return True


def check_constraint(x: np.ndarray, f: np.ndarray, c: ConstraintAtom) -> bool:
    val = x[c.idx] if c.kind == "var" else f[c.idx]
    if c.sign == "<=":
        return val <= c.value + 1e-12
    if c.sign == ">=":
        return val >= c.value - 1e-12
    return abs(val - c.value) <= 1e-9


def soft_penalty(x: np.ndarray, f: np.ndarray, c: ConstraintAtom) -> float:
    val = x[c.idx] if c.kind == "var" else f[c.idx]
    if c.sign == "<=":
        v = max(0.0, val - c.value)
    elif c.sign == ">=":
        v = max(0.0, c.value - val)
    else:
        v = abs(val - c.value)
    return c.penalty * v


def compute_penalty_and_feas(x: np.ndarray, f: np.ndarray, upv: UPV, cr_on: bool) -> Tuple[float, bool]:
    P = 0.0
    feas = True
    if upv.D and not check_var_D(x, upv.D):
        P += (1000.0 if cr_on else 10.0); feas = False
    for c in upv.C_h:
        ok = check_constraint(x, f, c)
        if not ok:
            P += (1000.0 if cr_on else 10.0); feas = False
    for c in upv.C_s:
        P += soft_penalty(x, f, c)
    return P, feas


def constraint_violation(x: np.ndarray, f: np.ndarray, c: ConstraintAtom) -> float:
    val = x[c.idx] if c.kind == "var" else f[c.idx]
    if c.sign == "<=":
        return max(0.0, val - c.value)
    if c.sign == ">=":
        return max(0.0, c.value - val)
    return abs(val - c.value)


def compute_violation(x: np.ndarray, f: np.ndarray, upv: UPV) -> float:
    v = 0.0
    if upv.D:
        for i, (lo, hi) in enumerate(upv.D):
            v += max(0.0, lo - x[i])
            v += max(0.0, x[i] - hi)
    for c in upv.C_h:
        v += constraint_violation(x, f, c)
    for c in upv.C_s:
        v += constraint_violation(x, f, c)
    return float(v)


def phi_objective(x: np.ndarray, f: np.ndarray, upv: UPV, cr_on: bool) -> float:
    """Eq.(3) scalar objective using UPV (w/r/penalty/constraints)."""
    a = asf_distance(f, upv.w, upv.r)
    p, _ = compute_penalty_and_feas(x, f, upv, cr_on)
    return float(a + p)


def compute_sat_prefgap(X: np.ndarray, F: np.ndarray, upv: UPV, cr_on: bool) -> Tuple[float, float]:
    if X is None or F is None or len(F) == 0:
        return 0.0, 1.0
    phi_vals = [phi_objective(X[i], F[i], upv, cr_on) for i in range(len(F))]
    phi_vals = np.asarray(phi_vals, dtype=float)
    phi_min = float(np.min(phi_vals))
    tau = max(1e-6, float(getattr(upv, "tau", 0.2)))
    sat = float(np.mean(np.exp(-(phi_vals - phi_min) / tau)))
    return sat, float(1.0 - sat)


def asf_distance(f: np.ndarray, w: np.ndarray, r: np.ndarray, rho: float = 1e-6) -> float:
    # Defensive: truncate to min dimension if shapes mismatch
    n = min(len(f), len(w), len(r))
    f, w, r = np.asarray(f[:n], dtype=float), np.asarray(w[:n], dtype=float), np.asarray(r[:n], dtype=float)
    d = scaled(w * np.abs(f - r))  # objective scales (preference model v2); identity by default
    return float(np.max(d) + rho * np.sum(d))


# ------------------------------------------------------------------ #
#  L1--L4 conflict resolution layers
# ------------------------------------------------------------------ #

def asf_distance_batch(F: np.ndarray, w: np.ndarray, r: np.ndarray) -> Tuple[int, float]:
    d = np.array([asf_distance(F[i], w, r) for i in range(F.shape[0])])
    idx = int(np.argmin(d))
    return idx, float(d[idx])


def l1_project_aspiration(state: "OptState", threshold: float = 0.12, eta: float = 0.25):
    if _ablation_active(state, "ABL_NO_CR"):     # §4.5 关闭 L1-L4 冲突消解
        return False
    if state.pop_F is None or len(state.pop_F) == 0:
        return None
    idx, dmin = asf_distance_batch(state.pop_F, state.upv.w, state.upv.r)
    if dmin > threshold:
        alpha = eta * (1.0 - state.upv.tau)
        best = state.pop_F[idx]
        old_r = state.upv.r.copy()
        state.upv.r = (1 - alpha) * state.upv.r + alpha * best
        state.add_log("CR\u2605L1", f"dmin={dmin:.4f}, alpha={alpha:.3f}, r: {old_r.tolist()} -> {state.upv.r.tolist()}")
        return True
    return False


def l2_reconcile_D_Ch(state: "OptState"):
    if _ablation_active(state, "ABL_NO_CR"):     # §4.5 关闭 L1-L4 冲突消解
        return False
    changed = False
    notes = []
    D = list(state.upv.D) if state.upv.D else [tuple(x) for x in state.problem.bounds]
    for c in state.upv.C_h:
        if c.kind != "var":
            continue
        lo, hi = D[c.idx]
        old = (lo, hi)
        if c.sign == "<=" and hi > c.value:
            hi = c.value; changed = True
        elif c.sign == ">=" and lo < c.value:
            lo = c.value; changed = True
        elif c.sign == "==" and (lo > c.value or hi < c.value):
            lo = hi = c.value; changed = True
        lo = max(lo, float(state.problem.bounds[c.idx, 0]))
        hi = min(hi, float(state.problem.bounds[c.idx, 1]))
        if old != (lo, hi):
            notes.append(f"x{c.idx+1}: {old} -> {(lo, hi)} by hard {c.sign} {c.value}")
        D[c.idx] = (lo, hi)
    if changed:
        oldD = list(state.upv.D)
        state.upv.D = D
        state.add_log("CR\u2605L2", f"D adjusted: {oldD} -> {D}; " + " | ".join(notes))
    return changed


def l3_adapt_soft_penalties(state: "OptState",
                            target_feas: Tuple[float, float] = (0.6, 0.85),
                            up_gain: float = 1.12, down_gain: float = 0.95):
    if _ablation_active(state, "ABL_NO_CR"):     # §4.5 关闭 L1-L4 冲突消解
        return False
    if not state.upv.C_s or state.pop_X is None:
        return False
    X, F = state.pop_X, state.pop_F
    if len(X) == 0:
        return False

    lower, upper = target_feas
    m = len(state.upv.C_s)
    viol = np.zeros(m, dtype=int)
    for i in range(len(X)):
        for k, c in enumerate(state.upv.C_s):
            val = X[i, c.idx] if c.kind == "var" else F[i, c.idx]
            ok = (val <= c.value + 1e-12) if c.sign == "<=" else \
                 (val >= c.value - 1e-12) if c.sign == ">=" else \
                 (abs(val - c.value) <= 1e-9)
            if not ok:
                viol[k] += 1
    viol_rate = viol / max(1, len(X))

    changes = []
    for k, c in enumerate(state.upv.C_s):
        old_pen = c.penalty
        if viol_rate[k] > (1 - upper):
            c.penalty = float(c.penalty * up_gain)
        elif viol_rate[k] < (1 - lower):
            c.penalty = float(c.penalty * down_gain)
        if abs(c.penalty - old_pen) / max(1e-12, old_pen) > 1e-6:
            changes.append(f"soft[{k}] pen {old_pen:.3f}->{c.penalty:.3f} (viol={viol_rate[k]:.2f})")
    if changes:
        state.add_log("CR\u2605L3", " | ".join(changes))
        return True
    return False


def l4_goal_consistency(state: "OptState", clamp_alpha: float = 0.5):
    if _ablation_active(state, "ABL_NO_CR"):     # §4.5 关闭 L1-L4 冲突消解
        return False
    changed = False
    notes = []
    for j in range(len(state.upv.b)):
        lo, hi = state.upv.b[j]
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        rj = float(state.upv.r[j])
        if rj < lo or rj > hi:
            new_rj = (1 - clamp_alpha) * rj + clamp_alpha * np.clip(rj, lo, hi)
            notes.append(f"r{j+1} {rj:.3f}->{new_rj:.3f} within [{lo:.3f},{hi:.3f}]")
            state.upv.r[j] = new_rj
            changed = True
    w_old = state.upv.w.copy()
    state.upv.w = normalize_w(state.upv.w)
    if np.max(np.abs(w_old - state.upv.w)) > 1e-12:
        notes.append(f"w normalized: {w_old.tolist()} -> {state.upv.w.tolist()}")
        changed = True
    if changed:
        state.add_log("CR\u2605L4", " | ".join(notes))
    return changed


# ------------------------------------------------------------------ #
#  build_state_vector  (Eq.(8))
# ------------------------------------------------------------------ #

def build_state_vector(upv: UPV, pf: ProcessFeatures, ir_summary: Optional[np.ndarray] = None) -> np.ndarray:
    """Eq.(8) s_t = [UPV(t), F(t)] (+ optional IR_summary with fixed dims)."""
    w = np.asarray(upv.w, dtype=float).reshape(-1)
    r = np.asarray(upv.r, dtype=float).reshape(-1)
    sw = float(upv.sigma.get("w", 0.5)) if isinstance(getattr(upv, "sigma", None), dict) else 0.5
    sr = float(upv.sigma.get("r", 0.5)) if isinstance(getattr(upv, "sigma", None), dict) else 0.5
    tau = float(getattr(upv, "tau", 0.2))
    alpha = float(upv.rho.get("cvar_alpha", 0.9)) if isinstance(getattr(upv, "rho", None), dict) else 0.9
    chance = float(upv.rho.get("chance", 0.9)) if isinstance(getattr(upv, "rho", None), dict) else 0.9
    if ir_summary is None:
        ir_summary = np.zeros(6, dtype=float)

    s = np.concatenate([
        w, r,
        np.array([tau, sw, sr, alpha, chance], dtype=float),
        pf.as_vector(),
        np.asarray(ir_summary, dtype=float).reshape(-1)
    ])
    # 防止策略输入出现 NaN/Inf（会导致 IQL 的 logits 变 NaN）
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    s = np.clip(s, -1e6, 1e6)
    return s


# ------------------------------------------------------------------ #
#  compile_model_ir  (UPV x OSV -> ModelIR)
# ------------------------------------------------------------------ #

def compile_model_ir(base_problem,
                     upv: UPV,
                     osv_sample: OSVSample,
                     pf: ProcessFeatures,
                     x_best: Optional[np.ndarray],
                     exp_cfg: Optional["ExperimentConfig"] = None) -> ModelIR:
    """UPV x OSV -> ModelIR: 实现"优化模型重构"的统一入口。"""
    from hitlo.core.experiment import ExperimentConfig, normalize_method_id

    if exp_cfg is None:
        exp_cfg = ExperimentConfig()
    if "ABL_NO_MODEL_RECONF" in exp_cfg.ablations:
        bounds0 = [(float(lo), float(hi)) for lo, hi in zip(base_problem.xl, base_problem.xu)]
        return ModelIR(bounds=bounds0)

    # --- C1 HV booster gate ---
    try:
        method_id = normalize_method_id(getattr(exp_cfg, "exp_group", ""))
    except Exception:
        method_id = str(getattr(exp_cfg, "exp_group", ""))
    sat_now  = float(getattr(pf, "sat", 0.0))
    feas_now = float(getattr(pf, "feas_ratio", 0.0))
    viol_now = float(getattr(pf, "mean_violation", 0.0))
    hv_boost_active = False
    if (method_id == "C1_PPCS_IQL_FULL") and bool(getattr(exp_cfg, "hv_boost_enabled", False)):
        sat_th  = float(getattr(exp_cfg, "hv_boost_sat_th", 0.50))
        feas_th = float(getattr(exp_cfg, "hv_boost_feas_th", 0.55))
        viol_th = float(getattr(exp_cfg, "hv_boost_viol_th", 1.0))

        # CO-ARCH: sat 的尺度很小，默认忽略 sat 门控（改用可行率+违反强度触发）
        # Note: isinstance check for COProblem is deferred to runtime;
        # we check for a 'co_arch' marker attribute instead.
        ignore_sat = bool(getattr(exp_cfg, "hv_boost_ignore_sat_on_coarch", True)) and \
                     getattr(base_problem, "_is_co_problem", False)
        if ignore_sat:
            feas_th = max(feas_th, float(getattr(exp_cfg, "hv_boost_coarch_feas_th", 0.90)))
            viol_th = min(viol_th, float(getattr(exp_cfg, "hv_boost_coarch_viol_th", 0.50)))
            sat_ok = True
        else:
            sat_ok = (sat_now >= sat_th)

        hv_boost_active = sat_ok and (feas_now >= feas_th) and (viol_now <= viol_th)

    ch = getattr(osv_sample, "channels", {}) if osv_sample is not None else {}
    o_pref  = np.asarray(ch.get("pref",  np.zeros(2)), dtype=float)
    o_rob   = np.asarray(ch.get("rob",   np.zeros(2)), dtype=float)
    o_param = np.asarray(ch.get("param", np.zeros(4)), dtype=float)
    o_model = np.asarray(ch.get("model", np.zeros(4)), dtype=float)
    l_op    = int(getattr(osv_sample, "op_index", 0))

    # 1) bounds: 信任域缩放（围绕当前偏好最优点）
    bounds0 = [(float(lo), float(hi)) for lo, hi in zip(base_problem.xl, base_problem.xu)]
    trust_raw = float(np.clip(0.5 * (o_param[2] if o_param.size >= 3 else 0.0) + 0.5, 0.0, 1.0))
    if x_best is None:
        bounds = bounds0
        tr_scale = 1.0
    else:
        tr_scale = 0.08 + 0.92 * trust_raw
        bounds = []
        for i, (lo, hi) in enumerate(bounds0):
            rng = hi - lo
            rad = tr_scale * rng
            c = float(x_best[i])
            bounds.append((max(lo, c - rad), min(hi, c + rad)))

    # hv_boost: sat/可行性足够好时，放宽信任域以改善 HV 覆盖
    if hv_boost_active:
        tr_min = float(getattr(exp_cfg, "hv_boost_tr_min", 0.20))
        if np.isfinite(tr_min):
            tr_scale = float(max(tr_scale, tr_min))
            if x_best is not None:
                bounds = []
                for i, (lo, hi) in enumerate(bounds0):
                    rng = hi - lo
                    rad = tr_scale * rng
                    c = float(x_best[i])
                    bounds.append((max(lo, c - rad), min(hi, c + rad)))

    # 2) penalty/bias: 偏好注入强度、约束压力共同决定
    pref_strength = float(np.clip(0.5 * (o_pref[0] if o_pref.size >= 1 else 0.0) + 0.5, 0.0, 1.0))
    penalty_raw   = float(np.clip(0.5 * (o_param[1] if o_param.size >= 2 else 0.0) + 0.5, 0.0, 1.0))
    penalty_scale = 0.5 + 6.0 * penalty_raw
    bias_lambda   = 0.02 + 0.25 * pref_strength  # preference shaping

    if hv_boost_active:
        bias_max = float(getattr(exp_cfg, "hv_boost_bias_max", 0.08))
        if np.isfinite(bias_max):
            bias_lambda = float(min(bias_lambda, bias_max))

    # 3) scalar_mode: 保持默认 ASF（operator l(t) 由 R1-R5 处理）
    scalar_mode  = "ASF"
    model_scalar = float(np.clip(0.5 * (o_model[0] if o_model.size >= 1 else 0.0) + 0.5, 0.0, 1.0))

    # 4) obj_band: 将"偏好愿望点 r"转换为阶段性目标带（强偏好时收紧）
    obj_band = None
    band_penalty = 0.0
    if x_best is not None and np.isfinite(np.sum(upv.r)) and (model_scalar > 0.55 or pref_strength > 0.55):
        shrink = 0.5 - 0.35 * pref_strength + 0.12 * pf.stall_flag
        shrink = float(np.clip(shrink, 0.06, 0.8))
        obj_band = []
        for j in range(len(upv.r)):
            rj = float(upv.r[j])
            width = 1.0 * shrink
            obj_band.append((rj - width, rj + width))
        band_penalty = 0.4 + 2.0 * penalty_raw

    if hv_boost_active and bool(getattr(exp_cfg, "hv_boost_disable_band", True)):
        obj_band = None
        band_penalty = 0.0

    # 5) risk 参数（示意）: risk_strength -> alpha
    rob_strength = float(np.clip(0.5 * (o_rob[0] if o_rob.size >= 1 else 0.0) + 0.5, 0.0, 1.0))
    risk_alpha   = float(np.clip(0.55 + 0.4 * rob_strength, 0.55, 0.995))

    mir = ModelIR(
        bounds=bounds,
        penalty_scale=float(penalty_scale),
        bias_lambda=float(bias_lambda),
        scalar_mode=scalar_mode,
        obj_band=obj_band,
        band_penalty=float(band_penalty),
        risk_alpha=risk_alpha,
        trust_radius_scale=float(tr_scale)
    )
    return mir
