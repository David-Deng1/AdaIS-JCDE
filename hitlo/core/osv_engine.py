"""
hitlo.core.osv_engine
=======================
OSV safety / gating functions: conflict & confidence scoring,
conflict-confidence gating, simplex projection, rolling statistics,
and the three Pi_safe projectors.

Extracted from the monolithic ``app_v11_iqlfix.py``.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from hitlo.core.types import (
    OSV,
    OSVSample,
    ProcessFeatures,
    ModelIR,
)

if TYPE_CHECKING:
    from hitlo.core.experiment import ExperimentConfig

__all__ = [
    "compute_conflict_score",
    "compute_confidence_score",
    "apply_conflict_confidence_gating",
    "_project_simplex_with_floor",
    "affine",
    "round_affine",
    "log10_map",
    "_softmax_temp",
    "RollingMedianIQR",
    "pi_safe_osv_sample",
    "pi_safe_params",
    "pi_safe_model_ir",
]


# ------------------------------------------------------------------ #
#  conflict / confidence scoring
# ------------------------------------------------------------------ #

def compute_conflict_score(state: "OptState") -> float:
    # 源1：L5冲突队列事件计数（语义层冲突）
    pending = len(getattr(state, "l5_pending", []) or [])
    queued = len(getattr(state, "l5_queue", []) or [])
    s = pending + queued
    score_queue = 1.0 - float(np.exp(-0.5 * s))

    # 源2：UPV参数更新散度（分布层冲突代理）
    score_drift = 0.0
    upv = getattr(state, "upv", None)
    if upv is not None:
        w_now = np.asarray(upv.w, dtype=float).ravel()
        r_now = np.asarray(upv.r, dtype=float).ravel()
        w_prev = np.asarray(getattr(state, "_conflict_prev_w", w_now), dtype=float).ravel()
        r_prev = np.asarray(getattr(state, "_conflict_prev_r", r_now), dtype=float).ravel()
        d_w = float(np.linalg.norm(w_now - w_prev)) if w_now.shape == w_prev.shape else 0.0
        d_r = float(np.linalg.norm(r_now - r_prev)) if r_now.shape == r_prev.shape else 0.0
        d_upv = d_w + d_r
        score_drift = float(1.0 / (1.0 + np.exp(-5.0 * (d_upv - 0.3))))
        state._conflict_prev_w = w_now.copy()
        state._conflict_prev_r = r_now.copy()

    score = max(score_queue, score_drift)
    state.last_conflict_score = score
    return float(np.clip(score, 0.0, 1.0))


def compute_confidence_score(state: "OptState") -> float:
    # 源1：证据量贡献（gamma加权的NL原子+排序对+视觉交互）
    n_nl   = int(getattr(state, "n_atoms_nl_total", 0))
    n_rank = int(getattr(state, "n_pairs_rank_total", 0))
    n_vis  = int(getattr(state, "n_atoms_visual_total", 0))
    evid = n_nl + n_rank + n_vis
    conf_evid = 1.0 - float(np.exp(-0.02 * evid))

    # 源2：GP后验不确定度贡献（sigma越小=越确定=conf越高）
    conf_sigma = 0.5  # 无GP信息时中性默认
    upv = getattr(state, "upv", None)
    if upv is not None and isinstance(getattr(upv, "sigma", None), dict):
        sigma_w = float(np.clip(upv.sigma.get("w", 0.5), 0.0, 1.0))
        sigma_r = float(np.clip(upv.sigma.get("r", 0.5), 0.0, 1.0))
        conf_sigma = 1.0 - 0.5 * (sigma_w + sigma_r)

    # 加权融合
    conf = 0.6 * conf_evid + 0.4 * conf_sigma
    return float(np.clip(conf, 0.0, 1.0))


# §4.5 OSV 6 通道消融映射（论文 §4.4 6 分组 → 代码 6 通道）
#   pref     → pref      (偏好塑形)
#   feas     → rob       (可行性引导, 鲁棒/约束惩罚通道)
#   explore  → alg       (探索-利用, 算法分布)
#   param    → param     (参数自适应)
#   rebuild  → model     (模型重构)
#   interact → op        (交互触发, 操作符)
_OSV_CHANNEL_ABL_MAP = {
    "ABL_NO_OSV_PREF":  "pref",
    "ABL_NO_OSV_ROB":   "rob",
    "ABL_NO_OSV_ALG":   "alg",
    "ABL_NO_OSV_PARAM": "param",
    "ABL_NO_OSV_MODEL": "model",
    "ABL_NO_OSV_OP":    "op",
}


def apply_conflict_confidence_gating(osv: OSV, conflict: float, confidence: float,
                                     exp_cfg: "ExperimentConfig") -> np.ndarray:
    if "ABL_NO_GATING" in exp_cfg.ablations:
        gate_eff = np.ones_like(osv.gate_p)
    else:
        base = np.asarray(osv.gate_p, dtype=float)
        scales = []
        for name in OSV.CHANNEL_ORDER:
            if name in ("pref", "model"):
                scale = np.clip(confidence * (1.0 - conflict), 0.0, 1.0)
            else:
                scale = np.clip(0.5 + 0.5 * confidence * (1.0 - 0.5 * conflict), 0.3, 1.0)
            scales.append(scale)
        gate_eff = np.clip(base * np.array(scales), 0.0, 1.0)
        if not np.any(gate_eff):
            gate_eff = np.ones_like(gate_eff)

    # §4.5 OSV 单通道消融: ABL_NO_OSV_<CHANNEL> 把对应通道 gate 置 0
    for abl in exp_cfg.ablations:
        ch = _OSV_CHANNEL_ABL_MAP.get(abl)
        if ch is not None and ch in OSV.CHANNEL_ORDER:
            idx = OSV.CHANNEL_ORDER.index(ch)
            gate_eff[idx] = 0.0
    return gate_eff


# ------------------------------------------------------------------ #
#  helper utilities
# ------------------------------------------------------------------ #

def _project_simplex_with_floor(alpha: np.ndarray, floor: np.ndarray) -> np.ndarray:
    alpha = np.maximum(alpha, 0.0)
    floor = np.maximum(floor, 0.0)
    if alpha.size == 0:
        return alpha
    floor = np.minimum(floor, 1.0)
    if np.sum(floor) >= 1.0:
        return floor / (np.sum(floor) + 1e-12)
    rem = 1.0 - float(np.sum(floor))
    a = np.maximum(alpha - floor, 0.0)
    s = float(np.sum(a))
    if s <= 1e-12:
        return floor + rem / alpha.size
    return floor + rem * (a / s)


def affine(z: float, lo: float, hi: float) -> float:
    return float(lo + (z + 1.0) * 0.5 * (hi - lo))


def round_affine(z: float, lo: float, hi: float) -> int:
    return int(round(affine(z, lo, hi)))


def log10_map(z: float, lo: float, hi: float) -> float:
    lo_l = np.log10(lo)
    hi_l = np.log10(hi)
    return float(10 ** affine(z, lo_l, hi_l))


def _softmax_temp(logits: np.ndarray, temp: float) -> np.ndarray:
    z = np.asarray(logits, dtype=float) * float(temp)
    z = z - np.max(z)
    e = np.exp(np.clip(z, -20.0, 20.0))
    return e / (np.sum(e) + 1e-12)


# ------------------------------------------------------------------ #
#  RollingMedianIQR
# ------------------------------------------------------------------ #

class RollingMedianIQR:
    def __init__(self, maxlen: int = 200, eps: float = 1e-6):
        self.values = deque(maxlen=maxlen)
        self.eps = float(eps)

    def update(self, v: float):
        try:
            v = float(v)
        except Exception:
            return
        if not np.isfinite(v):
            return
        self.values.append(v)

    def _stats(self) -> Tuple[float, float, float, float]:
        if not self.values:
            return 0.0, 1.0, 0.0, 1.0
        arr = np.asarray(self.values, dtype=float)
        med = float(np.median(arr))
        q25 = float(np.percentile(arr, 25))
        q75 = float(np.percentile(arr, 75))
        iqr = max(q75 - q25, self.eps)
        p95 = float(np.percentile(arr, 95))
        return med, iqr, p95, max(p95, self.eps)

    def norm_signed(self, v: float) -> float:
        med, iqr, _, _ = self._stats()
        return float(np.clip(0.5 + (float(v) - med) / (2.0 * iqr), 0.0, 1.0))

    def norm_pos(self, v: float) -> float:
        _, _, _, p95 = self._stats()
        return float(np.clip(max(0.0, float(v)) / p95, 0.0, 1.0))


# ------------------------------------------------------------------ #
#  Cold start prior (问题感知的冷启动先验)
# ------------------------------------------------------------------ #

def _cold_start_prior(algo_pool: List[str], n_constr: int) -> np.ndarray:
    """Generate a problem-aware cold-start allocation prior.

    When the policy has no training data (early ticks), this provides a
    reasonable default instead of uniform allocation:
    - Constrained problems: favour CP/MILP + NSGA-II (feasibility-oriented)
    - Unconstrained problems: favour NSGA-III + BO (coverage-oriented)
    """
    prior = np.ones(len(algo_pool), dtype=float)
    name_to_idx = {a: i for i, a in enumerate(algo_pool)}

    if n_constr > 0:
        # constrained: emphasise feasibility-oriented solvers
        for a, w in [("CP/MILP", 3.0), ("NSGA-II", 2.5), ("NSGA-III", 1.5),
                     ("MOEA/D", 1.5), ("BO", 0.5), ("RL", 0.8)]:
            if a in name_to_idx:
                prior[name_to_idx[a]] = w
    else:
        # unconstrained: emphasise coverage-oriented solvers
        for a, w in [("NSGA-III", 3.0), ("MOEA/D", 2.0), ("BO", 2.0),
                     ("NSGA-II", 1.5), ("CP/MILP", 0.5), ("RL", 1.0)]:
            if a in name_to_idx:
                prior[name_to_idx[a]] = w

    return prior / np.sum(prior)


# ------------------------------------------------------------------ #
#  Pi_safe projectors
# ------------------------------------------------------------------ #

def pi_safe_osv_sample(osv_sample: OSVSample, pf: ProcessFeatures,
                       exp_cfg: "ExperimentConfig", algo_pool: List[str]) -> Tuple[OSVSample, np.ndarray, int, np.ndarray]:
    """Safety projection Pi_safe for executing OSV.

    - 默认: 对 allocation simplex 施加 floor + 归一化; 对 op logits 加 noop bias; 对连续通道做范围裁剪。
    - ABL_NO_SAFETY / C2: 尽量不改动动作（仅做必要的数值清理/裁剪以避免数值崩溃）。
    """
    from hitlo.core.experiment import ExperimentConfig, normalize_method_id

    u_gate_raw = np.asarray(osv_sample.u_gate, dtype=float).copy()
    u_exec = u_gate_raw.copy()
    slices = osv_sample.channel_slices
    no_safety = ("ABL_NO_SAFETY" in exp_cfg.ablations)
    clip_only = ("ABL_CLIP_ONLY" in exp_cfg.ablations)
    no_consistency = ("ABL_NO_CONSISTENCY_PROJ" in exp_cfg.ablations)
    no_temporal = ("ABL_NO_TEMPORAL_SMOOTH" in exp_cfg.ablations)
    numeric_clip_count = 0
    alpha_raw = np.zeros(0, dtype=float)
    alpha_pre_project = np.zeros(0, dtype=float)
    alpha_shift_l1 = 0.0
    op_shift_l1 = 0.0
    lcc_count = 0
    raw_lcc_count = 0
    post_lcc_count = 0
    temporal_shift_l2 = 0.0
    hv_boost_active = 0

    def _safe_list(x: np.ndarray, max_items: int = 256) -> List[float]:
        arr = np.asarray(x, dtype=float).reshape(-1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size > max_items:
            arr = arr[:max_items]
        return [float(v) for v in arr]

    # --- minimal numeric sanitization ---
    for name, sl in slices.items():
        raw_vals = np.asarray(u_exec[sl], dtype=float)
        nonfinite_count = int(np.sum(~np.isfinite(raw_vals)))
        vals = np.nan_to_num(raw_vals, nan=0.0, posinf=0.0, neginf=0.0)
        if name in ("alg", "op"):
            lim = 20.0 if no_safety else 5.0
            vals_clipped = np.clip(vals, -lim, lim)
        else:
            # continuous channels are defined on [-1, 1]; keep this clamp even in no-safety
            vals_clipped = np.clip(vals, -1.0, 1.0)
        numeric_clip_count += nonfinite_count + int(np.sum(np.abs(vals_clipped - vals) > 1e-12))
        vals = vals_clipped
        u_exec[sl] = vals

    if (not no_safety) and (not no_temporal) and (not clip_only):
        prev = getattr(pf, "prev_exec_action", None)
        if prev is not None:
            prev_arr = np.asarray(prev, dtype=float).reshape(-1)
            if prev_arr.shape == u_exec.shape and np.all(np.isfinite(prev_arr)):
                tau = float(np.clip(getattr(exp_cfg, "temporal_smooth_tau", 0.15), 0.0, 0.95))
                before = u_exec.copy()
                u_exec = (1.0 - tau) * u_exec + tau * prev_arr
                temporal_shift_l2 = float(np.linalg.norm(u_exec - before))

    # --- decode allocation from logits ---
    sl_alg = slices.get("alg")
    alloc_logits = u_exec[sl_alg] if sl_alg else np.zeros(0)
    alpha = _softmax_temp(alloc_logits, exp_cfg.alloc_temp) if alloc_logits.size else np.zeros(0)
    alpha_raw = np.asarray(alpha, dtype=float).copy()

    feas      = float(getattr(pf, "feas_ratio", 0.0))
    mean_viol = float(getattr(pf, "mean_violation", 0.0))
    stall     = float(getattr(pf, "stall_flag", 0.0))

    if alpha.size:
        # --- cold start prior (修复2): 前 N 个 tick 混入问题感知先验 ---
        if not clip_only:
            cold_ticks = int(getattr(exp_cfg, "cold_start_ticks", 3))
            cold_blend = float(getattr(exp_cfg, "cold_start_blend", 0.7))
            tick_count = int(getattr(pf, "tick_count", cold_ticks))  # pf 携带 tick 信息
            if tick_count < cold_ticks:
                n_constr = int(getattr(pf, "n_constr", 0))
                prior = _cold_start_prior(algo_pool, n_constr)
                w = cold_blend * (1.0 - tick_count / max(1, cold_ticks))  # 线性衰减
                alpha = (1.0 - w) * alpha + w * prior

        if no_safety:
            s = float(np.sum(alpha))
            alpha_safe = alpha / s if s > 1e-12 else np.ones_like(alpha) / max(1, alpha.size)
            alpha_pre_project = np.asarray(alpha, dtype=float).copy()
            alpha_shift_l1 = float(np.sum(np.abs(alpha_safe - alpha_raw))) if alpha_raw.shape == alpha_safe.shape else 0.0
        elif clip_only or no_consistency:
            s = float(np.sum(alpha))
            alpha_safe = alpha / s if s > 1e-12 else np.ones_like(alpha) / max(1, alpha.size)
            alpha_pre_project = np.asarray(alpha, dtype=float).copy()
            alpha_shift_l1 = float(np.sum(np.abs(alpha_safe - alpha_raw))) if alpha_raw.shape == alpha_safe.shape else 0.0
            if alloc_logits.size:
                logits = np.log(alpha_safe + 1e-8)
                logits = logits - np.mean(logits)
                u_exec[sl_alg] = np.clip(logits / max(exp_cfg.alloc_temp, 1e-6), -5.0, 5.0)
        else:
            # heuristic floors (encourage EMO + keep some CP/MILP)
            floor = np.zeros_like(alpha)
            for i, a in enumerate(algo_pool):
                if a in ("NSGA-II", "NSGA-III", "MOEA/D"):
                    floor[i] = 0.05
                if a == "CP/MILP":
                    floor[i] = 0.05

            # risky: suppress BO, boost EMO/CP floors
            if feas < 0.2 or mean_viol > 5.0:
                for i, a in enumerate(algo_pool):
                    if a == "BO":
                        alpha[i] = min(alpha[i], 0.2)
                    if a in ("CP/MILP", "NSGA-II", "NSGA-III", "MOEA/D"):
                        floor[i] = max(floor[i], 0.1)

            # stall: keep some search diversity
            if stall > 0.6:
                for i, a in enumerate(algo_pool):
                    if a in ("NSGA-II", "NSGA-III", "MOEA/D", "RL"):
                        floor[i] = max(floor[i], 0.05)

            # Legacy HV-oriented diagnostic branch. Formal Ch5 runs keep
            # hv_boost_enabled=False, so C1 remains rule-prior + IQL residual
            # before the same consistency/safety projection path.
            method_id = normalize_method_id(getattr(exp_cfg, "exp_group", ""))
            if (method_id == "C1_PPCS_IQL_FULL") and bool(getattr(exp_cfg, "hv_boost_enabled", False)):
                sat    = float(getattr(pf, "sat", 0.0))
                sat_th = float(getattr(exp_cfg, "hv_boost_sat_th", 0.50))
                feas_th = float(getattr(exp_cfg, "hv_boost_feas_th", 0.55))
                viol_th = float(getattr(exp_cfg, "hv_boost_viol_th", 1.0))

                # CO-ARCH override
                pid = str(getattr(pf, "problem_id", ""))
                is_coarch = pid.strip().upper() == "CO-ARCH"
                ignore_sat = bool(getattr(exp_cfg, "hv_boost_ignore_sat_on_coarch", True)) and is_coarch
                if ignore_sat:
                    feas_th = max(feas_th, float(getattr(exp_cfg, "hv_boost_coarch_feas_th", 0.90)))
                    viol_th = min(viol_th, float(getattr(exp_cfg, "hv_boost_coarch_viol_th", 0.50)))
                    sat_ok = True
                else:
                    sat_ok = (sat >= sat_th)

                if sat_ok and (feas >= feas_th) and (mean_viol <= viol_th):
                    hv_boost_active = 1
                    hv_algos    = list(getattr(exp_cfg, "hv_boost_algos", ["CP/MILP", "NSGA-III", "MOEA/D", "NSGA-II"]))
                    strength    = float(getattr(exp_cfg, "hv_boost_strength", 0.60))
                    floor_hv    = float(getattr(exp_cfg, "hv_boost_floor", 0.10))
                    cp_suppress = float(getattr(exp_cfg, "hv_boost_cp_suppress", 0.0))
                    for i, a in enumerate(algo_pool):
                        if a in hv_algos:
                            alpha[i] *= (1.0 + strength)
                            floor[i] = max(float(floor[i]), floor_hv)
                        if a == "CP/MILP":
                            alpha[i] *= max(0.05, (1.0 - cp_suppress))
                    # stall 时，给 BO/RL 少量增益，避免过早陷入局部
                    if stall > 0.6:
                        for i, a in enumerate(algo_pool):
                            if a in ("BO", "RL"):
                                alpha[i] *= 1.15

            alpha_pre_project = np.asarray(alpha, dtype=float).copy()
            alpha_safe = _project_simplex_with_floor(alpha, floor)

            # enforce semantic caps after floor projection; otherwise simplex
            # renormalisation can re-introduce risky BO mass.
            risk_active = bool(feas < 0.2 or mean_viol > 5.0)
            if risk_active:
                name_to_idx = {a: i for i, a in enumerate(algo_pool)}
                bo_idx = name_to_idx.get("BO")
                safe_idx = [name_to_idx[a] for a in ("CP/MILP", "NSGA-II", "NSGA-III", "MOEA/D") if a in name_to_idx]
                if bo_idx is not None and bo_idx < len(alpha_safe) and float(alpha_safe[bo_idx]) > 0.2:
                    excess = float(alpha_safe[bo_idx] - 0.2)
                    alpha_safe[bo_idx] = 0.2
                    valid_safe = [i for i in safe_idx if i < len(alpha_safe) and i != bo_idx]
                    if valid_safe:
                        weights = np.asarray([max(float(alpha_safe[i]), 1e-8) for i in valid_safe], dtype=float)
                        weights = weights / max(float(np.sum(weights)), 1e-12)
                        for i, w in zip(valid_safe, weights):
                            alpha_safe[i] += excess * float(w)
                    alpha_safe = alpha_safe / max(float(np.sum(alpha_safe)), 1e-12)

            if stall > 0.6:
                name_to_idx = {a: i for i, a in enumerate(algo_pool)}
                search_idx = [name_to_idx[a] for a in ("NSGA-II", "NSGA-III", "MOEA/D", "RL") if a in name_to_idx]
                valid_search = [i for i in search_idx if i < len(alpha_safe)]
                search_mass = float(np.sum(alpha_safe[valid_search])) if valid_search else 0.0
                if valid_search and search_mass < 0.25:
                    need = 0.25 - search_mass
                    donors = [i for i in range(len(alpha_safe)) if i not in valid_search]
                    donor_mass = float(np.sum(alpha_safe[donors])) if donors else 0.0
                    take = min(need, max(0.0, donor_mass - 1e-8))
                    if take > 0.0:
                        for i in donors:
                            alpha_safe[i] -= take * float(alpha_safe[i]) / max(donor_mass, 1e-12)
                        for i in valid_search:
                            alpha_safe[i] += take / len(valid_search)
                        alpha_safe = alpha_safe / max(float(np.sum(alpha_safe)), 1e-12)
            alpha_shift_l1 = float(np.sum(np.abs(alpha_safe - alpha_raw))) if alpha_raw.shape == alpha_safe.shape else 0.0

            # write back safe logits (so applied action is aligned with execution)
            if alloc_logits.size:
                logits = np.log(alpha_safe + 1e-8)
                logits = logits - np.mean(logits)
                u_alloc_exec = np.clip(logits / max(exp_cfg.alloc_temp, 1e-6), -5.0, 5.0)
                u_exec[sl_alg] = u_alloc_exec
    else:
        alpha_safe = alpha

    # --- op logits safety (noop bias) ---
    sl_op = slices.get("op")
    op_logits = u_exec[sl_op] if sl_op else np.zeros(0)
    op_logits_raw = np.asarray(op_logits, dtype=float).copy()
    if op_logits.size and (not no_safety) and (not clip_only) and (not no_consistency):
        conf = float(getattr(pf, "conf", 1.0))
        if conf < 0.3 or mean_viol > 5.0:
            op_logits[0] = max(float(op_logits[0]), float(np.max(op_logits)) + max(1.0, float(exp_cfg.op_noop_bias)))
        op_logits = np.clip(op_logits, -5.0, 5.0)
        u_exec[sl_op] = op_logits
    if op_logits.size and op_logits_raw.shape == op_logits.shape:
        op_shift_l1 = float(np.sum(np.abs(op_logits - op_logits_raw)))
    op_idx = int(np.argmax(op_logits)) if op_logits.size else 0

    def _semantic_lcc(alpha_vec: np.ndarray, op_i: int) -> int:
        count = 0
        name_to_idx = {a: i for i, a in enumerate(algo_pool)}
        risk_active = bool(feas < 0.2 or mean_viol > 5.0)
        bo_idx = name_to_idx.get("BO")
        if risk_active and bo_idx is not None and bo_idx < len(alpha_vec) and float(alpha_vec[bo_idx]) > 0.2:
            count += 1
        search_idx = [name_to_idx[a] for a in ("NSGA-II", "NSGA-III", "MOEA/D", "RL") if a in name_to_idx]
        valid_idx = [i for i in search_idx if i < len(alpha_vec)]
        if stall > 0.6 and valid_idx and float(np.sum(alpha_vec[valid_idx])) < 0.25:
            count += 1
        conf_for_lcc = float(getattr(pf, "conf", 1.0))
        if op_i != 0 and (conf_for_lcc < 0.3 or mean_viol > 5.0):
            count += 1
        return int(count)

    raw_op_idx = int(np.argmax(op_logits_raw)) if op_logits_raw.size else 0
    if alpha_raw.size:
        raw_lcc_count = _semantic_lcc(alpha_raw, raw_op_idx)
        post_lcc_count = _semantic_lcc(alpha_safe, op_idx)
        lcc_count = raw_lcc_count

    # --- operator intensity + xi decode (Table 3/4) ---
    u_int = 0.0
    u_risk = 0.0
    sl_model = slices.get("model")
    if sl_model and (sl_model.stop - sl_model.start) > 0:
        u_int = float(np.clip(u_exec[sl_model.start], -1.0, 1.0))
    sl_rob = slices.get("rob")
    if sl_rob and (sl_rob.stop - sl_rob.start) > 0:
        u_risk = float(np.clip(u_exec[sl_rob.start], -1.0, 1.0))
    xi = np.zeros(6, dtype=float)
    if op_idx == 1:
        xi[1] = 0.5 * (u_int + 1.0)
    elif op_idx == 2:
        xi[2] = 0.5 * (u_int + 1.0)
    elif op_idx == 3:
        xi[3] = affine(u_int, 0.05, 0.5)
    elif op_idx == 4:
        xi[4] = 0.5 * (u_int + 1.0)
    elif op_idx == 5:
        xi[5] = affine(u_risk, 0.0, 10.0)

    osv_sample.applied = u_exec.copy()
    osv_sample.op_index = op_idx
    projection_delta = np.nan_to_num(u_exec - u_gate_raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw_norm = float(np.linalg.norm(np.nan_to_num(u_gate_raw, nan=0.0, posinf=0.0, neginf=0.0)))
    projection_l2 = float(np.linalg.norm(projection_delta))
    projection_norm = projection_l2 / max(1.0, raw_norm)
    air_flag = int((projection_l2 > 1e-8) or (numeric_clip_count > 0) or (raw_lcc_count > 0) or (post_lcc_count > 0))
    osv_sample.projection_diag = {
        "u_raw": _safe_list(u_gate_raw),
        "u_exec": _safe_list(u_exec),
        "projection_l2": projection_l2,
        "projection_norm": float(projection_norm),
        "air_flag": int(air_flag),
        "lcc_count": int(lcc_count),
        "raw_lcc_count": int(raw_lcc_count),
        "post_lcc_count": int(post_lcc_count),
        "numeric_clip_count": int(numeric_clip_count),
        "alpha_raw": _safe_list(alpha_raw),
        "alpha_pre_project": _safe_list(alpha_pre_project if alpha_pre_project.size else alpha_raw),
        "alpha_safe": _safe_list(alpha_safe),
        "alpha_shift_l1": float(alpha_shift_l1),
        "op_raw": int(raw_op_idx),
        "op_safe": int(op_idx),
        "op_shift_l1": float(op_shift_l1),
        "temporal_shift_l2": float(temporal_shift_l2),
        "hv_boost_active": int(hv_boost_active),
        "no_safety": int(no_safety),
        "clip_only": int(clip_only),
        "no_consistency": int(no_consistency),
        "no_temporal": int(no_temporal),
    }
    return osv_sample, alpha_safe, op_idx, xi


def pi_safe_params(state: "OptState", exp_cfg: "ExperimentConfig"):
    if "ABL_NO_SAFETY" in exp_cfg.ablations:
        return
    state.cx_eta  = float(np.clip(state.cx_eta, 1.0, 60.0))
    state.mut_eta = float(np.clip(state.mut_eta, 1.0, 80.0))
    beta_override = state.bo_params.get("beta_override", 1.0)
    if beta_override is None:
        beta_override = 1.0
    state.bo_params["beta_override"]  = float(np.clip(beta_override, 0.1, 20.0))
    state.bo_params["radius_scale"]   = float(np.clip(state.bo_params.get("radius_scale", 1.0), 0.2, 2.0))
    state.cpm_params["lambda_pen"]    = float(np.clip(state.cpm_params.get("lambda_pen", 1.0), 0.1, 10.0))
    state.cpm_params["ridge_l2"]      = float(np.clip(state.cpm_params.get("ridge_l2", 0.0), 0.0, 5.0))
    state.rl_params["sigma"]          = float(np.clip(state.rl_params.get("sigma", 0.2), 0.01, 2.0))
    state.rl_params["radius_scale"]   = float(np.clip(state.rl_params.get("radius_scale", 1.0), 0.2, 2.0))
    state.upv.rho["cvar_alpha"]       = float(np.clip(state.upv.rho.get("cvar_alpha", 0.9), 0.5, 0.999))


def pi_safe_model_ir(mir: ModelIR, problem, exp_cfg: "ExperimentConfig") -> ModelIR:
    if "ABL_NO_SAFETY" in exp_cfg.ablations:
        return mir
    mir.penalty_scale = float(np.clip(mir.penalty_scale, 0.1, 50.0))
    mir.bias_lambda   = float(np.clip(mir.bias_lambda, 0.0, 0.3))
    mir.trust_radius_scale = float(np.clip(mir.trust_radius_scale, 0.05, 1.5))
    if mir.obj_band:
        cleaned = []
        for lo, hi in mir.obj_band:
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
                cleaned = None
                break
            cleaned.append((float(lo), float(hi)))
        mir.obj_band = cleaned
        mir.band_penalty = float(np.clip(mir.band_penalty, 0.0, 20.0)) if cleaned else 0.0
    bounds0 = [(float(lo), float(hi)) for lo, hi in zip(problem.xl, problem.xu)]
    fixed_bounds = []
    for (lo, hi), (blo, bhi) in zip(mir.bounds, bounds0):
        lo2, hi2 = max(lo, blo), min(hi, bhi)
        if lo2 >= hi2:
            lo2, hi2 = blo, bhi
        fixed_bounds.append((lo2, hi2))
    mir.bounds = fixed_bounds
    return mir
