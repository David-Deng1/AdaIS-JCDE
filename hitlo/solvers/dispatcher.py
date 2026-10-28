"""
hitlo.solvers.dispatcher
===========================
OSV-driven algorithm selection, parameter mapping, and operator application,
extracted from the monolithic ``app_v11_iqlfix.py``.

Contains:
- ``select_algo_by_osv(state, osv_sample, pf)``
- ``apply_osv_knobs(state, u_exec, exp_cfg)``
- ``apply_operator(state, op_idx, u_exec, xi)``
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from hitlo.core.types import OSV, OSVSample, ProcessFeatures, UPV
from hitlo.config.settings import ExperimentConfig
from hitlo.solvers.emo import _normalize_w as normalize_w


# ---------------------------------------------------------------------------
# Mapping helpers (ported from app_v11_iqlfix.py L1392-1401)
# ---------------------------------------------------------------------------

def _affine(z: float, lo: float, hi: float) -> float:
    return float(lo + (z + 1.0) * 0.5 * (hi - lo))


def _round_affine(z: float, lo: float, hi: float) -> int:
    return int(round(_affine(z, lo, hi)))


def _log10_map(z: float, lo: float, hi: float) -> float:
    lo_l = np.log10(lo)
    hi_l = np.log10(hi)
    return float(10 ** _affine(z, lo_l, hi_l))


def _pi_safe_params(state, exp_cfg: ExperimentConfig):
    """Safety projection for solver parameters (Eq.(18) safety layer)."""
    if "ABL_NO_SAFETY" in exp_cfg.ablations:
        return
    state.cx_eta = float(np.clip(state.cx_eta, 1.0, 60.0))
    state.mut_eta = float(np.clip(state.mut_eta, 1.0, 80.0))
    beta_override = state.bo_params.get("beta_override", 1.0)
    if beta_override is None:
        beta_override = 1.0
    state.bo_params["beta_override"] = float(
        np.clip(beta_override, 0.1, 20.0))
    state.bo_params["radius_scale"] = float(
        np.clip(state.bo_params.get("radius_scale", 1.0), 0.2, 2.0))
    state.cpm_params["lambda_pen"] = float(
        np.clip(state.cpm_params.get("lambda_pen", 1.0), 0.1, 10.0))
    state.cpm_params["ridge_l2"] = float(
        np.clip(state.cpm_params.get("ridge_l2", 0.0), 0.0, 5.0))
    state.rl_params["sigma"] = float(
        np.clip(state.rl_params.get("sigma", 0.2), 0.01, 2.0))
    state.rl_params["radius_scale"] = float(
        np.clip(state.rl_params.get("radius_scale", 1.0), 0.2, 2.0))
    state.upv.rho["cvar_alpha"] = float(
        np.clip(state.upv.rho.get("cvar_alpha", 0.9), 0.5, 0.999))


# ---------------------------------------------------------------------------
# select_algo_by_osv
# ---------------------------------------------------------------------------

def select_algo_by_osv(state, alpha_safe: np.ndarray,
                       algo_pool: list) -> str:
    """Single-algorithm dispatch: sample one algorithm from safe probability.

    Unlike portfolio mode (which allocates budget to ALL algorithms
    proportionally), this mode picks exactly ONE algorithm per tick and
    gives it the entire evaluation budget.  Safety shielding is already
    applied upstream in ``pi_safe_osv_sample()``, so this function only
    performs multinomial sampling.

    Parameters
    ----------
    state : OptState
        Optimisation state (read-only access to ``seed``, ``tick_count``).
    alpha_safe : np.ndarray
        Post-safety-projection algorithm probability vector (sums to 1).
    algo_pool : list[str]
        Ordered list of algorithm names matching ``alpha_safe``.

    Returns
    -------
    str
        Name of the selected algorithm.
    """
    rng = np.random.default_rng(state.seed + state.tick_count)

    p = np.asarray(alpha_safe, dtype=float).copy()
    p = np.maximum(p, 1e-9)
    p = p / np.sum(p)

    choice = str(rng.choice(algo_pool, p=p))
    return choice


# ---------------------------------------------------------------------------
# apply_osv_knobs
# ---------------------------------------------------------------------------

def apply_osv_knobs(state, u_exec: np.ndarray,
                    exp_cfg: ExperimentConfig) -> Dict[str, float]:
    """Map OSV continuous channels to solver knobs (Table 4).

    Parameters
    ----------
    state : OptState
        Optimisation state (modified in-place: EMO / BO / CPM / RL params).
    u_exec : np.ndarray
        The applied (post-gating, post-safety) OSV action vector.
    exp_cfg : ExperimentConfig
        Experiment configuration (checked for ablation flags).

    Returns
    -------
    dict
        Decoded knob values for logging.
    """
    slices = (state.osv._channel_slices()
              if state.osv is not None
              else OSV.from_upv(state.upv,
                                state.algo_pool)._channel_slices())

    o_param = (np.asarray(u_exec[slices.get("param")], dtype=float)
               if slices.get("param") else np.zeros(4, dtype=float))
    o_rob = (np.asarray(u_exec[slices.get("rob")], dtype=float)
             if slices.get("rob") else np.zeros(2, dtype=float))
    o_model = (np.asarray(u_exec[slices.get("model")], dtype=float)
               if slices.get("model") else np.zeros(4, dtype=float))

    uexpl = float(np.clip(
        o_param[0] if o_param.size > 0 else 0.0, -1.0, 1.0))
    ucon = float(np.clip(
        o_param[1] if o_param.size > 1 else 0.0, -1.0, 1.0))
    uroi = float(np.clip(
        o_param[2] if o_param.size > 2 else 0.0, -1.0, 1.0))
    uloc = float(np.clip(
        o_param[3] if o_param.size > 3 else 0.0, -1.0, 1.0))
    urisk = float(np.clip(
        o_rob[0] if o_rob.size > 0 else 0.0, -1.0, 1.0))
    u_int = float(np.clip(
        o_model[0] if o_model.size > 0 else 0.0, -1.0, 1.0))

    if "ABL_NO_PARAM" not in exp_cfg.ablations:
        # EMO knobs (uexpl)
        state.cx_prob = float(
            np.clip(_affine(uexpl, 0.7, 0.95), 0.0, 1.0))
        state.mut_prob = float(
            np.clip(_affine(uexpl, 0.05, 0.3), 0.0, 1.0))
        state.cx_eta = float(
            np.clip(_affine(-uexpl, 5.0, 30.0), 1.0, 60.0))
        state.mut_eta = float(
            np.clip(_affine(-uexpl, 10.0, 50.0), 1.0, 80.0))
        state.emo_params["roi_bias"] = float(
            np.clip(0.5 * (uroi + 1.0), 0.0, 1.0))

        # BO knobs
        state.bo_params["kappa"] = float(
            np.clip(_affine(uexpl, 0.5, 5.0), 0.1, 20.0))
        state.bo_params["q"] = int(
            np.clip(_round_affine(uexpl, 1, 8), 1, 10))
        state.bo_params["delta_bo"] = float(
            np.clip(_affine(uloc, 0.05, 0.5), 0.05, 0.5))
        state.bo_params["beta_override"] = float(
            state.bo_params["kappa"])
        # OSV.rob (feas) → BO 约束惩罚权重 (S1.2: ABL_NO_OSV_ROB 关闭时 cv_w=0,
        # 让 BO 在强约束问题上完全失去可行性引导, 体现 ROB 通道独立必要性)
        if "ABL_NO_OSV_ROB" in exp_cfg.ablations:
            state.bo_params["cv_penalty_weight"] = 0.0
        else:
            state.bo_params["cv_penalty_weight"] = float(
                np.clip(_log10_map(urisk, 10.0, 10000.0), 10.0, 10000.0))
        # OSV.alg (explore) → BO 算法选择 (S1.3: ABL_NO_OSV_ALG 关闭 BO 评分,
        # 退化为纯 random sampling baseline, 体现 alg 通道独立必要性)
        state.bo_params["bo_disabled"] = bool(
            "ABL_NO_OSV_ALG" in exp_cfg.ablations)
        # OSV.model (rebuild) → BO GP 训练样本规模 (S1.4: ABL_NO_OSV_MODEL
        # 关闭时锁定 GP 训练样本数为冷启动 5 个, GP 永不学习新数据,
        # 体现 model 通道作为 surrogate 重训驱动的独立必要性)
        if "ABL_NO_OSV_MODEL" in exp_cfg.ablations:
            state.bo_params["gp_train_cap"] = 5  # GP 永远只用前 5 个观测
        else:
            state.bo_params["gp_train_cap"] = int(
                np.clip(_round_affine(u_int, 20, 1000), 20, 1000))
        # OSV.pref → BO 偏好引导开关 (S1.6: ABL_NO_OSV_PREF 让 BO ASF 计算失去
        # UPV.r/w 的偏好塑形, 用观测中性参考点 + 均匀权重替代,
        # 体现 pref 通道作为偏好塑形的独立必要性)
        state.bo_params["ignore_upv_pref"] = bool(
            "ABL_NO_OSV_PREF" in exp_cfg.ablations)

        # CP/MILP knobs
        state.cpm_params["sample_size"] = int(
            np.clip(_round_affine(uexpl, 200, 900), 50, 2000))
        state.cpm_params["lambda_pen"] = float(
            np.clip(_log10_map(ucon, 1.0, 1e4), 1.0, 1e4))
        state.cpm_params["ridge_l2"] = float(
            np.clip(_log10_map(ucon, 1e-4, 1e2), 1e-4, 1e2))
        state.cpm_params["tau_milp"] = float(
            np.clip(_round_affine(ucon, 10, 300), 10, 300))
        state.cpm_params["delta_cp"] = float(
            np.clip(_affine(uloc, 0.05, 0.5), 0.05, 0.5))

        # RL/CEM knobs (stored for step_rl)
        state.rl_params["alpha_cvar"] = float(
            np.clip(_affine(urisk, 0.5, 0.95), 0.5, 0.95))
        state.rl_params["lambda_risk"] = float(
            np.clip(_affine(urisk, 0.0, 10.0), 0.0, 10.0))
        state.rl_params["delta_cem"] = float(
            np.clip(_affine(uloc, 0.05, 0.5), 0.05, 0.5))

    _pi_safe_params(state, exp_cfg)
    state._last_osv_knobs = {
        "uexpl": uexpl, "ucon": ucon, "uroi": uroi, "uloc": uloc,
        "urisk": urisk, "u_int": u_int
    }
    return dict(state._last_osv_knobs)


# ---------------------------------------------------------------------------
# apply_operator
# ---------------------------------------------------------------------------

def apply_operator(state, op_idx: int, u_exec: np.ndarray,
                   xi: np.ndarray):
    """Apply operator effects R1-R5 (Table 3).

    Parameters
    ----------
    state : OptState
        Optimisation state (modified in-place).
    op_idx : int
        Operator index (0 = NOOP, 1..5 = R1..R5).
    u_exec : np.ndarray
        The applied OSV action vector.
    xi : np.ndarray
        Per-operator parameter vector.
    """
    if op_idx <= 0:
        return

    slices = (state.osv._channel_slices()
              if state.osv is not None
              else OSV.from_upv(state.upv,
                                state.algo_pool)._channel_slices())
    sl_rob = slices.get("rob")
    urisk = 0.0
    if sl_rob and (sl_rob.stop - sl_rob.start) > 0:
        urisk = float(np.clip(u_exec[sl_rob.start], -1.0, 1.0))

    if op_idx == 1:
        gamma_pen = float(np.clip(xi[1], 0.0, 1.0))
        state.cpm_params["lambda_pen"] *= float(
            np.exp(np.log(10.0) * gamma_pen))

    elif op_idx == 2:
        rho = float(np.clip(xi[2], 0.0, 1.0))
        if state.pref_gp is not None:
            pred = state.pref_gp.predict(state.pf.as_vector(),
                                         return_std=True)
            if pred is not None:
                w_hat, r_hat, _, _ = pred
                w_hat = normalize_w(np.maximum(1e-6, w_hat))
                state.upv.w = normalize_w(
                    (1.0 - rho) * state.upv.w + rho * w_hat)
                state.upv.r = ((1.0 - rho) * state.upv.r
                               + rho * np.asarray(r_hat, dtype=float))

    elif op_idx == 3:
        delta = float(np.clip(xi[3], 0.05, 0.5))
        state.bo_params["delta_bo"] = delta
        state.cpm_params["delta_cp"] = delta
        state.rl_params["delta_cem"] = delta

    elif op_idx == 4:
        kappa_roi = float(np.clip(xi[4], 0.0, 1.0))
        state.emo_params["roi_bias"] = kappa_roi

    elif op_idx == 5:
        alpha_cvar = float(
            np.clip(_affine(urisk, 0.5, 0.95), 0.5, 0.95))
        lambda_risk = float(
            np.clip(_affine(urisk, 0.0, 10.0), 0.0, 10.0))
        state.upv.rho["cvar_alpha"] = alpha_cvar
        state.rl_params["alpha_cvar"] = alpha_cvar
        state.rl_params["lambda_risk"] = lambda_risk
