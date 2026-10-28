"""
hitlo.solvers.bayesian
=========================
Lightweight Bayesian Optimisation step (GPR surrogates + local/global
candidates), extracted from the monolithic ``app_v11_iqlfix.py``.

Contains: ``step_bo(state, upv_opt, budget)``
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge

from hitlo.core.types import UPV
from hitlo.solvers.emo import _asf_distance as asf_distance
from hitlo.solvers.emo import _compute_penalty_and_feas as compute_penalty_and_feas


def _step_bo_random(state, rng, budget):
    """S1.3: ABL_NO_OSV_ALG 退化为纯 random sampling baseline.

    跳过 GP 拟合 + UCB 评分, 直接在搜索域均匀采样,
    体现 alg 通道关闭后 BO 失去算法选择能力 → 朴素探索.
    """
    xl = np.asarray(state.problem.xl, dtype=float)
    xu = np.asarray(state.problem.xu, dtype=float)
    n_constr = int(getattr(state.problem, "n_constr", 0) or 0)
    k = int(state.pop_size)
    if budget is not None:
        try:
            k = max(1, min(int(budget), int(state.pop_size)))
        except Exception:
            pass
    if k <= 0:
        return
    X_new = rng.uniform(xl, xu, size=(k, state.n_var))
    if getattr(state, "model_ir", None) is not None:
        X_new = state.model_ir.clamp_x(X_new)
    F_list = []
    G_list = []
    for i in range(len(X_new)):
        f, g, failed = state._safe_eval(X_new[i])
        if failed or f is None:
            f = np.ones(state.n_obj, dtype=float) * 1e6
            if n_constr > 0:
                g = np.ones(n_constr, dtype=float) * 1e3
        else:
            state._record_eval(1, 0)
            if n_constr > 0 and g is None:
                g = np.zeros(n_constr, dtype=float)
        F_list.append(np.asarray(f, dtype=float))
        if n_constr > 0:
            G_list.append(np.asarray(g, dtype=float))
    state.pop_X = X_new
    state.pop_F = np.asarray(F_list, dtype=float)
    state.pop_G = (np.asarray(G_list, dtype=float) if n_constr > 0 else None)
    state._update_history()


def step_bo(state, upv_opt: Optional[UPV] = None,
            budget: Optional[int] = None):
    """Lightweight BO step (GPR surrogates + local/global candidates).

    Key fixes vs. old version:
    - Respect per-epoch budget (no implicit min-k=4).
    - Disable GPR hyperparameter optimisation (optimizer=None) to remove
      convergence warnings and speed up.
    - Add constraint-aware surrogate penalty when base problem has
      constraints (n_constr > 0).

    Parameters
    ----------
    state : OptState
        Optimisation state (modified in-place).
    upv_opt : UPV, optional
        Override UPV for this step.
    budget : int, optional
        Evaluation budget for this tick.
    """
    rng = np.random.default_rng(state.seed + len(state.history_hv))

    # S1.3: ABL_NO_OSV_ALG 退化为纯 random sampling
    if bool(state.bo_params.get("bo_disabled", False)):
        return _step_bo_random(state, rng, budget)

    upv_use = upv_opt if (upv_opt is not None) else state.upv

    # Observations (prefer BO replay buffer; fallback to current population)
    X_obs = (state._bo_X if getattr(state, "_bo_X", None) is not None
             else state.pop_X)
    F_obs = (state._bo_F if getattr(state, "_bo_F", None) is not None
             else state.pop_F)
    G_obs = getattr(state, "_bo_G", None)
    if G_obs is None:
        G_obs = state.pop_G

    if X_obs is None or F_obs is None or len(X_obs) < 5:
        X_obs, F_obs, G_obs = state.pop_X, state.pop_F, state.pop_G

    X_obs = np.asarray(X_obs, dtype=float)
    F_obs = np.asarray(F_obs, dtype=float)
    if G_obs is not None:
        G_obs = np.asarray(G_obs, dtype=float)

    # Clean NaN/Inf
    mask = np.isfinite(X_obs).all(axis=1) & np.isfinite(F_obs).all(axis=1)
    if G_obs is not None:
        mask = mask & np.isfinite(G_obs).all(axis=1)
    if not np.all(mask):
        X_obs = X_obs[mask]
        F_obs = F_obs[mask]
        if G_obs is not None:
            G_obs = G_obs[mask]

    if X_obs.shape[0] < 5:
        # Fallback to current pop (as a last resort)
        X_obs = np.asarray(state.pop_X, dtype=float)
        F_obs = np.asarray(state.pop_F, dtype=float)
        G_obs = (np.asarray(state.pop_G, dtype=float)
                 if state.pop_G is not None else None)

    # S1.4: ABL_NO_OSV_MODEL 限定 GP 训练样本规模 (永不学习新观测)
    gp_cap = state.bo_params.get("gp_train_cap", None)
    if gp_cap is not None:
        gp_cap = int(gp_cap)
        if X_obs.shape[0] > gp_cap:
            X_obs = X_obs[:gp_cap]
            F_obs = F_obs[:gp_cap]
            if G_obs is not None:
                G_obs = G_obs[:gp_cap]

    # Fit GPR surrogates for objectives (fixed kernel; no optimizer)
    kernel_l = float(state.bo_params.get("kernel_l", 0.3))
    kernel = RBF(length_scale=kernel_l)
    gprs = []
    for j in range(state.n_obj):
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=state.seed,
            optimizer=None,
        )
        gpr.fit(X_obs, F_obs[:, j])
        gprs.append(gpr)

    # Optional constraint surrogates
    n_constr = int(getattr(state.problem, "n_constr", 0) or 0)
    g_models = None
    if (n_constr > 0 and G_obs is not None
            and G_obs.shape[0] == X_obs.shape[0]):
        g_models = []
        ridge_alpha = float(state.bo_params.get("ridge_l2", 1e-8))
        for k in range(n_constr):
            mdl = Ridge(alpha=ridge_alpha, fit_intercept=True)
            mdl.fit(X_obs, G_obs[:, k])
            g_models.append(mdl)

    # Candidate set: local + global
    q = int(state.bo_params.get("q", 2))
    n_local = max(1, int(q * int(state.pop_size)))
    n_global = int(state.bo_params.get("n_global", 40))

    # S1.6: ABL_NO_OSV_PREF 让 BO ASF 计算用中性参考点 + 均匀权重 (无偏好引导)
    if bool(state.bo_params.get("ignore_upv_pref", False)):
        r_use = np.min(F_obs, axis=0)
        w_use = np.ones_like(upv_use.w) / max(len(upv_use.w), 1)
    else:
        r_use = np.asarray(upv_use.r, dtype=float)
        w_use = np.asarray(upv_use.w, dtype=float)

    d = np.array([asf_distance(F_obs[i], w_use, r_use)
                  for i in range(len(F_obs))], dtype=float)
    x_center = X_obs[int(np.argmin(d))]
    radius = (float(state.bo_params.get("delta_bo", 0.2))
              * float(state.bo_params.get("radius_scale", 1.0)))
    xl = np.asarray(state.problem.xl, dtype=float)
    xu = np.asarray(state.problem.xu, dtype=float)

    X_loc = np.clip(rng.normal(x_center, radius,
                               size=(n_local, state.n_var)), xl, xu)
    X_glb = rng.uniform(xl, xu, size=(n_global, state.n_var))

    if upv_use.D:
        for i, (lo, hi) in enumerate(upv_use.D):
            X_loc[:, i] = np.clip(X_loc[:, i], lo, hi)
            X_glb[:, i] = np.clip(X_glb[:, i], lo, hi)

    X_cand = np.vstack([X_loc, X_glb])

    if getattr(state, "model_ir", None) is not None:
        X_cand = state.model_ir.clamp_x(X_cand)

    # UCB-like score on scalarised surrogate objectives
    alpha = float(upv_use.rho.get("cvar_alpha", 1.0))
    beta_override = state.bo_params.get("beta_override", None)
    kappa = state.bo_params.get("kappa", None)
    if beta_override is not None:
        beta = float(beta_override)
    elif kappa is not None:
        beta = float(kappa)
    else:
        beta = 1.0 + 2.0 * (1.0 - alpha)

    mu_list = []
    std_list = []
    for gpr in gprs:
        m, s = gpr.predict(X_cand, return_std=True)
        mu_list.append(m.reshape(-1, 1))
        std_list.append(s.reshape(-1, 1))
    MU = np.hstack(mu_list)
    SD = np.hstack(std_list)

    ASF = np.array([asf_distance(MU[i], w_use, r_use)
                    for i in range(len(X_cand))],
                   dtype=float).reshape(-1, 1)
    score = ASF - beta * np.mean(SD, axis=1, keepdims=True)

    # UPV penalty
    P = []
    for i in range(len(X_cand)):
        p, _ = compute_penalty_and_feas(X_cand[i], MU[i], upv_use,
                                        state.enable_cr)
        P.append(p)
    score = score + np.asarray(P, dtype=float).reshape(-1, 1)

    # Base-constraint penalty (predicted)
    if g_models is not None:
        G_hat = np.column_stack([mdl.predict(X_cand) for mdl in g_models])
        cv_hat = np.sum(np.maximum(0.0, G_hat), axis=1, keepdims=True)
        cv_w = float(state.bo_params.get("cv_penalty_weight", 1000.0))
        score = score + cv_w * cv_hat

    # Select top-k and evaluate
    k = int(state.pop_size)
    if budget is not None:
        try:
            k = max(1, min(int(budget), int(state.pop_size)))
        except Exception:
            pass
    if k <= 0:
        return

    idx = np.argsort(score.ravel())[:k]
    X_new = X_cand[idx]
    if getattr(state, "model_ir", None) is not None:
        X_new = state.model_ir.clamp_x(X_new)

    F_list = []
    G_list = []
    for i in range(len(X_new)):
        f, g, failed = state._safe_eval(X_new[i])
        if failed or f is None:
            f = np.ones(state.n_obj, dtype=float) * 1e6
            if n_constr > 0:
                g = np.ones(n_constr, dtype=float) * 1e3
        else:
            state._record_eval(1, 0)
            if n_constr > 0 and g is None:
                g = np.zeros(n_constr, dtype=float)

        F_list.append(np.asarray(f, dtype=float))
        if n_constr > 0:
            G_list.append(np.asarray(g, dtype=float))

    state.pop_X = X_new
    state.pop_F = np.asarray(F_list, dtype=float)
    state.pop_G = (np.asarray(G_list, dtype=float) if n_constr > 0 else None)

    # Update BO replay buffer
    try:
        state._bo_X = np.vstack([X_obs, state.pop_X])[
            -(3 * int(state.pop_size)):]
        state._bo_F = np.vstack([F_obs, state.pop_F])[
            -(3 * int(state.pop_size)):]
        if n_constr > 0:
            if G_obs is None:
                G_obs = state.pop_G
            state._bo_G = np.vstack([
                np.asarray(G_obs, dtype=float), state.pop_G
            ])[-(3 * int(state.pop_size)):]
        else:
            state._bo_G = None
    except Exception:
        state._bo_X = state.pop_X.copy()
        state._bo_F = state.pop_F.copy()
        state._bo_G = (state.pop_G.copy()
                       if (n_constr > 0 and state.pop_G is not None)
                       else None)

    state._update_history()
