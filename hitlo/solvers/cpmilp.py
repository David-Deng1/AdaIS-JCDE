"""
hitlo.solvers.cpmilp
=======================
Lightweight CP/MILP approximation solver step (linear surrogates + sampling),
extracted from the monolithic ``app_v11_iqlfix.py``.

Contains: ``step_cpm(state, upv_opt, budget)``
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge

from hitlo.core.types import UPV
from hitlo.solvers.emo import _asf_distance as asf_distance
from hitlo.solvers.emo import _compute_penalty_and_feas as compute_penalty_and_feas


def step_cpm(state, upv_opt: Optional[UPV] = None,
             budget: Optional[int] = None):
    """Lightweight CP/MILP approximation (linear surrogates + sampling).

    Key fixes vs. old version:
    - Respect per-epoch budget (no implicit min-k=4).
    - Add constraint-aware surrogate penalty when base problem has constraints.

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
    upv_use = upv_opt if (upv_opt is not None) else state.upv

    X_obs = np.asarray(state.pop_X, dtype=float)
    F_obs = np.asarray(state.pop_F, dtype=float)
    G_obs = (np.asarray(state.pop_G, dtype=float)
             if state.pop_G is not None else None)
    n_constr = int(getattr(state.problem, "n_constr", 0) or 0)

    # Clean NaN/Inf
    mask = (np.isfinite(X_obs).all(axis=1)
            & np.isfinite(F_obs).all(axis=1))
    if n_constr > 0 and G_obs is not None:
        mask = mask & np.isfinite(G_obs).all(axis=1)

    if not np.all(mask):
        X_obs = X_obs[mask]
        F_obs = F_obs[mask]
        if n_constr > 0 and G_obs is not None:
            G_obs = G_obs[mask]

    if X_obs.shape[0] < 2:
        mid = 0.5 * (np.asarray(state.problem.xl, dtype=float)
                     + np.asarray(state.problem.xu, dtype=float))
        X_obs = np.nan_to_num(
            np.asarray(state.pop_X, dtype=float),
            nan=mid, posinf=state.problem.xu, neginf=state.problem.xl)
        X_obs = np.clip(X_obs, state.problem.xl, state.problem.xu)
        F_obs = np.nan_to_num(
            np.asarray(state.pop_F, dtype=float),
            nan=1e6, posinf=1e6, neginf=1e6)
        if n_constr > 0 and state.pop_G is not None:
            G_obs = np.nan_to_num(
                np.asarray(state.pop_G, dtype=float),
                nan=1e3, posinf=1e3, neginf=1e3)

    # Linear surrogates for objectives
    models = []
    ridge_l2 = float(state.cpm_params.get("ridge_l2", 0.0) or 0.0)
    for j in range(state.n_obj):
        y = F_obs[:, j]
        alpha = ridge_l2 if ridge_l2 > 0 else 1e-8
        mdl = Ridge(alpha=alpha, fit_intercept=True)
        mdl.fit(X_obs, y)
        models.append(mdl)

    # Linear surrogates for constraints (if available)
    g_models = None
    if (n_constr > 0 and G_obs is not None
            and G_obs.shape[0] == X_obs.shape[0]):
        g_models = []
        alpha_g = ridge_l2 if ridge_l2 > 0 else 1e-6
        for k in range(n_constr):
            mdl = Ridge(alpha=alpha_g, fit_intercept=True)
            mdl.fit(X_obs, G_obs[:, k])
            g_models.append(mdl)

    # Candidate sampling
    N = int(state.cpm_params.get("sample_size", 600))
    xl = np.asarray(state.problem.xl, dtype=float)
    xu = np.asarray(state.problem.xu, dtype=float)
    X_cand = rng.uniform(xl, xu, size=(N, state.n_var))

    if upv_use.D:
        for i, (lo, hi) in enumerate(upv_use.D):
            X_cand[:, i] = np.clip(X_cand[:, i], lo, hi)

    if getattr(state, "model_ir", None) is not None:
        X_cand = state.model_ir.clamp_x(X_cand)

    # Surrogate predict + score
    MU = np.column_stack([mdl.predict(X_cand) for mdl in models])
    scores = []
    cv_w = float(state.cpm_params.get("cv_penalty_weight", 1000.0))

    if g_models is not None:
        G_hat = np.column_stack([mdl.predict(X_cand) for mdl in g_models])
        cv_hat = np.sum(np.maximum(0.0, G_hat), axis=1)
    else:
        cv_hat = np.zeros(len(X_cand), dtype=float)

    for i in range(len(X_cand)):
        f_hat = MU[i]
        a = asf_distance(f_hat, upv_use.w, upv_use.r)
        p, _ = compute_penalty_and_feas(X_cand[i], f_hat, upv_use,
                                        state.enable_cr)
        scores.append(
            a + float(state.cpm_params.get("lambda_pen", 1.0)) * p
            + cv_w * float(cv_hat[i])
        )

    k = int(state.pop_size)
    if budget is not None:
        try:
            k = max(1, min(int(budget), int(state.pop_size)))
        except Exception:
            pass
    if k <= 0:
        return

    idx = np.argsort(scores)[:k]
    X_new = X_cand[idx]
    if getattr(state, "model_ir", None) is not None:
        X_new = state.model_ir.clamp_x(X_new)

    # Evaluate selected points
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
