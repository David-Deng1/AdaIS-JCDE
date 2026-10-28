"""
hitlo.solvers.rl_cem
=======================
CEM-style policy search solver step on the decision space, extracted from the
monolithic ``app_v11_iqlfix.py``.

Contains: ``step_rl(state, upv_opt, budget)``
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from hitlo.core.types import UPV
from hitlo.solvers.emo import _asf_distance as asf_distance
from hitlo.solvers.emo import _compute_penalty_and_feas as compute_penalty_and_feas


# ---------------------------------------------------------------------------
# Helper affine maps (duplicated from dispatcher to keep rl_cem self-contained)
# ---------------------------------------------------------------------------

def _affine(z: float, lo: float, hi: float) -> float:
    return float(lo + (z + 1.0) * 0.5 * (hi - lo))


def _round_affine(z: float, lo: float, hi: float) -> int:
    return int(round(_affine(z, lo, hi)))


# ---------------------------------------------------------------------------
# step_rl
# ---------------------------------------------------------------------------

def step_rl(state, upv_opt: Optional[UPV] = None,
            budget: Optional[int] = None):
    """CEM-style policy search on decision space.

    Key fixes vs. old version:
    - Respect per-epoch budget: evaluates exactly ``budget`` new candidates
      (if given).
    - Uses existing population as additional samples for CEM update (no extra
      eval cost).
    - Adds base-constraint violation into cost (important for constrained
      benchmarks like CO-ARCH).
    - Risk knob (CVaR) now affects ordering via a tail penalty (instead of
      adding a constant).

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

    knobs = getattr(state, "_last_osv_knobs", {}) or {}
    uexpl = float(knobs.get("uexpl", 0.0))
    urisk = float(knobs.get("urisk", 0.0))
    uloc = float(knobs.get("uloc", 0.0))

    # How many *new* candidates to evaluate this tick
    N_base = int(np.clip(_round_affine(uexpl, 50, 500), 50, 500))
    if budget is not None:
        try:
            N_new = int(max(0, int(budget)))
        except Exception:
            N_new = N_base
    else:
        N_new = N_base

    if N_new <= 0:
        return

    elite_ratio = float(np.clip(_affine(uexpl, 0.05, 0.20), 0.05, 0.20))
    beta_smooth = float(np.clip(_affine(uexpl, 0.5, 0.9), 0.5, 0.9))
    alpha_cvar = float(np.clip(_affine(urisk, 0.5, 0.95), 0.5, 0.95))
    lambda_risk = float(np.clip(_affine(urisk, 0.0, 10.0), 0.0, 10.0))
    delta_cem = float(np.clip(_affine(uloc, 0.05, 0.5), 0.05, 0.5))

    # Init mean/std from population if missing
    if state._rl_mean is None:
        state._rl_mean = np.mean(state.pop_X, axis=0)
    if state._rl_std is None:
        state._rl_std = np.std(state.pop_X, axis=0) + 1e-3

    xl = np.asarray(state.problem.xl, dtype=float)
    xu = np.asarray(state.problem.xu, dtype=float)
    sigma = (float(state.rl_params.get("sigma", 0.15))
             * float(state.rl_params.get("radius_scale", 1.0)))

    # Sample new candidates around current distribution
    eps = rng.normal(0.0, sigma, size=(N_new, state.n_var))
    X_new = np.clip(state._rl_mean + state._rl_std * eps, xl, xu)

    if upv_use.D:
        for i, (lo, hi) in enumerate(upv_use.D):
            X_new[:, i] = np.clip(X_new[:, i], lo, hi)

    if getattr(state, "model_ir", None) is not None:
        X_new = state.model_ir.clamp_x(X_new)

    # Evaluate new candidates (costly)
    n_constr = int(getattr(state.problem, "n_constr", 0) or 0)
    F_new = []
    G_new = []
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

        F_new.append(np.asarray(f, dtype=float))
        if n_constr > 0:
            G_new.append(np.asarray(g, dtype=float))

    F_new = np.asarray(F_new, dtype=float)
    G_new = (np.asarray(G_new, dtype=float) if n_constr > 0 else None)

    # Combine with existing population (no extra eval cost)
    X_exist = np.asarray(state.pop_X, dtype=float)
    F_exist = np.asarray(state.pop_F, dtype=float)

    if n_constr > 0:
        G_exist = state.pop_G
        if G_exist is None or len(G_exist) != len(X_exist):
            # Try to recover constraints for existing pop (treat as cached;
            # no eval_count update)
            try:
                vals = state.problem.evaluate(X_exist,
                                              return_values_of=["G"])
                if isinstance(vals, dict):
                    G_exist = vals.get("G", None)
                elif isinstance(vals, (tuple, list)) and len(vals) > 0:
                    G_exist = vals[-1]
                else:
                    G_exist = None
            except Exception:
                G_exist = None
        G_exist = (np.asarray(G_exist, dtype=float)
                   if G_exist is not None
                   else np.zeros((len(X_exist), n_constr), dtype=float))
        G_all = (np.vstack([G_exist, G_new])
                 if G_new is not None else G_exist)
    else:
        G_all = None

    X_all = np.vstack([X_exist, X_new])
    F_all = np.vstack([F_exist, F_new])

    # Cost = ASF + UPV penalty + (optional) base constraint penalty
    cv_w = float(state.rl_params.get("cv_penalty_weight", 1000.0))
    costs = []
    for i in range(len(X_all)):
        a = asf_distance(F_all[i], upv_use.w, upv_use.r)
        p, _ = compute_penalty_and_feas(X_all[i], F_all[i], upv_use,
                                        state.enable_cr)
        cv = 0.0
        if n_constr > 0 and G_all is not None:
            cv = float(np.sum(np.maximum(0.0, G_all[i])))
        costs.append(float(a) + float(p) + cv_w * cv)
    costs = np.asarray(costs, dtype=float)

    # CVaR-like tail penalty (affects ordering)
    q = (float(np.quantile(costs, alpha_cvar))
         if len(costs) > 0 else 0.0)
    tail_pen = np.maximum(0.0, costs - q)
    costs_adj = costs + lambda_risk * tail_pen

    # Select elites from combined set
    n_elite = max(1, int(round(elite_ratio * len(X_all))))
    n_elite = int(np.clip(n_elite, 1, len(X_all)))
    elite_idx = np.argsort(costs_adj)[:n_elite]
    elite = X_all[elite_idx]

    m = np.mean(elite, axis=0)
    s = np.std(elite, axis=0) + 1e-3

    # Update distribution (smooth + trust shrink)
    state._rl_mean = beta_smooth * m + (1.0 - beta_smooth) * state._rl_mean
    state._rl_std = beta_smooth * s + (1.0 - beta_smooth) * state._rl_std
    state._rl_std = np.maximum(state._rl_std, 1e-3)
    state._rl_std = ((1.0 - delta_cem) * state._rl_std
                     + delta_cem * s)

    # New population: keep best k (feasibility-first via cv_w in costs_adj)
    k_pop = min(int(state.pop_size), len(X_all))
    pop_idx = np.argsort(costs_adj)[:k_pop]
    state.pop_X = X_all[pop_idx]
    state.pop_F = F_all[pop_idx]
    if n_constr > 0 and G_all is not None:
        state.pop_G = G_all[pop_idx]
    else:
        state.pop_G = None

    state._update_history()
