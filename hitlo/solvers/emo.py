"""
hitlo.solvers.emo
====================
Evolutionary multi-objective optimisation solver step, extracted from the
monolithic ``app_v11_iqlfix.py``.

Contains:
- ``WrappedProblem``  — pymoo problem wrapper that applies UPV penalty / shaping
- ``build_emo_algorithm(state, upv_opt)`` — construct the pymoo algorithm object
- ``step_emo(state, upv_opt, budget)`` — run one EMO tick
"""

from __future__ import annotations

import inspect
from typing import Dict, Optional, Tuple

import numpy as np

from hitlo.core.objective_scale import scaled
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.to_bound import ToBoundOutOfBoundsRepair
from pymoo.operators.sampling.rnd import FloatRandomSampling

from hitlo.core.types import UPV, ModelIR



# ---------------------------------------------------------------------------
# JCDE-R1 determinism fix: pymoo's @default_random_state decorator creates
# ``np.random.default_rng(None)`` (OS entropy) whenever a caller omits
# ``random_state``; NSGA-III niching reaches such a path, so identical runs could
# diverge on ties. Draw the missing seed from numpy's global RNG instead, which
# the experiment runner seeds from the run seed.
# ---------------------------------------------------------------------------
import numpy as _numpy
import pymoo.util as _pymoo_util


class _SeededRandomProxy:
    def __getattr__(self, name):
        return getattr(_numpy.random, name)

    @staticmethod
    def default_rng(seed=None):
        if seed is None:
            seed = int(_numpy.random.randint(0, 2**31 - 1))
        return _numpy.random.default_rng(seed)


class _NumpyProxy:
    random = _SeededRandomProxy()

    def __getattr__(self, name):
        return getattr(_numpy, name)


_pymoo_util.np = _NumpyProxy()

# ---------------------------------------------------------------------------
# pymoo compatibility helpers
# ---------------------------------------------------------------------------

def _pymoo_constr_init_kwargs(n_ieq: int = 0, n_eq: int = 0) -> Dict[str, int]:
    """Return the right constraint kwargs for the installed pymoo version."""
    try:
        sig = inspect.signature(ElementwiseProblem.__init__)
        if "n_ieq_constr" in sig.parameters:
            return {"n_ieq_constr": int(n_ieq), "n_eq_constr": int(n_eq)}
    except Exception:
        pass
    return {"n_constr": int(n_ieq) + int(n_eq)}


# ---------------------------------------------------------------------------
# Utility re-exports used across the solvers sub-package
# ---------------------------------------------------------------------------

def _normalize_w(w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 1e-12)
    return w / np.sum(w)


def _asf_distance(f: np.ndarray, w: np.ndarray, r: np.ndarray,
                  rho: float = 1e-6) -> float:
    d = scaled(w * np.abs(f - r))  # objective scales (preference model v2); identity by default
    return float(np.max(d) + rho * np.sum(d))


def _compute_penalty_and_feas(x: np.ndarray, f: np.ndarray,
                              upv: UPV, cr_on: bool) -> Tuple[float, bool]:
    """Compute UPV-driven penalty and feasibility flag."""
    P = 0.0
    feas = True

    # Variable domain check
    if upv.D:
        for i, (lo, hi) in enumerate(upv.D):
            if x[i] < lo - 1e-12 or x[i] > hi + 1e-12:
                P += (1000.0 if cr_on else 10.0)
                feas = False
                break

    # Hard constraints
    for c in upv.C_h:
        val = x[c.idx] if c.kind == "var" else f[c.idx]
        if c.sign == "<=":
            ok = val <= c.value + 1e-12
        elif c.sign == ">=":
            ok = val >= c.value - 1e-12
        else:
            ok = abs(val - c.value) <= 1e-9
        if not ok:
            P += (1000.0 if cr_on else 10.0)
            feas = False

    # Soft constraints
    for c in upv.C_s:
        val = x[c.idx] if c.kind == "var" else f[c.idx]
        if c.sign == "<=":
            v = max(0.0, val - c.value)
        elif c.sign == ">=":
            v = max(0.0, c.value - val)
        else:
            v = abs(val - c.value)
        P += c.penalty * v

    return P, feas


# Expose short names used by sibling modules
asf_distance = _asf_distance
normalize_w = _normalize_w
compute_penalty_and_feas = _compute_penalty_and_feas


# ---------------------------------------------------------------------------
# WarmStartSampling
# ---------------------------------------------------------------------------

class WarmStartSampling(Sampling):
    """Warm-start sampling for pymoo (robust across versions).

    Some pymoo versions do NOT reliably accept a raw numpy array as
    ``sampling=...``.  This sampling object returns the provided seed
    population *X* (and pads with random samples if more individuals are
    requested).  It also guards against shape mismatch to avoid
    ``Input dimension 0 are not equal to n_var ...`` crashes.
    """

    def __init__(self, X_seed=None):
        super().__init__()
        self.X_seed = None if X_seed is None else np.asarray(X_seed, dtype=float)

    def _do(self, problem, n_samples, **kwargs):
        n_samples = int(n_samples)
        n_var = int(getattr(problem, "n_var", 0) or 0)

        def _rand(m: int):
            # pymoo passes its seeded generator as ``random_state``; the old ``seed`` lookup was always None,
            # which made warm-start padding draw from OS entropy and runs non-reproducible.
            rng = kwargs.get("random_state")
            if rng is None:
                rng = np.random.default_rng(kwargs.get("seed", 0))
            xl = (np.asarray(getattr(problem, "xl", None), dtype=float)
                  if getattr(problem, "xl", None) is not None else None)
            xu = (np.asarray(getattr(problem, "xu", None), dtype=float)
                  if getattr(problem, "xu", None) is not None else None)
            if n_var <= 0:
                return np.zeros((m, 1))
            if xl is not None and xu is not None:
                return rng.uniform(xl, xu, size=(m, n_var))
            return rng.random((m, n_var))

        if self.X_seed is None or len(self.X_seed) == 0:
            return _rand(n_samples)

        X = self.X_seed.copy()
        # Shape guard
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if n_var > 0 and X.shape[1] != n_var:
            return _rand(n_samples)

        if len(X) >= n_samples:
            return X[:n_samples]
        pad = _rand(n_samples - len(X))
        return np.vstack([X, pad])


# ---------------------------------------------------------------------------
# WrappedProblem
# ---------------------------------------------------------------------------

class WrappedProblem(ElementwiseProblem):
    """Unified wrapper: applies the compiled ModelIR to the evaluation.

    - penalty_scale : constraint / feasibility pressure  (OSV.g[2])
    - bias_lambda   : preference shaping strength        (OSV.g[1])
    - scalar_mode   : ASF / TCH / WSUM preference distance (model reconf)
    - obj_band      : dynamic objective band (optional)
    """

    def __init__(self,
                 base: ElementwiseProblem,
                 upv: UPV,
                 enable_cr: bool,
                 model_ir: Optional[ModelIR] = None):
        xl = base.xl
        xu = base.xu
        self.model_ir = model_ir
        if model_ir is not None and model_ir.bounds is not None and len(model_ir.bounds) > 0:
            xl = np.array([b[0] for b in model_ir.bounds], dtype=float)
            xu = np.array([b[1] for b in model_ir.bounds], dtype=float)
        n_constr = int(getattr(base, "n_constr", 0) or 0)
        super().__init__(n_var=base.n_var, n_obj=base.n_obj, xl=xl, xu=xu,
                         **_pymoo_constr_init_kwargs(n_constr, 0))
        self.base = base
        self.upv = upv
        self.enable_cr = enable_cr

    def _pref_distance(self, f: np.ndarray) -> float:
        if self.model_ir is None:
            return _asf_distance(f, self.upv.w, self.upv.r)
        mode = (self.model_ir.scalar_mode or "ASF").upper()
        w = np.asarray(self.upv.w, dtype=float)
        r = np.asarray(self.upv.r, dtype=float)
        d = np.abs(f - r)
        if mode == "WSUM":
            return float(np.sum(w * d))
        if mode == "TCH":
            return float(np.max(w * d))
        # default ASF
        return _asf_distance(f, w, r)

    def _band_penalty(self, f: np.ndarray) -> float:
        if self.model_ir is None or not self.model_ir.obj_band:
            return 0.0
        p = 0.0
        for j, (lo, hi) in enumerate(self.model_ir.obj_band):
            v = float(f[j])
            if v < lo:
                p += (lo - v)
            elif v > hi:
                p += (v - hi)
        return float(self.model_ir.band_penalty * p)

    def _evaluate(self, x, out, *args, **kwargs):
        tmp = {}
        self.base._evaluate(x, tmp)
        f = tmp["F"].astype(float)

        g = tmp.get("G", None)

        # Original constraint penalty + dynamic objective band penalty
        P, _ = _compute_penalty_and_feas(
            np.asarray(x, dtype=float), f, self.upv, self.enable_cr
        )
        Pb = self._band_penalty(f)

        # Preference distance shaping
        bias = self._pref_distance(f)

        penalty_scale = (float(self.model_ir.penalty_scale)
                         if (self.model_ir is not None) else 1.0)
        bias_lambda = (float(self.model_ir.bias_lambda)
                       if (self.model_ir is not None) else 0.05)
        if not self.enable_cr:
            bias_lambda = 0.0

        out["F"] = f + (penalty_scale * (P + Pb) + bias_lambda * bias)
        if g is not None:
            try:
                out["G"] = np.asarray(g, dtype=float)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# build_emo_algorithm
# ---------------------------------------------------------------------------

def build_emo_algorithm(state, upv_opt: Optional[UPV] = None):
    """Construct the pymoo EMO algorithm from *state* attributes.

    Parameters
    ----------
    state : OptState
        Optimisation state object carrying solver parameters
        (``algo_name``, ``pop_size``, ``cx_eta``, ``mut_eta``, etc.).
    upv_opt : UPV, optional
        Override UPV; falls back to ``state.upv``.
    """
    # Guard against invalid SBX parameters
    cx_eta = state.cx_eta if (np.isfinite(state.cx_eta) and state.cx_eta > 0) else 15.0
    mut_eta = state.mut_eta if (np.isfinite(state.mut_eta) and state.mut_eta > 0) else 20.0
    cx_prob = float(np.clip(state.cx_prob, 0.0, 1.0))
    mut_prob = ((1.0 / state.n_var) if (state.mut_prob is None)
                else float(np.clip(state.mut_prob, 0.0, 1.0)))

    repair = ToBoundOutOfBoundsRepair()
    crossover = SBX(prob=cx_prob, eta=cx_eta, repair=repair)
    mutation = PM(prob=mut_prob, eta=mut_eta, repair=repair)
    sampling = (WarmStartSampling(state.pop_X)
                if state.pop_X is not None else FloatRandomSampling())
    upv_use = upv_opt if (upv_opt is not None) else state.upv

    if state.algo_name == "NSGA-III":
        try:
            ref_dirs = get_reference_directions("das-dennis", state.n_obj,
                                                n_points=state.pop_size)
        except Exception:
            # das-dennis n_points may not be achievable for this n_obj;
            # fall back to choosing n_partitions that yields >= pop_size points
            n_part = 1
            while True:
                from math import comb
                n_pts = comb(n_part + state.n_obj - 1, state.n_obj - 1)
                if n_pts >= state.pop_size:
                    break
                n_part += 1
                if n_part > 200:
                    break
            try:
                ref_dirs = get_reference_directions("das-dennis", state.n_obj,
                                                    n_partitions=n_part)
            except Exception:
                ref_dirs = (np.random.default_rng(int(state.seed) + 101)
                            .dirichlet(np.ones(state.n_obj), size=state.pop_size))
        roi_bias = float(state.emo_params.get("roi_bias", 0.0))
        if roi_bias > 0.0:
            roi_bias = float(np.clip(roi_bias, 0.0, 1.0))
            w_ref = _normalize_w(upv_use.w)
            ref_dirs = ((1.0 - roi_bias) * ref_dirs
                        + roi_bias * w_ref.reshape(1, -1))

        # Normalize and ensure |ref_dirs| == pop_size
        if ref_dirs is None or len(ref_dirs) == 0:
            ref_dirs = (np.random.default_rng(int(state.seed) + 101)
                        .dirichlet(np.ones(state.n_obj), size=state.pop_size))
        ref_dirs = np.asarray(ref_dirs, dtype=float)
        ref_dirs = ref_dirs / np.maximum(1e-12, np.sum(ref_dirs, axis=1,
                                                        keepdims=True))
        if len(ref_dirs) > int(state.pop_size):
            ref_dirs = ref_dirs[:int(state.pop_size)]
        elif len(ref_dirs) < int(state.pop_size):
            extra = (np.random.default_rng(int(state.seed) + 777)
                     .dirichlet(np.ones(state.n_obj),
                                size=int(state.pop_size) - len(ref_dirs)))
            ref_dirs = np.vstack([ref_dirs, extra])
        ref_dirs = ref_dirs / np.maximum(1e-12, np.sum(ref_dirs, axis=1,
                                                        keepdims=True))

        algo = NSGA3(pop_size=int(state.pop_size),
                     sampling=sampling, crossover=crossover,
                     mutation=mutation, eliminate_duplicates=True,
                     ref_dirs=ref_dirs)
        return algo

    if state.algo_name == "MOEA/D":
        try:
            ref_dirs = get_reference_directions("das-dennis", state.n_obj,
                                                n_points=state.pop_size)
        except Exception:
            n_part = 1
            while True:
                from math import comb
                n_pts = comb(n_part + state.n_obj - 1, state.n_obj - 1)
                if n_pts >= state.pop_size:
                    break
                n_part += 1
                if n_part > 200:
                    break
            try:
                ref_dirs = get_reference_directions("das-dennis", state.n_obj,
                                                    n_partitions=n_part)
            except Exception:
                ref_dirs = (np.random.default_rng(int(state.seed) + 202)
                            .dirichlet(np.ones(state.n_obj), size=state.pop_size))
        roi_bias = float(state.emo_params.get("roi_bias", 0.0))
        if roi_bias > 0.0:
            roi_bias = float(np.clip(roi_bias, 0.0, 1.0))
            w_ref = _normalize_w(upv_use.w)
            ref_dirs = ((1.0 - roi_bias) * ref_dirs
                        + roi_bias * w_ref.reshape(1, -1))

        if ref_dirs is None or len(ref_dirs) == 0:
            ref_dirs = (np.random.default_rng(int(state.seed) + 202)
                        .dirichlet(np.ones(state.n_obj), size=state.pop_size))
        ref_dirs = np.asarray(ref_dirs, dtype=float)
        ref_dirs = ref_dirs / np.maximum(1e-12, np.sum(ref_dirs, axis=1,
                                                        keepdims=True))
        if len(ref_dirs) > int(state.pop_size):
            ref_dirs = ref_dirs[:int(state.pop_size)]
        elif len(ref_dirs) < int(state.pop_size):
            extra = (np.random.default_rng(int(state.seed) + 888)
                     .dirichlet(np.ones(state.n_obj),
                                size=int(state.pop_size) - len(ref_dirs)))
            ref_dirs = np.vstack([ref_dirs, extra])
        ref_dirs = ref_dirs / np.maximum(1e-12, np.sum(ref_dirs, axis=1,
                                                        keepdims=True))

        algo = MOEAD(ref_dirs=ref_dirs,
                     n_neighbors=min(state.emo_params["moead_neighbors"],
                                     len(ref_dirs) - 1),
                     sampling=sampling, crossover=crossover,
                     mutation=mutation)
        return algo

    return NSGA2(pop_size=state.pop_size, sampling=sampling,
                 crossover=crossover, mutation=mutation,
                 eliminate_duplicates=True)


# ---------------------------------------------------------------------------
# step_emo
# ---------------------------------------------------------------------------

def step_emo(state, upv_opt: Optional[UPV] = None,
             budget: Optional[int] = None):
    """Run one EMO tick on *state*.

    Parameters
    ----------
    state : OptState
        Optimisation state (modified in-place).
    upv_opt : UPV, optional
        Override UPV for this step.
    budget : int, optional
        Evaluation budget for this tick.
    """
    # Clean NaN/Inf in population to prevent SBX pow warnings
    if state.pop_X is not None:
        X = np.asarray(state.pop_X, dtype=float)
        if not np.all(np.isfinite(X)):
            mid = 0.5 * (state.problem.xl + state.problem.xu)
            X = np.nan_to_num(X, nan=mid, posinf=state.problem.xu,
                              neginf=state.problem.xl)
            X = np.clip(X, state.problem.xl, state.problem.xu)
            state.pop_X = X

    term = get_termination("n_gen", state.gen_per_tick)
    upv_use = upv_opt if (upv_opt is not None) else state.upv
    wrapped = WrappedProblem(state.problem, upv_use, state.enable_cr,
                             model_ir=state.model_ir)
    algo = build_emo_algorithm(state, upv_opt=upv_use)
    algo_name_eff = state.algo_name

    if isinstance(algo, MOEAD) and wrapped.has_constraints():
        prev_algo = state.algo_name
        state.algo_name = "NSGA-II"
        algo = build_emo_algorithm(state, upv_opt=upv_use)
        state.algo_name = prev_algo
        algo_name_eff = "NSGA-II"

    # Bind seed to (base_seed, tick, algo_idx) to avoid identical random
    # sequences across ticks.
    try:
        algo_idx = (int(state.algo_pool.index(algo_name_eff))
                    if (hasattr(state, 'algo_pool')
                        and algo_name_eff in state.algo_pool)
                    else 0)
    except Exception:
        algo_idx = 0
    seed_run = int(state.seed
                   + 1000 * int(getattr(state, 'tick_count', 0))
                   + 17 * int(algo_idx))

    try:
        res = minimize(wrapped, algo, termination=term, seed=seed_run,
                       verbose=False, save_history=False)
    except AssertionError as e:
        # Robust fallback for rare pymoo initialisation glitches
        state.add_log("WARN",
                      f"EMO minimize assertion: {str(e)}; "
                      "fallback to random sampling.")
        state.pop_X = None
        algo = build_emo_algorithm(state, upv_opt=upv_use)
        res = minimize(wrapped, algo, termination=term, seed=seed_run,
                       verbose=False, save_history=False)

    new_pop = res.pop
    X_new = new_pop.get("X")
    state.pop_X = X_new

    # Use *base problem* F/G (not the shaped WrappedProblem values) for
    # logging, analysis and cross-method comparison.
    try:
        vals = state.problem.evaluate(X_new, return_values_of=["F", "G"])
        if isinstance(vals, (tuple, list)) and len(vals) >= 2:
            F_base, G_base = vals[0], vals[1]
        elif isinstance(vals, dict):
            F_base, G_base = vals.get("F"), vals.get("G")
        else:
            F_base, G_base = vals, None
    except Exception:
        F_base, G_base = new_pop.get("F"), new_pop.get("G")

    state.pop_F = (None if F_base is None
                   else np.asarray(F_base, dtype=float))
    n_constr = int(getattr(state.problem, "n_constr", 0) or 0)
    if n_constr > 0 and G_base is not None:
        state.pop_G = np.asarray(G_base, dtype=float)
    else:
        state.pop_G = None

    # Estimated evaluations: pop_size * gen_per_tick
    n_eval_rec = (int(budget)
                  if (budget is not None and int(budget) > 0)
                  else int(state.pop_size * state.gen_per_tick))
    state._record_eval(n_eval_rec, 0)
    state._update_history()
