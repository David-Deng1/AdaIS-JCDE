# -*- coding: utf-8 -*-
"""Base problem utilities: pymoo compatibility helpers, PymooProblemAdapter, WarmStartSampling."""

import inspect
from typing import Dict, Any, Optional

import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.problems import get_problem
from pymoo.core.sampling import Sampling


# ---------------------------
# pymoo compatibility helpers
# ---------------------------

def _pymoo_constr_init_kwargs(n_ieq: int = 0, n_eq: int = 0) -> Dict[str, int]:
    """Return the right constraint kwargs for the installed pymoo version.

    Newer pymoo versions (>=0.6) use `n_ieq_constr` and `n_eq_constr`, while older
    versions use `n_constr`.
    """
    try:
        sig = inspect.signature(ElementwiseProblem.__init__)
        if "n_ieq_constr" in sig.parameters:
            return {"n_ieq_constr": int(n_ieq), "n_eq_constr": int(n_eq)}
    except Exception:
        pass
    return {"n_constr": int(n_ieq) + int(n_eq)}


class PymooProblemAdapter(ElementwiseProblem):
    """Adapt a vectorized pymoo problem to an ElementwiseProblem.

    Our experimental pipeline evaluates candidates one-by-one via `_evaluate(x, out)`.
    Many pymoo benchmark problems are vectorized (`Problem`) and expect `X` to be a
    2D array. This adapter wraps any `get_problem(...)` object so it behaves like an
    `ElementwiseProblem` while preserving bounds/metadata.
    """

    def __init__(self, name: str, n_var_hint: Optional[int] = None, **kwargs):
        self.name = str(name)

        # Try passing n_var if supported; some problems (e.g., DASCMOP) hard-code n_var
        # and will raise if `n_var` is provided.
        inner = None
        if n_var_hint is not None:
            try:
                inner = get_problem(self.name, n_var=int(n_var_hint), **kwargs)
            except TypeError:
                inner = None
        if inner is None:
            inner = get_problem(self.name, **kwargs)

        self.inner = inner

        n_var = int(getattr(inner, "n_var", 0) or 0)
        n_obj = int(getattr(inner, "n_obj", 0) or 0)
        n_ieq = int(getattr(inner, "n_ieq_constr", getattr(inner, "n_constr", 0)) or 0)
        n_eq = int(getattr(inner, "n_eq_constr", 0) or 0)

        # normalize bounds to vectors
        xl = np.asarray(getattr(inner, "xl", None), dtype=float)
        xu = np.asarray(getattr(inner, "xu", None), dtype=float)
        if xl.ndim == 0:
            xl = np.full(n_var, float(xl), dtype=float)
        if xu.ndim == 0:
            xu = np.full(n_var, float(xu), dtype=float)
        if xl.size != n_var:
            xl = np.resize(xl, n_var)
        if xu.size != n_var:
            xu = np.resize(xu, n_var)

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
            **_pymoo_constr_init_kwargs(n_ieq=n_ieq, n_eq=n_eq),
        )

        # Provide both old and new-style constraint attributes for downstream code.
        #
        # IMPORTANT: in newer pymoo versions, `n_constr` is a read-only @property
        # (computed from n_ieq_constr + n_eq_constr). Assigning to it will raise:
        #   AttributeError: property 'n_constr' ... has no setter
        #
        # Therefore we only *set* n_constr for older pymoo releases where it is
        # a plain attribute.
        for k, v in (("n_ieq_constr", n_ieq), ("n_eq_constr", n_eq)):
            try:
                setattr(self, k, int(v))
            except Exception:
                # If pymoo exposes these as read-only properties, ignore.
                pass

        try:
            cls_attr = getattr(type(self), "n_constr", None)
            if not isinstance(cls_attr, property):
                self.n_constr = int(n_ieq) + int(n_eq)
        except Exception:
            pass

        self.bounds = np.column_stack([xl, xu])

    def _evaluate(self, x, out, *args, **kwargs):
        X = np.atleast_2d(np.asarray(x, dtype=float))
        inner_out: Dict[str, Any] = {}
        # Call underlying vectorized evaluation.
        self.inner._evaluate(X, inner_out, *args, **kwargs)

        F = inner_out.get("F", None)
        if F is None:
            raise RuntimeError(f"Inner problem {self.name} did not write out['F']")
        F = np.asarray(F, dtype=float)
        if F.ndim == 1:
            out["F"] = F
        else:
            out["F"] = F[0]

        if self.n_constr > 0:
            G = inner_out.get("G", None)
            H = inner_out.get("H", None)

            G_all = []
            if G is not None:
                G = np.asarray(G, dtype=float)
                if G.ndim == 1:
                    G = G.reshape((-1, 1))
                G_all.append(G)

            if H is not None:
                H = np.asarray(H, dtype=float)
                if H.ndim == 1:
                    H = H.reshape((-1, 1))
                # Convert equality constraints to inequality form: |h(x)| - eps <= 0
                eps = 1e-4
                G_all.append(np.abs(H) - eps)

            if len(G_all) == 0:
                out["G"] = np.zeros((self.n_constr,), dtype=float)
            else:
                GG = np.column_stack(G_all)
                out["G"] = GG[0]

    def __getattr__(self, item):
        # Forward convenience methods like pareto_front(), pareto_set(), etc.
        if item in ("inner", "name"):
            raise AttributeError
        return getattr(self.inner, item)


class WarmStartSampling(Sampling):
    """Warm-start sampling for pymoo (robust across versions).

    Some pymoo versions do NOT reliably accept a raw numpy array as `sampling=...`.
    This sampling object returns the provided seed population X (and pads with random
    samples if more individuals are requested). It also guards against shape mismatch
    to avoid `Input dimension 0 are not equal to n_var ...` crashes.
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
            xl = np.asarray(getattr(problem, "xl", None), dtype=float) if getattr(problem, "xl", None) is not None else None
            xu = np.asarray(getattr(problem, "xu", None), dtype=float) if getattr(problem, "xu", None) is not None else None
            if n_var <= 0:
                # Fallback: try infer from bounds length
                if xl is not None and xl.ndim == 1 and xl.size > 0:
                    n_v = int(xl.size)
                elif xu is not None and xu.ndim == 1 and xu.size > 0:
                    n_v = int(xu.size)
                else:
                    n_v = 1
            else:
                n_v = n_var

            # Ensure bounds vectors match n_v
            if xl is None or xl.size == 0:
                xl = -1.0 * np.ones(n_v, dtype=float)
            if xu is None or xu.size == 0:
                xu = 1.0 * np.ones(n_v, dtype=float)
            if xl.ndim == 0:
                xl = np.full(n_v, float(xl))
            if xu.ndim == 0:
                xu = np.full(n_v, float(xu))
            if xl.size != n_v:
                xl = np.resize(xl, n_v)
            if xu.size != n_v:
                xu = np.resize(xu, n_v)
            return rng.uniform(low=xl, high=xu, size=(int(m), int(n_v)))

        X = self.X_seed
        if X is None:
            return _rand(n_samples)

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            if n_var > 0 and X.size == n_var:
                X = X.reshape(1, n_var)
            else:
                return _rand(n_samples)

        if X.ndim != 2 or (n_var > 0 and X.shape[1] != n_var) or X.shape[1] <= 0:
            return _rand(n_samples)

        if X.shape[0] >= n_samples:
            return X[:n_samples].copy()

        pad = _rand(n_samples - X.shape[0])
        return np.vstack([X, pad])
