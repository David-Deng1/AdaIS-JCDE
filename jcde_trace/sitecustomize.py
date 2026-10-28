# -*- coding: utf-8 -*-
"""Observation-only run tracing for the JCDE-2026-572 reruns.

Inert unless ``JCDE_TRACE=1``. When active and this directory is first on
``PYTHONPATH``, it patches two engine entry points right after their modules load:

* ``OptState.step_opt``: records the preference estimate (w, r) before each tick and the
  population objectives/constraints and evaluation count after it.
* ``SimulatedDM.respond_*``: records each simulated response together with the noise-free
  answer implied by the ground-truth preference (w*, r*, C_h*), so response correctness can
  be scored independently of the scheduler's own quality signals.

Nothing here draws random numbers or mutates engine state, so traced and untraced runs are
bit-identical. At interpreter exit the trace is written next to the run's ``metrics.csv``
(``jcde_trace.npz`` and ``jcde_trace.json``), or to ``$JCDE_TRACE_DIR`` when set.
"""

import importlib.abc
import importlib.machinery
import os
import sys

# module name -> patches applied right after that module executes (shared by tracing and overrides)
_TARGETS = {}


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        patches = _TARGETS.get(name)
        if not patches:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        exec_original = spec.loader.exec_module

        def exec_module(module):
            exec_original(module)
            for patch in patches:
                patch(module)

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("JCDE_SEED_GLOBALS") == "1":
    # Third-party code (e.g. some pymoo operators) still draws from the global numpy / Python
    # generators, which are otherwise seeded from OS entropy. Seed them from the run's --seed.
    def _seed_globals():
        import random

        import numpy as np

        seed = 0
        if "--seed" in sys.argv and sys.argv.index("--seed") + 1 < len(sys.argv):
            seed = int(sys.argv[sys.argv.index("--seed") + 1])
        np.random.seed(seed)
        random.seed(seed)

    _seed_globals()

if os.environ.get("JCDE_TRACE") == "1":
    import atexit
    import importlib.abc
    import importlib.machinery
    import json
    import time
    from pathlib import Path

    _T0 = time.time()
    _TRACE = {
        "pop_F": [],
        "pop_G": [],
        "eval": [],
        "upv_w": [],
        "upv_r": [],
        "events": [],
        "sdm": None,
    }

    def _vec(value):
        import numpy as np

        try:
            return np.asarray(value, dtype=float).ravel().tolist()
        except Exception:
            return None

    def _patch_opt_state(module):
        import numpy as np

        original = module.OptState.step_opt

        def step_opt(self, *args, **kwargs):
            upv = getattr(self, "upv", None)
            _TRACE["upv_w"].append(_vec(getattr(upv, "w", None)))
            _TRACE["upv_r"].append(_vec(getattr(upv, "r", None)))
            try:
                return original(self, *args, **kwargs)
            finally:
                pop_F = getattr(self, "pop_F", None)
                pop_G = getattr(self, "pop_G", None)
                _TRACE["pop_F"].append(
                    None if pop_F is None else np.array(pop_F, dtype=float, copy=True)
                )
                _TRACE["pop_G"].append(
                    None if pop_G is None else np.array(pop_G, dtype=float, copy=True)
                )
                _TRACE["eval"].append(int(getattr(self, "eval_count", 0) or 0))
                if os.environ.get("JCDE_TRACE_DEBUG") == "1":
                    osv = getattr(self, "osv", None)
                    _TRACE.setdefault("debug", []).append(
                        {
                            "algo": str(getattr(self, "algo_name", "")),
                            "g_exec": _vec(getattr(osv, "g_exec", None)),
                            "pop_X_sum": float(
                                np.nansum(
                                    np.asarray(getattr(self, "pop_X", []), dtype=float)
                                )
                            ),
                            "pref_w": _vec(
                                getattr(getattr(self, "upv", None), "w", None)
                            ),
                        }
                    )

        module.OptState.step_opt = step_opt

    def _patch_sdm(module):
        import numpy as np

        cls = module.SimulatedDM
        original_init = cls.__init__
        original = {
            name: getattr(cls, name)
            for name in (
                "respond_pairwise",
                "respond_ranking",
                "respond_threshold",
                "respond_roi",
                "respond_nl_clarify",
            )
        }

        def _event(self, channel, **fields):
            fields.update(
                channel=channel,
                tick=len(_TRACE["pop_F"]),
                sigma_effective=float(self.sigma_effective),
            )
            _TRACE["events"].append(fields)

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            _TRACE["sdm"] = {
                "w_star": _vec(self.w_star),
                "r_star": _vec(self.r_star),
                "C_h_star": json.loads(json.dumps(self.C_h_star, default=str)),
                "sigma_base": float(self.sigma_base),
                "fatigue_rate": float(self.fatigue_rate),
            }
            if getattr(self, "PREF_MODEL", None):  # preference model v2: utility form and scales
                _TRACE["sdm"]["pref_model"] = json.loads(json.dumps(self.PREF_MODEL, default=str))

        def respond_pairwise(self, x_i, x_j, f_i, f_j):
            u_i, u_j = float(self._asf(f_i)), float(self._asf(f_j))
            truth = 0 if u_i <= u_j else 1
            out = original["respond_pairwise"](self, x_i, x_j, f_i, f_j)
            _event(
                self,
                "pairwise",
                returned=int(out),
                truth=truth,
                correct=int(out) == truth,
                asf=[u_i, u_j],
            )
            return out

        def respond_ranking(self, candidates_F, k=5):
            arr = np.asarray(candidates_F, dtype=float)
            asf = [float(self._asf(arr[i])) for i in range(len(arr))]
            truth = [int(i) for i in np.argsort(asf)][: min(k, len(arr))]
            out = original["respond_ranking"](self, candidates_F, k=k)
            returned = [int(i) for i in out]
            _event(
                self,
                "ranking",
                returned=returned,
                truth=truth,
                correct=returned == truth,
                n_misplaced=int(sum(a != b for a, b in zip(returned, truth))),
                asf=asf,
            )
            return out

        def respond_threshold(self, slot_name, proposed_value):
            true_value = self._get_ground_truth_slot(slot_name)
            out = original["respond_threshold"](self, slot_name, proposed_value)
            accepted, value = bool(out[0]), float(out[1])
            rel_error = (
                None
                if true_value is None
                else abs(value - true_value) / (abs(true_value) + 1e-6)
            )
            _event(
                self,
                "threshold",
                slot=slot_name,
                proposed=float(proposed_value),
                accepted=accepted,
                returned=value,
                truth=true_value,
                rel_error=rel_error,
            )
            return out

        def respond_roi(self, pop_F, projection_dims=(0, 1)):
            arr = np.asarray(pop_F, dtype=float)
            out = original["respond_roi"](self, pop_F, projection_dims)
            best = (
                arr[int(np.argmin([self._asf(f) for f in arr]))] if len(arr) else None
            )
            inside = None
            if best is not None:
                lo0, hi0, lo1, hi1 = (float(v) for v in out)
                d0, d1 = projection_dims
                inside = bool(lo0 <= best[d0] <= hi0 and lo1 <= best[d1] <= hi1)
            _event(
                self,
                "roi",
                returned=[float(v) for v in out],
                dims=list(projection_dims),
                asf_best_point=_vec(best),
                contains_asf_best=inside,
            )
            return out

        def respond_nl_clarify(self, slot_name):
            out = original["respond_nl_clarify"](self, slot_name)
            _event(
                self,
                "nl_clarify",
                slot=slot_name,
                returned=json.loads(json.dumps(out, default=str)),
                oracle=True,
            )
            return out

        cls.__init__ = __init__
        for fn in (
            respond_pairwise,
            respond_ranking,
            respond_threshold,
            respond_roi,
            respond_nl_clarify,
        ):
            setattr(cls, fn.__name__, fn)

    _TARGETS.setdefault("hitlo.engine.opt_state", []).append(_patch_opt_state)
    _TARGETS.setdefault("hitlo.core.simulated_dm", []).append(_patch_sdm)

    def _run_dir():
        explicit = os.environ.get("JCDE_TRACE_DIR")
        if explicit:
            return Path(explicit)
        argv = sys.argv
        for flag in ("--output_dir", "--out_dir", "--output-dir"):
            if flag in argv and argv.index(flag) + 1 < len(argv):
                base = Path(argv[argv.index(flag) + 1])
                fresh = [
                    p.parent
                    for p in base.glob("*/metrics.csv")
                    if p.stat().st_mtime >= _T0 - 1
                ]
                fresh += (
                    [base]
                    if (base / "metrics.csv").is_file()
                    and (base / "metrics.csv").stat().st_mtime >= _T0 - 1
                    else []
                )
                if fresh:
                    return max(fresh, key=lambda p: (p / "metrics.csv").stat().st_mtime)
        return None

    def _write():
        import numpy as np

        run_dir = _run_dir()
        if run_dir is None or not _TRACE["pop_F"]:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        pops = [p for p in _TRACE["pop_F"] if p is not None and p.ndim == 2]
        n_max = max((p.shape[0] for p in pops), default=0)
        n_obj = pops[0].shape[1] if pops else 0
        pop_F = np.full((len(_TRACE["pop_F"]), n_max, n_obj), np.nan)
        sizes = np.zeros(len(_TRACE["pop_F"]), dtype=int)
        for t, p in enumerate(_TRACE["pop_F"]):
            if p is not None and p.ndim == 2:
                pop_F[t, : p.shape[0]] = p
                sizes[t] = p.shape[0]
        gs = [g for g in _TRACE["pop_G"] if g is not None and g.ndim == 2]
        n_con = gs[0].shape[1] if gs else 0
        pop_G = np.full((len(_TRACE["pop_G"]), n_max, n_con), np.nan)
        for t, g in enumerate(_TRACE["pop_G"]):
            if g is not None and g.ndim == 2:
                pop_G[t, : g.shape[0]] = g
        np.savez_compressed(
            run_dir / "jcde_trace.npz",
            pop_F=pop_F,
            pop_G=pop_G,
            pop_size=sizes,
            eval_count=np.asarray(_TRACE["eval"], dtype=int),
        )
        meta = {
            "sdm": _TRACE["sdm"],
            "upv_w": _TRACE["upv_w"],
            "upv_r": _TRACE["upv_r"],
            "events": _TRACE["events"],
            "argv": sys.argv,
            "debug": _TRACE.get("debug"),
            "overrides": os.environ.get("JCDE_OVERRIDES"),
            "host": os.environ.get("JCDE_HOST"),
        }
        (run_dir / "jcde_trace.json").write_text(
            json.dumps(meta, default=str), encoding="utf-8"
        )

    atexit.register(_write)


if os.environ.get("JCDE_OVERRIDES") or os.environ.get("JCDE_HOST"):
    # Sensitivity / host overrides for the JCDE-R1 supplementary experiments (E5, E6). Values are
    # applied right after the owning module loads, before any runner binds them.
    import json as _json

    _OV = _json.loads(os.environ.get("JCDE_OVERRIDES") or "{}")
    _HOST = os.environ.get("JCDE_HOST", "")

    def _wrap_init(cls, values):
        original = cls.__init__

        def __init__(self, *args, **kwargs):
            original(self, *args, **kwargs)
            for key, val in values.items():
                setattr(self, key, val)

        cls.__init__ = __init__

    def _ov_scheduler(module):
        module.CHANNEL_COST.update({k: float(v) for k, v in _OV.get("cost", {}).items()})
        if _OV.get("sched"):
            _wrap_init(module.SchedulerConfig, _OV["sched"])

    def _ov_sdm(module):
        cls = module.SimulatedDM
        cls.CHANNEL_BASE_LATENCY.update({k: float(v) for k, v in _OV.get("sdm_latency", {}).items()})
        for key, val in _OV.get("sdm_class", {}).items():
            setattr(cls, key, val)
        if _OV.get("sdm_inst"):
            _wrap_init(cls, _OV["sdm_inst"])
        form = _OV.get("fatigue_form", "linear")
        if form != "linear":
            import numpy as np

            def _fatigue_progress(self):
                x = self.fatigue_rate * float(self._interaction_count) / self._K_total
                if form == "saturating":
                    return float(np.clip(1.0 - np.exp(-3.0 * x), 0.0, 1.0))
                if form == "step":
                    return 1.0 if x >= 0.5 else 0.0
                raise ValueError(f"unknown fatigue_form {form!r}")

            cls._fatigue_progress = _fatigue_progress

    def _ov_experiment(module):
        if _HOST == "nsga2":
            _wrap_init(module.ExperimentConfig, {"exp_group": "A1_FIXED_SINGLE", "fixed_single_solver": "NSGA-II"})
        elif _HOST == "single_nsga2":
            # default host with its preference coupling kept (objective shaping with the UPV ASF, ROI-biased
            # reference directions) but one generator only: the paper's Abl-3 "no solver mixture" switch
            original = module.ExperimentConfig.__init__

            def __init__(self, *args, **kwargs):
                original(self, *args, **kwargs)
                self.fixed_single_solver = "NSGA-II"
                if "ABL_NO_SCHEDULING" not in self.ablations:
                    self.ablations.append("ABL_NO_SCHEDULING")

            module.ExperimentConfig.__init__ = __init__
        elif _HOST:
            raise ValueError(f"unknown JCDE_HOST {_HOST!r}")

    _TARGETS.setdefault("hitlo.core.interaction_scheduler", []).append(_ov_scheduler)
    _TARGETS.setdefault("hitlo.core.simulated_dm", []).append(_ov_sdm)
    _TARGETS.setdefault("hitlo.core.experiment", []).append(_ov_experiment)

if _TARGETS:
    sys.meta_path.insert(0, _PatchFinder())
