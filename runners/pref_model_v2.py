"""Preference model v2 (JCDE revision; fixed on 2026-10-22, before any v2 run).

v1 (submitted code): the simulated decision maker (SDM) and the host aggregate preferences with
ASF(f) = max_i w_i |f_i - r_i| on raw objectives. On SAV-AERO, -Cn spans about 6.4 units and Cd0 about 0.5,
and the SDM aspirations (Cn 8-10) lie beyond the attainable front (Cn <= 6.5). Every SDM choice was decided
by Cn alone (10,500 sampled pairs: 100 % identical to a Cn-only rule) and w* had no effect.

v2:
* every ASF, in the host and in the SDM, divides the weighted deviations by fixed objective scales
  (hitlo.core.objective_scale); the host keeps its distance form, so thresholds keep their meaning;
* the SDM uses the monotone aspiration form ASF(f) = max_i w_i (f_i - r_i)/s_i + rho * sum_i w_i (f_i - r_i)/s_i,
  so a design better than the aspiration is not penalised;
* SAV-AERO aspirations are moved onto the attainable front by a fixed rule:
  scale = nadir - ideal of the front sampled by the 450 finished P1 reruns (final feasible populations);
  r* = the front point preferred by w* from the ideal, argmin max_i w_i (f_i - ideal_i)/s_i;
  'unattractive' bounds C_h* = r* + 0.15 s (Cd0 above, Cn below) with tolerance 0.05 s.
Benchmarks keep their w* and r*; their scales are the ranges of the analytic Pareto fronts.
"""

from __future__ import annotations

import numpy as np

from hitlo.core.objective_scale import get_scale, set_scale
from hitlo.core.simulated_dm import SDM_PROFILES, SDMProfile, SimulatedDM

RHO = 1e-6
SCALES = {
    "SAV-AERO": (
        0.5349,
        6.3675,
    ),  # Cd0, -Cn: front ideal (0.0295, -6.5464), nadir (0.5644, -0.1789)
    "ZDT1": (1.0, 1.0),
    "ZDT2": (1.0, 1.0),
    "ZDT3": (0.8518, 1.7733),
    "BNH": (136.0, 46.0),
    "TNK": (1.0, 1.0),
}
# key: (w*, r* as minimisation objectives (Cd0, -Cn), [(Cd0 bound, tol), (-Cn bound, tol)])
AERO = {
    "SAV-AERO_balanced": ((0.5, 0.5), (0.278, -3.80), ((0.359, 0.027), (-2.85, 0.32))),
    "SAV-AERO_mid": ((0.65, 0.35), (0.204, -2.96), ((0.285, 0.027), (-2.01, 0.32))),
    "SAV-AERO_edge": ((0.8, 0.2), (0.131, -1.40), ((0.211, 0.027), (-0.45, 0.32))),
}


def _asf_v2(self, f) -> float:
    w = self.w_star
    d = w * (np.asarray(f, dtype=float)[: len(w)] - self.r_star[: len(w)])
    s = get_scale()
    if s is not None:
        d = d / s[: len(d)]
    return float(np.max(d) + RHO * np.sum(d))


def activate(problem_id: str) -> dict:
    """Switch this process to preference model v2 (call before the SDM and the host are built)."""
    scale = SCALES.get(problem_id.upper())
    if scale is None:
        raise KeyError(f"preference model v2 has no objective scale for {problem_id}")
    set_scale(scale)
    for key, (w, r, bounds) in AERO.items():
        SDM_PROFILES[key] = SDMProfile(
            name=key,
            problem_id="SAV-AERO",
            n_obj=2,
            w_star=np.array(w),
            r_star=np.array(r),
            C_h_star=[
                {"slot": f"r_{i}", "value": v, "tolerance": t}
                for i, (v, t) in enumerate(bounds)
            ],
            description=f"{key} (preference model v2: aspiration on the attainable front)",
        )
    SimulatedDM._asf = _asf_v2
    SimulatedDM.PREF_MODEL = {
        "version": "v2",
        "sdm_utility": "monotone ASF on scaled objectives",
        "host_utility": "ASF distance on scaled objectives",
        "scale": list(scale),
        "rho": RHO,
    }
    return SimulatedDM.PREF_MODEL
