# -*- coding: utf-8 -*-
"""
hitlo.engine.convergence
===========================
Three-layer convergence detection for the HITLO optimization loop.

Layer 1: Hard budget guard (N_eval, wall-clock time)
Layer 2: Stagnation detector (HV, IGD sliding-window)
Layer 3: User satisfaction gate (external signal)

Usage::

    detector = ConvergenceDetector(max_evals=500, hv_window=10, hv_eps=0.001)
    status = detector.check(state)
    if status.should_stop:
        ...
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Status dataclass returned by ConvergenceDetector.check()
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceStatus:
    """Snapshot of convergence assessment at a single tick."""

    # Layer 1: budget
    budget_exhausted: bool = False
    budget_fraction: float = 0.0       # eval_count / max_evals
    evals_used: int = 0
    evals_max: int = 0

    # Layer 2: stagnation
    hv_stagnant: bool = False
    igd_stagnant: bool = False
    convergence_score: float = 0.0     # 0..1, composite
    consec_stagnant_ticks: int = 0
    converged: bool = False            # patience exceeded

    # Layer 2 detail
    hv_improvement_rate: float = 0.0   # avg ΔHV over window
    hv_current: float = 0.0
    hv_predicted_ceiling: float = 0.0  # simple extrapolation
    igd_improvement_rate: float = 0.0

    # Layer 3: user satisfaction
    user_satisfied: bool = False

    # Composite
    should_stop: bool = False          # any hard-stop triggered
    should_suggest: bool = False       # convergence detected, suggest to user
    suggestion_text: str = ""

    # Signal lights: "green" / "yellow" / "red"
    light_budget: str = "green"
    light_convergence: str = "green"
    light_satisfaction: str = "gray"


# ---------------------------------------------------------------------------
# ConvergenceDetector
# ---------------------------------------------------------------------------

class ConvergenceDetector:
    """Three-layer convergence detector.

    Parameters
    ----------
    max_evals : int
        Hard evaluation budget limit. 0 = no limit.
    max_wall_time : float
        Hard wall-clock limit in seconds. 0 = no limit.
    hv_window : int
        Sliding window size (ticks) for HV stagnation check.
    hv_eps : float
        HV improvement threshold below which stagnation is declared.
    igd_window : int
        Sliding window for IGD stagnation (defaults to hv_window).
    igd_eps : float
        IGD improvement threshold.
    patience : int
        Consecutive stagnant ticks before convergence is confirmed.
    auto_stop_on_budget : bool
        If True, hard-stop when budget is exhausted.
    auto_stop_on_convergence : bool
        If True, hard-stop when converged. If False, only suggest.
    """

    def __init__(
        self,
        max_evals: int = 5000,
        max_wall_time: float = 0,
        hv_window: int = 10,
        hv_eps: float = 0.001,
        igd_window: int = 0,
        igd_eps: float = 0.005,
        patience: int = 3,
        auto_stop_on_budget: bool = True,
        auto_stop_on_convergence: bool = False,
    ):
        self.max_evals = max_evals
        self.max_wall_time = max_wall_time
        self.hv_window = max(2, hv_window)
        self.hv_eps = hv_eps
        self.igd_window = max(2, igd_window or hv_window)
        self.igd_eps = igd_eps
        self.patience = max(1, patience)
        self.auto_stop_on_budget = auto_stop_on_budget
        self.auto_stop_on_convergence = auto_stop_on_convergence

        # Internal state
        self._consec_stagnant: int = 0
        self._start_time: Optional[float] = None
        self._last_suggested_tick: int = -100
        self._user_satisfied: bool = False

    def reset(self):
        """Reset detector state for a new run."""
        self._consec_stagnant = 0
        self._start_time = time.monotonic()
        self._last_suggested_tick = -100
        self._user_satisfied = False

    def set_user_satisfied(self, satisfied: bool = True):
        """Mark user satisfaction (Layer 3)."""
        self._user_satisfied = satisfied

    # ----- Main check -----

    def check(self, state) -> ConvergenceStatus:
        """Evaluate convergence against current OptState.

        Must be called with state.lock held (or from step_opt).
        """
        if self._start_time is None:
            self._start_time = time.monotonic()

        status = ConvergenceStatus()
        eval_count = int(getattr(state, "eval_count", 0))
        tick_count = int(getattr(state, "tick_count", 0))
        hv_hist = list(getattr(state, "history_hv", []))
        igd_hist = list(getattr(state, "history_igd", []))
        feas_hist = list(getattr(state, "history_feasible", []))
        n_feasible = int(len([f for f in feas_hist[-1:] if f and float(f) > 0])) if feas_hist else 0

        # ---- Layer 1: Hard budget ----
        status.evals_used = eval_count
        status.evals_max = self.max_evals

        if self.max_evals > 0:
            status.budget_fraction = min(1.0, eval_count / self.max_evals)
            if eval_count >= self.max_evals:
                status.budget_exhausted = True
        else:
            status.budget_fraction = 0.0

        # Wall-clock check
        if self.max_wall_time > 0 and self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            if elapsed >= self.max_wall_time:
                status.budget_exhausted = True

        # Budget traffic light
        if status.budget_exhausted:
            status.light_budget = "red"
        elif status.budget_fraction > 0.75:
            status.light_budget = "yellow"
        else:
            status.light_budget = "green"

        # ---- Layer 2: Stagnation ----
        status.hv_current = float(hv_hist[-1]) if hv_hist else 0.0

        # HV stagnation
        if len(hv_hist) >= self.hv_window:
            window = hv_hist[-self.hv_window:]
            hv_range = max(window) - min(window)
            hv_mean_delta = (window[-1] - window[0]) / max(1, len(window) - 1)
            status.hv_improvement_rate = hv_mean_delta
            status.hv_stagnant = hv_range < self.hv_eps

        # IGD stagnation
        if len(igd_hist) >= self.igd_window:
            window = igd_hist[-self.igd_window:]
            igd_range = max(window) - min(window)
            igd_mean_delta = (window[-1] - window[0]) / max(1, len(window) - 1)
            status.igd_improvement_rate = igd_mean_delta
            status.igd_stagnant = igd_range < self.igd_eps

        # HV ceiling prediction (simple exponential fit)
        status.hv_predicted_ceiling = self._predict_hv_ceiling(hv_hist)

        # Composite convergence score: 0 (not converged) to 1 (fully converged)
        score = 0.0
        if status.hv_stagnant:
            score += 0.5
        if status.igd_stagnant:
            score += 0.3
        if status.budget_fraction > 0.8:
            score += 0.2
        status.convergence_score = min(1.0, score)

        # Track consecutive stagnant ticks
        if status.hv_stagnant:
            self._consec_stagnant += 1
        else:
            self._consec_stagnant = 0
        status.consec_stagnant_ticks = self._consec_stagnant

        # Convergence confirmed when patience exceeded
        status.converged = self._consec_stagnant >= self.patience

        # Convergence traffic light
        if status.converged:
            status.light_convergence = "red"
        elif status.hv_stagnant or status.convergence_score > 0.4:
            status.light_convergence = "yellow"
        else:
            status.light_convergence = "green"

        # ---- Layer 3: User satisfaction ----
        status.user_satisfied = self._user_satisfied
        if self._user_satisfied:
            status.light_satisfaction = "green"
        else:
            status.light_satisfaction = "gray"

        # ---- Composite decisions ----
        # Hard stop
        if status.budget_exhausted and self.auto_stop_on_budget:
            status.should_stop = True
            status.suggestion_text = (
                f"评估预算已耗尽（{eval_count}/{self.max_evals}），优化自动停止。"
            )
        elif status.converged and self.auto_stop_on_convergence:
            status.should_stop = True
            status.suggestion_text = (
                f"优化已收敛（HV停滞{self._consec_stagnant}轮），自动停止。"
            )
        elif self._user_satisfied:
            status.should_stop = True
            status.suggestion_text = "用户确认满意，优化终止。"

        # Suggest stop (non-hard)
        if status.converged and not status.should_stop:
            # Avoid spamming: only suggest every 5 ticks
            if tick_count - self._last_suggested_tick >= 5:
                status.should_suggest = True
                self._last_suggested_tick = tick_count

                # Build suggestion
                pf_count = 0
                pop_F = getattr(state, "pop_F", None)
                if pop_F is not None and len(pop_F) > 0:
                    from hitlo.engine.metrics import is_nondominated
                    nd_mask = is_nondominated(np.asarray(pop_F, dtype=float))
                    pf_count = int(np.sum(nd_mask))

                feas_pct = float(feas_hist[-1]) * 100 if feas_hist else 0
                hv_pct = abs(status.hv_improvement_rate) * 100

                status.suggestion_text = (
                    f"收敛评估报告\n"
                    f"- 最近{self.hv_window}轮HV改善仅 {hv_pct:.2f}%，趋势持平\n"
                    f"- 当前帕累托前沿含 {pf_count} 个非支配解"
                    f"（可行率 {feas_pct:.1f}%）\n"
                    f"- 已使用 {eval_count} 次评估"
                    + (f"（预算 {self.max_evals}）" if self.max_evals > 0 else "")
                    + "\n\n"
                    f"建议考虑终止优化。"
                )

        return status

    # ----- HV ceiling predictor -----

    @staticmethod
    def _predict_hv_ceiling(hv_hist: List[float], min_points: int = 5) -> float:
        """Predict HV ceiling via simple exponential extrapolation.

        Fits HV(t) ≈ HV_∞ · (1 - c·e^{-k·t}) using least-squares on the
        last portion of the history.  Falls back to the current HV if too
        few points or fitting fails.
        """
        if len(hv_hist) < min_points:
            return float(hv_hist[-1]) if hv_hist else 0.0

        hv = np.asarray(hv_hist[-50:], dtype=float)
        n = len(hv)
        t = np.arange(n, dtype=float)

        try:
            # Simple approach: fit a saturating exponential
            # HV(t) = a - b * exp(-k*t)
            # Use last value as initial a estimate
            a_est = hv[-1] * 1.05
            if a_est <= 0:
                return float(hv[-1])

            # Use log-linear regression on (a_est - HV) to estimate k
            diff = a_est - hv
            diff = np.clip(diff, 1e-12, None)
            log_diff = np.log(diff)

            # Linear fit: log(a-HV) = log(b) - k*t
            if n >= 3:
                A_mat = np.vstack([np.ones(n), t]).T
                coeffs = np.linalg.lstsq(A_mat, log_diff, rcond=None)[0]
                k = -coeffs[1]
                if k > 0:
                    # Ceiling = a_est (our estimate)
                    return float(a_est)

            return float(hv[-1])
        except Exception:
            return float(hv[-1]) if hv_hist else 0.0
