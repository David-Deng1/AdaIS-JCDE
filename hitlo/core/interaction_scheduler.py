# -*- coding: utf-8 -*-
"""
hitlo.core.interaction_scheduler
==================================
Adaptive interaction scheduling for §3.4: decides *when* and *which channel*
to query the decision maker, enforcing cooldown, budget, and fatigue controls.

Three trigger signals:
  - I_h (high uncertainty): low confidence on key UPV slots
  - I_d (high conflict): large conflict score
  - I_s (stagnation): consecutive stagnant ticks

Channel selection follows phase-adapted strategy (Table 3-5) with
information-gain (IG) proxy weighting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import os

import numpy as np

__all__ = [
    "SchedulerConfig",
    "InteractionScheduler",
    "CHANNEL_COST",
]


# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #

@dataclass
class SchedulerConfig:
    """Configuration for the adaptive interaction scheduler.

    Parameters
    ----------
    tau_h : float
        Uncertainty trigger threshold: fire I_h when max(1-gamma_s) > tau_h.
    tau_d : float
        Conflict trigger threshold: fire I_d when ||d_t||_inf > tau_d.
    T_stag : int
        Stagnation trigger: fire I_s after T_stag consecutive stagnant ticks.
    lambda_cost : float
        Per-interaction cost weight for the reward function.
    B_total : int
        Total interaction budget for the run.
    B_reserve : int
        Reserved budget for late-stage clarifications.
    T_min : float
        Minimum cooldown period (ticks).
    T_max : float
        Maximum cooldown period (ticks).
    kappa : float
        Exponential decay rate for cooldown as a function of last IG.
    tau_Q_high : float
        Upper threshold for quality indicator Q_t. Q_t > tau_Q_high → "normal".
    tau_Q_low : float
        Lower threshold for Q_t. Q_t < tau_Q_low → "pause" (fatigue detected).
    alpha1 : float
        Weight of response latency ratio (t_k/t̄) in Q_t (Eq. 3-35).
    alpha2 : float
        Weight of revision count (n_rev) in Q_t (Eq. 3-35).
    alpha3 : float
        Weight of preference flip count (n_flip) in Q_t (Eq. 3-35).
    W_Q : int
        Window size for counting preference flips in Q_t computation.
    feas_early_threshold : float
        Feasibility ratio below which phase = "early".
    delta_hv_late_threshold : float
        HV improvement rate below which phase = "late" (if feas >= 0.8).
    warmup_ticks : int
        Minimum number of optimisation ticks before adaptive scheduling can
        trigger. This avoids spending preference queries before the current
        population contains an informative feasible/frontier set.
    """
    tau_h: float = 0.6
    tau_d: float = 0.4
    T_stag: int = 5
    lambda_cost: float = 0.5       # Eq.(3-30) cost weight, balances IG vs cognitive cost
    B_total: int = 30
    B_reserve: int = 5
    T_min: float = 0.0
    T_max: float = 6.0
    kappa: float = 3.0
    tau_Q_high: float = 0.6     # Q_t > 0.6 → normal (high quality)
    tau_Q_low: float = 0.3      # Q_t < 0.3 → pause (low quality / fatigued)
    alpha1: float = 0.8         # latency ratio weight（iter_1 验证 0.3 信号幅度过小：fat0→0.7 Q̄_t 仅差 0.029；放大至 0.8 后 Q_t 更易跌入 warning/pause 区）
    alpha2: float = 0.5         # revision count weight
    alpha3: float = 0.8         # preference flip weight
    W_Q: int = 5                # flip window size
    feas_early_threshold: float = 0.3
    delta_hv_late_threshold: float = 0.001
    warmup_ticks: int = 0


# ------------------------------------------------------------------ #
#  Channel priority table (Table 3-5)
# ------------------------------------------------------------------ #

# Phase -> ranked list of preferred channels (highest priority first)
# Per Table 3-5: early=feasibility-dominated, mid=trade-off exploration, late=refinement
_PHASE_CHANNEL_PRIORITY: Dict[str, List[str]] = {
    "early": ["threshold", "nl_clarify", "roi", "ranking", "pairwise"],
    "mid":   ["roi", "ranking", "pairwise", "threshold", "nl_clarify"],
    "late":  ["pairwise", "nl_clarify", "ranking", "roi", "threshold"],
}

# §3.4.2 Channel interaction cost C(a), normalised to [0,1] (60s upper bound)
CHANNEL_COST: Dict[str, float] = {
    "pairwise":   0.10,    # ~10s, low cognitive load
    "threshold":  0.15,    # ~15s, low cognitive load
    "roi":        0.20,    # ~20s, moderate cognitive load
    "ranking":    0.30,    # ~30s, moderate cognitive load
    "nl_clarify": 0.60,    # ~60s, high cognitive load
}


# ------------------------------------------------------------------ #
#  InteractionScheduler
# ------------------------------------------------------------------ #

class InteractionScheduler:
    """Adaptive interaction frequency and timing controller.

    Implements the three-signal trigger model (I_h, I_d, I_s),
    phase-aware channel selection, cooldown adaptation, and
    fatigue-aware quality monitoring (Q_t).

    Parameters
    ----------
    config : SchedulerConfig
        All tuneable parameters.
    """

    def __init__(self, config: SchedulerConfig):
        self.cfg = config

        # internal state
        self._cooldown_remaining: int = 0
        self._budget_used: int = 0
        self._interaction_log: List[Dict[str, Any]] = []
        self._Q_t: float = 1.0  # start at max quality (no fatigue)
        self._response_level: str = "normal"
        self._last_IG: float = 0.0
        self._abnormal_protection: bool = False    # §3.4.3 abnormal situation flag
        self._w_history: List[np.ndarray] = []     # weight snapshots for flip detection

    # ================================================================
    #  Trigger signals
    # ================================================================

    def compute_I_h(self, upv_state: Dict[str, float]) -> bool:
        """Uncertainty trigger: max_{s in S_key} (1 - gamma_s) > tau_h.

        Parameters
        ----------
        upv_state : dict
            Mapping of slot_name -> gamma (confidence) value.
            Typically from ``osv_engine.compute_confidence_score()`` per slot,
            or from UPV.sigma values.

        Returns
        -------
        bool
            True if high-uncertainty trigger fires.
        """
        if not upv_state:
            return False
        max_uncertainty = max(1.0 - gamma for gamma in upv_state.values())
        return max_uncertainty > self.cfg.tau_h

    def compute_I_d(self, conflict_scores: Dict[str, float]) -> bool:
        """Conflict trigger: ||d_t||_inf > tau_d.

        Parameters
        ----------
        conflict_scores : dict
            Mapping of source_name -> conflict score in [0, 1].
            Typically the per-source breakdown from
            ``osv_engine.compute_conflict_score()``.

        Returns
        -------
        bool
            True if high-conflict trigger fires.
        """
        if not conflict_scores:
            return False
        d_inf = max(conflict_scores.values())
        return d_inf > self.cfg.tau_d

    def compute_I_s(self, convergence_status: Any) -> bool:
        """Stagnation trigger: consecutive_stagnant_ticks > T_stag.

        Parameters
        ----------
        convergence_status : ConvergenceStatus
            Status object from ``ConvergenceDetector.check()``.

        Returns
        -------
        bool
            True if stagnation trigger fires.
        """
        consec = int(getattr(convergence_status, "consec_stagnant_ticks", 0))
        return consec > self.cfg.T_stag

    # ================================================================
    #  Main trigger decision
    # ================================================================

    def should_trigger(
        self,
        upv_state: Dict[str, float],
        conflict_scores: Dict[str, float],
        convergence_status: Any,
        cooldown_remaining: Optional[int] = None,
        budget_remaining: Optional[int] = None,
        feas_ratio: Optional[float] = None,
    ) -> bool:
        """Decide whether to trigger an interaction at this tick.

        An interaction is triggered iff:
          1. At least one signal (I_h, I_d, I_s) fires, AND
          2. Cooldown has expired, AND
          3. Budget is not exhausted (accounting for reserve), AND
          4. Response level is not "pause".

        Also checks §3.4.3 abnormal situation protection.

        Parameters
        ----------
        upv_state : dict
            Slot confidence mapping for I_h.
        conflict_scores : dict
            Per-source conflict scores for I_d.
        convergence_status : object
            ConvergenceStatus for I_s.
        cooldown_remaining : int, optional
            Override internal cooldown counter. If None, uses internal state.
        budget_remaining : int, optional
            Override budget remaining. If None, computed from internal state.
        feas_ratio : float, optional
            Current feasibility ratio for abnormal situation detection.

        Returns
        -------
        bool
            True if an interaction should be triggered.
        """
        if cooldown_remaining is None:
            cooldown_remaining = self._cooldown_remaining
        if budget_remaining is None:
            budget_remaining = self.cfg.B_total - self._budget_used

        # gate: cooldown
        if cooldown_remaining > 0:
            return False

        # gate: budget (preserve reserve for late stage)
        if budget_remaining <= self.cfg.B_reserve:
            # only allow if stagnation is severe (I_s)
            if not self.compute_I_s(convergence_status):
                return False

        if budget_remaining <= 0:
            return False

        # §3.4.3 abnormal situation protection
        # FR < 0.05 and constraint-related gamma > 0.6 → only allow threshold
        if feas_ratio is not None and feas_ratio < 0.05:
            r_gamma = [g for k, g in upv_state.items() if k.startswith("r_")]
            if r_gamma and float(np.mean(r_gamma)) > 0.6:
                self._abnormal_protection = True
            else:
                self._abnormal_protection = False
        else:
            self._abnormal_protection = False

        # gate: fatigue pause
        if self._response_level == "pause":
            return False

        # check signals
        i_h = self.compute_I_h(upv_state)
        i_d = self.compute_I_d(conflict_scores)
        i_s = self.compute_I_s(convergence_status)

        return i_h or i_d or i_s

    # ================================================================
    #  Channel selection
    # ================================================================

    def select_channel(
        self,
        phase: str,
        candidate_IGs: Dict[str, float],
    ) -> str:
        """Select the best interaction channel for the current phase.

        Per Eq. (3-30): a* = argmax [ IG(a|ξ_t) - λ·C(a) ]
        with a phase-priority bonus term.

        If abnormal protection is active (§3.4.3), forces "threshold".

        Parameters
        ----------
        phase : str
            Current optimisation phase: "early", "mid", or "late".
        candidate_IGs : dict
            Mapping of channel_name -> IG proxy score.

        Returns
        -------
        str
            Selected channel name.
        """
        # §3.4.3 abnormal protection → force threshold
        if self._abnormal_protection:
            return "threshold"

        phase = phase if phase in _PHASE_CHANNEL_PRIORITY else "mid"
        priority_list = _PHASE_CHANNEL_PRIORITY[phase]

        # A.2: warning 级别下，仅允许低认知成本通道（threshold + pairwise），
        #      跳过 nl_clarify / ranking / roi 等高成本通道，避免在反馈质量下降时加重用户负担
        if self._response_level == "warning":
            LOW_COST_ALLOWED = {"threshold", "pairwise"}
            filtered = [c for c in priority_list if c in LOW_COST_ALLOWED]
            priority_list = filtered if filtered else ["pairwise"]  # 兜底

        best_channel = priority_list[0]
        best_score = -float("inf")

        for rank, channel in enumerate(priority_list):
            ig_c = candidate_IGs.get(channel, 0.0)
            cost_c = CHANNEL_COST.get(channel, 0.3)
            # phase priority bonus decays with rank position
            priority_bonus = 1.0 / (1.0 + 0.5 * rank)
            # Eq. (3-30): IG - λ·C(a) + small priority bonus
            score_c = ig_c - self.cfg.lambda_cost * cost_c + 0.1 * priority_bonus
            if score_c > best_score:
                best_score = score_c
                best_channel = channel

        return best_channel

    def compute_IG_proxy(
        self,
        action_type: str,
        state: Dict[str, Any],
    ) -> float:
        """Compute proxy information gain for a candidate action.

        Based on expected gamma improvement (for uncertainty-reducing actions)
        or expected conflict reduction (for clarification actions).

        Parameters
        ----------
        action_type : str
            One of "pairwise", "ranking", "threshold", "roi", "nl_clarify".
        state : dict
            Current state dict with keys:
              - "gamma_slots": dict of slot -> confidence
              - "conflict_score": float
              - "phase": str
              - "feas_ratio": float

        Returns
        -------
        float
            Proxy IG value (higher = more informative).
        """
        gamma_slots = state.get("gamma_slots", {})
        conflict = state.get("conflict_score", 0.0)
        phase = state.get("phase", "mid")

        if action_type == "pairwise":
            # expected confidence boost on WEIGHT slots only (pairwise doesn't update r)
            w_gamma = [g for k, g in gamma_slots.items() if k.startswith("w_")]
            low_w = [1.0 - g for g in w_gamma if g < 0.6]
            if low_w:
                return float(np.mean(low_w)) * 0.8
            return 0.2

        elif action_type == "ranking":
            # ranking provides broader preference signal on WEIGHT slots
            w_gamma = [g for k, g in gamma_slots.items() if k.startswith("w_")]
            low_w = [1.0 - g for g in w_gamma if g < 0.7]
            if low_w:
                return float(np.mean(low_w)) * 0.6
            return 0.15

        elif action_type == "threshold":
            # threshold reduces constraint ambiguity AND reference point uncertainty
            sigma_r = state.get("sigma_r", 0.5)
            r_uncertainty_ig = sigma_r * 0.8
            conflict_ig = conflict * 0.7 if conflict > 0.3 else 0.0
            return max(r_uncertainty_ig, conflict_ig, 0.1)

        elif action_type == "roi":
            # ROI narrows search region — scales with r-uncertainty (moderate)
            sigma_r = state.get("sigma_r", 0.5)
            base = {"early": 0.3, "mid": 0.4, "late": 0.5}.get(phase, 0.4)
            return base + sigma_r * 0.3

        elif action_type == "nl_clarify":
            # NL clarification targets lowest-confidence slot
            if gamma_slots:
                min_gamma = min(gamma_slots.values())
                return (1.0 - min_gamma) * 0.9
            return 0.3

        return 0.1

    # ================================================================
    #  Cooldown adaptation
    # ================================================================

    def update_cooldown(self, IG_last: float) -> int:
        """Compute and set the next cooldown period.

        T_cool = T_min + (T_max - T_min) * exp(-kappa * IG_last)

        High information gain -> short cooldown (interact again soon).
        Low information gain -> long cooldown (wait longer).

        Parameters
        ----------
        IG_last : float
            Information gain from the most recent interaction.

        Returns
        -------
        int
            Cooldown period in ticks.
        """
        self._last_IG = float(IG_last)
        T_cool = self.cfg.T_min + (self.cfg.T_max - self.cfg.T_min) * math.exp(
            -self.cfg.kappa * max(0.0, IG_last)
        )
        # A.1: warning 级别下冷却 ×3.0（让 AdaS-Full 在反馈质量下降时主动节流，
        #      配合 A.2 的低认知成本通道筛选，形成差异化行为；原实现 warning 无任何效果）
        if self._response_level == "warning":
            T_cool *= 3.0
        self._cooldown_remaining = int(round(T_cool))
        return self._cooldown_remaining

    def tick_cooldown(self) -> int:
        """Decrement cooldown by one tick. Returns remaining cooldown."""
        self._cooldown_remaining = max(0, self._cooldown_remaining - 1)
        return self._cooldown_remaining

    # ================================================================
    #  Fatigue monitoring (Q_t)
    # ================================================================

    def compute_Q_t(self, interaction_log: List[Dict[str, Any]]) -> float:
        """Compute interaction quality indicator Q_t (Eq. 3-35).

        Q_t = 1 / (1 + α₁·(t_k/t̄) + α₂·n_rev + α₃·n_flip)

        Where:
          - t_k / t̄ : latest latency / historical mean latency (>1 = hesitant)
          - n_rev   : revision count from latest interaction (high = uncertain)
          - n_flip  : preference flip count in recent W_Q window (high = unstable)

        Returns Q_t ∈ (0, 1]. Higher = better quality (no fatigue).

        Parameters
        ----------
        interaction_log : list[dict]
            List of interaction records with keys:
              - "latency": float (seconds)
              - "quality": float (0..1)
              - "timestamp": float (epoch seconds)

        Returns
        -------
        float
            Quality indicator in (0, 1]. Higher = better quality.
        """
        self._interaction_log = interaction_log
        if not interaction_log or len(interaction_log) < 1:
            self._Q_t = 1.0  # no interactions yet → maximum quality
            return 1.0

        # --- signal 1: latency term ---
        # JCDE_QT_LATENCY selects the form (default "ratio" = the submitted held-out/benchmark engine):
        #   ratio          t_k / mean(all latencies incl. t_k), pooled over channels
        #   ratio_channel  t_k / mean(latencies of the same channel incl. t_k)            (Reviewer 4-4)
        #   excess         max(0, t_k / mean(previous latencies) - 1), first response 0    (Eq. 3.11)
        #   excess_channel max(0, t_k / mean(previous same-channel latencies) - 1)         (Eq. 3.11 + R4-4)
        mode = os.environ.get("JCDE_QT_LATENCY", "ratio")
        latencies = [r.get("latency", 0.0) for r in interaction_log]
        t_k = latencies[-1] if latencies else 0.0
        if mode in ("ratio_channel", "excess_channel"):
            channel = interaction_log[-1].get("channel")
            pool = [r.get("latency", 0.0) for r in interaction_log if r.get("channel") == channel]
        else:
            pool = latencies
        if mode.startswith("excess"):
            previous = pool[:-1]
            if not previous:
                latency_ratio = 0.0
            else:
                t_bar = float(np.mean(previous))
                if t_bar < 1e-6:
                    t_bar = 1.0
                latency_ratio = max(0.0, t_k / t_bar - 1.0)
        elif mode in ("ratio", "ratio_channel"):
            t_bar = float(np.mean(pool)) if pool else 1.0
            if t_bar < 1e-6:
                t_bar = 1.0
            latency_ratio = t_k / t_bar
        else:
            raise ValueError(f"unknown JCDE_QT_LATENCY {mode!r}")

        # --- signal 2: n_rev (revision count from latest quality) ---
        latest_quality = interaction_log[-1].get("quality", 1.0)
        n_rev = int((1.0 - latest_quality) * 3)

        # --- signal 3: n_flip (preference flips in window W_Q) ---
        W_Q = self.cfg.W_Q
        n_flip = self._count_preference_flips(W_Q)

        # --- Eq. 3-35 ---
        denominator = 1.0 + (
            self.cfg.alpha1 * latency_ratio
            + self.cfg.alpha2 * n_rev
            + self.cfg.alpha3 * n_flip
        )
        Q_t = 1.0 / denominator
        self._Q_t = float(np.clip(Q_t, 1e-6, 1.0))
        return self._Q_t

    def _count_preference_flips(self, W_Q: int) -> int:
        """Count weight direction flips in the most recent W_Q interactions.

        A flip occurs when adjacent weight updates change direction
        (e.g., w_i increases then decreases).
        """
        history = self._w_history
        if len(history) < 2:
            return 0
        recent = history[-W_Q:] if len(history) >= W_Q else history
        flips = 0
        for k in range(1, len(recent) - 1):
            delta_prev = recent[k] - recent[k - 1]
            delta_curr = recent[k + 1] - recent[k]
            # flip = sign change in any dimension
            sign_change = np.sign(delta_prev) * np.sign(delta_curr)
            flips += int(np.any(sign_change < 0))
        return flips

    def record_weight_snapshot(self, w: np.ndarray) -> None:
        """Record a weight snapshot for flip detection."""
        self._w_history.append(np.asarray(w, dtype=float).copy())

    def get_response_level(self, Q_t: float) -> str:
        """Determine response level from quality indicator Q_t.

        Q_t is higher = better quality (Eq. 3-35):
          - Q_t > tau_Q_high (0.6) → "normal"   (good quality, proceed)
          - tau_Q_low ≤ Q_t ≤ tau_Q_high → "warning" (degrading quality)
          - Q_t < tau_Q_low (0.3) → "pause"     (low quality, stop interactions)

        Parameters
        ----------
        Q_t : float
            Quality indicator from ``compute_Q_t()``.

        Returns
        -------
        str
            "normal" | "warning" | "pause"
        """
        if Q_t > self.cfg.tau_Q_high:
            self._response_level = "normal"
        elif Q_t >= self.cfg.tau_Q_low:
            self._response_level = "warning"
        else:
            self._response_level = "pause"
        return self._response_level

    # ================================================================
    #  Phase detection
    # ================================================================

    def detect_phase(self, feas_ratio: float, delta_hv: float) -> str:
        """Detect the current optimisation phase from process features.

        Per Table 3-5:
          - Early: FR_t < 0.3
          - Mid:   0.3 ≤ FR_t < 0.8 and ΔHV_t > 0
          - Late:  FR_t ≥ 0.8 and ΔHV_t ≈ 0

        Parameters
        ----------
        feas_ratio : float
            Fraction of feasible solutions in current population.
        delta_hv : float
            Recent HV improvement rate.

        Returns
        -------
        str
            "early" | "mid" | "late"
        """
        if feas_ratio < self.cfg.feas_early_threshold:
            return "early"
        if feas_ratio >= 0.8 and delta_hv < self.cfg.delta_hv_late_threshold:
            return "late"
        return "mid"

    # ================================================================
    #  Budget accounting
    # ================================================================

    def record_interaction(
        self,
        channel: str,
        IG_actual: float,
        latency: float = 0.0,
        quality: float = 1.0,
    ) -> None:
        """Record a completed interaction and update internal state.

        Parameters
        ----------
        channel : str
            Channel used (e.g., "pairwise", "ranking").
        IG_actual : float
            Measured information gain.
        latency : float
            Response time in seconds.
        quality : float
            Response quality score in [0, 1].
        """
        import time as _time

        self._budget_used += 1
        self._interaction_log.append({
            "channel": channel,
            "IG": float(IG_actual),
            "latency": float(latency),
            "quality": float(quality),
            "timestamp": _time.time(),
        })
        self.update_cooldown(IG_actual)

        # update fatigue
        Q_t = self.compute_Q_t(self._interaction_log)
        self.get_response_level(Q_t)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.cfg.B_total - self._budget_used)

    @property
    def cooldown_remaining(self) -> int:
        return self._cooldown_remaining
