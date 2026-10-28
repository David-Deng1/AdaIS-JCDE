# -*- coding: utf-8 -*-
"""
hitlo.core.simulated_dm
=========================
Parameterised Simulated Decision Maker (SDM) for §3.4 experiment validation.

Provides deterministic-yet-noisy responses to pairwise comparison, ranking,
threshold confirmation, ROI selection, and NL clarification queries.  Noise
level grows with fatigue (interaction count).

Each SDM profile is defined by a ground-truth UPV ``(w*, r*, C_h*)``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hitlo.core.upv_engine import asf_distance, asf_distance_batch

__all__ = [
    "SDMProfile",
    "SDM_PROFILES",
    "SimulatedDM",
]


# ------------------------------------------------------------------ #
#  SDM profile definition
# ------------------------------------------------------------------ #

@dataclass
class SDMProfile:
    """Ground-truth preference profile for a simulated DM."""
    name: str
    problem_id: str
    n_obj: int
    w_star: np.ndarray
    r_star: np.ndarray
    C_h_star: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""


def _make_profiles() -> Dict[str, SDMProfile]:
    """Build the built-in catalogue of 9 problem x 2 persona profiles."""
    profiles: Dict[str, SDMProfile] = {}

    # ---- ZDT1 (2-obj, convex PF) ----
    profiles["ZDT1_edge"] = SDMProfile(
        name="ZDT1_edge", problem_id="ZDT1", n_obj=2,
        w_star=np.array([0.8, 0.2]),
        r_star=np.array([0.2, 0.3]),
        description="ZDT1 edge: strongly prefer f1",
    )
    profiles["ZDT1_balanced"] = SDMProfile(
        name="ZDT1_balanced", problem_id="ZDT1", n_obj=2,
        w_star=np.array([0.5, 0.5]),
        r_star=np.array([0.35, 0.35]),
        description="ZDT1 balanced: equal preference",
    )

    # ---- ZDT2 (2-obj, non-convex PF) ----
    profiles["ZDT2_edge"] = SDMProfile(
        name="ZDT2_edge", problem_id="ZDT2", n_obj=2,
        w_star=np.array([0.3, 0.7]),
        r_star=np.array([0.6, 0.1]),
        description="ZDT2 edge: strongly prefer f2",
    )
    profiles["ZDT2_balanced"] = SDMProfile(
        name="ZDT2_balanced", problem_id="ZDT2", n_obj=2,
        w_star=np.array([0.5, 0.5]),
        r_star=np.array([0.4, 0.4]),
        description="ZDT2 balanced",
    )

    # ---- ZDT3 (2-obj, disconnected PF) ----
    profiles["ZDT3_edge"] = SDMProfile(
        name="ZDT3_edge", problem_id="ZDT3", n_obj=2,
        w_star=np.array([0.75, 0.25]),
        r_star=np.array([0.15, 0.0]),
        description="ZDT3 edge: prefer low f1 region",
    )
    profiles["ZDT3_balanced"] = SDMProfile(
        name="ZDT3_balanced", problem_id="ZDT3", n_obj=2,
        w_star=np.array([0.45, 0.55]),
        r_star=np.array([0.3, 0.2]),
        description="ZDT3 balanced",
    )

    # ---- ZDT4 (2-obj, multi-modal) ----
    profiles["ZDT4_edge"] = SDMProfile(
        name="ZDT4_edge", problem_id="ZDT4", n_obj=2,
        w_star=np.array([0.85, 0.15]),
        r_star=np.array([0.1, 0.5]),
        description="ZDT4 edge: strongly prefer f1",
    )
    profiles["ZDT4_balanced"] = SDMProfile(
        name="ZDT4_balanced", problem_id="ZDT4", n_obj=2,
        w_star=np.array([0.5, 0.5]),
        r_star=np.array([0.3, 0.6]),
        description="ZDT4 balanced",
    )

    # ---- DTLZ2 (M-obj, concave PF) ----
    profiles["DTLZ2_edge"] = SDMProfile(
        name="DTLZ2_edge", problem_id="DTLZ2", n_obj=2,
        w_star=np.array([0.7, 0.3]),
        r_star=np.array([0.3, 0.7]),
        description="DTLZ2 edge: prefer f1",
    )
    profiles["DTLZ2_balanced"] = SDMProfile(
        name="DTLZ2_balanced", problem_id="DTLZ2", n_obj=2,
        w_star=np.array([0.5, 0.5]),
        r_star=np.array([0.5, 0.5]),
        description="DTLZ2 balanced",
    )

    # ---- BNH (2-obj, constrained) ----
    profiles["BNH_edge"] = SDMProfile(
        name="BNH_edge", problem_id="BNH", n_obj=2,
        w_star=np.array([0.8, 0.2]),
        r_star=np.array([10.0, 5.0]),
        C_h_star=[{"kind": "obj", "idx": 0, "sign": "<=", "value": 80.0}],
        description="BNH edge: constrained, prefer f1",
    )
    profiles["BNH_balanced"] = SDMProfile(
        name="BNH_balanced", problem_id="BNH", n_obj=2,
        w_star=np.array([0.5, 0.5]),
        r_star=np.array([30.0, 10.0]),
        description="BNH balanced",
    )

    # ---- TNK (2-obj, constrained, trigonometric) ----
    profiles["TNK_edge"] = SDMProfile(
        name="TNK_edge", problem_id="TNK", n_obj=2,
        w_star=np.array([0.3, 0.7]),
        r_star=np.array([0.8, 0.3]),
        description="TNK edge: prefer f2",
    )
    profiles["TNK_balanced"] = SDMProfile(
        name="TNK_balanced", problem_id="TNK", n_obj=2,
        w_star=np.array([0.5, 0.5]),
        r_star=np.array([0.6, 0.6]),
        description="TNK balanced",
    )

    # ---- OSY (6-var, 2-obj, 6-constraint) ----
    profiles["OSY_edge"] = SDMProfile(
        name="OSY_edge", problem_id="OSY", n_obj=2,
        w_star=np.array([0.65, 0.35]),
        r_star=np.array([-100.0, 10.0]),
        C_h_star=[{"kind": "obj", "idx": 1, "sign": "<=", "value": 60.0}],
        description="OSY edge: prefer f1 with upper bound on f2",
    )
    profiles["OSY_balanced"] = SDMProfile(
        name="OSY_balanced", problem_id="OSY", n_obj=2,
        w_star=np.array([0.5, 0.5]),
        r_star=np.array([-150.0, 30.0]),
        description="OSY balanced",
    )

    # ---- WELDED-BEAM (4-var, 2-obj, 4-constraint) ----
    profiles["WELDED-BEAM_edge"] = SDMProfile(
        name="WELDED-BEAM_edge", problem_id="WELDED-BEAM", n_obj=2,
        w_star=np.array([0.75, 0.25]),
        r_star=np.array([5.0, 0.005]),
        description="WELDED-BEAM edge: prefer cost over deflection",
    )
    profiles["WELDED-BEAM_balanced"] = SDMProfile(
        name="WELDED-BEAM_balanced", problem_id="WELDED-BEAM", n_obj=2,
        w_star=np.array([0.5, 0.5]),
        r_star=np.array([15.0, 0.01]),
        description="WELDED-BEAM balanced",
    )

    # ---- SAV-AERO (2-obj: Cd0 vs -Cn, 6 constraints) ----
    # f1=零升阻力Cd0(越小越好), f2=负法向力系数-Cn(越小越好，即Cn越大越好=机动性)
    profiles["SAV-AERO_edge"] = SDMProfile(
        name="SAV-AERO_edge", problem_id="SAV-AERO", n_obj=2,
        w_star=np.array([0.8, 0.2]),
        r_star=np.array([0.05, -10.0]),
        C_h_star=[
            {"slot": "r_0", "value": 0.08, "tolerance": 0.03},   # Cd0阈值
            {"slot": "r_1", "value": -8.0, "tolerance": 2.0},    # Cn阈值
        ],
        description="SAV-AERO edge: 强调低阻力（效率优先），法向力次之",
    )
    profiles["SAV-AERO_balanced"] = SDMProfile(
        name="SAV-AERO_balanced", problem_id="SAV-AERO", n_obj=2,
        w_star=np.array([0.5, 0.5]),
        r_star=np.array([0.1, -8.0]),
        C_h_star=[
            {"slot": "r_0", "value": 0.12, "tolerance": 0.05},
            {"slot": "r_1", "value": -6.0, "tolerance": 2.0},
        ],
        description="SAV-AERO balanced: 阻力与法向力均衡",
    )

    return profiles


SDM_PROFILES: Dict[str, SDMProfile] = _make_profiles()


# ------------------------------------------------------------------ #
#  SimulatedDM
# ------------------------------------------------------------------ #

class SimulatedDM:
    """Parameterised simulated decision maker for controlled experiments.

    Parameters
    ----------
    w_star : np.ndarray
        Ground-truth weight vector.
    r_star : np.ndarray
        Ground-truth reference (aspiration) point.
    C_h_star : list[dict]
        Ground-truth hard constraints (list of ConstraintAtom-like dicts).
    problem : object
        pymoo Problem instance (used for variable bounds).
    sigma_noise : float
        Base noise level in [0, 1]. Higher = more inconsistent responses.
    fatigue_rate : float
        Per-interaction growth rate of effective noise.
    rng_seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        w_star: np.ndarray,
        r_star: np.ndarray,
        C_h_star: List[Dict[str, Any]],
        problem: Any,
        sigma_noise: float = 0.1,
        fatigue_rate: float = 0.0,
        rng_seed: Optional[int] = None,
    ):
        self.w_star = np.asarray(w_star, dtype=float).ravel()
        self.r_star = np.asarray(r_star, dtype=float).ravel()
        self.C_h_star = list(C_h_star) if C_h_star else []
        self.problem = problem
        self.sigma_base = float(np.clip(sigma_noise, 0.0, 1.0))
        self.sigma_effective = self.sigma_base
        self.fatigue_rate = float(fatigue_rate)
        self._interaction_count = 0
        self._K_total = 50  # normalisation constant for fatigue curve
        self.rng = np.random.default_rng(rng_seed)

    # ---- utility ----

    def _asf(self, f: np.ndarray) -> float:
        """Compute ASF distance to ground-truth aspiration point."""
        return asf_distance(f, self.w_star, self.r_star)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-float(np.clip(x, -20.0, 20.0))))

    # ---- fatigue model ----

    # Per-channel base latency in seconds (cf. §3.4.2 Eq. (3-32) 预估交互耗时)
    CHANNEL_BASE_LATENCY: Dict[str, float] = {
        "pairwise":   10.0,
        "ranking":    30.0,
        "threshold":  15.0,
        "roi":        20.0,
        "nl_clarify": 60.0,
    }
    # Latency fatigue coefficient (β): latency stretch factor at k=K_total when fat=1
    BETA_LAT: float = 2.0
    # Quality fatigue coefficient (γ): quality drop factor at k=K_total when fat=1
    GAMMA_QUAL: float = 0.6

    def update_fatigue(self, interaction_count: int) -> None:
        """Update effective noise level based on cumulative interaction count.

        sigma_effective = sigma_base + fatigue_rate * k / K_total
        """
        self._interaction_count = interaction_count
        k = float(interaction_count)
        self.sigma_effective = self.sigma_base + self.fatigue_rate * k / self._K_total

    def _fatigue_progress(self) -> float:
        """归一化疲劳进度 p = fatigue_rate * k / K_total，钳在 [0, 1]."""
        k = float(self._interaction_count)
        return float(np.clip(self.fatigue_rate * k / self._K_total, 0.0, 1.0))

    def simulate_latency(self, channel: str) -> float:
        """在当前疲劳水平下模拟决策者响应耗时（秒）。

        latency = base(channel) · (1 + β · p) · ε,  ε ~ N(1, 0.15²)

        其中 p 为归一化疲劳进度。依据 Boksem et al. (2005) —— 心理疲劳
        导致响应时间显著增长；基线耗时参考 §3.4.2 的 t̂(a) 估计。
        """
        base = self.CHANNEL_BASE_LATENCY.get(channel, 15.0)
        p = self._fatigue_progress()
        noise = float(self.rng.normal(1.0, 0.15))
        return float(max(1.0, base * (1.0 + self.BETA_LAT * p) * max(0.4, noise)))

    def simulate_quality(self, channel: str) -> float:
        """在当前疲劳水平下模拟反馈质量 ∈ [0, 1]（高=低修订率）。

        quality = (1 - sigma_effective) · (1 - γ · p) + ξ,  ξ ~ N(0, 0.05²)

        依据 Boksem et al. (2005) 疲劳增加错误率的结论；sigma_effective
        已捕捉响应一致性，γ·p 额外注入"修订/犹豫"行为随疲劳的单调下降。
        """
        p = self._fatigue_progress()
        q_base = (1.0 - float(self.sigma_effective)) * (1.0 - self.GAMMA_QUAL * p)
        jitter = float(self.rng.normal(0.0, 0.05))
        return float(np.clip(q_base + jitter, 0.0, 1.0))

    # ---- response channels ----

    def respond_pairwise(
        self,
        x_i: np.ndarray,
        x_j: np.ndarray,
        f_i: np.ndarray,
        f_j: np.ndarray,
    ) -> int:
        """Pairwise comparison: return 0 if x_i preferred, 1 if x_j preferred.

        Noise flip probability:  p_flip = sigma * sigmoid(1 / |U_i - U_j|)
        When |U_i - U_j| is small (hard to distinguish), flip probability rises.
        """
        u_i = self._asf(f_i)
        u_j = self._asf(f_j)
        true_choice = 0 if u_i <= u_j else 1  # lower ASF = better

        diff = abs(u_i - u_j)
        # avoid division by zero
        discriminability = 1.0 / max(diff, 1e-8)
        p_flip = self.sigma_effective * self._sigmoid(discriminability)

        if self.rng.random() < p_flip:
            return 1 - true_choice
        return true_choice

    def respond_ranking(
        self,
        candidates_F: np.ndarray,
        k: int = 5,
    ) -> List[int]:
        """Rank candidates by ASF; apply adjacent-swap noise.

        Returns list of indices in preference order (best first).
        """
        candidates_F = np.asarray(candidates_F, dtype=float)
        n = len(candidates_F)
        k = min(k, n)

        # compute ASF for each candidate
        asf_vals = np.array([self._asf(candidates_F[i]) for i in range(n)])
        order = list(np.argsort(asf_vals))  # ascending = best first

        # adjacent-swap noise
        p_swap = self.sigma_effective
        for i in range(len(order) - 1):
            if self.rng.random() < p_swap:
                order[i], order[i + 1] = order[i + 1], order[i]

        return order[:k]

    def respond_threshold(
        self,
        slot_name: str,
        proposed_value: float,
    ) -> Tuple[bool, float]:
        """Threshold confirmation: compare proposed value against ground truth.

        If within tolerance, accept (True, proposed_value).
        Otherwise, reject and return (False, true_value + noise).
        """
        # map slot names to ground-truth values
        true_value = self._get_ground_truth_slot(slot_name)
        if true_value is None:
            # unknown slot -> always accept
            return True, proposed_value

        tolerance = 0.15 * (abs(true_value) + 1e-6)
        if abs(proposed_value - true_value) <= tolerance:
            return True, proposed_value
        else:
            noise = self.rng.normal(0, self.sigma_effective * 0.1 * (abs(true_value) + 1e-6))
            return False, true_value + noise

    def respond_roi(
        self,
        pop_F: np.ndarray,
        projection_dims: Tuple[int, int] = (0, 1),
    ) -> Tuple[float, float, float, float]:
        """Return a bounding box in objective space around the w*-optimal region.

        Returns (f_dim0_min, f_dim0_max, f_dim1_min, f_dim1_max).
        """
        pop_F = np.asarray(pop_F, dtype=float)
        if len(pop_F) == 0:
            return (0.0, 1.0, 0.0, 1.0)

        # find best solution according to w*
        asf_vals = np.array([self._asf(pop_F[i]) for i in range(len(pop_F))])
        best_idx = int(np.argmin(asf_vals))
        best_f = pop_F[best_idx]

        # define neighbourhood: select solutions within top-20% ASF
        threshold = np.percentile(asf_vals, 20)
        mask = asf_vals <= threshold
        if np.sum(mask) < 2:
            mask = asf_vals <= np.percentile(asf_vals, 40)

        neighbourhood_F = pop_F[mask]
        d0, d1 = projection_dims

        f0_min = float(np.min(neighbourhood_F[:, d0]))
        f0_max = float(np.max(neighbourhood_F[:, d0]))
        f1_min = float(np.min(neighbourhood_F[:, d1]))
        f1_max = float(np.max(neighbourhood_F[:, d1]))

        # add small noise to bbox edges
        noise_scale = self.sigma_effective * 0.05
        f0_range = max(f0_max - f0_min, 1e-6)
        f1_range = max(f1_max - f1_min, 1e-6)
        f0_min += self.rng.normal(0, noise_scale * f0_range)
        f0_max += self.rng.normal(0, noise_scale * f0_range)
        f1_min += self.rng.normal(0, noise_scale * f1_range)
        f1_max += self.rng.normal(0, noise_scale * f1_range)

        # ensure valid bbox
        if f0_min > f0_max:
            f0_min, f0_max = f0_max, f0_min
        if f1_min > f1_max:
            f1_min, f1_max = f1_max, f1_min

        return (f0_min, f0_max, f1_min, f1_max)

    def respond_nl_clarify(
        self,
        slot_name: str,
    ) -> Dict[str, Any]:
        """Return the ground-truth value for a given UPV slot.

        Simulates the DM providing a precise NL clarification.
        """
        result: Dict[str, Any] = {"slot": slot_name}

        if slot_name.startswith("w_"):
            try:
                idx = int(slot_name.split("_")[1])
                result["value"] = float(self.w_star[idx])
                result["source"] = "weight"
            except (IndexError, ValueError):
                result["value"] = None
        elif slot_name.startswith("r_"):
            try:
                idx = int(slot_name.split("_")[1])
                result["value"] = float(self.r_star[idx])
                result["source"] = "reference"
            except (IndexError, ValueError):
                result["value"] = None
        elif slot_name.startswith("C_h_"):
            try:
                idx = int(slot_name.split("_")[2])
                if idx < len(self.C_h_star):
                    result["value"] = copy.deepcopy(self.C_h_star[idx])
                    result["source"] = "constraint"
                else:
                    result["value"] = None
            except (IndexError, ValueError):
                result["value"] = None
        else:
            result["value"] = self._get_ground_truth_slot(slot_name)

        return result

    # ---- internal helpers ----

    def _get_ground_truth_slot(self, slot_name: str) -> Optional[float]:
        """Resolve a slot name to its ground-truth scalar value."""
        if slot_name.startswith("w_"):
            try:
                idx = int(slot_name.split("_")[1])
                return float(self.w_star[idx])
            except (IndexError, ValueError):
                return None
        elif slot_name.startswith("r_"):
            try:
                idx = int(slot_name.split("_")[1])
                return float(self.r_star[idx])
            except (IndexError, ValueError):
                return None
        elif slot_name.startswith("C_h_"):
            try:
                idx = int(slot_name.split("_")[2])
                if idx < len(self.C_h_star):
                    return float(self.C_h_star[idx].get("value", 0.0))
            except (IndexError, ValueError):
                return None
        return None

    @classmethod
    def from_profile(
        cls,
        profile_name: str,
        problem: Any,
        sigma_noise: float = 0.1,
        fatigue_rate: float = 0.0,
        rng_seed: Optional[int] = None,
    ) -> "SimulatedDM":
        """Create a SimulatedDM from a named profile in SDM_PROFILES."""
        if profile_name not in SDM_PROFILES:
            raise KeyError(
                f"Unknown SDM profile '{profile_name}'. "
                f"Available: {sorted(SDM_PROFILES.keys())}"
            )
        p = SDM_PROFILES[profile_name]
        return cls(
            w_star=p.w_star.copy(),
            r_star=p.r_star.copy(),
            C_h_star=copy.deepcopy(p.C_h_star),
            problem=problem,
            sigma_noise=sigma_noise,
            fatigue_rate=fatigue_rate,
            rng_seed=rng_seed,
        )
