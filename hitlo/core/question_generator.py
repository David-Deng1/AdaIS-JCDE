# -*- coding: utf-8 -*-
"""
hitlo.core.question_generator
================================
Question generator for §3.4 interaction channels.

Selects informative pairwise comparison pairs, ranking candidates,
threshold questions, ROI projections, and NL clarification prompts
based on current population state and UPV confidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hitlo.core.types import UPV
from hitlo.core.upv_engine import asf_distance, asf_distance_batch, phi_objective

__all__ = [
    "QuestionGenerator",
]


class QuestionGenerator:
    """Generates structured questions for each interaction channel.

    Uses population diversity, UPV confidence, and problem structure
    to construct maximally informative queries.
    """

    def __init__(self, rng_seed: Optional[int] = None):
        self.rng = np.random.default_rng(rng_seed)

    # ================================================================
    #  Pairwise selection (Eq. 3-29: discriminability D)
    # ================================================================

    def select_pairwise(
        self,
        pop_F: np.ndarray,
        upv: UPV,
        k_pairs: int = 3,
    ) -> List[Tuple[int, int]]:
        """Select informative pairwise comparison pairs.

        Computes discriminability D(i,j) for all candidate pairs and
        selects pairs with mid-range D values (not too easy, not too hard).

        D(i,j) = |ASF(f_i, w, r) - ASF(f_j, w, r)|

        Pairs with D in the 30th-70th percentile range are most informative:
        too-easy pairs (high D) waste a question, too-hard pairs (low D)
        yield noisy responses.

        Parameters
        ----------
        pop_F : np.ndarray
            Population objective values, shape (N, M).
        upv : UPV
            Current UPV state.
        k_pairs : int
            Number of pairs to select.

        Returns
        -------
        list[tuple[int, int]]
            List of (i, j) index pairs into pop_F.
        """
        pop_F = np.asarray(pop_F, dtype=float)
        N = len(pop_F)
        if N < 2:
            return []

        w = np.asarray(upv.w, dtype=float)
        r = np.asarray(upv.r, dtype=float)

        # compute ASF for all solutions
        asf_vals = np.array([asf_distance(pop_F[i], w, r) for i in range(N)])

        # build candidate pairs with discriminability scores
        # Eq. 3-29: D(x_i,x_j) = |E_UPV[U(x_i)] - E_UPV[U(x_j)]| · ||f(x_i)-f(x_j)||_∞
        pairs: List[Tuple[int, int, float]] = []
        for i in range(N):
            for j in range(i + 1, N):
                d_utility = abs(asf_vals[i] - asf_vals[j])
                d_objective = float(np.max(np.abs(pop_F[i] - pop_F[j])))
                d_ij = d_utility * d_objective
                pairs.append((i, j, d_ij))

        if not pairs:
            return []

        # select pairs with D in the 30th-70th percentile (mid-range)
        d_values = np.array([p[2] for p in pairs])
        p30 = np.percentile(d_values, 30)
        p70 = np.percentile(d_values, 70)

        mid_pairs = [(i, j, d) for (i, j, d) in pairs if p30 <= d <= p70]

        # if not enough mid-range pairs, relax to 20th-80th
        if len(mid_pairs) < k_pairs:
            p20 = np.percentile(d_values, 20)
            p80 = np.percentile(d_values, 80)
            mid_pairs = [(i, j, d) for (i, j, d) in pairs if p20 <= d <= p80]

        # if still not enough, use all pairs
        if len(mid_pairs) < k_pairs:
            mid_pairs = list(pairs)

        # randomly sample from qualified pairs
        k_pairs = min(k_pairs, len(mid_pairs))
        indices = self.rng.choice(len(mid_pairs), size=k_pairs, replace=False)
        return [(mid_pairs[idx][0], mid_pairs[idx][1]) for idx in indices]

    # ================================================================
    #  Ranking selection (cluster + feasibility stratified sampling)
    # ================================================================

    def select_ranking(
        self,
        pop_F: np.ndarray,
        pop_G: Optional[np.ndarray],
        k: int = 5,
    ) -> List[int]:
        """Select diverse candidates for ranking query.

        Uses simple k-means-style clustering on objective space combined
        with feasibility stratified sampling to ensure the DM sees both
        feasible and boundary solutions.

        Parameters
        ----------
        pop_F : np.ndarray
            Population objective values, shape (N, M).
        pop_G : np.ndarray or None
            Constraint violation values, shape (N, n_constr) or None.
        k : int
            Number of candidates to select for ranking.

        Returns
        -------
        list[int]
            Indices of selected candidates in pop_F.
        """
        pop_F = np.asarray(pop_F, dtype=float)
        N = len(pop_F)
        if N <= k:
            return list(range(N))

        # determine feasibility
        if pop_G is not None and len(pop_G) > 0:
            pop_G = np.asarray(pop_G, dtype=float)
            # feasible = all constraints <= 0
            feas_mask = np.all(pop_G <= 1e-8, axis=1)
        else:
            feas_mask = np.ones(N, dtype=bool)

        n_feasible = int(np.sum(feas_mask))
        n_infeasible = N - n_feasible

        # stratified allocation: at least 1 infeasible if available
        if n_infeasible > 0 and n_feasible > 0:
            k_infeas = max(1, min(k // 3, n_infeasible))
            k_feas = k - k_infeas
        elif n_feasible == 0:
            k_infeas = k
            k_feas = 0
        else:
            k_infeas = 0
            k_feas = k

        selected: List[int] = []

        # select from feasible pool via clustering
        if k_feas > 0:
            feas_indices = np.where(feas_mask)[0]
            feas_F = pop_F[feas_indices]
            cluster_picks = self._kmeans_sample(feas_F, k_feas)
            selected.extend(feas_indices[cluster_picks].tolist())

        # select from infeasible pool via clustering
        if k_infeas > 0:
            infeas_indices = np.where(~feas_mask)[0]
            infeas_F = pop_F[infeas_indices]
            cluster_picks = self._kmeans_sample(infeas_F, k_infeas)
            selected.extend(infeas_indices[cluster_picks].tolist())

        return selected[:k]

    def _kmeans_sample(
        self,
        F: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Simple k-means++ inspired diverse sampling.

        Iteratively selects the point farthest from already selected points
        (max-min distance), similar to k-means++ initialisation.

        Parameters
        ----------
        F : np.ndarray
            Objective values of the candidate pool, shape (N, M).
        k : int
            Number of samples.

        Returns
        -------
        np.ndarray
            Indices into F.
        """
        N = len(F)
        if N <= k:
            return np.arange(N)

        # normalise objectives for distance computation
        F_min = np.min(F, axis=0)
        F_range = np.max(F, axis=0) - F_min
        F_range = np.where(F_range < 1e-12, 1.0, F_range)
        F_norm = (F - F_min) / F_range

        # start with a random point
        selected = [int(self.rng.integers(N))]
        min_dists = np.full(N, np.inf)

        for _ in range(k - 1):
            # update min distances to selected set
            last = F_norm[selected[-1]]
            dists = np.sum((F_norm - last) ** 2, axis=1)
            min_dists = np.minimum(min_dists, dists)
            min_dists[selected] = -1.0  # exclude already selected

            # select farthest point
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)

        return np.array(selected)

    # ================================================================
    #  Threshold question
    # ================================================================

    def generate_threshold_question(
        self,
        upv: UPV,
        slot_name: str,
        pop_F: np.ndarray,
    ) -> Dict[str, Any]:
        """Generate a threshold confirmation question.

        Based on the current population distribution, proposes a threshold
        value and several alternatives for the DM to choose from.

        Parameters
        ----------
        upv : UPV
            Current UPV state.
        slot_name : str
            Name of the UPV slot to query (e.g., "r_0", "r_1").
        pop_F : np.ndarray
            Population objective values for context.

        Returns
        -------
        dict
            Question structure with keys:
              - "slot": slot name
              - "proposed": proposed value
              - "alternatives": list of alternative values
              - "context": population statistics for the relevant dimension
        """
        pop_F = np.asarray(pop_F, dtype=float)
        result: Dict[str, Any] = {"slot": slot_name, "type": "threshold"}

        # determine which objective dimension this slot refers to
        if slot_name.startswith("r_"):
            try:
                dim = int(slot_name.split("_")[1])
            except (IndexError, ValueError):
                dim = 0
        else:
            dim = 0

        if dim < pop_F.shape[1]:
            f_col = pop_F[:, dim]
            f_min = float(np.min(f_col))
            f_max = float(np.max(f_col))
            f_median = float(np.median(f_col))
            f_q25 = float(np.percentile(f_col, 25))
            f_q75 = float(np.percentile(f_col, 75))
        else:
            f_min = f_max = f_median = f_q25 = f_q75 = 0.0

        # current UPV value for this slot
        current_value = self._get_upv_slot_value(upv, slot_name)

        # propose value based on current + population median
        if current_value is not None:
            proposed = 0.5 * current_value + 0.5 * f_median
        else:
            proposed = f_median

        # generate alternatives at different quantile positions
        alternatives = [f_q25, f_median, f_q75]
        # add a point between proposed and current if different
        if current_value is not None and abs(proposed - current_value) > 1e-8:
            alternatives.append(0.5 * (proposed + current_value))
        # deduplicate and sort
        alternatives = sorted(set(round(v, 6) for v in alternatives))

        result["proposed"] = round(proposed, 6)
        result["alternatives"] = alternatives
        result["context"] = {
            "f_min": f_min,
            "f_max": f_max,
            "f_median": f_median,
            "f_q25": f_q25,
            "f_q75": f_q75,
        }
        return result

    # ================================================================
    #  ROI question
    # ================================================================

    def generate_roi_question(
        self,
        pop_F: np.ndarray,
        phase: str,
    ) -> Dict[str, Any]:
        """Generate an ROI (Region of Interest) question.

        Selects appropriate projection dimensions based on the current
        optimisation phase.

        Parameters
        ----------
        pop_F : np.ndarray
            Population objective values, shape (N, M).
        phase : str
            Current phase: "early", "mid", or "late".

        Returns
        -------
        dict
            Question structure with keys:
              - "projection_dims": tuple of 2 objective indices
              - "current_range": dict of min/max per dimension
              - "phase": current phase
              - "instruction": human-readable instruction
        """
        pop_F = np.asarray(pop_F, dtype=float)
        M = pop_F.shape[1] if pop_F.ndim == 2 else 1

        # select projection dimensions
        if M <= 2:
            projection_dims = (0, min(1, M - 1))
        else:
            # for higher-dim: early focuses on first two objectives,
            # mid/late selects the two most spread objectives
            if phase == "early":
                projection_dims = (0, 1)
            else:
                # pick the two dimensions with highest variance
                variances = np.var(pop_F, axis=0)
                top2 = np.argsort(variances)[-2:]
                projection_dims = (int(min(top2)), int(max(top2)))

        d0, d1 = projection_dims
        current_range = {}
        if len(pop_F) > 0:
            current_range = {
                f"f{d0}_min": float(np.min(pop_F[:, d0])),
                f"f{d0}_max": float(np.max(pop_F[:, d0])),
                f"f{d1}_min": float(np.min(pop_F[:, d1])),
                f"f{d1}_max": float(np.max(pop_F[:, d1])),
            }

        instruction_map = {
            "early": "请在目标空间中框选您感兴趣的大致区域（可以较宽泛）。",
            "mid":   "请缩小感兴趣区域以聚焦搜索方向。",
            "late":  "请精确标注最终偏好区域以辅助收敛。",
        }

        return {
            "type": "roi",
            "projection_dims": projection_dims,
            "current_range": current_range,
            "phase": phase,
            "instruction": instruction_map.get(phase, instruction_map["mid"]),
        }

    # ================================================================
    #  NL clarification
    # ================================================================

    def generate_nl_clarify(
        self,
        upv: UPV,
        low_gamma_slots: List[str],
    ) -> Dict[str, Any]:
        """Generate a closed-form NL clarification question.

        Targets the lowest-confidence slots and constructs a focused
        question to elicit a precise response from the DM.

        Parameters
        ----------
        upv : UPV
            Current UPV state.
        low_gamma_slots : list[str]
            Slot names with low confidence, ordered by ascending gamma.

        Returns
        -------
        dict
            Question structure with keys:
              - "type": "nl_clarify"
              - "target_slot": the slot being queried
              - "current_value": current value in UPV
              - "question_text": generated question string
              - "response_format": expected format
        """
        if not low_gamma_slots:
            return {
                "type": "nl_clarify",
                "target_slot": None,
                "question_text": "当前所有偏好槽位置信度均较高，无需进一步澄清。",
            }

        target_slot = low_gamma_slots[0]
        current_value = self._get_upv_slot_value(upv, target_slot)

        # generate question based on slot type
        if target_slot.startswith("w_"):
            try:
                idx = int(target_slot.split("_")[1])
                question = (
                    f"关于第{idx+1}个目标的重要性："
                    f"当前设定权重为 {current_value:.3f}，"
                    f"您希望调整吗？请给出0到1之间的权重值。"
                )
                response_format = "float in [0, 1]"
            except (IndexError, ValueError):
                question = f"请明确'{target_slot}'的期望值。"
                response_format = "float"
        elif target_slot.startswith("r_"):
            try:
                idx = int(target_slot.split("_")[1])
                question = (
                    f"关于第{idx+1}个目标的期望值（参考点）："
                    f"当前设定为 {current_value:.4f}，"
                    f"您期望的目标值是多少？"
                )
                response_format = "float"
            except (IndexError, ValueError):
                question = f"请明确'{target_slot}'的期望值。"
                response_format = "float"
        elif target_slot.startswith("C_h_"):
            question = (
                f"关于硬约束'{target_slot}'："
                f"当前阈值为 {current_value}，"
                f"这个约束边界是否准确？请确认或修正。"
            )
            response_format = "float or 'confirm'"
        else:
            question = f"请进一步明确'{target_slot}'的偏好设定。"
            response_format = "free text"

        return {
            "type": "nl_clarify",
            "target_slot": target_slot,
            "current_value": current_value,
            "question_text": question,
            "response_format": response_format,
        }

    # ================================================================
    #  Internal helpers
    # ================================================================

    @staticmethod
    def _get_upv_slot_value(upv: UPV, slot_name: str) -> Optional[float]:
        """Get the scalar value of a UPV slot by name."""
        if slot_name.startswith("w_"):
            try:
                idx = int(slot_name.split("_")[1])
                w = np.asarray(upv.w, dtype=float)
                if idx < len(w):
                    return float(w[idx])
            except (IndexError, ValueError):
                pass
        elif slot_name.startswith("r_"):
            try:
                idx = int(slot_name.split("_")[1])
                r = np.asarray(upv.r, dtype=float)
                if idx < len(r):
                    return float(r[idx])
            except (IndexError, ValueError):
                pass
        elif slot_name == "tau":
            return float(upv.tau)
        elif slot_name.startswith("C_h_"):
            try:
                idx = int(slot_name.split("_")[2])
                if idx < len(upv.C_h):
                    return float(upv.C_h[idx].value)
            except (IndexError, ValueError):
                pass
        return None
