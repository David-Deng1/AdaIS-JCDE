"""Ground-truth evaluation helpers for the JCDE revision (read-only: no RNG use, no state mutation).

The simulated decision maker holds a ground-truth preference (w*, r*) and judges solutions by the
achievement scalarizing function ASF(f; w*, r*) (lower is better). These helpers log, per tick,
how good the current population is under that ground truth, and label each feedback event as
consistent or not with the noise-free ground-truth answer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from hitlo.core.upv_engine import asf_distance

TRUTH_FIELDS = ["n_feasible_pop", "u_true_best", "u_true_rec", "rec_f0", "rec_f1"]


def _feasible_mask(pop_F: np.ndarray, pop_G: Optional[np.ndarray]) -> np.ndarray:
    if pop_G is None:
        return np.ones(len(pop_F), dtype=bool)
    G = np.asarray(pop_G, dtype=float)
    if G.ndim == 1:
        G = G.reshape(len(pop_F), -1)
    if G.shape[1] == 0:
        return np.ones(len(pop_F), dtype=bool)
    return np.all(G <= 0.0, axis=1)


def truth_row(state: Any, sdm: Any) -> Dict[str, float]:
    """Ground-truth utility of the best feasible design and of the design the system would recommend.

    u_true_best: min ASF(f; w*, r*) over the feasible population (what an oracle would pick).
    u_true_rec:  ASF(f_rec; w*, r*) where f_rec minimises ASF under the system's current estimate
                 (w, r) - i.e. the ground-truth quality of the design the system would recommend.
    """
    nan = float("nan")
    pop_F = getattr(state, "pop_F", None)
    if pop_F is None or len(pop_F) == 0:
        return {
            "n_feasible_pop": 0,
            "u_true_best": nan,
            "u_true_rec": nan,
            "rec_f0": nan,
            "rec_f1": nan,
        }
    F = np.asarray(pop_F, dtype=float)
    feas = _feasible_mask(F, getattr(state, "pop_G", None))
    if not np.any(feas):
        return {
            "n_feasible_pop": 0,
            "u_true_best": nan,
            "u_true_rec": nan,
            "rec_f0": nan,
            "rec_f1": nan,
        }
    Ff = F[feas]
    u_true = np.array([sdm._asf(f) for f in Ff])  # the SDM's own utility (v1: ASF with w*, r*)
    w_est = np.asarray(state.upv.w, dtype=float)
    r_est = np.asarray(state.upv.r, dtype=float)
    u_est = np.array([asf_distance(f, w_est, r_est) for f in Ff])
    rec = int(np.argmin(u_est))
    return {
        "n_feasible_pop": int(feas.sum()),
        "u_true_best": round(float(u_true.min()), 8),
        "u_true_rec": round(float(u_true[rec]), 8),
        "rec_f0": round(float(Ff[rec, 0]), 8),
        "rec_f1": round(float(Ff[rec, 1]), 8) if Ff.shape[1] > 1 else nan,
    }


def label_response(
    channel: str,
    question: Dict[str, Any],
    record: Dict[str, Any],
    sdm: Any,
    pop_F: np.ndarray,
) -> Dict[str, Any]:
    """Compare a feedback event with the noise-free ground-truth answer (no RNG calls)."""
    F = np.asarray(pop_F, dtype=float)
    asf = sdm._asf  # the SDM's own utility (v1: ASF with w*, r*)
    try:
        if channel == "pairwise":
            res = record.get("results", [])
            ok = [
                int(r["choice"] == (0 if asf(F[r["i"]]) <= asf(F[r["j"]]) else 1))
                for r in res
            ]
            return {
                "truth_correct": float(np.mean(ok)) if ok else None,
                "truth_n": len(ok),
            }
        if channel == "ranking":
            cand = list(question.get("candidates", []))
            ranking = list(record.get("ranking", []))
            if not cand or not ranking:
                return {"truth_correct": None}
            true_order = list(np.argsort([asf(F[c]) for c in cand]))
            k = len(ranking)
            top1 = int(ranking[0] == true_order[0])
            overlap = len(set(ranking[:k]) & set(true_order[:k])) / k
            return {"truth_correct": float(top1), "truth_topk_overlap": overlap}
        if channel == "threshold":
            slot = record.get("slot")
            true_value = sdm._get_ground_truth_slot(slot) if slot else None
            if true_value is None:
                return {"truth_correct": None}
            tol = 0.15 * (abs(true_value) + 1e-6)
            should_accept = abs(float(record.get("proposed", 0.0)) - true_value) <= tol
            value_err = abs(float(record.get("value", 0.0)) - true_value) / (
                abs(true_value) + 1e-6
            )
            return {
                "truth_correct": float(bool(record.get("accepted")) == should_accept),
                "truth_value_rel_err": round(value_err, 6),
            }
        if channel == "roi":
            bbox = record.get("bbox")
            d0, d1 = record.get("dims", (0, 1))
            if bbox is None or len(F) == 0:
                return {"truth_correct": None}
            # Noise-free box from the same rule the SDM uses (top-20% ASF neighbourhood).
            u = np.array([asf(f) for f in F])
            mask = u <= np.percentile(u, 20)
            if mask.sum() < 2:
                mask = u <= np.percentile(u, 40)
            nb = F[mask]
            true_box = (
                nb[:, d0].min(),
                nb[:, d0].max(),
                nb[:, d1].min(),
                nb[:, d1].max(),
            )
            ix = max(0.0, min(bbox[1], true_box[1]) - max(bbox[0], true_box[0]))
            iy = max(0.0, min(bbox[3], true_box[3]) - max(bbox[2], true_box[2]))
            area = lambda b: max(b[1] - b[0], 1e-12) * max(b[3] - b[2], 1e-12)  # noqa: E731
            inter = ix * iy
            iou = inter / (area(bbox) + area(true_box) - inter)
            return {
                "truth_correct": float(iou >= 0.5),
                "truth_iou": round(float(iou), 4),
            }
        if channel == "nl_clarify":
            return {
                "truth_correct": 1.0,
                "truth_note": "simulated NL returns ground-truth slot value",
            }
    except Exception as exc:  # labels must never break a run
        return {"truth_correct": None, "truth_error": repr(exc)}
    return {"truth_correct": None}
