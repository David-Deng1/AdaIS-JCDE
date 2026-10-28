#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_scheduling_exp.py
======================
Automated experiment runner for §3.4: Adaptive Interaction Scheduling.

Runs a single (method, problem, seed, profile) experiment, producing:
  - metrics.csv          per-tick performance + scheduling columns
  - interaction_log.jsonl detailed interaction records

Usage examples::

    python run_scheduling_exp.py --method AdaS-Full --problem ZDT1 --seed 0 --profile P-balanced
    python run_scheduling_exp.py --method Baseline-F --problem BNH --seed 42 --profile P-edge --fatigue_rate 0.02
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
#  Imports from HITLO
# ---------------------------------------------------------------------------
from hitlo.engine.opt_state import OptState
from hitlo.core.experiment import ExperimentConfig
from hitlo.core.types import UPV, ConstraintAtom
from hitlo.core.upv_engine import (
    normalize_w,
    apply_patch_to_upv,
    asf_distance,
    asf_distance_batch,
    compute_sat_prefgap,
)
from hitlo.core.osv_engine import compute_conflict_score, compute_confidence_score
from hitlo.core.simulated_dm import SimulatedDM, SDM_PROFILES
from hitlo.core.objective_scale import scaled
from hitlo.core.interaction_scheduler import (
    InteractionScheduler,
    SchedulerConfig,
    CHANNEL_COST,
)
from hitlo.core.question_generator import QuestionGenerator
from hitlo.problems.catalog import PROBLEM_CATALOG

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # sibling helpers (truth_metrics, relay)
from truth_metrics import TRUTH_FIELDS, label_response, truth_row  # noqa: E402


# ===================================================================
#  Problem -> default n_var lookup
# ===================================================================


def _default_nvar(problem_id: str) -> int:
    pid = problem_id.upper()
    meta = PROBLEM_CATALOG.get(pid)
    if meta:
        return meta["default_nvar"]
    return 30


# ===================================================================
#  SDM profile resolution
# ===================================================================

_PROFILE_SUFFIX = {"P-balanced": "balanced", "P-edge": "edge"}


def _resolve_sdm_profile(problem_id: str, profile: str) -> str:
    """Resolve (problem, profile_label) -> SDM_PROFILES key."""
    pid = problem_id.upper()
    suffix = _PROFILE_SUFFIX.get(profile, profile)
    key = f"{pid}_{suffix}"
    if key in SDM_PROFILES:
        return key
    # try partial match
    for k in SDM_PROFILES:
        if k.startswith(pid) and k.endswith(suffix):
            return k
    raise KeyError(
        f"No SDM profile for problem='{problem_id}', profile='{profile}'. "
        f"Available: {sorted(SDM_PROFILES.keys())}"
    )


# ===================================================================
#  Method -> SchedulerConfig mapping
# ===================================================================

# Available methods for the scheduling experiment
METHODS = [
    "AdaIS",  # calibrated adaptive interaction scheduling
    "AdaS-Full",  # complete adaptive scheduling
    "Baseline-F2",  # fixed-interval (every 2 ticks), random channel
    "Baseline-F",  # fixed-interval (every 4 ticks), random channel
    "Baseline-F8",  # fixed-interval (every 8 ticks), random channel
    "Baseline-F16",  # fixed-interval (every 16 ticks), random channel
    "Baseline-R",  # random trigger (p=0.25/tick), random channel
    "Ablation-P",  # disable I_s (process stagnation signal)
    "Ablation-O",  # disable I_h and I_d (observation signals)
    "AdaS-UniCh",  # uniform-random channel selection
    "AdaS-RandQ",  # random question selection
    "AdaS-GreedyQ",  # greedy (lambda_cost=0)
    "AdaS-NoQ",  # no quality monitoring
    "AdaS-FixedT8",  # NoT: fixed 8-tick timing, valuation and quality monitoring kept
    "AdaS-RandomT",  # Random-T: count-matched pre-registered random timing, AdaIS channel rule and quality handling
    "AdaS-NoQpre",  # no pre-query quality pathway (gating, cooling, suspension); update weight kept
    "AdaS-NoQpost",  # pre-query quality pathway kept; update weight fixed at 1
    "VoI-Timing-MC",  # Bayesian value-rate timing with the AdaIS channel rule and quality handling
    "VoI-Timing-MC-AllCh",  # E9a: the timing posterior also absorbs ranking/threshold/ROI/NL answers
]


@dataclass
class MethodConfig:
    """Per-method scheduler configuration + flags."""

    scheduler_cfg: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Trigger overrides
    fixed_interval: int = 0  # >0 => trigger every N ticks (Baseline-F)
    random_trigger_p: float = 0.0  # >0 => per-tick random trigger prob (Baseline-R)
    disable_I_h: bool = False
    disable_I_d: bool = False
    disable_I_s: bool = False

    # Channel selection overrides
    uniform_channel: bool = False  # ignore phase priority, uniform random

    # Question selection overrides
    random_question: bool = False  # random question instead of IG-based

    # Quality monitoring
    disable_quality_monitoring: bool = False

    # Adaptive-only experimental controls
    bootstrap_channels: List[str] = field(default_factory=list)

    # Designs folded in from the derived runners (JCDE-R1, route P3)
    fixed_respects_quality_gate: bool = (
        False  # NoT: fixed timing still honours suspension
    )
    random_timing_schedule: bool = False  # Random-T: pre-registered tick schedule
    no_prequery: bool = False  # NoQpre: no channel restriction, cooling or suspension
    no_postweight: bool = False  # NoQpost: update weight fixed at 1
    voi_timing: bool = False  # VoI-Timing-MC
    voi_allch: bool = False  # E9a: VoI posterior updated from every channel, not only pairwise


def _build_method_config(method: str) -> MethodConfig:
    """Build method-specific configuration."""
    mc = MethodConfig()

    if method in {"AdaIS", "AdaS-Full"}:
        pass  # default = full adaptive scheduling

    elif method.startswith("Baseline-F"):
        if method == "Baseline-F":
            mc.fixed_interval = 4
        else:
            mc.fixed_interval = int(method.replace("Baseline-F", ""))
        mc.uniform_channel = True

    elif method == "Baseline-R":
        # 朴素"无调度"baseline：每轮都触发交互（用满 B_total 预算），通道完全均匀随机。
        # 这一配置用以严格对照"是否需要调度"——若 AdaS 在用户负担类指标上未显著优于 Random，
        # 则结构化调度的工程价值无法成立。p=1.0 让 N_int 严格等于 B_total，避免 p=0.25 时
        # 触发次数被截断而与 AdaS 接近的视觉干扰。
        mc.random_trigger_p = 1.0
        mc.uniform_channel = True

    elif method == "Ablation-P":
        mc.disable_I_s = True

    elif method == "Ablation-O":
        mc.disable_I_h = True
        mc.disable_I_d = True

    elif method == "AdaS-UniCh":
        mc.uniform_channel = True

    elif method == "AdaS-RandQ":
        mc.random_question = True

    elif method == "AdaS-GreedyQ":
        mc.scheduler_cfg.lambda_cost = 0.0

    elif method == "AdaS-NoQ":
        mc.disable_quality_monitoring = True

    elif method == "AdaS-FixedT8":
        mc.fixed_interval = 8
        mc.fixed_respects_quality_gate = True

    elif method == "AdaS-RandomT":
        mc.random_timing_schedule = True

    elif method == "AdaS-NoQpre":
        mc.no_prequery = True

    elif method == "AdaS-NoQpost":
        mc.no_postweight = True

    elif method == "VoI-Timing-MC":
        mc.voi_timing = True

    elif method == "VoI-Timing-MC-AllCh":
        mc.voi_timing = True
        mc.voi_allch = True

    else:
        raise ValueError(f"Unknown method: {method}. Choose from {METHODS}")

    return mc


# ===================================================================
#  Interaction execution helpers
# ===================================================================

ALL_CHANNELS = ["pairwise", "ranking", "threshold", "roi", "nl_clarify"]


def _select_channel(
    scheduler: InteractionScheduler,
    method_cfg: MethodConfig,
    phase: str,
    state_dict: Dict[str, Any],
    rng: np.random.Generator,
) -> str:
    """Select the interaction channel."""
    if method_cfg.uniform_channel:
        return rng.choice(ALL_CHANNELS)

    if method_cfg.bootstrap_channels:
        used = len(getattr(scheduler, "_interaction_log", []) or [])
        if used < len(method_cfg.bootstrap_channels):
            channel = method_cfg.bootstrap_channels[used]
            if channel in ALL_CHANNELS:
                return channel

    candidate_IGs = {}
    for ch in ALL_CHANNELS:
        candidate_IGs[ch] = scheduler.compute_IG_proxy(ch, state_dict)
    return scheduler.select_channel(phase, candidate_IGs)


def _generate_question(
    channel: str,
    qg: QuestionGenerator,
    state: OptState,
    upv: UPV,
    phase: str,
    method_cfg: MethodConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Generate a question for the selected channel."""
    pop_F = state.pop_F
    pop_G = state.pop_G

    if method_cfg.random_question:
        channel = rng.choice(ALL_CHANNELS)

    if channel == "pairwise":
        pairs = qg.select_pairwise(pop_F, upv, k_pairs=3)
        return {"channel": "pairwise", "pairs": pairs}

    elif channel == "ranking":
        candidates = qg.select_ranking(pop_F, pop_G, k=5)
        return {"channel": "ranking", "candidates": candidates}

    elif channel == "threshold":
        slots = _get_slot_confidences(upv)
        r_slots = sorted(k for k in slots if k.startswith("r_"))
        w_slots = sorted(k for k in slots if k.startswith("w_"))
        n_obj = len(r_slots)
        if not hasattr(_generate_question, "_thr_cycle"):
            _generate_question._thr_cycle = 0
        cyc = _generate_question._thr_cycle
        _generate_question._thr_cycle += 1
        # Phase 1: first n_obj calls cover every r_ slot exactly once
        if cyc < n_obj and r_slots:
            target = r_slots[cyc % n_obj]
        # Phase 2: then cover every w_ slot once
        elif cyc < 2 * n_obj and w_slots:
            target = w_slots[(cyc - n_obj) % n_obj]
        # Phase 3: pick lowest-confidence slot overall, cycling to break ties
        else:
            all_keys = r_slots + w_slots
            target = all_keys[cyc % len(all_keys)] if all_keys else "r_0"
        q = qg.generate_threshold_question(upv, target, pop_F)
        return {"channel": "threshold", "question": q}

    elif channel == "roi":
        q = qg.generate_roi_question(pop_F, phase)
        return {"channel": "roi", "question": q}

    elif channel == "nl_clarify":
        slots = _get_slot_confidences(upv)
        low_gamma = sorted(slots, key=slots.get)[:3]
        q = qg.generate_nl_clarify(upv, low_gamma)
        return {"channel": "nl_clarify", "question": q}

    return {"channel": channel}


def _get_slot_confidences(upv: UPV) -> Dict[str, float]:
    """Derive per-slot confidence proxy from UPV sigma."""
    sigma = getattr(upv, "sigma", {}) or {}
    sigma_w = float(sigma.get("w", 0.5))
    sigma_r = float(sigma.get("r", 0.5))
    n_obj = len(np.asarray(upv.w).ravel())
    slots = {}
    for i in range(n_obj):
        slots[f"w_{i}"] = 1.0 - sigma_w
        slots[f"r_{i}"] = 1.0 - sigma_r
    return slots


def _execute_interaction(
    channel: str,
    question: Dict[str, Any],
    sdm: SimulatedDM,
    state: OptState,
    upv: UPV,
    strength: float = 1.0,
    corrupt=None,
) -> Tuple[Dict[str, Any], float]:
    """Execute the interaction: SDM responds, patch is computed, UPV is updated.

    ``strength`` is the Eq. (3.15) update weight beta (P3 mode); 1.0 reproduces the submitted updates.
    ``corrupt`` (common-corruption matrix) is a callable that decides whether this response is corrupted.

    Returns (response_record, IG_actual).
    """
    pop_X = state.pop_X
    pop_F = state.pop_F
    n_obj = len(np.asarray(upv.w).ravel())
    record: Dict[str, Any] = {"channel": channel}
    ig_actual = 0.0

    if channel == "pairwise":
        pairs = question.get("pairs", [])
        results = []
        for i, j in pairs:
            if i < len(pop_F) and j < len(pop_F):
                choice = sdm.respond_pairwise(pop_X[i], pop_X[j], pop_F[i], pop_F[j])
                if corrupt is not None:
                    corrupted = corrupt()
                    choice = 1 - choice if corrupted else choice
                    results.append(
                        {"i": i, "j": j, "choice": choice, "corrupted": corrupted}
                    )
                else:
                    results.append({"i": i, "j": j, "choice": choice})
                # apply patch: adjust weights towards preferred solution
                preferred_F = pop_F[i] if choice == 0 else pop_F[j]
                _nudge_weights_from_pairwise(upv, preferred_F, n_obj, strength)
        record["results"] = results
        ig_actual = min(len(results) * 0.15, 0.6)

    elif channel == "ranking":
        candidates = question.get("candidates", [])
        if candidates:
            cand_F = pop_F[candidates]
            ranking = sdm.respond_ranking(cand_F, k=min(5, len(candidates)))
            if corrupt is not None:
                record["corrupted"] = corrupt()
                ranking = (
                    list(reversed(ranking))
                    if record["corrupted"] and ranking
                    else ranking
                )
            record["ranking"] = ranking
            # adjust weights based on top-ranked solution
            if ranking:
                best_local_idx = ranking[0]
                if best_local_idx < len(cand_F):
                    _nudge_weights_from_pairwise(
                        upv, cand_F[best_local_idx], n_obj, strength
                    )
            ig_actual = 0.3

    elif channel == "threshold":
        q = question.get("question", {})
        slot = q.get("slot", "r_0")
        proposed = q.get("proposed", 0.0)
        accepted, value = sdm.respond_threshold(slot, proposed)
        if corrupt is not None:
            record["corrupted"] = corrupt()
            if record["corrupted"]:
                direction = 1.0 if corrupt.rng.random() < 0.5 else -1.0
                value = float(value) + direction * (
                    abs(float(value)) + abs(float(proposed)) + 1.0
                )
                accepted = False
        record["slot"] = slot
        record["proposed"] = proposed
        record["accepted"] = accepted
        record["value"] = value
        _apply_slot_update(upv, slot, value, n_obj, strength)
        ig_actual = 0.4 if not accepted else 0.15

    elif channel == "roi":
        q = question.get("question", {})
        dims = q.get("projection_dims", (0, 1))
        bbox = sdm.respond_roi(pop_F, dims)
        record["bbox"] = bbox
        record["dims"] = dims
        # ROI informs search direction but does NOT directly update r —
        # r updates come from threshold/nl_clarify which query exact slot values.
        # The bbox information feeds back via IG computation and conflict scores.
        ig_actual = 0.25

    elif channel == "nl_clarify":
        q = question.get("question", {})
        target_slot = q.get("target_slot")
        if target_slot:
            resp = sdm.respond_nl_clarify(target_slot)
            if corrupt is not None and resp.get("text") is None:
                record["corrupted"] = corrupt()
                v = resp.get("value")
                if record["corrupted"] and v is not None and not isinstance(v, dict):
                    direction = 1.0 if corrupt.rng.random() < 0.5 else -1.0
                    resp = {
                        **resp,
                        "value": float(v) + direction * (abs(float(v)) + 1.0),
                    }
            record["response"] = resp
            value = resp.get("value")
            if resp.get("text") is not None:  # free text from the E10' relay respondent
                from nl_grounding import ground

                g = ground(resp["text"], target_slot)
                record["grounding"] = g
                if g["grounded"]:
                    _apply_slot_update(
                        upv, g["slot"], float(g["value"]), n_obj, strength
                    )
                    ig_actual = 0.5
                else:
                    ig_actual = 0.1
            elif value is not None and not isinstance(value, dict):
                _apply_slot_update(upv, target_slot, float(value), n_obj, strength)
                ig_actual = 0.5
            else:
                ig_actual = 0.1

    # update sigma (confidence) after interaction
    sigma = (
        dict(upv.sigma)
        if isinstance(upv.sigma, dict)
        else {"w": 0.5, "r": 0.5, "b": 0.5, "D": 0.5}
    )
    decay = 1.0 - strength * (1.0 - 0.85)  # 0.85 when strength == 1
    if channel in ("pairwise", "ranking"):
        sigma["w"] = max(0.05, sigma.get("w", 0.5) * decay)
    elif channel in ("threshold", "nl_clarify"):
        sigma["r"] = max(0.05, sigma.get("r", 0.5) * decay)
    elif channel == "roi":
        sigma["r"] = max(0.05, sigma.get("r", 0.5) * decay)
        sigma["w"] = max(
            0.05, sigma.get("w", 0.5) * (1.0 - strength * 0.10)
        )  # 0.90 when strength == 1
    upv.sigma = sigma

    return record, ig_actual * strength


def _nudge_weights_from_pairwise(
    upv: UPV, preferred_F: np.ndarray, n_obj: int, strength: float = 1.0
):
    """Slightly adjust UPV weights towards dimensions where preferred solution excels."""
    w = np.asarray(upv.w, dtype=float).copy()
    r = np.asarray(upv.r, dtype=float).copy()
    pf = np.asarray(preferred_F, dtype=float)
    n = min(len(w), len(pf), n_obj)

    # compute delta: dimensions where preferred is closer to r
    diff = scaled(np.abs(pf[:n] - r[:n]))  # objective scales (preference model v2); identity by default
    # increase weight where gap is small (solution is close to aspiration)
    diff_norm = diff / (np.max(diff) + 1e-12)
    adjustment = 0.02 * float(np.clip(strength, 0.0, 1.0)) * (1.0 - diff_norm)
    w[:n] = w[:n] + adjustment
    upv.w = normalize_w(w)


def _apply_slot_update(
    upv: UPV, slot: str, value: float, n_obj: int, strength: float = 1.0
):
    """Apply a single slot value update to UPV as a convex step of size ``strength`` (1.0 = overwrite)."""
    s = float(np.clip(strength, 0.0, 1.0))
    if slot.startswith("w_"):
        try:
            idx = int(slot.split("_")[1])
            w = np.asarray(upv.w, dtype=float).copy()
            if idx < len(w):
                w[idx] = (1.0 - s) * w[idx] + s * max(1e-6, float(value))
                upv.w = normalize_w(w)
        except (IndexError, ValueError):
            pass
    elif slot.startswith("r_"):
        try:
            idx = int(slot.split("_")[1])
            r = np.asarray(upv.r, dtype=float).copy()
            if idx < len(r):
                r[idx] = (1.0 - s) * r[idx] + s * float(value)
                upv.r = r
        except (IndexError, ValueError):
            pass


# ===================================================================
#  P3 ("as described") mechanisms - all off unless --p3 is given
# ===================================================================


def _select_channel_p3(scheduler, method_cfg, phase, state_dict, rng, args):
    """Eq. (3.10) net-value channel choice with the burden term active, warning filter and no-query gate.

    V(c) = G(c) - [lambda_min + lambda_B / (B_t + eps)] * kappa(c) + omega * priority(c); the channel with
    the largest V is asked only when V > 0 (Algorithm 1, step 10). The initial-coverage prefix (Table A.2)
    is used only while its next channel is admissible in the current quality state.
    Returns None when no query should be issued.
    """
    admissible = list(ALL_CHANNELS)
    if getattr(scheduler, "_abnormal_protection", False):
        admissible = ["threshold"]
    elif getattr(scheduler, "_response_level", "normal") == "warning":
        admissible = [c for c in admissible if c in {"threshold", "pairwise"}]
    if (
        method_cfg.uniform_channel
    ):  # Fixed-k / Random-C: all five channels; NoC: the quality-admissible set
        return str(
            rng.choice(
                ALL_CHANNELS if args.method.startswith("Baseline-") else admissible
            )
        )
    from hitlo.core.interaction_scheduler import _PHASE_CHANNEL_PRIORITY

    if method_cfg.bootstrap_channels:
        used = len(getattr(scheduler, "_interaction_log", []) or [])
        if (
            used < len(method_cfg.bootstrap_channels)
            and method_cfg.bootstrap_channels[used] in admissible
        ):
            return method_cfg.bootstrap_channels[used]
    key = phase if phase in _PHASE_CHANNEL_PRIORITY else "mid"
    order = [
        c for c in _PHASE_CHANNEL_PRIORITY.get(key, ALL_CHANNELS) if c in admissible
    ]
    order += [c for c in admissible if c not in order]
    lam = args.p3_lambda_min + args.p3_lambda_B / (
        float(scheduler.budget_remaining) + 1e-6
    )
    best, best_v = None, -float("inf")
    for rank, c in enumerate(order):
        v = (
            scheduler.compute_IG_proxy(c, state_dict)
            - lam * CHANNEL_COST.get(c, 0.3)
            + args.p3_omega / (1.0 + 0.5 * rank)
        )
        if v > best_v:
            best, best_v = c, v
    return best if best_v > 0 else None


def _quality_update_weight_p3(response_level, q, channel, phase, budget_remaining):
    """Table A.3 update weight, evaluated on the observable quality score q (not the simulator's latent quality)."""
    q = float(np.clip(q, 0.0, 1.0))
    reliable = channel in {"threshold", "nl_clarify"}
    if reliable and (phase == "late" or budget_remaining <= 2) and q >= 0.38:
        return 1.0 if q >= 0.48 else 0.85
    if reliable and budget_remaining <= 4 and q >= 0.42:
        return 1.0
    if response_level == "pause":
        return 0.0 if q < 0.36 else 0.55
    if q >= 0.55:
        return 1.0
    if q >= 0.48:
        return 0.88
    if response_level == "warning":
        return float(np.clip(0.35 + 0.95 * q, 0.25, 0.78))
    if q < 0.35:
        return float(np.clip(0.18 + 0.80 * q, 0.12, 0.52))
    return float(np.clip(0.48 + 0.95 * (q - 0.35), 0.42, 0.82))


def _wrap_nl_p3(sdm, rng, args):
    """P3 NL respondent for the simulated DM (the submitted simulator returned the exact ground truth).

    ``pipeline`` (default): reproduces what the E10' grounding pipeline delivers. A weight clarification is
    a qualitative importance statement ("drag matters more" / "equally" / "normal force matters more",
    from the sign of w*_i - 1/n with a 0.05 dead band) that grounds to 0.7 / 0.5 / 0.3; it fails to ground
    with probability ``p3_nl_fail_w`` and is read with the wrong direction with probability
    ``p3_nl_wrong_w``. A reference clarification states a number (truth plus threshold-level noise) and
    fails to ground with probability ``p3_nl_fail_r``. The three rates are estimated from the E10' agent
    transcripts.
    ``exact``: truth plus threshold-level noise, failing with probability ``p3_nl_fail_r`` (sensitivity).
    """
    oracle = sdm.respond_nl_clarify
    n_obj = len(np.asarray(sdm.w_star).ravel())

    def respond(slot_name):
        res = dict(oracle(slot_name))
        v = res.get("value")
        if v is None or isinstance(v, dict):
            return res
        if slot_name.startswith("w_") and args.p3_nl_model == "pipeline":
            if rng.random() < args.p3_nl_fail_w:
                return {**res, "value": None, "source": "ungrounded"}
            lean = float(v) - 1.0 / n_obj
            level = 0.5 if abs(lean) <= 0.05 else (0.7 if lean > 0 else 0.3)
            if level != 0.5 and rng.random() < args.p3_nl_wrong_w:
                level = round(1.0 - level, 2)  # importance read the wrong way round
            return {**res, "value": level, "source": "qualitative_importance"}
        if rng.random() < args.p3_nl_fail_r:
            return {**res, "value": None, "source": "ungrounded"}
        sd = args.p3_nl_noise * sdm.sigma_effective * 0.1 * (abs(float(v)) + 1e-6)
        res["value"] = float(v) + float(rng.normal(0.0, sd))
        return res

    sdm.respond_nl_clarify = respond


def _observable_revisions(rng, latent_quality):
    """Observable revision events for one response: n ~ Binomial(3, 1 - q). The monitor sees n, never q.

    Returned as the 'quality' field the scheduler stores, encoded so that int((1 - quality) * 3) == n.
    """
    n = int(rng.binomial(3, float(np.clip(1.0 - latent_quality, 0.0, 1.0))))
    return n, 1.0 - (n + 0.5) / 3.0


def _corruption_draw(rng, latent_quality):
    """Common-corruption response model (archived corruption runner): a response from a fatigued simulated DM
    is corrupted with probability clip(0.35 + 2 (0.75 - q), 0, 0.95) when its latent quality q < 0.75."""
    if latent_quality >= 0.75:
        return False
    return bool(
        rng.random() < float(np.clip(0.35 + (0.75 - latent_quality) * 2.00, 0.0, 0.95))
    )


class _RandomTiming:
    """Random-T: pre-registered tick schedule (count-matched to AdaIS); suspension, exhausted budget and a
    non-positive channel value defer the pending request to the next tick."""

    def __init__(self, path, scenario_key, T):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if str(payload.get("scenario_key", "")) != str(scenario_key):
            raise ValueError("Random-T scenario identity mismatch")
        ticks = [int(v) for v in payload.get("planned_ticks", [])]
        if ticks != sorted(set(ticks)) or any(v < 0 or v >= T for v in ticks):
            raise ValueError(
                "Random-T planned ticks must be sorted, unique and inside the run"
            )
        if int(payload.get("planned_K", len(ticks))) != len(ticks):
            raise ValueError("Random-T planned_K does not equal schedule length")
        self.planned = set(ticks)
        self.K = len(ticks)
        self.pending = []
        self.audit = []

    def trigger(self, tick, response_level, budget_remaining):
        if tick in self.planned:
            self.pending.append(
                {"planned_tick": tick, "deferred": False, "deferral_reason": ""}
            )
        if not self.pending:
            return None
        reason = (
            "quality_pause"
            if response_level == "pause"
            else "budget_exhausted"
            if budget_remaining <= 0
            else ""
        )
        if reason:
            for req in self.pending:
                req["deferred"] = True
                req["deferral_reason"] = req["deferral_reason"] or reason
            return None
        return self.pending.pop(0)

    def requeue(self, req, reason):
        req["deferred"] = True
        req["deferral_reason"] = req["deferral_reason"] or reason
        self.pending.insert(0, req)

    def realized(self, req, tick):
        self.audit.append(
            {
                **req,
                "realized_tick": tick,
                "pending_count": len(self.pending),
                "planned_K": self.K,
            }
        )

    def write(self, path):
        rows = self.audit + [
            {**req, "realized_tick": None, "deferred": True,
             "deferral_reason": req["deferral_reason"] or "window_exhaustion", "planned_K": self.K}
            for req in self.pending
        ]  # fmt: skip
        realized_K = sum(r["realized_tick"] is not None for r in rows)
        with open(path, "w", encoding="utf-8") as fh:
            for r in sorted(rows, key=lambda r: r["planned_tick"]):
                fh.write(
                    json.dumps(
                        {**r, "realized_K": realized_K}, ensure_ascii=False, default=str
                    )
                    + "\n"
                )


class _VoiTiming:
    """VoI-Timing-MC (archived runner): query when the best-pair expected value of information per unit cost
    exceeds the EWMA value of continuing; pairwise answers on the chosen archive pair update the particle
    posterior. The channel itself comes from the AdaIS channel rule."""

    def __init__(self, n_obj, seed):
        sys.path.insert(0, str(Path(__file__).resolve().parent / "derived"))
        import adas_voi_timing_core as core

        self.core = core
        self.C = core.CONSTANTS
        self.posterior = core.ParticlePosterior.from_prior(
            n_obj, seed=int(self.C["seed_namespaces"]["particle_prior"]) + int(seed)
        )
        self.tracker = core.ContinueValueTracker(window=int(self.C["ewma_window"]))
        self.records = []

    def _archive(self, state):
        empty = (
            np.empty((0, state.n_var)),
            np.empty((0, state.n_obj)),
            np.empty((0, state.n_obj)),
        )
        raw_F = getattr(state, "feas_archive_F", None)
        raw_X = getattr(state, "feas_archive_X", None)
        if (
            raw_F is None
            or raw_X is None
            or len(raw_F) == 0
            or len(raw_F) != len(raw_X)
        ):
            return empty
        raw_F, raw_X = np.asarray(raw_F, float), np.asarray(raw_X, float)
        sel_F = self.core.feasible_nondominated_archive(
            raw_F,
            feasible=np.ones(len(raw_F), dtype=bool),
            cap=int(self.C["archive_cap"]),
        )
        sel_X, used = [], set()
        for row in sel_F:
            match = next(
                int(i)
                for i in np.flatnonzero(np.all(raw_F == row, axis=1))
                if int(i) not in used
            )
            used.add(match)
            sel_X.append(raw_X[match])
        norm_F, _, _ = self.core.normalize_archive(sel_F)
        return np.asarray(sel_X, float), sel_F, norm_F

    def decide(self, tick, state, budget_remaining, response_level):
        warmup = int(self.C["warmup_ticks"])
        ctx = {"pair": None, "X": None, "F": None, "N": None}
        query_value = continue_value = 0.0
        if tick >= warmup:  # lazy warm-up: no archive work before a query is possible
            ctx["X"], ctx["F"], ctx["N"] = self._archive(state)
            if len(ctx["N"]) > 0:
                continue_value = self.tracker.observe(
                    self.posterior.expected_recommendation_utility(ctx["N"])
                )
            if len(ctx["N"]) >= 2:
                ctx["pair"], query_value = self.core.select_best_pair(
                    self.posterior, ctx["N"]
                )
        decision = self.core.should_query(
            query_value, continue_value, tick=tick, warmup_ticks=warmup, budget_remaining=budget_remaining,
            paused=(response_level == "pause"), already_queried=False,
            query_cost=float(self.C["query_cost"]), continuation_cost=float(self.C["continuation_cost"]),
        )  # fmt: skip
        rec = self.core.make_decision_record(
            tick=tick, decision=decision, query_value=query_value, continue_value=continue_value,
            selected_pair=ctx["pair"], posterior=self.posterior,
            archive_size=0 if ctx["N"] is None else len(ctx["N"]),
        )  # fmt: skip
        rec.update(
            {
                "response_level": response_level,
                "selected_channel": None,
                "voi_posterior_updated": False,
            }
        )
        self.records.append(rec)
        ctx["record"] = rec
        return bool(decision.query and ctx["pair"] is not None), ctx

    @staticmethod
    def pair_question(ctx):
        i, j = ctx["pair"]
        return {"channel": "pairwise", "pairs": [ctx["pair"]],
                "archive_objectives": [ctx["F"][i].tolist(), ctx["F"][j].tolist()]}  # fmt: skip

    def ask_pair(self, sdm, upv, ctx, strength, corrupt):
        """One real archive pair; the timing posterior and the host UPV are both updated from the answer."""
        i, j = ctx["pair"]
        choice = int(
            sdm.respond_pairwise(ctx["X"][i], ctx["X"][j], ctx["F"][i], ctx["F"][j])
        )
        corrupted = bool(corrupt()) if corrupt else False
        if corrupted:
            choice = 1 - choice
        self.posterior.update(ctx["N"][i], ctx["N"][j], preferred=choice)
        _nudge_weights_from_pairwise(upv, ctx["F"][i] if choice == 0 else ctx["F"][j],
                                     len(np.asarray(upv.w).ravel()), strength)  # fmt: skip
        sigma = (
            dict(upv.sigma)
            if isinstance(upv.sigma, dict)
            else {"w": 0.5, "r": 0.5, "b": 0.5, "D": 0.5}
        )
        sigma["w"] = max(0.05, sigma.get("w", 0.5) * (1.0 - strength * (1.0 - 0.85)))
        upv.sigma = sigma
        ctx["record"].update({"observed_preference": choice, "voi_posterior_updated": True,
                              "posterior_ess_after": float(self.posterior.effective_sample_size)})  # fmt: skip
        return {"channel": "pairwise", "results": [{"i": int(i), "j": int(j), "choice": choice,
                                                    "corrupted": corrupted}]}, 0.15 * strength  # fmt: skip

    # E9a likelihoods for non-pairwise answers (constants fixed before any E9a run)
    BOUND_EPS = 0.25  # a particle whose recommended design meets / misses a stated bound: 1 - eps / eps
    WEIGHT_SD = 0.15  # NL weight statement: Gaussian likelihood on the particle weight

    def absorb(self, channel, question, response, state, ctx):
        """E9a: update the timing posterior from a ranking, threshold, ROI or NL answer."""
        F_arch, N = ctx.get("F"), ctx.get("N")
        if N is None or len(N) == 0:
            return
        _, ideal, nadir = self.core.normalize_archive(F_arch)
        span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
        post = self.posterior
        updated = False
        if channel == "ranking" and response.get("ranking"):
            cand = np.asarray(state.pop_F, float)[question.get("candidates", [])]
            norm = (cand - ideal) / span
            order = [int(k) for k in response["ranking"] if int(k) < len(norm)]
            for a, b in zip(order[:-1], order[1:]):
                post.update(norm[a], norm[b], preferred=0)
                updated = True
        else:
            rec = F_arch[np.argmax(self.core._utility_matrix(N, post.particles, rho=post.rho), axis=1)]
            like = None
            slot, value = None, None
            if channel == "threshold":
                slot = response.get("slot")
                value = response.get("proposed") if response.get("accepted") else response.get("value")
            elif channel == "nl_clarify":
                r = response.get("response") or {}
                slot, value = r.get("slot"), r.get("value")
                if isinstance(value, dict) or value is None:
                    slot = None
            if channel == "roi" and response.get("bbox") is not None:
                f0a, f0b, f1a, f1b = response["bbox"]
                d0, d1 = response.get("dims", (0, 1))
                inside = (rec[:, d0] >= f0a) & (rec[:, d0] <= f0b) & (rec[:, d1] >= f1a) & (rec[:, d1] <= f1b)
                like = np.where(inside, 1.0 - self.BOUND_EPS, self.BOUND_EPS)
            elif slot and slot.startswith("r_"):
                i, v = int(slot.split("_")[1]), float(value)
                if i < rec.shape[1]:
                    like = np.where(rec[:, i] <= v + 0.15 * abs(v), 1.0 - self.BOUND_EPS, self.BOUND_EPS)
            elif slot and slot.startswith("w_"):
                i = int(slot.split("_")[1])
                if i < post.particles.shape[1]:
                    like = np.maximum(np.exp(-0.5 * ((post.particles[:, i] - float(value)) / self.WEIGHT_SD) ** 2), 1e-3)
            if like is not None:
                post.weights *= like
                post._normalize_weights()
                updated = True
        if updated:
            ctx["record"].update({"voi_posterior_updated": True, "posterior_update_channel": channel,
                                  "posterior_ess_after": float(post.effective_sample_size)})  # fmt: skip

    def write(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


# ===================================================================
#  Trigger decision with method overrides
# ===================================================================


def _should_trigger(
    tick: int,
    scheduler: InteractionScheduler,
    method_cfg: MethodConfig,
    upv_state: Dict[str, float],
    conflict_scores: Dict[str, float],
    convergence_status: Any,
    rng: np.random.Generator,
    feas_ratio: float = 1.0,
) -> bool:
    """Determine if an interaction should occur at this tick."""
    # Baseline-F: fixed interval
    if method_cfg.fixed_interval > 0:
        if scheduler.budget_remaining <= 0:
            return False
        if (
            method_cfg.fixed_respects_quality_gate
            and getattr(scheduler, "_response_level", "normal") == "pause"
        ):
            return False
        return (tick % method_cfg.fixed_interval == 0) and (tick > 0)

    # Baseline-R: random per-tick trigger
    if method_cfg.random_trigger_p > 0:
        if scheduler.budget_remaining <= 0:
            return False
        return rng.random() < method_cfg.random_trigger_p

    # Adaptive: use scheduler with signal overrides
    if tick < int(getattr(method_cfg.scheduler_cfg, "warmup_ticks", 0)):
        return False

    if method_cfg.disable_I_h:
        upv_state = {k: 1.0 for k in upv_state}  # fake high confidence
    if method_cfg.disable_I_d:
        conflict_scores = {k: 0.0 for k in conflict_scores}
    if method_cfg.disable_I_s:
        # create a dummy status with no stagnation
        class _NoStag:
            consec_stagnant_ticks = 0

        convergence_status = _NoStag()

    return scheduler.should_trigger(
        upv_state,
        conflict_scores,
        convergence_status,
        feas_ratio=feas_ratio,
    )


# ===================================================================
#  Main experiment loop
# ===================================================================


def run_experiment(args: argparse.Namespace) -> Path:
    """Run a single scheduling experiment and write outputs."""
    t_start = time.time()
    if not getattr(args, "no_seed_globals", False):
        # JCDE-R1: pymoo's NSGA-III niching breaks ties with numpy's *global* RNG, which was never
        # seeded, so runs with identical inputs could diverge. Seed global RNGs from the run seed.
        import random as _random

        np.random.seed(int(args.seed))
        _random.seed(int(args.seed))

    # ---- setup output directory ----
    out_dir = (
        Path(args.output_dir)
        / f"{args.method}_{args.problem}_{args.profile}_s{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- problem & OptState ----
    problem_id = args.problem.upper()
    if args.pref_model == "v2":
        import pref_model_v2

        pref_model_v2.activate(problem_id)
    n_var = _default_nvar(problem_id)

    state = OptState(problem_id=problem_id, n_var=n_var)
    state.seed = args.seed
    state.rng = np.random.default_rng(args.seed)

    exp_cfg = ExperimentConfig(
        exp_group="C1_PPCS_IQL_FULL",
        horizon_T=args.T,
        epoch_budget_B=50,
        seed_base=args.seed,
        run_mode="eval",
    )
    # load IQL checkpoint if available
    ckpt_path = _find_ckpt(problem_id)
    if ckpt_path:
        exp_cfg.iql_ckpt_path = ckpt_path
    state.experiment_config = exp_cfg
    state.enable_cr = True
    state.init_population()

    # ---- SDM ----
    profile_key = args.sdm_profile_key or _resolve_sdm_profile(problem_id, args.profile)
    if args.dm == "relay":
        from llm_relay_dm import make_relay_dm

        sdm = make_relay_dm(
            profile_key,
            state.problem,
            relay_dir=args.relay_dir,
            problem_id=problem_id,
            sigma_noise=args.sigma_noise,
            fatigue_rate=args.fatigue_rate,
            rng_seed=args.seed + 10000,
        )
    else:
        if profile_key not in SDM_PROFILES:
            import llm_relay_dm  # noqa: F401  registers the E10' persona profile (SAV-AERO_mid)
        sdm = SimulatedDM.from_profile(
            profile_key,
            state.problem,
            sigma_noise=args.sigma_noise,
            fatigue_rate=args.fatigue_rate,
            rng_seed=args.seed + 10000,
        )

    if args.p3 and args.dm != "relay":
        _wrap_nl_p3(sdm, np.random.default_rng(args.seed + 60000), args)

    # set initial UPV to a neutral starting point (NOT the ground truth)
    # so the scheduler has room to learn
    n_obj = state.n_obj
    state.upv.w = normalize_w(np.ones(n_obj))
    state.upv.r = np.zeros(n_obj)
    # Start with high uncertainty — scheduler must earn confidence via interactions
    state.upv.sigma = {"w": 0.85, "r": 0.85, "b": 0.7, "D": 0.7}

    # ---- method config ----
    method_cfg = _build_method_config(args.method)

    # apply CLI overrides to scheduler config
    if getattr(args, "tau_h", None) is not None:
        method_cfg.scheduler_cfg.tau_h = args.tau_h
    if getattr(args, "tau_d", None) is not None:
        method_cfg.scheduler_cfg.tau_d = args.tau_d
    if getattr(args, "T_stag", None) is not None:
        method_cfg.scheduler_cfg.T_stag = args.T_stag
    if getattr(args, "lambda_cost", None) is not None:
        method_cfg.scheduler_cfg.lambda_cost = args.lambda_cost
    if getattr(args, "B_total", None) is not None:
        method_cfg.scheduler_cfg.B_total = args.B_total
    if getattr(args, "B_reserve", None) is not None:
        method_cfg.scheduler_cfg.B_reserve = args.B_reserve
    if getattr(args, "T_min", None) is not None:
        method_cfg.scheduler_cfg.T_min = args.T_min
    if getattr(args, "T_max", None) is not None:
        method_cfg.scheduler_cfg.T_max = args.T_max
    if getattr(args, "kappa", None) is not None:
        method_cfg.scheduler_cfg.kappa = args.kappa
    if getattr(args, "tau_Q_high", None) is not None:
        method_cfg.scheduler_cfg.tau_Q_high = args.tau_Q_high
    if getattr(args, "tau_Q_low", None) is not None:
        method_cfg.scheduler_cfg.tau_Q_low = args.tau_Q_low
    if getattr(args, "warmup_ticks", None) is not None:
        method_cfg.scheduler_cfg.warmup_ticks = args.warmup_ticks
    if getattr(args, "bootstrap_channels", None):
        method_cfg.bootstrap_channels = [
            ch.strip() for ch in str(args.bootstrap_channels).split(",") if ch.strip()
        ]

    # ---- scheduler ----
    scheduler = InteractionScheduler(method_cfg.scheduler_cfg)
    qg = QuestionGenerator(rng_seed=args.seed + 20000)
    exp_rng = np.random.default_rng(args.seed + 30000)
    rev_rng = np.random.default_rng(args.seed + 50000)  # P3 observable revision events
    corruption_rng = np.random.default_rng(args.seed + 40000)  # common-corruption draws
    if args.corrupt_feedback and not args.p3:
        raise ValueError("--corrupt_feedback is implemented for --p3 runs only")
    random_timing = voi_timing = None
    if method_cfg.random_timing_schedule:
        random_timing = _RandomTiming(
            args.timing_schedule_path, args.timing_scenario_key, args.T
        )
    if method_cfg.voi_timing:
        voi_timing = _VoiTiming(state.n_obj, args.seed)
    if args.p3:
        os.environ.setdefault(
            "JCDE_QT_LATENCY", "excess_channel"
        )  # Eq. (3.11) + per-channel baseline
    _generate_question._thr_cycle = 0  # reset per experiment

    # ---- metrics storage ----
    csv_path = out_dir / "metrics.csv"
    jsonl_path = out_dir / "interaction_log.jsonl"

    csv_fields = [
        "tick",
        "eval_count",
        "hv",
        "hv_feas",
        "igd",
        "hv_delta",
        "feas_ratio",
        "mean_violation",
        "diversity",
        "sat",
        "pref_gap",
        "conflict",
        "confidence",
        "Q_t",
        "response_level",
        "triggered",
        "channel_type",
        "IG_value",
        "cognitive_cost",
        "conflict_resolved",
        "gamma_improved",
        "effective_interaction",
        "cooldown_remaining",
        "cooldown_set",
        "budget_remaining",
        "interaction_count",
        "phase",
        "cumulative_cognitive_cost",
        "dm_response_time",
        "dm_quality",
        "w_dist",
        "r_dist",
        "wall_time",
    ] + TRUTH_FIELDS

    csv_rows: List[Dict[str, Any]] = []
    interaction_records: List[Dict[str, Any]] = []
    total_interactions = 0
    cumulative_cognitive_cost = 0.0
    prev_hv = 0.0  # for hv_delta computation

    # ---- progress display ----
    T = args.T
    bar_len = 40

    # ---- scheduler-maintained state (isolated from pref-GP overwrites) ----
    _sched_sigma = (
        dict(state.upv.sigma)
        if isinstance(state.upv.sigma, dict)
        else {
            "w": 0.85,
            "r": 0.85,
            "b": 0.7,
            "D": 0.7,
        }
    )
    _sched_r = np.asarray(state.upv.r, dtype=float).copy()

    # ---- main loop ----
    for tick in range(T):
        # -- (a) execute one optimisation step --
        try:
            state.step_opt()
        except Exception as e:
            sys.stderr.write(f"[tick {tick}] step_opt error: {e}\n")
            # record error tick and continue
            csv_rows.append({f: 0.0 for f in csv_fields})
            csv_rows[-1]["tick"] = tick
            continue

        # Restore scheduler-maintained state (step_opt's pref-GP may overwrite sigma and r)
        state.upv.sigma = dict(_sched_sigma)
        state.upv.r = _sched_r.copy()

        # -- extract current metrics --
        hv = float(state.history_hv[-1]) if state.history_hv else 0.0
        hv_feas = float(state.history_hv_feas[-1]) if state.history_hv_feas else 0.0
        igd = float(state.history_igd[-1]) if state.history_igd else 0.0
        feas_ratio = (
            float(state.history_feasible[-1]) if state.history_feasible else 0.0
        )
        mean_viol = float(state.history_viol[-1]) if state.history_viol else 0.0
        diversity = (
            float(state.history_diversity[-1]) if state.history_diversity else 0.0
        )
        sat = float(state.history_sat[-1]) if state.history_sat else 0.0
        pref_gap = float(state.history_pref_gap[-1]) if state.history_pref_gap else 1.0

        # -- compute scheduler signals --
        conflict = compute_conflict_score(state)
        confidence = compute_confidence_score(state)
        upv_state = _get_slot_confidences(state.upv)
        conflict_scores = {"queue": conflict, "drift": conflict}
        conv_status = state.convergence_status

        # delta_hv for phase detection
        delta_hv = (
            float(state.history_dhv_smooth[-1]) if state.history_dhv_smooth else 0.0
        )
        phase = scheduler.detect_phase(feas_ratio, abs(delta_hv))

        # fatigue monitoring
        Q_t = scheduler.compute_Q_t(scheduler._interaction_log)
        if not method_cfg.disable_quality_monitoring:
            response_level = scheduler.get_response_level(Q_t)
            if method_cfg.no_prequery:
                scheduler._response_level = (
                    "normal"  # gating off; the update weight still sees response_level
                )
        else:
            response_level = "normal"
            scheduler._response_level = "normal"

        # -- (b) trigger decision --
        timing_request = voi_ctx = None
        if random_timing is not None:
            timing_request = random_timing.trigger(
                tick, response_level, scheduler.budget_remaining
            )
            triggered = timing_request is not None
        elif voi_timing is not None:
            triggered, voi_ctx = voi_timing.decide(
                tick, state, scheduler.budget_remaining, response_level
            )
        else:
            triggered = _should_trigger(
                tick,
                scheduler,
                method_cfg,
                upv_state,
                conflict_scores,
                conv_status,
                exp_rng,
                feas_ratio=feas_ratio,
            )

        channel_type = ""
        ig_value = 0.0
        cognitive_cost = 0.0
        cooldown_set = 0
        conflict_resolved = 0
        gamma_improved = 0
        effective_interaction = 0
        dm_response_time = 0.0  # ADM 响应时延（秒），仅在 triggered 时记录
        dm_quality = 0.0  # ADM 反馈质量 [0,1]，受疲劳传导（Boksem 2005, Hart 1988）

        # snapshot pre-interaction state for delta tracking
        gamma_before = dict(upv_state)
        conflict_before = conflict

        if triggered:
            # -- (c) generate and execute interaction --
            _sigma = state.upv.sigma if isinstance(state.upv.sigma, dict) else {}
            state_dict = {
                "gamma_slots": upv_state,
                "conflict_score": conflict,
                "phase": phase,
                "feas_ratio": feas_ratio,
                "sigma_r": float(_sigma.get("r", 0.5)),
            }

            if args.p3:
                channel_type = _select_channel_p3(
                    scheduler, method_cfg, phase, state_dict, exp_rng, args
                )
            else:
                channel_type = _select_channel(
                    scheduler,
                    method_cfg,
                    phase,
                    state_dict,
                    exp_rng,
                )
            if channel_type is None:  # P3 no-query gate: best net value <= 0
                triggered = False
                channel_type = ""
                if timing_request is not None:
                    random_timing.requeue(timing_request, "no_positive_value")
            if voi_ctx is not None:
                voi_ctx["record"]["selected_channel"] = channel_type or None
        if triggered:
            question = _generate_question(
                channel_type,
                qg,
                state,
                state.upv,
                phase,
                method_cfg,
                exp_rng,
            )
            if voi_ctx is not None and channel_type == "pairwise":
                question = _VoiTiming.pair_question(voi_ctx)

            # record weight snapshot before interaction for flip detection
            scheduler.record_weight_snapshot(state.upv.w)

            update_weight = 1.0
            if args.p3:
                dm_response_time = sdm.simulate_latency(channel_type)
                dm_quality = sdm.simulate_quality(channel_type)
                n_rev_obs, quality_obs = _observable_revisions(rev_rng, dm_quality)
                monitored = not (
                    method_cfg.disable_quality_monitoring
                    or method_cfg.no_postweight
                    or args.method.startswith("Baseline-")
                )  # Fixed-k and Random-C have no quality monitor; NoQ / NoQpost keep beta = 1
                if monitored:
                    probe = list(scheduler._interaction_log) + [
                        {
                            "channel": channel_type,
                            "latency": dm_response_time,
                            "quality": quality_obs,
                        }
                    ]
                    q_obs = float(copy.copy(scheduler).compute_Q_t(probe))
                    update_weight = _quality_update_weight_p3(
                        response_level,
                        q_obs,
                        channel_type,
                        phase,
                        scheduler.budget_remaining,
                    )
            corrupt = None
            if args.corrupt_feedback:
                q_latent = dm_quality

                def corrupt():
                    return _corruption_draw(corruption_rng, q_latent)

                corrupt.rng = corruption_rng
            if voi_ctx is not None and channel_type == "pairwise":
                response, ig_value = voi_timing.ask_pair(
                    sdm, state.upv, voi_ctx, update_weight, corrupt
                )
            else:
                response, ig_value = _execute_interaction(
                    channel_type,
                    question,
                    sdm,
                    state,
                    state.upv,
                    strength=update_weight,
                    corrupt=corrupt,
                )
                if voi_ctx is not None and method_cfg.voi_allch:
                    voi_timing.absorb(channel_type, question, response, state, voi_ctx)
            if timing_request is not None:
                random_timing.realized(timing_request, tick)
            truth_label = label_response(
                channel_type, question, response, sdm, state.pop_F
            )

            total_interactions += 1
            sdm.update_fatigue(total_interactions)

            # cognitive cost from CHANNEL_COST
            cognitive_cost = CHANNEL_COST.get(channel_type, 0.3)
            cumulative_cognitive_cost += cognitive_cost

            # sync scheduler state after interaction updated upv.sigma and upv.r
            _sched_sigma = (
                dict(state.upv.sigma)
                if isinstance(state.upv.sigma, dict)
                else _sched_sigma
            )
            _sched_r = np.asarray(state.upv.r, dtype=float).copy()

            # compute post-interaction signals for delta tracking
            gamma_after = _get_slot_confidences(state.upv)
            conflict_after = compute_conflict_score(state)

            # conflict_resolved: conflict dropped > 0.1
            if (conflict_before - conflict_after) > 0.1:
                conflict_resolved = 1

            # gamma_improved: any key slot gamma increased > 0.1
            for k in gamma_before:
                if k in gamma_after and (gamma_after[k] - gamma_before[k]) > 0.1:
                    gamma_improved = 1
                    break

            # compute hv_delta for effective_interaction check
            hv_delta = hv - prev_hv

            # effective_interaction: any of hv_delta>0, gamma_improved, conflict_resolved
            if hv_delta > 0 or gamma_improved or conflict_resolved:
                effective_interaction = 1

            # record interaction event to event stream
            state.event_stream.push(
                channel=channel_type,
                target="scheduler",
                content=response,
                context={
                    "tick": tick,
                    "hv": hv,
                    "feas_ratio": feas_ratio,
                    "conflict": conflict,
                    "confidence": confidence,
                    "phase": phase,
                },
                meta={
                    "IG": ig_value,
                    "Q_t": Q_t,
                    "interaction_count": total_interactions,
                },
            )

            # update scheduler state —— 用带疲劳传导的 latency/quality 代替常量
            if not args.p3:
                dm_response_time = sdm.simulate_latency(channel_type)
                dm_quality = sdm.simulate_quality(channel_type)
            scheduler.record_interaction(
                channel=channel_type,
                IG_actual=ig_value,
                latency=dm_response_time,
                quality=quality_obs if args.p3 else dm_quality,
            )
            cooldown_set = scheduler.cooldown_remaining

            # record detailed interaction (expanded fields)
            interaction_records.append(
                {
                    "tick": tick,
                    "channel": channel_type,
                    "question": _safe_serialize(question),
                    "response": _safe_serialize(response),
                    "IG": ig_value,
                    "phase": phase,
                    "conflict": conflict,
                    "confidence": confidence,
                    "Q_t": Q_t,
                    "budget_remaining": scheduler.budget_remaining,
                    "cognitive_cost": cognitive_cost,
                    "cooldown_set": cooldown_set,
                    "gamma_before": {k: round(v, 4) for k, v in gamma_before.items()},
                    "gamma_after": {k: round(v, 4) for k, v in gamma_after.items()},
                    "conflict_before": round(conflict_before, 4),
                    "conflict_after": round(conflict_after, 4),
                    "hv_before": round(hv, 6),
                    "hv_after_3tick": None,  # to be backfilled
                    "dm_latency": round(dm_response_time, 4),
                    "dm_quality": round(dm_quality, 4),
                    "truth": truth_label,
                    "update_weight": round(update_weight, 4),
                    "n_rev_observed": n_rev_obs if args.p3 else None,
                }
            )

            state.add_log(
                "SCHED",
                f"tick={tick} ch={channel_type} IG={ig_value:.3f} "
                f"budget_left={scheduler.budget_remaining}",
            )
        else:
            # tick cooldown
            scheduler.tick_cooldown()

        # -- compute hv_delta --
        hv_delta = hv - prev_hv
        prev_hv = hv

        # -- (e) compute distance to ground truth --
        w_dist = float(np.linalg.norm(np.asarray(state.upv.w) - sdm.w_star))
        r_dist = float(np.linalg.norm(np.asarray(state.upv.r) - sdm.r_star))

        # -- record row --
        csv_rows.append(
            {
                "tick": tick,
                "eval_count": state.eval_count,
                "hv": round(hv, 6),
                "hv_feas": round(hv_feas, 6),
                "igd": round(igd, 6),
                "hv_delta": round(hv_delta, 6),
                "feas_ratio": round(feas_ratio, 4),
                "mean_violation": round(mean_viol, 6),
                "diversity": round(diversity, 6),
                "sat": round(sat, 6),
                "pref_gap": round(pref_gap, 6),
                "conflict": round(conflict, 4),
                "confidence": round(confidence, 4),
                "Q_t": round(Q_t, 4),
                "response_level": response_level,
                "triggered": int(triggered),
                "channel_type": channel_type,
                "IG_value": round(ig_value, 4),
                "cognitive_cost": round(cognitive_cost, 4),
                "conflict_resolved": conflict_resolved,
                "gamma_improved": gamma_improved,
                "effective_interaction": effective_interaction,
                "cooldown_remaining": scheduler.cooldown_remaining,
                "cooldown_set": cooldown_set,
                "budget_remaining": scheduler.budget_remaining,
                "interaction_count": total_interactions,
                "phase": phase,
                "cumulative_cognitive_cost": round(cumulative_cognitive_cost, 4),
                "dm_response_time": round(dm_response_time, 4),
                "dm_quality": round(dm_quality, 4),
                "w_dist": round(w_dist, 6),
                "r_dist": round(r_dist, 6),
                "wall_time": round(time.time() - t_start, 2),
                **truth_row(state, sdm),
            }
        )

        # -- progress bar --
        pct = (tick + 1) / T
        filled = int(bar_len * pct)
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stderr.write(
            f"\r[{bar}] {pct * 100:5.1f}% | tick {tick + 1}/{T} "
            f"| HV={hv:.4f} feas={feas_ratio:.2f} "
            f"| int={total_interactions}/{method_cfg.scheduler_cfg.B_total}"
        )
        sys.stderr.flush()

    sys.stderr.write("\n")

    # ---- backfill hv_after_3tick in interaction_log ----
    for rec in interaction_records:
        rec_tick = rec.get("tick", -1)
        target_tick = rec_tick + 3
        if target_tick < len(csv_rows):
            rec["hv_after_3tick"] = csv_rows[target_tick].get("hv", None)
        else:
            # use final tick if not enough ticks remain
            rec["hv_after_3tick"] = csv_rows[-1].get("hv", None) if csv_rows else None

    # ---- write metrics.csv ----
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    # ---- write interaction_log.jsonl ----
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in interaction_records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # ---- save final population (pop_F, pop_G) for Pareto-front plotting & PHI calc ----
    try:
        pop_F_final = np.asarray(state.pop_F, dtype=float)
        np.save(out_dir / "pop_F_final.npy", pop_F_final)
        if hasattr(state, "pop_G") and state.pop_G is not None:
            pop_G_final = np.asarray(state.pop_G, dtype=float)
            np.save(out_dir / "pop_G_final.npy", pop_G_final)
    except Exception as _e:
        print(f"[WARN] pop_F save failed: {_e}")

    if random_timing is not None:
        random_timing.write(args.timing_audit_path or out_dir / "timing_audit.jsonl")
    if voi_timing is not None:
        voi_timing.write(out_dir / "voi_decision_log.jsonl")

    if getattr(sdm, "transcript", None) is not None:  # E10' relay respondent
        with open(out_dir / "relay_transcript.json", "w", encoding="utf-8") as f:
            json.dump(sdm.transcript, f, ensure_ascii=False, indent=1, default=str)

    # ---- summary ----
    elapsed = time.time() - t_start
    final_hv = csv_rows[-1]["hv"] if csv_rows else 0.0
    final_feas = csv_rows[-1]["feas_ratio"] if csv_rows else 0.0
    print(
        f"\nDone: method={args.method} problem={args.problem} "
        f"seed={args.seed} profile={args.profile}"
    )
    print(
        f"  T={T} ticks, {total_interactions} interactions, "
        f"final HV={final_hv:.4f}, feas={final_feas:.2f}, "
        f"w_dist={csv_rows[-1].get('w_dist', 0):.4f}, "
        f"time={elapsed:.1f}s"
    )
    print(f"  Output: {out_dir}")

    return out_dir


# ===================================================================
#  Helpers
# ===================================================================


def _find_ckpt(problem_id: str) -> str:
    """Try to locate a pretrained IQL checkpoint for the problem."""
    candidates = [
        f"ckpts/{problem_id}.pt",
        f"ckpts/{problem_id.lower()}.pt",
        f"ckpts/{problem_id.upper()}.pt",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return ""


def _safe_serialize(obj: Any) -> Any:
    """Make an object JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ===================================================================
#  CLI
# ===================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="§3.4 Adaptive Interaction Scheduling Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--method",
        type=str,
        default="AdaS-Full",
        choices=METHODS,
        help="Scheduling method variant",
    )
    p.add_argument(
        "--problem", type=str, default="ZDT1", help="Problem ID (e.g. ZDT1, BNH, DTLZ2)"
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument(
        "--profile",
        type=str,
        default="P-balanced",
        choices=["P-balanced", "P-edge"],
        help="SDM preference profile",
    )
    p.add_argument(
        "--fatigue_rate", type=float, default=0.0, help="SDM fatigue growth rate"
    )
    p.add_argument(
        "--sigma_noise", type=float, default=0.1, help="SDM base noise level"
    )
    p.add_argument("--T", type=int, default=80, help="Number of optimisation ticks")
    p.add_argument(
        "--output_dir",
        type=str,
        default="results/scheduling_exp",
        help="Root output directory",
    )

    # Scheduler parameter overrides (for parameter sweep experiments)
    p.add_argument("--tau_h", type=float, default=None)
    p.add_argument("--tau_d", type=float, default=None)
    p.add_argument("--T_stag", type=int, default=None)
    p.add_argument("--lambda_cost", type=float, default=None)
    p.add_argument("--B_total", type=int, default=None)
    p.add_argument("--B_reserve", type=int, default=None)
    p.add_argument("--T_min", type=float, default=None)
    p.add_argument("--T_max", type=float, default=None)
    p.add_argument("--kappa", type=float, default=None)
    p.add_argument("--tau_Q_high", type=float, default=None)
    p.add_argument("--tau_Q_low", type=float, default=None)
    p.add_argument("--warmup_ticks", type=int, default=None)
    p.add_argument(
        "--bootstrap_channels",
        type=str,
        default=None,
        help="Comma-separated adaptive-only channel sequence used before IG-based channel selection",
    )

    # JCDE revision options (defaults reproduce the submitted configuration)
    p.add_argument(
        "--dm",
        choices=["sdm", "relay"],
        default="sdm",
        help="Decision maker: simulated (sdm) or external agent via file relay (E10')",
    )
    p.add_argument(
        "--relay_dir", type=str, default=None, help="Relay folder for --dm relay"
    )
    p.add_argument(
        "--p3",
        action="store_true",
        help="P3: cost-gated net-value channel choice, beta-weighted updates on observable quality, "
        "observable revision events, excess per-channel latency",
    )
    p.add_argument(
        "--p3_lambda_min",
        type=float,
        default=0.0,
        help="Eq. (3.10) lambda_min (Table A.2)",
    )
    p.add_argument(
        "--p3_lambda_B", type=float, default=1.0, help="Eq. (3.10) lambda_B (Table A.2)"
    )
    p.add_argument(
        "--p3_omega",
        type=float,
        default=0.1,
        help="Eq. (3.10) omega_pi priority weight",
    )
    p.add_argument(
        "--p3_nl_model",
        choices=["pipeline", "exact"],
        default="pipeline",
        help="P3 simulated NL respondent",
    )
    p.add_argument(
        "--p3_nl_fail_w",
        type=float,
        default=0.0,
        help="P3 NL: weight statement fails to ground (E10')",
    )
    p.add_argument(
        "--p3_nl_wrong_w",
        type=float,
        default=0.0,
        help="P3 NL: weight statement read the wrong way (E10')",
    )
    p.add_argument(
        "--p3_nl_fail_r",
        type=float,
        default=0.0,
        help="P3 NL: reference statement fails to ground (E10')",
    )
    p.add_argument(
        "--p3_nl_noise",
        type=float,
        default=1.0,
        help="P3 simulated NL: noise relative to a threshold answer",
    )
    p.add_argument(
        "--corrupt_feedback",
        action="store_true",
        help="Common-corruption response model (P3 only)",
    )
    p.add_argument(
        "--timing_schedule_path",
        type=str,
        default=None,
        help="AdaS-RandomT: pre-registered schedule JSON",
    )
    p.add_argument(
        "--timing_scenario_key",
        type=str,
        default=None,
        help="AdaS-RandomT: scenario identity in the schedule",
    )
    p.add_argument(
        "--timing_audit_path",
        type=str,
        default=None,
        help="AdaS-RandomT: realization audit JSONL",
    )
    p.add_argument(
        "--no_seed_globals",
        action="store_true",
        help="Do not seed numpy/random global RNGs (reproduces the unseeded submitted runs)",
    )
    p.add_argument(
        "--sdm_profile_key",
        type=str,
        default=None,
        help="Explicit SDM_PROFILES key (overrides --profile), e.g. SAV-AERO_mid",
    )
    p.add_argument(
        "--pref_model",
        choices=["v1", "v2"],
        default="v1",
        help="v2: scaled ASF in host and SDM, monotone SDM utility, attainable SAV-AERO aspirations "
        "(pref_model_v2.py); v1 reproduces the submitted preference model",
    )

    return p.parse_args()


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
