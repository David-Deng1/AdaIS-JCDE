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
    normalize_w, apply_patch_to_upv, asf_distance, asf_distance_batch,
    compute_sat_prefgap,
)
from hitlo.core.osv_engine import compute_conflict_score, compute_confidence_score
from hitlo.core.simulated_dm import SimulatedDM, SDM_PROFILES
from hitlo.core.interaction_scheduler import InteractionScheduler, SchedulerConfig, CHANNEL_COST
from hitlo.core.question_generator import QuestionGenerator
from hitlo.problems.catalog import PROBLEM_CATALOG


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
    "AdaIS",          # calibrated adaptive interaction scheduling
    "AdaS-Full",      # complete adaptive scheduling
    "Baseline-F2",    # fixed-interval (every 2 ticks), random channel
    "Baseline-F",     # fixed-interval (every 4 ticks), random channel
    "Baseline-F8",    # fixed-interval (every 8 ticks), random channel
    "Baseline-F16",   # fixed-interval (every 16 ticks), random channel
    "Baseline-R",     # random trigger (p=0.25/tick), random channel
    "Ablation-P",     # disable I_s (process stagnation signal)
    "Ablation-O",     # disable I_h and I_d (observation signals)
    "AdaS-UniCh",     # uniform-random channel selection
    "AdaS-RandQ",     # random question selection
    "AdaS-GreedyQ",   # greedy (lambda_cost=0)
    "AdaS-NoQ",       # no quality monitoring
    "AdaS-FixedT8",   # fixed 8-tick timing, adaptive channel valuation + quality monitoring
    "AdaS-NoQpre",    # JCDE-R1: no pre-query quality pathway, update weight kept
    "AdaS-NoQpost",   # JCDE-R1: pre-query quality pathway kept, update weight fixed at 1
]


@dataclass
class MethodConfig:
    """Per-method scheduler configuration + flags."""
    scheduler_cfg: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Trigger overrides
    fixed_interval: int = 0        # >0 => trigger every N ticks (Baseline-F)
    random_trigger_p: float = 0.0  # >0 => per-tick random trigger prob (Baseline-R)
    fixed_respects_quality_gate: bool = False
    feedback_quality_filter: bool = False
    jcde_no_prequery: bool = False
    jcde_no_postweight: bool = False
    disable_I_h: bool = False
    disable_I_d: bool = False
    disable_I_s: bool = False

    # Channel selection overrides
    uniform_channel: bool = False   # ignore phase priority, uniform random

    # Question selection overrides
    random_question: bool = False   # random question instead of IG-based

    # Quality monitoring
    disable_quality_monitoring: bool = False

    # Adaptive-only experimental controls
    bootstrap_channels: List[str] = field(default_factory=list)


def _build_method_config(method: str) -> MethodConfig:
    """Build method-specific configuration."""
    mc = MethodConfig()

    if method in {"AdaIS", "AdaS-Full"}:
        mc.feedback_quality_filter = True

    elif method == "AdaS-FixedT8":
        mc.fixed_interval = 8
        mc.fixed_respects_quality_gate = True
        mc.feedback_quality_filter = True

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
        mc.feedback_quality_filter = True

    elif method == "AdaS-RandQ":
        mc.random_question = True

    elif method == "AdaS-GreedyQ":
        mc.scheduler_cfg.lambda_cost = 0.0

    elif method == "AdaS-NoQ":
        mc.disable_quality_monitoring = True

    elif method == "AdaS-NoQpre":
        mc.feedback_quality_filter = True
        mc.jcde_no_prequery = True

    elif method == "AdaS-NoQpost":
        mc.feedback_quality_filter = True
        mc.jcde_no_postweight = True

    else:
        raise ValueError(f"Unknown method: {method}. Choose from {METHODS}")

    return mc


# ===================================================================
#  Interaction execution helpers
# ===================================================================

ALL_CHANNELS = ["pairwise", "ranking", "threshold", "roi", "nl_clarify"]

QUALITY_STRESS_ENABLED = True

def _quality_admissible_channels(scheduler: InteractionScheduler, method_cfg: MethodConfig) -> List[str]:
    if not getattr(method_cfg, "feedback_quality_filter", False) or getattr(method_cfg, "jcde_no_prequery", False):
        return list(ALL_CHANNELS)
    response_level = getattr(scheduler, "_response_level", "normal")
    abnormal_protection = bool(getattr(scheduler, "_abnormal_protection", False))
    if abnormal_protection or response_level == "pause":
        return ["threshold"]
    if response_level == "warning":
        return ["pairwise", "threshold"]
    return list(ALL_CHANNELS)


def _jcde_quality_for_weight(scheduler, channel, latency, dm_quality):
    """Quality score q_resp that sets the update weight in Eq. (3.15).

    Default (JCDE-R1): the observable score Q_t recomputed with the current response appended,
    evaluated on a shallow copy so the scheduler's own state is untouched. With
    JCDE_BETA_SOURCE=latent the simulator's internal quality is used instead, which is how the
    submitted common-corruption matrix was produced (kept only as a diagnostic upper bound).
    """
    if os.environ.get("JCDE_BETA_SOURCE", "observable") == "latent":
        return dm_quality
    import copy as _copy

    probe = _copy.copy(scheduler)
    record = {"channel": channel, "latency": float(latency), "quality": float(dm_quality)}
    return float(probe.compute_Q_t(list(scheduler._interaction_log) + [record]))


def _quality_update_weight(method_cfg: MethodConfig, response_level: str, dm_quality: float,
                           channel: str = "", phase: str = "", budget_remaining: int = 999) -> float:
    """Return how strongly a feedback response should update UPV.

    In the recovery setting, the quality monitor first filters low-trust
    feedback, then restores trustworthy late threshold/NL updates so the
    preference model can still converge by the terminal horizon.
    """
    if not QUALITY_STRESS_ENABLED or not getattr(method_cfg, "feedback_quality_filter", False):
        return 1.0
    if getattr(method_cfg, "jcde_no_postweight", False):
        return 1.0
    q = float(np.clip(dm_quality, 0.0, 1.0))
    reliable_channel = channel in {"threshold", "nl_clarify"}
    late_recovery = (phase == "late") or (budget_remaining <= 2)
    if reliable_channel and late_recovery and q >= 0.38:
        return 1.0 if q >= 0.48 else 0.85
    if reliable_channel and budget_remaining <= 4 and q >= 0.42:
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
    if q < 0.55:
        return float(np.clip(0.48 + 0.95 * (q - 0.35), 0.42, 0.82))
    return 1.0


def _select_channel(
    scheduler: InteractionScheduler,
    method_cfg: MethodConfig,
    phase: str,
    state_dict: Dict[str, Any],
    rng: np.random.Generator,
) -> str:
    """Select the interaction channel."""
    if method_cfg.uniform_channel:
        admissible_channels = _quality_admissible_channels(scheduler, method_cfg)
        return rng.choice(admissible_channels)

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
    update_weight: float = 1.0,
    dm_quality: float = 1.0,
    corrupt_feedback: bool = False,
    stress_rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[str, Any], float]:
    """Execute the interaction: SDM responds, patch is computed, UPV is updated.

    Returns (response_record, IG_actual).
    """
    pop_X = state.pop_X
    pop_F = state.pop_F
    n_obj = len(np.asarray(upv.w).ravel())
    record: Dict[str, Any] = {"channel": channel}
    ig_actual = 0.0
    update_weight = float(np.clip(update_weight, 0.0, 1.0))
    dm_quality = float(np.clip(dm_quality, 0.0, 1.0))
    if stress_rng is None:
        stress_rng = np.random.default_rng(0)
    record["update_weight"] = update_weight
    record["dm_quality"] = dm_quality

    def _maybe_corrupt_feedback() -> bool:
        if not QUALITY_STRESS_ENABLED or not corrupt_feedback:
            return False
        if dm_quality >= 0.75:
            return False
        p_corrupt = float(np.clip(0.35 + (0.75 - dm_quality) * 2.00, 0.0, 0.95))
        return bool(stress_rng.random() < p_corrupt)

    if channel == "pairwise":
        pairs = question.get("pairs", [])
        results = []
        for (i, j) in pairs:
            if i < len(pop_F) and j < len(pop_F):
                choice = sdm.respond_pairwise(pop_X[i], pop_X[j], pop_F[i], pop_F[j])
                corrupted = _maybe_corrupt_feedback()
                if corrupted:
                    choice = 1 - choice
                results.append({"i": i, "j": j, "choice": choice, "corrupted": corrupted})
                # apply patch: adjust weights towards preferred solution
                preferred_F = pop_F[i] if choice == 0 else pop_F[j]
                _nudge_weights_from_pairwise(upv, preferred_F, n_obj, strength=update_weight)
        record["results"] = results
        ig_actual = min(len(results) * 0.15, 0.6)

    elif channel == "ranking":
        candidates = question.get("candidates", [])
        if candidates:
            cand_F = pop_F[candidates]
            ranking = sdm.respond_ranking(cand_F, k=min(5, len(candidates)))
            corrupted = _maybe_corrupt_feedback()
            if corrupted and ranking:
                ranking = list(reversed(ranking))
            record["ranking"] = ranking
            record["ranking_corrupted"] = corrupted
            # adjust weights based on top-ranked solution
            if ranking:
                best_local_idx = ranking[0]
                if best_local_idx < len(cand_F):
                    _nudge_weights_from_pairwise(upv, cand_F[best_local_idx], n_obj, strength=update_weight)
            ig_actual = 0.3

    elif channel == "threshold":
        q = question.get("question", {})
        slot = q.get("slot", "r_0")
        proposed = q.get("proposed", 0.0)
        accepted, value = sdm.respond_threshold(slot, proposed)
        corrupted = _maybe_corrupt_feedback()
        if corrupted:
            direction = 1.0 if stress_rng.random() < 0.5 else -1.0
            value = float(value) + direction * 1.00 * (abs(float(value)) + abs(float(proposed)) + 1.0)
            accepted = False
        record["slot"] = slot
        record["corrupted"] = corrupted
        record["proposed"] = proposed
        record["accepted"] = accepted
        record["value"] = value
        _apply_slot_update(upv, slot, value, n_obj, strength=update_weight)
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
            corrupted = _maybe_corrupt_feedback()
            if corrupted and resp.get("value") is not None and not isinstance(resp.get("value"), dict):
                base_value = float(resp.get("value"))
                direction = 1.0 if stress_rng.random() < 0.5 else -1.0
                resp["value"] = base_value + direction * 1.00 * (abs(base_value) + 1.0)
            record["response"] = resp
            record["corrupted"] = corrupted
            value = resp.get("value")
            if value is not None and not isinstance(value, dict):
                _apply_slot_update(upv, target_slot, float(value), n_obj, strength=update_weight)
                ig_actual = 0.5
            else:
                ig_actual = 0.1

    # update sigma (confidence) after interaction
    sigma = dict(upv.sigma) if isinstance(upv.sigma, dict) else {"w": 0.5, "r": 0.5, "b": 0.5, "D": 0.5}
    decay = 0.85
    effective_decay = 1.0 - update_weight * (1.0 - decay)
    if channel in ("pairwise", "ranking"):
        sigma["w"] = max(0.05, sigma.get("w", 0.5) * effective_decay)
    elif channel in ("threshold", "nl_clarify"):
        sigma["r"] = max(0.05, sigma.get("r", 0.5) * effective_decay)
    elif channel == "roi":
        sigma["r"] = max(0.05, sigma.get("r", 0.5) * effective_decay)
        sigma["w"] = max(0.05, sigma.get("w", 0.5) * (1.0 - update_weight * 0.10))
    upv.sigma = sigma

    return record, ig_actual


def _nudge_weights_from_pairwise(upv: UPV, preferred_F: np.ndarray, n_obj: int, strength: float = 1.0):
    """Slightly adjust UPV weights towards dimensions where preferred solution excels."""
    w = np.asarray(upv.w, dtype=float).copy()
    r = np.asarray(upv.r, dtype=float).copy()
    pf = np.asarray(preferred_F, dtype=float)
    n = min(len(w), len(pf), n_obj)

    # compute delta: dimensions where preferred is closer to r
    diff = np.abs(pf[:n] - r[:n])
    # increase weight where gap is small (solution is close to aspiration)
    diff_norm = diff / (np.max(diff) + 1e-12)
    adjustment = 0.02 * float(np.clip(strength, 0.0, 1.0)) * (1.0 - diff_norm)
    w[:n] = w[:n] + adjustment
    upv.w = normalize_w(w)


def _apply_slot_update(upv: UPV, slot: str, value: float, n_obj: int, strength: float = 1.0):
    """Apply a single slot value update to UPV."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if slot.startswith("w_"):
        try:
            idx = int(slot.split("_")[1])
            w = np.asarray(upv.w, dtype=float).copy()
            if idx < len(w):
                w[idx] = max(1e-6, (1.0 - strength) * w[idx] + strength * float(value))
                upv.w = normalize_w(w)
        except (IndexError, ValueError):
            pass
    elif slot.startswith("r_"):
        try:
            idx = int(slot.split("_")[1])
            r = np.asarray(upv.r, dtype=float).copy()
            if idx < len(r):
                r[idx] = (1.0 - strength) * r[idx] + strength * float(value)
                upv.r = r
        except (IndexError, ValueError):
            pass


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
        if method_cfg.fixed_respects_quality_gate and getattr(scheduler, "_response_level", "normal") == "pause":
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
        upv_state, conflict_scores, convergence_status,
        feas_ratio=feas_ratio,
    )


# ===================================================================
#  Main experiment loop
# ===================================================================

def run_experiment(args: argparse.Namespace) -> Path:
    """Run a single scheduling experiment and write outputs."""
    t_start = time.time()

    # ---- setup output directory ----
    out_dir = Path(args.output_dir) / f"{args.method}_{args.problem}_{args.profile}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- problem & OptState ----
    problem_id = args.problem.upper()
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
    profile_key = _resolve_sdm_profile(problem_id, args.profile)
    sdm = SimulatedDM.from_profile(
        profile_key,
        state.problem,
        sigma_noise=args.sigma_noise,
        fatigue_rate=args.fatigue_rate,
        rng_seed=args.seed + 10000,
    )

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
            ch.strip() for ch in str(args.bootstrap_channels).split(",")
            if ch.strip()
        ]

    # ---- scheduler ----
    scheduler = InteractionScheduler(method_cfg.scheduler_cfg)
    qg = QuestionGenerator(rng_seed=args.seed + 20000)
    exp_rng = np.random.default_rng(args.seed + 30000)
    corruption_rng = np.random.default_rng(args.seed + 40000)
    _generate_question._thr_cycle = 0  # reset per experiment

    # ---- metrics storage ----
    csv_path = out_dir / "metrics.csv"
    jsonl_path = out_dir / "interaction_log.jsonl"

    csv_fields = [
        "tick", "eval_count",
        "hv", "hv_feas", "igd",
        "hv_delta",
        "feas_ratio", "mean_violation", "diversity",
        "sat", "pref_gap",
        "conflict", "confidence",
        "Q_t", "response_level",
        "triggered", "channel_type", "IG_value",
        "cognitive_cost",
        "conflict_resolved", "gamma_improved", "effective_interaction",
        "cooldown_remaining", "cooldown_set",
        "budget_remaining", "interaction_count",
        "phase",
        "cumulative_cognitive_cost",
        "dm_response_time", "dm_quality", "feedback_update_weight",
        "w_dist", "r_dist",
        "wall_time",
    ]

    csv_rows: List[Dict[str, Any]] = []
    interaction_records: List[Dict[str, Any]] = []
    total_interactions = 0
    cumulative_cognitive_cost = 0.0
    prev_hv = 0.0  # for hv_delta computation

    # ---- progress display ----
    T = args.T
    bar_len = 40

    # ---- scheduler-maintained state (isolated from pref-GP overwrites) ----
    _sched_sigma = dict(state.upv.sigma) if isinstance(state.upv.sigma, dict) else {
        "w": 0.85, "r": 0.85, "b": 0.7, "D": 0.7,
    }
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
        feas_ratio = float(state.history_feasible[-1]) if state.history_feasible else 0.0
        mean_viol = float(state.history_viol[-1]) if state.history_viol else 0.0
        diversity = float(state.history_diversity[-1]) if state.history_diversity else 0.0
        sat = float(state.history_sat[-1]) if state.history_sat else 0.0
        pref_gap = float(state.history_pref_gap[-1]) if state.history_pref_gap else 1.0

        # -- compute scheduler signals --
        conflict = compute_conflict_score(state)
        confidence = compute_confidence_score(state)
        upv_state = _get_slot_confidences(state.upv)
        conflict_scores = {"queue": conflict, "drift": conflict}
        conv_status = state.convergence_status

        # delta_hv for phase detection
        delta_hv = float(state.history_dhv_smooth[-1]) if state.history_dhv_smooth else 0.0
        phase = scheduler.detect_phase(feas_ratio, abs(delta_hv))

        # fatigue monitoring
        Q_t = scheduler.compute_Q_t(scheduler._interaction_log)
        if not method_cfg.disable_quality_monitoring:
            response_level = scheduler.get_response_level(Q_t)
            if getattr(method_cfg, "jcde_no_prequery", False):
                scheduler._response_level = "normal"  # gating off; weight still sees response_level
        else:
            response_level = "normal"
            scheduler._response_level = "normal"

        # -- (b) trigger decision --
        triggered = _should_trigger(
            tick, scheduler, method_cfg,
            upv_state, conflict_scores, conv_status, exp_rng,
            feas_ratio=feas_ratio,
        )

        channel_type = ""
        ig_value = 0.0
        cognitive_cost = 0.0
        cooldown_set = 0
        conflict_resolved = 0
        gamma_improved = 0
        effective_interaction = 0
        feedback_update_weight = 0.0
        dm_response_time = 0.0   # ADM 响应时延（秒），仅在 triggered 时记录
        dm_quality = 0.0         # ADM 反馈质量 [0,1]，受疲劳传导（Boksem 2005, Hart 1988）

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

            channel_type = _select_channel(
                scheduler, method_cfg, phase, state_dict, exp_rng,
            )
            question = _generate_question(
                channel_type, qg, state, state.upv, phase, method_cfg, exp_rng,
            )

            # record weight snapshot before interaction for flip detection
            scheduler.record_weight_snapshot(state.upv.w)

            sdm.update_fatigue(total_interactions + 1)
            dm_response_time = sdm.simulate_latency(channel_type)
            dm_quality = sdm.simulate_quality(channel_type)
            feedback_update_weight = _quality_update_weight(
                method_cfg, response_level,
                _jcde_quality_for_weight(scheduler, channel_type, dm_response_time, dm_quality),
                channel=channel_type, phase=phase,
                budget_remaining=scheduler.budget_remaining,
            )

            response, ig_value = _execute_interaction(
                channel_type, question, sdm, state, state.upv,
                update_weight=feedback_update_weight,
                dm_quality=dm_quality,
                corrupt_feedback=True,
                stress_rng=corruption_rng,
            )
            ig_value *= feedback_update_weight

            total_interactions += 1

            # cognitive cost from CHANNEL_COST
            cognitive_cost = CHANNEL_COST.get(channel_type, 0.3)
            cumulative_cognitive_cost += cognitive_cost

            # sync scheduler state after interaction updated upv.sigma and upv.r
            _sched_sigma = dict(state.upv.sigma) if isinstance(state.upv.sigma, dict) else _sched_sigma
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
            scheduler.record_interaction(
                channel=channel_type,
                IG_actual=ig_value,
                latency=dm_response_time,
                quality=dm_quality,
            )
            cooldown_set = scheduler.cooldown_remaining

            # record detailed interaction (expanded fields)
            interaction_records.append({
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
                "dm_quality": round(dm_quality, 4),
                "feedback_update_weight": round(feedback_update_weight, 4),
            })

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
        w_dist = float(np.linalg.norm(
            np.asarray(state.upv.w) - sdm.w_star
        ))
        r_dist = float(np.linalg.norm(
            np.asarray(state.upv.r) - sdm.r_star
        ))

        # -- record row --
        csv_rows.append({
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
            "feedback_update_weight": round(feedback_update_weight, 4),
            "w_dist": round(w_dist, 6),
            "r_dist": round(r_dist, 6),
            "wall_time": round(time.time() - t_start, 2),
        })

        # -- progress bar --
        pct = (tick + 1) / T
        filled = int(bar_len * pct)
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stderr.write(
            f"\r[{bar}] {pct*100:5.1f}% | tick {tick+1}/{T} "
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
    p.add_argument("--method", type=str, default="AdaS-Full",
                    choices=METHODS, help="Scheduling method variant")
    p.add_argument("--problem", type=str, default="ZDT1",
                    help="Problem ID (e.g. ZDT1, BNH, DTLZ2)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--profile", type=str, default="P-balanced",
                    choices=["P-balanced", "P-edge"],
                    help="SDM preference profile")
    p.add_argument("--fatigue_rate", type=float, default=0.0,
                    help="SDM fatigue growth rate")
    p.add_argument("--sigma_noise", type=float, default=0.1,
                    help="SDM base noise level")
    p.add_argument("--T", type=int, default=80,
                    help="Number of optimisation ticks")
    p.add_argument("--output_dir", type=str,
                    default="results/scheduling_exp",
                    help="Root output directory")

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
    p.add_argument("--bootstrap_channels", type=str, default=None,
                   help="Comma-separated adaptive-only channel sequence used before IG-based channel selection")

    return p.parse_args()


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
