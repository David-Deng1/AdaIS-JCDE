# -*- coding: utf-8 -*-
"""
hitlo.nlp.validation_pipeline
================================
Six-stage NL-to-UPV validation pipeline (Ch3 §3.3).

Pipeline stages:
  S1  Intent Segmentation    – split NL into typed intent segments
  S2  Layered RAG Retrieval  – route each segment to appropriate knowledge layers
  S3  N-Sample Generation    – parallel LLM calls for consistency estimation
  S4  Four-Gate Validation   – G1 format / G2 evidence / G3 consistency / G4 history
  S5  Confidence Calibration – compute per-atom γ score
  S6  Minimal Clarification  – generate closed-form questions for low-γ items

Usage::

    from hitlo.nlp.validation_pipeline import run_pipeline
    output = run_pipeline(nl_text, state, n_samples=3)
"""

from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .llm_client import (
    http_chat,
    extract_json_block,
    API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
)
from .prompt_templates import build_patch_prompt

if TYPE_CHECKING:
    from hitlo.engine.opt_state import OptState


# ====================================================================
#  Data classes
# ====================================================================

# Slot types that map to UPV composition (7 canonical slots)
SLOT_TYPES = (
    "weight", "reference", "band",
    "constraint_hard", "constraint_soft",
    "robustness", "strategy",
)

# Mapping from slot type to preferred RAG knowledge layers
SLOT_TO_LAYERS: Dict[str, List[str]] = {
    "weight":          ["L1"],                 # terminology (variable/obj names)
    "reference":       ["L1"],
    "band":            ["L1", "L2"],
    "constraint_hard": ["L2", "L4"],           # design standards + heuristic rules
    "constraint_soft": ["L2", "L4"],
    "robustness":      ["L2", "L4", "L3"],     # standards + rules + historical logs
    "strategy":        ["L3", "L4"],           # historical logs + heuristic rules
}

# Op types that correspond to each slot type
SLOT_TO_OPS: Dict[str, List[str]] = {
    "weight":          ["set_weight"],
    "reference":       ["set_reference"],
    "band":            ["set_band"],
    "constraint_hard": ["add_constraint"],
    "constraint_soft": ["add_constraint"],
    "robustness":      ["set_cvar", "set_chance", "set_gamma", "set_decay",
                        "set_init_penalty"],
    "strategy":        ["set_trust", "set_priority"],
}


@dataclass
class IntentSegment:
    """A single preference intent parsed from user NL input.

    Attributes
    ----------
    text : str
        The original sub-sentence or clause text.
    slot_type : str
        Predicted target slot in UPV (one of ``SLOT_TYPES``).
    numeric_values : list of float
        Numeric values extracted by the rule layer (may be empty).
    source_span : tuple of int
        Character offset (start, end) in the original NL text.
    """
    text: str
    slot_type: str
    numeric_values: List[float] = field(default_factory=list)
    source_span: Tuple[int, int] = (0, 0)


@dataclass
class ValidationResult:
    """Validation outcome for a single patch operation.

    Attributes
    ----------
    op : dict
        The original patch operation dict (e.g. ``{"op": "set_weight", ...}``).
    g1_pass : bool
        G1 format gate: JSON parseable, fields complete, values in range.
    g2_score : float
        G2 evidence gate: how well the cited evidence supports this op [0, 1].
    g3_consensus : float
        G3 N-sample consistency: agreement rate across N LLM outputs [0, 1].
    g4_pass : bool
        G4 history gate: no unreasonable jump from UPV(t-1).
    gamma : float
        Calibrated confidence score for this atom.
    evidence_ids : list of str
        RAG document IDs cited as supporting evidence.
    intent_segment : IntentSegment or None
        The originating intent segment (for provenance tracing).
    """
    op: Dict[str, Any]
    g1_pass: bool = True
    g2_score: float = 1.0
    g3_consensus: float = 1.0
    g4_pass: bool = True
    gamma: float = 1.0
    evidence_ids: List[str] = field(default_factory=list)
    intent_segment: Optional[IntentSegment] = None

    @property
    def all_gates_pass(self) -> bool:
        return self.g1_pass and self.g4_pass and self.g2_score >= 0.3

    @property
    def needs_clarification(self) -> bool:
        return self.gamma < 0.5 or not self.g1_pass or not self.g4_pass


@dataclass
class PipelineOutput:
    """Complete output of the six-stage validation pipeline.

    Attributes
    ----------
    validated_ops : list of ValidationResult
        All validated patch operations with gate scores and gamma.
    clarification_needed : list of dict
        Low-confidence items requiring user clarification.
        Each dict: ``{"text": str, "kind": str, "choices": list}``.
    provenance_chain : list of dict
        Full evidence chain for each operation (for UI visualization).
    raw_patch : dict
        The winning (majority-vote) raw patch before validation.
    segments : list of IntentSegment
        S1 segmentation results.
    n_samples_used : int
        Actual N used for generation.
    pipeline_time_ms : float
        Wall-clock time of the full pipeline in milliseconds.
    """
    validated_ops: List[ValidationResult] = field(default_factory=list)
    clarification_needed: List[Dict[str, Any]] = field(default_factory=list)
    provenance_chain: List[Dict[str, Any]] = field(default_factory=list)
    raw_patch: Dict[str, Any] = field(default_factory=dict)
    segments: List[IntentSegment] = field(default_factory=list)
    n_samples_used: int = 1
    pipeline_time_ms: float = 0.0

    @property
    def high_confidence_ops(self) -> List[ValidationResult]:
        """Operations with gamma >= 0.5 and all critical gates passed."""
        return [v for v in self.validated_ops if not v.needs_clarification]

    @property
    def mean_gamma(self) -> float:
        if not self.validated_ops:
            return 0.0
        return float(np.mean([v.gamma for v in self.validated_ops]))

    def to_summary_text(self) -> str:
        """Generate a human-readable summary for the chat bubble."""
        n_total = len(self.validated_ops)
        n_high = len(self.high_confidence_ops)
        n_clarify = len(self.clarification_needed)

        parts = []
        # Describe applied operations in natural language
        for vr in self.high_confidence_ops:
            desc = _op_to_chinese(vr.op)
            if desc:
                parts.append(f"  · {desc}")

        header = f"已识别 {n_total} 个偏好意图"
        if n_high == n_total:
            header += "，全部高置信直接应用。"
        elif n_clarify > 0:
            header += f"，{n_high} 个高置信已应用，{n_clarify} 个需要确认。"
        else:
            header += f"，{n_high} 个已应用。"

        lines = [header]
        if parts:
            lines.extend(parts)
        return "\n".join(lines)


# ====================================================================
#  S1: Intent Segmentation
# ====================================================================

# Clause-splitting patterns (Chinese sentence connectors / punctuation)
_CLAUSE_SPLITTERS = re.compile(
    r"[，,；;。！!？?\n]+"
    r"|(?:并且|同时|另外|此外|而且|但是|不过|然后|接着)"
)

# Slot-type classification rules (keyword → slot_type)
_SLOT_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("constraint_hard", ["必须", "强制", "不得", "不能", "不允许", "硬约束",
                         "shall", "must"]),
    ("constraint_soft", ["希望", "尽量", "最好", "prefer", "软约束", "try"]),
    ("robustness",      ["鲁棒", "CVaR", "cvar", "概率约束", "尾部风险",
                         "保守", "激进", "噪声", "不确定", "可靠度", "chance"]),
    ("strategy",        ["信任", "探索", "利用", "收缩", "搜索范围", "trust",
                         "explore", "exploit", "priority", "优先"]),
    ("band",            ["范围", "区间", "band", "上界", "下界", "上限", "下限"]),
    ("reference",       ["参考点", "reference", "理想点", "aspiration"]),
    ("weight",          ["权重", "重要", "weight", "比重", "占比"]),
]

# Numeric extraction patterns
# Note: f2/x3 indices are NOT captured as numeric values (non-capturing group)
_NUM_PATTERN = re.compile(
    r"(?:(\d+(?:\.\d+)?)\s*[%％])"                              # percentage
    r"|(?:(?:到|为|=|等于|设为|提高到|降低到|从)\s*(\d+(?:\.\d+)?))" # target value
    r"|(?:\[\s*(\d+(?:\.\d+)?)\s*[,，]\s*(\d+(?:\.\d+)?)\s*\])" # range [a, b]
    r"|(?:([<>≤≥]=?)\s*(\d+(?:\.\d+)?))"                        # comparison
    r"|(?:(?<![fFxX\d])(\d+\.\d+)(?!\d))"                       # standalone decimal
)


def s1_segment(nl_text: str) -> List[IntentSegment]:
    """S1: Split NL text into independent intent segments with slot classification.

    Uses rule-based clause splitting + keyword-based slot routing.
    """
    text = nl_text.strip()
    if not text:
        return []

    # Split into clauses
    clauses = _CLAUSE_SPLITTERS.split(text)
    clauses = [c.strip() for c in clauses if c.strip()]

    # If no split happened, treat the whole text as one segment
    if not clauses:
        clauses = [text]

    segments = []
    for clause in clauses:
        slot_type = _classify_slot(clause)
        numerics = _extract_numerics(clause)
        # Find span in original text
        start = text.find(clause)
        end = start + len(clause) if start >= 0 else 0
        segments.append(IntentSegment(
            text=clause,
            slot_type=slot_type,
            numeric_values=numerics,
            source_span=(max(0, start), end),
        ))

    return segments


def _classify_slot(clause: str) -> str:
    """Classify a clause into a slot type by keyword matching."""
    clause_lower = clause.lower()
    for slot_type, keywords in _SLOT_KEYWORDS:
        for kw in keywords:
            if kw.lower() in clause_lower:
                return slot_type
    # Default: if contains objective/variable reference, treat as weight
    if re.search(r"[fF]\d+|目标\d*|f_\d+", clause):
        return "weight"
    if re.search(r"[xX]\d+|变量\d*|x_\d+", clause):
        return "constraint_hard"
    return "weight"  # safest default


def _extract_numerics(clause: str) -> List[float]:
    """Extract numeric values from a clause."""
    nums = []
    for m in _NUM_PATTERN.finditer(clause):
        for g in m.groups():
            if g and re.match(r"\d+(?:\.\d+)?$", g):
                try:
                    nums.append(float(g))
                except ValueError:
                    pass
    return nums


# ====================================================================
#  S2: Layered RAG Retrieval
# ====================================================================

def s2_retrieve(segments: List[IntentSegment], top_k: int = 3
                ) -> Dict[str, List[Any]]:
    """S2: Route each segment to appropriate knowledge layers and retrieve.

    Returns a dict mapping segment index (as str) to list of RAGResult objects.
    """
    results: Dict[str, List[Any]] = {}
    try:
        from hitlo.nlp.rag_store import get_rag_store
        store = get_rag_store()
        if store.size == 0:
            return results
    except Exception:
        return results

    for i, seg in enumerate(segments):
        preferred_layers = SLOT_TO_LAYERS.get(seg.slot_type, ["L1", "L2"])
        hits = store.retrieve(seg.text, top_k=top_k, layers=preferred_layers)
        # If few results from preferred layers, fall back to all layers
        if len(hits) < 2:
            hits_all = store.retrieve(seg.text, top_k=top_k)
            # Merge, dedup by doc_id
            seen = {r.doc.doc_id for r in hits}
            for r in hits_all:
                if r.doc.doc_id not in seen:
                    hits.append(r)
                    seen.add(r.doc.doc_id)
        results[str(i)] = hits[:top_k]

    return results


def _format_rag_for_prompt(rag_results: Dict[str, List[Any]]) -> str:
    """Format RAG results into a context string for the LLM prompt."""
    if not rag_results:
        return ""
    lines = ["[领域知识检索结果]"]
    seen_ids = set()
    for _seg_idx, hits in rag_results.items():
        for r in hits:
            doc_id = r.doc.doc_id
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            lines.append(
                f"  [{r.doc.layer}] {doc_id}: {r.doc.title} — "
                f"{r.doc.content[:120]}"
            )
    return "\n".join(lines[:15])  # cap at 15 entries


# ====================================================================
#  S3: N-Sample Structured Generation
# ====================================================================

def s3_generate(
    nl_text: str,
    rag_context: str = "",
    n_samples: int = 3,
    temperature: float = 0.5,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """S3: Generate N independent LLM patch responses in parallel.

    For N=1, uses temperature=0 for deterministic output.
    For N>1, uses the specified temperature and ThreadPoolExecutor.

    Returns a list of parsed patch dicts (may contain fewer than N on failures).
    """
    model = model or DEFAULT_MODEL
    api_key = api_key or API_KEY
    base_url = base_url or DEFAULT_BASE_URL

    # Build enhanced prompt with RAG context
    system_msg = (
        "You convert Chinese natural language into strict JSON "
        "patches for a multi-objective Unified Preference Vector (UPV)."
    )
    if rag_context:
        system_msg += (
            "\n\n以下是与用户输入相关的领域知识，请用于消歧和提供证据支持：\n"
            + rag_context
        )

    user_msg = build_patch_prompt(nl_text)

    def _single_call(temp: float) -> Optional[Dict[str, Any]]:
        payload = {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": temp,
        }
        obj = http_chat(payload, model=model, api_key=api_key, base_url=base_url)
        if obj is None:
            return None
        try:
            content = obj["choices"][0]["message"]["content"]
        except Exception:
            content = obj.get("content") or json.dumps(obj)
        data = extract_json_block(content)
        if data is None:
            return None
        # Ensure standard structure
        if "mode" not in data:
            data["mode"] = "patch"
        if "ops" not in data or not isinstance(data["ops"], list):
            data["ops"] = []
        return data

    if n_samples <= 1:
        result = _single_call(0.0)
        return [result] if result else []

    # Parallel N-sample generation
    results = []
    with ThreadPoolExecutor(max_workers=min(n_samples, 5)) as executor:
        futures = {
            executor.submit(_single_call, temperature): i
            for i in range(n_samples)
        }
        for future in as_completed(futures):
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception:
                pass

    return results


# ====================================================================
#  S4: Four-Gate Validation
# ====================================================================

# --- G1: Format Validation ---

_VALID_OPS = {
    "set_weight", "set_cvar", "set_chance", "set_gamma", "set_decay",
    "set_init_penalty", "set_trust", "set_priority",
    "set_reference", "set_band", "add_constraint",
}

# Value ranges for numeric fields
_VALUE_RANGES: Dict[str, Tuple[float, float]] = {
    "set_weight":       (0.0, 1.0),
    "set_cvar":         (0.0, 1.0),
    "set_chance":       (0.0, 1.0),
    "set_gamma":        (0.0, 10.0),
    "set_decay":        (0.0, 1.0),
    "set_init_penalty": (0.0, 1000.0),
    "set_trust":        (0.0, 1.0),
}


def g1_format_validate(op: Dict[str, Any], n_obj: int = 2, n_var: int = 10
                       ) -> bool:
    """G1: Validate op format, field completeness, and value ranges."""
    op_type = op.get("op")
    if op_type not in _VALID_OPS:
        return False

    if op_type == "set_weight":
        idx = op.get("index")
        val = op.get("value")
        if idx is None or val is None:
            return False
        try:
            if not (0 <= int(idx) < n_obj):
                return False
            if not (0.0 <= float(val) <= 1.0):
                return False
        except (TypeError, ValueError):
            return False

    elif op_type in ("set_cvar", "set_chance", "set_gamma", "set_decay",
                     "set_init_penalty", "set_trust"):
        val = op.get("value")
        if val is None:
            return False
        try:
            lo, hi = _VALUE_RANGES.get(op_type, (0.0, 1e6))
            if not (lo <= float(val) <= hi):
                return False
        except (TypeError, ValueError):
            return False

    elif op_type == "set_reference":
        vals = op.get("values")
        if not isinstance(vals, list) or len(vals) == 0:
            return False

    elif op_type == "set_band":
        j = op.get("obj_index")
        if j is None:
            return False
        try:
            if not (0 <= int(j) < n_obj):
                return False
        except (TypeError, ValueError):
            return False

    elif op_type == "add_constraint":
        kind = op.get("kind")
        sign = op.get("sign")
        if kind not in ("var", "obj"):
            return False
        if sign not in ("<=", ">=", "=="):
            return False
        idx = op.get("idx")
        if idx is None:
            return False
        try:
            max_idx = n_var if kind == "var" else n_obj
            if not (0 <= int(idx) < max_idx):
                return False
        except (TypeError, ValueError):
            return False

    return True


# --- G2: Evidence Consistency ---

def g2_evidence_score(op: Dict[str, Any], rag_results: Dict[str, List[Any]]
                      ) -> float:
    """G2: Score how well RAG evidence supports this operation.

    Returns a score in [0, 1]. If no RAG results available, returns a neutral 0.5.
    """
    if not rag_results:
        return 0.5  # neutral when no evidence system

    # Collect all retrieved text
    all_texts = []
    for hits in rag_results.values():
        for r in hits:
            all_texts.append(r.doc.content.lower())

    if not all_texts:
        return 0.5

    # Simple keyword overlap check between op and retrieved evidence
    op_type = op.get("op", "")
    op_keywords = set()

    if op_type == "set_weight":
        op_keywords.add(f"f{op.get('index', 0) + 1}")
        op_keywords.update(["权重", "weight", "目标"])
    elif op_type == "add_constraint":
        kind = op.get("kind", "var")
        idx = op.get("idx", 0)
        prefix = "x" if kind == "var" else "f"
        op_keywords.add(f"{prefix}{idx + 1}")
        op_keywords.update(["约束", "constraint", "限制"])
    elif "cvar" in op_type or "chance" in op_type:
        op_keywords.update(["鲁棒", "robustness", "cvar", "概率", "可靠"])
    elif "trust" in op_type or "priority" in op_type:
        op_keywords.update(["信任", "策略", "探索", "trust"])

    if not op_keywords:
        return 0.5

    # Count keyword hits in evidence
    hit_count = 0
    for text in all_texts:
        for kw in op_keywords:
            if kw.lower() in text:
                hit_count += 1
                break

    score = min(1.0, hit_count / max(1, len(all_texts)))
    return max(0.3, score)  # floor at 0.3 to avoid over-penalizing


# --- G3: N-Sample Consistency ---

def g3_consensus_rate(all_patches: List[Dict[str, Any]]) -> Dict[str, float]:
    """G3: Compute per-op-type consensus rate across N samples.

    Returns a dict mapping op signature to agreement rate [0, 1].
    """
    if len(all_patches) <= 1:
        return {}  # not applicable for N=1

    N = len(all_patches)
    # Build op signature → count mapping
    sig_counts: Dict[str, int] = {}
    for patch in all_patches:
        seen_sigs = set()
        for op in patch.get("ops", []):
            sig = _op_signature(op)
            if sig not in seen_sigs:
                sig_counts[sig] = sig_counts.get(sig, 0) + 1
                seen_sigs.add(sig)

    # Consensus = count / N for each signature
    return {sig: count / N for sig, count in sig_counts.items()}


def _op_signature(op: Dict[str, Any]) -> str:
    """Generate a canonical signature for an op (for consensus comparison).

    Groups similar operations: e.g. set_weight(0, 0.7) and set_weight(0, 0.65)
    both produce "set_weight:0" so they match on intent if not exact value.
    """
    op_type = op.get("op", "unknown")
    if op_type == "set_weight":
        return f"set_weight:{op.get('index', '?')}"
    elif op_type == "set_band":
        return f"set_band:{op.get('obj_index', '?')}"
    elif op_type == "add_constraint":
        return f"add_constraint:{op.get('kind', '?')}:{op.get('idx', '?')}:{op.get('sign', '?')}"
    elif op_type in ("set_reference",):
        return "set_reference"
    else:
        return op_type


# --- G4: History Consistency ---

def g4_history_check(op: Dict[str, Any], current_upv: Any, M: int = 2
                     ) -> bool:
    """G4: Check that the operation doesn't produce an unreasonable jump from UPV(t-1).

    Thresholds:
      - Weight change >= 0.4 in a single step → suspicious
      - Trust change >= 0.5 in a single step → suspicious
      - CVaR/chance flip (>= 0.4 swing) → suspicious
    """
    if current_upv is None:
        return True  # no history to compare

    op_type = op.get("op")

    try:
        if op_type == "set_weight":
            idx = int(op.get("index", 0))
            new_val = float(op.get("value", 0.5))
            old_w = list(current_upv.w) if hasattr(current_upv.w, "__iter__") else [0.5]
            if 0 <= idx < len(old_w):
                if abs(new_val - float(old_w[idx])) >= 0.4:
                    return False

        elif op_type == "set_trust":
            new_val = float(op.get("value", 0.2))
            old_tau = float(getattr(current_upv, "tau", 0.2))
            if abs(new_val - old_tau) >= 0.5:
                return False

        elif op_type in ("set_cvar", "set_chance"):
            new_val = float(op.get("value", 0.9))
            rho = dict(getattr(current_upv, "rho", {}))
            key = "cvar_alpha" if op_type == "set_cvar" else "chance"
            old_val = float(rho.get(key, 0.9))
            if abs(new_val - old_val) >= 0.4:
                return False
    except Exception:
        pass  # if anything goes wrong, don't block

    return True


# ====================================================================
#  S5: Confidence Calibration
# ====================================================================

# Channel prior for NL channel (conservative, reflecting LLM skepticism)
CHANNEL_PRIOR_NL = 0.7


def s5_calibrate_gamma(
    vr: ValidationResult,
    consensus_rates: Dict[str, float],
    n_samples: int,
) -> float:
    """S5: Compute calibrated confidence γ for a single validated operation.

    Formula:
        γ = w_g1 * g1 + w_g2 * g2_score + w_g3 * consensus + w_g4 * g4
            × channel_prior

    Where:
        g1 = 1.0 if format pass, else 0.0
        g2_score = evidence consistency [0, 1]
        consensus = N-sample agreement rate [0, 1] (1.0 if N=1)
        g4 = 1.0 if history pass, else 0.0
        channel_prior = 0.7 (NL channel skepticism)
    """
    # Gate weights (sum = 1.0)
    w_g1, w_g2, w_g3, w_g4 = 0.30, 0.15, 0.35, 0.20

    g1_val = 1.0 if vr.g1_pass else 0.0
    g2_val = vr.g2_score

    # Get consensus for this op
    sig = _op_signature(vr.op)
    if n_samples > 1 and consensus_rates:
        g3_val = consensus_rates.get(sig, 0.0)
    else:
        # N=1 mode: no consensus info, use neutral value
        g3_val = 0.7  # mildly optimistic
        w_g3 = 0.10   # reduce weight of consensus gate
        # Redistribute weight
        w_g1 += 0.10
        w_g4 += 0.15

    g4_val = 1.0 if vr.g4_pass else 0.0

    raw_score = w_g1 * g1_val + w_g2 * g2_val + w_g3 * g3_val + w_g4 * g4_val
    gamma = raw_score * CHANNEL_PRIOR_NL

    return float(np.clip(gamma, 0.0, 1.0))


# ====================================================================
#  S6: Minimal Clarification Generation
# ====================================================================

def s6_generate_clarifications(
    validated_ops: List[ValidationResult],
    gamma_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """S6: Generate closed-form clarification questions for low-confidence items.

    Only produces binary / discrete / threshold confirmation questions.
    """
    clarifications = []

    for vr in validated_ops:
        if vr.gamma >= gamma_threshold:
            continue

        op = vr.op
        op_type = op.get("op", "")

        if not vr.g1_pass:
            # Format failure: ask for value confirmation
            desc = _op_to_chinese(op) or f"操作 {op_type}"
            clarifications.append({
                "text": f"无法解析操作[{desc}]，请确认或重新表述。",
                "kind": "clarify",
                "choices": ["跳过该项", "重新输入"],
                "source_op": op,
                "gamma": vr.gamma,
            })

        elif not vr.g4_pass:
            # History jump: confirm the large change
            desc = _op_to_chinese(op) or f"操作 {op_type}"
            clarifications.append({
                "text": f"[{desc}] 变化幅度较大，确认执行？",
                "kind": "confirm",
                "choices": ["确认执行", "取消该修改"],
                "source_op": op,
                "gamma": vr.gamma,
            })

        elif vr.g3_consensus < 0.5:
            # Low consistency: ask for disambiguation
            desc = _op_to_chinese(op) or f"操作 {op_type}"
            clarifications.append({
                "text": f"[{desc}] 的解读存在歧义（一致率 {vr.g3_consensus:.0%}），请确认。",
                "kind": "clarify",
                "choices": ["确认当前解读", "重新描述需求"],
                "source_op": op,
                "gamma": vr.gamma,
            })

    return clarifications


# ====================================================================
#  Majority-vote patch selection
# ====================================================================

def _select_majority_patch(all_patches: List[Dict[str, Any]]
                           ) -> Dict[str, Any]:
    """Select the 'best' patch from N samples via majority voting.

    For each op signature, pick the most common version.
    If N=1, just return the single patch.
    """
    if len(all_patches) == 1:
        return all_patches[0]
    if not all_patches:
        return {"mode": "patch", "ops": []}

    # Collect all ops by signature
    sig_to_ops: Dict[str, List[Dict[str, Any]]] = {}
    sig_to_count: Dict[str, int] = {}
    for patch in all_patches:
        seen = set()
        for op in patch.get("ops", []):
            sig = _op_signature(op)
            if sig not in seen:
                sig_to_count[sig] = sig_to_count.get(sig, 0) + 1
                seen.add(sig)
            sig_to_ops.setdefault(sig, []).append(op)

    # Keep ops that appear in majority (> N/2) of patches
    N = len(all_patches)
    threshold = N / 2.0
    majority_ops = []
    for sig, count in sig_to_count.items():
        if count >= threshold:
            # Pick the median value version
            ops_list = sig_to_ops[sig]
            majority_ops.append(_pick_median_op(ops_list))

    return {"mode": "patch", "ops": majority_ops}


def _pick_median_op(ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the median-valued op from a list of similar ops."""
    if len(ops) == 1:
        return ops[0]

    # Extract numeric "value" field and pick median
    vals = []
    for op in ops:
        v = op.get("value")
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass

    if vals:
        median_val = float(np.median(vals))
        # Find the op closest to median
        best_op = min(ops, key=lambda o: abs(float(o.get("value", 0)) - median_val))
        return best_op

    return ops[0]  # fallback: first op


# ====================================================================
#  Helper: op → Chinese description
# ====================================================================

def _op_to_chinese(op: Dict[str, Any]) -> str:
    """Convert a patch op to a short Chinese description."""
    op_type = op.get("op", "")
    try:
        if op_type == "set_weight":
            return f"将 f{int(op['index'])+1} 权重设为 {float(op['value']):.2f}"
        elif op_type == "set_cvar":
            return f"设置 CVaR α = {float(op['value']):.2f}"
        elif op_type == "set_chance":
            return f"设置概率约束置信度 = {float(op['value']):.2f}"
        elif op_type == "set_gamma":
            return f"设置鲁棒性裕度 γ = {float(op['value']):.2f}"
        elif op_type == "set_decay":
            return f"设置时间衰减 = {float(op['value']):.3f}"
        elif op_type == "set_init_penalty":
            return f"设置初始惩罚权重 = {float(op['value']):.1f}"
        elif op_type == "set_trust":
            return f"设置信任度 τ = {float(op['value']):.2f}"
        elif op_type == "set_priority":
            return f"设置优先策略: {op.get('value', '?')}"
        elif op_type == "set_reference":
            vals = op.get("values", [])
            return f"设置参考点 r = {vals}"
        elif op_type == "set_band":
            j = int(op.get("obj_index", 0))
            lo = op.get("lo", "-∞")
            hi = op.get("hi", "+∞")
            return f"设置 f{j+1} 可接受区间 [{lo}, {hi}]"
        elif op_type == "add_constraint":
            kind_cn = "变量" if op.get("kind") == "var" else "目标"
            idx = int(op.get("idx", 0)) + 1
            sign = op.get("sign", "<=")
            val = float(op.get("value", 0))
            hard_cn = "硬" if op.get("hard", True) else "软"
            prefix = "x" if op.get("kind") == "var" else "f"
            return f"新增{hard_cn}约束: {prefix}{idx} {sign} {val}"
    except Exception:
        pass
    return ""


# ====================================================================
#  N-sample mode configuration helper
# ====================================================================

def resolve_n_samples(mode: str = "balanced") -> Tuple[int, float]:
    """Resolve N-sample mode string to (n_samples, temperature).

    Parameters
    ----------
    mode : str
        One of "fast" (N=1), "balanced" (N=3), "precise" (N=5).

    Returns
    -------
    tuple of (int, float)
        (n_samples, temperature)
    """
    if mode == "fast":
        return 1, 0.0
    elif mode == "precise":
        return 5, 0.5
    else:  # balanced (default)
        return 3, 0.5


# ====================================================================
#  Main pipeline entry point
# ====================================================================

def run_pipeline(
    nl_text: str,
    state: "OptState",
    n_samples: Optional[int] = None,
    temperature: Optional[float] = None,
    gamma_threshold: float = 0.5,
) -> PipelineOutput:
    """Execute the full six-stage NL validation pipeline.

    Parameters
    ----------
    nl_text : str
        User's natural language input.
    state : OptState
        Current optimization state (for UPV history, problem info, config).
    n_samples : int or None
        Override N-sample count. If None, reads from state config.
    temperature : float or None
        Override LLM sampling temperature. If None, uses mode default.
    gamma_threshold : float
        Threshold below which items trigger clarification (S6).

    Returns
    -------
    PipelineOutput
        Complete pipeline output with validated ops, clarifications,
        and provenance chain.
    """
    t0 = time.perf_counter()

    # Resolve N-sample configuration
    if n_samples is None:
        mode = "balanced"
        # Try reading from app config if available
        app_cfg = getattr(state, "app_config", None)
        if app_cfg and hasattr(app_cfg, "n_sample_mode"):
            mode = getattr(app_cfg, "n_sample_mode", "balanced")
        n_samples, default_temp = resolve_n_samples(mode)
    else:
        default_temp = 0.5 if n_samples > 1 else 0.0

    if temperature is None:
        temperature = default_temp

    n_obj = getattr(state, "n_obj", 2)
    n_var = getattr(state, "n_var", 10)
    current_upv = getattr(state, "upv", None)

    output = PipelineOutput(n_samples_used=n_samples)

    # ---- S1: Intent Segmentation ----
    segments = s1_segment(nl_text)
    output.segments = segments

    # ---- S2: Layered RAG Retrieval ----
    rag_results = s2_retrieve(segments, top_k=3)
    rag_context = _format_rag_for_prompt(rag_results)

    # Inject dialogue + optimization context (multi-turn awareness)
    dialogue_ctx = ""
    try:
        dm = getattr(state, "dialogue_memory", None)
        if dm and dm.size > 0:
            dialogue_ctx = dm.to_prompt_text(k=5)
        from hitlo.nlp.dialogue_manager import get_opt_context_summary
        opt_summary = get_opt_context_summary(state).to_prompt_text()
        if dialogue_ctx:
            rag_context = f"{rag_context}\n\n{dialogue_ctx}\n\n{opt_summary}"
        else:
            rag_context = f"{rag_context}\n\n{opt_summary}"
    except Exception:
        pass

    # ---- S3: N-Sample Structured Generation ----
    all_patches = s3_generate(
        nl_text,
        rag_context=rag_context,
        n_samples=n_samples,
        temperature=temperature,
    )

    if not all_patches:
        # Total failure: return empty output with clarification
        output.clarification_needed.append({
            "text": "LLM 无法解析您的输入，请重新表述。",
            "kind": "clarify",
            "choices": ["重新输入"],
        })
        output.pipeline_time_ms = (time.perf_counter() - t0) * 1000
        return output

    # Compute N-sample consensus (G3)
    consensus_rates = g3_consensus_rate(all_patches)

    # Select majority-vote patch
    best_patch = _select_majority_patch(all_patches)
    output.raw_patch = best_patch

    # ---- S4 + S5: Per-op validation + calibration ----
    for op in best_patch.get("ops", []):
        # G1: Format
        g1 = g1_format_validate(op, n_obj=n_obj, n_var=n_var)

        # G2: Evidence
        g2 = g2_evidence_score(op, rag_results)

        # G3: Consensus (looked up from pre-computed rates)
        sig = _op_signature(op)
        g3 = consensus_rates.get(sig, 1.0 if n_samples <= 1 else 0.0)

        # G4: History
        g4 = g4_history_check(op, current_upv, M=n_obj)

        # Find matching segment for provenance
        matched_seg = _match_op_to_segment(op, segments)

        vr = ValidationResult(
            op=op,
            g1_pass=g1,
            g2_score=g2,
            g3_consensus=g3,
            g4_pass=g4,
            evidence_ids=[
                r.doc.doc_id
                for hits in rag_results.values()
                for r in hits
            ][:5],
            intent_segment=matched_seg,
        )

        # S5: Calibrate gamma
        vr.gamma = s5_calibrate_gamma(vr, consensus_rates, n_samples)

        output.validated_ops.append(vr)

        # Build provenance chain entry
        output.provenance_chain.append({
            "op": op,
            "segment": matched_seg.text if matched_seg else nl_text,
            "slot_type": matched_seg.slot_type if matched_seg else "unknown",
            "rag_evidence": [r.doc.doc_id for hits in rag_results.values() for r in hits][:3],
            "g1": g1, "g2": g2, "g3": g3, "g4": g4,
            "gamma": vr.gamma,
            "description": _op_to_chinese(op),
        })

    # ---- S6: Minimal Clarification ----
    output.clarification_needed = s6_generate_clarifications(
        output.validated_ops, gamma_threshold
    )

    output.pipeline_time_ms = (time.perf_counter() - t0) * 1000
    return output


def _match_op_to_segment(op: Dict[str, Any], segments: List[IntentSegment]
                         ) -> Optional[IntentSegment]:
    """Heuristically match a patch op to its originating intent segment."""
    op_type = op.get("op", "")

    # Determine which slot type this op belongs to
    target_slots = []
    for slot, ops_list in SLOT_TO_OPS.items():
        if op_type in ops_list:
            target_slots.append(slot)

    if not target_slots:
        return segments[0] if segments else None

    # Find segment with matching slot type
    for seg in segments:
        if seg.slot_type in target_slots:
            return seg

    return segments[0] if segments else None
