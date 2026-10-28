# -*- coding: utf-8 -*-
"""
hitlo.nlp.dialogue_manager
=============================
Multi-turn dialogue context management and optimization state awareness.

Provides:
- ``DialogueMemory`` -- sliding window of structured dialogue turns
- ``build_context_prompt`` -- compose dialogue history + opt state into LLM context
- ``classify_intent`` -- rule-based intent pre-classification
- ``OptContextSummary`` -- snapshot of current optimization state for prompt injection
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hitlo.engine.opt_state import OptState


# ====================================================================
#  Data structures
# ====================================================================

@dataclass
class DialogueTurn:
    """A single turn in the dialogue history."""
    user_text: str
    parsed_ops: List[Dict[str, Any]] = field(default_factory=list)
    upv_snapshot: Optional[Dict[str, Any]] = None
    opt_context: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0
    gamma_mean: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class OptContextSummary:
    """Compact snapshot of optimization state for prompt injection."""
    tick: int = 0
    hv: float = 0.0
    hv_delta: float = 0.0
    feas_ratio: float = 0.0
    mean_violation: float = 0.0
    n_eval: int = 0
    current_algo: str = ""
    convergence_status: str = "running"
    n_obj: int = 2
    n_var: int = 10
    upv_w: List[float] = field(default_factory=list)
    upv_tau: float = 0.2

    def to_prompt_text(self) -> str:
        """Format as a Chinese-language context block for LLM injection."""
        lines = [
            "[当前优化状态]",
            f"  迭代轮次: {self.tick}, 累计评估: {self.n_eval}",
            f"  超体积(HV): {self.hv:.4f} (delta={self.hv_delta:+.4f})",
            f"  可行率: {self.feas_ratio:.1%}, 平均违反: {self.mean_violation:.3f}",
            f"  当前算法: {self.current_algo}",
            f"  收敛状态: {self.convergence_status}",
            f"  目标数: {self.n_obj}, 变量数: {self.n_var}",
            f"  当前UPV权重: {[f'{w:.2f}' for w in self.upv_w]}",
            f"  信任度(tau): {self.upv_tau:.2f}",
        ]
        return "\n".join(lines)


# ====================================================================
#  DialogueMemory
# ====================================================================

class DialogueMemory:
    """Sliding-window dialogue context manager.

    Maintains the most recent K turns of structured dialogue history,
    enabling anaphora resolution and context-aware LLM prompting.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._turns: List[DialogueTurn] = []

    @property
    def turns(self) -> List[DialogueTurn]:
        return self._turns

    @property
    def size(self) -> int:
        return len(self._turns)

    def add_turn(self, turn: DialogueTurn):
        """Add a new dialogue turn, evicting oldest if at capacity."""
        self._turns.append(turn)
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]

    def get_recent(self, k: int = 5) -> List[DialogueTurn]:
        """Get the most recent k turns."""
        return self._turns[-k:]

    def clear(self):
        self._turns.clear()

    def to_prompt_text(self, k: int = 5) -> str:
        """Format recent turns as a dialogue history block for LLM prompt."""
        recent = self.get_recent(k)
        if not recent:
            return ""

        lines = ["[对话历史]"]
        for i, turn in enumerate(recent):
            lines.append(f"  [{i+1}] 用户: {turn.user_text[:80]}")
            if turn.parsed_ops:
                ops_desc = ", ".join(
                    op.get("op", "?") for op in turn.parsed_ops[:3]
                )
                lines.append(f"      操作: {ops_desc}")
            if turn.upv_snapshot:
                w = turn.upv_snapshot.get("w", [])
                lines.append(f"      UPV权重: {[f'{v:.2f}' for v in w[:4]]}")
        return "\n".join(lines)


# ====================================================================
#  Intent Classification
# ====================================================================

# Intent categories and their keywords
_INTENT_PATTERNS: List[tuple] = [
    # (intent, keywords_list, priority)
    ("follow_up",       ["上一条", "刚才", "之前", "前面", "那个", "继续",
                         "再", "还要", "此外"], 10),
    ("query",           ["什么是", "请解释", "为什么", "怎么样", "现在",
                         "当前状态", "帮我看看", "查看"], 8),
    ("constraint_set",  ["约束", "限制", "必须", "不得", "不能", "不允许",
                         "至少", "至多", "不超过", "不低于"], 6),
    ("weight_adjust",   ["权重", "重要", "比重", "占比", "优先考虑",
                         "更关注"], 5),
    ("strategy_change", ["策略", "信任", "探索", "利用", "保守", "激进",
                         "收缩", "扩大搜索"], 4),
    ("robustness",      ["鲁棒", "CVaR", "概率约束", "可靠", "不确定",
                         "噪声", "风险"], 4),
]


def classify_intent(text: str) -> str:
    """Classify user NL input into an intent category.

    Returns one of: constraint_set, weight_adjust, strategy_change,
    robustness, query, follow_up.
    """
    text_lower = text.lower()
    best_intent = "weight_adjust"  # default
    best_score = 0

    for intent, keywords, priority in _INTENT_PATTERNS:
        score = 0
        for kw in keywords:
            if kw.lower() in text_lower:
                score += priority
        if score > best_score:
            best_score = score
            best_intent = intent

    return best_intent


def is_dialogue_query(text: str) -> bool:
    """Check if the input is a pure dialogue query (not a preference command)."""
    intent = classify_intent(text)
    return intent == "query"


# ====================================================================
#  Context prompt builder
# ====================================================================

def build_context_prompt(
    memory: DialogueMemory,
    opt_state: "OptState",
    k_turns: int = 5,
) -> str:
    """Build the full context injection block for the LLM prompt.

    Combines:
    1. Dialogue history (recent k turns)
    2. Current optimization state summary

    Parameters
    ----------
    memory : DialogueMemory
        The dialogue history manager.
    opt_state : OptState
        Current optimization state.
    k_turns : int
        Number of recent turns to include.

    Returns
    -------
    str
        Context block to prepend to the LLM system message.
    """
    parts = []

    # Dialogue history
    history = memory.to_prompt_text(k=k_turns)
    if history:
        parts.append(history)

    # Optimization state
    ctx = get_opt_context_summary(opt_state)
    parts.append(ctx.to_prompt_text())

    return "\n\n".join(parts)


def get_opt_context_summary(state: "OptState") -> OptContextSummary:
    """Extract a compact optimization state summary from OptState."""
    try:
        hv_hist = getattr(state, "hv_history", []) or []
        hv = hv_hist[-1] if hv_hist else 0.0
        hv_prev = hv_hist[-2] if len(hv_hist) >= 2 else hv
        hv_delta = hv - hv_prev

        feas_hist = getattr(state, "feas_history", []) or []
        feas = feas_hist[-1] if feas_hist else 0.0

        viol_hist = getattr(state, "viol_history", []) or []
        viol = viol_hist[-1] if viol_hist else 0.0

        # Convergence status
        stall_count = getattr(state, "stall_count", 0)
        if stall_count >= 3:
            conv_status = "stagnated"
        elif getattr(state, "converged", False):
            conv_status = "converged"
        else:
            conv_status = "running"

        # Current algorithm
        algo = getattr(state, "current_algo", "")
        if not algo:
            algo = getattr(state, "last_algo", "unknown")

        upv = getattr(state, "upv", None)
        w = list(upv.w) if upv is not None and hasattr(upv, "w") else []
        tau = float(upv.tau) if upv is not None and hasattr(upv, "tau") else 0.2

        return OptContextSummary(
            tick=int(getattr(state, "tick", 0)),
            hv=float(hv),
            hv_delta=float(hv_delta),
            feas_ratio=float(feas),
            mean_violation=float(viol),
            n_eval=int(getattr(state, "total_evals", 0)),
            current_algo=str(algo),
            convergence_status=conv_status,
            n_obj=int(getattr(state, "n_obj", 2)),
            n_var=int(getattr(state, "n_var", 10)),
            upv_w=[float(x) for x in w],
            upv_tau=float(tau),
        )
    except Exception:
        return OptContextSummary()


# ====================================================================
#  Proactive suggestion triggers
# ====================================================================

# Suggestion types with cooldown (in ticks)
_SUGGESTION_COOLDOWNS: Dict[str, int] = {
    "hv_stagnation": 20,
    "feas_drop": 15,
    "algo_switch": 10,
}


def check_proactive_suggestions(state: "OptState") -> List[Dict[str, Any]]:
    """Check optimization state for events that warrant proactive suggestions.

    Returns a list of suggestion dicts (compatible with L5 queue format).
    """
    suggestions = []
    tick = int(getattr(state, "tick", 0))

    # Track last suggestion ticks
    if not hasattr(state, "_suggest_last_tick"):
        state._suggest_last_tick = {}

    def _can_suggest(kind: str) -> bool:
        last = state._suggest_last_tick.get(kind, -999)
        cooldown = _SUGGESTION_COOLDOWNS.get(kind, 15)
        return (tick - last) >= cooldown

    # 1) HV stagnation
    hv_hist = getattr(state, "hv_history", []) or []
    if len(hv_hist) >= 15 and _can_suggest("hv_stagnation"):
        recent_hv = hv_hist[-15:]
        hv_range = max(recent_hv) - min(recent_hv)
        if hv_range < 1e-4:
            suggestions.append({
                "text": (
                    f"HV 已连续 15 代无明显提升（振幅 {hv_range:.6f}）。"
                    "建议：放松约束、增加探索偏置、或切换求解器。"
                ),
                "kind": "suggest",
                "choices": ["增大探索偏置", "放松软约束", "暂不调整"],
            })
            state._suggest_last_tick["hv_stagnation"] = tick

    # 2) Feasibility ratio drop
    feas_hist = getattr(state, "feas_history", []) or []
    if len(feas_hist) >= 5 and _can_suggest("feas_drop"):
        if feas_hist[-1] < 0.1 and feas_hist[-5] >= 0.2:
            suggestions.append({
                "text": (
                    f"可行率从 {feas_hist[-5]:.1%} 骤降至 {feas_hist[-1]:.1%}。"
                    "建议：检查约束设置是否过严，或增大惩罚容忍度。"
                ),
                "kind": "suggest",
                "choices": ["放宽硬约束", "增大惩罚容忍度", "暂不调整"],
            })
            state._suggest_last_tick["feas_drop"] = tick

    # 3) Algorithm switch event
    current_algo = getattr(state, "current_algo", "")
    last_algo = getattr(state, "_last_reported_algo", "")
    if current_algo and current_algo != last_algo and _can_suggest("algo_switch"):
        if last_algo:  # skip first report
            suggestions.append({
                "text": (
                    f"求解器已从 {last_algo} 切换为 {current_algo}。"
                    "这是OSV调度的自动决策，基于当前过程特征。"
                ),
                "kind": "suggest",
                "choices": [],
            })
            state._suggest_last_tick["algo_switch"] = tick
        state._last_reported_algo = current_algo

    return suggestions
