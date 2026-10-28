"""Transparent rule-based grounding of free-text clarifications for the E10' study (SAV-AERO).

Grounds a sentence to one preference slot when it states (a) a numeric aspiration for Cd0 or Cn, or
(b) a relative-importance statement between drag and normal force. Anything else is ungrounded and
produces no hard update (the paper's "ungrounded text does not update the preference model").
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

_NUM = r"(-?\d+(?:\.\d+)?)"
# Objective mentions; Cd0 is rewritten to a digit-free token first so its "0" is never read as a value.
_DRAG = r"(?:cdzero|zero[- ]lift drag(?: coefficient)?|drag(?: coefficient)?)"
_CN = r"(?:\bcn\b|normal[- ]force(?: coefficient)?)"
_CD0_VAL = re.compile(_DRAG + r"[^0-9\-.;]{0,30}?" + _NUM, re.I)
_CN_VAL = re.compile(_CN + r"[^0-9\-.;]{0,30}?" + _NUM, re.I)
_TERM = rf"({_DRAG}|{_CN})"
_A_OVER_B = re.compile(
    _TERM
    + r"[^.;]{0,40}?\b(?:more important than|over|rather than|matters more than|before|ahead of)\b"
    r"[^.;]{0,20}?" + _TERM,
    re.I,
)
_FAVOUR = re.compile(
    r"\b(?:prioriti[sz]e|favou?r|prefer|emphasi[sz]e|focus on)\s+(?:low(?:er)?\s+|high(?:er)?\s+)?"
    + _TERM,
    re.I,
)
_IS_FIRST = re.compile(
    _TERM
    + r"[^.;]{0,25}?\b(?:is|are)\s+(?:more important|the (?:main )?priority|my (?:main )?priority"
    r"|first|dominant|what matters most)",
    re.I,
)
_EQUAL = re.compile(
    r"\bequal(?:ly)?\b[^.;]{0,25}\b(?:weight|importance|important|priority|matter)|"
    r"\b(?:matter|count|weigh)s? equally\b|\bbalanced\b|\bsame (?:weight|importance|priority)\b",
    re.I,
)
_PCT = re.compile(
    r"(\d{1,3})\s*(?:%|percent)[^.;]{0,25}"
    + _DRAG
    + "|"
    + _DRAG
    + r"[^.;]{0,25}?(\d{1,3})\s*(?:%|percent)",
    re.I,
)


def _is_drag(term: str) -> bool:
    return re.fullmatch(_DRAG, term.strip(), re.I) is not None


def ground(text: str, target_slot: str) -> Dict[str, Any]:
    """Return {'grounded': bool, 'slot': str|None, 'value': float|None, 'rule': str}."""
    t = re.sub(r"\bc[_ ]?d[_ ]?0\b", "cdzero", text or "", flags=re.I)
    out: Dict[str, Any] = {
        "grounded": False,
        "slot": None,
        "value": None,
        "rule": "none",
    }
    if target_slot.startswith("r_"):
        idx = int(target_slot.split("_")[1])
        m = (_CD0_VAL if idx == 0 else _CN_VAL).search(t)
        if m:
            v = float(m.group(1))
            out.update(
                grounded=True,
                slot=target_slot,
                value=v if idx == 0 else -abs(v),
                rule="numeric_aspiration",
            )
            return out
    pct = _PCT.search(t)
    if pct:
        share = float(pct.group(1) or pct.group(2)) / 100.0
        if 0.0 < share < 1.0:
            out.update(
                grounded=True, slot="w_0", value=share, rule="drag_share_percent"
            )
            return out
    for rule, rx in (
        ("a_over_b", _A_OVER_B),
        ("favour", _FAVOUR),
        ("is_first", _IS_FIRST),
    ):
        m = rx.search(t)
        if m:
            drag_first = _is_drag(m.group(1))
            out.update(
                grounded=True,
                slot="w_0",
                value=0.7 if drag_first else 0.3,
                rule=("drag" if drag_first else "normal_force")
                + "_more_important:"
                + rule,
            )
            return out
    if _EQUAL.search(t):
        out.update(grounded=True, slot="w_0", value=0.5, rule="equal_importance")
    return out


def ground_or_none(text: str, target_slot: str) -> Optional[Dict[str, Any]]:
    g = ground(text, target_slot)
    return g if g["grounded"] else None
