# -*- coding: utf-8 -*-
"""
hitlo.nlp.nl2upv
~~~~~~~~~~~~~~~~~~
Natural-language to Unified Preference Vector (UPV) conversion.

Provides:
- Pydantic data models: ``ConstraintAtom``, ``Robustness``, ``Strategy``, ``UPVPydantic``
- ``normalize_weights`` utility
- ``regex_parse_atoms`` -- regex fallback parser
- ``parse_nl_to_atoms`` -- LLM-first, regex-fallback atom parser
- ``atoms_to_upv`` -- atoms dict to ``UPVPydantic`` object
- ``apply_patch_to_upv`` -- apply an LLM-produced JSON patch to an existing UPV
"""

import re
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator

from .llm_client import (
    call_openai_atoms,
    API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
)


# =========================================================
#  Pydantic UPV structure (aligned with UPV composition table)
# =========================================================
class ConstraintAtom(BaseModel):
    """Single constraint: variable constraint or objective constraint."""

    kind: str  # 'var' | 'obj'
    idx: int  # 0-based
    sign: str  # '<=' | '>=' | '=='
    value: float
    hard: bool = True
    penalty: float = 10.0

    @field_validator("kind")
    @classmethod
    def _kind_ok(cls, v):
        if v not in ("var", "obj"):
            raise ValueError("kind must be 'var' or 'obj'")
        return v

    @field_validator("sign")
    @classmethod
    def _sign_ok(cls, v):
        if v not in ("<=", ">=", "=="):
            raise ValueError("sign must be one of <=, >=, ==")
        return v


class Robustness(BaseModel):
    """
    Robustness parameters:
    - p_chance: chance confidence
    - alpha_CVaR: CVaR quantile
    - gamma: robustness margin
    - kappa0: initial soft-constraint penalty weight
    - time_decay: c_s time decay
    - noise_std: observation noise
    """

    cvar_alpha: float = Field(default=0.9, ge=0.0, le=1.0)
    chance: float = Field(default=0.9, ge=0.0, le=1.0)
    tail_q: float = Field(default=0.9, ge=0.0, le=1.0)
    gamma: float = Field(default=0.0, ge=0.0)
    kappa0: float = Field(default=1.0, ge=0.0)
    time_decay: float = Field(default=0.05, ge=0.0, le=1.0)
    noise_std: float = Field(default=0.0, ge=0.0)


class Strategy(BaseModel):
    """
    Strategy / trust-region block:
    - trust: trust in current OSV / surrogate model
    - explore_bias: exploration preference (>0 explore, <0 exploit)
    - priority: global constraint priority descriptor
    """

    trust: float = Field(default=0.2, ge=0.0, le=1.0)
    explore_bias: float = 0.0
    priority: str = "balanced"


class UPVPydantic(BaseModel):
    """
    Pydantic version of UPV (for NL2UPV and Dash main program interaction).

    Note: renamed from ``UPV`` to ``UPVPydantic`` to avoid collision with
    the internal ``core.types.UPV`` dataclass.
    """

    # Goals
    w: List[float]  # weights, len=M
    r: List[float]  # reference point, len=M
    b: List[Tuple[float, float]]  # acceptable band per objective [L, U]

    # Constraints
    C_h: List[ConstraintAtom] = Field(default_factory=list)  # hard constraints
    C_s: List[ConstraintAtom] = Field(default_factory=list)  # soft constraints

    # Domain + Trust
    D: List[Tuple[float, float]] = Field(default_factory=list)  # variable domain
    tau: float = Field(default=0.2, ge=0.0, le=1.0)  # trust region scale

    # Robustness / Strategy
    rho: Robustness = Field(default_factory=Robustness)
    pi: Strategy = Field(default_factory=Strategy)

    # Fields aligned with internal UPV (optional)
    kappa: Dict[str, float] = Field(
        default_factory=lambda: {"hard": 1e-6, "var": 0.2, "robust": 0.2}
    )
    sigma: Dict[str, float] = Field(
        default_factory=lambda: {"w": 0.5, "r": 0.5, "b": 0.5, "D": 0.5}
    )

    # Meta information
    meta: Dict[str, Any] = Field(default_factory=dict)


# =========================================================
#  Utility
# =========================================================
def normalize_weights(w: List[float]) -> List[float]:
    """Normalize a list of non-negative weights to sum to 1."""
    arr = np.array([max(0.0, float(x)) for x in w], dtype=float)
    s = arr.sum()
    if s <= 0:
        arr = np.ones_like(arr) / max(1, len(arr))
    else:
        arr = arr / s
    return arr.tolist()


# =========================================================
#  Regex fallback parser
# =========================================================
def regex_parse_atoms(user_text: str) -> Dict[str, Any]:
    """
    Parse natural language into atoms dict using regex patterns.
    Used as a fallback when the LLM call fails.
    """
    t = user_text.strip().lower()
    out: Dict[str, Any] = {}

    m = re.search(r"w\s*=\s*\[([0-9eE\.\,\s\-]+)\]", t)
    if m:
        out["weights"] = [float(x) for x in m.group(1).split(",")]

    m = re.search(r"r\s*=\s*\[([0-9eE\.\,\s\-]+)\]", t)
    if m:
        out["reference"] = [float(x) for x in m.group(1).split(",")]

    m = re.findall(
        r"band\s*f(\d+)\s*\[\s*([0-9eE\.\-]+)\s*,\s*([0-9eE\.\-]+)\s*\]", t
    )
    if m:
        out["bands"] = [
            {"obj_index": int(i) - 1, "lo": float(lo), "hi": float(hi)}
            for i, lo, hi in m
        ]

    m = re.findall(
        r"(hard|soft)\s+constraint\s*:\s*([xf])(\d+)\s*(<=|>=|==)\s*"
        r"([0-9eE\.\-]+)(?:\s*penalty\s*=\s*([0-9eE\.\-]+))?",
        t,
    )
    if m:
        cons = []
        for hs, xo, idx, sign, val, pen in m:
            cons.append(
                {
                    "kind": "var" if xo == "x" else "obj",
                    "idx": int(idx) - 1,
                    "sign": sign,
                    "value": float(val),
                    "hard": hs == "hard",
                    "penalty": float(pen) if (pen and hs == "soft") else 10.0,
                }
            )
        out["constraints"] = cons

    m = re.search(r"cvar\s*=\s*([0-9]*\.?[0-9]+)", t)
    if m:
        out.setdefault("robustness", {})
        out["robustness"]["cvar_alpha"] = float(m.group(1))

    m = re.search(r"chance\s*=\s*([0-9]*\.?[0-9]+)", t)
    if m:
        out.setdefault("robustness", {})
        out["robustness"]["chance"] = float(m.group(1))

    m = re.search(r"trust\s*=\s*([0-9]*\.?[0-9]+)", t)
    if m:
        out.setdefault("strategy", {})
        out["strategy"]["trust"] = float(m.group(1))

    return out


# =========================================================
#  NL -> atoms (LLM-first, regex-fallback)
# =========================================================
def parse_nl_to_atoms(
    user_text: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    use_rag: bool = True,
) -> Dict[str, Any]:
    """
    Parse natural language into an atoms dict.

    Tries the LLM first (optionally with RAG context); falls back to regex.

    Parameters
    ----------
    user_text : str
        User input in natural language.
    use_rag : bool
        Whether to augment the LLM prompt with RAG retrieval results.
    """
    model = model or DEFAULT_MODEL
    api_key = api_key or API_KEY
    base_url = base_url or DEFAULT_BASE_URL

    # RAG retrieval (G7)
    rag_context = ""
    rag_provenance = []
    if use_rag:
        try:
            from hitlo.nlp.rag_store import get_rag_store
            store = get_rag_store()
            if store.size > 0:
                results = store.retrieve(user_text, top_k=5)
                rag_context = store.format_context(results)
                rag_provenance = [
                    f"[{r.doc.layer}] {r.doc.doc_id}: {r.doc.title} (score={r.score:.2f})"
                    for r in results
                ]
        except Exception as e:
            print(f"[NL] RAG retrieval failed (non-fatal): {e}")

    data: Optional[Dict[str, Any]] = None
    try:
        data = call_openai_atoms(
            user_text, model=model, api_key=api_key, base_url=base_url,
            rag_context=rag_context,
        )
    except Exception as e:
        print("[NL] call_openai_atoms failed, fallback regex:", e)

    if not data:
        data = regex_parse_atoms(user_text)

    if "weights" in data:
        data["weights"] = normalize_weights(data["weights"])

    # Minimal fallback: handle ultra-simple sentences like "w1=0.2"
    if (not data) or (not isinstance(data, dict)):
        m = re.search(r"[wW]\s*1\s*[:=]?\s*([0-9]*\.?[0-9]+)", user_text)
        if m:
            try:
                v = float(m.group(1))
                data = {"weights": [v]}
            except Exception:
                data = {}
        else:
            data = {}

    # Attach RAG provenance if available
    if rag_provenance:
        data["_rag_provenance"] = rag_provenance

    return data


# =========================================================
#  Atoms -> Pydantic UPV
# =========================================================
def atoms_to_upv(atoms: Dict[str, Any], M: int, Nvar: int) -> UPVPydantic:
    """
    Convert an atoms dict (from ``parse_nl_to_atoms``) into a
    ``UPVPydantic`` object.

    Parameters
    ----------
    atoms : dict
        Parsed atoms (weights, reference, bands, constraints, ...).
    M : int
        Number of objectives.
    Nvar : int
        Number of design variables.

    Returns
    -------
    UPVPydantic
    """
    # 1) w
    if isinstance(atoms.get("weights"), list) and atoms["weights"]:
        w = normalize_weights(atoms["weights"])
    else:
        w = [1.0 / max(1, M)] * max(1, M)
    if len(w) != M:
        w = (w + [1.0 / M] * M)[:M]

    # 2) r
    r = atoms.get("reference", [0.0] * M)
    r = (list(r) + [0.0] * M)[:M]
    r = [float(x) for x in r]

    # 3) objective bands b
    bands: List[Tuple[float, float]] = [
        (-float("inf"), float("inf")) for _ in range(M)
    ]
    for band in atoms.get("bands", []) or []:
        try:
            j = int(band.get("obj_index", -1))
            if 0 <= j < M:
                lo = float(band.get("lo", -float("inf")))
                hi = float(band.get("hi", float("inf")))
                bands[j] = (lo, hi)
        except Exception:
            pass

    # 4) constraints
    C_h: List[ConstraintAtom] = []
    C_s: List[ConstraintAtom] = []
    for c in atoms.get("constraints", []) or []:
        try:
            ca = ConstraintAtom(
                kind=c["kind"],
                idx=int(c["idx"]),
                sign=c["sign"],
                value=float(c["value"]),
                hard=bool(c.get("hard", True)),
                penalty=float(c.get("penalty", 10.0)),
            )
            (C_h if ca.hard else C_s).append(ca)
        except Exception as e:
            print("[atoms_to_upv] skip constraint:", e)

    # 5) variable domain: leave empty, filled by problem.bounds in main program
    D: List[Tuple[float, float]] = []

    # 6) robustness / strategy
    rho = (
        Robustness(**(atoms.get("robustness") or {}))
        if "robustness" in atoms
        else Robustness()
    )
    strat = atoms.get("strategy") or {}
    tau = float(strat.get("trust", 0.2))
    pi = Strategy(
        trust=tau,
        explore_bias=float(strat.get("explore_bias", 0.0)),
        priority=str(strat.get("priority", "balanced")),
    )

    # 7) initial kappa / sigma
    kappa = {"hard": 1e-6, "var": rho.kappa0, "robust": rho.kappa0}
    sigma = {"w": 0.5, "r": 0.5, "b": 0.5, "D": 0.5}

    return UPVPydantic(
        w=w,
        r=r,
        b=bands,
        C_h=C_h,
        C_s=C_s,
        D=D,
        tau=tau,
        rho=rho,
        pi=pi,
        kappa=kappa,
        sigma=sigma,
        meta={"source": "NL"},
    )


# =========================================================
#  Apply patch to UPV
# =========================================================
def apply_patch_to_upv(current_upv, patch: Dict[str, Any], M: int):
    """
    Apply an LLM-produced JSON patch incrementally to the **current** UPV.

    Supported operations:
      - set_weight(index, value)
      - set_cvar(value)
      - set_chance(value)
      - set_gamma(value)
      - set_decay(value)         -> writes to rho['time_decay']
      - set_init_penalty(value)  -> writes to kappa['var'] / kappa['robust']
      - set_trust(value)
      - set_priority(value)      -> writes to pi['priority']
      - set_reference(values[])
      - set_band(obj_index, lo, hi)
      - add_constraint(kind, idx, sign, value, hard, penalty)

    Parameters
    ----------
    current_upv : UPVPydantic or compatible object
        The UPV to modify in-place.
    patch : dict
        Patch dict with ``ops`` list.
    M : int
        Number of objectives.

    Returns
    -------
    The modified ``current_upv``.
    """
    if current_upv is None:
        raise ValueError("current_upv is None")

    # Editable copies
    w = [
        float(x)
        for x in (
            list(current_upv.w)
            if hasattr(current_upv.w, "__iter__")
            else [current_upv.w]
        )
    ]
    r = [
        float(x)
        for x in (
            list(current_upv.r)
            if getattr(current_upv, "r", None) is not None
            else [0.0] * M
        )
    ]
    b = (
        [tuple(map(float, bi)) for bi in current_upv.b]
        if getattr(current_upv, "b", None)
        else [(-float("inf"), float("inf")) for _ in range(M)]
    )

    tau = float(getattr(current_upv, "tau", 0.2))
    rho = dict(
        getattr(current_upv, "rho", {"cvar_alpha": 0.9, "chance": 0.9})
    )
    kappa = dict(
        getattr(
            current_upv, "kappa", {"hard": 1e-6, "var": 0.2, "robust": 0.2}
        )
    )
    pi = dict(
        getattr(
            current_upv, "pi", {"explore_bias": 0.0, "priority": "balanced"}
        )
    )

    C_h = list(current_upv.C_h)
    C_s = list(current_upv.C_s)

    # Size alignment
    if len(w) != M:
        w = (w + [1.0 / M] * M)[:M]
    if len(r) != M:
        r = (r + [0.0] * M)[:M]
    if len(b) != M:
        b = (b + [(-float("inf"), float("inf"))] * M)[:M]

    def _norm(arr: List[float]) -> List[float]:
        arr = [max(0.0, float(x)) for x in arr]
        s = sum(arr)
        return [x / s for x in arr] if s > 0 else [1.0 / M] * M

    # Apply patch operations
    for op in patch.get("ops", []):
        t = op.get("op")
        if t == "set_weight":
            idx = int(op.get("index", -1))
            val = float(op.get("value", 0.0))
            if 0 <= idx < M:
                w[idx] = max(0.0, val)

        elif t == "set_cvar":
            rho["cvar_alpha"] = float(
                op.get("value", rho.get("cvar_alpha", 0.9))
            )

        elif t == "set_chance":
            rho["chance"] = float(op.get("value", rho.get("chance", 0.9)))

        elif t == "set_gamma":
            rho["gamma"] = float(op.get("value", rho.get("gamma", 0.0)))

        elif t == "set_decay":
            rho["time_decay"] = float(
                op.get("value", rho.get("time_decay", 0.05))
            )

        elif t == "set_init_penalty":
            v = float(op.get("value", kappa.get("var", 0.2)))
            kappa["var"] = v
            kappa["robust"] = v

        elif t == "set_trust":
            tau = float(op.get("value", tau))

        elif t == "set_priority":
            pi["priority"] = str(
                op.get("value", pi.get("priority", "balanced"))
            )

        elif t == "set_reference":
            vals = op.get("values", [])
            if isinstance(vals, list) and len(vals) > 0:
                r = (list(map(float, vals)) + [0.0] * M)[:M]

        elif t == "set_band":
            j = int(op.get("obj_index", -1))
            lo = op.get("lo", -float("inf"))
            hi = op.get("hi", float("inf"))
            try:
                lo = (
                    -float("inf")
                    if str(lo).lower() in ("-inf", "-infinity")
                    else float(lo)
                )
                hi = (
                    float("inf")
                    if str(hi).lower() in ("inf", "infinity")
                    else float(hi)
                )
            except Exception:
                lo, hi = -float("inf"), float("inf")
            if 0 <= j < M:
                b[j] = (lo, hi)

        elif t == "add_constraint":
            kind = op.get("kind")
            idx = int(op.get("idx", -1))
            sign = op.get("sign")
            val = float(op.get("value", 0.0))
            hard = bool(op.get("hard", True))
            pen = float(op.get("penalty", 10.0))
            if (
                kind in ("obj", "var")
                and sign in (">=", "<=", "==")
                and idx >= 0
            ):
                ca = ConstraintAtom(
                    kind=kind,
                    idx=idx,
                    sign=sign,
                    value=val,
                    hard=hard,
                    penalty=pen,
                )
                (C_h if hard else C_s).append(ca)

    # Normalize weights
    w = _norm(w)

    # Write back to current_upv
    current_upv.w = np.asarray(w, dtype=float)
    current_upv.r = np.asarray(r, dtype=float)
    current_upv.b = [(float(lo), float(hi)) for (lo, hi) in b]
    current_upv.tau = float(tau)
    current_upv.rho = dict(rho)
    current_upv.kappa = dict(kappa)
    current_upv.pi = dict(pi)
    current_upv.C_h = C_h
    current_upv.C_s = C_s
    return current_upv
