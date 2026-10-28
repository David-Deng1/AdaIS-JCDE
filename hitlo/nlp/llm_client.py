# -*- coding: utf-8 -*-
"""
hitlo.nlp.llm_client
~~~~~~~~~~~~~~~~~~~~~~
LLM HTTP client utilities for NL2UPV.

- Communicates with OpenAI-compatible endpoints (e.g. Spark).
- API key loaded from env var ``LEHITLO_LLM_API_KEY`` with fallback.
- Provides ``call_openai_atoms`` (full UPV) and ``llm_nl_to_patch`` (incremental patch).
"""

import os
import json
import re
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

from .prompt_templates import build_atoms_prompt, build_patch_prompt

# =========================================================
#  Configuration (lazy-loaded from config.yaml → env var → fallback)
# =========================================================
API_KEY: str = ""
DEFAULT_BASE_URL: str = ""
DEFAULT_MODEL: str = ""
DEBUG_LLM: bool = True

_config_loaded: bool = False


def _ensure_config() -> None:
    """Lazy-load LLM configuration.

    Priority (highest wins): env var > config.yaml > hardcoded fallback.
    Called automatically before every LLM request.
    """
    global _config_loaded, API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL
    if _config_loaded:
        return
    _config_loaded = True

    # 1) Read from config.yaml via AppConfig
    try:
        from hitlo.config.settings import load_config
        cfg = load_config()
        API_KEY = cfg.llm_api_key or ""
        DEFAULT_BASE_URL = cfg.llm_base_url or ""
        DEFAULT_MODEL = cfg.llm_model or ""
    except Exception as e:
        print(f"[LLM] load_config failed (non-fatal): {e}")

    # 2) Env var override (highest priority)
    env_key = os.environ.get("LEHITLO_LLM_API_KEY", "")
    if env_key:
        API_KEY = env_key

    # 3) Hardcoded fallback if still empty
    if not DEFAULT_BASE_URL:
        DEFAULT_BASE_URL = "https://spark-api-open.xf-yun.com/v1"
    if not DEFAULT_MODEL:
        DEFAULT_MODEL = "generalv3.5"

    if DEBUG_LLM:
        print(f"[LLM] config loaded: model={DEFAULT_MODEL}, "
              f"base_url={DEFAULT_BASE_URL[:40]}..., "
              f"api_key={'***' + API_KEY[-4:] if len(API_KEY) > 4 else '(empty)'}")


# =========================================================
#  Core HTTP helper
# =========================================================
def http_chat(
    payload: Dict[str, Any],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Send a chat-completions request to an OpenAI-compatible endpoint.

    Parameters
    ----------
    payload : dict
        Request body (must contain ``messages``).
    model : str, optional
        Model identifier. Defaults to ``DEFAULT_MODEL``.
    api_key : str, optional
        Bearer token. Defaults to ``API_KEY``.
    base_url : str, optional
        API root (should end with ``/v1`` or it will be appended).

    Returns
    -------
    dict or None
        Parsed JSON response, or *None* on failure.
    """
    _ensure_config()
    api_key = api_key or API_KEY
    base_url = base_url or DEFAULT_BASE_URL
    model = model or DEFAULT_MODEL
    if not api_key:
        print("[LLM] api_key is empty")
        return None

    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    url = base_url.rstrip("/") + "/chat/completions"

    payload = dict(payload)
    payload.setdefault("model", model)
    payload.setdefault("temperature", 0)

    if DEBUG_LLM:
        print("\n" + "=" * 60)
        print("[LLM] >>>  REQUEST  >>>")
        print(f"  model: {payload.get('model', '(default)')}")
        print(f"  temperature: {payload.get('temperature', '(default)')}")
        for msg in payload.get("messages", []):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            print(f"  [{role}] {content}")
        print("=" * 60)

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            obj = json.loads(resp.read().decode("utf-8", errors="ignore"))
        if DEBUG_LLM:
            # 提取 LLM 回复内容
            llm_content = "(parse error)"
            try:
                llm_content = obj["choices"][0]["message"]["content"]
            except Exception:
                llm_content = json.dumps(obj, ensure_ascii=False)[:500]
            # 提取 usage 信息
            usage = obj.get("usage", {})
            usage_str = (
                f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
                f"completion_tokens={usage.get('completion_tokens', '?')}, "
                f"total_tokens={usage.get('total_tokens', '?')}"
            ) if usage else "(no usage info)"
            print("\n" + "-" * 60)
            print("[LLM] <<<  RESPONSE  <<<")
            print(f"  model: {obj.get('model', '?')}")
            print(f"  usage: {usage_str}")
            print(f"  content:\n{llm_content}")
            print("-" * 60 + "\n")
        return obj
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print("[LLM] HTTPError:", e.code, body[:300])
    except Exception as e:
        print("[LLM] Request error:", e)
    return None


# =========================================================
#  JSON extraction
# =========================================================
def extract_json_block(s: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first JSON object from a string that may contain
    markdown fences, surrounding prose, etc.

    Returns
    -------
    dict or None
    """
    if not isinstance(s, str):
        return None
    # Try ```json ... ``` fenced block first
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", s, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try outermost braces
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        frag = s[i : j + 1]
        try:
            return json.loads(frag)
        except Exception:
            pass
    # Try raw parse
    try:
        return json.loads(s)
    except Exception:
        return None


# =========================================================
#  High-level: NL -> atoms (full UPV)
# =========================================================
def call_openai_atoms(
    user_text: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    rag_context: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Call the LLM to convert natural language into a full atoms/UPV JSON.

    The returned dict may contain keys such as ``weights``, ``reference``,
    ``bands``, ``constraints``, ``robustness``, ``strategy``.

    Parameters
    ----------
    rag_context : str
        Optional RAG retrieval context to prepend to the prompt.

    Returns
    -------
    dict or None
    """
    # Avoid circular import: normalize_weights lives in nl2upv
    from .nl2upv import normalize_weights

    # Build system message, optionally with RAG context
    system_msg = (
        "You convert Chinese natural language into strict JSON "
        "describing a multi-objective preference model (UPV)."
    )
    if rag_context:
        system_msg += (
            "\n\n以下是与用户输入相关的领域知识，请用于消歧、确定槽位映射和提供证据：\n"
            + rag_context
        )

    payload = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": build_atoms_prompt(user_text)},
        ]
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

    # 1-based -> 0-based & normalize weights
    if "bands" in data:
        for b in data["bands"]:
            b["obj_index"] = int(b["obj_index"]) - 1
    if "constraints" in data:
        for c in data["constraints"]:
            c["idx"] = int(c["idx"]) - 1
    if "weights" in data:
        data["weights"] = normalize_weights(data["weights"])
    return data


# =========================================================
#  High-level: NL -> patch (incremental modification)
# =========================================================
def llm_nl_to_patch(
    user_text: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    dialogue_context: str = "",
    opt_context: str = "",
) -> Dict[str, Any]:
    """
    Call the LLM to produce a JSON patch for incrementally modifying
    the current UPV.

    Supported ops:
      - set_weight / set_cvar / set_chance / set_gamma / set_decay
      - set_init_penalty / set_trust / set_priority
      - set_reference / set_band / add_constraint

    Returns
    -------
    dict
        Patch object with ``mode`` == ``"patch"`` and ``ops`` list.

    Raises
    ------
    RuntimeError
        If the LLM call fails or the response cannot be parsed.
    """
    _ensure_config()
    api_key = api_key or API_KEY
    base_url = base_url or DEFAULT_BASE_URL
    model = model or DEFAULT_MODEL
    if not api_key:
        raise RuntimeError("LLM API Key is empty. Set LEHITLO_LLM_API_KEY env var.")

    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    # Build system message, optionally with dialogue/opt context
    system_content = (
        "You convert Chinese natural language into strict JSON "
        "patches for a multi-objective Unified Preference Vector (UPV)."
    )
    if dialogue_context or opt_context:
        if dialogue_context:
            system_content += f"\n\n{dialogue_context}"
        if opt_context:
            system_content += f"\n\n{opt_context}"
        system_content += (
            "\n\nPlease use the above context to resolve references like "
            "'the previous constraint' or 'that weight'. "
            "Consider the current optimization state when interpreting intent."
        )

    # Use contextual prompt if context is available
    if dialogue_context or opt_context:
        from .prompt_templates import build_contextual_patch_prompt
        user_prompt = build_contextual_patch_prompt(
            user_text, dialogue_context, opt_context
        )
    else:
        user_prompt = build_patch_prompt(user_text)

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ],
    }

    obj = http_chat(payload, model=model, api_key=api_key, base_url=base_url)
    if obj is None:
        raise RuntimeError(
            "LLM patch call failed; no fallback here (logic bug or network)."
        )

    try:
        content = obj["choices"][0]["message"]["content"]
    except Exception:
        content = obj.get("content") or json.dumps(obj)

    data = extract_json_block(content)
    if data is None:
        raise RuntimeError("LLM patch response cannot be parsed as JSON")

    if "mode" not in data:
        data["mode"] = "patch"
    if "ops" not in data or not isinstance(data["ops"], list):
        data["ops"] = []

    # 1-based -> 0-based and fallback normalization
    for op in data["ops"]:
        if op.get("op") == "set_band":
            if "obj_index" in op:
                op["obj_index"] = int(op["obj_index"])
        if op.get("op") == "add_constraint":
            if "idx" in op:
                op["idx"] = int(op["idx"])
            if "kind" in op and op["kind"] not in ("var", "obj"):
                op["kind"] = "obj"

    return data
