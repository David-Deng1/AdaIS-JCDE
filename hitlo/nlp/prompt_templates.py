# -*- coding: utf-8 -*-
"""
hitlo.nlp.prompt_templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prompt templates used by the LLM client for NL-to-UPV conversion.

- ``build_atoms_prompt``: full UPV construction from natural language.
- ``build_patch_prompt``: incremental UPV patch from natural language.
"""

import json


def build_atoms_prompt(user_text: str) -> str:
    """
    Build the LLM prompt for converting natural language into a
    complete atoms/UPV JSON.

    Parameters
    ----------
    user_text : str
        Chinese natural language describing optimization preferences.

    Returns
    -------
    str
        Formatted prompt string.
    """
    schema = {
        "weights": "array len M, non-negative, sum->1 (omit if absent)",
        "reference": "array len M (omit if absent)",
        "bands": "list of {obj_index:int(1-based), lo:float, hi:float}",
        "constraints": (
            "list of {kind:'var'|'obj', idx:int(1-based), sign:'<='|'>='|'==', "
            "value:float, hard:bool, penalty:float(optional)}"
        ),
        "robustness": (
            "{cvar_alpha:float(0,1], chance:float(0,1], tail_q:float(0,1], "
            "gamma:float>=0, kappa0:float>=0, time_decay:float[0,1], noise_std:float>=0}"
        ),
        "strategy": (
            "{trust:float(0,1], explore_bias:float(optional), priority:str(optional)}"
        ),
    }
    prompt = f"""
You are an expert multi-objective optimization assistant.
Convert the following Chinese natural language into a STRICT JSON object that describes
an Unified Preference Vector (UPV).

Return ONLY a JSON object, no extra text.

Fields (omit missing ones):
{json.dumps(schema, indent=2, ensure_ascii=False)}

Rules:
- Convert all indices from 1-based to 0-based in JSON.
- Normalize weights to sum to 1 if provided.
- '必须/强制/shall' => hard constraint; '希望/尽量/最好/prefer' => soft constraint with a reasonable penalty.
- '机会约束/概率约束/大多数情形' => robustness.chance or tail_q
- '最坏情况/尾部风险/CVaR' => robustness.cvar_alpha / gamma / kappa0
- '越保守' => larger gamma or larger cvar_alpha; '越激进' => smaller gamma or cvar_alpha.

User text:
\"\"\"{user_text}\"\"\"
"""
    return prompt.strip()


def build_contextual_patch_prompt(
    user_text: str,
    dialogue_context: str = "",
    opt_context: str = "",
) -> str:
    """Build the LLM prompt for patch generation with dialogue + optimization context.

    Parameters
    ----------
    user_text : str
        Chinese natural language describing the desired change.
    dialogue_context : str
        Formatted dialogue history from DialogueMemory.to_prompt_text().
    opt_context : str
        Formatted optimization state from OptContextSummary.to_prompt_text().

    Returns
    -------
    str
        Formatted prompt string with context sections.
    """
    base = build_patch_prompt(user_text)

    context_sections = []
    if dialogue_context:
        context_sections.append(dialogue_context)
    if opt_context:
        context_sections.append(opt_context)

    if not context_sections:
        return base

    context_block = "\n\n".join(context_sections)
    # Insert context before the user text section
    insertion = (
        f"\n\n[Context]\n{context_block}\n"
        "\nPlease consider the above dialogue history and optimization state "
        "when interpreting the user's intent. Resolve anaphora like "
        "'the previous constraint', 'that weight', etc. using the dialogue history.\n"
    )
    # Insert before "User text:"
    return base.replace(
        'User text:',
        insertion + 'User text:',
    )


def build_patch_prompt(user_text: str) -> str:
    """
    Build the LLM prompt for producing a JSON patch that modifies the
    current UPV incrementally.

    Parameters
    ----------
    user_text : str
        Chinese natural language describing the desired change.

    Returns
    -------
    str
        Formatted prompt string.
    """
    schema = {
        "mode": '"patch"',
        "ops": [
            {"op": "set_weight", "index": 0, "value": 0.2},
            {"op": "set_cvar", "value": 0.9},
            {"op": "set_chance", "value": 0.9},
            {"op": "set_gamma", "value": 0.1},
            {"op": "set_decay", "value": 0.05},
            {"op": "set_init_penalty", "value": 5.0},
            {"op": "set_trust", "value": 0.3},
            {"op": "set_priority", "value": "safe-first"},
            {"op": "set_reference", "values": [0.2, 0.6]},
            {"op": "set_band", "obj_index": 1, "lo": -1.0, "hi": 2.0},
            {
                "op": "add_constraint",
                "kind": "obj",
                "idx": 1,
                "sign": ">=",
                "value": 0.5,
                "hard": False,
                "penalty": 20.0,
            },
        ],
    }
    prompt = f"""
You are an expert multi-objective optimization assistant.
Given the user's Chinese natural language, output a STRICT JSON patch that modifies
the current Unified Preference Vector (UPV).

Patch JSON schema (example):
{json.dumps(schema, indent=2, ensure_ascii=False)}

Rules:
- indices in JSON are 0-based.
- If user only mentions "increase/decrease weight of f1", use set_weight with a reasonable value.
- '提高鲁棒性/更保守' => increase robustness.cvar_alpha or gamma via set_cvar / set_gamma / set_decay.
- '放宽概率约束' => decrease robustness.chance via set_chance.
- '惩罚更重/更宽松' => adjust set_init_penalty.
- '信任当前模型/收缩搜索范围' => increase set_trust.
Return ONLY the JSON patch object.
User text:
\"\"\"{user_text}\"\"\"
"""
    return prompt.strip()
