# -*- coding: utf-8 -*-
"""
hitlo.nlp
~~~~~~~~~~~
NLP module for the HITLO framework.

Provides natural-language-to-UPV conversion (NL2UPV), including:
- LLM-based and regex-based atom parsing
- Atom-to-UPV construction
- Patch-based incremental UPV modification
- Prompt templates and LLM client utilities
"""

from .nl2upv import parse_nl_to_atoms, atoms_to_upv, UPVPydantic, ConstraintAtom
from .llm_client import http_chat, extract_json_block
from .prompt_templates import build_atoms_prompt, build_patch_prompt
from .validation_pipeline import run_pipeline, PipelineOutput, ValidationResult
