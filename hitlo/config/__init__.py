"""
hitlo.config
===============
Configuration sub-package for the HITLO platform.

Public API
----------
* ``AppConfig`` / ``ExperimentConfig`` — typed configuration dataclasses.
* ``load_config(path)`` — load YAML + env-var overrides into an ``AppConfig``.
* ``CO_THRESH`` / ``CO_THRESH_STRICT`` / ``CO_THRESH_FEASIBLE`` — constraint
  threshold dictionaries for the CO-ARCH problem.
* Color constants from ``color_scheme`` (e.g. ``PRIMARY``, ``PLOTLY_COLORS``).
"""

from hitlo.config.settings import (
    AppConfig,
    ExperimentConfig,
    load_config,
    CO_THRESH,
    CO_THRESH_STRICT,
    CO_THRESH_FEASIBLE,
)

from hitlo.config.color_scheme import (
    PRIMARY,
    PRIMARY_LIGHT,
    ACCENT_TEAL,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    ACCENT_RED,
    ACCENT_GREEN,
    BG_PAGE,
    BG_CARD,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    PLOTLY_COLORS,
)

__all__ = [
    # Config classes & loader
    "AppConfig",
    "ExperimentConfig",
    "load_config",
    # CO thresholds
    "CO_THRESH",
    "CO_THRESH_STRICT",
    "CO_THRESH_FEASIBLE",
    # Colors
    "PRIMARY",
    "PRIMARY_LIGHT",
    "ACCENT_TEAL",
    "ACCENT_ORANGE",
    "ACCENT_PURPLE",
    "ACCENT_RED",
    "ACCENT_GREEN",
    "BG_PAGE",
    "BG_CARD",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "PLOTLY_COLORS",
]
