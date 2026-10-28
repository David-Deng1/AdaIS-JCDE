"""
hitlo.config.settings
========================
Central application configuration for the HITLO platform.

* ``AppConfig`` is a frozen dataclass with sensible defaults that can be
  overridden by a YAML file and / or environment variables.
* ``load_config()`` is the single entry-point used at startup.

Priority (highest wins): environment variables > YAML file > dataclass defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None

# ---------------------------------------------------------------------------
# CO constraint thresholds (ported from app_v11_iqlfix.py L591-618)
# ---------------------------------------------------------------------------

CO_THRESH_STRICT: Dict[str, Any] = {
    "V_min": 120.0,
    "Vavg_min": 90.0,
    "R_min": 320.0,
    "H_min": 18.0,
    "Q_max": 70.0,
    "N_max": 6.0,
    "E_max": 0.8,
    "P_min": 0.6,
    "w_cost": [0.4, 0.3, 0.3, 0.1],  # C_pre, C_small, C_large, C_dev
}

# Feasible thresholds: relaxed bounds that yield a non-empty feasible set
# while retaining moderate difficulty on the toy CO-ARCH model.
CO_THRESH_FEASIBLE: Dict[str, Any] = {
    "V_min": 60.0,       # max_speed >= 60
    "Vavg_min": 55.0,    # avg_speed >= 55
    "R_min": 180.0,      # range_cov >= 180
    "H_min": 14.0,       # max_alt >= 14
    "Q_max": 70.0,       # q_max <= 70   (unchanged)
    "N_max": 6.0,        # n_max <= 6    (unchanged)
    "E_max": 0.8,        # terminal_err <= 0.8 (unchanged)
    "P_min": 0.55,       # P_success >= 0.55
    "w_cost": [0.4, 0.3, 0.3, 0.1],
}

# Default active threshold set (switch to CO_THRESH_STRICT.copy() for strict mode)
CO_THRESH: Dict[str, Any] = CO_THRESH_FEASIBLE.copy()


# ---------------------------------------------------------------------------
# ExperimentConfig (ported from app_v11_iqlfix.py L1164-1291)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Per-run experiment configuration aligned with paper method groups / ablations.

    Method groups (paper compared methods):
        A1_FIXED_SINGLE, A2_FIXED_PORTFOLIO, A3_UPV_DIRECT,
        B1_RULE_OSV, C1_PPCS_IQL_FULL, C2_PPCS_IQL_NO_SAFETY

    Ablations (paper Abl-1 ~ Abl-5):
        ABL_NO_MODEL_RECONF, ABL_NO_PARAM, ABL_NO_SCHEDULING,
        ABL_NO_PROCESS_FEATURES, ABL_NO_GATING, ABL_NO_SAFETY
    """

    # -- method / ablation --
    exp_group: str = "C1_PPCS_IQL_FULL"
    ablations: List[str] = field(default_factory=list)

    # -- reproducibility --
    seed_base: int = 0
    n_reps: int = 1

    # -- budget / horizon --
    horizon_T: int = 80
    epoch_budget_B: int = 50

    # -- fixed / baseline solver configuration --
    fixed_single_solver: str = "NSGA-II"
    upv_direct_solver: str = "NSGA-III"
    upv_direct_roi_bias: float = 0.85

    # -- reward + HV smoothing --
    hv_beta: float = 0.9
    hv_eps: float = 1e-4
    lambda_c: float = 1.0
    lambda_p: float = 1.0
    lambda_b: float = 0.2
    lambda_q: float = 0.1

    # -- legacy HV-oriented diagnostic switches --
    # Formal Ch5 runs keep these disabled so C1 remains
    # rule-prior + IQL residual + safety projection.
    reward_hv_scale: float = 1.20
    hv_boost_enabled: bool = False
    hv_boost_sat_th: float = 0.50
    hv_boost_feas_th: float = 0.55
    hv_boost_viol_th: float = 1.0
    hv_boost_strength: float = 0.80
    hv_boost_floor: float = 0.12
    hv_boost_cp_suppress: float = 0.0
    hv_boost_tr_min: float = 0.25
    hv_boost_bias_max: float = 0.05
    hv_boost_disable_band: bool = True
    hv_boost_algos: List[str] = field(
        default_factory=lambda: ["CP/MILP", "NSGA-III", "MOEA/D", "NSGA-II"]
    )

    # -- legacy CO-ARCH trigger parameters --
    hv_boost_ignore_sat_on_coarch: bool = True
    hv_boost_coarch_feas_th: float = 0.90
    hv_boost_coarch_viol_th: float = 0.50

    # -- feasible elite archive & rescue --
    feasible_archive_enabled: bool = True
    feasible_archive_max_factor: float = 5.0
    feasible_rescue_th: float = 0.10
    rollback_on_zero_feas: bool = True
    hv_use_archive: bool = True

    # -- legacy HV block scheduling --
    hv_block_enabled: bool = False
    hv_block_len: int = 3
    hv_block_topk: int = 2
    hv_block_main_share: float = 0.85

    # -- offline RL (IQL) --
    offline_min_transitions: int = 1000
    offline_train_steps: int = 5000
    offline_train_ratio: float = 0.8
    iql_expectile: float = 0.7
    iql_beta: float = 3.0
    iql_gamma: float = 0.98
    offline_collect_noise: float = 0.12

    # -- IQL checkpoint / evaluation mode --
    iql_ckpt_path: str = ""
    iql_freeze: bool = False

    # -- allocation + gating --
    alloc_temp: float = 3.0
    op_noop_bias: float = 2.0
    query_budget: float = 1.0

    # -- OOD fallback --
    ood_k: int = 10
    # kNN support distance in z-scored high-dimensional state space.
    # Calibrated to avoid degenerate always-fallback behavior.
    ood_threshold: float = 8.0
    ood_enabled: bool = True

    # -- profiles / regimes (for auto) --
    profile_id: int = 0
    regime_id: int = 0
    eval_scenario: str = "seen"

    # -- run mode / metric definition --
    run_mode: str = "eval"
    sat_use_archive: bool = True

    # -- IQL policy improvement (C1) --
    iql_pi_enabled: bool = True
    iql_pi_include_baselines: bool = True
    iql_pi_include_mix: bool = True
    iql_pi_mix: float = 0.5
    iql_residual_accept_enabled: bool = True
    iql_residual_accept_margin: float = 0.0
    iql_residual_accept_require_safe: bool = True
    iql_residual_accept_channel_guard_enabled: bool = True
    iql_residual_accept_channel_tol: float = 0.05

    # -- termination / convergence --
    max_evals: int = 5000
    max_wall_time: float = 0
    hv_stagnation_window: int = 10
    hv_stagnation_eps: float = 0.001
    igd_stagnation_eps: float = 0.005
    convergence_patience: int = 3
    auto_stop_on_budget: bool = True
    auto_stop_on_convergence: bool = False


# ---------------------------------------------------------------------------
# AppConfig — top-level application settings
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Top-level application configuration.

    Loaded from ``defaults.yaml`` (or a user-supplied YAML), then selectively
    overridden by environment variables prefixed with ``LEHITLO_``.
    """

    # -- paths --
    problem_dir: str = "problems"

    # -- LLM --
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"

    # -- server --
    host: str = "0.0.0.0"
    port: int = 7860
    debug: bool = False

    # -- NL validation pipeline (Ch3 §3.3) --
    n_sample_mode: str = "balanced"       # "fast"(N=1) / "balanced"(N=3) / "precise"(N=5)
    n_sample_temperature: float = 0.5     # LLM sampling temperature for N>1
    nl_gamma_threshold: float = 0.5       # confidence threshold for auto-apply

    # -- experiment defaults --
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # -- CO thresholds (active set) --
    co_thresh: Dict[str, Any] = field(default_factory=lambda: CO_THRESH.copy())


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_ENV_PREFIX = "LEHITLO_"

# Fields that should be interpreted as specific Python types when read from
# environment variables (everything else stays as ``str``).
_TYPE_MAP = {
    "port": int,
    "debug": lambda v: v.lower() in ("1", "true", "yes"),
}


def _apply_env(cfg: AppConfig) -> None:
    """Override *flat* AppConfig fields with ``LEHITLO_*`` env vars."""
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(_ENV_PREFIX):
            continue
        field_name = env_key[len(_ENV_PREFIX):].lower()
        if hasattr(cfg, field_name):
            converter = _TYPE_MAP.get(field_name, str)
            object.__setattr__(cfg, field_name, converter(env_val))


def _dict_to_experiment(d: Dict[str, Any]) -> ExperimentConfig:
    """Build an ``ExperimentConfig`` from a (possibly partial) dict."""
    known = {f.name for f in ExperimentConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in d.items() if k in known}
    return ExperimentConfig(**filtered)


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load configuration with precedence: env vars > YAML > defaults.

    Parameters
    ----------
    path : str
        Path to a YAML configuration file.  If the file does not exist the
        pure dataclass defaults (+ env overrides) are returned without error.

    Returns
    -------
    AppConfig
    """
    data: Dict[str, Any] = {}

    # --- 1. Read YAML (if present) ---
    config_path = Path(path)
    if not config_path.is_absolute():
        # Also try next to the package root
        pkg_dir = Path(__file__).resolve().parent.parent.parent
        alt = pkg_dir / config_path
        if alt.exists():
            config_path = alt

    if config_path.exists() and yaml is not None:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    # --- 2. Build AppConfig from YAML data ---
    experiment_data = data.pop("experiment", {})
    co_thresh_data = data.pop("co_thresh", None)

    # Filter to known AppConfig fields only
    known_app = {f.name for f in AppConfig.__dataclass_fields__.values()}
    app_kwargs: Dict[str, Any] = {k: v for k, v in data.items() if k in known_app}

    if experiment_data:
        app_kwargs["experiment"] = _dict_to_experiment(experiment_data)

    if co_thresh_data is not None:
        app_kwargs["co_thresh"] = co_thresh_data

    cfg = AppConfig(**app_kwargs)

    # --- 3. Apply environment variable overrides ---
    _apply_env(cfg)

    return cfg
