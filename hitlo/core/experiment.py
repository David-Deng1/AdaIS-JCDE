"""
hitlo.core.experiment
=======================
Experiment configuration dataclass and paper method-ID helpers.

Extracted from the monolithic ``app_v11_iqlfix.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

__all__ = [
    "ExperimentConfig",
    "PAPER_METHOD_IDS",
    "normalize_method_id",
]


# ------------------------------------------------------------------ #
#  Paper method IDs & aliases
# ------------------------------------------------------------------ #

# 论文方法 ID：兼容旧版组别命名（G0~G4）
PAPER_METHOD_IDS = [
    "A1_FIXED_SINGLE",
    "A2_FIXED_PORTFOLIO",
    "A3_UPV_DIRECT",
    "B1_RULE_OSV",
    "C1_PPCS_IQL_FULL",
    "C2_PPCS_IQL_NO_SAFETY",
]

_PAPER_METHOD_ALIASES = {
    # 旧版 exp_group -> 论文 method
    "G0_BASE_NO_PREF": "A1_FIXED_SINGLE",
    "G1_UPV_ONLY":     "A3_UPV_DIRECT",
    "G2_OSV_RULE":     "B1_RULE_OSV",
    "G3_OSV_AWR":      "C1_PPCS_IQL_FULL",
    "G4_OSV_IQL":      "C1_PPCS_IQL_FULL",
    # 兼容你论文中可能出现的写法
    "A1": "A1_FIXED_SINGLE",
    "A2": "A2_FIXED_PORTFOLIO",
    "A3": "A3_UPV_DIRECT",
    "B1": "B1_RULE_OSV",
    "C1": "C1_PPCS_IQL_FULL",
    "C2": "C2_PPCS_IQL_NO_SAFETY",
}


def normalize_method_id(exp_group: str) -> str:
    """把 UI/脚本给的 exp_group 统一成论文 method ID。"""
    g = (exp_group or "").strip()
    return _PAPER_METHOD_ALIASES.get(g, g)


# ------------------------------------------------------------------ #
#  ExperimentConfig dataclass
# ------------------------------------------------------------------ #

# [EXP] 试验配置（提前定义，供安全/门控函数使用）
@dataclass
class ExperimentConfig:
    # ===== 与论文实验设计对齐的 method/ablation 入口 =====
    # exp_group: 论文中对比方法/组别的 ID（A1~C2），同时兼容旧版 G0~G4 的别名
    # ablations: Abl-1~Abl-5（以及一些开发期 ablation）

    # -------- 方法（paper compared methods）--------
    # A1_FIXED_SINGLE, A2_FIXED_PORTFOLIO, A3_UPV_DIRECT,
    # B1_RULE_OSV, C1_PPCS_IQL_FULL, C2_PPCS_IQL_NO_SAFETY
    exp_group: str = "C1_PPCS_IQL_FULL"

    # -------- Ablations（论文 Abl-1~Abl-5）--------
    # ABL_NO_MODEL_RECONF  -> Abl-1 No reconfiguration (disable operator library R1~R5 + model reconfiguration)
    # ABL_NO_PARAM         -> Abl-2 No parameter adaptation
    # ABL_NO_SCHEDULING    -> Abl-3 No solver mixture (single fixed solver)
    # ABL_NO_PROCESS_FEATURES -> Abl-4 No process features F(t)
    # ABL_NO_GATING        -> Abl-5 No uncertainty/conflict gating
    # ABL_NO_SAFETY        -> (C2) No safety projector Pi_safe
    # ABL_CLIP_ONLY        -> Projection ablation: numeric clipping only
    # ABL_NO_CONSISTENCY_PROJ -> Projection ablation: no semantic consistency constraints
    # ABL_NO_SUPPORT_CHECK -> OOD/support ablation: record support distance but do not fallback
    # ABL_NO_TEMPORAL_SMOOTH -> Projection ablation: disable temporal smoothing
    # ABL_IQL_DIRECT      -> Offline-RL ablation: IQL action without rule-residual anchoring
    # ABL_NO_UPV          -> Offline-RL ablation: neutralize UPV/OSV preference state
    ablations: List[str] = field(default_factory=list)

    # -------- 复现实验 --------
    seed_base: int = 0
    n_reps: int = 1

    # -------- 预算 / 时域 --------
    # horizon_T: A2 的 stage schedule 需要用到；auto 脚本也会用它做批量 run。
    horizon_T: int = 80
    epoch_budget_B: int = 50

    # -------- 固定/基线用的 solver 配置 --------
    fixed_single_solver: str = "NSGA-II"
    upv_direct_solver: str = "NSGA-III"
    upv_direct_roi_bias: float = 0.85

    # -------- reward + HV smoothing --------
    hv_beta: float = 0.9
    hv_eps: float = 1e-4
    lambda_c: float = 1.0
    lambda_p: float = 1.0
    lambda_b: float = 0.2
    lambda_q: float = 0.1

    # -------- legacy HV-oriented diagnostic switches --------
    # Formal Ch5 experiments must keep these disabled. C1 is defined as
    # rule-prior anchoring + IQL residual correction + consistency/safety projection,
    # not as a method-specific HV booster.
    reward_hv_scale: float = 1.20

    hv_boost_enabled: bool = False
    hv_boost_sat_th: float = 0.50
    hv_boost_feas_th: float = 0.55
    hv_boost_viol_th: float = 1.0
    hv_boost_strength: float = 0.80          # 对 HV-friendly 算法的 alpha 乘以 (1+strength)
    hv_boost_floor: float = 0.12             # HV-friendly 算法的 simplex floor
    hv_boost_cp_suppress: float = 0.0       # 对 CP/MILP 的 alpha 乘以 (1-cp_suppress)
    hv_boost_tr_min: float = 0.25            # trust region 最小缩放（仅 hv_boost 时）
    hv_boost_bias_max: float = 0.05          # preference shaping 的上限（仅 hv_boost 时）
    hv_boost_disable_band: bool = True       # hv_boost 时关闭 obj-band
    hv_boost_algos: List[str] = field(default_factory=lambda: ["CP/MILP", "NSGA-III", "MOEA/D", "NSGA-II"])

    # Legacy CO-ARCH trigger parameters retained only for backward-compatible
    # diagnostics when hv_boost_enabled is explicitly set outside formal runs.
    hv_boost_ignore_sat_on_coarch: bool = True
    hv_boost_coarch_feas_th: float = 0.90
    hv_boost_coarch_viol_th: float = 0.50

    # -------- Feasible elite archive & rescue (stabilize HV/feas curves) --------
    feasible_archive_enabled: bool = True
    feasible_archive_max_factor: float = 5.0   # max archive size = factor * pop_size
    feasible_rescue_th: float = 0.10          # if feasible_ratio < th, pull elites from archive
    rollback_on_zero_feas: bool = True        # if feasibility collapses, rollback to previous pop
    hv_use_archive: bool = True              # compute HV on cumulative feasible archive (best-so-far)

    # -------- legacy HV block scheduling --------
    # Disabled in formal experiments for consistency with the thesis route.
    hv_block_enabled: bool = False
    hv_block_len: int = 3                # 一个 block 连续保持的 tick 数
    hv_block_topk: int = 2               # block 中使用的 solver 数（1 或 2）
    hv_block_main_share: float = 0.85    # top-1 solver 的预算占比（其余给 top-2）
    hv_block_mix: float = 0.7            # block alpha 与策略 alpha 的软混合权重（1.0=硬覆盖，0.0=无效果）

    # -------- offline RL (IQL) --------
    offline_min_transitions: int = 1000
    offline_train_steps: int = 5000
    offline_train_ratio: float = 0.8
    iql_expectile: float = 0.7
    iql_beta: float = 3.0
    iql_gamma: float = 0.98
    offline_collect_noise: float = 0.12

    # -------- IQL checkpoint / evaluation mode --------
    iql_ckpt_path: str = ""          # if provided, load pretrained IQL and start from eval (skip collect/train)
    iql_freeze: bool = False          # if True, disable online updates during evaluation

    # -------- allocation + gating --------
    alloc_temp: float = 3.0
    op_noop_bias: float = 2.0
    query_budget: float = 1.0

    # -------- OOD fallback --------
    ood_k: int = 10
    # kNN support distance is computed in z-scored high-dimensional state space.
    # A threshold near 8 keeps only the upper-tail support states in fallback,
    # avoiding the degenerate "always fallback" behavior observed at 2.5.
    ood_threshold: float = 8.0
    ood_enabled: bool = True

    # -------- dispatch mode (portfolio vs single) --------
    # "portfolio": 所有算法按 alpha 分配预算并行执行（论文默认）
    # "single":    按 alpha 概率采样选单个算法，全部预算给它
    dispatch_mode: str = "portfolio"

    # -------- credit assignment (在线信用分配) --------
    credit_enabled: bool = True
    credit_window: int = 5          # 滑动窗口长度（tick 数）
    credit_lambda: float = 0.3      # 乘性修正强度

    # -------- cold start (冷启动先验) --------
    cold_start_ticks: int = 3       # 前 N 个 tick 使用冷启动先验
    cold_start_blend: float = 0.7   # 冷启动先验的初始混合权重（随 tick 线性衰减）

    # -------- profiles / regimes (for auto) --------
    profile_id: int = 0
    regime_id: int = 0
    eval_scenario: str = "seen"

    # -------- run mode / metric definition (for paper alignment) --------
    # run_mode: "collect" will enable stochastic perturbations for baseline actions to enrich offline dataset.
    #          : "eval" keeps baselines deterministic for fair comparison.
    run_mode: str = "eval"

    # Sat/PrefGap definition: paper uses the cumulative archive; code default now aligns with that.
    sat_use_archive: bool = True

    # -------- IQL policy improvement (C1) --------
    # If enabled, C1 will score a small candidate set (IQL action + safe baselines [+ optional mix])
    # with its learned critic and pick the best at each tick. This often stabilizes and improves HV/Sat.
    iql_pi_enabled: bool = True
    iql_pi_include_baselines: bool = True
    iql_pi_include_mix: bool = True
    iql_pi_mix: float = 0.5  # mix weight for A2 vs B1 (A2 weight)
    # Evaluation runs can use the actor mean instead of sampling from the
    # Gaussian policy. This makes preset reruns fully reproducible and avoids
    # hidden run-to-run variance from PyTorch sampling in thesis experiments.
    iql_eval_deterministic_actor: bool = False
    # Conservative B1-floor acceptance for C1 residual actions. The IQL
    # residual is executed only when its projected action clears the B1
    # critic score by this margin and passes post-projection safety checks.
    iql_residual_accept_enabled: bool = True
    iql_residual_accept_margin: float = 0.0
    iql_residual_accept_require_safe: bool = True
    iql_residual_accept_channel_guard_enabled: bool = True
    iql_residual_accept_channel_tol: float = 0.05
    iql_residual_accept_channel_adaptive_tol_enabled: bool = False
    iql_residual_accept_channel_tol_relaxed: float = 0.05
    iql_residual_accept_channel_relax_sat_th: float = 0.12
    iql_residual_accept_feas_guard_enabled: bool = True
    iql_residual_accept_min_feas_ratio: float = 0.98
    iql_residual_accept_max_violation: float = 1e-6
    iql_residual_accept_trust_guard_enabled: bool = False
    iql_residual_accept_min_residual_to_b1_norm: float = 0.0
    # Optional late-stage B1 anchoring for formal C1 tuning.  When set in
    # (0, 1), C1 stops executing learned residuals after this horizon fraction
    # and returns to the rule-prior action.  The default keeps the method
    # unchanged.
    iql_residual_terminal_b1_frac: float = 0.0

    # -------- residual anchoring and consistency projection --------
    # C1 默认把 IQL actor proposal 解释为规则基线附近的残差修正；
    # ABL_IQL_DIRECT 则直接执行 actor proposal，用于证明规则先验的必要性。
    iql_residual_anchor: str = "B1"
    iql_residual_scale: float = 0.50
    iql_residual_candidate_scales: List[float] = field(default_factory=list)
    temporal_smooth_tau: float = 0.15
