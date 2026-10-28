# -*- coding: utf-8 -*-
"""OptState: 中央优化状态管理器"""

import os
import re
import json
import threading
import copy
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from hitlo.core.types import (
    ConstraintAtom, UPV, OSV, OSVSample, ProcessFeatures, ModelIR, Transition
)
from hitlo.core.upv_engine import (
    normalize_w, build_state_vector, compile_model_ir, pyd_upv_to_internal,
    compute_penalty_and_feas, compute_violation, phi_objective, compute_sat_prefgap,
    asf_distance, asf_distance_batch, make_neutral_upv, make_single_slot_upv,
    l1_project_aspiration, l2_reconcile_D_Ch, l3_adapt_soft_penalties, l4_goal_consistency,
)
from hitlo.core.osv_engine import (
    compute_conflict_score, compute_confidence_score, apply_conflict_confidence_gating,
    pi_safe_osv_sample, pi_safe_params, pi_safe_model_ir, RollingMedianIQR,
)
from hitlo.core.experiment import ExperimentConfig, normalize_method_id, PAPER_METHOD_IDS
from hitlo.problems import make_problem
from hitlo.solvers import (
    step_emo, step_bo, step_cpm, step_rl,
    build_emo_algorithm, select_algo_by_osv, apply_osv_knobs, apply_operator,
)
from hitlo.policies import AWRPolicy, IQLPolicy, PreferenceGP, OfflineDataset, RollingRobustScale
from hitlo.engine.metrics import is_nondominated, crowding_distance, select_representative_indices
from hitlo.engine.convergence import ConvergenceDetector, ConvergenceStatus

from pymoo.indicators.hv import HV

try:
    from pymoo.indicators.igd import IGD
except ImportError:
    IGD = None

try:
    from exp_logger import ExperimentLogger, get_reference_pf, compute_igd
except Exception:
    ExperimentLogger = None
    def get_reference_pf(*args, **kwargs): return None
    def compute_igd(*args, **kwargs): return None


# ===========================================================================
#  OptState
# ===========================================================================

class OptState:
    def __init__(self, problem_id: str = "ZDT1", n_var: int = 30):
        self.lock = threading.Lock()
        self.problem_id = problem_id
        self.problem = make_problem(problem_id, n_var)
        self.n_var = int(self.problem.n_var)
        self.n_obj = int(self.problem.n_obj)

        self.upv = UPV(
            w=normalize_w(np.ones(self.n_obj)),
            r=np.zeros(self.n_obj),
            b=[(-np.inf, np.inf) for _ in range(self.n_obj)],
            D=[tuple(x) for x in self.problem.bounds]
        )

        # Dialogue memory for multi-turn NL interaction (Ch3 SS3.3)
        from hitlo.nlp.dialogue_manager import DialogueMemory
        self.dialogue_memory = DialogueMemory(max_turns=10)

        # 算法 & 运行参数
        self.algo_name = "NSGA-II"      # "NSGA-II" | "NSGA-III" | "MOEA/D" | "BO" | "CP/MILP" | "RL"
        self.pop_size = 80
        self.gen_per_tick = 10
        self.seed = 123
        self.rng = np.random.default_rng(self.seed)

        # EMO 操作符参数
        self.cx_prob = 0.9
        self.cx_eta = 15.0
        self.mut_eta = 20.0
        self.mut_prob = None  # None => 1/n_var

        # 具体算法参数容器（面板初值）
        self.emo_params = {
            "cx_prob": 0.9, "cx_eta": 15.0, "mut_eta": 20.0, "mut_prob": None,
            "moead_neighbors": 15, "roi_bias": 0.0
        }
        self.bo_params = {
            "n_local_factor": 6.0, "n_global": 40, "kernel_l": 0.3,
            "beta_override": None, "radius_scale": 1.0,
            "kappa": 2.0, "q": 2, "delta_bo": 0.2
        }
        self.cpm_params = {
            "use_pulp": False, "sample_size": 600, "lambda_pen": 1.0, "ridge_l2": 0.0,
            "tau_milp": 60.0, "delta_cp": 0.2
        }
        self.rl_params = {
            "N_mult": 12, "elite_frac": 0.25, "radius_scale": 1.0, "sigma": 0.15,
            "alpha_cvar": 0.9, "lambda_risk": 1.0, "delta_cem": 0.2
        }
        try:
            import pulp  # noqa
            self.has_pulp = True
        except Exception:
            self.has_pulp = False

        # 运行数据
        self.pop_X = None
        self.pop_F = None
        self.pop_G = None
        self.pop_birth_tick = None  # np.ndarray, shape=(n_pop,), dtype=int
        # Feasible elite archive (best-so-far)
        self.feas_archive_X = None
        self.feas_archive_F = None

        self.eval_count = 0
        self.hv_ref = np.array([1.0, 1.0], dtype=float)

        self.history_hv = []
        self.history_fmin = []
        self.history_feasible = []
        self.history_penalty = []
        self.history_osv = []
        self.history_igd = []           # 新增：IGD 序列
        self.history_upv = []
        self.history_sat = []
        self.history_pref_gap = []
        self.history_viol = []
        self.history_diversity = []
        self.history_conf = []
        self.history_risk = []
        self.history_dhv_smooth = []
        self.history_logs: List[str] = []
        self.pref_gp = PreferenceGP(pf_dim=len(ProcessFeatures().as_vector()),
                                    w_dim=self.n_obj, r_dim=self.n_obj, seed=self.seed)
        self.experiment_config = ExperimentConfig()
        self.pf_ref = None
        self.dhv_normer = RollingMedianIQR()
        self.viol_normer = RollingMedianIQR()
        self.history_hv_feas = []
        self.history_dhv_raw = []
        self.history_dhv_norm = []
        self.delta_hv_smooth = 0.0
        self.delta_hv_raw = 0.0
        self.last_epoch_eval = 0
        self.last_epoch_fail = 0
        self.violation_fail_surrogate = 10.0
        self.interaction_events = 0
        self._last_interaction_events = 0

        # 统一交互事件流（Ch3 §3.2(5)）
        from hitlo.core.event_stream import EventStream
        self.event_stream = EventStream()

        # 因果决策图（Ch6 §6.2 日志回放）
        from hitlo.core.causality import CausalGraph
        self.causal_graph = CausalGraph()
        self.offline_dataset = OfflineDataset(base_dir=".")
        self.offline_phase = "collect"
        self.offline_trained = False

        # OSV & 算法组合策略
        self.osv = None  # OSV 将在 step_opt 中按 algo_pool 初始化

        # -------- OSV policy Π (AWR / offline-style) --------
        # 算法池：OSV 的 algo logits 对应的候选集合（可在后续扩展）
        self.algo_pool = ["NSGA-II", "NSGA-III", "MOEA/D", "BO", "CP/MILP", "RL"]

        # 策略映射 Π：输出 OSV 分布参数 (g_mean, g_std, gate_p)，再进行通道采样
        action_dim = 2 + 2 + len(self.algo_pool) + 4 + 4 + OSV.OP_DIM  # pref+rob+alg+param+model+op
        ctrl_dim = len(OSV.CHANNEL_ORDER)
        self.pi = AWRPolicy(action_dim=action_dim,
                            ctrl_dim=ctrl_dim,
                            seed=self.seed)

        # 过程特征与 ModelIR
        self.pf = ProcessFeatures()
        self.model_ir = ModelIR(bounds=[tuple(x) for x in self.problem.bounds])
        self._pref_buffer: List[np.ndarray] = []  # 存储 pf 向量用于 GP

        # 用于构造离线日志/训练的 transition 缓存
        self._last_state_vec = None
        self._last_action_latent = None
        self._last_mask = None
        self._last_projection_diag: Dict[str, Any] = {}
        self._last_support_dist = float("nan")
        self._last_support_flag = 0
        self._last_support_reason = ""
        self._warned_osv_missing = False
        self.algo_pool = ["NSGA-II", "NSGA-III", "MOEA/D", "BO", "CP/MILP", "RL"]

        # Legacy HV-block scheduling state (disabled in formal Ch5 runs).
        self._hv_block_remaining: int = 0
        self._hv_block_alpha: Optional[np.ndarray] = None
        self._hv_block_picks: List[str] = []
        self.algo_stats: Dict[str, Dict[str, float]] = {
            name: {"n": 0, "sum_reward": 0.0} for name in self.algo_pool
        }
        self.last_algo_real: Optional[str] = None
        self.tick_count: int = 0

        # ==== 实验模式与日志 ====
        self.expt_mode = "upv_full"  # 'single' | 'naive' | 'upv_full'
        self.enable_cr = True  # default; may be overridden by UI/auto scripts
        self.logging_on = False
        self.run_id = None
        self.logger: Optional[ExperimentLogger] = None  # type: Optional[ExperimentLogger]

        # 计数器
        self.tick_count = 0
        self.eval_count = 0
        self.n_atoms_nl_total = 0
        self.n_atoms_visual_total = 0
        self.n_pairs_rank_total = 0
        self.n_clarify_total = 0

        # 收敛提示
        self.converged = False
        self.conv_hint = ""

        # 收敛检测器 (三层架构)
        self.convergence_detector = ConvergenceDetector()
        self.convergence_status: Optional[ConvergenceStatus] = None

        self.running = False
        self.stop_event = threading.Event()
        self.bt_pairs = []
        self.pcp_ranges: Dict[int, Tuple[float, float]] = {}

        # BO 状态
        self._bo_X = None
        self._bo_F = None
        self._bo_G = None

        # RL 状态
        self._rl_mean = None
        self._rl_std = None

        # L5 队列
        self.l5_queue: List[Dict[str, Any]] = []
        self._l5_counter: int = 0
        self._l5_recent: set = set()
        self.l5_pending: List[Dict[str, Any]] = []

    # ---- L5 接口 ----
    def enqueue_l5(self, kind: str, text: str, choices: Optional[List[str]] = None):
        sig = (kind, text, tuple(choices or []))
        # 避免刷屏：若刚刚推送过同样的澄清/确认，跳过
        if sig in self._l5_recent:
            return
        self._l5_recent.add(sig)
        # 限制记忆窗口，防止 set 无限制增长
        if len(self._l5_recent) > 50:
            self._l5_recent = set(list(self._l5_recent)[-25:])
        self.l5_queue.append({"kind": kind, "text": text, "choices": choices or []})

    def pop_all_l5(self) -> List[Dict[str, Any]]:
        items = list(self.l5_queue)
        self.l5_queue.clear()
        return items

    def l5_scan_conflicts(self, latest_feas: Optional[float] = None):
        # 仅在 UPV Full + CR 时才启用主动澄清
        if not (self.enable_cr and getattr(self, "expt_mode", "upv_full") == "upv_full"):
            return
        # 1) D vs bounds
        for i, (lo, hi) in enumerate(self.upv.D or []):
            blo, bhi = float(self.problem.bounds[i, 0]), float(self.problem.bounds[i, 1])
            if lo > bhi or hi < blo:
                self.enqueue_l5("clarify", f"x{i+1} 区间 {lo:.3f}..{hi:.3f} 与边界 {blo:.3f}..{bhi:.3f} 无交集。",
                                ["自动裁剪到边界", "保持不变"])
        # 2) C_h(var) vs D
        for c in self.upv.C_h:
            if c.kind != "var": continue
            lo, hi = self.upv.D[c.idx]
            if c.sign == "<=" and lo > c.value:
                self.enqueue_l5("confirm", f"硬约束 x{c.idx+1} ≤ {c.value} 与当前 D 下界 {lo:.3f} 冲突。",
                                [f"把 D 上界裁到 {c.value}", "保持不变"])
            if c.sign == ">=" and hi < c.value:
                self.enqueue_l5("confirm", f"硬约束 x{c.idx+1} ≥ {c.value} 与当前 D 上界 {hi:.3f} 冲突。",
                                [f"把 D 下界提到 {c.value}", "保持不变"])
        # 3) r vs b
        for j, (blo, bhi) in enumerate(self.upv.b):
            if np.isfinite(blo) and self.upv.r[j] < blo:
                self.enqueue_l5("clarify", f"愿望 r{j+1}={self.upv.r[j]:.3f} 低于带宽下界 {blo:.3f}。",
                                ["夹取到带宽内", "保持不变"])
            if np.isfinite(bhi) and self.upv.r[j] > bhi:
                self.enqueue_l5("clarify", f"愿望 r{j+1}={self.upv.r[j]:.3f} 高于带宽上界 {bhi:.3f}。",
                                ["夹取到带宽内", "保持不变"])
        # 4) 可行率过低
        feas = latest_feas if latest_feas is not None else (self.history_feasible[-1] if self.history_feasible else 0.0)
        if feas < 0.1:
            self.enqueue_l5("warn", f"当前可行率仅 {feas:.2f}，建议放宽 D 或降低软罚。", ["放宽 D（+20%）", "降低软罚（×0.9）", "保持不变"])

    # 兼容旧调用名
    def _l5_proactive_check(self, latest_feas: Optional[float] = None):
        try:
            return self.l5_scan_conflicts(latest_feas=latest_feas)
        except Exception as e:
            self.add_log("L5", f"proactive_check error: {e}")
            return None

    # ---- 杂项 ----
    def add_log(self, kind: str, msg: str):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {kind}: {msg}"
        self.history_logs.append(line)
        if len(self.history_logs) > 1000:
            self.history_logs = self.history_logs[-1000:]

    def init_population(self):
        rng = getattr(self, "rng", None)
        if rng is None:
            rng = np.random.default_rng(self.seed)
            self.rng = rng
        X0 = rng.uniform(low=self.problem.xl, high=self.problem.xu, size=(self.pop_size, self.n_var))
        self.pop_X = X0
        F_list = []
        G_list = []
        n_constr = int(getattr(self.problem, "n_constr", 0) or 0)
        for i in range(self.pop_size):
            out = {}
            try:
                self.problem._evaluate(X0[i], out)
            except Exception as e:
                self.add_log("ERR", f"init_population _evaluate[{i}] failed: {e}")
                raise
            F_list.append(out.get("F"))
            if n_constr > 0 and out.get("G", None) is not None:
                G_list.append(out.get("G"))
        self.pop_F = np.array(F_list)
        self.pop_G = (np.array(G_list, dtype=float)
                      if (n_constr > 0 and len(G_list) == self.pop_size) else None)

        # 记录初始化评估次数（用于更精确的 evals 轴）
        self.eval_count = int(getattr(self, "eval_count", 0) + int(self.pop_size))

        # HV 参考点：以初始种群最差值 + 10% 边距锁定
        F = self.pop_F
        maxv = np.max(F, axis=0)
        span = np.maximum(np.abs(maxv), 1.0)
        self.hv_ref = maxv + 0.1 * span

        self.pop_birth_tick = np.zeros(len(self.pop_X), dtype=int)  # all born at tick 0

        self.log_upv()
        self.add_log("INIT", f"population={self.pop_size}, n_var={self.n_var}")

        # 同步收敛检测器配置
        exp = getattr(self, "experiment_config", None)
        if exp is not None:
            self.convergence_detector = ConvergenceDetector(
                max_evals=int(getattr(exp, "max_evals", 5000)),
                max_wall_time=float(getattr(exp, "max_wall_time", 0)),
                hv_window=int(getattr(exp, "hv_stagnation_window", 10)),
                hv_eps=float(getattr(exp, "hv_stagnation_eps", 0.001)),
                igd_eps=float(getattr(exp, "igd_stagnation_eps", 0.005)),
                patience=int(getattr(exp, "convergence_patience", 3)),
                auto_stop_on_budget=bool(getattr(exp, "auto_stop_on_budget", True)),
                auto_stop_on_convergence=bool(getattr(exp, "auto_stop_on_convergence", False)),
            )
        self.convergence_detector.reset()
        self.convergence_status = None

        # 初始化 BO/RL 状态
        self._bo_X = self.pop_X.copy()
        self._bo_F = self.pop_F.copy()
        self._bo_G = (self.pop_G.copy() if self.pop_G is not None else None)
        self._rl_mean = np.mean(self.pop_X, axis=0)
        self._rl_std = np.std(self.pop_X, axis=0) + 1e-3


    def _update_feasible_archive(
        self,
        X_new: np.ndarray,
        F_new: np.ndarray,
        G_new: Optional[np.ndarray] = None,
        exp_cfg: Optional[ExperimentConfig] = None,
        *,
        tag: str = "",
    ) -> None:
        """Update cumulative feasible archive (best-so-far nondominated set).

        - The archive only keeps *feasible* solutions when constraints exist.
        - We keep the nondominated subset and trim by crowding distance.

        This archive is used for:
        1) Stabilizing HV curves (HV can be computed on archive)
        2) Rescuing population if feasibility collapses (CO-ARCH)
        """
        try:
            cfg = exp_cfg if exp_cfg is not None else getattr(self, "experiment_config", ExperimentConfig())
        except Exception:
            cfg = ExperimentConfig()

        if not bool(getattr(cfg, "feasible_archive_enabled", True)):
            return

        if X_new is None or F_new is None:
            return
        X_new = np.asarray(X_new, dtype=float)
        F_new = np.asarray(F_new, dtype=float)
        if len(X_new) == 0 or len(F_new) == 0:
            return

        n_constr = int(getattr(self.problem, "n_constr", 0) or 0)
        feas_mask = np.ones(len(F_new), dtype=bool)
        if n_constr > 0 and G_new is not None:
            try:
                G_new = np.asarray(G_new, dtype=float)
                if G_new.ndim == 1:
                    G_new = G_new.reshape(-1, 1)
                if len(G_new) == len(F_new):
                    cv = np.sum(np.maximum(0.0, G_new), axis=1)
                    feas_mask = cv <= 1e-12
            except Exception:
                pass

        if not np.any(feas_mask):
            return

        Xf = X_new[feas_mask]
        Ff = F_new[feas_mask]

        if getattr(self, "feas_archive_X", None) is None or getattr(self, "feas_archive_F", None) is None:
            X_all = Xf.copy()
            F_all = Ff.copy()
        else:
            X_all = np.vstack([np.asarray(self.feas_archive_X, dtype=float), Xf])
            F_all = np.vstack([np.asarray(self.feas_archive_F, dtype=float), Ff])

        # keep nondominated
        try:
            nd = is_nondominated(F_all)
            X_nd = X_all[nd]
            F_nd = F_all[nd]
        except Exception:
            X_nd, F_nd = X_all, F_all

        # trim
        try:
            max_factor = float(getattr(cfg, "feasible_archive_max_factor", 5.0))
        except Exception:
            max_factor = 5.0
        max_size = int(max(1, round(max_factor * float(getattr(self, "pop_size", 80) or 80))))

        if len(X_nd) > max_size:
            try:
                cd = crowding_distance(F_nd)
                order = np.argsort(-cd)[:max_size]
                X_nd = X_nd[order]
                F_nd = F_nd[order]
            except Exception:
                X_nd = X_nd[:max_size]
                F_nd = F_nd[:max_size]

        self.feas_archive_X = np.asarray(X_nd, dtype=float)
        self.feas_archive_F = np.asarray(F_nd, dtype=float)

        if tag:
            try:
                self.add_log("ARCH", f"update tag={tag}, size={len(self.feas_archive_F)}")
            except Exception:
                pass


    def _rescue_population_from_archive(
        self,
        exp_cfg: ExperimentConfig,
        *,
        reason: str = "",
    ) -> bool:
        """Try to rescue current population when feasibility collapses or becomes too low.

        Returns True if rescue happened (population modified).
        """
        if not bool(getattr(exp_cfg, "feasible_archive_enabled", True)):
            return False
        X_arch = getattr(self, "feas_archive_X", None)
        F_arch = getattr(self, "feas_archive_F", None)
        if X_arch is None or F_arch is None or len(X_arch) == 0:
            return False

        n_constr = int(getattr(self.problem, "n_constr", 0) or 0)
        if n_constr <= 0:
            # unconstrained: just inject archive elites to stabilize diversity / HV
            try:
                X_all = np.vstack([np.asarray(self.pop_X, dtype=float), np.asarray(X_arch, dtype=float)])
                F_all = np.vstack([np.asarray(self.pop_F, dtype=float), np.asarray(F_arch, dtype=float)])
                nd = is_nondominated(F_all)
                X_nd, F_nd = X_all[nd], F_all[nd]
                if len(X_nd) > self.pop_size:
                    cd = crowding_distance(F_nd)
                    order = np.argsort(-cd)[:self.pop_size]
                    X_nd, F_nd = X_nd[order], F_nd[order]
                self.pop_X = np.asarray(X_nd, dtype=float)
                self.pop_F = np.asarray(F_nd, dtype=float)
                self.pop_G = None
                # Rebuild birth_tick: all rescued individuals marked as current tick
                self.pop_birth_tick = np.full(len(self.pop_X), getattr(self, 'tick_count', 0), dtype=int)
                self.add_log("RESCUE", f"unconstrained rescue using archive ({reason})")
                return True
            except Exception:
                return False

        # constrained: prefer feasible archive to keep feasibility alive
        try:
            X_cur = np.asarray(self.pop_X, dtype=float) if self.pop_X is not None else np.zeros((0, self.n_var), dtype=float)
            F_cur = np.asarray(self.pop_F, dtype=float) if self.pop_F is not None else np.zeros((0, self.n_obj), dtype=float)
            G_cur = getattr(self, 'pop_G', None)
            cv_cur = None
            feas_cur = None
            if G_cur is not None and np.size(G_cur) > 0:
                G_cur = np.asarray(G_cur, dtype=float)
                if G_cur.ndim == 1:
                    G_cur = G_cur.reshape(-1, 1)
                if len(G_cur) == len(F_cur):
                    cv_cur = np.sum(np.maximum(0.0, G_cur), axis=1)
                    feas_cur = cv_cur <= 1e-12

            if feas_cur is not None and np.any(feas_cur):
                X_pool = np.vstack([X_cur[feas_cur], np.asarray(X_arch, dtype=float)])
                F_pool = np.vstack([F_cur[feas_cur], np.asarray(F_arch, dtype=float)])
            else:
                X_pool = np.asarray(X_arch, dtype=float)
                F_pool = np.asarray(F_arch, dtype=float)

            # keep nondominated feasible pool
            nd = is_nondominated(F_pool)
            X_nd, F_nd = X_pool[nd], F_pool[nd]
            if len(X_nd) > self.pop_size:
                cd = crowding_distance(F_nd)
                order = np.argsort(-cd)[:self.pop_size]
                X_nd, F_nd = X_nd[order], F_nd[order]

            # if still not enough, backfill with least-violation individuals from current pop
            n_constr = int(getattr(self.problem, 'n_constr', 0) or 0)
            g_zero = np.zeros((n_constr,), dtype=float) if n_constr > 0 else None

            X_sel = [x for x in X_nd]
            F_sel = [f for f in F_nd]
            G_sel = []
            if n_constr > 0:
                # For pool-selected feasible solutions, set G to zeros (feasible by construction)
                G_sel = [g_zero.copy() for _ in range(len(X_sel))]

            if len(X_sel) < self.pop_size and cv_cur is not None:
                order = np.argsort(cv_cur)
                for idx in order:
                    if feas_cur is not None and bool(feas_cur[idx]):
                        continue
                    X_sel.append(X_cur[idx])
                    F_sel.append(F_cur[idx])
                    if n_constr > 0 and G_cur is not None and np.size(G_cur) > 0 and idx < len(G_cur):
                        G_sel.append(np.asarray(G_cur[idx], dtype=float).ravel())
                    elif n_constr > 0:
                        # Unknown constraint values; mark as slightly infeasible via positive surrogate
                        G_sel.append(np.ones((n_constr,), dtype=float) * 1e-3)
                    if len(X_sel) >= self.pop_size:
                        break

            self.pop_X = np.asarray(X_sel[:self.pop_size], dtype=float)
            self.pop_F = np.asarray(F_sel[:self.pop_size], dtype=float)

            # rebuild pop_G for consistent feasibility measurement
            if n_constr > 0:
                if len(G_sel) < len(self.pop_X):
                    # pad if needed
                    G_sel.extend([g_zero.copy() for _ in range(len(self.pop_X) - len(G_sel))])
                self.pop_G = np.asarray(G_sel[:len(self.pop_X)], dtype=float)
            else:
                self.pop_G = None

            # Rebuild birth_tick: all rescued individuals marked as current tick
            self.pop_birth_tick = np.full(len(self.pop_X), getattr(self, 'tick_count', 0), dtype=int)
            self.add_log('RESCUE', f'constrained rescue using feasible archive ({reason})')
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    #  apply_seed / set_problem / log helpers
    # ------------------------------------------------------------------

    def apply_seed(self, seed: int):
        seed = int(seed)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if self.pref_gp is not None:
            self.pref_gp = PreferenceGP(pf_dim=len(ProcessFeatures().as_vector()),
                                        w_dim=self.n_obj, r_dim=self.n_obj, seed=self.seed)
        action_dim = 2 + 2 + len(self.algo_pool) + 4 + 4 + OSV.OP_DIM
        ctrl_dim = len(OSV.CHANNEL_ORDER)
        self.pi = AWRPolicy(action_dim=action_dim, ctrl_dim=ctrl_dim, seed=self.seed)
        self.osv = None

    def set_problem(self, problem_id: str, n_var: int, n_obj: int | None = None):
        self.problem_id = problem_id
        self.problem = make_problem(problem_id, int(n_var), n_obj=n_obj)
        self.n_var = int(self.problem.n_var)
        self.n_obj = int(self.problem.n_obj)

        # 对齐 UPV
        self.upv.w = normalize_w(np.ones(self.n_obj))
        self.upv.r = np.zeros(self.n_obj)
        self.upv.b = [(-np.inf, np.inf) for _ in range(self.n_obj)]
        self.upv.D = [tuple(x) for x in self.problem.bounds]

        # 清理历史
        self.pop_X = None; self.pop_F = None
        self.feas_archive_X = None
        self.feas_archive_F = None
        self.history_hv.clear(); self.history_fmin.clear()
        self.history_feasible.clear(); self.history_penalty.clear(); self.history_osv.clear()
        self.history_upv.clear()
        self.pcp_ranges.clear()

        self.add_log("PROBLEM", f"switch to {self.problem_id} (n_var={self.n_var})")

    def log_upv(self):
        self.history_upv.append({
            "w1": float(self.upv.w[0]), "w2": float(self.upv.w[1]),
            "r1": float(self.upv.r[0]), "r2": float(self.upv.r[1]),
            "tau": float(self.upv.tau),
            "cvar": float(self.upv.rho.get("cvar_alpha", 1.0))
        })

    def log_osv(self):
        if self.osv is None:
            return
        exec_vec = np.asarray(self.osv.g_exec if self.osv.g_exec is not None else self.osv._default_exec(), dtype=float)
        slices = self.osv._channel_slices()
        row = {
            "eta": float(self.osv.eta),
            "lambda_proxy": float(self.osv.lambda_proxy),
            "trust_region": float(self.osv.trust_region)
        }
        # 逐通道展开
        for name, sl in slices.items():
            vals = np.asarray(exec_vec[sl], dtype=float)
            if name == "alg":
                for i, alg in enumerate(self.osv.algo_names):
                    row[f"alg_{alg}"] = float(vals[i])
            else:
                for j, v in enumerate(vals):
                    row[f"{name}{j}"] = float(v)
        diag = dict(getattr(self, "_last_projection_diag", {}) or {})
        for k, v in diag.items():
            if isinstance(v, (list, tuple, dict, np.ndarray)):
                row[f"{k}_json"] = json.dumps(np.asarray(v).tolist() if isinstance(v, np.ndarray) else v,
                                              ensure_ascii=True, sort_keys=True)
            else:
                row[k] = v
        self.history_osv.append(row)

    def _capture_projection_diag(self,
                                 osv_sample: Optional[OSVSample],
                                 a_latent: Optional[np.ndarray],
                                 gate_eff: Optional[np.ndarray],
                                 behavior_id: str) -> None:
        """Collect one tick of OSV consistency-projection diagnostics for Ch5."""
        diag: Dict[str, Any] = {}
        if osv_sample is not None:
            diag.update(dict(getattr(osv_sample, "projection_diag", {}) or {}))

        gate = np.asarray(gate_eff if gate_eff is not None else [], dtype=float).reshape(-1)
        if gate.size:
            diag["gate_mean"] = float(np.mean(gate))
            for idx, name in enumerate(OSV.CHANNEL_ORDER):
                if idx < gate.size:
                    diag[f"gate_{name}"] = float(gate[idx])
        else:
            diag["gate_mean"] = float("nan")

        residual_norm = float("nan")
        try:
            latent = np.asarray(a_latent, dtype=float).reshape(-1)
            base = np.asarray(osv_sample.u_default, dtype=float).reshape(-1) if osv_sample is not None else np.zeros(0)
            if latent.size == base.size and latent.size > 0:
                residual_norm = float(np.linalg.norm(latent - base) / (np.linalg.norm(base) + 1e-9))
        except Exception:
            residual_norm = float("nan")
        diag["residual_norm"] = residual_norm

        support_dist = float(getattr(self, "_last_support_dist", float("nan")))
        support_flag = int(getattr(self, "_last_support_flag", 0) or 0)
        support_reason = str(getattr(self, "_last_support_reason", "") or "")
        fallback_flag = int(str(behavior_id or "").startswith("OOD_FALLBACK"))
        fallback_reason = "ood_fallback" if fallback_flag else ""
        if support_flag and not fallback_reason:
            fallback_reason = support_reason or "low_support"
        diag["support_dist"] = support_dist
        diag["support_flag"] = support_flag
        diag["support_reason"] = support_reason
        diag["fallback_flag"] = fallback_flag
        diag["fallback_reason"] = fallback_reason
        diag["behavior_id"] = str(behavior_id or "")
        lcc_count = int(diag.get("lcc_count", 0) or 0)
        diag["lcc_count"] = int(lcc_count)
        diag["support_lcc_count"] = int(support_flag)
        if lcc_count > 0 or support_flag:
            diag["air_flag"] = 1
        self._last_projection_diag = diag

    # ------------------------------------------------------------------
    #  _record_eval / _safe_eval
    # ------------------------------------------------------------------

    def _record_eval(self, n_eval: int, n_fail: int = 0):
        self.last_epoch_eval += int(n_eval)
        self.last_epoch_fail += int(n_fail)
        self.eval_count = int(getattr(self, "eval_count", 0) + int(n_eval))

    def _safe_eval(self, x: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """安全评估：返回 (F, G, failed)。

        - F: 目标值向量
        - G: 约束向量（若问题无约束则为 None）
        - failed: 是否失败（异常或数值非有限）

        重要：有些算子/问题在数值不稳定时可能返回 NaN/Inf（但不抛异常）。
        若让 NaN 进入 PF/奖励/IQL 训练，会导致 actor logits 变 NaN 并在评估时报错。
        """
        try:
            x = np.asarray(x, dtype=float).reshape(-1)
            xl = np.asarray(getattr(self.problem, 'xl', None), dtype=float)
            xu = np.asarray(getattr(self.problem, 'xu', None), dtype=float)
            if xl is not None and xu is not None and xl.size == x.size and xu.size == x.size:
                mid = 0.5 * (xl + xu)
                x = np.nan_to_num(x, nan=mid, posinf=xu, neginf=xl)
                x = np.clip(x, xl, xu)
            else:
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            out = {}
            self.problem._evaluate(x, out)
            f = out.get('F', None)
            g = out.get('G', None)
            if f is None:
                self._record_eval(1, 1)
                return None, None, True
            f = np.asarray(f, dtype=float).reshape(-1)
            if not np.isfinite(f).all():
                self._record_eval(1, 1)
                return None, None, True
            if g is not None:
                g = np.asarray(g, dtype=float).reshape(-1)
                if not np.isfinite(g).all():
                    self._record_eval(1, 1)
                    return None, None, True
            return f, g, False
        except Exception:
            self._record_eval(1, 1)
            return None, None, True

    # ------------------------------------------------------------------
    #  _compute_pf_from_history / _update_pref_gp
    # ------------------------------------------------------------------

    def _compute_pf_from_history(self, W: int = 5) -> ProcessFeatures:
        def _last(arr, default=0.0):
            return float(arr[-1]) if arr else float(default)
        def _diff(arr):
            return float(arr[-1] - arr[-2]) if len(arr) >= 2 else 0.0
        d_hv_s = _last(self.history_dhv_smooth, 0.0)
        stall = 1.0 if d_hv_s < float(self.experiment_config.hv_eps) else 0.0
        pf = ProcessFeatures(
            d_hv_smooth=d_hv_s,
            stall_flag=stall,
            feas_ratio=_last(self.history_feasible, 0.0),
            mean_violation=_last(self.history_viol, 0.0),
            diversity=_last(self.history_diversity, 0.0),
            sat=_last(self.history_sat, 0.0),
            conf=_last(self.history_conf, 0.0),
            risk_proxy=_last(self.history_risk, 0.0),
            d_feas_ratio=_diff(self.history_feasible),
            d_mean_violation=_diff(self.history_viol),
            d_diversity=_diff(self.history_diversity),
            d_sat=_diff(self.history_sat),
            d_conf=_diff(self.history_conf),
            d_risk_proxy=_diff(self.history_risk),
        )
        return pf

    def _update_pref_gp(self, pf_vec: np.ndarray):
        exp_cfg = getattr(self, 'experiment_config', ExperimentConfig())
        if 'ABL_NO_PREF_GP' in exp_cfg.ablations:
            return

        try:
            if self.pref_gp is None:
                return
            self.pref_gp.add_sample(pf_vec, self.upv.w, self.upv.r, holdout_ratio=0.2)
            self.pref_gp.fit()
            pred = self.pref_gp.predict(pf_vec, return_std=True)
            if pred is None:
                return
            w_hat, r_hat, w_std, r_std = pred

            # v8 S1.9: PrefGP baseline 变体 (规则预测器替代 GP, 验证 GP 概率预测增益)
            if 'ABL_PREFGP_FROZEN' in exp_cfg.ablations:
                # v_frozen: 偏好在最后一次交互后冻结 (Wang 2013 PICEA 类似设定)
                if len(self.pref_gp.Yw) > 0:
                    w_hat = self.pref_gp.Yw[-1].ravel().astype(float)
                    r_hat = self.pref_gp.Yr[-1].ravel().astype(float)
                    w_std = np.ones_like(w_hat) * 0.5
                    r_std = np.ones_like(r_hat) * 0.5
            elif 'ABL_PREFGP_LINEAR_DRIFT' in exp_cfg.ablations:
                # v_linearDrift: 用最后两次观测差线性外推下一时刻
                if len(self.pref_gp.Yw) >= 2:
                    dw = (self.pref_gp.Yw[-1] - self.pref_gp.Yw[-2]).ravel()
                    dr = (self.pref_gp.Yr[-1] - self.pref_gp.Yr[-2]).ravel()
                    w_hat = (self.pref_gp.Yw[-1].ravel() + dw).astype(float)
                    r_hat = (self.pref_gp.Yr[-1].ravel() + dr).astype(float)
                    w_std = np.ones_like(w_hat) * 0.5
                    r_std = np.ones_like(r_hat) * 0.5

            w_hat = np.maximum(1e-6, w_hat)
            w_hat = normalize_w(w_hat)

            # --- 精度自适应融合（替代固定0.7/0.3） ---
            _eps = 0.05
            # §4.5 ABL_AVG_FUSION: 用 alpha=0.5 平均融合替代精度加权 (验证精度加权的增益)
            _avg_fusion = 'ABL_AVG_FUSION' in exp_cfg.ablations
            # w的精度加权：prec = 1/sigma，alpha = prec_obs/(prec_prior+prec_obs)
            prec_prior_w = 1.0 / max(float(self.upv.sigma.get("w", 0.5)), _eps)
            prec_obs_w = 1.0 / max(float(np.mean(w_std)) if w_std is not None else 0.5, _eps)
            alpha_w = 0.5 if _avg_fusion else \
                float(np.clip(prec_obs_w / (prec_prior_w + prec_obs_w), 0.10, 0.90))
            self.upv.w = normalize_w((1.0 - alpha_w) * self.upv.w + alpha_w * w_hat)

            # r的精度加权
            prec_prior_r = 1.0 / max(float(self.upv.sigma.get("r", 0.5)), _eps)
            prec_obs_r = 1.0 / max(float(np.mean(r_std)) if r_std is not None else 0.5, _eps)
            alpha_r = 0.5 if _avg_fusion else \
                float(np.clip(prec_obs_r / (prec_prior_r + prec_obs_r), 0.10, 0.90))
            self.upv.r = (1.0 - alpha_r) * self.upv.r + alpha_r * np.asarray(r_hat, dtype=float)

            if w_std is not None and r_std is not None:
                self.upv.sigma["w"] = float(np.mean(w_std))
                self.upv.sigma["r"] = float(np.mean(r_std))
            self.log_upv()
            self.add_log("PREF-GP", f"precision-fuse alpha_w={alpha_w:.2f} alpha_r={alpha_r:.2f}; w≈{self.upv.w.tolist()}")
        except Exception as _e:
            self.add_log("WARN", f"pref_gp update skipped: {_e}")

    # ------------------------------------------------------------------
    #  _baseline_osv_action
    # ------------------------------------------------------------------

    def _baseline_osv_action(
        self,
        rng: np.random.Generator,
        pf: ProcessFeatures,
        *,
        force_behavior_id: Optional[str] = None,
        stage_frac: Optional[float] = None,
        noise_std: float = 0.08,
        enable_perturb: bool = True,
    ) -> Tuple[np.ndarray, str, np.ndarray, float]:
        'Rule-based baseline OSV controller (for A2/B1 + offline dataset perturbations).'

        exp_cfg = copy.deepcopy(getattr(self, "experiment_config", ExperimentConfig()))
        exp_cfg.exp_group = normalize_method_id(exp_cfg.exp_group)
        # 论文 C2: no-safety (disable Π_safe & confidence gating)
        if exp_cfg.exp_group == "C2_PPCS_IQL_NO_SAFETY":
            for _a in ("ABL_NO_SAFETY", "ABL_NO_GATING"):
                if _a not in exp_cfg.ablations:
                    exp_cfg.ablations.append(_a)
        # 论文 A1/A2/A3: baseline (disable param adaptation & model reconfiguration by definition)
        if exp_cfg.exp_group in ("A1_FIXED_SINGLE", "A2_FIXED_PORTFOLIO", "A3_UPV_DIRECT"):
            for _a in ("ABL_NO_PARAM", "ABL_NO_MODEL_RECONF"):
                if _a not in exp_cfg.ablations:
                    exp_cfg.ablations.append(_a)
            if exp_cfg.exp_group in ("A1_FIXED_SINGLE", "A3_UPV_DIRECT") and "ABL_NO_SCHEDULING" not in exp_cfg.ablations:
                exp_cfg.ablations.append("ABL_NO_SCHEDULING")
        if self.osv is None:
            self.osv = OSV.from_upv(self.upv, algo_names=self.algo_pool)
        slices = self.osv._channel_slices()
        total = self.osv.total_dim
        u = np.zeros(total, dtype=float)

        # ---- determine behavior id ----
        if force_behavior_id is not None:
            behavior_id = str(force_behavior_id)
        elif stage_frac is not None:
            behavior_id = "A2"
        else:
            is_risky = (pf.feas_ratio < 0.3) or (pf.mean_violation > 5.0) or (pf.stall_flag > 0.6)
            behavior_id = "A2" if is_risky else "B1"

        # ---- helpers ----
        sl_param = slices.get("param")
        sl_rob = slices.get("rob")
        sl_pref = slices.get("pref")
        sl_model = slices.get("model")
        sl_op = slices.get("op")
        sl_alg = slices.get("alg")

        def _set_alloc(weights: Dict[str, float]):
            if not sl_alg:
                return
            w = np.array([float(weights.get(name, 0.0)) for name in self.algo_pool], dtype=float)
            if not np.isfinite(w).all() or float(np.sum(w)) <= 1e-12:
                w = np.ones(len(self.algo_pool), dtype=float)
            w = w / (float(np.sum(w)) + 1e-12)
            logits = np.log(w + 1e-8)
            logits = logits - float(np.mean(logits))
            temp = max(float(exp_cfg.alloc_temp), 1e-6)
            u[sl_alg] = np.clip(logits / temp, -5.0, 5.0)

        def _set_op_index(op_idx: int):
            if not sl_op:
                return
            k = sl_op.stop - sl_op.start
            if k <= 0:
                return
            op_idx = int(np.clip(int(op_idx), 0, k - 1))
            op_logits = np.zeros(k, dtype=float)
            op_logits[op_idx] = 1.0
            u[sl_op] = op_logits

        # ---- continuous channels (uexpl/ucon/uroi/uloc, urisk, u_pref, u_int) ----
        if behavior_id == "A2":
            frac = float(stage_frac) if stage_frac is not None else 0.5
            if frac < 1.0 / 3.0:
                weights = {"NSGA-III": 0.35, "MOEA/D": 0.25, "NSGA-II": 0.20, "BO": 0.20}
            elif frac < 2.0 / 3.0:
                weights = {"MOEA/D": 0.35, "NSGA-III": 0.25, "NSGA-II": 0.20, "BO": 0.10, "RL": 0.10}
            else:
                weights = {"CP/MILP": 0.35, "MOEA/D": 0.25, "NSGA-II": 0.20, "NSGA-III": 0.10, "BO": 0.05, "RL": 0.05}
            _set_alloc(weights)

            # conservative knobs (more constraint focus, less ROI bias)
            if sl_param and (sl_param.stop - sl_param.start) >= 4:
                u[sl_param.start + 0] = 0.2   # u_expl
                u[sl_param.start + 1] = 0.8   # u_con
                u[sl_param.start + 2] = -0.2  # u_loc
                u[sl_param.start + 3] = -0.3  # u_roi
            if sl_rob and (sl_rob.stop - sl_rob.start) >= 2:
                u[sl_rob.start + 0] = 0.3     # u_risk
                u[sl_rob.start + 1] = 0.0
            if sl_model and (sl_model.stop - sl_model.start) >= 4:
                u[sl_model.start + 0] = 0.3   # u_int
                u[sl_model.start + 1] = 0.0
                u[sl_model.start + 2] = 0.0
                u[sl_model.start + 3] = 0.0
            if sl_pref and (sl_pref.stop - sl_pref.start) >= 2:
                u[sl_pref.start + 0] = 0.0
                u[sl_pref.start + 1] = 0.0

            _set_op_index(0)  # NOOP

        else:  # "B1" rule-OSV
            conf = float(compute_confidence_score(self))
            conflict = float(compute_conflict_score(self))
            feas = float(pf.feas_ratio)
            viol = float(pf.mean_violation)
            stall = float(pf.stall_flag)

            # knobs: more exploration on stall, more penalty on low feas
            if sl_param and (sl_param.stop - sl_param.start) >= 4:
                u_expl = np.clip(0.2 + 0.8 * stall, -1.0, 1.0)
                u_con  = np.clip(0.6 + 0.6 * (1.0 - feas), -1.0, 1.0)
                u_loc  = np.clip(-0.3 + 0.6 * (1.0 - stall), -1.0, 1.0)
                u_roi  = np.clip(0.3 + 0.7 * max(0.0, (0.5 - conflict)), -1.0, 1.0)
                u[sl_param.start + 0] = float(u_expl)
                u[sl_param.start + 1] = float(u_con)
                u[sl_param.start + 2] = float(u_loc)
                u[sl_param.start + 3] = float(u_roi)

            if sl_rob and (sl_rob.stop - sl_rob.start) >= 2:
                u_risk = np.clip(0.2 + 0.6 * (1.0 - feas) + 0.2 * float(pf.risk_proxy), -1.0, 1.0)
                u[sl_rob.start + 0] = float(u_risk)
                u[sl_rob.start + 1] = 0.0

            if sl_pref and (sl_pref.stop - sl_pref.start) >= 2:
                # encourage stronger preference injection when confident and low conflict
                u_pref = np.clip(2.0 * (conf - 0.5), -1.0, 1.0)
                u_smooth = np.clip(1.0 - 2.0 * conflict, -1.0, 1.0)
                u[sl_pref.start + 0] = float(u_pref)
                u[sl_pref.start + 1] = float(u_smooth)

            if sl_model and (sl_model.stop - sl_model.start) >= 4:
                # reconfig intensity rises with constraint pressure
                u_int = np.clip(0.2 + 0.8 * (1.0 - feas), -1.0, 1.0)
                u[sl_model.start + 0] = float(u_int)
                u[sl_model.start + 1] = 0.0
                u[sl_model.start + 2] = 0.0
                u[sl_model.start + 3] = 0.0

            # allocations: prefer EMO, add some BO when confident, add CP/MILP when risky
            weights = {"NSGA-II": 0.25, "NSGA-III": 0.25, "MOEA/D": 0.25, "BO": 0.15, "RL": 0.10}
            if feas < 0.3 or viol > 5.0:
                weights["CP/MILP"] = 0.20
                weights["BO"] = 0.05
            if conf > 0.7 and conflict < 0.4:
                weights["BO"] = min(0.25, weights.get("BO", 0.0) + 0.1)
            _set_alloc(weights)

            # operator selection (R1..R5)
            op_idx = 0
            if feas < 0.2 or viol > 6.0:
                op_idx = 1  # R1: constraints/penalty
            elif stall > 0.65:
                op_idx = 3  # R3: local exploration / trust region
            elif conf < 0.35 and conflict < 0.6:
                op_idx = 2  # R2: preference smoothing / damping
            elif float(pf.risk_proxy) > 0.6:
                op_idx = 5  # R5: robustness
            elif float(pf.sat) < 0.35 and float(pf.diversity) > 0.25:
                op_idx = 4  # R4: ROI focus
            _set_op_index(op_idx)

        # ---- stochastic perturbations (offline dataset collection) ----
        perturb_mask = np.zeros_like(u, dtype=bool)
        eff_noise = float(noise_std) if (enable_perturb and noise_std > 0.0) else 0.0
        if eff_noise > 0.0:
            # do NOT perturb operator logits to keep behavior interpretable
            if sl_param:
                perturb_mask[sl_param] = True
            if sl_pref:
                perturb_mask[sl_pref] = True
            if sl_model:
                perturb_mask[sl_model] = True
            if sl_rob:
                perturb_mask[sl_rob] = True
            if sl_alg:
                perturb_mask[sl_alg] = True
            if sl_op:
                perturb_mask[sl_op] = False

            u = u + rng.normal(0.0, eff_noise, size=u.shape) * perturb_mask.astype(float)

        return u, behavior_id, perturb_mask, eff_noise

    # ------------------------------------------------------------------
    #  _estimate_cvar / _avg_reward
    # ------------------------------------------------------------------

    # ---- OSV 辅助函数 ----
    def _estimate_cvar(self, alpha: float) -> Optional[float]:
        """基于当前种群，用 ASF 距离估计 CVaR@alpha（尾部风险）"""
        if self.pop_F is None or len(self.pop_F) == 0:
            return None
        F = np.asarray(self.pop_F, dtype=float)
        losses = np.array([asf_distance(f, self.upv.w, self.upv.r) for f in F], dtype=float)
        losses = losses[np.isfinite(losses)]
        if losses.size == 0:
            return None
        q = np.quantile(losses, alpha)
        tail = losses[losses >= q]
        if tail.size == 0:
            return float(q)
        return float(np.mean(tail))

    def _avg_reward(self, algos: List[str]) -> float:
        vals = []
        for name in algos:
            st = self.algo_stats.get(name)
            if st and st["n"] > 0:
                vals.append(st["sum_reward"] / st["n"])
        return float(np.mean(vals)) if vals else 0.0

    # ------------------------------------------------------------------
    #  _is_ood_state
    # ------------------------------------------------------------------

    def _is_ood_state(self, s_t: np.ndarray) -> bool:
        """基于离线数据集对当前状态做轻量 OOD 检测（kNN 距离, z-score 标准化）。

        说明：
        - 该函数在 C1/C2 的 policy eval 阶段（非 collect）用于 OOD fallback。
        - 目的不是精确密度估计，而是提供一个鲁棒的"分布外"代理，避免策略在明显异常状态下失控。
        - 若离线数据不足、维度不一致或数值异常，会采取保守策略（多判为 OOD 或直接返回 False）。
        """
        def _finish(flag: bool, score: float = float("nan"), reason: str = "") -> bool:
            self._last_support_dist = float(score)
            self._last_support_flag = int(bool(flag))
            self._last_support_reason = str(reason or "")
            return bool(flag)

        exp_cfg = getattr(self, "experiment_config", ExperimentConfig())
        if not getattr(exp_cfg, "ood_enabled", False):
            return _finish(False, float("nan"), "disabled")

        k_req = int(getattr(exp_cfg, "ood_k", 10) or 10)
        support_states = getattr(self, "_support_states", None)
        if support_states is not None:
            try:
                states = np.asarray(support_states, dtype=float)
            except Exception:
                states = np.zeros((0, 0), dtype=float)
        else:
            ds = getattr(self, "offline_dataset", None)
            if ds is None:
                return _finish(False, float("nan"), "no_dataset")
            if len(getattr(ds, "transitions", [])) < k_req:
                return _finish(False, float("nan"), "insufficient_dataset")
            try:
                states = np.asarray([t.s for t in ds.transitions], dtype=float)
            except Exception:
                return _finish(False, float("nan"), "dataset_error")

        if states.shape[0] < k_req:
            return _finish(False, float("nan"), "insufficient_support")

        if states.ndim != 2:
            return _finish(False, float("nan"), "bad_dataset_shape")

        s = np.asarray(s_t, dtype=float).reshape(1, -1)
        if s.shape[1] != states.shape[1]:
            # 维度不一致：更安全的做法是直接视为 OOD
            return _finish(True, float("nan"), "dim_mismatch")

        # 防止 NaN/Inf 传播
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

        mu = states.mean(axis=0)
        std = states.std(axis=0) + 1e-6
        Xn = (states - mu) / std
        sn = (s - mu) / std

        d = np.linalg.norm(Xn - sn, axis=1)
        k = min(k_req, len(d))
        if k <= 0:
            return False

        d_k = np.partition(d, k - 1)[:k]
        score = float(np.mean(d_k))
        thr = float(getattr(exp_cfg, "ood_threshold", 8.0))
        flag = (not np.isfinite(score)) or (score > thr)
        return _finish(flag, score, "ood" if flag else "in_support")

    # ------------------------------------------------------------------
    #  Delegation methods (forward to hitlo.solvers)
    # ------------------------------------------------------------------

    def _step_emo(self, upv_opt=None, budget=None):
        return step_emo(self, upv_opt, budget)

    def _step_bo(self, upv_opt=None, budget=None):
        return step_bo(self, upv_opt, budget)

    def _step_cpm(self, upv_opt=None, budget=None):
        return step_cpm(self, upv_opt, budget)

    def _step_rl(self, upv_opt=None, budget=None):
        return step_rl(self, upv_opt, budget)

    def _build_emo_algorithm(self, upv_opt=None):
        return build_emo_algorithm(self, upv_opt)

    def _apply_osv_knobs(self, u_exec, exp_cfg):
        return apply_osv_knobs(self, u_exec, exp_cfg)

    def _apply_operator(self, op_idx, u_exec, xi):
        return apply_operator(self, op_idx, u_exec, xi)

    def step_opt(self):
        """单步优化（一个 tick）。

        新版逻辑（与你最新论文一致）：
          1) 由 UPV + 过程特征 F(t) 构造状态 s_t
          2) 策略映射 Π(s_t) 输出 OSV 分布参数；采样得到 OSVSample（带通道 mask + algo_probs）
          3) OSVSample 同时驱动：
             - 算法调度/切换（选择主算法）
             - 参数自适应（EMO/BO/CP/RL 的关键旋钮）
             - 模型重构（compile_model_ir -> ModelIR -> WrappedProblem/采样域）
          4) 运行一个 tick，计算指标；更新 Π（离线/在线 AWR）
        """
        old_algo = self.algo_name
        rng = np.random.default_rng(self.seed + self.tick_count)

        # ---------- 0) Build F(t) from history window (Appendix A.0.7) ----------
        if self.pop_X is None or self.pop_F is None or len(self.pop_X) == 0:
            self._step_emo()
            return

        # snapshot (for rollback / stability)
        X_prev = np.asarray(self.pop_X, dtype=float).copy()
        F_prev = np.asarray(self.pop_F, dtype=float).copy()
        G_prev = (np.asarray(self.pop_G, dtype=float).copy()
                  if getattr(self, 'pop_G', None) is not None and np.size(getattr(self, 'pop_G', None)) > 0 else None)

        # Save old birth_tick mapping for post-step inheritance
        _old_birth = {}
        if getattr(self, 'pop_birth_tick', None) is not None and len(self.pop_birth_tick) == len(X_prev):
            for i in range(len(X_prev)):
                _old_birth[tuple(X_prev[i].tolist())] = int(self.pop_birth_tick[i])

        self.pf = self._compute_pf_from_history()
        try:
            setattr(self.pf, 'problem_id', str(self.problem_id))
            setattr(self.pf, 'tick_count', int(self.tick_count))
            setattr(self.pf, 'n_constr', int(getattr(self.problem, 'n_constr', 0) or 0))
            if getattr(self, "osv", None) is not None and getattr(self.osv, "g_exec", None) is not None:
                setattr(self.pf, "prev_exec_action", np.asarray(self.osv.g_exec, dtype=float).copy())
        except Exception:
            pass
        pf_vec = self.pf.as_vector()
        self._update_pref_gp(pf_vec)

        # init/update feasible archive using current population (no extra evals)
        try:
            self._update_feasible_archive(self.pop_X, self.pop_F, getattr(self, 'pop_G', None), exp_cfg=getattr(self, 'experiment_config', None), tag='tick_pre')
        except Exception:
            pass

        # IR summary (fixed dims) for state vector
        ir_summary = np.zeros(6, dtype=float)
        if getattr(self, "model_ir", None) is not None:
            ir = self.model_ir
            scalar_idx = {"WSUM": 0.0, "TCH": 0.5, "ASF": 1.0}.get(getattr(ir, "scalar_mode", "WSUM"), 0.0)
            ir_summary = np.array([
                float(getattr(ir, "trust_radius_scale", 1.0)),
                float(getattr(ir, "penalty_scale", 1.0)),
                float(getattr(ir, "bias_lambda", 0.05)),
                float(getattr(ir, "risk_alpha", self.upv.rho.get("cvar_alpha", 0.9))),
                float(scalar_idx),
                float(1.0 if getattr(ir, "band_penalty", 0.0) > 0 else 0.0)
            ], dtype=float)

        # ---------- 1) 构造状态 s_t ----------
        # 注意：为了支持 Abl-4（去掉过程特征）以及 C2（no-safety/no-gating），我们在这里先固化 exp_cfg。
        exp_cfg = copy.deepcopy(getattr(self, "experiment_config", ExperimentConfig()))
        exp_cfg.exp_group = normalize_method_id(exp_cfg.exp_group)

        # 论文 C2: no-safety (disable Π_safe & confidence gating)
        if exp_cfg.exp_group == "C2_PPCS_IQL_NO_SAFETY":
            for _a in ("ABL_NO_SAFETY", "ABL_NO_GATING"):
                if _a not in exp_cfg.ablations:
                    exp_cfg.ablations.append(_a)

        # 论文 A1/A2/A3: baseline (disable param adaptation & model reconfiguration by definition)
        if exp_cfg.exp_group in ("A1_FIXED_SINGLE", "A2_FIXED_PORTFOLIO", "A3_UPV_DIRECT"):
            for _a in ("ABL_NO_PARAM", "ABL_NO_MODEL_RECONF"):
                if _a not in exp_cfg.ablations:
                    exp_cfg.ablations.append(_a)
            if exp_cfg.exp_group in ("A1_FIXED_SINGLE", "A3_UPV_DIRECT") and "ABL_NO_SCHEDULING" not in exp_cfg.ablations:
                exp_cfg.ablations.append("ABL_NO_SCHEDULING")

        # Abl-4: no process features -> 状态中 pf 置零，但评估指标仍用真实 pf。
        pf_for_state = ProcessFeatures() if ("ABL_NO_PROCESS_FEATURES" in exp_cfg.ablations) else self.pf

        # §4.5 ABL_SINGLE_SLOT: 强制 UPV 单槽位 (仅 S1 = w), 中性化 S2-S6 槽位.
        # 在 self.upv 层级生效, 让所有依赖 r/b/C_h/C_s/pi/rho 的下游逻辑（asf/L4/scalarization）
        # 均看到单槽位状态; ADM 通过 NL 通道写入的 r/b/C_h/C_s 会在下个 tick 被覆盖.
        if "ABL_SINGLE_SLOT" in exp_cfg.ablations:
            self.upv = make_single_slot_upv(self.upv)

        if "ABL_NO_UPV" in exp_cfg.ablations:
            upv_state = make_neutral_upv(self.upv)
        elif "ABL_SINGLE_SLOT" in exp_cfg.ablations:
            upv_state = self.upv  # 已被中性化, 直接传
        else:
            upv_state = self.upv
        s_t = build_state_vector(upv_state, pf_for_state, ir_summary=ir_summary)
        self._last_state_vec = s_t

        self.last_epoch_eval = 0
        self.last_epoch_fail = 0
        self._last_projection_diag = {}
        self._last_support_dist = float("nan")
        self._last_support_flag = 0
        self._last_support_reason = ""

        # ---------- 2) Pi(s_t) -> OSV action (Eq.(9)) ----------
        if self.osv is None:
            self.osv = OSV.from_upv(self.upv, algo_names=self.algo_pool)

        conflict = compute_conflict_score(self)
        confidence = compute_confidence_score(self)
        gate_eff = apply_conflict_confidence_gating(self.osv, conflict, confidence, exp_cfg)
        if "ABL_NO_GATING" in exp_cfg.ablations:
            gate_eff = np.ones(len(OSV.CHANNEL_ORDER), dtype=float)
        single_channel = None
        for abl in exp_cfg.ablations:
            if abl.startswith("ABL_SINGLE_CHANNEL"):
                parts = abl.split("=")
                if len(parts) == 2:
                    single_channel = parts[1].strip().lower()
        if single_channel:
            gate_eff = np.array([1 if name == single_channel else 0 for name in OSV.CHANNEL_ORDER], dtype=float)

        # §4.5 ABL_NO_OSV_X: 关闭对应 OSV 通道 (gate_eff[X]=0 让该通道输出失效).
        # ch4_batchrunner 的 v_noOSV_pref/feas/explore/param/rebuild/interact 六分组消融在此实施.
        _OSV_ABL_MAP = {
            "ABL_NO_OSV_PREF":  "pref",
            "ABL_NO_OSV_ROB":   "rob",
            "ABL_NO_OSV_ALG":   "alg",
            "ABL_NO_OSV_PARAM": "param",
            "ABL_NO_OSV_MODEL": "model",
            "ABL_NO_OSV_OP":    "op",
        }
        for _abl, _ch in _OSV_ABL_MAP.items():
            if _abl in exp_cfg.ablations:
                try:
                    _idx = OSV.CHANNEL_ORDER.index(_ch)
                    gate_eff[_idx] = 0.0
                except (ValueError, IndexError):
                    pass

        behavior_id = exp_cfg.exp_group
        perturb_mask = None
        noise_std = 0.0

        method_id = exp_cfg.exp_group  # 已经 normalize_method_id 过

        # ---- Baselines (paper A1/A2/A3/B1) & learned controllers (C*) ----
        if method_id in ("A1_FIXED_SINGLE", "A2_FIXED_PORTFOLIO", "A3_UPV_DIRECT", "B1_RULE_OSV"):
            # Construct a deterministic (or rule-based) OSV action and then run the same Π_safe / runtime path
            u_default = self.osv._default_exec()
            sls = self.osv._channel_slices()
            sl_alg = sls.get("alg")
            sl_op = sls.get("op")

            # In offline dataset collection runs, inject controlled perturbations into baseline OSV channels
            # (before Π_safe) to improve IQL coverage. Baseline evaluation runs stay deterministic.
            collect_mode = str(getattr(exp_cfg, "run_mode", "eval")).lower().strip() == "collect"
            base_noise = float(getattr(exp_cfg, "offline_collect_noise", 0.0)) if collect_mode else 0.0
            base_enable_perturb = bool(collect_mode and base_noise > 0.0)

            # Helper: fixed single-solver action (only affects scheduling logits; other channels stay default)
            def _fixed_single_u(target_algo: str) -> np.ndarray:
                u = np.asarray(u_default, dtype=float).copy()
                if sl_alg is not None:
                    logits = -4.0 * np.ones(len(self.algo_pool), dtype=float)
                    if target_algo in self.algo_pool:
                        logits[self.algo_pool.index(target_algo)] = 4.0
                    u[sl_alg] = logits
                if sl_op is not None:
                    op_dim = sl_op.stop - sl_op.start
                    op_logits = -3.0 * np.ones(op_dim, dtype=float)
                    op_logits[0] = 3.0  # NOOP
                    u[sl_op] = op_logits
                return u

            if method_id == "A1_FIXED_SINGLE":
                u_tilde = _fixed_single_u(exp_cfg.fixed_single_solver)
                behavior_id = "A1"
            elif method_id == "A3_UPV_DIRECT":
                target = getattr(exp_cfg, "upv_direct_solver", "NSGA-III")
                u_tilde = _fixed_single_u(target)
                behavior_id = "A3"
            elif method_id == "A2_FIXED_PORTFOLIO":
                # 固定阶段调度：仅依赖 tick/horizon，不依赖 UPV 或过程特征
                t_frac = float(self.tick_count) / float(max(1, int(exp_cfg.horizon_T) - 1))
                u_tilde, behavior_id, perturb_mask, noise_std = self._baseline_osv_action(
                    rng, self.pf,
                    force_behavior_id="A2",
                    stage_frac=t_frac,
                    noise_std=base_noise,
                    enable_perturb=base_enable_perturb,
                )
                behavior_id = "A2"
            else:  # B1_RULE_OSV
                u_tilde, behavior_id, perturb_mask, noise_std = self._baseline_osv_action(
                    rng, self.pf,
                    force_behavior_id="B1",
                    noise_std=base_noise,
                    enable_perturb=base_enable_perturb,
                )
                behavior_id = "B1"

            osv_sample = self.osv.sample_from_action(u_tilde, gate_eff)
            osv_sample.behavior_id = behavior_id
            osv_sample.profile_id = int(exp_cfg.profile_id)
            osv_sample.regime_id = int(exp_cfg.regime_id)
            osv_sample.perturb_mask = perturb_mask
            osv_sample.noise_std = float(noise_std)

            osv_sample, alpha_safe, op_idx, xi = pi_safe_osv_sample(osv_sample, self.pf, exp_cfg, self.algo_pool)
            a_latent = np.asarray(u_tilde, dtype=float)

        elif method_id in ("C1_PPCS_IQL_FULL", "C2_PPCS_IQL_NO_SAFETY"):
            # IQL controller (learning-based PPCS)
            if not isinstance(self.pi, IQLPolicy):
                self.pi = IQLPolicy(action_dim=self.osv.total_dim, op_dim=OSV.OP_DIM, alg_dim=len(self.algo_pool),
                                    seed=self.seed,
                                    expectile=exp_cfg.iql_expectile,
                                    beta=exp_cfg.iql_beta,
                                    gamma=exp_cfg.iql_gamma)

            # Optional: load a pretrained IQL checkpoint (evaluation-only runs)
            ckpt_path = str(getattr(exp_cfg, "iql_ckpt_path", "") or "").strip()
            if ckpt_path and os.path.isfile(ckpt_path) and (not getattr(self.pi, "_ckpt_loaded", False)):
                try:
                    self.pi.load(ckpt_path)
                    self.pi._ckpt_loaded = True
                    self.offline_phase = "eval"
                    self.offline_trained = True
                except Exception as _e:
                    try:
                        self.add_log("WARN", f"Failed to load IQL ckpt: {ckpt_path} ({_e})")
                    except Exception:
                        pass

            if self.offline_phase == "collect":
                # Offline dataset collection uses safe baselines A2/B1 with stochastic perturbations
                u_tilde, b_id, perturb_mask, noise_std = self._baseline_osv_action(
                    rng, self.pf,
                    force_behavior_id=None,
                    stage_frac=None,
                    noise_std=float(exp_cfg.offline_collect_noise),
                    enable_perturb=True,
                )
                behavior_id = b_id
            else:
                support_flag = bool(exp_cfg.ood_enabled and self._is_ood_state(s_t))
                ood_flag = bool(support_flag)
                if "ABL_NO_SUPPORT_CHECK" in exp_cfg.ablations:
                    ood_flag = False
                    if support_flag:
                        self._last_support_reason = "support_check_disabled"
                if "ABL_NO_SAFETY" in exp_cfg.ablations:
                    ood_flag = False
                if ood_flag:
                    u_tilde, b_id, perturb_mask, noise_std = self._baseline_osv_action(
                        rng, self.pf,
                        force_behavior_id=None,
                        stage_frac=None,
                        noise_std=0.0,
                        enable_perturb=False,
                    )
                    behavior_id = "OOD_FALLBACK_" + str(b_id)
                else:
                    # Actor proposal
                    u_cont, op_idx, op_logits = self.pi.act(s_t)
                    u_actor_direct = np.concatenate([u_cont, op_logits], axis=0)
                    direct_iql = "ABL_IQL_DIRECT" in exp_cfg.ablations
                    if direct_iql:
                        u_tilde_actor = np.asarray(u_actor_direct, dtype=float)
                    else:
                        anchor = str(getattr(exp_cfg, "iql_residual_anchor", "B1") or "B1").upper()
                        try:
                            t_frac = float(self.tick_count) / float(max(1, int(exp_cfg.horizon_T) - 1))
                        except Exception:
                            t_frac = 0.0
                        u_anchor, anchor_id, _, _ = self._baseline_osv_action(
                            rng, self.pf,
                            force_behavior_id="A2" if anchor == "A2" else "B1",
                            stage_frac=t_frac if anchor == "A2" else None,
                            noise_std=0.0,
                            enable_perturb=False,
                        )
                        scale = float(np.clip(getattr(exp_cfg, "iql_residual_scale", 0.70), 0.0, 1.0))
                        u_tilde_actor = np.asarray(u_anchor, dtype=float) + scale * (
                            np.asarray(u_actor_direct, dtype=float) - np.asarray(u_anchor, dtype=float)
                        )

                    # Optional: critic-guided policy improvement over a small candidate set
                    # (IQL action + safe baselines [+ optional mix]). This often improves stability and HV/Sat.
                    if bool(getattr(exp_cfg, "iql_pi_enabled", True)) and isinstance(self.pi, IQLPolicy) and (self.offline_phase == "eval"):
                        candidates = [("IQL_DIRECT" if direct_iql else "IQL_RESID", u_tilde_actor)]

                        include_baselines = bool(getattr(exp_cfg, "iql_pi_include_baselines", True)) and (not direct_iql)
                        if include_baselines:
                            try:
                                t_frac = float(self.tick_count) / float(max(1, int(exp_cfg.horizon_T) - 1))
                            except Exception:
                                t_frac = 0.0

                            u_a2, _, _, _ = self._baseline_osv_action(
                                rng, self.pf,
                                force_behavior_id="A2",
                                stage_frac=t_frac,
                                noise_std=0.0,
                                enable_perturb=False,
                            )
                            u_b1, _, _, _ = self._baseline_osv_action(
                                rng, self.pf,
                                force_behavior_id="B1",
                                noise_std=0.0,
                                enable_perturb=False,
                            )
                            candidates.append(("A2", u_a2))
                            candidates.append(("B1", u_b1))

                            if bool(getattr(exp_cfg, "iql_pi_include_mix", True)) and (not direct_iql):
                                w_mix = float(getattr(exp_cfg, "iql_pi_mix", 0.5))
                                w_mix = float(np.clip(w_mix, 0.0, 1.0))
                                try:
                                    u_mix = w_mix * np.asarray(u_a2, dtype=float) + (1.0 - w_mix) * np.asarray(u_b1, dtype=float)
                                    candidates.append((f"MIX_{w_mix:.2f}", u_mix))
                                except Exception:
                                    pass

                        best_name = None
                        best_u = None
                        best_q = float("-inf")
                        for cname, u_can in candidates:
                            try:
                                s_can = self.osv.sample_from_action(np.asarray(u_can, dtype=float), gate_eff)
                                s_can, _, _, _ = pi_safe_osv_sample(s_can, self.pf, exp_cfg, self.algo_pool)
                                q_can = float(self.pi.q_min(s_t, s_can.applied))
                                if q_can > best_q:
                                    best_q = q_can
                                    best_name = cname
                                    best_u = np.asarray(u_can, dtype=float)
                            except Exception:
                                continue

                        if best_u is not None and best_name is not None:
                            u_tilde = best_u
                            behavior_id = f"{method_id}_PI_{best_name}"
                        else:
                            u_tilde = u_tilde_actor
                            behavior_id = method_id
                    else:
                        u_tilde = u_tilde_actor
                        behavior_id = method_id

            osv_sample = self.osv.sample_from_action(u_tilde, gate_eff)
            osv_sample.behavior_id = behavior_id
            osv_sample.profile_id = int(exp_cfg.profile_id)
            osv_sample.regime_id = int(exp_cfg.regime_id)
            osv_sample.perturb_mask = perturb_mask
            osv_sample.noise_std = float(noise_std)

            osv_sample, alpha_safe, op_idx, xi = pi_safe_osv_sample(osv_sample, self.pf, exp_cfg, self.algo_pool)
            a_latent = np.asarray(u_tilde, dtype=float)

        else:
            # Default: AWR (online) or any other experimental controller
            if not isinstance(self.pi, AWRPolicy):
                self.pi = AWRPolicy(action_dim=self.osv.total_dim, ctrl_dim=len(OSV.CHANNEL_ORDER), seed=self.seed)

            g_mean, g_std, gate_p = self.pi.predict_dist(s_t) if self.pi is not None else (
                self.osv.g_mean, self.osv.g_std, self.osv.gate_p
            )
            g_mean = np.nan_to_num(np.asarray(g_mean, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            g_std = np.nan_to_num(np.asarray(g_std, dtype=float), nan=0.25, posinf=1.0, neginf=0.25)
            g_std = np.clip(g_std, 1e-3, 5.0)
            gate_p = np.nan_to_num(np.asarray(gate_p, dtype=float), nan=0.5, posinf=0.5, neginf=0.5)
            gate_p = np.clip(gate_p, 0.0, 1.0)
            self.osv.g_mean = g_mean
            self.osv.g_std  = g_std
            self.osv.gate_p = gate_p
            self.osv.w = np.asarray(self.upv.w, dtype=float).copy()
            self.osv.r = np.asarray(self.upv.r, dtype=float).copy()

            osv_sample = self.osv.sample(rng=rng, use_gate=True, gate_eff=gate_eff)
            osv_sample, alpha_safe, op_idx, xi = pi_safe_osv_sample(osv_sample, self.pf, exp_cfg, self.algo_pool)
            behavior_id = "C0_PPCS_AWR"
            a_latent = np.asarray(self.osv.g_mean, dtype=float)

        self._last_action_latent = a_latent
        self._last_mask = gate_eff
        if self.osv is not None:
            self.osv.g_exec = np.asarray(osv_sample.applied, dtype=float).copy()
        self._capture_projection_diag(osv_sample, a_latent, gate_eff, behavior_id)

        # Causal graph: OSV action node
        _osv_node_id = None
        try:
            _osv_node_id = self.causal_graph.add_node(
                tick=self.tick_count,
                event_type="osv_action",
                payload={
                    "behavior": behavior_id,
                    "algo_probs": {k: round(v, 3) for k, v in osv_sample.algo_probs.items()},
                    "op_idx": osv_sample.op_index,
                },
            )
        except Exception:
            pass

        osv_sample.behavior_id = behavior_id
        osv_sample.perturb_mask = perturb_mask
        osv_sample.noise_std = float(noise_std)
        osv_sample.profile_id = int(exp_cfg.profile_id)
        osv_sample.regime_id = int(exp_cfg.regime_id)
        Xc = np.asarray(self.pop_X, dtype=float)
        Fc = np.asarray(self.pop_F, dtype=float)
        ch = osv_sample.channels
        o_pref = np.asarray(ch.get("pref", np.zeros(2)), dtype=float)
        o_rob = np.asarray(ch.get("rob", np.zeros(2)), dtype=float)
        o_param = np.asarray(ch.get("param", np.zeros(4)), dtype=float)
        o_model = np.asarray(ch.get("model", np.zeros(4)), dtype=float)

        # ---------- 3) 由 OSVSample 编译 ModelIR（模型重构） ----------
        # 当前偏好最优点（用于信任域中心）
        # ---------- 3) 由 OSVSample 编译 ModelIR（模型重构） ----------
        idx_best, _ = asf_distance_batch(Fc, self.upv.w, self.upv.r)
        x_best = Xc[idx_best] if idx_best is not None else None

        if "ABL_NO_MODEL_RECONF" in exp_cfg.ablations:
            # 基线/消融：禁用模型重构，仅保留最基本的 IR（并禁用 preference bias）
            b0 = self.problem.bounds
            self.model_ir = ModelIR(
                bounds=b0,
                penalty_scale=1.0,
                bias_lambda=0.0,
                trust_radius_scale=1.0,
                scalar_mode="ASF",
                obj_band=None,
                band_penalty=0.0,
                risk_alpha=float(self.upv.rho.get("cvar_alpha", 0.2)) if isinstance(self.upv.rho, dict) else 0.2,
            )
        else:
            self.model_ir = compile_model_ir(self.problem, upv_state, osv_sample, self.pf, x_best, exp_cfg=exp_cfg)
            self.model_ir = pi_safe_model_ir(self.model_ir, self.problem, exp_cfg)

        # Baselines A1/A2/A3：不做目标偏置塑形（只靠 ROI/标量化或固定策略）
        if exp_cfg.exp_group in ("A1_FIXED_SINGLE", "A2_FIXED_PORTFOLIO", "A3_UPV_DIRECT"):
            self.model_ir.bias_lambda = 0.0

        # ---------- 4) OSV 驱动算法预算分配 ----------
        method_id = exp_cfg.exp_group
        alpha = alpha_safe if 'alpha_safe' in locals() else np.array([osv_sample.algo_probs.get(name, 0.0) for name in self.algo_pool], dtype=float)
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

        # A1/A3: 固定单求解器
        if method_id in ("A1_FIXED_SINGLE", "A3_UPV_DIRECT"):
            alpha = np.zeros_like(alpha)
            target = exp_cfg.fixed_single_solver if method_id == "A1_FIXED_SINGLE" else getattr(exp_cfg, "upv_direct_solver", "NSGA-III")
            if target not in self.algo_pool:
                target = "NSGA-II" if "NSGA-II" in self.algo_pool else self.algo_pool[0]
            alpha[self.algo_pool.index(target)] = 1.0

        # A2: 固定 portfolio stage schedule（仅依赖时间，不依赖 UPV 或过程特征）
        if method_id == "A2_FIXED_PORTFOLIO":
            frac = float(self.tick_count) / max(1.0, float(exp_cfg.horizon_T - 1))
            if frac < 0.33:
                wts = {"NSGA-III": 0.35, "MOEA/D": 0.25, "NSGA-II": 0.20, "BO": 0.20}
            elif frac < 0.66:
                wts = {"MOEA/D": 0.35, "NSGA-III": 0.25, "NSGA-II": 0.20, "BO": 0.10, "RL": 0.10}
            else:
                wts = {"CP/MILP": 0.35, "MOEA/D": 0.25, "NSGA-II": 0.20, "NSGA-III": 0.10, "BO": 0.05, "RL": 0.05}
            alpha = np.array([float(wts.get(name, 0.0)) for name in self.algo_pool], dtype=float)
            alpha = alpha / alpha.sum() if alpha.sum() > 0 else np.ones_like(alpha) / len(alpha)

        # Abl-3: no solver mixture -> 固定单求解器
        if "ABL_NO_SCHEDULING" in exp_cfg.ablations and method_id not in ("A1_FIXED_SINGLE", "A3_UPV_DIRECT"):
            alpha = np.zeros_like(alpha)
            target = exp_cfg.fixed_single_solver if exp_cfg.fixed_single_solver in self.algo_pool else self.algo_pool[0]
            alpha[self.algo_pool.index(target)] = 1.0

        # Π_safe（no-safety 时跳过）
        if "ABL_NO_SAFETY" not in exp_cfg.ablations:
            alpha = np.clip(alpha, 1e-12, None)
            alpha = alpha / alpha.sum() if alpha.sum() > 0 else np.ones_like(alpha) / len(alpha)
        else:
            s = float(np.sum(alpha))
            alpha = alpha / s if s > 0 else np.ones_like(alpha) / len(alpha)

        # --- 在线信用分配修正（修复3）---
        # 根据滑动窗口中各算法的历史 ΔHV 贡献对 alpha 做乘性修正
        _credit_on = bool(getattr(exp_cfg, "credit_enabled", True))
        _credit_lam = float(getattr(exp_cfg, "credit_lambda", 0.3))
        if _credit_on and hasattr(self, "_credit_window") and len(self._credit_window) > 0:
            # 计算窗口内各算法的平均归一化信用
            credit_sum = np.zeros(len(self.algo_pool), dtype=float)
            for rec in self._credit_window:
                for i, a in enumerate(self.algo_pool):
                    credit_sum[i] += float(rec.get(a, 0.0))
            credit_avg = credit_sum / max(1, len(self._credit_window))
            c_max = float(np.max(credit_avg))
            if c_max > 1e-12:
                credit_norm = credit_avg / c_max  # 归一化到 [0, 1]
            else:
                credit_norm = np.zeros_like(credit_avg)
            # 乘性修正: alpha_adj = alpha * (1 + λ * credit_norm)
            alpha = alpha * (1.0 + _credit_lam * credit_norm)
            alpha = alpha / float(np.sum(alpha)) if float(np.sum(alpha)) > 0 else alpha

        # --- Legacy HV-oriented block scheduling ---
        # Formal Ch5 runs keep this disabled so C1 remains rule-prior + IQL
        # residual correction followed by the same projection and scheduling path.
        hv_boost_active2 = False
        if (method_id == "C1_PPCS_IQL_FULL") and bool(getattr(exp_cfg, "hv_boost_enabled", False)):
            sat_now  = float(getattr(self.pf, "sat", 0.0))
            feas_now = float(getattr(self.pf, "feas_ratio", 0.0))
            viol_now = float(getattr(self.pf, "mean_violation", 0.0))
            sat_th  = float(getattr(exp_cfg, "hv_boost_sat_th", 0.50))
            feas_th = float(getattr(exp_cfg, "hv_boost_feas_th", 0.55))
            viol_th = float(getattr(exp_cfg, "hv_boost_viol_th", 1.0))

            # CO-ARCH override: sat 尺度很小，默认忽略 sat 门控（用可行率+违反强度触发）
            pid = str(getattr(self, "problem_id", ""))
            is_coarch = pid.strip().upper() == "CO-ARCH"
            ignore_sat = bool(getattr(exp_cfg, "hv_boost_ignore_sat_on_coarch", True)) and is_coarch
            if ignore_sat:
                feas_th = max(feas_th, float(getattr(exp_cfg, "hv_boost_coarch_feas_th", 0.90)))
                viol_th = min(viol_th, float(getattr(exp_cfg, "hv_boost_coarch_viol_th", 0.50)))
                sat_ok = True
            else:
                sat_ok = (sat_now >= sat_th)

            hv_boost_active2 = sat_ok and (feas_now >= feas_th) and (viol_now <= viol_th)

        alpha_for_budget = np.asarray(alpha, dtype=float).copy()
        if hv_boost_active2 and bool(getattr(exp_cfg, "hv_block_enabled", False)):
            # init stateful block container if missing (backward compatible)
            if not hasattr(self, "_hv_block_remaining"):
                self._hv_block_remaining = 0
                self._hv_block_alpha = None
                self._hv_block_picks = []

            block_len = int(getattr(exp_cfg, "hv_block_len", 3))
            topk = int(getattr(exp_cfg, "hv_block_topk", 2))
            main_share = float(getattr(exp_cfg, "hv_block_main_share", 0.85))
            main_share = float(np.clip(main_share, 0.50, 1.0))
            topk = int(np.clip(topk, 1, 2))

            # start new block if expired
            if int(getattr(self, "_hv_block_remaining", 0)) <= 0 or getattr(self, "_hv_block_alpha", None) is None:
                # Legacy diagnostic: pick top-k solvers inside hv_boost_algos.
                hv_algos = list(getattr(exp_cfg, "hv_boost_algos", ["CP/MILP", "NSGA-III", "MOEA/D", "NSGA-II"]))
                cand = [i for i, a in enumerate(self.algo_pool) if a in set(hv_algos)]
                if len(cand) >= 1:
                    order = sorted(cand, key=lambda i: -float(alpha_for_budget[i]))
                else:
                    order = np.argsort(-alpha_for_budget).tolist()
                picks = [int(order[i]) for i in range(min(len(order), topk))]
                block_alpha = np.zeros_like(alpha_for_budget)
                if len(picks) == 1:
                    block_alpha[picks[0]] = 1.0
                else:
                    block_alpha[picks[0]] = main_share
                    block_alpha[picks[1]] = 1.0 - main_share

                s2 = float(np.sum(block_alpha))
                block_alpha = block_alpha / s2 if s2 > 1e-12 else alpha_for_budget
                self._hv_block_alpha = np.asarray(block_alpha, dtype=float)
                self._hv_block_picks = [self.algo_pool[i] for i in picks]
                self._hv_block_remaining = int(max(1, block_len))
                self.add_log("SCHED-BLOCK", f"HV-block start picks={self._hv_block_picks}, alpha_block={block_alpha.tolist()}, len={self._hv_block_remaining}")

            # 修复4：软性混合代替硬覆盖，保留策略输出的部分影响力
            # mix=0.7 表示 block 方向占 70%，策略原始输出保留 30%
            hv_block_mix = float(getattr(exp_cfg, "hv_block_mix", 0.7))
            block_target = np.asarray(getattr(self, "_hv_block_alpha", alpha_for_budget), dtype=float)
            alpha_for_budget = hv_block_mix * block_target + (1.0 - hv_block_mix) * alpha_for_budget

            # countdown
            self._hv_block_remaining = int(getattr(self, "_hv_block_remaining", 1)) - 1
            if self._hv_block_remaining <= 0:
                self.add_log("SCHED-BLOCK", f"HV-block end picks={getattr(self, '_hv_block_picks', [])}")
                self._hv_block_alpha = None
                self._hv_block_picks = []
        else:
            # Reset any existing legacy block when inactive.
            if hasattr(self, "_hv_block_remaining") and int(getattr(self, "_hv_block_remaining", 0)) > 0:
                self.add_log("SCHED-BLOCK", "HV-block reset (hv_boost inactive)")
            self._hv_block_remaining = 0
            self._hv_block_alpha = None
            self._hv_block_picks = []

        alpha = np.asarray(alpha_for_budget, dtype=float)
        alpha = np.clip(alpha, 1e-12, None)
        alpha = alpha / float(np.sum(alpha)) if float(np.sum(alpha)) > 0 else np.ones_like(alpha) / len(alpha)

        B_total = int(getattr(exp_cfg, "epoch_budget_B", self.pop_size * self.gen_per_tick))
        dispatch_mode = getattr(exp_cfg, "dispatch_mode", "portfolio")

        if dispatch_mode == "single":
            # --- Single mode: 采样选一个算法，全部预算给它 ---
            chosen = select_algo_by_osv(self, alpha, self.algo_pool)
            B_floor = np.zeros(len(self.algo_pool), dtype=int)
            chosen_idx = self.algo_pool.index(chosen) if chosen in self.algo_pool else 0
            B_floor[chosen_idx] = B_total
            self.add_log("SCHED", f"mode=single, chosen={chosen}, B_total={B_total}")
        else:
            # --- Portfolio mode: 按概率比例分配预算给所有算法 ---
            B_floor = np.floor(alpha * B_total).astype(int)
            rem = B_total - int(np.sum(B_floor))
            if rem > 0:
                order = np.argsort(-alpha)
                for i in range(rem):
                    B_floor[order[i % len(order)]] += 1
            self.add_log("SCHED", f"mode=portfolio, alpha={alpha.tolist()}, budget={B_floor.tolist()}, B_total={B_total}")

        self._last_alpha = np.array(alpha, dtype=float)
        self._last_budget = np.array(B_floor, dtype=int)
        self._last_B_total = int(B_total)
        self._last_dispatch_mode = dispatch_mode
        _diag = dict(getattr(self, "_last_projection_diag", {}) or {})
        _diag["budget_total"] = int(B_total)
        _diag["dispatch_mode"] = str(dispatch_mode)
        self._last_projection_diag = _diag

        # ---------- 5) OSV 驱动参数自适应（不同算法从 OSV 的不同通道采样） ----------
        knobs = self._apply_osv_knobs(osv_sample.applied, exp_cfg)

        # A3: UPV-direct 编译为 ROI 偏置（不属于操作符库 R）
        # C1: PPCS 完整模式同样启用 ROI 偏置, 让 UPV.r 真正干预 NSGA-II 种群分布,
        # 否则 v_full 行为与 v_baseline_portfolio (A2) 几乎等价 (都走 π_rule 不读 UPV.r).
        # ABL_NO_OSV_PREF 强制关闭 ROI 偏置 (相当于 OSV.pref 通道完全失效).
        # v6 (2026-05-08): 提到 0.95 让 v_full 真正聚焦决策者 ROI, 拉开与 baseline 的偏好域差距.
        if method_id == "A3_UPV_DIRECT":
            self.emo_params["roi_bias"] = float(getattr(exp_cfg, "upv_direct_roi_bias", 0.95))
        elif method_id in ("A1_FIXED_SINGLE", "A2_FIXED_PORTFOLIO"):
            self.emo_params["roi_bias"] = 0.0
        elif method_id in ("C1_PPCS_IQL_FULL", "C2_PPCS_IQL_NO_SAFETY"):
            if "ABL_NO_OSV_PREF" in exp_cfg.ablations:
                self.emo_params["roi_bias"] = 0.0
            else:
                self.emo_params["roi_bias"] = float(getattr(exp_cfg, "upv_direct_roi_bias", 0.95))

        op_apply = int(op_idx) if 'op_idx' in locals() else 0
        self._last_op_raw = int(op_idx) if 'op_idx' in locals() else 0
        if "ABL_NO_MODEL_RECONF" in exp_cfg.ablations:
            op_apply = 0
        self._last_op_apply = int(op_apply)
        _diag = dict(getattr(self, "_last_projection_diag", {}) or {})
        _diag["op_raw"] = int(getattr(self, "_last_op_raw", 0))
        _diag["op_apply"] = int(getattr(self, "_last_op_apply", 0))
        _diag["operator_suppressed"] = int(int(getattr(self, "_last_op_raw", 0)) != int(getattr(self, "_last_op_apply", 0)))
        self._last_projection_diag = _diag
        self._apply_operator(op_apply, osv_sample.applied, xi)

        # ---------- 6) 预算分配执行 ----------
        upv_opt = make_neutral_upv(self.upv) if (method_id in ("A1_FIXED_SINGLE", "A2_FIXED_PORTFOLIO") or ("ABL_NO_UPV" in exp_cfg.ablations)) else self.upv


        # baseline snapshot (elitism): always keep current pop in candidate pool
        X0 = np.asarray(self.pop_X, dtype=float) if self.pop_X is not None else None
        if X0 is None or len(X0) == 0:
            self.init_population()
            X0 = np.asarray(self.pop_X, dtype=float)
        try:
            vals0 = self.problem.evaluate(X0, return_values_of=["F", "G"])
            if isinstance(vals0, (tuple, list)) and len(vals0) >= 2:
                F0, G0 = vals0[0], vals0[1]
            elif isinstance(vals0, dict):
                F0, G0 = vals0.get("F"), vals0.get("G")
            else:
                F0, G0 = vals0, None
        except Exception:
            F0 = np.asarray(self.pop_F, dtype=float) if self.pop_F is not None else None
            G0 = np.asarray(self.pop_G, dtype=float) if getattr(self, 'pop_G', None) is not None else None
        n_constr0 = int(getattr(self.problem, "n_constr", 0) or 0)
        if n_constr0 <= 0:
            G0 = None
        if F0 is None:
            F0 = np.zeros((len(X0), self.n_obj), dtype=float)
        else:
            F0 = np.asarray(F0, dtype=float)
        if G0 is not None:
            G0 = np.asarray(G0, dtype=float)

        X_candidates = [X0.copy()]
        F_candidates = [F0.copy()]
        G_candidates = [None if G0 is None else G0.copy()]
        algo_eval = [("ELITE", 0)]

        # --- baseline HV for credit assignment (修复3) ---
        _credit_hv_base = 0.0
        _credit_enabled = bool(getattr(exp_cfg, "credit_enabled", True))
        _credit_algo_dhv = {}
        if _credit_enabled:
            try:
                _feas0 = F0
                n_c0 = int(getattr(self.problem, "n_constr", 0) or 0)
                if n_c0 > 0 and G0 is not None:
                    cv0 = np.sum(np.maximum(0.0, G0), axis=1)
                    _feas0 = F0[cv0 <= 1e-12]
                if len(_feas0) > 0:
                    _credit_hv_base = float(HV(ref_point=self.hv_ref)(_feas0))
            except Exception:
                pass

        # parallel-portfolio semantics: each algorithm runs from the same baseline snapshot
        for algo_name, b in zip(self.algo_pool, B_floor):
            if b <= 0:
                algo_eval.append((algo_name, 0))
                continue

            # restore baseline before running this algorithm
            self.pop_X = X0.copy()
            self.pop_F = F0.copy()
            self.pop_G = None if G0 is None else G0.copy()

            self.algo_name = algo_name

            try:
                if algo_name in ["NSGA-II", "NSGA-III", "MOEA/D"]:
                    gen_backup = self.gen_per_tick
                    pop_backup = self.pop_size
                    try:
                        b_int = int(b)
                    except Exception:
                        b_int = int(pop_backup)

                    # choose effective pop size so that small budgets can still evolve (>=2 gens when possible)
                    pop_eff = int(min(pop_backup, max(8, int(b_int // 2))))
                    if pop_eff < pop_backup:
                        try:
                            rng = np.random.default_rng(int(self.seed) + 991 * int(getattr(self, 'tick_count', 0)) + 17 * int(self.algo_pool.index(algo_name)))
                            idxs = rng.choice(len(self.pop_X), size=pop_eff, replace=(len(self.pop_X) < pop_eff))
                            self.pop_X = np.asarray(self.pop_X, dtype=float)[idxs]
                            self.pop_F = np.asarray(self.pop_F, dtype=float)[idxs]
                            if self.pop_G is not None:
                                self.pop_G = np.asarray(self.pop_G, dtype=float)[idxs]
                        except Exception:
                            pass

                    self.pop_size = pop_eff
                    if b_int >= 2 * pop_eff:
                        self.gen_per_tick = max(2, int(np.ceil(float(b_int) / max(1.0, float(pop_eff)))))
                    else:
                        self.gen_per_tick = max(1, int(np.ceil(float(b_int) / max(1.0, float(pop_eff)))))

                    self._step_emo(upv_opt=upv_opt, budget=b_int)
                    self.gen_per_tick = gen_backup
                    self.pop_size = pop_backup

                elif algo_name == "BO":
                    self._step_bo(upv_opt=upv_opt, budget=b)
                elif algo_name == "CP/MILP":
                    self._step_cpm(upv_opt=upv_opt, budget=b)
                elif algo_name == "RL":
                    self._step_rl(upv_opt=upv_opt, budget=b)
                else:
                    self._step_emo(upv_opt=upv_opt, budget=b_int)
            except Exception as exc:
                # 修复5：算法执行失败保护 — 记录并跳过
                self.add_log("SOLVER-ERR", f"{algo_name} failed: {exc}")
                self.pop_X = X0.copy()
                self.pop_F = F0.copy()
                self.pop_G = None if G0 is None else G0.copy()

            X_candidates.append(np.asarray(self.pop_X, dtype=float))
            F_candidates.append(np.asarray(self.pop_F, dtype=float))
            G_candidates.append(np.asarray(self.pop_G, dtype=float) if (getattr(self, 'pop_G', None) is not None) else None)
            algo_eval.append((algo_name, int(b)))

            # --- per-algorithm ΔHV credit (修复3) ---
            if _credit_enabled and b > 0:
                try:
                    F_algo = np.asarray(self.pop_F, dtype=float)
                    n_c0 = int(getattr(self.problem, "n_constr", 0) or 0)
                    if n_c0 > 0 and self.pop_G is not None:
                        cv_a = np.sum(np.maximum(0.0, np.asarray(self.pop_G, dtype=float)), axis=1)
                        F_algo = F_algo[cv_a <= 1e-12]
                    hv_algo = float(HV(ref_point=self.hv_ref)(F_algo)) if len(F_algo) > 0 else 0.0
                    _credit_algo_dhv[algo_name] = max(0.0, hv_algo - _credit_hv_base)
                except Exception:
                    _credit_algo_dhv[algo_name] = 0.0

        self.last_algo_real = "MIX"

        # --- update credit assignment sliding window (修复3) ---
        if _credit_enabled and _credit_algo_dhv:
            if not hasattr(self, "_credit_window"):
                self._credit_window = deque(maxlen=int(getattr(exp_cfg, "credit_window", 5)))
            self._credit_window.append(_credit_algo_dhv)
            self.add_log("CREDIT", f"dhv={{{', '.join(f'{k}:{v:.4f}' for k,v in _credit_algo_dhv.items())}}}")

        # Merge candidates (constraint-first). If constraints exist, re-evaluate G on base problem for consistency.
        if X_candidates:
            X_all = np.vstack([np.asarray(x, dtype=float) for x in X_candidates if x is not None and len(x) > 0])
            F_all = np.vstack([np.asarray(f, dtype=float) for f in F_candidates if f is not None and len(f) > 0])

            n_constr = int(getattr(self.problem, "n_constr", 0) or 0)
            G_all = None
            if n_constr > 0:
                try:
                    vals = self.problem.evaluate(X_all, return_values_of=["G"])
                    if isinstance(vals, dict):
                        G_all = vals.get("G")
                    elif isinstance(vals, (tuple, list)):
                        G_all = vals[0] if len(vals) > 0 else None
                    else:
                        G_all = vals
                    if G_all is not None:
                        G_all = np.asarray(G_all, dtype=float)
                except Exception:
                    G_all = None

            if n_constr <= 0 or G_all is None or G_all.size == 0:
                nd = is_nondominated(F_all)
                X_nd = X_all[nd]; F_nd = F_all[nd]
                if len(X_nd) > self.pop_size:
                    cd = crowding_distance(F_nd)
                    order = np.argsort(-cd)[:self.pop_size]
                    self.pop_X = X_nd[order]
                    self.pop_F = F_nd[order]
                    self.pop_G = None
                else:
                    self.pop_X = X_nd
                    self.pop_F = F_nd
                    self.pop_G = None
            else:
                cv = np.sum(np.maximum(0.0, np.asarray(G_all, dtype=float)), axis=1)
                feas = cv <= 1e-12
                X_feas = X_all[feas]
                F_feas = F_all[feas]
                G_feas = G_all[feas]

                X_sel = []
                F_sel = []
                G_sel = []

                if len(X_feas) > 0:
                    nd = is_nondominated(F_feas)
                    idx_nd = np.where(nd)[0]
                    if len(idx_nd) >= self.pop_size:
                        cd = crowding_distance(F_feas[idx_nd])
                        pick = idx_nd[np.argsort(-cd)[:self.pop_size]]
                        self.pop_X = X_feas[pick]
                        self.pop_F = F_feas[pick]
                        self.pop_G = G_feas[pick]
                    else:
                        pick_nd = idx_nd.tolist()
                        X_sel.extend(X_feas[pick_nd])
                        F_sel.extend(F_feas[pick_nd])
                        G_sel.extend(G_feas[pick_nd])

                        remain = [i for i in range(len(X_feas)) if i not in set(pick_nd)]
                        if remain:
                            cd_all = crowding_distance(F_feas)
                            remain_sorted = sorted(remain, key=lambda i: -cd_all[i])
                            need = max(0, self.pop_size - len(X_sel))
                            for i in remain_sorted[:need]:
                                X_sel.append(X_feas[i])
                                F_sel.append(F_feas[i])
                                G_sel.append(G_feas[i])

                if len(X_sel) < self.pop_size:
                    order = np.argsort(cv)
                    for idx in order:
                        if feas[idx]:
                            continue
                        X_sel.append(X_all[idx])
                        F_sel.append(F_all[idx])
                        G_sel.append(G_all[idx])
                        if len(X_sel) >= self.pop_size:
                            break

                self.pop_X = np.asarray(X_sel[:self.pop_size], dtype=float)
                self.pop_F = np.asarray(F_sel[:self.pop_size], dtype=float)
                self.pop_G = np.asarray(G_sel[:self.pop_size], dtype=float)


        # --- ARCH/RESCUE hook (v11) ---
        # Update cumulative feasible archive using *all* candidates produced in this tick.
        try:
            if 'X_all' in locals() and 'F_all' in locals():
                self._update_feasible_archive(X_all, F_all, locals().get('G_all', None), exp_cfg=exp_cfg, tag=f"tick_{int(getattr(self, 'tick_count', 0))}")
        except Exception:
            pass

        # If feasibility collapses (or becomes too low), rescue from archive or rollback to previous pop.
        try:
            n_constr = int(getattr(self.problem, 'n_constr', 0) or 0)
            if n_constr > 0:
                G_pop = getattr(self, 'pop_G', None)
                feas_ratio_pop = 0.0
                if G_pop is not None and np.size(G_pop) > 0:
                    G_pop = np.asarray(G_pop, dtype=float)
                    if G_pop.ndim == 1:
                        G_pop = G_pop.reshape(-1, 1)
                    cv_pop = np.sum(np.maximum(0.0, G_pop), axis=1) if len(G_pop) else np.zeros((0,), dtype=float)
                    feas_ratio_pop = float(np.mean(cv_pop <= 1e-12)) if len(cv_pop) else 0.0
                else:
                    # fallback: approximate using UPV constraints only
                    feas_flags = []
                    for i in range(len(self.pop_X)):
                        _, feas_i = compute_penalty_and_feas(np.asarray(self.pop_X[i], dtype=float), np.asarray(self.pop_F[i], dtype=float), self.upv, self.enable_cr)
                        feas_flags.append(bool(feas_i))
                    feas_ratio_pop = float(np.mean(feas_flags)) if feas_flags else 0.0

                rescue_th = float(getattr(exp_cfg, 'feasible_rescue_th', 0.10))
                # Only rollback when feasibility *collapses* (prev>0 -> now==0).
                # If feasibility has been zero all along (hard constrained problems), rolling back every tick
                # would freeze exploration and make all methods identical.
                if feas_ratio_pop <= 0.0 and bool(getattr(exp_cfg, 'rollback_on_zero_feas', True)):
                    feas_prev = 0.0
                    if G_prev is not None and np.size(G_prev) > 0:
                        try:
                            Gp = np.asarray(G_prev, dtype=float)
                            if Gp.ndim == 1:
                                Gp = Gp.reshape(-1, 1)
                            cvp = np.sum(np.maximum(0.0, Gp), axis=1) if len(Gp) else np.zeros((0,), dtype=float)
                            feas_prev = float(np.mean(cvp <= 1e-12)) if len(cvp) else 0.0
                        except Exception:
                            feas_prev = 0.0

                    if feas_prev > 0.0:
                        # First try archive rescue; if not possible, rollback.
                        rescued = self._rescue_population_from_archive(exp_cfg, reason='feas=0')
                        if not rescued:
                            self.pop_X = np.asarray(X_prev, dtype=float)
                            self.pop_F = np.asarray(F_prev, dtype=float)
                            self.pop_G = (None if G_prev is None else np.asarray(G_prev, dtype=float))
                            self.add_log('ROLLBACK', 'feasibility collapsed -> rollback to previous population')
                    else:
                        # still zero feasible: keep searching (no rollback)
                        if not bool(getattr(self, "_skip_zero_feas_rollback_logged", False)):
                            self.add_log('INFO', 'feasibility is still zero; skip rollback to keep exploring')
                            setattr(self, "_skip_zero_feas_rollback_logged", True)
                elif feas_ratio_pop < rescue_th:
                    # soft rescue when feasibility is too low
                    self._rescue_population_from_archive(exp_cfg, reason=f'feas={feas_ratio_pop:.3f}<th')
        except Exception:
            pass

        self.algo_name = old_algo  # 恢复 UI 选择

        # ---------- 7) Metrics & reward terms (Eq.(11), A.0.3~A.0.5) ----------
        X = np.asarray(self.pop_X, dtype=float)
        F = np.asarray(self.pop_F, dtype=float)

        feas_flags, violations, F_feas = [], [], []
        G = getattr(self, 'pop_G', None)
        use_g = (G is not None and np.size(G) > 0)
        if use_g:
            G = np.asarray(G, dtype=float)
        for i in range(len(X)):
            f_i = F[i].astype(float)
            if use_g and i < len(G):
                g_i = np.asarray(G[i], dtype=float).ravel()
                v_i = float(np.sum(np.maximum(0.0, g_i)))
                feas = v_i <= 1e-12
                violations.append(v_i)
                feas_flags.append(bool(feas))
            else:
                _, feas = compute_penalty_and_feas(X[i], f_i, self.upv, self.enable_cr)
                feas_flags.append(bool(feas))
                violations.append(compute_violation(X[i], f_i, self.upv))
            if feas:
                F_feas.append(f_i)
        if self.last_epoch_fail > 0:
            violations += [float(self.violation_fail_surrogate)] * int(self.last_epoch_fail)
        feas_ratio2 = float(np.mean(feas_flags)) if len(feas_flags) else 0.0
        mean_violation = float(np.mean(violations)) if violations else 0.0
        self.viol_normer.update(mean_violation)
        _, _, _, v_p95 = self.viol_normer._stats()
        viol_norm = float(min(1.0, mean_violation / max(1e-6, v_p95)))

        F_feas = np.asarray(F_feas, dtype=float)
        hv_indicator = HV(ref_point=self.hv_ref)

        # HV on archive (best-so-far) makes curves more stable / interpretable.
        F_hv = F_feas
        try:
            exp_cfg_hv = exp_cfg if 'exp_cfg' in locals() else getattr(self, 'experiment_config', ExperimentConfig())

            if bool(getattr(exp_cfg_hv, 'hv_use_archive', False)):
                F_arch = getattr(self, 'feas_archive_F', None)
                if F_arch is not None and len(F_arch) > 0:
                    F_hv = np.asarray(F_arch, dtype=float)
        except Exception:
            pass

        hv = float(hv_indicator(F_hv)) if (hasattr(F_hv, 'size') and F_hv.size) else 0.0
        fmin = np.min(F_hv, axis=0).tolist() if (hasattr(F_hv, 'size') and F_hv.size) else (np.min(F, axis=0).tolist() if F.size else [])
        hv_prev = self.history_hv_feas[-1] if self.history_hv_feas else hv
        delta_hv_raw = float(hv - hv_prev)
        self.delta_hv_raw = delta_hv_raw
        beta = float(exp_cfg.hv_beta)
        delta_hv_smooth = beta * float(self.delta_hv_smooth) + (1.0 - beta) * delta_hv_raw
        self.delta_hv_smooth = delta_hv_smooth
        stall_flag = 1.0 if delta_hv_smooth < float(exp_cfg.hv_eps) else 0.0
        self.dhv_normer.update(delta_hv_smooth)
        dhv_norm = float(self.dhv_normer.norm_pos(delta_hv_smooth))

        exp_cfg_sat = exp_cfg if 'exp_cfg' in locals() else getattr(self, 'experiment_config', ExperimentConfig())

        sat_use_archive = bool(getattr(exp_cfg_sat, 'sat_use_archive', True))

        if sat_use_archive:

            X_sat = getattr(self, 'feas_archive_X', None)

            F_sat = getattr(self, 'feas_archive_F', None)

            if X_sat is not None and F_sat is not None and len(F_sat) > 0:

                sat, pref_gap = compute_sat_prefgap(np.asarray(X_sat, dtype=float), np.asarray(F_sat, dtype=float), self.upv, self.enable_cr)

            else:

                sat, pref_gap = compute_sat_prefgap(X, F, self.upv, self.enable_cr)

        else:

            sat, pref_gap = compute_sat_prefgap(X, F, self.upv, self.enable_cr)
        diversity = float(np.mean(np.var(F, axis=0))) if F.size else 0.0
        conf = float(compute_confidence_score(self))
        alpha_cvar = float(self.upv.rho.get("cvar_alpha", 0.9))
        cvar2 = self._estimate_cvar(alpha_cvar) or 0.0
        risk2 = float(np.tanh(cvar2 / (1.0 + abs(cvar2))))

        cost_t = int(self.last_epoch_eval)  # last_epoch_eval already counts failed evals
        fail_t = int(self.last_epoch_fail)
        cost_norm = float(np.clip(cost_t / max(1.0, float(exp_cfg.epoch_budget_B)), 0.0, 1.0))
        delta_interaction = int(self.interaction_events - self._last_interaction_events)
        query_cost = float(1.0 if delta_interaction > 0 else 0.0)
        query_norm = float(np.clip(query_cost / max(1.0, float(exp_cfg.query_budget)), 0.0, 1.0))
        self._last_interaction_events = int(self.interaction_events)

        self.history_hv.append(hv)
        self.history_hv_feas.append(hv)
        self.history_fmin.append(fmin)
        self.history_dhv_raw.append(delta_hv_raw)
        self.history_dhv_norm.append(dhv_norm)
        self.history_feasible.append(feas_ratio2)
        self.history_penalty.append(mean_violation)
        self.history_viol.append(mean_violation)
        self.history_diversity.append(diversity)
        self.history_sat.append(sat)
        self.history_pref_gap.append(pref_gap)
        self.history_conf.append(conf)
        self.history_risk.append(risk2)
        self.history_dhv_smooth.append(delta_hv_smooth)

        # Causal graph: metric change node + state snapshot
        try:
            _src_ids = [_osv_node_id] if '_osv_node_id' in dir() and _osv_node_id is not None else []
            self.causal_graph.add_node(
                tick=self.tick_count,
                event_type="metric_change",
                source_ids=_src_ids,
                payload={
                    "hv": round(hv, 6),
                    "feas": round(feas_ratio2, 4),
                    "sat": round(sat, 4),
                    "algo": self.algo_name,
                },
            )
            # Save snapshot every 10 ticks (for replay)
            if self.tick_count % 10 == 0 or self.tick_count <= 3:
                self.causal_graph.save_snapshot(self.tick_count, {
                    "w": self.upv.w.tolist(),
                    "r": self.upv.r.tolist(),
                    "tau": self.upv.tau,
                    "hv": hv,
                    "feas": feas_ratio2,
                    "sat": sat,
                    "algo": self.algo_name,
                })
        except Exception:
            pass

        # IGD (if available)
        igd = None
        try:
            if self.pf_ref is not None:
                igd = float(IGD(self.pf_ref)(F))
        except Exception:
            igd = None
        self.history_igd.append(float(igd) if igd is not None else float('nan'))

        prev_feas = self.history_feasible[-2] if len(self.history_feasible) >= 2 else 0.0
        prev_viol = self.history_viol[-2] if len(self.history_viol) >= 2 else 0.0
        prev_div = self.history_diversity[-2] if len(self.history_diversity) >= 2 else 0.0
        prev_sat = self.history_sat[-2] if len(self.history_sat) >= 2 else 0.0
        prev_conf = self.history_conf[-2] if len(self.history_conf) >= 2 else 0.0
        prev_risk = self.history_risk[-2] if len(self.history_risk) >= 2 else 0.0

        pf_next = ProcessFeatures(
            d_hv_smooth=delta_hv_smooth,
            stall_flag=stall_flag,
            feas_ratio=feas_ratio2,
            mean_violation=mean_violation,
            diversity=diversity,
            sat=sat,
            conf=conf,
            risk_proxy=risk2,
            d_feas_ratio=feas_ratio2 - prev_feas,
            d_mean_violation=mean_violation - prev_viol,
            d_diversity=diversity - prev_div,
            d_sat=sat - prev_sat,
            d_conf=conf - prev_conf,
            d_risk_proxy=risk2 - prev_risk
        )
        self.pf = pf_next
        self._update_pref_gp(pf_next.as_vector())

        # s_{t+1}
        pf_state_next = ProcessFeatures() if ("ABL_NO_PROCESS_FEATURES" in exp_cfg.ablations) else pf_next
        upv_state_next = make_neutral_upv(self.upv) if ("ABL_NO_UPV" in exp_cfg.ablations) else self.upv
        s_next = build_state_vector(upv_state_next, pf_state_next, ir_summary=ir_summary)

        reward_parts = {
            "delta_hv_raw": delta_hv_raw,
            "delta_hv_smooth": delta_hv_smooth,
            "dhv_norm": dhv_norm,
            "viol_raw": mean_violation,
            "viol_norm": viol_norm,
            "sat": sat,
            "pref_gap": pref_gap,
            "cost": cost_t,
            "cost_norm": cost_norm,
            "query_cost": query_cost,
            "query_norm": query_norm
        }
        # ---------- 8) Update policy & offline dataset (Eq.(12)) ----------
        u_exec = np.asarray(osv_sample.applied, dtype=float)
        xi_vec = np.asarray(xi, dtype=float)
        self._update_osv_and_algorithm(
            s_t=s_t,
            a_latent=a_latent,
            mask=gate_eff,
            u_exec=u_exec,
            s_next=s_next,
            reward_parts=reward_parts,
            alpha=alpha,
            l_op=int(getattr(osv_sample, "op_index", 0)),
            xi=xi_vec,
            hv=hv,
            feas_ratio=feas_ratio2,
            mean_violation=mean_violation,
            diversity=diversity,
            osv_sample=osv_sample
        )
        if (exp_cfg.exp_group in ("C1_PPCS_IQL_FULL", "C2_PPCS_IQL_NO_SAFETY") and isinstance(self.pi, IQLPolicy)):
            if self.offline_phase == "collect" and len(self.offline_dataset.transitions) >= exp_cfg.offline_min_transitions:
                self.offline_phase = "train"
            if self.offline_phase == "train" and isinstance(self.pi, IQLPolicy) and not self.offline_trained:
                for k in range(int(exp_cfg.offline_train_steps)):
                    self.pi.maybe_train(self.tick_count + k)
                self.offline_trained = True
                self.offline_phase = "eval"
        # Update pop_birth_tick: inherit old birth ticks, new individuals get current tick
        if self.pop_X is not None and len(self.pop_X) > 0:
            new_birth = np.full(len(self.pop_X), self.tick_count, dtype=int)
            for i in range(len(self.pop_X)):
                key = tuple(np.asarray(self.pop_X[i], dtype=float).tolist())
                if key in _old_birth:
                    new_birth[i] = _old_birth[key]
            self.pop_birth_tick = new_birth

        self.tick_count += 1

        # ---- Convergence assessment (post-tick) ----
        try:
            cs = self.convergence_detector.check(self)
            self.convergence_status = cs
            self.converged = cs.converged
            if cs.converged and not self.conv_hint:
                self.conv_hint = "HV stagnation detected"
            if cs.should_stop:
                self.running = False
                self.stop_event.set()
                self.add_log("CONV", f"Auto-stop: {cs.suggestion_text}")
            elif cs.should_suggest:
                # 通过已有 L5 队列推送对话消息（复用 flush_l5_into_chat）
                self.enqueue_l5("warn", f"[收敛评估] {cs.suggestion_text}",
                                ["停止并审查方案", "再跑10轮", "调整偏好继续"])
                self.add_log("CONV", f"Suggest stop: score={cs.convergence_score:.2f}")
        except Exception as e:
            self.add_log("CONV", f"convergence check error: {e}")

    def _update_osv_and_algorithm(self,
                                  s_t: Optional[np.ndarray] = None,
                                  a_latent: Optional[np.ndarray] = None,
                                  mask: Optional[np.ndarray] = None,
                                  u_exec: Optional[np.ndarray] = None,
                                  s_next: Optional[np.ndarray] = None,
                                  reward_parts: Optional[Dict[str, float]] = None,
                                  alpha: Optional[np.ndarray] = None,
                                  l_op: int = 0,
                                  xi: Optional[np.ndarray] = None,
                                  hv: Optional[float] = None,
                                  feas_ratio: Optional[float] = None,
                                  mean_violation: Optional[float] = None,
                                  diversity: Optional[float] = None,
                                  osv_sample: Optional[OSVSample] = None):
        """Update policy and log transition (Eq.(11)(12))."""
        if s_t is None:
            s_t = getattr(self, "_last_state_vec", None)
        else:
            self._last_state_vec = s_t
        if a_latent is None:
            a_latent = getattr(self, "_last_action_latent", None)
        else:
            self._last_action_latent = a_latent
        if mask is None:
            mask = getattr(self, "_last_mask", None)
        else:
            self._last_mask = mask

        exp_cfg = copy.deepcopy(getattr(self, "experiment_config", ExperimentConfig()))
        exp_cfg.exp_group = normalize_method_id(exp_cfg.exp_group)
        # 论文 C2: no-safety (disable Π_safe & confidence gating)
        if exp_cfg.exp_group == "C2_PPCS_IQL_NO_SAFETY":
            for _a in ("ABL_NO_SAFETY", "ABL_NO_GATING"):
                if _a not in exp_cfg.ablations:
                    exp_cfg.ablations.append(_a)
        # 论文 A1/A2/A3: baseline (disable param adaptation & model reconfiguration by definition)
        if exp_cfg.exp_group in ("A1_FIXED_SINGLE", "A2_FIXED_PORTFOLIO", "A3_UPV_DIRECT"):
            for _a in ("ABL_NO_PARAM", "ABL_NO_MODEL_RECONF"):
                if _a not in exp_cfg.ablations:
                    exp_cfg.ablations.append(_a)
            if exp_cfg.exp_group in ("A1_FIXED_SINGLE", "A3_UPV_DIRECT") and "ABL_NO_SCHEDULING" not in exp_cfg.ablations:
                exp_cfg.ablations.append("ABL_NO_SCHEDULING")
        if reward_parts is None:
            reward_parts = {
                "delta_hv_raw": 0.0,
                "delta_hv_smooth": 0.0,
                "dhv_norm": 0.0,
                "viol_raw": 0.0,
                "viol_norm": 0.0,
                "sat": 0.0,
                "pref_gap": 0.0,
                "cost": 0.0,
                "cost_norm": 0.0,
                "query_cost": 0.0,
                "query_norm": 0.0
            }
        hv_scale = float(getattr(exp_cfg, "reward_hv_scale", 1.0))
        # 仅对 IQL 组别使用 HV reward 放大，避免影响基线/规则组的统计与可复现性
        if exp_cfg.exp_group not in ("C1_PPCS_IQL_FULL", "C2_PPCS_IQL_NO_SAFETY"):
            hv_scale = 1.0

        def _sf(v, default=0.0):
            try:
                v = float(v)
            except Exception:
                return float(default)
            if not np.isfinite(v):
                return float(default)
            return float(v)

        r_t = (
            hv_scale * _sf(reward_parts.get("dhv_norm", 0.0), 0.0)
            - float(exp_cfg.lambda_c) * _sf(reward_parts.get("viol_norm", 0.0), 0.0)
            - float(exp_cfg.lambda_p) * _sf(reward_parts.get("pref_gap", 0.0), 0.0)
            - float(exp_cfg.lambda_b) * _sf(reward_parts.get("cost_norm", 0.0), 0.0)
            - float(exp_cfg.lambda_q) * _sf(reward_parts.get("query_norm", 0.0), 0.0)
        )
        if not np.isfinite(r_t):
            r_t = 0.0

        if isinstance(self.pi, IQLPolicy):
            if s_t is not None and u_exec is not None and s_next is not None:
                self.pi.push_transition(s_t, u_exec, r_t, s_next, extra_info=reward_parts)
                if (not getattr(exp_cfg, "iql_freeze", False)) and getattr(self, "offline_phase", "eval") != "collect":
                    self.pi.maybe_train(self.tick_count)
        else:
            if self.pi is not None and s_t is not None and a_latent is not None and mask is not None:
                self.pi.push(s_t, a_latent, mask, r_t)
                self.pi.maybe_train(self.tick_count)
            else:
                # baseline controllers / no-policy modes: skip policy update silently
                if self.pi is None:
                    pass
                else:
                    if not getattr(self, "_warned_osv_missing", False):
                        self.add_log("WARN", "OSV update skipped: missing state/action; will resume after next full tick.")
                        self._warned_osv_missing = True

        if u_exec is not None and s_t is not None and s_next is not None:
            alpha = np.asarray(alpha if alpha is not None else np.zeros(len(self.algo_pool)), dtype=float)
            xi = np.asarray(xi if xi is not None else np.zeros(6, dtype=float), dtype=float)
            behavior_id = str(getattr(osv_sample, "behavior_id", ""))
            g_vec = np.asarray(getattr(osv_sample, "g_vec", np.zeros_like(u_exec)), dtype=np.float32)
            perturb_mask = getattr(osv_sample, "perturb_mask", None)
            noise_std = float(getattr(osv_sample, "noise_std", 0.0))
            profile_id = int(getattr(osv_sample, "profile_id", 0))
            regime_id = int(getattr(osv_sample, "regime_id", 0))
            tr = Transition(
                s=np.asarray(s_t, dtype=np.float32),
                u=np.asarray(u_exec, dtype=np.float32),
                r=np.float32(r_t),
                s_next=np.asarray(s_next, dtype=np.float32),
                t=np.int32(self.tick_count),
                seed=np.int32(self.seed),
                profile_id=np.int32(profile_id),
                regime_id=np.int32(regime_id),
                behavior_id=behavior_id,
                g_vec=g_vec,
                perturb_mask=None if perturb_mask is None else np.asarray(perturb_mask, dtype=bool),
                noise_std=np.float32(noise_std),
                dhv_raw=np.float32(reward_parts.get("delta_hv_raw", 0.0)),
                dhv_smooth=np.float32(reward_parts.get("delta_hv_smooth", 0.0)),
                dhv_norm=np.float32(reward_parts.get("dhv_norm", 0.0)),
                viol_raw=np.float32(reward_parts.get("viol_raw", 0.0)),
                viol_norm=np.float32(reward_parts.get("viol_norm", 0.0)),
                sat=np.float32(reward_parts.get("sat", 0.0)),
                pref_gap=np.float32(reward_parts.get("pref_gap", 0.0)),
                cost=np.int32(reward_parts.get("cost", 0.0)),
                cost_norm=np.float32(reward_parts.get("cost_norm", 0.0)),
                query_cost=np.float32(reward_parts.get("query_cost", 0.0)),
                query_norm=np.float32(reward_parts.get("query_norm", 0.0)),
                alpha_decoded=np.asarray(alpha, dtype=np.float32),
                op_idx=np.int16(l_op),
                xi=np.asarray(xi, dtype=np.float32),
                hv_feas=np.float32(hv if hv is not None else 0.0),
                feas_ratio=np.float32(feas_ratio if feas_ratio is not None else 0.0),
                mean_violation=np.float32(mean_violation if mean_violation is not None else 0.0),
                diversity=np.float32(diversity if diversity is not None else 0.0),
            )
            self.offline_dataset.add(tr)
            if self.tick_count % 50 == 0:
                self.offline_dataset.save_npz()

        if self.last_algo_real:
            st = self.algo_stats.setdefault(self.last_algo_real, {"n": 0, "sum_reward": 0.0})
            st["n"] += 1
            st["sum_reward"] += float(r_t)

    def _update_history(self):
        """Update cached metrics (used by single-algorithm steps)."""
        X, F = self.pop_X, self.pop_F
        if X is None or F is None or len(X) == 0:
            return
        X = np.asarray(X, dtype=float)
        F = np.asarray(F, dtype=float)
        feas_flags, violations, F_feas = [], [], []
        G = getattr(self, 'pop_G', None)
        use_g = (G is not None and np.size(G) > 0)
        if use_g:
            G = np.asarray(G, dtype=float)
        for i in range(len(X)):
            f_i = F[i].astype(float)
            if use_g and i < len(G):
                g_i = np.asarray(G[i], dtype=float).ravel()
                v_i = float(np.sum(np.maximum(0.0, g_i)))
                feas = v_i <= 1e-12
                violations.append(v_i)
                feas_flags.append(bool(feas))
            else:
                _, feas = compute_penalty_and_feas(X[i], f_i, self.upv, self.enable_cr)
                feas_flags.append(bool(feas))
                violations.append(compute_violation(X[i], f_i, self.upv))
            if feas:
                F_feas.append(f_i)
        feas_ratio = float(np.mean(feas_flags)) if feas_flags else 0.0
        mean_violation = float(np.mean(violations)) if violations else 0.0
        self.viol_normer.update(mean_violation)

        F_feas = np.asarray(F_feas, dtype=float)
        hv_indicator = HV(ref_point=self.hv_ref)

        # HV on archive (best-so-far) makes curves more stable / interpretable.
        F_hv = F_feas
        try:
            exp_cfg_hv = exp_cfg if 'exp_cfg' in locals() else getattr(self, 'experiment_config', ExperimentConfig())

            if bool(getattr(exp_cfg_hv, 'hv_use_archive', False)):
                F_arch = getattr(self, 'feas_archive_F', None)
                if F_arch is not None and len(F_arch) > 0:
                    F_hv = np.asarray(F_arch, dtype=float)
        except Exception:
            pass

        hv = float(hv_indicator(F_hv)) if (hasattr(F_hv, 'size') and F_hv.size) else 0.0
        fmin = np.min(F_hv, axis=0).tolist() if (hasattr(F_hv, 'size') and F_hv.size) else (np.min(F, axis=0).tolist() if F.size else [])
        hv_prev = self.history_hv_feas[-1] if self.history_hv_feas else hv
        delta_hv_raw = float(hv - hv_prev)
        beta = float(self.experiment_config.hv_beta)
        self.delta_hv_smooth = beta * float(self.delta_hv_smooth) + (1.0 - beta) * delta_hv_raw
        self.dhv_normer.update(self.delta_hv_smooth)
        dhv_norm = float(self.dhv_normer.norm_pos(self.delta_hv_smooth))

        exp_cfg_sat = exp_cfg if 'exp_cfg' in locals() else getattr(self, 'experiment_config', ExperimentConfig())

        sat_use_archive = bool(getattr(exp_cfg_sat, 'sat_use_archive', True))

        if sat_use_archive:

            X_sat = getattr(self, 'feas_archive_X', None)

            F_sat = getattr(self, 'feas_archive_F', None)

            if X_sat is not None and F_sat is not None and len(F_sat) > 0:

                sat, pref_gap = compute_sat_prefgap(np.asarray(X_sat, dtype=float), np.asarray(F_sat, dtype=float), self.upv, self.enable_cr)

            else:

                sat, pref_gap = compute_sat_prefgap(X, F, self.upv, self.enable_cr)

        else:

            sat, pref_gap = compute_sat_prefgap(X, F, self.upv, self.enable_cr)
        diversity = float(np.mean(np.var(F, axis=0))) if F.size else 0.0
        conf = float(compute_confidence_score(self))
        alpha_cvar = float(self.upv.rho.get("cvar_alpha", 0.9))
        cvar = self._estimate_cvar(alpha_cvar) or 0.0
        risk = float(np.tanh(cvar / (1.0 + abs(cvar))))

        self.history_hv.append(hv)
        self.history_hv_feas.append(hv)
        self.history_fmin.append(fmin)
        self.history_dhv_raw.append(delta_hv_raw)
        self.history_dhv_norm.append(dhv_norm)
        self.history_feasible.append(feas_ratio)
        self.history_penalty.append(mean_violation)
        self.history_viol.append(mean_violation)
        self.history_diversity.append(diversity)
        self.history_sat.append(sat)
        self.history_pref_gap.append(pref_gap)
        self.history_conf.append(conf)
        self.history_risk.append(risk)
        self.history_dhv_smooth.append(self.delta_hv_smooth)
