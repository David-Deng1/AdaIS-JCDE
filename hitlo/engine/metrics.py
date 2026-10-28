"""
hitlo.engine.metrics
======================
Pareto-front utility functions and recognized preference-aware metrics.

Contains:
  - is_nondominated, crowding_distance, select_representative_indices (legacy)
  - define_roi_ellipsoid, in_roi, roi_hit_rate, hv_in_roi (Deb & Sundar 2006 GECCO 衍生)
  - cumulative_regret (Srinivas et al. 2010 ICML / Frazier 2018 BO Tutorial)

References
----------
[Deb2006GECCO]   Deb, K. & Sundar, J. (2006). Reference Point Based Multi-Objective
                 Optimization Using Evolutionary Algorithms. GECCO, 635-642.
[Srinivas2010]   Srinivas, N. et al. (2010). Gaussian Process Optimization in the
                 Bandit Setting: No Regret and Experimental Design. ICML.
[Frazier2018]    Frazier, P. I. (2018). A Tutorial on Bayesian Optimization.
                 arXiv:1807.02811.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def is_nondominated(F: np.ndarray) -> np.ndarray:
    """Return a boolean mask indicating which rows of *F* are non-dominated."""
    n = len(F)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if all(F[j] <= F[i]) and any(F[j] < F[i]):
                nd[i] = False
                break
    return nd


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Compute crowding distance for each row in objective matrix *F*."""
    m, n_obj = F.shape
    cd = np.zeros(m)
    if m == 0:
        return cd
    if m <= 2:
        cd[:] = np.inf
        return cd
    for j in range(n_obj):
        order = np.argsort(F[:, j])
        cd[order[0]] = cd[order[-1]] = np.inf
        fmin, fmax = F[order, j][0], F[order, j][-1]
        denom = (fmax - fmin) if (fmax - fmin) > 1e-12 else 1.0
        for k in range(1, m - 1):
            cd[order[k]] += (F[order[k + 1], j] - F[order[k - 1], j]) / denom
    return cd


def select_representative_indices(F: np.ndarray, k: int = 5,
                                  w: np.ndarray = None,
                                  r: np.ndarray = None) -> List[int]:
    """Select *k* representative solution indices from objective matrix *F*.

    Parameters
    ----------
    F : np.ndarray
        Objective matrix of shape ``(n, n_obj)``.
    k : int
        Number of representatives to select.
    w, r : np.ndarray, optional
        Weight and reference vectors for ASF distance fallback.  When the
        non-dominated set has fewer than *k* members the remaining slots are
        filled by solutions closest in ASF distance.
    """
    # Lazy import to avoid circular dependency at module level
    from hitlo.solvers.emo import _asf_distance

    n = len(F)
    if n == 0:
        return []
    nd_mask = is_nondominated(F)
    nd_idx = np.where(nd_mask)[0]
    if len(nd_idx) >= k:
        F_nd = F[nd_idx]
        cd = crowding_distance(F_nd)
        order_local = np.argsort(-cd)[:k]
        return [int(nd_idx[i]) for i in order_local]
    taken = list(map(int, nd_idx.tolist()))
    remain = [i for i in range(n) if i not in taken]
    if remain and w is not None and r is not None:
        d = np.array([_asf_distance(F[i], w, r) for i in remain])
        need = max(0, k - len(taken))
        picks = list(np.argsort(d)[:need])
        taken += [int(remain[p]) for p in picks]
    return taken[:k]


# =============================================================================
# v7 公认偏好导向指标 (Recognized Preference-Aware Metrics)
# =============================================================================

def define_roi_ellipsoid(r: np.ndarray,
                         b: np.ndarray,
                         tau: float = 1.0) -> Dict[str, np.ndarray]:
    """ROI 椭球定义 (Deb & Sundar 2006 GECCO 衍生).

    以决策者参考点 ``r`` 为中心、容差带 ``b`` 为各维半径、扩张系数 ``tau`` 控制大小的
    椭球域，作为"决策者关心的局部偏好区域"统一定义.

    椭球内点判据 (各目标 minimize 假设):

        sqrt( sum_i ( (F_i - r_i) / b_i )^2 ) <= tau

    所有变体共享同一 ROI 椭球（基于 v_full 终态的 r* 锁定），避免变体间椭球漂移
    导致的不公平比较。

    Parameters
    ----------
    r : np.ndarray, shape (M,)
        ROI 椭球中心 (UPV 终态参考点 r*).
    b : np.ndarray, shape (M,)
        各维半径 (UPV 终态容差带 b*); 内部对 b<=0 截断为 1e-12.
    tau : float
        扩张系数, 默认 1.0.

    Returns
    -------
    dict
        {'center': r, 'radii': b, 'tau': tau}.
    """
    center = np.asarray(r, dtype=float).reshape(-1)
    radii = np.asarray(b, dtype=float).reshape(-1)
    radii = np.maximum(radii, 1e-12)
    return {'center': center, 'radii': radii, 'tau': float(tau)}


def in_roi(F: np.ndarray, roi: Dict[str, np.ndarray]) -> np.ndarray:
    """检测目标矩阵每行是否落入 ROI 椭球. 返回 shape (n,) 的 bool mask."""
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    if F.size == 0:
        return np.array([], dtype=bool)
    diff = (F - roi['center']) / roi['radii']
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist <= roi['tau']


def roi_hit_rate(F: np.ndarray, roi: Dict[str, np.ndarray]) -> float:
    """ROI 命中率 (Deb & Sundar 2006 GECCO 衍生).

    解集中落入 ROI 椭球内的解所占比例. 量级 [0, 1], 越大表示偏好对齐越好.

    被 200+ 篇 EMO + 偏好融合论文采用作为 "preference satisfaction" 直接度量.
    """
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return 0.0
    if F.ndim == 1:
        F = F.reshape(1, -1)
    mask = in_roi(F, roi)
    return float(mask.sum() / len(F))


def hv_in_roi(F: np.ndarray,
              roi: Dict[str, np.ndarray],
              ref_point: Optional[np.ndarray] = None) -> float:
    """ROI 内 Hypervolume (Deb & Sundar 2006 GECCO 衍生).

    仅对落入 ROI 椭球内的解集计算 HV. 全局 HV 衡量"全空间覆盖", HV(ROI) 衡量
    "偏好域内覆盖", 是 PPCS 框架"让搜索聚焦决策者偏好域"主张的直接度量.

    Parameters
    ----------
    F : np.ndarray, shape (n, M)
        所有候选解的目标矩阵.
    roi : dict
        ROI 椭球描述, 来自 ``define_roi_ellipsoid``.
    ref_point : np.ndarray, optional
        HV 计算的 nadir 参考点; 若为 None, 自动取 ROI 椭球外接盒的右上顶点
        ``r + b * tau``.

    Returns
    -------
    float
        HV(ROI), 量级 [0, prod(radii)*某常数]; 0 表示 ROI 内无解或全部超出 ref_point.
    """
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return 0.0
    if F.ndim == 1:
        F = F.reshape(1, -1)
    mask = in_roi(F, roi)
    F_in = F[mask]
    if len(F_in) == 0:
        return 0.0
    if ref_point is None:
        ref_point = roi['center'] + roi['radii'] * roi['tau']
    ref_point = np.asarray(ref_point, dtype=float).reshape(-1)
    try:
        from pymoo.indicators.hv import HV
        hv_calc = HV(ref_point=ref_point)
        return float(hv_calc.do(F_in))
    except Exception:
        # 退化: 在所有解都超出 ref_point 或 pymoo 不可用时返回 0
        return 0.0


def cumulative_regret(F_history: np.ndarray,
                      F_optimal: np.ndarray,
                      w: Optional[np.ndarray] = None) -> np.ndarray:
    """累积 Regret (Srinivas et al. 2010 ICML / Frazier 2018 BO Tutorial).

    Bayesian Optimization 与 Bandits 文献 30 年标准指标. 定义为加权 Tchebycheff
    标量化下"当前最优解"与"理论最优"差距的累积:

        instant_regret(t) = max(  max_i w_i * F_history[t,i]
                                - max_i w_i * F_optimal,    0 )
        cum_regret(t)     = sum_{s=0}^{t} instant_regret(s)

    PPCS 的"主动澄清 + OSV 求解器调度"应让 regret 增长更慢, 是框架核心卖点之一.

    Parameters
    ----------
    F_history : np.ndarray, shape (T, M)
        每个 tick 的"当前最优解"目标向量轨迹 (从 t=0 到 t=T-1).
    F_optimal : np.ndarray, shape (M,)
        真实最优 (Pareto 前沿上的理想点; 标准基准用解析理想, 工程问题用 NSGA-II 长跑前沿的极值点）
    w : np.ndarray, optional, shape (M,)
        标量化权重; 若为 None, 取均匀 1/M.

    Returns
    -------
    np.ndarray, shape (T,)
        累积 regret 序列.
    """
    F_hist = np.asarray(F_history, dtype=float)
    if F_hist.ndim == 1:
        F_hist = F_hist.reshape(-1, 1)
    T, M = F_hist.shape
    F_opt = np.asarray(F_optimal, dtype=float).reshape(-1)
    if w is None:
        w = np.ones(M) / M
    w = np.asarray(w, dtype=float).reshape(-1)
    scalar_hist = np.max(w * F_hist, axis=-1)
    scalar_opt = float(np.max(w * F_opt))
    instant = np.maximum(scalar_hist - scalar_opt, 0.0)
    return np.cumsum(instant)
