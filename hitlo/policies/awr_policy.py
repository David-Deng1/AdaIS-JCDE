"""
hitlo.policies.awr_policy
===========================
Advantage-Weighted Regression (AWR/AWAC-style) lightweight policy.
Extracted from app_v11_iqlfix.py (AWRPolicy class).
"""

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge, SGDClassifier


class AWRPolicy:
    """一个轻量但"有研究难度"的 Π 实现：Advantage-Weighted Regression (AWR/AWAC 风格).

    - 目的：在昂贵优化/交互场景下，优先利用离线日志数据学习 OSV 控制策略；
    - 形式：用带权回归学习 g_mean(s)，权重与回报 r 相关（高回报样本更影响策略）；
    - 优点：实现简单、可复现、可做消融；比纯规则更"硬"，又比完整离线 SAC 更易落地。
    """
    def __init__(self, action_dim: int, ctrl_dim: int, seed: int = 0):
        self.action_dim = int(action_dim)
        self.ctrl_dim = int(ctrl_dim)
        self.rng = np.random.default_rng(seed)
        self._state_dim = None  # 延迟确定
        self._actor = None      # 每维一个 Ridge（支持 sample_weight）
        self._gate = None       # 每个控制通道一个 Logistic SGDClassifier（可 partial_fit）
        self._gate_inited = False

        self.buffer = deque(maxlen=8000)  # (s, a_latent, mask, r)

        # 训练超参
        self.temp = 0.5
        self.ridge_alpha = 1e-2
        self.train_every = 10
        self.min_batch = 200

    @property
    def state_dim(self) -> int:
        return int(self._state_dim or 0)

    def _ensure_models(self, state_dim: int):
        if self._actor is None:
            self._state_dim = int(state_dim)
            self._actor = [Ridge(alpha=self.ridge_alpha) for _ in range(self.action_dim)]
            self._gate = [SGDClassifier(loss="log_loss", alpha=1e-4, random_state=0) for _ in range(self.ctrl_dim)]
            self._gate_inited = False

    def push(self, s: np.ndarray, a_latent: np.ndarray, mask: np.ndarray, r: float):
        s = np.asarray(s, dtype=float).reshape(1,-1)
        a = np.asarray(a_latent, dtype=float).reshape(1,-1)
        m = np.asarray(mask, dtype=int).reshape(1,-1)
        self.buffer.append((s, a, m, float(r)))

    def predict_dist(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """输出 (g_mean, g_std, gate_p)。g_mean 为 unbounded latent。"""
        s = np.asarray(s, dtype=float).reshape(1,-1)
        self._ensure_models(s.shape[1])

        if len(self.buffer) < self.min_batch or any(not hasattr(m, "coef_") for m in self._actor):
            # 未训练：输出零均值、固定方差、门控 0.6
            g_mean = np.zeros(self.action_dim, dtype=float)
            g_std = np.ones(self.action_dim, dtype=float) * 0.25
            gate_p = np.ones(self.ctrl_dim, dtype=float) * 0.6
            return g_mean, g_std, gate_p

        g_mean = np.array([mdl.predict(s)[0] for mdl in self._actor], dtype=float)
        g_std = np.ones(self.action_dim, dtype=float) * 0.18  # 也可学习，这里先固定

        # 门控：若 gate 模型尚未初始化，则回退
        if not self._gate_inited:
            gate_p = np.ones(self.ctrl_dim, dtype=float) * 0.6
        else:
            gate_p = np.array([self._gate[k].predict_proba(s)[0,1] for k in range(self.ctrl_dim)], dtype=float)
        return g_mean, g_std, gate_p

    def maybe_train(self, step: int, force: bool = False, **kwargs) -> Optional[Dict[str, float]]:
        if len(self.buffer) < self.min_batch:
            return
        if step % self.train_every != 0:
            return

        # 组 batch
        S = np.vstack([b[0] for b in self.buffer])   # (N, d)
        A = np.vstack([b[1] for b in self.buffer])   # (N, action_dim)
        M = np.vstack([b[2] for b in self.buffer])   # (N, ctrl_dim)
        R = np.array([b[3] for b in self.buffer], dtype=float)

        self._ensure_models(S.shape[1])

        # 计算 advantage-like weights（只用 reward 做 proxy）
        r0 = R - np.mean(R)
        w = np.exp(np.clip(r0 / max(1e-6, self.temp), -10.0, 10.0))
        w = np.clip(w, 1e-3, 50.0)

        # 训练 actor：每个维度一个加权 Ridge
        for j in range(self.action_dim):
            self._actor[j].fit(S, A[:,j], sample_weight=w)

        # 训练 gate：用"高回报样本更倾向于哪些通道启用"来学 p(m_k=1|s)
        # label：通道是否启用（mask）
        for k in range(self.ctrl_dim):
            y = M[:,k].astype(int)
            # SGDClassifier 需要 classes 初始化
            if not self._gate_inited:
                self._gate[k].partial_fit(S, y, classes=np.array([0,1], dtype=int), sample_weight=w)
            else:
                self._gate[k].partial_fit(S, y, sample_weight=w)
        self._gate_inited = True
