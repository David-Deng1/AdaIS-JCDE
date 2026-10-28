"""
hitlo.policies.preference_gp
==============================
UPV preference prediction model using independent-dimension Gaussian Processes.
Extracted from app_v11_iqlfix.py (PreferenceGP class).
"""

from typing import List, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


class PreferenceGP:
    """UPV 偏好预测模型：混合（独立维度）高斯过程，根据 pf -> (w,r) 学习。"""
    def __init__(self, pf_dim: int, w_dim: int, r_dim: int, seed: int = 0, max_samples: int = 300):
        # 使用 RBF 核 + alpha 参数控制噪声，避免 WhiteKernel 的收敛警告
        kernel = RBF(length_scale=2.0)
        self.gp_w = [GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=True, random_state=seed+j) for j in range(w_dim)]
        self.gp_r = [GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=True, random_state=seed+100+j) for j in range(r_dim)]
        self.X: List[np.ndarray] = []
        self.Yw: List[np.ndarray] = []
        self.Yr: List[np.ndarray] = []
        # v8 S1.8: hold-out 集合 (Rasmussen 2006 NLPD 评估协议)
        self.X_holdout: List[np.ndarray] = []
        self.Yw_holdout: List[np.ndarray] = []
        self.Yr_holdout: List[np.ndarray] = []
        self.pf_dim = int(pf_dim)
        self.w_dim = int(w_dim)
        self.r_dim = int(r_dim)
        self.trained = False
        self.max_samples = max_samples

    def add_sample(self, pf_vec: np.ndarray, w: np.ndarray, r: np.ndarray,
                   holdout_ratio: float = 0.0):
        """v8 S1.8: holdout_ratio > 0 时按确定性 1/k 划分加入 hold-out (k=1/ratio)."""
        pf_vec = np.asarray(pf_vec, dtype=float).reshape(1, -1)
        w = np.asarray(w, dtype=float).reshape(1, -1)
        r = np.asarray(r, dtype=float).reshape(1, -1)
        if pf_vec.shape[1] != self.pf_dim:
            return
        # 确定性 hold-out 划分: 每 k 次 add 中第 k 次加入 hold-out (1/k 比例)
        if holdout_ratio > 0:
            k = max(2, int(round(1.0 / holdout_ratio)))
            n_total = len(self.X) + len(self.X_holdout)
            if (n_total + 1) % k == 0:
                self.X_holdout.append(pf_vec)
                self.Yw_holdout.append(w)
                self.Yr_holdout.append(r)
                # hold-out 也限规模避免无限增长
                if len(self.X_holdout) > self.max_samples // 4:
                    self.X_holdout = self.X_holdout[-(self.max_samples // 4):]
                    self.Yw_holdout = self.Yw_holdout[-(self.max_samples // 4):]
                    self.Yr_holdout = self.Yr_holdout[-(self.max_samples // 4):]
                return
        self.X.append(pf_vec)
        self.Yw.append(w)
        self.Yr.append(r)
        if len(self.X) > self.max_samples:
            self.X = self.X[-self.max_samples:]
            self.Yw = self.Yw[-self.max_samples:]
            self.Yr = self.Yr[-self.max_samples:]

    def evaluate_nlpd_holdout(self) -> dict:
        """v8 S1.8: 用内部 hold-out 集计算 NLPD."""
        if len(self.X_holdout) == 0:
            return {'nlpd_total': float('nan'), 'nlpd_w': float('nan'),
                    'nlpd_r': float('nan'), 'mse_w': float('nan'),
                    'mse_r': float('nan'), 'n_holdout': 0,
                    'n_train': len(self.X)}
        X_h = np.vstack(self.X_holdout)
        Yw_h = np.vstack(self.Yw_holdout)
        Yr_h = np.vstack(self.Yr_holdout)
        return self.evaluate_nlpd(X_h, Yw_h, Yr_h)

    def fit(self):
        if len(self.X) < 5:
            return
        X = np.vstack(self.X)
        Yw = np.vstack(self.Yw)
        Yr = np.vstack(self.Yr)
        for j,gp in enumerate(self.gp_w):
            try:
                gp.fit(X, Yw[:, j])
            except Exception:
                pass
        for j,gp in enumerate(self.gp_r):
            try:
                gp.fit(X, Yr[:, j])
            except Exception:
                pass
        self.trained = True

    def predict(self, pf_vec: np.ndarray, return_std: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
        if not self.trained or len(self.X) < 5:
            return None
        pf_vec = np.asarray(pf_vec, dtype=float).reshape(1, -1)
        if pf_vec.shape[1] != self.pf_dim:
            return None
        try:
            if return_std:
                w_out = [gp.predict(pf_vec, return_std=True) for gp in self.gp_w]
                r_out = [gp.predict(pf_vec, return_std=True) for gp in self.gp_r]
                w_hat = np.array([m[0] for (m, s) in w_out], dtype=float)
                r_hat = np.array([m[0] for (m, s) in r_out], dtype=float)
                w_std = np.array([s[0] for (m, s) in w_out], dtype=float)
                r_std = np.array([s[0] for (m, s) in r_out], dtype=float)
                return w_hat, r_hat, w_std, r_std
            w_hat = np.array([gp.predict(pf_vec)[0] for gp in self.gp_w], dtype=float)
            r_hat = np.array([gp.predict(pf_vec)[0] for gp in self.gp_r], dtype=float)
            return w_hat, r_hat, None, None
        except Exception:
            return None

    # =========================================================================
    # v7 评估接口 (Negative Log Predictive Density / Mean Squared Error)
    # =========================================================================

    def evaluate_nlpd(self,
                      X_holdout: np.ndarray,
                      Yw_holdout: np.ndarray,
                      Yr_holdout: np.ndarray) -> dict:
        """对 hold-out 偏好观测计算 NLPD 与 MSE.

        NLPD (Negative Log Predictive Density), Rasmussen & Williams 2006 §5.4
        式 (5.10), 是 GP 文献评估概率预测质量的金标准, 同时考虑均值与方差:

            NLPD = -(1/N) * sum_i log p(y_i | x_i, D)
                 = (1/N) * sum_i [ 0.5*log(2*pi*sigma_i^2) + 0.5*(y_i - mu_i)^2 / sigma_i^2 ]

        MSE (Mean Squared Error) 是均值预测的辅助度量.

        Parameters
        ----------
        X_holdout : np.ndarray, shape (N, pf_dim)
            hold-out 过程特征观测.
        Yw_holdout : np.ndarray, shape (N, w_dim)
            hold-out 真实权重观测.
        Yr_holdout : np.ndarray, shape (N, r_dim)
            hold-out 真实参考点观测.

        Returns
        -------
        dict
            {
              'nlpd_w': float, 'nlpd_r': float, 'nlpd_total': float,
              'mse_w':  float, 'mse_r':  float,
              'n_holdout': int, 'n_train': int,
            }
            训练样本不足时返回 nan; 上层调用应检查 isnan.
        """
        if not self.trained or len(self.X) < 5:
            return {
                'nlpd_w': float('nan'), 'nlpd_r': float('nan'),
                'nlpd_total': float('nan'),
                'mse_w':  float('nan'), 'mse_r':  float('nan'),
                'n_holdout': int(len(X_holdout)) if X_holdout is not None else 0,
                'n_train': int(len(self.X)),
            }

        X_h = np.asarray(X_holdout, dtype=float)
        if X_h.ndim == 1:
            X_h = X_h.reshape(1, -1)
        if X_h.size == 0 or X_h.shape[1] != self.pf_dim:
            return {
                'nlpd_w': float('nan'), 'nlpd_r': float('nan'),
                'nlpd_total': float('nan'),
                'mse_w':  float('nan'), 'mse_r':  float('nan'),
                'n_holdout': int(len(X_h)),
                'n_train': int(len(self.X)),
            }

        Yw_h = np.asarray(Yw_holdout, dtype=float).reshape(-1, self.w_dim)
        Yr_h = np.asarray(Yr_holdout, dtype=float).reshape(-1, self.r_dim)
        N = X_h.shape[0]

        # Predict per-dim mean & std
        Mu_w = np.zeros((N, self.w_dim), dtype=float)
        Sd_w = np.zeros((N, self.w_dim), dtype=float)
        Mu_r = np.zeros((N, self.r_dim), dtype=float)
        Sd_r = np.zeros((N, self.r_dim), dtype=float)
        # v8 S1.8 修复: sigma 截断从 1e-6 提至 1e-2 避免 (y-mu)^2/sigma^2 数值爆炸
        # (1e-6 截断让方差极小区域的 NLPD 达 1e10 量级, 失去与 baseline 的可比性)
        SIGMA_FLOOR = 1e-2
        try:
            for j, gp in enumerate(self.gp_w):
                m, s = gp.predict(X_h, return_std=True)
                Mu_w[:, j] = m.reshape(-1)
                Sd_w[:, j] = np.maximum(s.reshape(-1), SIGMA_FLOOR)
            for j, gp in enumerate(self.gp_r):
                m, s = gp.predict(X_h, return_std=True)
                Mu_r[:, j] = m.reshape(-1)
                Sd_r[:, j] = np.maximum(s.reshape(-1), SIGMA_FLOOR)
        except Exception:
            return {
                'nlpd_w': float('nan'), 'nlpd_r': float('nan'),
                'nlpd_total': float('nan'),
                'mse_w':  float('nan'), 'mse_r':  float('nan'),
                'n_holdout': int(N),
                'n_train': int(len(self.X)),
            }

        # NLPD per dim, then mean over dims & samples
        log2pi = float(np.log(2.0 * np.pi))
        nlpd_w_per = 0.5 * (log2pi + np.log(Sd_w ** 2) + (Yw_h - Mu_w) ** 2 / (Sd_w ** 2))
        nlpd_r_per = 0.5 * (log2pi + np.log(Sd_r ** 2) + (Yr_h - Mu_r) ** 2 / (Sd_r ** 2))
        nlpd_w = float(np.mean(nlpd_w_per))
        nlpd_r = float(np.mean(nlpd_r_per))
        # 加权 (w 与 r 同等权重) 总 NLPD
        nlpd_total = 0.5 * (nlpd_w + nlpd_r)

        mse_w = float(np.mean((Yw_h - Mu_w) ** 2))
        mse_r = float(np.mean((Yr_h - Mu_r) ** 2))

        return {
            'nlpd_w': nlpd_w,
            'nlpd_r': nlpd_r,
            'nlpd_total': nlpd_total,
            'mse_w':  mse_w,
            'mse_r':  mse_r,
            'n_holdout': int(N),
            'n_train': int(len(self.X)),
        }
