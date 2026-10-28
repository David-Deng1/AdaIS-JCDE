"""
hitlo.policies.offline_dataset
================================
Offline transition dataset and rolling robust scaler.
Extracted from app_v11_iqlfix.py (RollingRobustScale, OfflineDataset classes).
"""

import json
import os
from collections import deque
from typing import List

import numpy as np

from hitlo.core.types import Transition


class RollingRobustScale:
    def __init__(self, maxlen: int = 200, q: float = 0.95):
        self.values = deque(maxlen=maxlen)
        self.q = float(q)
    def update(self, v: float):
        try:
            v = float(v)
        except Exception:
            return
        if not np.isfinite(v):
            return
        self.values.append(v)
    def scale(self) -> float:
        if not self.values:
            return 1.0
        return float(np.percentile(np.asarray(self.values, dtype=float), 100.0 * self.q))


class OfflineDataset:
    def __init__(self, base_dir: str = ".", prefix: str = "offline_D"):
        self.base_dir = base_dir
        self.prefix = prefix
        self.transitions: List[Transition] = []
        self.jsonl_path = os.path.join(self.base_dir, f"{self.prefix}.jsonl")
        os.makedirs(self.base_dir, exist_ok=True)
        if not os.path.exists(self.jsonl_path):
            with open(self.jsonl_path, "w", encoding="utf-8") as f:
                f.write("")

    def _behavior_code(self, behavior_id: str) -> int:
        """Map string behavior_id to a compact integer code.

        This is used for filtering offline transitions during IQL/AWR training.

        IMPORTANT: keep this mapping consistent with train_iql_v11.py.
        """
        sid = str(behavior_id or "").strip().upper()
        # Backward/forward compatible: allow prefixed fallback ids, e.g. "OOD_FALLBACK_A2".
        if sid.startswith("OOD_FALLBACK"):
            return 5
        mapping = {
            "A2": 1,
            "B1": 2,
            "IQL": 3,
            "AWR": 4,
            "OOD_FALLBACK": 5,
            "A1": 6,
            "A3": 7,
        }
        return int(mapping.get(sid, 0))

    def add(self, tr: Transition):
        self.transitions.append(tr)
        info = {
            "t": int(tr.t),
            "seed": int(tr.seed),
            "profile_id": int(tr.profile_id),
            "regime_id": int(tr.regime_id),
            "behavior_id": str(tr.behavior_id),
            "dhv_raw": float(tr.dhv_raw),
            "dhv_smooth": float(tr.dhv_smooth),
            "dhv_norm": float(tr.dhv_norm),
            "viol_raw": float(tr.viol_raw),
            "viol_norm": float(tr.viol_norm),
            "sat": float(tr.sat),
            "pref_gap": float(tr.pref_gap),
            "cost": int(tr.cost),
            "cost_norm": float(tr.cost_norm),
            "query_cost": float(tr.query_cost),
            "query_norm": float(tr.query_norm),
            "g_vec": [float(x) for x in tr.g_vec.tolist()],
            "alpha_decoded": [float(x) for x in tr.alpha_decoded.tolist()],
            "op_idx": int(tr.op_idx),
            "xi": [float(x) for x in tr.xi.tolist()],
            "hv_feas": float(tr.hv_feas),
            "feas_ratio": float(tr.feas_ratio),
            "mean_violation": float(tr.mean_violation),
            "diversity": float(tr.diversity),
            "noise_std": float(tr.noise_std),
        }
        if tr.perturb_mask is not None:
            info["perturb_mask"] = [int(x) for x in tr.perturb_mask.tolist()]
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(info, ensure_ascii=True) + "\n")

    def save_npz(self):
        if not self.transitions:
            return
        S = np.stack([t.s for t in self.transitions], axis=0).astype(np.float32)
        U = np.stack([t.u for t in self.transitions], axis=0).astype(np.float32)
        R = np.array([t.r for t in self.transitions], dtype=np.float32)
        S_next = np.stack([t.s_next for t in self.transitions], axis=0).astype(np.float32)
        T = np.array([t.t for t in self.transitions], dtype=np.int32)
        Seed = np.array([t.seed for t in self.transitions], dtype=np.int32)
        Profile = np.array([t.profile_id for t in self.transitions], dtype=np.int32)
        Regime = np.array([t.regime_id for t in self.transitions], dtype=np.int32)
        Behavior = np.array([self._behavior_code(t.behavior_id) for t in self.transitions], dtype=np.int32)
        G = np.stack([t.g_vec for t in self.transitions], axis=0).astype(np.float32)
        Perturb = np.stack([t.perturb_mask if t.perturb_mask is not None else np.zeros_like(t.g_vec) for t in self.transitions], axis=0).astype(np.int8)
        NoiseStd = np.array([t.noise_std for t in self.transitions], dtype=np.float32)
        Alpha = np.stack([t.alpha_decoded for t in self.transitions], axis=0).astype(np.float32)
        Op = np.array([t.op_idx for t in self.transitions], dtype=np.int16)
        Xi = np.stack([t.xi for t in self.transitions], axis=0).astype(np.float32)
        DhvRaw = np.array([t.dhv_raw for t in self.transitions], dtype=np.float32)
        DhvSmooth = np.array([t.dhv_smooth for t in self.transitions], dtype=np.float32)
        DhvNorm = np.array([t.dhv_norm for t in self.transitions], dtype=np.float32)
        ViolRaw = np.array([t.viol_raw for t in self.transitions], dtype=np.float32)
        ViolNorm = np.array([t.viol_norm for t in self.transitions], dtype=np.float32)
        Sat = np.array([t.sat for t in self.transitions], dtype=np.float32)
        PrefGap = np.array([t.pref_gap for t in self.transitions], dtype=np.float32)
        Cost = np.array([t.cost for t in self.transitions], dtype=np.int32)
        CostNorm = np.array([t.cost_norm for t in self.transitions], dtype=np.float32)
        QueryCost = np.array([t.query_cost for t in self.transitions], dtype=np.float32)
        QueryNorm = np.array([t.query_norm for t in self.transitions], dtype=np.float32)
        HvFeas = np.array([t.hv_feas for t in self.transitions], dtype=np.float32)
        Feas = np.array([t.feas_ratio for t in self.transitions], dtype=np.float32)
        Viol = np.array([t.mean_violation for t in self.transitions], dtype=np.float32)
        Div = np.array([t.diversity for t in self.transitions], dtype=np.float32)
        npz_path = os.path.join(self.base_dir, f"{self.prefix}.npz")
        np.savez(npz_path,
                 S=S, U=U, R=R, S_next=S_next,
                 t=T, seed=Seed, profile_id=Profile, regime_id=Regime, behavior_id=Behavior,
                 g_vec=G, perturb_mask=Perturb, noise_std=NoiseStd,
                 alpha_decoded=Alpha, op_idx=Op, xi=Xi,
                 dhv_raw=DhvRaw, dhv_smooth=DhvSmooth, dhv_norm=DhvNorm,
                 viol_raw=ViolRaw, viol_norm=ViolNorm, sat=Sat, pref_gap=PrefGap,
                 cost=Cost, cost_norm=CostNorm, query_cost=QueryCost, query_norm=QueryNorm,
                 hv_feas=HvFeas, feas_ratio=Feas, mean_violation=Viol, diversity=Div)
