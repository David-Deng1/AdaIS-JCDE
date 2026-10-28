from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SAV_MODELS_DIR = Path(__file__).parent / "sav_models"
SAV_TRAINED_DIR = SAV_MODELS_DIR / "trained"
SAV_COST_DIR = SAV_MODELS_DIR / "cost_modules"


AERO_VAR_NAMES: List[str] = [
    "LNOSE",
    "DNOSE",
    "BNOSE",
    "LCENTR",
    "DCENTR",
    "DEXIT",
    "XCG",
    "XHINGE",
    "XLE",
    "SWEEP",
    "SSPAN_root",
    "SSPAN_tip",
    "CHORD_root",
    "CHORD_tip",
    "LMAXU_root",
    "LMAXU_tip",
    "LFLATU_root",
    "LFLATU_tip",
    "ZUPPER_root",
    "ZUPPER_tip",
    "DELMAX",
    "FYMAX",
]

AERO_VAR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "LNOSE":       (0.0,   1.5),
    "DNOSE":       (0.02,  0.6),
    "BNOSE":       (0.0,   0.2),
    "LCENTR":      (0.0,   5.0),
    "DCENTR":      (0.02,  0.6),
    "DEXIT":       (0.0,   0.6),
    "XCG":         (0.0,   6.0),
    "XHINGE":      (0.409, 6.0),
    "XLE":         (0.4,   6.0),
    "SWEEP":       (20.0,  89.0),
    "SSPAN_root":  (0.02,  1.0),
    "SSPAN_tip":   (0.02,  1.0),
    "CHORD_root":  (0.05,  1.5),
    "CHORD_tip":   (0.0,   1.5),
    "LMAXU_root":  (0.05,  0.5),
    "LMAXU_tip":   (0.05,  0.5),
    "LFLATU_root": (0.05,  0.8),
    "LFLATU_tip":  (0.05,  0.8),
    "ZUPPER_root": (0.005, 0.1),
    "ZUPPER_tip":  (0.005, 0.1),
    "DELMAX":      (0.0,   35.0),
    "FYMAX":       (1.0,   30.0),
}

AERO_CONSTANTS: Dict[str, object] = {
    "TNOSE":  "KARMAN",
    "SREF":   0.0735415,
    "NPANEL": 4,
    "SECTYP": 0.0,
    "X0":     0.0,
    "STA":    0.0,
    "ROUGH":  0.0,
}

AERO_OUTPUT_NAMES: List[str] = [
    "D_Cy1",
    "D_Xcp1",
    "D_Mx1",
    "D_CA1",
    "D_CA1z",
    "D_CD0",
    "D_CD0z",
    "T_Cy1",
    "T_CA1",
    "T_Mx1",
    "T_Mz1",
    "Mz_wz",
    "Mz_wx",
    "Mx_wx",
    "Mx_wz",
    "Mz_ad",
]

AERO_CONSTR_LABELS: List[str] = [
    "g1: LMAXU_root+LFLATU_root<1（根部翼型）",
    "g2: LMAXU_tip+LFLATU_tip<1（梢部翼型）",
    "g3: XLE≥0.5·LNOSE（翼根前缘位置）",
    "g4: LNOSE/DNOSE>1.5（头部细长比）",
    "g5: PHI构型约束（翼面布局）",
    "g6: 静稳定裕度∈[1%,5%]",
]

AERO_SUBSYSTEMS: List[Dict] = [
    {"name": "弹身几何",  "key": "body_geom",    "n_var": 7,  "icon": "fa-ruler-combined"},
    {"name": "翼面几何",  "key": "fin_geom",     "n_var": 13, "icon": "fa-drafting-compass"},
    {"name": "操纵参数",  "key": "control_param", "n_var": 2,  "icon": "fa-sliders-h"},
]


def make_bounds_arrays(
    var_names: List[str],
    var_bounds: Dict[str, Tuple[float, float]],
    n_var: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n = n_var or len(var_names)
    lo = np.zeros(n, dtype=float)
    hi = np.ones(n, dtype=float)
    for i, name in enumerate(var_names[:n]):
        bounds = var_bounds.get(name, (0.0, 1.0))
        lo[i] = bounds[0]
        hi[i] = bounds[1]
    return lo, hi


def unflatten(x: np.ndarray, var_names: List[str]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).ravel()
    return {name: float(x[i]) if i < len(x) else 0.0
            for i, name in enumerate(var_names)}
