from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from dash import html
from pymoo.core.problem import ElementwiseProblem

from .base import _pymoo_constr_init_kwargs
from .sav_common import (
    AERO_VAR_NAMES,
    AERO_VAR_BOUNDS,
    AERO_CONSTANTS,
    AERO_OUTPUT_NAMES,
    AERO_CONSTR_LABELS,
    AERO_SUBSYSTEMS,
    make_bounds_arrays,
    unflatten,
)

logger = logging.getLogger(__name__)

VAR_NAMES = AERO_VAR_NAMES
SUBSYSTEMS = AERO_SUBSYSTEMS
CONSTR_LABELS = AERO_CONSTR_LABELS

_DESIGN_MACH = 2.0
_DESIGN_AOA_DEG = 2.0
_DESIGN_AOA_RAD = np.radians(_DESIGN_AOA_DEG)
_GAMMA = 1.4
_RE_PER_M = 2.0e7
_N_FIN = 4
_T_INF = 223.0
_RHO_INF = 0.365
_P_INF = 22632.0

_SM_LO = 0.01
_SM_HI = 0.05

_K_BODY = 1.2


def aero_bounds(n_var: int = 22):
    lo, hi = make_bounds_arrays(AERO_VAR_NAMES, AERO_VAR_BOUNDS, n_var)
    return lo, hi


def aero_unflatten(x: np.ndarray) -> Dict[str, float]:
    return unflatten(x, AERO_VAR_NAMES)


def _friction_drag(L_total: float, D_ref: float, S_ref: float, Mach: float) -> float:
    Re_L = _RE_PER_M * np.clip(L_total, 0.1, 20.0)
    log_Re = np.log10(np.clip(Re_L, 1e3, 1e12))
    Cf_incomp = 0.455 / (log_Re ** 2.58)
    Cf = Cf_incomp / (1.0 + 0.144 * Mach ** 2) ** 0.65
    S_wet = np.pi * np.clip(D_ref, 0.01, 2.0) * np.clip(L_total, 0.1, 20.0)
    S_ref_safe = np.clip(S_ref, 1e-6, 10.0)
    return Cf * S_wet / S_ref_safe


def _pressure_drag_nose(L_nose: float, D_nose: float, Mach: float) -> float:
    fineness = np.clip(L_nose, 0.01, 5.0) / np.clip(D_nose, 0.01, 2.0)
    beta = np.sqrt(np.clip(Mach ** 2 - 1.0, 0.01, 100.0))
    half_angle = np.clip(D_nose / (2.0 * np.clip(L_nose, 0.01, 5.0)), 0.001, 1.0)
    Cd_nose = 0.5 * (half_angle ** 1.5) / beta
    Cd_nose *= 0.70
    fineness_factor = np.clip(1.0 / (1.0 + 0.1 * fineness), 0.3, 1.0)
    return Cd_nose * fineness_factor


def _base_drag(D_exit: float, D_centr: float, Mach: float) -> float:
    D_centr_safe = np.clip(D_centr, 0.01, 2.0)
    ratio = np.clip(D_exit / D_centr_safe, 0.0, 1.0)
    beta = np.sqrt(np.clip(Mach ** 2 - 1.0, 0.01, 100.0))
    K_base = 0.25
    return K_base * (1.0 - ratio) ** 2 / beta


def _fin_zero_lift_drag(
    n_fin: int,
    chord_root: float, chord_tip: float,
    sspan_root: float, sspan_tip: float,
    zupper_root: float, zupper_tip: float,
    sweep_rad: float, Mach: float, S_ref: float,
) -> float:
    span = np.clip(sspan_tip - sspan_root, 0.001, 5.0)
    c_mean = 0.5 * (np.clip(chord_root, 0.01, 3.0) + np.clip(chord_tip, 0.001, 3.0))
    S_fin_single = c_mean * span
    S_fin_total = n_fin * S_fin_single

    Re_c = _RE_PER_M * np.clip(c_mean, 0.01, 3.0)
    log_Re = np.log10(np.clip(Re_c, 1e3, 1e10))
    Cf_fin = 0.455 / (log_Re ** 2.58)
    Cf_fin /= (1.0 + 0.144 * Mach ** 2) ** 0.65

    S_ref_safe = np.clip(S_ref, 1e-6, 10.0)
    Cd_f = 2.0 * Cf_fin * S_fin_total / S_ref_safe

    t_c_avg = 0.5 * (np.clip(zupper_root, 0.001, 0.5) + np.clip(zupper_tip, 0.001, 0.5))
    beta = np.sqrt(np.clip(Mach ** 2 - 1.0, 0.01, 100.0))
    cos_sweep = np.clip(np.cos(sweep_rad), 0.01, 1.0)
    Cd_t = 4.0 * (t_c_avg ** 2) * cos_sweep / beta * S_fin_total / S_ref_safe

    return Cd_f + Cd_t


def _body_normal_force(
    D_centr: float, L_nose: float, L_centr: float,
    S_ref: float, alpha: float,
) -> tuple:
    S_base = np.pi / 4.0 * np.clip(D_centr, 0.01, 2.0) ** 2
    S_ref_safe = np.clip(S_ref, 1e-6, 10.0)
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)

    Cn_body = 2.0 * sin_a * cos_a + _K_BODY * (sin_a ** 2) * (S_base / S_ref_safe)

    L_total = np.clip(L_nose, 0.0, 5.0) + np.clip(L_centr, 0.0, 10.0)
    Xcp_body = 0.63 * L_total

    return Cn_body, Xcp_body


def _fin_normal_force(
    chord_root: float, chord_tip: float,
    sspan_root: float, sspan_tip: float,
    sweep_rad: float, Mach: float, S_ref: float,
    alpha: float, n_fin: int,
    XLE: float,
) -> tuple:
    span = np.clip(sspan_tip - sspan_root, 0.001, 5.0)
    c_root_safe = np.clip(chord_root, 0.01, 3.0)
    c_tip_safe = np.clip(chord_tip, 0.001, 3.0)
    c_mean = 0.5 * (c_root_safe + c_tip_safe)
    AR = 2.0 * span / np.clip(c_mean, 0.01, 5.0)

    beta_sq = np.clip(Mach ** 2 - 1.0, 0.01, 100.0)
    tan_sweep = np.tan(np.clip(sweep_rad, 0.01, np.pi / 2 - 0.02))

    denom_inner = AR ** 2 * beta_sq * (1.0 + tan_sweep ** 2 / np.clip(beta_sq, 0.1, 100.0)) + 4.0
    Cn_alpha_single = (2.0 * np.pi * AR) / (2.0 + np.sqrt(np.clip(denom_inner, 1.0, 1e6)))

    S_fin_single = c_mean * span
    S_ref_safe = np.clip(S_ref, 1e-6, 10.0)

    K_BW = 1.4

    Cn_fin = n_fin * Cn_alpha_single * alpha * K_BW * S_fin_single / S_ref_safe

    Xcp_fin = np.clip(XLE, 0.0, 10.0) + 0.25 * c_root_safe

    return Cn_fin, Xcp_fin


def _rudder_effectiveness(
    chord_root: float, chord_tip: float,
    sspan_root: float, sspan_tip: float,
    sweep_rad: float, Mach: float, S_ref: float,
    D_centr: float, delmax_rad: float,
) -> tuple:
    span = np.clip(sspan_tip - sspan_root, 0.001, 5.0)
    c_mean = 0.5 * (np.clip(chord_root, 0.01, 3.0) + np.clip(chord_tip, 0.001, 3.0))
    AR = 2.0 * span / np.clip(c_mean, 0.01, 5.0)

    beta_sq = np.clip(Mach ** 2 - 1.0, 0.01, 100.0)
    tan_sweep = np.tan(np.clip(sweep_rad, 0.01, np.pi / 2 - 0.02))
    denom_inner = AR ** 2 * beta_sq * (1.0 + tan_sweep ** 2 / np.clip(beta_sq, 0.1, 100.0)) + 4.0
    Cn_alpha_single = (2.0 * np.pi * AR) / (2.0 + np.sqrt(np.clip(denom_inner, 1.0, 1e6)))

    S_fin_single = c_mean * span
    S_ref_safe = np.clip(S_ref, 1e-6, 10.0)

    K_rud = 0.6
    n_rud = 2

    T_Cy1 = n_rud * Cn_alpha_single * delmax_rad * K_rud * S_fin_single / S_ref_safe

    D_ref = np.clip(D_centr, 0.01, 2.0)
    T_Mz1_unit = T_Cy1 / D_ref

    return T_Cy1, T_Mz1_unit


def _add_landscape_perturbation(x_vec: np.ndarray, seed_base: int = 42) -> float:
    n = len(x_vec)
    perturbation = 0.0
    for k in range(1, 4):
        phase = np.sum(x_vec * np.arange(1, n + 1) * (k * 0.37 + seed_base * 0.01))
        perturbation += 0.01 * np.sin(phase * k * 2.71828)
    return perturbation


class SAVAeroProblem(ElementwiseProblem):

    def __init__(self, n_var: int = 22):
        self.n_var_actual = min(n_var, len(AERO_VAR_NAMES))
        xl, xu = aero_bounds(self.n_var_actual)
        self.bounds = np.column_stack([xl, xu])
        super().__init__(
            n_var=self.n_var_actual,
            n_obj=2,
            xl=xl,
            xu=xu,
            **_pymoo_constr_init_kwargs(6, 0),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x, dtype=float).ravel()

        lo, hi = aero_bounds(self.n_var_actual)
        x = np.clip(x, lo, hi)

        d = aero_unflatten(x)

        LNOSE   = d["LNOSE"]
        DNOSE   = d["DNOSE"]
        LCENTR  = d["LCENTR"]
        DCENTR  = d["DCENTR"]
        DEXIT   = d["DEXIT"]
        XCG     = d["XCG"]
        XHINGE  = d["XHINGE"]
        XLE     = d["XLE"]
        SWEEP   = d["SWEEP"]
        SSPAN_r = d["SSPAN_root"]
        SSPAN_t = d["SSPAN_tip"]
        CHORD_r = d["CHORD_root"]
        CHORD_t = d["CHORD_tip"]
        LMAXU_r = d["LMAXU_root"]
        LMAXU_t = d["LMAXU_tip"]
        LFLATU_r = d["LFLATU_root"]
        LFLATU_t = d["LFLATU_tip"]
        ZUPPER_r = d["ZUPPER_root"]
        ZUPPER_t = d["ZUPPER_tip"]
        DELMAX  = d["DELMAX"]
        FYMAX   = d["FYMAX"]

        S_ref = float(AERO_CONSTANTS["SREF"])
        n_fin = int(AERO_CONSTANTS["NPANEL"])

        Mach = _DESIGN_MACH
        alpha = _DESIGN_AOA_RAD
        sweep_rad = np.radians(np.clip(SWEEP, 1.0, 89.0))
        delmax_rad = np.radians(np.clip(DELMAX, 0.1, 45.0))

        L_total = np.clip(LNOSE, 0.0, 5.0) + np.clip(LCENTR, 0.0, 10.0)
        L_total = np.clip(L_total, 0.1, 15.0)

        Cd_friction = _friction_drag(L_total, DCENTR, S_ref, Mach)
        Cd_pressure = _pressure_drag_nose(LNOSE, DNOSE, Mach)
        Cd_base = _base_drag(DEXIT, DCENTR, Mach)
        Cd_fin = _fin_zero_lift_drag(
            n_fin, CHORD_r, CHORD_t, SSPAN_r, SSPAN_t,
            ZUPPER_r, ZUPPER_t, sweep_rad, Mach, S_ref,
        )

        D_CD0 = Cd_friction + Cd_pressure + Cd_base + Cd_fin

        K_interference = 1.08
        D_CD0 *= K_interference

        D_CD0 *= (1.0 + _add_landscape_perturbation(x, seed_base=17))
        D_CD0 = np.clip(D_CD0, 0.01, 5.0)

        Cn_body, Xcp_body = _body_normal_force(DCENTR, LNOSE, LCENTR, S_ref, alpha)

        Cn_fin, Xcp_fin = _fin_normal_force(
            CHORD_r, CHORD_t, SSPAN_r, SSPAN_t,
            sweep_rad, Mach, S_ref, alpha, n_fin, XLE,
        )

        D_Cy1 = Cn_body + Cn_fin
        D_Cy1 *= (1.0 + _add_landscape_perturbation(x, seed_base=31))
        D_Cy1 = np.clip(D_Cy1, 0.001, 50.0)

        Cn_total = np.clip(Cn_body + Cn_fin, 1e-6, 100.0)
        D_Xcp1 = (Cn_body * Xcp_body + Cn_fin * Xcp_fin) / Cn_total

        T_Cy1, T_Mz1_unit = _rudder_effectiveness(
            CHORD_r, CHORD_t, SSPAN_r, SSPAN_t,
            sweep_rad, Mach, S_ref, DCENTR, delmax_rad,
        )

        D_ref = np.clip(DCENTR, 0.01, 2.0)
        T_Mz1 = T_Cy1 * (Xcp_fin - XCG) / np.clip(D_ref, 0.01, 2.0)

        stability_margin = (D_Xcp1 - XCG) / np.clip(D_ref, 0.01, 2.0)

        delta_zupper = abs(ZUPPER_r - ZUPPER_t)
        D_Mx1 = 0.01 * delta_zupper * D_Cy1

        D_CA1 = D_CD0 * np.cos(alpha) + D_Cy1 * np.sin(alpha)
        D_CA1z = D_CA1 * 1.05

        D_CD0z = D_CD0 * 1.05

        f1 = D_CD0
        f2 = -D_Cy1

        out["F"] = np.array([f1, f2], dtype=float)

        g1 = LMAXU_r + LFLATU_r - 1.0

        g2 = LMAXU_t + LFLATU_t - 1.0

        g3 = 0.5 * LNOSE - XLE

        fineness_ratio = LNOSE / np.clip(DNOSE, 0.001, 10.0)
        g4 = 1.5 - fineness_ratio

        g5 = (SSPAN_r + 0.5 * DCENTR) - SSPAN_t

        g6 = max(_SM_LO - stability_margin, stability_margin - _SM_HI)

        out["G"] = np.array([g1, g2, g3, g4, g5, g6], dtype=float)

        out["D_Cy1"] = float(D_Cy1)
        out["D_CD0"] = float(D_CD0)
        out["D_Xcp1"] = float(D_Xcp1)
        out["D_Mx1"] = float(D_Mx1)
        out["D_CA1"] = float(D_CA1)
        out["D_CA1z"] = float(D_CA1z)
        out["D_CD0z"] = float(D_CD0z)
        out["T_Cy1"] = float(T_Cy1)
        out["T_Mz1"] = float(T_Mz1)
        out["stability_margin"] = float(stability_margin)
        out["Cn_body"] = float(Cn_body)
        out["Cn_fin"] = float(Cn_fin)
        out["Xcp_body"] = float(Xcp_body)
        out["Xcp_fin"] = float(Xcp_fin)
        out["L_total"] = float(L_total)
        out["fineness_ratio"] = float(fineness_ratio)
        out["Cd_friction"] = float(Cd_friction)
        out["Cd_pressure"] = float(Cd_pressure)
        out["Cd_base"] = float(Cd_base)
        out["Cd_fin"] = float(Cd_fin)


def build_aero_topology() -> html.Div:
    from hitlo.config.color_scheme import (
        PRIMARY, ACCENT_CYAN, ACCENT_BLUE, ACCENT_AMBER,
        ACCENT_ROSE, BG_INSET, TEXT_PRIMARY, TEXT_SECONDARY,
    )

    _box_style_base = {
        "borderRadius": "8px",
        "padding": "8px 10px",
        "textAlign": "center",
        "fontSize": "0.72rem",
        "lineHeight": "1.4",
        "minWidth": "80px",
    }

    def _input_box(title: str, subtitle: str, color: str = ACCENT_CYAN) -> html.Div:
        return html.Div([
            html.Div(title, style={
                "fontWeight": "700", "color": "#fff", "fontSize": "0.75rem",
            }),
            html.Div(subtitle, style={
                "color": "rgba(255,255,255,0.7)", "fontSize": "0.65rem",
            }),
        ], style={
            **_box_style_base,
            "background": PRIMARY,
            "border": f"1px solid {color}",
            "flex": "1",
        })

    def _arrow_down() -> html.Div:
        return html.Div(
            "\u25BC",
            style={
                "textAlign": "center", "color": TEXT_SECONDARY,
                "fontSize": "0.9rem", "margin": "4px 0",
            },
        )

    def _process_box(title: str, subtitle: str, color: str = ACCENT_BLUE) -> html.Div:
        return html.Div([
            html.Div(title, style={
                "fontWeight": "700", "color": color, "fontSize": "0.73rem",
            }),
            html.Div(subtitle, style={
                "color": TEXT_SECONDARY, "fontSize": "0.62rem",
            }),
        ], style={
            **_box_style_base,
            "background": BG_INSET,
            "border": f"1px solid {color}",
        })

    def _output_box(title: str, items: list, color: str) -> html.Div:
        return html.Div([
            html.Div(title, style={
                "fontWeight": "700", "color": color,
                "fontSize": "0.73rem", "marginBottom": "3px",
            }),
            *[html.Div(it, style={
                "color": TEXT_SECONDARY, "fontSize": "0.62rem",
            }) for it in items],
        ], style={
            **_box_style_base,
            "background": "#fff",
            "border": f"1px solid {color}",
            "flex": "1",
        })

    row_inputs = html.Div([
        _input_box("弹身几何(7)", "LNOSE/DNOSE/LCENTR/DCENTR/..."),
        _input_box("翼面几何(13)", "XLE/SWEEP/CHORD/SSPAN/ZUPPER/..."),
        _input_box("操纵参数(2)", "DELMAX/FYMAX"),
    ], style={"display": "flex", "gap": "8px", "justifyContent": "center"})

    arrow1 = _arrow_down()

    row_condition = html.Div(
        [_process_box("工况: Ma=2.0, \u03b1=2\u00b0", "超声速巡航 / 设计攻角")],
        style={"display": "flex", "justifyContent": "center"},
    )

    arrow2 = _arrow_down()

    row_model = html.Div([
        _process_box("弹身气动", "细长体理论\nCn_body, Xcp_body", ACCENT_BLUE),
        _process_box("翼面气动", "线性化升力线斜率\nCn_fin, Cd_fin", ACCENT_BLUE),
        _process_box("阻力分解", "摩擦+波阻+底阻\nCd_friction/pressure/base", ACCENT_AMBER),
    ], style={"display": "flex", "gap": "6px", "justifyContent": "center"})

    arrow3 = _arrow_down()

    row_outputs_mid = html.Div(
        [_process_box(
            "气动特性汇聚",
            "Cn, Cd0, Cp, 舵效Cm, 稳定裕度sm",
            ACCENT_AMBER,
        )],
        style={"display": "flex", "justifyContent": "center"},
    )

    arrow4 = _arrow_down()

    row_final = html.Div([
        _output_box("目标函数", [
            "f1: Cd0\u2193 (零升阻力)",
            "f2: -Cy1\u2193 (法向力\u2191)",
        ], ACCENT_CYAN),
        _output_box("约束条件", [
            "g1: 翼型约束(根)",
            "g2: 翼型约束(梢)",
            "g3: 翼前缘位置",
            "g4: 头部细长比\u22651.5",
            "g5: 翼面构型(PHI)",
            "g6: 静稳定裕度\u2208[1%,5%]",
        ], ACCENT_ROSE),
    ], style={"display": "flex", "gap": "12px", "justifyContent": "center"})

    return html.Div([
        html.Div(
            "问题拓扑",
            style={
                "fontSize": "0.75rem", "fontWeight": "600",
                "color": TEXT_SECONDARY, "marginBottom": "8px",
            },
        ),
        html.Div(
            [row_inputs, arrow1, row_condition, arrow2, row_model,
             arrow3, row_outputs_mid, arrow4, row_final],
            style={"padding": "8px", "borderRadius": "6px", "background": BG_INSET},
        ),
    ])
