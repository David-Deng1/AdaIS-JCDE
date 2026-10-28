# -*- coding: utf-8 -*-
"""Problem catalog and factory function ``make_problem``."""

from pymoo.core.problem import ElementwiseProblem

from .base import PymooProblemAdapter
from .benchmarks import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, DTLZ1, DTLZ2, DTLZ3, DTLZ4, KUR
from .sav_aero import (
    SAVAeroProblem, aero_bounds, aero_unflatten, build_aero_topology,
)
from .sav_common import (
    AERO_VAR_NAMES, AERO_SUBSYSTEMS, AERO_CONSTR_LABELS,
)


# —— 工厂 ——
PROBLEM_CATALOG = {
    "SAV-AERO": {"cls": SAVAeroProblem, "default_nvar": 22,
                  "label": "细长旋成体气动优化设计"},
    "ZDT1":   {"cls": ZDT1,  "default_nvar": 30, "label": "ZDT1 (convex front)"},
    "ZDT2":   {"cls": ZDT2,  "default_nvar": 30, "label": "ZDT2 (non-convex)"},
    "ZDT3":   {"cls": ZDT3,  "default_nvar": 30, "label": "ZDT3 (disconnected)"},
    "ZDT4":   {"cls": ZDT4,  "default_nvar": 10, "label": "ZDT4 (multimodal)"},
    "ZDT6":   {"cls": ZDT6,  "default_nvar": 30, "label": "ZDT6 (bias f1)"},
    "DTLZ1":  {"cls": DTLZ1, "default_nvar": 6,  "label": "DTLZ1 (M-obj, linear PF)", "scalable_obj": True, "default_nobj": 3},
    "DTLZ2":  {"cls": DTLZ2, "default_nvar": 11, "label": "DTLZ2 (M-obj, concave PF)", "scalable_obj": True, "default_nobj": 3},
    "DTLZ3":  {"cls": DTLZ3, "default_nvar": 11, "label": "DTLZ3 (M-obj, many local)", "scalable_obj": True, "default_nobj": 3},
    "DTLZ4":  {"cls": DTLZ4, "default_nvar": 11, "label": "DTLZ4 (M-obj, bias)", "scalable_obj": True, "default_nobj": 3},
    "KUR":    {"cls": KUR,   "default_nvar": 3,  "label": "Kursawe"},

    # --- Classic constrained MOPs (pymoo) ---
    # 2-objective, few constraints
    "BNH":    {"cls": (lambda n_var=2: PymooProblemAdapter("bnh", n_var_hint=n_var)),
               "default_nvar": 2, "label": "BNH (2-obj, 2 constr)"},
    "TNK":    {"cls": (lambda n_var=2: PymooProblemAdapter("tnk", n_var_hint=n_var)),
               "default_nvar": 2, "label": "TNK (2-obj, 2 constr)"},
    "SRN":    {"cls": (lambda n_var=2: PymooProblemAdapter("srn", n_var_hint=n_var)),
               "default_nvar": 2, "label": "SRN (2-obj, 2 constr)"},
    "OSY":    {"cls": (lambda n_var=6: PymooProblemAdapter("osy", n_var_hint=n_var)),
               "default_nvar": 6, "label": "OSY (2-obj, 6 constr)"},
    "WELDED-BEAM": {"cls": (lambda n_var=4: PymooProblemAdapter("welded_beam", n_var_hint=n_var)),
                    "default_nvar": 4, "label": "Welded Beam (2-obj, 4 constr)"},
    "TRUSS2D": {"cls": (lambda n_var=3: PymooProblemAdapter("truss2d", n_var_hint=n_var)),
               "default_nvar": 3, "label": "Truss2D (2-obj, 1 constr)"},

    # 3-objective engineering-style CMOP with many constraints
    "CARSIDE": {"cls": (lambda n_var=7: PymooProblemAdapter("carside", n_var_hint=n_var)),
               "default_nvar": 7, "label": "Carside (3-obj, 10 constr)"},

    # --- Many-constraint CMOP suites (pymoo) ---
    # DASCMOP: 2-obj with 11 constraints (difficulty 1..16). We fix difficulty=8 (0.5,0.5,0.5) as default.
    "DASCMOP1": {"cls": (lambda n_var=30: PymooProblemAdapter("dascmop1", n_var_hint=n_var, difficulty=8)),
                 "default_nvar": 30, "label": "DASCMOP1 (2-obj, 11 constr, diff=8)"},
    "DASCMOP3": {"cls": (lambda n_var=30: PymooProblemAdapter("dascmop3", n_var_hint=n_var, difficulty=8)),
                 "default_nvar": 30, "label": "DASCMOP3 (2-obj, 11 constr, diff=8)"},
    # 3-obj DASCMOP with 7 constraints
    "DASCMOP7": {"cls": (lambda n_var=30: PymooProblemAdapter("dascmop7", n_var_hint=n_var, difficulty_factors=8)),
                 "default_nvar": 30, "label": "DASCMOP7 (3-obj, 7 constr, diff=8)"},

    # MW suite (default n_var=15); include a representative 4-constraint instance
    "MW11":   {"cls": (lambda n_var=15: PymooProblemAdapter("mw11", n_var_hint=n_var)),
               "default_nvar": 15, "label": "MW11 (2-obj, 4 constr)"},
    "MW5":    {"cls": (lambda n_var=15: PymooProblemAdapter("mw5", n_var_hint=n_var)),
               "default_nvar": 15, "label": "MW5 (2-obj, 3 constr)"},

    # CTP suite (2-obj, 2 constr)
    "CTP1":   {"cls": (lambda n_var=2: PymooProblemAdapter("ctp1", n_var_hint=n_var)),
               "default_nvar": 2, "label": "CTP1 (2-obj, 2 constr)"},

}

# ---------------------------------------------------------------------------
# Optional display metadata (decorative enrichment for problem profile cards)
# ---------------------------------------------------------------------------
PROBLEM_META = {
    "ZDT1": {"family": "ZDT", "pf_shape": "凸", "tags": ["无约束", "可缩放"], "desc": "经典双目标，凸帕累托前沿"},
    "ZDT2": {"family": "ZDT", "pf_shape": "非凸", "tags": ["无约束", "可缩放"], "desc": "非凸帕累托前沿"},
    "ZDT3": {"family": "ZDT", "pf_shape": "不连续", "tags": ["无约束", "可缩放"], "desc": "不连续帕累托前沿"},
    "ZDT4": {"family": "ZDT", "pf_shape": "凸", "tags": ["无约束", "多模态"], "desc": "多模态，大量局部最优"},
    "ZDT6": {"family": "ZDT", "pf_shape": "非凸", "tags": ["无约束", "有偏"], "desc": "密度偏向 f1 端"},
    "DTLZ1": {"family": "DTLZ", "pf_shape": "线性", "tags": ["无约束", "多模态"], "desc": "线性前沿，多局部最优"},
    "DTLZ2": {"family": "DTLZ", "pf_shape": "凹", "tags": ["无约束", "可缩放"], "desc": "球面型凹前沿"},
    "DTLZ3": {"family": "DTLZ", "pf_shape": "凹", "tags": ["无约束", "多模态"], "desc": "凹前沿，大量局部最优"},
    "DTLZ4": {"family": "DTLZ", "pf_shape": "凹", "tags": ["无约束", "有偏"], "desc": "密度偏向特定区域"},
    "KUR": {"family": "经典", "pf_shape": "不连续", "tags": ["无约束"], "desc": "3变量，复杂非凸景观"},
    "BNH": {"family": "经典", "tags": ["有约束"], "desc": "2变量2约束基准"},
    "TNK": {"family": "经典", "tags": ["有约束"], "desc": "三角函数约束"},
    "SRN": {"family": "经典", "tags": ["有约束"], "desc": "标准参考问题"},
    "OSY": {"family": "经典", "tags": ["有约束", "多约束"], "desc": "6变量6约束"},
    "WELDED-BEAM": {"family": "工程", "tags": ["有约束", "工程问题"], "desc": "焊接梁结构优化"},
    "TRUSS2D": {"family": "工程", "tags": ["有约束", "工程问题"], "desc": "二维桁架设计"},
    "CARSIDE": {"family": "工程", "tags": ["有约束", "三目标", "工程问题"], "desc": "7变量10约束汽车侧碰撞"},
    "DASCMOP1": {"family": "DASCMOP", "tags": ["有约束", "多约束", "可缩放"], "desc": "难度8，11个约束"},
    "DASCMOP3": {"family": "DASCMOP", "tags": ["有约束", "多约束", "可缩放"], "desc": "难度8，不连续前沿"},
    "DASCMOP7": {"family": "DASCMOP", "tags": ["有约束", "三目标", "可缩放"], "desc": "3目标7约束"},
    "MW11": {"family": "MW", "tags": ["有约束", "可缩放"], "desc": "15变量4约束"},
    "MW5": {"family": "MW", "tags": ["有约束", "可缩放"], "desc": "15变量3约束"},
    "CTP1": {"family": "CTP", "tags": ["有约束"], "desc": "2变量2约束"},
    "SAV-AERO": {
        "family": "工程代理",
        "pf_shape": "非凸",
        "tags": ["有约束", "工程问题", "代理模型", "气动优化"],
        "desc": "22变量细长旋成体气动外形优化（2目标6约束，半经验气动模型）",
        "has_subsystems": True,
    },
}

# ---------------------------------------------------------------------------
# Generic registries: variable names, subsystems, topology renderers
# ---------------------------------------------------------------------------
VAR_NAMES_REGISTRY: dict[str, list[str]] = {
    "SAV-AERO": list(AERO_VAR_NAMES),
}

# Objective labels: short semantic names for each objective per problem.
# Format: "f{j+1} ({meaning})" — used by scatter, PCP, trend, design table.
OBJ_NAMES_REGISTRY: dict[str, list[str]] = {
    "SAV-AERO": ["f1 (零升阻力Cd0)", "f2 (负法向力-Cn)"],
    "BNH":      ["f1",              "f2"],
    "OSY":      ["f1",              "f2"],
    "TNK":      ["f1",              "f2"],
}


def get_obj_labels(pid: str | None, n_obj: int) -> list[str]:
    """Return objective display labels for *pid*.

    Falls back to ``['f1', 'f2', ...]`` when the problem has no semantic names.
    """
    pid = (pid or "").upper()
    reg = OBJ_NAMES_REGISTRY.get(pid)
    if reg and len(reg) >= n_obj:
        return list(reg[:n_obj])
    return [f"f{j+1}" for j in range(n_obj)]

SUBSYSTEM_REGISTRY: dict[str, list[dict]] = {
    "SAV-AERO": AERO_SUBSYSTEMS,
}

TOPOLOGY_RENDERERS: dict[str, callable] = {
    "SAV-AERO": build_aero_topology,
}


def get_problem_options():
    """从 PROBLEM_CATALOG 自动生成下拉选项列表。"""
    return [{"label": meta["label"], "value": pid} for pid, meta in PROBLEM_CATALOG.items()]


def get_problem_meta(pid: str) -> dict:
    """返回问题展示元数据。优先从 PROBLEM_META 取，不足部分从问题实例动态推断。"""
    pid = (pid or "ZDT1").upper()
    cat = PROBLEM_CATALOG.get(pid, PROBLEM_CATALOG["ZDT1"])
    base = {
        "family": "未分类",
        "default_nvar": cat.get("default_nvar", 30),
        "label": cat.get("label", pid),
        "tags": [],
        "desc": "",
        "pf_shape": "",
        "has_subsystems": False,
    }
    extra = PROBLEM_META.get(pid, {})
    base.update(extra)
    return base


def make_problem(problem_id: str, n_var: int, n_obj: int | None = None) -> ElementwiseProblem:
    pid = (problem_id or "ZDT1").upper()
    meta = PROBLEM_CATALOG.get(pid)
    if not meta:
        meta = PROBLEM_CATALOG["ZDT1"]
    cls = meta["cls"]
    # Build kwargs: always pass n_var; pass M (n_obj) when provided
    # and the problem class accepts it (DTLZ family).
    try:
        if n_obj is not None and meta.get("scalable_obj"):
            prob = cls(n_var=int(n_var), M=int(n_obj))
        else:
            prob = cls(n_var=int(n_var))
    except TypeError:
        prob = cls()

    # pymoo version compatibility: newer pymoo uses n_ieq_constr/n_eq_constr.
    # Many places in this codebase expect `problem.n_constr`.
    if not hasattr(prob, "n_constr"):
        try:
            n_ieq = int(getattr(prob, "n_ieq_constr", 0) or 0)
            n_eq = int(getattr(prob, "n_eq_constr", 0) or 0)
            setattr(prob, "n_constr", n_ieq + n_eq)
        except Exception:
            setattr(prob, "n_constr", 0)

    return prob
