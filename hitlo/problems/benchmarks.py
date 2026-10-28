# -*- coding: utf-8 -*-
"""Classic multi-objective benchmark problems: ZDT1-6, DTLZ1-4, KUR."""

import numpy as np

from pymoo.core.problem import ElementwiseProblem

from .base import _pymoo_constr_init_kwargs


# ------------------ 问题定义：经典多目标测试集 ------------------

# —— ZDT 家族（2 目标）——
class ZDT1(ElementwiseProblem):
    def __init__(self, n_var=30):
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, xl=bounds[:, 0], xu=bounds[:, 1],
                         **_pymoo_constr_init_kwargs(0, 0))
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x)
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        out["F"] = np.array([f1, f2], dtype=float)

class ZDT2(ElementwiseProblem):
    def __init__(self, n_var=30):
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, xl=bounds[:, 0], xu=bounds[:, 1],
                         **_pymoo_constr_init_kwargs(0, 0))
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x)
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1.0 - (f1 / g) ** 2.0)
        out["F"] = np.array([f1, f2], dtype=float)

class ZDT3(ElementwiseProblem):
    def __init__(self, n_var=30):
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, xl=bounds[:, 0], xu=bounds[:, 1],
                         **_pymoo_constr_init_kwargs(0, 0))
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x)
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1))
        out["F"] = np.array([f1, f2], dtype=float)

class ZDT4(ElementwiseProblem):
    def __init__(self, n_var=10):
        bounds = np.zeros((n_var, 2), dtype=float)
        bounds[0] = [0.0, 1.0]
        bounds[1:] = [-5.0, 5.0]
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, xl=bounds[:, 0], xu=bounds[:, 1],
                         **_pymoo_constr_init_kwargs(0, 0))
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x)
        f1 = x[0]
        g = 1.0 + 10.0 * (self.n_var - 1) + np.sum(x[1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[1:]))
        f2 = g * (1.0 - np.sqrt(f1 / g))
        out["F"] = np.array([f1, f2], dtype=float)

class ZDT6(ElementwiseProblem):
    def __init__(self, n_var=30):
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, xl=bounds[:, 0], xu=bounds[:, 1],
                         **_pymoo_constr_init_kwargs(0, 0))
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x)
        f1 = 1.0 - np.exp(-4.0 * x[0]) * (np.sin(6.0 * np.pi * x[0])) ** 6
        g = 1.0 + 9.0 * (np.mean(x[1:]) ** 0.25)
        f2 = g * (1.0 - (f1 / g) ** 2.0)
        out["F"] = np.array([f1, f2], dtype=float)

# —— DTLZ（可缩放目标维度 M）——
def _dtlz_split(n_var, M):
    return max(1, n_var - M + 1)


class DTLZ1(ElementwiseProblem):
    def __init__(self, n_var=6, M=2):
        M = int(M)
        self.M = M
        n_var = max(n_var, M)
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=M, xl=bounds[:, 0], xu=bounds[:, 1],
                         **_pymoo_constr_init_kwargs(0, 0))

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.M
        k = _dtlz_split(self.n_var, M)
        xm = x[self.n_var - k:]
        g = 100.0 * (k + np.sum((xm - 0.5) ** 2 - np.cos(20.0 * np.pi * (xm - 0.5))))
        f = np.empty(M, dtype=float)
        for i in range(M):
            f[i] = 0.5 * (1.0 + g)
            for j in range(M - 1 - i):
                f[i] *= x[j]
            if i > 0:
                f[i] *= (1.0 - x[M - 1 - i])
        out["F"] = f


class DTLZ2(ElementwiseProblem):
    def __init__(self, n_var=11, M=2):
        M = int(M)
        self.M = M
        n_var = max(n_var, M)
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=M, xl=bounds[:, 0], xu=bounds[:, 1],
                         **_pymoo_constr_init_kwargs(0, 0))

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.M
        k = _dtlz_split(self.n_var, M)
        g = np.sum((x[self.n_var - k:] - 0.5) ** 2)
        theta = 0.5 * np.pi * x[:M - 1]
        f = np.empty(M, dtype=float)
        for i in range(M):
            f[i] = 1.0 + g
            for j in range(M - 1 - i):
                f[i] *= np.cos(theta[j])
            if i > 0:
                f[i] *= np.sin(theta[M - 1 - i])
        out["F"] = f


class DTLZ3(ElementwiseProblem):
    def __init__(self, n_var=11, M=2):
        M = int(M)
        self.M = M
        n_var = max(n_var, M)
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=M, xl=bounds[:, 0], xu=bounds[:, 1],
                         **_pymoo_constr_init_kwargs(0, 0))

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.M
        k = _dtlz_split(self.n_var, M)
        xm = x[self.n_var - k:]
        g = 100.0 * (k + np.sum((xm - 0.5) ** 2 - np.cos(20.0 * np.pi * (xm - 0.5))))
        theta = 0.5 * np.pi * x[:M - 1]
        f = np.empty(M, dtype=float)
        for i in range(M):
            f[i] = 1.0 + g
            for j in range(M - 1 - i):
                f[i] *= np.cos(theta[j])
            if i > 0:
                f[i] *= np.sin(theta[M - 1 - i])
        out["F"] = f


class DTLZ4(ElementwiseProblem):
    def __init__(self, n_var=11, M=2, alpha=100.0):
        M = int(M)
        self.M = M
        self.alpha = alpha
        n_var = max(n_var, M)
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=M, xl=bounds[:, 0], xu=bounds[:, 1],
                         **_pymoo_constr_init_kwargs(0, 0))

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.M
        k = _dtlz_split(self.n_var, M)
        g = np.sum((x[self.n_var - k:] - 0.5) ** 2)
        theta = 0.5 * np.pi * (x[:M - 1] ** self.alpha)
        f = np.empty(M, dtype=float)
        for i in range(M):
            f[i] = 1.0 + g
            for j in range(M - 1 - i):
                f[i] *= np.cos(theta[j])
            if i > 0:
                f[i] *= np.sin(theta[M - 1 - i])
        out["F"] = f

# —— Kursawe（2 目标，3 变量）——
class KUR(ElementwiseProblem):
    def __init__(self, n_var=3):
        assert n_var >= 3, "Kursawe 通常使用 3 变量"
        bounds = np.array([[-5.0, 5.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, xl=bounds[:, 0], xu=bounds[:, 1],
                         **_pymoo_constr_init_kwargs(0, 0))
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x, dtype=float)
        s1 = 0.0
        for i in range(self.n_var - 1):
            s1 += -10.0 * np.exp(-0.2 * np.sqrt(x[i] ** 2 + x[i + 1] ** 2))
        s2 = np.sum(np.abs(x) ** 0.8 + 5.0 * np.sin(x ** 3))
        out["F"] = np.array([s1, s2], dtype=float)
