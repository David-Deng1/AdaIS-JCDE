"""Ground-truth evaluation of traced runs (JCDE-R1): utility regret, preference recovery and paired effects.

Every run is scored against the simulated decision maker's ground truth (w*, r*) with the same
achievement scalarizing function the simulator uses: ASF(f) = max_i w_i|f_i - r_i| + rho * sum_i w_i|f_i - r_i|.

* u_best(t): ASF of the best feasible design in the population at tick t (what an oracle would pick).
* u_rec(t):  ASF of the design the system would recommend at tick t, i.e. the feasible design that
             minimises ASF under the system's own preference estimate (w_t, r_t) entering tick t.
* Regret is measured against U*, the best ASF found by any run on the same problem and ground truth,
  and normalised by the median initial gap (u_best at tick 0 minus U*), so 1 = no progress, 0 = best known.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

RHO = 1e-6


def asf(F: np.ndarray, w: np.ndarray, r: np.ndarray) -> np.ndarray:
    d = np.asarray(w, dtype=float) * np.abs(
        np.asarray(F, dtype=float) - np.asarray(r, dtype=float)
    )
    return d.max(axis=-1) + RHO * d.sum(axis=-1)


def asf_mono(F: np.ndarray, w: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Monotone aspiration ASF (preference model v2 SDM utility; inputs already scaled)."""
    d = np.asarray(w, dtype=float) * (
        np.asarray(F, dtype=float) - np.asarray(r, dtype=float)
    )
    return d.max(axis=-1) + RHO * d.sum(axis=-1)


def feasible_mask(G: Optional[np.ndarray], n: int) -> np.ndarray:
    if G is None:
        return np.ones(n, dtype=bool)
    G = np.asarray(G, dtype=float)
    if G.ndim == 1 or G.shape[-1] == 0:
        return np.ones(n, dtype=bool)
    return np.all(np.nan_to_num(G, nan=np.inf) <= 0.0, axis=-1)


def _arg(cmd: List[str], name: str, default=None):
    return cmd[cmd.index(name) + 1] if name in cmd else default


@dataclass
class Run:
    run_dir: Path
    matrix: str
    variant: str
    method: str
    problem: str
    fatigue: float
    budget: int
    seed: int
    sigma: float
    env: Dict[str, str]
    w_star: np.ndarray
    r_star: np.ndarray
    ticks: (
        pd.DataFrame
    )  # per tick: tick, eval, hv, n_int, cost, latency, u_best, u_rec, w_err, r_err

    @property
    def key(self):
        return (self.problem, self.fatigue, self.budget, self.seed)


def load_run(run_dir: Path, runs_root: Path) -> Optional[Run]:
    run_dir = Path(run_dir)
    done = run_dir / "_jcde_done.json"
    if not done.is_file() or not (run_dir / "jcde_trace.npz").is_file():
        return None
    meta = json.loads(done.read_text(encoding="utf-8"))
    cmd = meta["cmd"]
    rel = run_dir.relative_to(runs_root).parts
    matrix = rel[0]
    variant = "/".join(
        p
        for p in rel[1:-1]
        if not re.match(r"^(SAV-AERO|ZDT\d|DTLZ\d|BNH|TNK|fat[\d.]+_B\d+)$", p)
        and p != _arg(cmd, "--method")  # per-method output folders are not variants
        and not re.match(
            r"^(jcde_r1|random_t|U2)$", p
        )  # Random-T builder nests jcde_r1/random_t/U2
    )
    trace = json.loads((run_dir / "jcde_trace.json").read_text(encoding="utf-8"))
    sdm = trace.get("sdm") or {}
    w_star, r_star = np.asarray(sdm["w_star"], float), np.asarray(sdm["r_star"], float)
    pm = sdm.get(
        "pref_model"
    )  # preference model v2: scaled objectives, monotone SDM utility
    if pm:
        s = np.asarray(pm["scale"], float)
        truth_u = lambda F: asf_mono(np.asarray(F, float) / s, w_star, r_star / s)  # noqa: E731
        est_u = lambda F, w, r: asf(
            np.asarray(F, float) / s, w, np.asarray(r, float) / s
        )  # noqa: E731
    else:
        truth_u = lambda F: asf(F, w_star, r_star)  # noqa: E731
        est_u = asf
    z = np.load(run_dir / "jcde_trace.npz", allow_pickle=True)
    pop_F = z["pop_F"]
    pop_G = z["pop_G"] if "pop_G" in z.files else None
    upv_w = trace.get("upv_w") or []
    upv_r = trace.get("upv_r") or []
    m = pd.read_csv(run_dir / "metrics.csv")
    n_t = min(len(pop_F), len(m))
    rows = []
    for t in range(n_t):
        F = pop_F[t]
        feas = feasible_mask(
            None if pop_G is None or pop_G.ndim < 3 else pop_G[t], len(F)
        )
        feas &= np.all(np.isfinite(F), axis=-1)
        if not feas.any():
            u_best = u_rec = np.nan
        else:
            Ff = F[feas]
            ut = truth_u(Ff)
            u_best = float(ut.min())
            w_t = (
                np.asarray(upv_w[t], float)
                if t < len(upv_w) and upv_w[t] is not None
                else None
            )
            r_t = (
                np.asarray(upv_r[t], float)
                if t < len(upv_r) and upv_r[t] is not None
                else None
            )
            u_rec = (
                float(ut[int(np.argmin(est_u(Ff, w_t, r_t)))])
                if w_t is not None and r_t is not None
                else np.nan
            )
        w_t = (
            np.asarray(upv_w[t], float)
            if t < len(upv_w) and upv_w[t] is not None
            else np.full_like(w_star, np.nan)
        )
        r_t = (
            np.asarray(upv_r[t], float)
            if t < len(upv_r) and upv_r[t] is not None
            else np.full_like(r_star, np.nan)
        )
        rows.append(
            (
                t,
                m["eval_count"].iloc[t],
                m["hv"].iloc[t],
                m["interaction_count"].iloc[t],
                m["cumulative_cognitive_cost"].iloc[t],
                m["dm_response_time"].iloc[t],
                u_best,
                u_rec,
                float(np.abs(w_t - w_star).sum()),
                r_t[0],
                r_t[1] if len(r_t) > 1 else np.nan,
            )
        )
    ticks = pd.DataFrame(
        rows,
        columns=[
            "tick",
            "eval",
            "hv",
            "n_int",
            "cost",
            "latency",
            "u_best",
            "u_rec",
            "w_err_l1",
            "r0_est",
            "r1_est",
        ],
    )
    return Run(
        run_dir,
        matrix,
        variant,
        _arg(cmd, "--method"),
        _arg(cmd, "--problem"),
        float(_arg(cmd, "--fatigue_rate")),
        int(_arg(cmd, "--B_total")),
        int(_arg(cmd, "--seed")),
        float(_arg(cmd, "--sigma_noise", 0.1)),
        meta.get("env") or {},
        w_star,
        r_star,
        ticks,
    )


def iter_runs(runs_root: Path, matrices: Iterable[str]) -> Iterable[Run]:
    for matrix in matrices:
        for done in sorted((runs_root / matrix).rglob("_jcde_done.json")):
            run = load_run(done.parent, runs_root)
            if run is not None:
                yield run


def truth_key(run: Run) -> tuple:
    return (run.problem, tuple(np.round(run.w_star, 6)), tuple(np.round(run.r_star, 6)))


def references(runs: List[Run]) -> pd.DataFrame:
    """U* (best known) and U0 (median initial best) per problem and ground truth, pooled over all runs given."""
    rows = {}
    for run in runs:
        k = truth_key(run)
        u = run.ticks["u_best"].to_numpy()
        rec = rows.setdefault(k, {"u_star": np.inf, "u0": []})
        if np.isfinite(u).any():
            rec["u_star"] = min(rec["u_star"], float(np.nanmin(u)))
        finite = u[np.isfinite(u)]
        if (
            finite.size
        ):  # first tick with a feasible design (SAV-AERO starts infeasible)
            rec["u0"].append(float(finite[0]))
    out = [
        {
            "problem": k[0],
            "w_star": k[1],
            "r_star": k[2],
            "u_star": v["u_star"],
            "u0": float(np.median(v["u0"])) if v["u0"] else np.nan,
            "n_runs": len(v["u0"]),
        }
        for k, v in rows.items()
    ]
    return pd.DataFrame(out)


def summarise(runs: List[Run], refs: pd.DataFrame) -> pd.DataFrame:
    ref = {
        (r.problem, tuple(r.w_star), tuple(r.r_star)): (r.u_star, r.u0)
        for r in refs.itertuples()
    }
    rows = []
    for run in runs:
        u_star, u0 = ref[truth_key(run)]
        gap = max(u0 - u_star, 1e-12)
        t = run.ticks
        reg_rec = (t["u_rec"] - u_star) / gap
        reg_best = (t["u_best"] - u_star) / gap
        rows.append(
            {
                "matrix": run.matrix,
                "variant": run.variant,
                "method": run.method,
                "problem": run.problem,
                "fatigue": run.fatigue,
                "budget": run.budget,
                "seed": run.seed,
                "env": json.dumps(run.env, sort_keys=True) if run.env else "",
                "terminal_hv": t["hv"].iloc[-1],
                "hv_auc": float(
                    np.trapezoid(t["hv"], t["eval"])
                    / (t["eval"].iloc[-1] - t["eval"].iloc[0])
                ),
                "regret_rec": float(reg_rec.iloc[-1]),
                "regret_best": float(reg_best.iloc[-1]),
                "regret_rec_auc": float(np.nanmean(reg_rec)),
                "regret_best_auc": float(np.nanmean(reg_best)),
                "w_err_l1": float(t["w_err_l1"].iloc[-1]),
                "N_int": int(t["n_int"].iloc[-1]),
                "C_cog": float(t["cost"].iloc[-1]),
                "T_resp": float(t.loc[t["latency"] > 0, "latency"].mean())
                if (t["latency"] > 0).any()
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def paired_effect(
    summary: pd.DataFrame,
    a: str,
    b: str,
    metric: str,
    lower_is_better: bool,
    keys=("problem", "fatigue", "budget", "seed"),
    reps: int = 20_000,
    seed: int = 20261021,
) -> dict:
    """Mean paired difference (a minus b, sign-adjusted so positive favours a) with a cluster bootstrap CI.

    Primary CI (ci_low, ci_high, ci_excludes_zero; decision of 10-24): cells that share a seed share the SDM noise,
    the corruption draws and the optimizer RNG, so when `keys` contain the seed the clusters are (problem, seed) and
    the resampled statistic is the ratio of cluster sums over cluster sizes. Without a seed in `keys` every paired
    row is its own independent unit (e.g. one LHS sample in E6'). The cell-level CI that treats every row as
    independent is kept as cell_ci_low / cell_ci_high; clusters_favour counts clusters whose summed advantage > 0.
    """
    A = summary[summary["method"] == a].set_index(list(keys))[metric]
    B = summary[summary["method"] == b].set_index(list(keys))[metric]
    joined = pd.concat([A.rename("a"), B.rename("b")], axis=1, join="inner").dropna()
    if joined.empty:
        return {"a": a, "b": b, "metric": metric, "n": 0}
    adv = (
        (joined["b"] - joined["a"]) if lower_is_better else (joined["a"] - joined["b"])
    )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(adv), size=(reps, len(adv)))
    boot = adv.to_numpy()[idx].mean(axis=1)
    cell_lo, cell_hi = np.quantile(boot, (0.025, 0.975))
    levels = [k for k in ("problem", "seed") if k in keys] if "seed" in keys else []
    if levels:
        per = adv.groupby(level=levels).agg(["sum", "size"])
        lo, hi = cluster_ci(per["sum"].to_numpy(), per["size"].to_numpy(), reps, seed)
        n_cl, n_fav = len(per), int((per["sum"] > 0).sum())
    else:
        lo, hi, n_cl, n_fav = cell_lo, cell_hi, len(adv), int((adv > 0).sum())
    return {
        "a": a,
        "b": b,
        "metric": metric,
        "n": len(adv),
        "mean_a": joined["a"].mean(),
        "mean_b": joined["b"].mean(),
        "advantage": adv.mean(),
        "ci_low": lo,
        "ci_high": hi,
        "p_favour_a": float((adv > 0).mean()),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "n_clusters": n_cl,
        "clusters_favour": n_fav,
        "cell_ci_low": cell_lo,
        "cell_ci_high": cell_hi,
    }


def cluster_ci(
    sums: np.ndarray, sizes: np.ndarray, reps: int = 20_000, seed: int = 20261021
) -> tuple:
    """95% percentile CI of a mean when whole clusters (with these sums and sizes) are resampled."""
    idx = np.random.default_rng(seed).integers(0, len(sums), (reps, len(sums)))
    lo, hi = np.quantile(sums[idx].sum(1) / sizes[idx].sum(1), (0.025, 0.975))
    return lo, hi
