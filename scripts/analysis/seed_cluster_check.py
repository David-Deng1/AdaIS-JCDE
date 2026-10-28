"""Sensitivity of every paired effect to the bootstrap unit (10-24). paired_effect resamples cells (problem, fatigue,
budget, seed) as if independent, but cells sharing a seed share the SDM noise, the corruption draws and the optimizer
RNG (the E7 regret tail sits on seeds 59-60 across all fatigue x budget cells). Here the same advantages are
resampled with (problem, seed) as clusters. Applied to every comparator and metric alike, favourable or not.
Since the seed clusters were adopted as the primary inference (10-24), te.paired_effect returns them as
ci_low / ci_high; this script recomputes them independently and serves as the cross-check.

Outputs: seed_cluster_effects.csv (cell CI, seed-cluster CI, clusters favouring AdaIS) and seed_cluster_beta_gap.csv.

Usage: python seed_cluster_check.py [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth_eval as te  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jcde_paths  # noqa: E402
A = jcde_paths.ANALYSIS
SETS = {"P-balanced": "g2/g2_runs.csv", "P-edge": "e2_edge/g2_runs.csv", "corruption": "e7_corruption/e7_runs.csv",
        "nominal-ablation": "e_ablation/g2_runs.csv", "benchmark": "benchmark/bench_runs.csv",
        "e9a": "e9a_voi_allch/e9a_runs.csv"}  # fmt: skip
METRICS = [("regret_rec", True), ("terminal_hv", False), ("hv_auc", False), ("w_err_l1", True), ("N_int", True),
           ("C_cog", True)]  # fmt: skip
KEYS = ["problem", "fatigue", "budget", "seed"]
REPS, SEED = 20_000, 20261021


def boot_ratio(s: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    idx = np.random.default_rng(SEED).integers(0, len(s), (REPS, len(s)))
    lo, hi = np.quantile(s[idx].sum(1) / n[idx].sum(1), (0.025, 0.975))
    return lo, hi


def cluster(
    d: pd.DataFrame, b: str, metric: str, lower: bool
) -> tuple[float, float, int, int]:
    """Seed-cluster CI of the mean paired advantage, and how many clusters favour AdaIS."""
    a_ = d[d.method == "AdaIS"].set_index(KEYS)[metric]
    b_ = d[d.method == b].set_index(KEYS)[metric]
    j = pd.concat([a_.rename("a"), b_.rename("b")], axis=1, join="inner").dropna()
    adv = (j["b"] - j["a"]) if lower else (j["a"] - j["b"])
    per = adv.groupby(level=["problem", "seed"]).agg(["sum", "size"])
    lo, hi = boot_ratio(per["sum"].to_numpy(), per["size"].to_numpy())
    return lo, hi, int((per["sum"] > 0).sum()), len(per)


def effects(name: str, d: pd.DataFrame) -> list[dict]:
    rows = []
    scopes = [("pooled", d)] + (
        list(d.groupby("problem")) if d.problem.nunique() > 1 else []
    )
    for scope, g in scopes:
        for b in sorted(set(g.method) - {"AdaIS"}):
            for metric, lower in METRICS:
                if name == "P-balanced" and metric == "w_err_l1":
                    continue  # truth weights equal the initial ones: not informative
                e = te.paired_effect(g, "AdaIS", b, metric, lower)
                if not e.get("n"):
                    continue
                lo, hi, npos, ncl = cluster(g, b, metric, lower)
                rows.append({"set": name, "scope": scope, "b": b, "metric": metric, "n": e["n"],
                             "advantage": e["advantage"], "cell_lo": e["cell_ci_low"], "cell_hi": e["cell_ci_high"],
                             "seed_lo": lo, "seed_hi": hi, "clusters_positive": npos, "clusters": ncl,
                             "cell_sig": bool(e["cell_ci_low"] > 0 or e["cell_ci_high"] < 0), "seed_sig": bool(lo > 0 or hi < 0)})  # fmt: skip
    return rows


def beta_gap(e7: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in e7.groupby("method"):
        g = g.assign(d=g.beta_on_clean - g.beta_on_corrupted).dropna(subset=["d"])
        per = g.groupby("seed")["d"].agg(["sum", "size"])
        lo, hi = boot_ratio(per["sum"].to_numpy(), per["size"].to_numpy())
        rows.append({"method": m, "n_runs": len(g), "beta_gap": g.d.mean(), "seed_lo": lo, "seed_hi": hi,
                     "seeds_positive": int((per["sum"] > 0).sum()), "seeds": len(per)})  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(A / "seed_cluster"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, f in SETS.items():
        rows += effects(name, pd.read_csv(A / f))
    for (host, var), g in pd.read_csv(A / "e5_host/e5_runs.csv").groupby(
        ["host", "variant"]
    ):
        rows += effects(f"host:{host}:{var}", g)
    for form, g in pd.read_csv(A / "e6_fatigue/e6_fatigue_runs.csv").groupby("form"):
        rows += effects(f"fatigue:{form}", g)
    r = pd.DataFrame(rows)
    r.to_csv(out / "seed_cluster_effects.csv", index=False)
    gap = beta_gap(pd.read_csv(A / "e7_corruption/e7_runs.csv"))
    gap.to_csv(out / "seed_cluster_beta_gap.csv", index=False)
    lost, gained = (r.cell_sig & ~r.seed_sig), (~r.cell_sig & r.seed_sig)
    print(f"effects {len(r)}  cell-sig {int(r.cell_sig.sum())}  seed-sig {int(r.seed_sig.sum())}  "
          f"lost {int(lost.sum())} (favourable {int((lost & (r.advantage > 0)).sum())}, "
          f"adverse {int((lost & (r.advantage < 0)).sum())})  gained {int(gained.sum())}")  # fmt: skip
    print(gap.round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
