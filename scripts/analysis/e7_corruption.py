"""E7 / E8 (R4-5, R4-6, R2-2, R3-8): quality monitor on the common-corruption matrix, judged by evidence that does
not come from the monitor itself.

Corruption is drawn by the simulator (independent of the monitor's flags), so every interaction carries a ground-
truth "corrupted" label. Per run: number of corrupted answers, the update weight beta they received, the corrupted
evidence actually absorbed (sum of beta over corrupted answers) and its share of all absorbed evidence; plus the
truth-based endpoints (regret, weight-recovery error) from truth_eval. Paired effects of AdaIS against NoQ, NoQ-pre,
NoQ-post, Fixed-8 and Random-C are formed per (fatigue, budget, seed), overall and per fatigue level.

Also written: the beta gap against the corruption labels (e7_beta_gap), the regret distribution per method
(e7_regret_tail), AdaIS against NoQ-post block by block (e7_noqpost_pairs), and what the high-regret runs share
(e7_tail_profile: reference-point error in scaled units, threshold answers and their beta).

Usage: python e7_corruption.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth_eval as te  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jcde_paths  # noqa: E402
RUNS = jcde_paths.P3_RUNS
MATRIX = "p3_corruption"


def exposure(run_dir: Path) -> dict:
    """Corruption exposure over preference evidence only: ROI answers never update w or r (submitted model,
    kept in P3) and the corruption model never draws on them, so they are left out of every count."""
    n = n_corr = 0
    absorbed = absorbed_corr = 0.0
    for line in (
        (run_dir / "interaction_log.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        rec = json.loads(line)
        if rec.get("channel") == "roi":
            continue
        resp = rec.get("response") or {}
        corr = resp.get("corrupted")
        if corr is None and resp.get("results"):
            corr = any(r.get("corrupted") for r in resp["results"])
        beta = float(rec.get("update_weight", 1.0))
        n += 1
        absorbed += beta
        if corr:
            n_corr += 1
            absorbed_corr += beta
    return {"n_answers": n, "n_corrupted": n_corr, "absorbed": absorbed, "absorbed_corrupted": absorbed_corr,
            "corrupt_share_absorbed": absorbed_corr / absorbed if absorbed else 0.0,
            "beta_on_corrupted": absorbed_corr / n_corr if n_corr else float("nan"),
            "beta_on_clean": (absorbed - absorbed_corr) / (n - n_corr) if n > n_corr else float("nan")}  # fmt: skip


def reference_point(run) -> dict:
    """Final reference-point error in scaled objective units (signed: + = looser than the truth) and the threshold
    answers that carry it (count, mean beta)."""
    tr = json.loads((run.run_dir / "jcde_trace.json").read_text(encoding="utf-8"))
    scale = np.asarray(tr["sdm"]["pref_model"]["scale"], float)
    last = run.ticks.iloc[-1]
    err = (np.array([last["r0_est"], last["r1_est"]]) - run.r_star) / scale
    thr = [rec for rec in map(json.loads, (run.run_dir / "interaction_log.jsonl").read_text(encoding="utf-8")
           .splitlines()) if rec.get("channel") == "threshold"]  # fmt: skip
    beta = [float(rec.get("update_weight", 1.0)) for rec in thr]
    r0 = [
        bool((rec.get("response") or {}).get("corrupted"))
        for rec in thr
        if (rec.get("response") or {}).get("slot") == "r_0"
    ]
    return {"r0_err_scaled": err[0], "r1_err_scaled": err[1], "n_threshold": len(beta),
            "beta_threshold": float(np.mean(beta)) if beta else float("nan"),
            "r0_corrupted_any": float(any(r0)), "r0_last_corrupted": float(r0[-1]) if r0 else float("nan")}  # fmt: skip


def tail_profile(summary: pd.DataFrame, cut: float = 2.0) -> pd.DataFrame:
    """Runs with regret above `cut` against the rest, per method: what differs in them (R4-6 discussion)."""
    s = summary.assign(tail=summary["regret_rec"] > cut)
    g = s.groupby(["method", "tail"])
    return g[["regret_rec", "w_err_l1", "r0_err_scaled", "r1_err_scaled", "n_threshold", "beta_threshold",
              "r0_corrupted_any", "r0_last_corrupted"]].mean().assign(
        n_runs=g.size(), seeds=g["seed"].apply(lambda x: " ".join(map(str, sorted(set(x)))))).reset_index()  # fmt: skip


def beta_gap(
    summary: pd.DataFrame, reps: int = 20_000, seed: int = 20261021
) -> pd.DataFrame:
    """Monitor discrimination against the simulator's corruption labels: per run, mean beta on clean answers minus
    mean beta on corrupted ones (runs without a corrupted answer drop out). Primary CI: runs resampled by seed
    cluster (te.paired_effect convention); the run-level CI is kept as cell_ci_low / cell_ci_high."""
    rows = []
    for method, g in summary.groupby("method"):
        g = g.assign(d=g["beta_on_clean"] - g["beta_on_corrupted"]).dropna(subset=["d"])
        d = g["d"].to_numpy()
        if not len(d):
            continue
        boot = d[np.random.default_rng(seed).integers(0, len(d), (reps, len(d)))].mean(
            axis=1
        )
        cell_lo, cell_hi = np.quantile(boot, (0.025, 0.975))
        per = g.groupby("seed")["d"].agg(["sum", "size"])
        lo, hi = te.cluster_ci(
            per["sum"].to_numpy(), per["size"].to_numpy(), reps, seed
        )
        rows.append({"method": method, "n_runs": len(d), "beta_gap": d.mean(), "ci_low": lo, "ci_high": hi,
                     "runs_gap_positive": int((d > 0).sum()), "n_clusters": len(per),
                     "clusters_favour": int((per["sum"] > 0).sum()), "cell_ci_low": cell_lo,
                     "cell_ci_high": cell_hi})  # fmt: skip
    return pd.DataFrame(rows)


def regret_tail(summary: pd.DataFrame) -> pd.DataFrame:
    """Distribution of the recommendation regret per method: a mean difference can come from a few heavy runs."""
    q = summary.groupby("method")["regret_rec"]
    return pd.DataFrame({"n": q.size(), "mean": q.mean(), "median": q.median(), "p10": q.quantile(0.1),
                         "p90": q.quantile(0.9), "n_above_2": q.apply(lambda r: int((r > 2).sum()))}).reset_index()  # fmt: skip


def noqpost_pairs(summary: pd.DataFrame) -> pd.DataFrame:
    """AdaIS against NoQ-post block by block (advantage > 0 favours AdaIS), with each side's weight error."""
    k = ["fatigue", "budget", "seed"]
    A = summary[summary["method"] == "AdaIS"].set_index(k)
    B = summary[summary["method"] == "AdaS-NoQpost"].set_index(k)
    j = pd.concat([A["regret_rec"].rename("regret_adais"), B["regret_rec"].rename("regret_noqpost"),
                   A["w_err_l1"].rename("w_err_adais"), B["w_err_l1"].rename("w_err_noqpost")], axis=1, join="inner")  # fmt: skip
    j["advantage"] = j["regret_noqpost"] - j["regret_adais"]
    return j.reset_index().sort_values("advantage")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(jcde_paths.ANALYSIS / "e7_corruption"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    runs = list(te.iter_runs(RUNS, [MATRIX]))
    summary = te.summarise(runs, te.references(runs)).reset_index(drop=True)
    summary = pd.concat(
        [
            summary,
            pd.DataFrame([exposure(r.run_dir) for r in runs]),
            pd.DataFrame([reference_point(r) for r in runs]),
        ],
        axis=1,
    )
    summary.to_csv(out / "e7_runs.csv", index=False)
    cols = ["regret_rec", "w_err_l1", "terminal_hv", "N_int", "C_cog", "n_corrupted", "absorbed_corrupted",
            "corrupt_share_absorbed", "beta_on_corrupted", "beta_on_clean"]  # fmt: skip
    means = summary.groupby("method")[cols].mean().round(4)
    means.to_csv(out / "e7_means.csv")
    rows = []
    metrics = [("regret_rec", True), ("w_err_l1", True), ("terminal_hv", False), ("N_int", True), ("C_cog", True),
               ("absorbed_corrupted", True), ("corrupt_share_absorbed", True)]  # fmt: skip
    for other in (
        "AdaS-NoQ",
        "AdaS-NoQpre",
        "AdaS-NoQpost",
        "Baseline-F8",
        "Baseline-R",
        "AdaS-FixedT8",
        "AdaS-UniCh",
    ):
        for metric, lower in metrics:
            rows.append(te.paired_effect(summary, "AdaIS", other, metric, lower))
    effects = pd.DataFrame(rows)
    effects.to_csv(out / "e7_paired_effects.csv", index=False)
    by_fat = []  # the submitted high-fatigue boundary (Fig. 17) is the fatigue = 1.3 slice of this matrix
    for fat, g in summary.groupby("fatigue"):
        for other in ("AdaS-NoQ", "AdaS-NoQpost", "Baseline-F8"):
            for metric, lower in metrics[:5]:
                by_fat.append(
                    {
                        "fatigue": fat,
                        **te.paired_effect(g, "AdaIS", other, metric, lower),
                    }
                )
    pd.DataFrame(by_fat).to_csv(out / "e7_by_fatigue.csv", index=False)
    gap, tail, pairs = beta_gap(summary), regret_tail(summary), noqpost_pairs(summary)
    tail_profile(summary).to_csv(out / "e7_tail_profile.csv", index=False)
    gap.to_csv(out / "e7_beta_gap.csv", index=False)
    tail.to_csv(out / "e7_regret_tail.csv", index=False)
    pairs.to_csv(out / "e7_noqpost_pairs.csv", index=False)
    pd.set_option("display.width", 220)
    print(gap.round(4).to_string(index=False))
    print(tail.round(3).to_string(index=False))
    print(means.to_string())
    keep = ["b", "metric", "n", "mean_a", "mean_b", "advantage", "ci_low", "ci_high"]
    print(
        effects[effects["b"].str.startswith("AdaS-NoQ")][keep]
        .round(4)
        .to_string(index=False)
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
