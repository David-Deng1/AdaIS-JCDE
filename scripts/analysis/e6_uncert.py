"""E6' (R3-4c): propagation of the set behavioural parameters of the parametric decision maker.

The 60 Latin-hypercube samples (p3_uncert_design.json: channel costs, latency scale, BETA_LAT, GAMMA_QUAL, answer
noise sigma, fatigue) each run AdaIS, VoI-Timing-MC and Fixed-8 on the same seed. Per comparator and metric:
mean paired advantage of AdaIS with a bootstrap CI over samples, the share of samples in which AdaIS is better
(Wilson 95% interval), and the Spearman correlation of the advantage with each sampled parameter.
Independent unit: the LHS sample (each sample has its own seed and parameter set), so the bootstrap resamples
samples (te.paired_effect with keys = (variant,)); there is no seed cluster to form.

Usage: python e6_uncert.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth_eval as te  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jcde_paths  # noqa: E402
RUNS = jcde_paths.P3_RUNS
DESIGN = RUNS / "specs" / "p3_uncert_design.json"
MATRIX = "p3_uncert"
METRICS = [("regret_rec", True), ("C_cog", True), ("N_int", True), ("terminal_hv", False), ("hv_auc", False),
           ("w_err_l1", True)]  # fmt: skip


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(jcde_paths.ANALYSIS / "e6_uncert"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    runs = list(te.iter_runs(RUNS, [MATRIX]))
    summary = te.summarise(runs, te.references(runs))
    summary.to_csv(out / "e6u_runs.csv", index=False)
    design = pd.DataFrame(json.loads(DESIGN.read_text(encoding="utf-8")))
    design["variant"] = design["sample"].map(lambda k: f"s{k:03d}")
    params = [c for c in design.columns if c not in ("sample", "seed", "variant")]
    rows, corr = [], []
    for other in ("VoI-Timing-MC", "Baseline-F8"):
        for metric, lower in METRICS:
            eff = te.paired_effect(
                summary, "AdaIS", other, metric, lower, keys=("variant",)
            )
            A = summary[summary["method"] == "AdaIS"].set_index("variant")[metric]
            B = summary[summary["method"] == other].set_index("variant")[metric]
            adv = ((B - A) if lower else (A - B)).dropna()
            k = int((adv > 0).sum())
            lo, hi = wilson(k, len(adv))
            rows.append(
                {
                    **eff,
                    "n_better": k,
                    "n_tied": int((adv == 0).sum()),
                    "share_lo": lo,
                    "share_hi": hi,
                }
            )
            d = design.set_index("variant").loc[adv.index]
            for p in params:
                r, pv = spearmanr(d[p], adv)
                corr.append(
                    {"b": other, "metric": metric, "param": p, "rho": r, "p": pv}
                )
    effects = pd.DataFrame(rows)
    effects.to_csv(out / "e6u_paired_effects.csv", index=False)
    corr = pd.DataFrame(corr)
    corr.to_csv(out / "e6u_param_correlations.csv", index=False)
    pd.set_option("display.width", 220)
    keep = ["b", "metric", "n", "mean_a", "mean_b", "advantage", "ci_low", "ci_high", "n_better", "n_tied",
            "share_lo", "share_hi"]  # fmt: skip
    print(effects[keep].round(4).to_string(index=False))
    print(corr[corr["p"] < 0.01].round(4).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
