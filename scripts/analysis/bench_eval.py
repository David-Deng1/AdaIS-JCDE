"""Benchmark sanity check (R1-9, R2-9, R3-5): AdaIS against the ablations and fixed-pace / random baselines on
ZDT1-3, BNH and TNK with the known-utility decision maker (held-out seeds 11-20).

Effects are paired by (problem, fatigue, budget, seed) and reported per problem only: regret scales differ
between problems (constrained BNH / TNK have a small U0 - U* gap), so a pooled mean would be driven by them.
A sign table (AdaIS better / worse / no detectable difference, 95% block-bootstrap CI) summarises each cell.

Usage: python bench_eval.py [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth_eval as te  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jcde_paths  # noqa: E402
RUNS = jcde_paths.P3_RUNS
METRICS = [
    ("regret_rec", True),
    ("terminal_hv", False),
    ("hv_auc", False),
    ("C_cog", True),
    ("N_int", True),
]
OTHERS = ["AdaS-NoQ", "AdaS-FixedT8", "AdaS-UniCh", "Baseline-F2", "Baseline-F", "Baseline-F8", "Baseline-F16",
          "Baseline-R"]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(jcde_paths.ANALYSIS / "benchmark"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    runs = list(te.iter_runs(RUNS, ["p3_benchmark"]))
    summary = te.summarise(runs, te.references(runs))
    summary.to_csv(out / "bench_runs.csv", index=False)
    summary.groupby(["problem", "method"])[[m for m, _ in METRICS]].mean().round(
        4
    ).to_csv(out / "bench_means.csv")
    rows = []
    for problem in sorted(summary["problem"].unique()):
        s = summary[summary["problem"] == problem]
        for other in OTHERS:
            for metric, lower in METRICS:
                rows.append(
                    {
                        "problem": problem,
                        **te.paired_effect(s, "AdaIS", other, metric, lower),
                    }
                )
    eff = pd.DataFrame(rows)
    eff.to_csv(out / "bench_paired_effects.csv", index=False)
    eff["sign"] = eff.apply(
        lambda r: "+" if r["ci_low"] > 0 else ("-" if r["ci_high"] < 0 else "0"), axis=1
    )
    signs = eff.pivot_table(
        index=["metric", "b"], columns="problem", values="sign", aggfunc="first"
    )
    signs.to_csv(out / "bench_sign_table.csv")
    pd.set_option("display.width", 200)
    print(signs.to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
