"""G2 (route P3, preference model v2): direction of the ground-truth regret on the held-out aero matrix.

AdaIS against every comparator in p3_core (and count-matched Random-T once p3_core_random_t exists), paired by
(fatigue, budget, seed). Positive advantage favours AdaIS. Regret uses truth_eval.py with U* and U0 pooled over
all runs of the matrix. Only blocks where both arms finished are paired, so a partial matrix gives a preview.

Usage: python g2_direction.py [--matrices p3_core p3_core_random_t] [--out DIR]
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
METRICS = [("regret_rec", True), ("regret_best", True), ("w_err_l1", True), ("terminal_hv", False),
           ("hv_auc", False), ("N_int", True), ("C_cog", True), ("T_resp", True)]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="+", default=["p3_core", "p3_core_random_t"])
    ap.add_argument("--out", default=str(jcde_paths.ANALYSIS / "g2"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    matrices = [m for m in args.matrices if (RUNS / m).exists()]
    runs = list(te.iter_runs(RUNS, matrices))
    refs = te.references(runs)
    summary = te.summarise(runs, refs)
    summary.to_csv(out / "g2_runs.csv", index=False)
    means = summary.groupby("method")[[m for m, _ in METRICS]].mean().round(4)
    means["n"] = summary.groupby("method").size()
    rows = []
    for other in sorted(set(summary["method"]) - {"AdaIS"}):
        for metric, lower in METRICS:
            rows.append(te.paired_effect(summary, "AdaIS", other, metric, lower))
    effects = pd.DataFrame(rows)
    effects.to_csv(out / "g2_paired_effects.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"matrices: {matrices}; runs: {len(summary)}")
    print(means.to_string())
    cols = [
        "b",
        "metric",
        "n",
        "mean_a",
        "mean_b",
        "advantage",
        "ci_low",
        "ci_high",
        "p_favour_a",
    ]
    print(
        effects[effects["metric"].isin(["regret_rec", "terminal_hv", "C_cog"])][cols]
        .round(4)
        .to_string(index=False)
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
