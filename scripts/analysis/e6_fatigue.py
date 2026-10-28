"""E6 fatigue forms (R3-4e): AdaIS against NoQ and Random-C under linear, saturating and step fatigue.

The linear form is the common-corruption matrix itself (p3_corruption, variant ""); the saturating and step forms
are the p3_fatigue variants with the same corruption model. Budget 10 only (the budget p3_fatigue was run at).
References are pooled over both matrices; effects are paired by (fatigue, budget, seed) within each form.

Usage: python e6_fatigue.py [--out DIR]
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
    ("w_err_l1", True),
    ("terminal_hv", False),
    ("C_cog", True),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(jcde_paths.ANALYSIS / "e6_fatigue"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    runs = list(te.iter_runs(RUNS, ["p3_corruption", "p3_fatigue"]))
    s = te.summarise(runs, te.references(runs))
    s["form"] = s["variant"].map(
        lambda v: v if v in ("saturating", "step") else "linear"
    )
    s = s[(s["budget"] == 10) & s["method"].isin(["AdaIS", "AdaS-NoQ", "Baseline-R"])]
    s.to_csv(out / "e6_fatigue_runs.csv", index=False)
    rows = []
    for form, g in s.groupby("form"):
        for b in ("AdaS-NoQ", "Baseline-R"):
            for metric, lower in METRICS:
                r = te.paired_effect(g, "AdaIS", b, metric, lower)
                r["form"] = form
                rows.append(r)
    e = pd.DataFrame(rows)
    e.to_csv(out / "e6_fatigue_effects.csv", index=False)
    print(
        e[["form", "b", "metric", "n", "advantage", "ci_low", "ci_high"]]
        .round(4)
        .to_string(index=False)
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
