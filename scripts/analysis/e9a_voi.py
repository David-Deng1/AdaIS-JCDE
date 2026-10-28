"""E9a (R4-7): VoI-Timing-MC with a posterior updated by every channel (VoI-Timing-MC-AllCh) against the original
pairwise-only VoI-Timing-MC and against AdaIS, on the held-out aero blocks of p3_core.

References (U*, U0) are pooled over p3_core and p3_e9a; effects are paired by (fatigue, budget, seed).

Usage: python e9a_voi.py [--out DIR]
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
    ("hv_auc", False),
    ("C_cog", True),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(jcde_paths.ANALYSIS / "e9a_voi_allch"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    runs = list(te.iter_runs(RUNS, ["p3_core", "p3_e9a"]))
    s = te.summarise(runs, te.references(runs))
    s = s[s["method"].isin(["AdaIS", "VoI-Timing-MC", "VoI-Timing-MC-AllCh"])]
    s.to_csv(out / "e9a_runs.csv", index=False)
    print(
        s.groupby("method")[
            ["regret_rec", "w_err_l1", "terminal_hv", "hv_auc", "N_int", "C_cog"]
        ]
        .mean()
        .round(4)
        .to_string()
    )
    rows = []
    for a, b in (
        ("VoI-Timing-MC-AllCh", "VoI-Timing-MC"),
        ("AdaIS", "VoI-Timing-MC-AllCh"),
    ):
        for metric, lower in METRICS:
            rows.append(te.paired_effect(s, a, b, metric, lower))
    e = pd.DataFrame(rows)
    e.to_csv(out / "e9a_paired_effects.csv", index=False)
    print(
        e[["a", "b", "metric", "n", "advantage", "ci_low", "ci_high"]]
        .round(4)
        .to_string(index=False)
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
