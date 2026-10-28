"""E6 scans (R3-4a/b, R2-5, R4-4, R4-6): sensitivity of the P3 design to the NL respondent and to the set constants.

Each scan arm is paired by (fatigue, budget, seed) with the default arm of the matrix it perturbs:
    nl_exact            AdaIS and Fixed-8 with exact NL slot values  vs  p3_core (pipeline NL respondent)
    kappa_nl{k}         AdaIS with NL burden coefficient k           vs  p3_core AdaIS (kappa_NL = 0.60)
    quality_a1{x}_a3{y} AdaIS with quality-score weights alpha1, alpha3  vs  p3_corruption AdaIS (0.8 / 0.8)
    tauQ{hi}-{lo}       AdaIS with monitor thresholds tau_Q          vs  p3_corruption AdaIS (0.55 / 0.25)
The corruption-side arms also report the corrupted evidence absorbed (e7_corruption.exposure) and the comparison
with AdaS-NoQ, i.e. whether the monitor's advantage survives the perturbation. References (U*, U0) are pooled over
all runs loaded, so every arm shares one regret scale. CIs resample seeds as clusters (te.paired_effect convention,
keys contain the seed).

Usage: python e6_scans.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth_eval as te  # noqa: E402
from e7_corruption import exposure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jcde_paths  # noqa: E402
RUNS = jcde_paths.P3_RUNS
KEEP = {
    "p3_core": ("AdaIS", "Baseline-F8"),
    "p3_corruption": ("AdaIS", "AdaS-NoQ"),
    "p3_scans": None,
}
METRICS = [("regret_rec", True), ("terminal_hv", False), ("hv_auc", False), ("C_cog", True), ("N_int", True),
           ("w_err_l1", True)]  # fmt: skip
KEYS = ("fatigue", "budget", "seed")


def load() -> list:
    runs = []
    for matrix, methods in KEEP.items():
        for done in sorted((RUNS / matrix).rglob("_jcde_done.json")):
            if methods and not done.parent.name.startswith(
                tuple(f"{m}_SAV-AERO" for m in methods)
            ):
                continue
            run = te.load_run(done.parent, RUNS)
            if run is not None:
                runs.append(run)
    return runs


def channels(run_dir: Path) -> dict:
    lines = (run_dir / "interaction_log.jsonl").read_text(encoding="utf-8").splitlines()
    n = Counter(json.loads(line)["channel"] for line in lines)
    return {
        "n_nl": n.get("nl_clarify", 0),
        "nl_share": n.get("nl_clarify", 0) / max(sum(n.values()), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(jcde_paths.ANALYSIS / "e6_scans"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load()
    summary = te.summarise(runs, te.references(runs)).reset_index(drop=True)
    extra = [
        {
            **channels(r.run_dir),
            **(exposure(r.run_dir) if r.matrix != "p3_core" else {}),
        }
        for r in runs
    ]
    summary = pd.concat([summary, pd.DataFrame(extra)], axis=1)
    base = {"p3_core": "pipeline", "p3_corruption": "default"}
    summary["arm"] = [f"{m}|{v or base[mx]}" for m, v, mx in zip(summary["method"], summary["variant"],
                                                                 summary["matrix"])]  # fmt: skip
    summary.to_csv(out / "e6s_runs.csv", index=False)
    cols = [
        "regret_rec",
        "terminal_hv",
        "C_cog",
        "N_int",
        "n_nl",
        "nl_share",
        "w_err_l1",
        "corrupt_share_absorbed",
    ]
    summary.groupby("arm")[cols].mean().round(4).to_csv(out / "e6s_means.csv")

    def pair(a: str, b: str, metrics=METRICS, blocks=None) -> list:
        s = summary.assign(method=summary["arm"])
        if blocks is not None:
            s = s[s.set_index(list(KEYS)).index.isin(blocks)]
        return [te.paired_effect(s, a, b, m, lower, keys=KEYS) for m, lower in metrics]

    rows = []
    rows += pair("AdaIS|nl_exact", "AdaIS|pipeline")
    rows += pair("AdaIS|nl_exact", "Baseline-F8|nl_exact")
    exact_blocks = (
        summary[summary["arm"] == "AdaIS|nl_exact"].set_index(list(KEYS)).index
    )
    rows += pair(
        "AdaIS|pipeline", "Baseline-F8|pipeline", blocks=exact_blocks
    )  # same blocks as the exact arm
    for arm in sorted(a for a in summary["arm"].unique() if "|kappa_nl" in a):
        rows += pair(arm, "AdaIS|pipeline", METRICS + [("n_nl", False)])
    corr_metrics = METRICS + [("corrupt_share_absorbed", True)]
    for arm in sorted(
        a for a in summary["arm"].unique() if "|quality_" in a or "|tauQ" in a
    ):
        rows += pair(arm, "AdaIS|default", corr_metrics)
        rows += pair(arm, "AdaS-NoQ|default", corr_metrics)
    rows += pair("AdaIS|default", "AdaS-NoQ|default", corr_metrics)
    effects = pd.DataFrame(rows)
    effects.to_csv(out / "e6s_paired_effects.csv", index=False)
    pd.set_option("display.width", 230)
    print(summary.groupby("arm")[cols].mean().round(3).to_string())
    keep = [
        "a",
        "b",
        "metric",
        "n",
        "advantage",
        "ci_low",
        "ci_high",
        "ci_excludes_zero",
    ]
    print(effects[keep].round(4).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
