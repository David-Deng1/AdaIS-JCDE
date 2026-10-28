"""E5 (host transfer): does the AdaIS comparison survive a different optimizer host?

Two single hosts. JCDE_HOST = nsga2 (p3_host) is the A1 path: NSGA-II optimising with a neutral UPV, so its search
ignores the elicited preferences and populations (hence HV) are identical across methods; only the recommendation
and the burden can differ. JCDE_HOST = single_nsga2 (p3_host_single, aero only) keeps the default host's
preference coupling and drops only the solver mixture (one NSGA-II generator).
    bench   ZDT1-3, BNH, TNK x fatigue x budget x seeds 11-15 (nsga2): AdaIS vs AdaS-NoQ, Fixed-8, Random-C
    aero    SAV-AERO P-balanced x fatigue x budget x seeds 51-60 (both hosts): AdaIS vs VoI-Timing-MC and vs the
            count-matched Random-T (p3_host_random_t, p3_host_single_random_t)
For the aero arm the same paired effects on the default host (p3_core, p3_core_random_t) are printed alongside,
so the transfer is read as a change in effect size, not only a sign. Pairing keys: (problem, fatigue, budget, seed).

Usage: python e5_host.py [--out DIR]
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
DEFAULT_HOST = {
    "p3_core": ("AdaIS", "VoI-Timing-MC"),
    "p3_core_random_t": ("AdaS-RandomT",),
}


def load(matrices: dict) -> list:
    runs = []
    for matrix, methods in matrices.items():
        for done in sorted((RUNS / matrix).rglob("_jcde_done.json")):
            if methods and not done.parent.name.startswith(
                tuple(f"{m}_" for m in methods)
            ):
                continue
            run = te.load_run(done.parent, RUNS)
            if run is not None:
                runs.append(run)
    return runs


def effects(summary: pd.DataFrame, pairs: list, label: str) -> list:
    rows = []
    for problem in [None, *sorted(summary["problem"].unique())]:
        s = summary if problem is None else summary[summary["problem"] == problem]
        if problem is None and summary["problem"].nunique() == 1:
            continue
        for a, b in pairs:
            for metric, lower in METRICS:
                rows.append(
                    {
                        "host": label,
                        "scope": problem or "pooled",
                        **te.paired_effect(s, a, b, metric, lower),
                    }
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(jcde_paths.ANALYSIS / "e5_host"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    host = load({"p3_host": None, "p3_host_random_t": None})
    single = load({"p3_host_single": None, "p3_host_single_random_t": None})
    default = load(DEFAULT_HOST)
    # one reference pool per problem and ground truth across all hosts, so the aero regrets share a scale
    refs = te.references(host + single + default)
    s_host = te.summarise(host, refs)
    s_single = (
        te.summarise(single, refs) if single else pd.DataFrame(columns=s_host.columns)
    )
    s_def = te.summarise(default, refs)
    s_host["host"], s_single["host"], s_def["host"] = "nsga2", "single_nsga2", "default"
    pd.concat([s_host, s_single, s_def]).to_csv(out / "e5_runs.csv", index=False)
    bench = s_host[s_host["problem"] != "SAV-AERO"]
    aero = s_host[s_host["problem"] == "SAV-AERO"]
    aero_pairs = [("AdaIS", "VoI-Timing-MC"), ("AdaIS", "AdaS-RandomT")]
    rows = effects(
        bench,
        [("AdaIS", m) for m in ("AdaS-NoQ", "Baseline-F8", "Baseline-R")],
        "nsga2",
    )
    rows += effects(aero, aero_pairs, "nsga2")
    if not s_single.empty:
        rows += effects(s_single, aero_pairs, "single_nsga2")
    rows += effects(s_def, aero_pairs, "default")
    eff = pd.DataFrame(rows)
    eff.to_csv(out / "e5_paired_effects.csv", index=False)
    means = (
        pd.concat([s_host, s_single, s_def])
        .groupby(["host", "problem", "method"])[[m for m, _ in METRICS]]
        .mean()
    )
    means.round(4).to_csv(out / "e5_means.csv")
    pd.set_option("display.width", 220)
    keep = [
        "host",
        "scope",
        "a",
        "b",
        "metric",
        "n",
        "advantage",
        "ci_low",
        "ci_high",
        "ci_excludes_zero",
    ]
    print(
        eff[eff["scope"].isin(["pooled", "SAV-AERO"])][keep]
        .round(4)
        .to_string(index=False)
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
