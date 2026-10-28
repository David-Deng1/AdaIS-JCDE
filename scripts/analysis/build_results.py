"""Build the JCDE-R1 result tables from every finished traced run.

Outputs (jcde_r1_runs/analysis/): refs.csv, summary.csv, paired_effects.csv, corruption_events.csv,
corruption_summary.csv. Safe to rerun at any time; partial matrices are analysed as far as they go.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth_eval as te  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jcde_paths  # noqa: E402
RUNS = Path(os.environ.get("JCDE_RUNS_ROOT", str(jcde_paths.R1_RUNS)))  # jcde_p3_runs for route P3
OUT = RUNS / "analysis"
PUBLIC = {
    "AdaIS": "AdaIS",
    "Baseline-F2": "Fixed-2",
    "Baseline-F": "Fixed-4",
    "Baseline-F8": "Fixed-8",
    "Baseline-F16": "Fixed-16",
    "Baseline-R": "Random-C",
    "AdaS-RandomT": "Random-T",
    "VoI-Timing-MC": "VoI-MC",
    "AdaS-FixedT8": "NoT",
    "AdaS-UniCh": "NoC",
    "AdaS-NoQ": "NoQ",
    "AdaS-NoQpre": "NoQ-pre",
    "AdaS-NoQpost": "NoQ-post",
    "AdaS-Full": "AdaS-Full",
}
METRICS = [
    ("terminal_hv", False),
    ("hv_auc", False),
    ("regret_rec", True),
    ("regret_best", True),
    ("regret_rec_auc", True),
    ("w_err_l1", True),
    ("N_int", True),
    ("C_cog", True),
    ("T_resp", True),
]

# (group label, matrix filter for AdaIS side, matrix filter for the comparator side, comparator public names)
COMPARISONS = [
    (
        "heldout_balanced",
        {"heldout_aero"},
        {"heldout_aero", "heldout_random_t", "heldout_voi_mc"},
        ["Random-T", "VoI-MC", "Fixed-2", "Fixed-4", "Fixed-8", "Fixed-16", "Random-C"],
    ),
    (
        "heldout_edge",
        {"a6_heldout_edge"},
        {"a6_heldout_edge", "a6_edge_random_t"},
        ["Random-T", "VoI-MC", "Fixed-2", "Fixed-4", "Fixed-8", "Fixed-16", "Random-C"],
    ),
    (
        "aero_ablation",
        {"a4_aero_ablation"},
        {"a4_aero_ablation"},
        ["NoT", "NoC", "NoQ"],
    ),
    (
        "benchmark",
        {"g1_benchmark"},
        {"g1_benchmark", "g2_bench_ablation"},
        ["NoQ", "NoT", "NoC", "Fixed-2", "Fixed-4", "Fixed-8", "Fixed-16", "Random-C"],
    ),
    (
        "host_nsga2",
        {"e5_host"},
        {"e5_host", "e5_host_random_t"},
        ["NoQ", "Fixed-8", "Random-C", "VoI-MC", "Random-T"],
    ),
    ("voi_allch", {"heldout_aero"}, {"e9a_voi_allch"}, ["VoI-MC"]),
]
TIMING = ["Random-T", "VoI-MC", "Fixed-2", "Fixed-4", "Fixed-8", "Fixed-16", "Random-C"]
COMPARISONS_P3 = [
    ("p3_core", {"p3_core"}, {"p3_core", "p3_core_random_t"}, TIMING),
    ("p3_edge", {"p3_edge"}, {"p3_edge", "p3_edge_random_t"}, TIMING),
    ("p3_ablation", {"p3_core"}, {"p3_ablation"}, ["NoT", "NoC", "NoQ"]),
    ("p3_corruption", {"p3_corruption"}, {"p3_corruption"}, ["NoT", "NoC", "NoQ", "NoQ-pre", "NoQ-post", "Fixed-8", "Random-C"]),
    ("p3_benchmark", {"p3_benchmark"}, {"p3_benchmark"}, ["NoQ", "NoT", "NoC", "Fixed-2", "Fixed-4", "Fixed-8", "Fixed-16", "Random-C"]),
    ("host_nsga2", {"p3_host"}, {"p3_host", "p3_host_random_t"}, ["NoQ", "Fixed-8", "Random-C", "VoI-MC", "Random-T"]),
]
if RUNS.name == "jcde_p3_runs":
    COMPARISONS = COMPARISONS_P3
CORRUPTION_MATRICES = ("a5_corruption", "e6_quality", "e6_fatigue", "p3_corruption", "p3_scans")


def load_all() -> list:
    matrices = sorted(
        p.name
        for p in RUNS.iterdir()
        if p.is_dir()
        and not p.name.startswith(("_", "jobs", "specs", "status", "analysis"))
    )
    return list(te.iter_runs(RUNS, matrices))


def corruption_events(run) -> pd.DataFrame:
    rows = []
    path = run.run_dir / "interaction_log.jsonl"
    if not path.is_file():
        return pd.DataFrame()
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        resp = rec.get("response") or {}
        flags = [
            r.get("corrupted") for r in resp.get("results", []) if isinstance(r, dict)
        ]
        flags += [resp.get("ranking_corrupted"), resp.get("corrupted")]
        flags = [bool(f) for f in flags if f is not None]
        rows.append(
            {
                "tick": rec.get("tick"),
                "channel": rec.get("channel"),
                "dm_quality": rec.get("dm_quality"),
                "beta": rec.get("feedback_update_weight", rec.get("update_weight")),
                "Q_t": rec.get("Q_t"),
                "corrupted": any(flags) if flags else np.nan,
            }
        )
    return pd.DataFrame(rows)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score_pos > score_neg) with ties counted half (Mann-Whitney)."""
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float(
        (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
        / (len(pos) * len(neg))
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_all()
    print(f"{len(runs)} traced runs loaded", flush=True)
    refs = te.references(runs)
    refs.to_csv(OUT / "refs.csv", index=False)
    summary = te.summarise(runs, refs)
    summary["method"] = summary["method"].map(lambda m: PUBLIC.get(m, m))
    summary.to_csv(OUT / "summary.csv", index=False)

    effects = []
    for group, a_mats, b_mats, comparators in COMPARISONS:
        a_side = summary[
            summary["matrix"].isin(a_mats)
            & (summary["method"] == "AdaIS")
            & (summary["variant"] == "")
        ]
        if group == "host_nsga2":
            a_side = summary[
                summary["matrix"].isin(a_mats) & (summary["method"] == "AdaIS")
            ]
        for comp in comparators:
            b_side = summary[
                summary["matrix"].isin(b_mats) & (summary["method"] == comp)
            ]
            if a_side.empty or b_side.empty:
                continue
            both = pd.concat([a_side, b_side])
            for metric, lower in METRICS:
                res = te.paired_effect(both, "AdaIS", comp, metric, lower)
                if res.get("n"):
                    effects.append({"group": group, **res})
    pd.DataFrame(effects).to_csv(OUT / "paired_effects.csv", index=False)

    ev_rows, cs_rows = [], []
    for run in runs:
        if run.matrix not in CORRUPTION_MATRICES:
            continue
        ev = corruption_events(run)
        if ev.empty:
            continue
        tag = dict(
            matrix=run.matrix,
            variant=run.variant,
            method=PUBLIC.get(run.method, run.method),
            fatigue=run.fatigue,
            budget=run.budget,
            seed=run.seed,
        )
        ev_rows.append(ev.assign(**tag))
        lab = ev["corrupted"].astype(float)
        beta = pd.to_numeric(ev["beta"], errors="coerce")
        cs_rows.append(
            {
                **tag,
                "n_events": len(ev),
                "low_quality_rate": float((ev["dm_quality"] < 0.5).mean()),
                "corrupted_rate": float(lab.mean()),
                "beta_mean_corrupted": float(beta[lab == 1].mean())
                if (lab == 1).any()
                else np.nan,
                "beta_mean_clean": float(beta[lab == 0].mean())
                if (lab == 0).any()
                else np.nan,
                "corrupted_mass": float((beta.fillna(1.0) * lab).sum()),
            }
        )
    if ev_rows:
        ev_all = pd.concat(ev_rows, ignore_index=True)
        ev_all.to_csv(OUT / "corruption_events.csv", index=False)
        cs = pd.DataFrame(cs_rows)
        aucs = []
        for (variant, method), g in ev_all.groupby(["variant", "method"]):
            ok = (
                g["corrupted"].notna()
                & pd.to_numeric(g["beta"], errors="coerce").notna()
            )
            if ok.sum() > 10:
                aucs.append(
                    {
                        "variant": variant,
                        "method": method,
                        "n": int(ok.sum()),
                        "auc_one_minus_beta": auc(
                            1 - pd.to_numeric(g.loc[ok, "beta"]).to_numpy(),
                            g.loc[ok, "corrupted"].astype(int).to_numpy(),
                        ),
                        "auc_one_minus_Qt": auc(
                            1 - pd.to_numeric(g.loc[ok, "Q_t"]).to_numpy(),
                            g.loc[ok, "corrupted"].astype(int).to_numpy(),
                        ),
                    }
                )
        cs.to_csv(OUT / "corruption_summary.csv", index=False)
        pd.DataFrame(aucs).to_csv(OUT / "corruption_detection_auc.csv", index=False)
    print("tables written to", OUT, flush=True)


if __name__ == "__main__":
    main()
