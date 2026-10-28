"""E10' analysis: LLM-based decision-maker agents vs the parametric SDM on the same personas and settings.

Agent sessions: jcde_r1_runs/e10_llm_dm/<persona>/<policy>/ (phase A = Baseline-F8, phase B = AdaIS-P3,
seeds 51-53). Simulated twin: jcde_p3_runs/p3_e10_twin/<persona>/ (same personas, settings and policies;
seeds 51-53 pair with the agent sessions, 54-60 widen the reference).

Per run: terminal HV, ground-truth regret of the recommended design and of the best design (truth_eval.py;
U* and U0 pooled over agent and twin runs of the same persona), preference-recovery error, interaction count
and burden, and the share of answers that agree with the persona's noise-free answer (interaction_log
"truth" field). Paired AdaIS - Fixed-8 effects are formed per persona and seed, for the agents and for the
twin, and the direction of the two is compared. With three seeds per persona the comparison is descriptive
(counts of sessions, no bootstrap CI; the unit is one persona x seed session).

Usage: python e10_analysis.py [--out DIR]
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
E10 = jcde_paths.E10_RUNS
P3 = jcde_paths.P3_RUNS
TWIN = P3 / "p3_e10_twin"
POLICY = {"Baseline-F8": "Fixed-8", "AdaIS": "AdaIS-P3"}
METRICS = [
    "terminal_hv",
    "regret_rec",
    "regret_best",
    "w_err_l1",
    "N_int",
    "C_cog",
    "answer_acc",
    "n_nl",
]


def answers(run_dir: Path) -> dict:
    recs = [
        json.loads(x)
        for x in (run_dir / "interaction_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x
    ]
    # NL is scored by e10_grounding.py (its truth label is a placeholder 1.0); ROI uses a strict IoU >= 0.5 label,
    # so it is reported as mean IoU instead
    correct = [r["truth"]["truth_correct"] for r in recs if r["channel"] not in ("nl_clarify", "roi")
               and isinstance(r.get("truth"), dict) and r["truth"].get("truth_correct") is not None]  # fmt: skip
    iou = [
        r["truth"]["truth_iou"]
        for r in recs
        if r["channel"] == "roi" and "truth_iou" in (r.get("truth") or {})
    ]
    return {
        "answer_acc": float(np.mean(correct)) if correct else np.nan,
        "roi_iou": float(np.mean(iou)) if iou else np.nan,
        "n_answers_scored": len(correct),
        "n_nl": sum(r["channel"] == "nl_clarify" for r in recs),
        "channels": " ".join(r["channel"] for r in recs),
    }


def channel_agreement(runs: list, meta: list) -> pd.DataFrame:
    """Per channel, the share of answers that agree with the persona's noise-free answer (R2-5); ROI as mean IoU,
    NL left out (scored by e10_grounding.py). Matched seeds 51-53 only, so agent and SDM answer the same sessions."""
    rows = []
    for run, m in zip(runs, meta):
        if not 51 <= run.seed <= 53:
            continue
        pop_F = np.load(run.run_dir / "jcde_trace.npz", allow_pickle=True)["pop_F"]
        for line in (
            (run.run_dir / "interaction_log.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ):
            r = json.loads(line) if line else {}
            truth = r.get("truth") if isinstance(r.get("truth"), dict) else {}
            if r.get("channel") == "roi" and "truth_iou" in truth:
                # the label's true box lies inside the current population; does the answered box reach it at all?
                F = np.asarray(pop_F[min(int(r["tick"]), len(pop_F) - 1)], float)
                b, (d0, d1) = r["response"]["bbox"], r["response"].get("dims", (0, 1))
                inside = (b[0] <= F[:, d0].max() and b[1] >= F[:, d0].min() and b[2] <= F[:, d1].max()
                          and b[3] >= F[:, d1].min())  # fmt: skip
                rows.append({"source": m["source"], "channel": "roi", "score": truth["truth_iou"],
                             "roi_overlaps_population": float(inside)})  # fmt: skip
            elif (
                r.get("channel") not in (None, "roi", "nl_clarify")
                and truth.get("truth_correct") is not None
            ):
                rows.append(
                    {
                        "source": m["source"],
                        "channel": r["channel"],
                        "score": float(truth["truth_correct"]),
                    }
                )
    t = pd.DataFrame(rows)
    g = t.groupby(["source", "channel"])
    return pd.DataFrame({"n_answers": g["score"].size(), "agreement_or_iou": g["score"].mean(),
                         "roi_overlaps_population": g["roi_overlaps_population"].mean()}).reset_index()  # fmt: skip


def collect() -> tuple[list, list]:
    runs, meta = [], []
    for persona in ("P1", "P2", "P3"):
        for done in sorted((E10 / persona).rglob("_jcde_done.json")):
            run = te.load_run(done.parent, E10)
            if run is not None:
                runs.append(run)
                meta.append(
                    {"persona": persona, "source": "agent", **answers(done.parent)}
                )
        for done in sorted((TWIN / persona).rglob("_jcde_done.json")):
            run = te.load_run(done.parent, P3)
            if run is not None:
                runs.append(run)
                meta.append(
                    {"persona": persona, "source": "sdm", **answers(done.parent)}
                )
    return runs, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(jcde_paths.ANALYSIS / "e10"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    runs, meta = collect()
    refs = te.references(runs)
    df = pd.concat(
        [te.summarise(runs, refs).reset_index(drop=True), pd.DataFrame(meta)], axis=1
    )
    df["policy"] = df["method"].map(POLICY)
    df["matched"] = df["seed"].between(51, 53)
    df.to_csv(out / "e10_runs.csv", index=False)
    refs.to_csv(out / "e10_references.csv", index=False)

    groups = []
    for (persona, source, policy), g in df.groupby(["persona", "source", "policy"]):
        for subset, sel in (
            ("seeds 51-53", g["matched"]),
            ("seeds 51-60", g["seed"] > 0),
        ):
            if source == "agent" and subset == "seeds 51-60":
                continue
            h = g[sel]
            groups.append({"persona": persona, "source": source, "policy": policy, "subset": subset, "n": len(h),
                           **{m: h[m].mean() for m in METRICS}})  # fmt: skip
    summary = pd.DataFrame(groups)
    summary.to_csv(out / "e10_summary.csv", index=False)

    paired = []
    m = df[df["matched"]]
    for (persona, source, seed), g in m.groupby(["persona", "source", "seed"]):
        a, f = g[g["method"] == "AdaIS"], g[g["method"] == "Baseline-F8"]
        if len(a) == 1 and len(f) == 1:
            paired.append({"persona": persona, "source": source, "seed": seed,
                           **{f"d_{k}": float(a[k].iloc[0] - f[k].iloc[0]) for k in METRICS}})  # fmt: skip
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "e10_paired.csv", index=False)

    agree = []
    if not paired.empty:
        wide = paired.pivot_table(index=["persona", "seed"], columns="source")
        for k in ("d_regret_rec", "d_terminal_hv", "d_C_cog", "d_N_int"):
            if (k, "agent") in wide and (k, "sdm") in wide:
                both = wide[[(k, "agent"), (k, "sdm")]].dropna()
                agree.append({"effect": k, "n_pairs": len(both),
                              "agent_mean": both[(k, "agent")].mean(), "sdm_mean": both[(k, "sdm")].mean(),
                              "same_sign": int((np.sign(both[(k, "agent")]) == np.sign(both[(k, "sdm")])).sum())})  # fmt: skip
    agree = pd.DataFrame(agree)
    agree.to_csv(out / "e10_direction_agreement.csv", index=False)
    by_channel = channel_agreement(runs, meta)
    by_channel.to_csv(out / "e10_channel_agreement.csv", index=False)

    pd.set_option("display.width", 200)
    print(
        f"runs: agent {int((df.source == 'agent').sum())}, sdm twin {int((df.source == 'sdm').sum())}"
    )
    print(summary.round(3).to_string(index=False))
    if not paired.empty:
        print(
            paired.groupby("source")[[f"d_{k}" for k in METRICS[:6]]]
            .mean()
            .round(3)
            .to_string()
        )
    if not agree.empty:
        print(agree.round(3).to_string(index=False))
    print(by_channel.round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
