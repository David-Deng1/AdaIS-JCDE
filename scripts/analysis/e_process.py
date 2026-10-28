"""Interaction-process statistics (R1-6, R2-7, R2-8, R4-7; submitted Fig. 15): when each method asks, through which
channel, whether the quality monitor pauses elicitation, and how much budget is left unused.

Read from each run's per-tick metrics.csv (triggered, channel_type, response_level, budget_remaining). Thirds follow
the submitted Fig. 15: ticks 0-26, 27-52, 53-79. A run "pauses" when response_level reaches "pause"; in P3 the
pause is permanent, so the budget left at that tick is never spent. Pause tick and budget left at pause are averaged
over paused runs only. Fixed-k and Random-C log the monitor state but do not act on it (their trigger counts are
the same in every matrix), so their "paused" share is a logged state, not a stop.

Usage: python e_process.py [--matrices ...] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth_eval as te  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jcde_paths  # noqa: E402
RUNS = jcde_paths.P3_RUNS
CHANNELS = ["threshold", "roi", "ranking", "pairwise", "nl_clarify"]
THIRDS = [(0, 26), (27, 52), (53, 79)]


def process_row(run_dir: Path) -> dict:
    m = pd.read_csv(run_dir / "metrics.csv")
    trig = m[m["triggered"] == 1]
    pause = m.index[m["response_level"].astype(str) == "pause"]
    row = {
        f"third{i + 1}": int(trig["tick"].between(a, b).sum())
        for i, (a, b) in enumerate(THIRDS)
    }
    row.update({f"ch_{c}": int((trig["channel_type"] == c).sum()) for c in CHANNELS})
    row.update({
        "n_triggers": len(trig),
        "last_trigger_tick": int(trig["tick"].max()) if len(trig) else -1,
        "paused": bool(len(pause)),
        "pause_tick": int(m.loc[pause[0], "tick"]) if len(pause) else float("nan"),  # paused runs only
        "budget_left_at_pause": int(m.loc[pause[0], "budget_remaining"]) if len(pause) else float("nan"),
        "budget_left_end": int(m["budget_remaining"].iloc[-1]),
        "exhausted_before_last_third": bool(m.loc[m["tick"] <= THIRDS[2][0] - 1, "budget_remaining"].min() <= 0),
    })  # fmt: skip
    voi = run_dir / "voi_decision_log.jsonl"
    if voi.is_file():  # R4-7: which VoI queries updated the posterior, and how far its effective sample size fell
        log = [json.loads(x) for x in voi.read_text(encoding="utf-8").splitlines() if x]
        q = [r for r in log if r.get("decision") == "query"]
        upd = [r for r in q if r.get("voi_posterior_updated")]
        row.update({"voi_queries": len(q), "voi_posterior_updates": len(upd),
                    "voi_ess_final": float(upd[-1]["posterior_ess_after"]) if upd else float(log[0]["posterior_ess"])})  # fmt: skip
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--matrices",
        nargs="+",
        default=["p3_core", "p3_core_random_t", "p3_edge", "p3_corruption", "p3_e9a"],
    )
    ap.add_argument("--out", default=str(jcde_paths.ANALYSIS / "process"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for matrix in args.matrices:
        for done in sorted((RUNS / matrix).rglob("_jcde_done.json")):
            run = te.load_run(done.parent, RUNS)
            if run is None:
                continue
            rows.append({"matrix": matrix, "variant": run.variant, "method": run.method, "fatigue": run.fatigue,
                         "budget": run.budget, "seed": run.seed, **process_row(done.parent)})  # fmt: skip
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "process_runs.csv", index=False)
    cols = ["n_triggers", "third1", "third2", "third3", *[f"ch_{c}" for c in CHANNELS], "last_trigger_tick",
            "paused", "pause_tick", "budget_left_at_pause", "budget_left_end", "exhausted_before_last_third",
            *[c for c in ("voi_queries", "voi_posterior_updates", "voi_ess_final") if c in runs]]  # fmt: skip
    g = runs.groupby(["matrix", "method"])
    means = g[cols].mean()
    means["n_runs"] = g.size()
    means["max_last_trigger_tick"] = g["last_trigger_tick"].max()
    means["runs_with_nl"] = g["ch_nl_clarify"].apply(lambda s: int((s > 0).sum()))
    means["runs_no_last_third"] = g["third3"].apply(lambda s: int((s == 0).sum()))
    means = means.round(3)
    means.to_csv(out / "process_means.csv")
    by_fat = runs[runs["matrix"] == "p3_corruption"].groupby(["method", "fatigue"])[["paused", "pause_tick", "budget_left_at_pause", "n_triggers"]].mean().round(3)  # fmt: skip
    by_fat.to_csv(out / "process_pause_by_fatigue.csv")
    pd.set_option("display.width", 250)
    print(means.to_string())
    print(by_fat.to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
