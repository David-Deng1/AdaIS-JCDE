"""E6 Morris screening (R3-4b): which of the 15 set constants move AdaIS's outcomes on ZDT1 (known-utility DM)?

The design is regenerated with the builder's SALib call (same problem, N = 10, 4 levels, seed) and checked against
p3_morris_design.json; each design point's output is the mean over seeds 1 and 2. Reported per output: mu* (mean
absolute elementary effect, with its bootstrap CI), mu and sigma, ranked by mu*. A ranking statement, not a
performance claim. Independent unit: the Morris trajectory (seeds are averaged inside each design point), so the
seed-cluster convention of te.paired_effect does not apply here.

Usage: python e6_morris.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.sample import morris as morris_sample

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth_eval as te  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jcde_paths  # noqa: E402
RUNS = jcde_paths.P3_RUNS
DESIGN = RUNS / "specs" / "p3_morris_design.json"
OUTPUTS = ["regret_rec", "terminal_hv", "hv_auc", "C_cog", "N_int"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(jcde_paths.ANALYSIS / "e6_morris"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    problem = design["problem"]
    X = morris_sample.sample(problem, N=10, num_levels=4, seed=20261022)
    rec = (
        pd.DataFrame(design["points"])
        .set_index("point")[problem["names"]]
        .to_numpy(float)
    )
    ints = [
        i
        for i, n in enumerate(problem["names"])
        if n in ("T_stag", "B_reserve", "warmup_ticks")
    ]
    flt = [i for i in range(X.shape[1]) if i not in ints]
    assert np.allclose(np.round(X[:, flt], 5), rec[:, flt]), (
        "regenerated design differs from the recorded one"
    )
    runs = list(te.iter_runs(RUNS, ["p3_morris"]))
    summary = te.summarise(runs, te.references(runs))
    summary.to_csv(out / "morris_runs.csv", index=False)
    by_point = (
        summary.assign(point=summary["variant"].str[1:].astype(int))
        .groupby("point")[OUTPUTS]
        .mean()
    )
    missing = sorted(set(range(len(X))) - set(by_point.index))
    if missing:
        raise SystemExit(
            f"{len(missing)} design points without runs, e.g. {missing[:5]}"
        )
    rows = []
    for name in OUTPUTS:
        Y = by_point.loc[range(len(X)), name].to_numpy(float)
        res = morris_analyze.analyze(problem, X, Y, num_levels=4, num_resamples=2000, conf_level=0.95,
                                     seed=20261023)  # fmt: skip
        for i, p in enumerate(problem["names"]):
            rows.append({"output": name, "param": p, "mu_star": res["mu_star"][i], "mu_star_conf": res["mu_star_conf"][i],
                         "mu": res["mu"][i], "sigma": res["sigma"][i]})  # fmt: skip
    table = pd.DataFrame(rows)
    table["rank"] = table.groupby("output")["mu_star"].rank(ascending=False).astype(int)
    table.sort_values(["output", "rank"]).to_csv(
        out / "morris_indices.csv", index=False
    )
    pd.set_option("display.width", 200)
    for name in OUTPUTS:
        t = table[table["output"] == name].sort_values("rank").head(6)
        print(
            f"\n{name}\n"
            + t[["param", "mu_star", "mu_star_conf", "mu", "sigma"]]
            .round(4)
            .to_string(index=False)
        )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
