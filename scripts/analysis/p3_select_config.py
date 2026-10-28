"""Apply the pre-registered P3 calibration rule (see build_specs_p3.py) to the finished p3_calib runs.

Writes jcde_p3_runs/p3_config.json (used by every later P3 spec) and the E10' phase-B configuration.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_specs_p3 as b  # noqa: E402


def main() -> None:
    rows = []
    for line in (b.SPECS / "p3_calib.jsonl").read_text(encoding="utf-8").splitlines():
        spec = json.loads(line)
        run_dir = Path(spec["run_dir"])
        if not (run_dir / "_jcde_done.json").is_file():
            sys.exit(f"calibration incomplete: {run_dir}")
        m = pd.read_csv(run_dir / "metrics.csv")
        cand = spec["run_id"].split("|")[0].split("/")[1]
        rows.append({"candidate": cand, "terminal_hv": float(m["hv"].iloc[-1]),
                     "C_cog": float(m["cumulative_cognitive_cost"].iloc[-1]),
                     "N_int": int(m["interaction_count"].iloc[-1])})  # fmt: skip
    df = pd.DataFrame(rows)
    table = df.groupby("candidate").agg(terminal_hv=("terminal_hv", "mean"), C_cog=("C_cog", "mean"),
                                        N_int=("N_int", "mean"), n=("terminal_hv", "size")).reset_index()  # fmt: skip
    best = table["terminal_hv"].max()
    tied = table[table["terminal_hv"] >= 0.99 * best].sort_values(
        ["C_cog", "terminal_hv"], ascending=[True, False]
    )
    pick = tied.iloc[0]["candidate"]
    prefix, lb = pick.split("_lB")
    cfg = {
        "prefix": prefix,
        "lambda_B": float(lb),
        "selected": pick,
        "rule": "highest mean terminal HV on seeds 41-45; within 1% of best -> lowest mean C_cog",
        "tied_candidates": tied["candidate"].tolist(),
        "table": table.round(6).to_dict("records"),
        "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    b.CONFIG.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    b.E10_CONFIG.write_text(
        json.dumps({"args": b.selected(b.U2), "source": str(b.CONFIG)}, indent=1),
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    print(f"selected: {pick}  (tied: {cfg['tied_candidates']})")


if __name__ == "__main__":
    main()
