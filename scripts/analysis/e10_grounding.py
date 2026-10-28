"""E10' NL grounding outcomes -> parameters of the P3 simulated NL respondent (writes nl_calibration.json).

Every nl_clarify answer from the LLM decision-maker agents is classified against the persona's ground truth:
* weight target (w_i): ungrounded / grounded with the correct direction / grounded with the wrong direction
  (direction = sign of w*_i - 1/n with a 0.05 dead band; an equal-importance truth is correct only at 0.5);
* reference target (r_i): ungrounded / grounded (numeric aspiration; its error is reported).
Rates are pooled over personas and sessions, with Wilson 95% intervals. Only finished sessions are used.

Usage: python e10_grounding.py [--phase A] [--out D:/.../jcde_p3_runs/nl_calibration.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jcde_paths  # noqa: E402
E10 = jcde_paths.E10_RUNS
PREP = jcde_paths.E10_MATERIALS


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)


def classify(rec: dict, w_star: list) -> dict:
    resp = rec["response"]
    slot = (resp.get("response") or {}).get("slot") or ""
    g = resp.get("grounding") or {}
    out = {"target": slot, "text": (resp.get("response") or {}).get("text"), "grounded": bool(g.get("grounded")),
           "g_slot": g.get("slot"), "g_value": g.get("value"), "rule": g.get("rule")}  # fmt: skip
    if slot.startswith("w_"):
        i = int(slot.split("_")[1])
        lean = w_star[i] - 1.0 / len(w_star)
        truth_level = 0.5 if abs(lean) <= 0.05 else (0.7 if lean > 0 else 0.3)
        if not out["grounded"]:
            out["outcome"] = "ungrounded"
        else:
            j = int(str(out["g_slot"]).split("_")[1]) if out["g_slot"] else i
            v = (
                out["g_value"] if j == i else 1.0 - out["g_value"]
            )  # two objectives: w_j fixes w_i
            got = 0.5 if abs(v - 0.5) < 1e-9 else (0.7 if v > 0.5 else 0.3)
            out["outcome"] = "correct" if got == truth_level else "wrong"
        out["truth_level"] = truth_level
    else:
        out["outcome"] = "grounded" if out["grounded"] else "ungrounded"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        default=None,
        help="restrict to one phase (A/B) via the session manifests",
    )
    ap.add_argument(
        "--out",
        default=str(jcde_paths.P3_RUNS / "nl_calibration.json"),
    )
    args = ap.parse_args()
    personas = json.loads((PREP / "personas.json").read_text(encoding="utf-8"))
    rows = []
    for pdir in sorted(E10.glob("P[123]")):
        manifest = json.loads((pdir / "session_manifest.json").read_text(encoding="utf-8")) \
            if (pdir / "session_manifest.json").exists() else []  # fmt: skip
        phase_of = {Path(m["output"]).name: m.get("phase") for m in manifest}
        for done in sorted(pdir.rglob("_jcde_done.json")):
            run = done.parent
            if args.phase and phase_of.get(run.name, "A") != args.phase:
                continue
            for line in (
                (run / "interaction_log.jsonl").read_text(encoding="utf-8").splitlines()
            ):
                rec = json.loads(line)
                if rec["channel"] == "nl_clarify":
                    rows.append({"persona": pdir.name, "run": run.name,
                                 **classify(rec, personas[pdir.name]["w_star"])})  # fmt: skip
    w = [r for r in rows if r["target"].startswith("w_")]
    r_ = [r for r in rows if r["target"].startswith("r_")]
    w_fail = sum(r["outcome"] == "ungrounded" for r in w)
    w_grounded_dir = [
        r for r in w if r["outcome"] != "ungrounded" and r["truth_level"] != 0.5
    ]
    w_wrong = sum(r["outcome"] == "wrong" for r in w_grounded_dir)
    r_fail = sum(r["outcome"] == "ungrounded" for r in r_)
    res = {
        "source": str(E10),
        "phase": args.phase or "all",
        "n_nl": len(rows),
        "weight": {"n": len(w), "ungrounded": w_fail, "p_fail_w": round(w_fail / len(w), 4) if w else None,
                   "p_fail_w_ci": wilson(w_fail, len(w)), "n_directional_grounded": len(w_grounded_dir),
                   "wrong": w_wrong,
                   "p_wrong_w": round(w_wrong / len(w_grounded_dir), 4) if w_grounded_dir else None,
                   "p_wrong_w_ci": wilson(w_wrong, len(w_grounded_dir)),
                   "equal_truth_misread": sum(r["outcome"] == "wrong" and r["truth_level"] == 0.5 for r in w)},
        "reference": {"n": len(r_), "ungrounded": r_fail, "p_fail_r": round(r_fail / len(r_), 4) if r_ else None,
                      "p_fail_r_ci": wilson(r_fail, len(r_))},
        "records": rows,
    }  # fmt: skip
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in res.items() if k != "records"},
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
