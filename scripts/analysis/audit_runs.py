"""Integrity audit of the P3 run root before the results freeze (G3): is every analysed run the run that was specified?

Per matrix: every spec entry has a done marker with rc 0; the command recorded at completion equals the spec (so no
spec was rebuilt under a finished run) and the host and overrides the trace says took effect equal the spec's
environment (the done marker does not keep the environment); every run carries --p3 and --pref_model v2; the files hashed
at completion are unchanged (sha256); the trace loads and its final tick has finite truth metrics; no two runs share
a (variant, method, problem, fatigue, budget, seed) cell; and no done marker sits under a matrix folder without a
spec entry (iter_runs would silently pick such an orphan up).

Usage: python audit_runs.py [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth_eval as te  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jcde_paths  # noqa: E402
RUNS = jcde_paths.P3_RUNS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(p: str) -> str:
    return str(Path(p)).lower()


def audit_matrix(spec_path: Path) -> tuple[dict, list]:
    name = spec_path.stem
    specs = [
        json.loads(l)
        for l in spec_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    issues = []
    n_done = n_hash_ok = n_trace_ok = 0
    cells = Counter()
    spec_dirs = set()
    for s in specs:
        run_dir = Path(s["run_dir"])
        spec_dirs.add(norm(s["run_dir"]))
        done = run_dir / "_jcde_done.json"
        if not done.is_file():
            issues.append((name, s["run_id"], "no done marker"))
            continue
        meta = json.loads(done.read_text(encoding="utf-8"))
        if meta.get("rc") != 0 or not meta.get("ok"):
            issues.append(
                (name, s["run_id"], f"rc={meta.get('rc')} ok={meta.get('ok')}")
            )
        n_done += 1
        if meta.get("cmd") != s["cmd"]:
            issues.append((name, s["run_id"], "command differs from spec"))
        # the done marker does not keep the environment; the trace records the host and overrides that took effect
        env = s.get("env") or {}
        unknown = set(env) - {"JCDE_HOST", "JCDE_OVERRIDES"}
        if unknown:
            issues.append((name, s["run_id"], f"unchecked env keys {sorted(unknown)}"))
        try:
            tr = json.loads((run_dir / "jcde_trace.json").read_text(encoding="utf-8"))
            want_ov = (
                json.loads(env["JCDE_OVERRIDES"]) if "JCDE_OVERRIDES" in env else None
            )
            got_ov = tr.get("overrides")
            got_ov = json.loads(got_ov) if isinstance(got_ov, str) else got_ov  # the trace keeps the raw env string
            if tr.get("host") != env.get("JCDE_HOST") or got_ov != want_ov:
                issues.append((name, s["run_id"], f"trace host/overrides {tr.get('host')}/{tr.get('overrides')} "
                               f"!= spec {env.get('JCDE_HOST')}/{want_ov}"))  # fmt: skip
        except (OSError, ValueError) as exc:
            issues.append((name, s["run_id"], f"trace json unreadable: {exc}"))
        cmd = s["cmd"]
        if "--p3" not in cmd or te._arg(cmd, "--pref_model") != "v2":
            issues.append((name, s["run_id"], "missing --p3 or --pref_model v2"))
        bad = [f for f, h in (meta.get("files") or {}).items()
               if not (run_dir / f).is_file() or sha256(run_dir / f) != h]  # fmt: skip
        if not meta.get("files"):
            issues.append((name, s["run_id"], "no file hashes recorded"))
        elif bad:
            issues.append((name, s["run_id"], f"hash mismatch: {bad}"))
        else:
            n_hash_ok += 1
        try:
            run = te.load_run(run_dir, RUNS)
            last = run.ticks.iloc[-1]
            if (
                not np.isfinite(last["u_best"])
                or not np.isfinite(last["u_rec"])
                or not np.isfinite(last["hv"])
            ):
                issues.append(
                    (name, s["run_id"], "non-finite final u_best / u_rec / hv")
                )
            else:
                n_trace_ok += 1
            cells[
                (
                    run.variant,
                    run.method,
                    run.problem,
                    run.fatigue,
                    run.budget,
                    run.seed,
                )
            ] += 1
        except Exception as exc:  # noqa: BLE001 - every failure is a finding
            issues.append(
                (name, s["run_id"], f"trace load failed: {type(exc).__name__}: {exc}")
            )
    dup = [c for c, k in cells.items() if k > 1]
    for c in dup:
        issues.append((name, str(c), "duplicate cell"))
    top = name.split("/")[0]
    orphans = [
        d.parent
        for d in (RUNS / top).rglob("_jcde_done.json")
        if norm(str(d.parent)) not in spec_dirs
    ]
    for o in orphans:
        issues.append(
            (name, str(o.relative_to(RUNS)), "orphan done marker (not in spec)")
        )
    row = {"matrix": name, "specified": len(specs), "done": n_done, "hash_ok": n_hash_ok, "trace_ok": n_trace_ok,
           "duplicate_cells": len(dup), "orphans": len(orphans), "issues": sum(1 for i in issues if i[0] == name)}  # fmt: skip
    print(f"{name:<26} {row['done']:>5}/{row['specified']:<5} hash {n_hash_ok:>5} trace {n_trace_ok:>5} "
          f"dup {len(dup)} orphan {len(orphans)} issues {row['issues']}", flush=True)  # fmt: skip
    return row, issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(jcde_paths.ANALYSIS / "audit"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    rows, issues = [], []
    for spec in sorted((RUNS / "specs").glob("*.jsonl")):
        row, iss = audit_matrix(spec)
        rows.append(row)
        issues += iss
    table = pd.DataFrame(rows)
    table.loc[len(table)] = {
        "matrix": "TOTAL",
        **table.drop(columns="matrix").sum().to_dict(),
    }
    table.to_csv(out / "audit_matrices.csv", index=False)
    pd.DataFrame(issues, columns=["matrix", "run", "issue"]).to_csv(
        out / "audit_issues.csv", index=False
    )
    print(table.to_string(index=False))
    print(f"{len(issues)} issues -> {out}")


if __name__ == "__main__":
    main()
