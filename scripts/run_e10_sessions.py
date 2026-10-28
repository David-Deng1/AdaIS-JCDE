"""E10' out-of-model check: run optimizer sessions whose decision maker is an external LLM agent (file relay).

One persona per invocation. Sessions run one after another; each writes its questions to
<relay_root>/session_XX/ and waits for the agent's answers. When the phase's sessions finish, the file
<relay_root>/ALL_DONE is written so the agent can stop.

Two phases (route P3):
    A  Fixed-8 sessions (sessions 01-03). Fixed-8 has no quality monitor and no valuation, so it does not
       depend on the P3 calibration; its NL answers also give the grounding rate used to calibrate the
       simulated NL respondent.
    B  AdaIS (P3) sessions (sessions 04-06), run after calibration with the configuration frozen in
       E10_LLM决策者实验/adais_p3_config.json.
The agent never sees which policy a session uses; the map is written outside the relay folder.

Usage:
    python run_e10_sessions.py P1 --phase A [--relay_root DIR] [--out_root DIR] [--seeds 51 52 53]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orchestrator import engine_env  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jcde_paths  # noqa: E402
PY = jcde_paths.PYTHON
PREP = jcde_paths.E10_MATERIALS
FATIGUE, BUDGET = 0.3, 10
PHASES = {"A": ("Baseline-F8", 0), "B": ("AdaIS", 3)}  # policy, session-number offset


def adais_args() -> list:
    cfg_path = PREP / "adais_p3_config.json"
    if not cfg_path.exists():
        sys.exit(
            f"phase B needs the frozen P3 configuration in {cfg_path} (written after calibration)"
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return [str(a) for a in cfg["args"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("persona", choices=["P1", "P2", "P3"])
    ap.add_argument("--phase", choices=sorted(PHASES), required=True)
    ap.add_argument("--relay_root", default=None)
    ap.add_argument(
        "--out_root", default=str(jcde_paths.E10_RUNS)
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[51, 52, 53])
    args = ap.parse_args()
    persona = json.loads((PREP / "personas.json").read_text(encoding="utf-8"))[
        args.persona
    ]
    relay_root = Path(args.relay_root or PREP / "relay" / args.persona)
    relay_root.mkdir(parents=True, exist_ok=True)
    (relay_root / "ALL_DONE").unlink(missing_ok=True)
    policy, offset = PHASES[args.phase]
    extra = ["--p3", "--pref_model", "v2"] + (adais_args() if policy == "AdaIS" else [])
    out_persona = Path(args.out_root) / args.persona
    manifest_path = out_persona / "session_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else []
    )
    for k, seed in enumerate(args.seeds, offset + 1):
        session = relay_root / f"session_{k:02d}"
        out = out_persona / policy
        run_dir = out / f"{policy}_SAV-AERO_P-balanced_s{seed}"
        entry = {
            "session": session.name,
            "phase": args.phase,
            "policy": policy,
            "seed": seed,
            "output": str(run_dir),
        }
        manifest = [m for m in manifest if m["session"] != session.name] + [entry]
        if (run_dir / "_jcde_done.json").exists():
            print(
                f"[{args.persona}] session {session.name} already done ({policy}, seed {seed})",
                flush=True,
            )
            continue
        print(
            f"[{args.persona}] session {session.name}: waiting for agent answers in {session}",
            flush=True,
        )
        cmd = [
            PY, str(CODE / "runners" / "run_adais_jcde.py"),
            "--method", policy, "--problem", "SAV-AERO", "--seed", str(seed), "--T", "80",
            "--B_total", str(BUDGET), "--fatigue_rate", str(FATIGUE), "--sigma_noise", "0.1",
            "--output_dir", str(out), "--dm", "relay", "--relay_dir", str(session),
            "--sdm_profile_key", persona["profile_key"], *extra,
        ]  # fmt: skip
        cwd = Path(
            tempfile.mkdtemp(prefix="jcde_e10_")
        )  # the engine writes offline data into its cwd
        started = time.time()
        try:
            rc = subprocess.run(cmd, cwd=cwd, env=engine_env()).returncode
        finally:
            shutil.rmtree(cwd, ignore_errors=True)
        if rc:
            sys.exit(
                f"session {session.name} failed (rc={rc}); fix and rerun this script to resume"
            )
        (run_dir / "_jcde_done.json").write_text(
            json.dumps({"cmd": cmd, "elapsed_s": round(time.time() - started, 1),
                        "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )  # fmt: skip
        out_persona.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8"
        )
    out_persona.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (relay_root / "ALL_DONE").write_text(
        f"phase {args.phase} finished\n", encoding="utf-8"
    )
    print(
        f"[{args.persona}] phase {args.phase}: all {len(args.seeds)} sessions done",
        flush=True,
    )


if __name__ == "__main__":
    main()
