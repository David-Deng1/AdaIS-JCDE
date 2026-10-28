"""Run the route-P3 matrices one after another (unattended): calibration, selection, then held-out matrices.

Each stage is resumable: re-running this script skips finished runs and rebuilds specs that depend on
finished runs. Progress lines go to stdout and to jcde_p3_runs/status/chain.log.

A run stage that ends incomplete (killed runs, failures) is retried once. A Random-T spec is built only
from a complete source matrix (its K comes from every AdaIS run of that source); otherwise it is deferred
to the sweep, which builds and runs it once the source is complete.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orchestrator import is_done, load_spec  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jcde_paths  # noqa: E402
OUT = jcde_paths.P3_RUNS
PY = jcde_paths.PYTHON
LOG = OUT / "status" / "chain.log"
WORKERS = int(os.environ.get("JCDE_WORKERS", "8"))

STAGES = [
    ("build", "p3_calib"),
    ("run", "p3_calib"),
    ("select", None),
    ("build", "p3_core"),
    ("run", "p3_core"),
    ("build", "p3_core_random_t"),
    ("run", "p3_core_random_t"),
    ("build", "p3_edge"),
    ("run", "p3_edge"),
    ("build", "p3_edge_random_t"),
    ("run", "p3_edge_random_t"),
    ("build", "p3_ablation"),
    ("run", "p3_ablation"),
    ("build", "p3_corruption"),
    ("run", "p3_corruption"),
    ("build", "p3_e10_twin"),
    ("run", "p3_e10_twin"),
    ("build", "p3_e9a"),
    ("run", "p3_e9a"),
    ("build", "p3_fatigue"),
    ("run", "p3_fatigue"),
    ("build", "p3_uncert"),
    ("run", "p3_uncert"),
    ("build", "p3_scans"),
    ("run", "p3_scans"),
    ("build", "p3_host"),
    ("run", "p3_host"),
    ("build", "p3_host_random_t"),
    ("run", "p3_host_random_t"),
    ("build", "p3_host_single"),
    ("run", "p3_host_single"),
    ("build", "p3_host_single_random_t"),
    ("run", "p3_host_single_random_t"),
    ("build", "p3_morris"),
    ("run", "p3_morris"),
    ("build", "p3_benchmark"),
    ("run", "p3_benchmark"),
    ("sweep", None),
]


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def orchestrate(name: str) -> int:
    env = dict(os.environ, JCDE_STATUS_DIR=str(OUT / "status"))
    cmd = [PY, str(CODE / "scripts" / "orchestrator.py"), "run", str(OUT / "specs" / f"{name}.jsonl"),
           "--workers", str(WORKERS)]  # fmt: skip
    return subprocess.run(cmd, cwd=CODE, env=env).returncode


def progress(name: str) -> tuple[int, int] | None:
    spec = OUT / "specs" / f"{name}.jsonl"
    if not spec.exists():
        return None
    runs = load_spec(spec)
    return sum(is_done(r) for r in runs), len(runs)


def complete(name: str) -> bool:
    p = progress(name)
    return p is not None and p[0] == p[1]


def build(name: str) -> None:
    log(f"building spec {name}")
    subprocess.run(
        [PY, str(CODE / "scripts" / "build_specs_p3.py"), name], cwd=CODE, check=True
    )


def run_stage(name: str) -> None:
    for attempt in (1, 2):
        log(f"running {name}" + (" (retry of incomplete runs)" if attempt == 2 else ""))
        rc = orchestrate(name)
        done, total = progress(name)
        log(f"finished {name} (rc={rc}, {done}/{total} done)")
        if done == total:
            return


def main() -> None:
    start_at = sys.argv[1] if len(sys.argv) > 1 else None
    started = start_at is None
    deferred: list[str] = []
    for action, name in STAGES:
        if not started:
            started = name == start_at
            if not started:
                continue
        if action == "build":
            spec = OUT / "specs" / f"{name}.jsonl"
            if name.endswith("_random_t"):
                source = name.removesuffix("_random_t")
                if not complete(source):
                    log(
                        f"deferring {name}: source {source} incomplete {progress(source)}"
                    )
                    deferred.append(name)
                    continue
            elif spec.exists():
                continue  # keep the spec that earlier runs were launched from
            build(name)
        elif action == "select":
            if (OUT / "p3_config.json").exists():
                continue
            log("selecting the P3 configuration on the calibration seeds")
            subprocess.run(
                [PY, str(CODE / "scripts" / "analysis" / "p3_select_config.py")],
                cwd=CODE,
                check=True,
            )
        elif action == "sweep":
            for _, spec in [s for s in STAGES if s[0] == "run"]:
                if spec in deferred or progress(spec) is None:
                    continue
                log(f"sweep {spec}")
                orchestrate(spec)
            for name_rt in deferred:
                source = name_rt.removesuffix("_random_t")
                if complete(source):
                    build(name_rt)
                    run_stage(name_rt)
                else:
                    log(
                        f"still incomplete, {name_rt} not built: {source} {progress(source)}"
                    )
        elif name in deferred:
            continue
        else:
            run_stage(name)
    summary = {name: progress(name) for action, name in STAGES if action == "run"}
    log(f"chain complete: {summary}")


if __name__ == "__main__":
    main()
