"""JCDE-2026-572 revision: resumable batch runner for regenerating every experiment matrix.

A matrix is a JSONL spec (one run per line) with fields:
    run_id   unique id within the matrix
    cmd      argv list (python executable first)
    run_dir  directory where the runner writes metrics.csv
    env      optional extra environment variables
    after    optional list of "matrix:run_id" that must finish first (informational)

Every run executes in a fresh temporary working directory (the engine writes an offline
dataset into the cwd, which must not leak between runs), with BLAS threads pinned to 1,
the minimal adais_jcde_code engine first on PYTHONPATH, tracing enabled and global RNGs seeded from --seed.
A run counts as done when run_dir/_jcde_done.json exists; reruns skip done runs.

Usage:
    python orchestrator.py run <spec.jsonl> [--workers N]
    python orchestrator.py status <spec.jsonl>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]  # adais_jcde_code (desensitized minimal engine)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jcde_paths  # noqa: E402
PYTHON = Path(jcde_paths.PYTHON)
ROOT = Path(__file__).resolve().parent
STATUS_DIR = Path(os.environ.get("JCDE_STATUS_DIR", str(jcde_paths.R1_RUNS / "status")))
TIMEOUT_S = 3600
STALE_CLAIM_S = TIMEOUT_S + 600
# Memory gate (16 GB laptop; one run peaks at ~640 MB): a run starts only while this much physical
# memory would stay available after it, and starts are spaced so each new process shows up first.
MIN_AVAIL_MB = int(os.environ.get("JCDE_MIN_AVAIL_MB", "1500"))
RUN_RESERVE_MB = 700
LAUNCH_GAP_S = 6.0
_launch_lock = threading.Lock()
_last_launch = [0.0]


def available_mb() -> float | None:
    if os.name != "nt":
        return None
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong)
            for n in ("total", "avail", "total_pf", "avail_pf", "total_v", "avail_v", "avail_xv")
        ]

    s = MemoryStatusEx()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.avail / 2**20


def wait_for_memory() -> None:
    with _launch_lock:
        while True:
            gap = LAUNCH_GAP_S - (time.time() - _last_launch[0])
            if gap > 0:
                time.sleep(gap)
            avail = available_mb()
            if avail is None or avail >= MIN_AVAIL_MB + RUN_RESERVE_MB:
                break
            time.sleep(10)
        _last_launch[0] = time.time()


def pid_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)  # signal 0 only probes on POSIX (on Windows os.kill would terminate)
        except OSError:
            return False
        return True
    import ctypes

    k32 = ctypes.windll.kernel32
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
    k32.CloseHandle(handle)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def engine_env(extra: dict | None = None, pythonpath_extra: list | None = None) -> dict:
    env = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(ENGINE / "jcde_trace"), str(ENGINE), *(pythonpath_extra or [])])
    env["JCDE_TRACE"] = "1"
    env["JCDE_SEED_GLOBALS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("JCDE_TRACE_DEBUG", None)
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def load_spec(path: Path) -> list[dict]:
    runs = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [r["run_id"] for r in runs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate run_id in {path}")
    return runs


def is_done(run: dict) -> bool:
    return (Path(run["run_dir"]) / "_jcde_done.json").is_file()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def claim(run: dict) -> bool:
    """Atomically claim a run so several orchestrator pools can share one spec without duplicates."""
    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "_jcde_claim"
    for _ in range(2):
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if is_done(run):
                return False
            try:
                owner = int(marker.read_text().split()[0])
            except (OSError, ValueError, IndexError):
                owner = None
            fresh = time.time() - marker.stat().st_mtime < STALE_CLAIM_S
            if fresh and (owner is None or owner == os.getpid() or pid_alive(owner)):
                return False
            marker.unlink(missing_ok=True)  # stale claim, or its pool was killed
            continue
        os.write(fd, f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}".encode())
        os.close(fd)
        return True
    return False


def execute(run: dict, matrix: str) -> dict:
    run_dir = Path(run["run_dir"])
    if is_done(run):
        return {"matrix": matrix, "run_id": run["run_id"], "ok": True, "skipped": True}
    wait_for_memory()
    if is_done(run) or not claim(run):
        return {"matrix": matrix, "run_id": run["run_id"], "ok": True, "skipped": True}
    cwd = Path(tempfile.mkdtemp(prefix="jcde_cwd_"))
    started = time.time()
    try:
        proc = subprocess.run(
            run["cmd"],
            cwd=cwd,
            env=engine_env(run.get("env"), run.get("pythonpath_extra")),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_S,
        )
        rc, err = proc.returncode, proc.stderr[-1500:]
    except subprocess.TimeoutExpired:
        rc, err = -9, f"timeout after {TIMEOUT_S}s"
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    elapsed = time.time() - started
    metrics = run_dir / "metrics.csv"
    ok = rc == 0 and metrics.is_file()
    if ok:  # the trace is written at interpreter exit; under memory pressure it can fail with rc 0
        try:
            import numpy as np

            with np.load(run_dir / "jcde_trace.npz", allow_pickle=True) as z:
                z["pop_F"]
            json.loads((run_dir / "jcde_trace.json").read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            ok, rc, err = False, rc or -7, f"trace unreadable: {exc!r}"
    if not ok:
        (run_dir / "_jcde_claim").unlink(missing_ok=True)  # allow a retry by any pool
    record = {
        "matrix": matrix,
        "run_id": run["run_id"],
        "rc": rc,
        "ok": ok,
        "elapsed_s": round(elapsed, 1),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stderr_tail": "" if ok else err,
    }
    if ok:
        files = {
            p.name: sha256(p)
            for p in sorted(run_dir.iterdir())
            if p.is_file() and not p.name.startswith("_")
        }
        (run_dir / "_jcde_done.json").write_text(
            json.dumps(
                {**record, "cmd": run["cmd"], "files": files},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
    return record


def cmd_run(spec: Path, workers: int) -> None:
    matrix = spec.stem
    runs = load_spec(spec)
    todo = [r for r in runs if not is_done(r)]
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    log = STATUS_DIR / f"{matrix}.jsonl"
    print(
        f"[{matrix}] {len(runs)} runs, {len(runs) - len(todo)} already done, {len(todo)} to run, workers={workers}",
        flush=True,
    )
    t0, done, failed = time.time(), 0, 0
    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        log.open("a", encoding="utf-8") as fh,
    ):
        futures = {pool.submit(execute, r, matrix): r for r in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            if rec.get("skipped"):
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done += 1
            failed += 0 if rec["ok"] else 1
            if done % 10 == 0 or not rec["ok"] or done == len(todo):
                rate = (time.time() - t0) / done
                eta = rate * (len(todo) - done) / 60
                print(
                    f"[{matrix}] {done}/{len(todo)} done, {failed} failed, ETA {eta:.0f} min"
                    + (
                        ""
                        if rec["ok"]
                        else f" | FAIL {rec['run_id']}: {rec['stderr_tail'][-200:]}"
                    ),
                    flush=True,
                )
    print(
        f"[{matrix}] finished: {done - failed} ok, {failed} failed, {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )


def cmd_status(spec: Path) -> None:
    runs = load_spec(spec)
    n_done = sum(is_done(r) for r in runs)
    print(f"{spec.stem}: {n_done}/{len(runs)} done")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "status"])
    parser.add_argument("spec", type=Path, nargs="+")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    for spec in args.spec:
        if args.action == "run":
            cmd_run(spec, args.workers)
        else:
            cmd_status(spec)


if __name__ == "__main__":
    sys.exit(main())
