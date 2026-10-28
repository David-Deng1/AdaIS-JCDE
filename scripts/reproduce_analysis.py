"""Regenerate every table, evidence file and figure of the paper from the run outputs, in dependency order.

Outputs go to jcde_paths.ANALYSIS (set ADAIS_ANALYSIS_DIR to write elsewhere). With --compare DIR, every non-image
output is then compared byte for byte with DIR, for example the analysis folder of the released data archive.

Usage:
    python reproduce_analysis.py                                   # write into <data root>/jcde_p3_runs/analysis
    ADAIS_ANALYSIS_DIR=/tmp/a python reproduce_analysis.py --compare <data root>/jcde_p3_runs/analysis
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jcde_paths  # noqa: E402

AN = Path(__file__).resolve().parent / "analysis"
A = jcde_paths.ANALYSIS
STEPS = [
    ["audit_runs.py"],  # integrity of all runs against their specs
    ["g2_direction.py"],  # held-out comparison (p3_core + count-matched Random-T)
    [
        "g2_direction.py",
        "--matrices",
        "p3_edge",
        "p3_edge_random_t",
        "--out",
        str(A / "e2_edge"),
    ],
    [
        "g2_direction.py",
        "--matrices",
        "p3_core",
        "p3_ablation",
        "--out",
        str(A / "e_ablation"),
    ],
    ["e7_corruption.py"],
    ["e9a_voi.py"],
    ["e6_fatigue.py"],
    ["e6_uncert.py"],
    ["e6_scans.py"],
    ["e6_morris.py"],
    ["e5_host.py"],
    ["bench_eval.py"],
    ["e_process.py"],
    ["seed_cluster_check.py"],
    ["e10_analysis.py"],
    [
        "e10_grounding.py",
        "--phase",
        "A",
        "--out",
        str(A / "e10" / "nl_grounding_v2_phaseA_check.json"),
    ],
    ["evidence_map.py"],
    ["evidence_claims.py"],
    ["fig_p3.py"],
    ["fig_paper.py"],
    ["e10_paper.py"],
]
IMAGES = {".png", ".pdf", ".svg"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def outputs(root: Path) -> list[Path]:
    """Non-image files below root, skipping hidden folders such as tool caches."""
    return sorted(
        p.relative_to(root)
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() not in IMAGES
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )


def compare(new: Path, ref: Path) -> int:
    files = outputs(ref)
    bad = [f for f in files if not (new / f).is_file() or sha(new / f) != sha(ref / f)]
    extra = [f for f in outputs(new) if not (ref / f).exists()]
    print(
        f"compared {len(files)} non-image files: {len(files) - len(bad)} identical, {len(bad)} different or missing"
    )
    for f in bad:
        print(f"  DIFF {f}")
    for f in extra:
        print(f"  NEW  {f}")
    return len(bad)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--compare",
        default=None,
        help="reference analysis folder for a byte-for-byte check",
    )
    args = ap.parse_args()
    A.mkdir(parents=True, exist_ok=True)
    for step in STEPS:
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {' '.join(step)}", flush=True)
        res = subprocess.run([sys.executable, str(AN / step[0]), *step[1:]], capture_output=True, text=True,
                             encoding="utf-8", errors="replace")  # fmt: skip
        if res.returncode:
            print(res.stdout[-2000:], res.stderr[-4000:], sep="\n")
            sys.exit(f"step failed: {step[0]}")
        print(f"    done in {time.time() - t0:.0f} s", flush=True)
    if args.compare:
        sys.exit(1 if compare(A, Path(args.compare)) else 0)


if __name__ == "__main__":
    main()
