"""Run P3 matrices outside the main chain with the chain's own build / run / retry logic, e.g. a matrix added
after the running chain has loaded its stage list: python run_extra_p3.py p3_host_single p3_host_single_random_t

A Random-T matrix is built only once its source matrix is complete. Specs that already exist are kept.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_chain_p3 import OUT, build, complete, log, progress, run_stage  # noqa: E402


def main() -> None:
    for name in sys.argv[1:]:
        if name.endswith("_random_t"):
            source = name.removesuffix("_random_t")
            if not complete(source):
                log(
                    f"extra: {name} not built, source {source} incomplete {progress(source)}"
                )
                continue
            build(name)
        elif not (OUT / "specs" / f"{name}.jsonl").exists():
            build(name)
        run_stage(name)
    log(f"extra complete: { {n: progress(n) for n in sys.argv[1:]} }")


if __name__ == "__main__":
    main()
