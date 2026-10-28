"""Where the run data, the analysis outputs and the out-of-model materials live.

Every pipeline and analysis script takes its paths from here. The defaults follow the layout of the released data
archive unpacked next to this repository:

    <data root>/jcde_p3_runs/             simulation runs, specs, calibration files and the analysis/ folder
    <data root>/jcde_r1_runs/e10_llm_dm/  sessions answered by the LLM-based decision-maker agents
    <data root>/e10_materials/            persona briefs, operating prompt, relay transcripts, frozen AdaIS configuration

Environment variables override the defaults:

    ADAIS_DATA_ROOT     data root (default: the folder that contains this repository)
    ADAIS_ANALYSIS_DIR  where analysis outputs are written and read (default: <data root>/jcde_p3_runs/analysis)
    ADAIS_PYTHON        interpreter used to launch simulation runs (default: the running interpreter)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ADAIS_DATA_ROOT", str(REPO.parent)))
P3_RUNS = DATA / "jcde_p3_runs"
R1_RUNS = DATA / "jcde_r1_runs"
E10_RUNS = R1_RUNS / "e10_llm_dm"
E10_MATERIALS = DATA / "e10_materials"
ANALYSIS = Path(os.environ.get("ADAIS_ANALYSIS_DIR", str(P3_RUNS / "analysis")))
PYTHON = os.environ.get("ADAIS_PYTHON", sys.executable)
