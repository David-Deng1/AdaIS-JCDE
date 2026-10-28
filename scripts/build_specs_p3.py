"""Orchestrator specs for route P3 (JCDE-R1): every arm runs on the single P3 runner with --p3.

Inputs (written by earlier stages, never edited by hand):
    jcde_p3_runs/nl_calibration.json   E10' phase-A grounding rates -> simulated NL respondent
    jcde_p3_runs/p3_config.json        configuration selected on the calibration seeds (after p3_calib)

Calibration protocol (pre-registered 2026-10-21, before any P3 held-out run):
    candidates   initial-coverage prefix {none, threshold-ranking-threshold} x lambda_B {1, 2, 4}
    blocks       SAV-AERO, P-balanced, fatigue {0, 0.3, 0.7} x budget {5, 10, 20} x seeds 41-45
    criterion    highest mean terminal HV; candidates within 1 % of the best are tied and the tie goes to
                 the lowest mean interaction burden (C_cog). All other parameters are the frozen U2 values.
Held-out evaluation uses seeds 51-60 (aero) and 11-20 (benchmarks), each run once.
2026-10-23, before any P3 benchmark or host run: the benchmark held-out seeds moved from 1-10 to 11-20 (host
benchmark arm 1-5 -> 11-15), because the carried-over benchmark gates (BENCH_GATES) were selected on seeds 1-10
in the submitted study. Morris screening keeps seeds 1 and 2 (sensitivity ranking, not a performance claim).

Preference model v2 (2026-10-22, adopted after the E10' agents disagreed with the SDM on 93 % of
pairwise answers): every run passes --pref_model v2 (runners/pref_model_v2.py: scaled ASF in host and SDM,
monotone SDM utility, SAV-AERO aspirations on the attainable front). The calibration protocol above is
unchanged and is rerun under v2. The NL respondent keeps the phase-A grounding rates frozen on 10-21
(nl_calibration.json); the v2 E10' sessions are reported as a check, not used to recalibrate.
The v1 runs are archived in jcde_p3_runs_prefv1/.

Usage: python build_specs_p3.py <matrix> [...]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
RUNNER = CODE / "runners" / "run_adais_jcde.py"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jcde_paths  # noqa: E402
PY = jcde_paths.PYTHON
OUT = jcde_paths.P3_RUNS
SPECS = OUT / "specs"
NL_CAL = OUT / "nl_calibration.json"
CONFIG = OUT / "p3_config.json"
E10_CONFIG = jcde_paths.E10_MATERIALS / "adais_p3_config.json"

FATIGUE, BUDGETS = (0.0, 0.3, 0.7), (5, 10, 20)
HELDOUT, CALIB, BENCH_SEEDS = range(51, 61), range(41, 46), range(11, 21)
BENCH = ("ZDT1", "ZDT2", "ZDT3", "BNH", "TNK")
CORR_FATIGUE = (0.7, 1.0, 1.3)
PREFIXES = {"none": None, "TRT": "threshold,ranking,threshold"}
LAMBDA_B = (1.0, 2.0, 4.0)
# Frozen U2 gates (aero) and the benchmark gates; the submitted prefixes are replaced by the P3 prefix.
U2 = ["--tau_h", "0.35", "--tau_d", "0.25", "--T_stag", "4", "--lambda_cost", "0.0", "--B_reserve", "2",
      "--T_max", "2.5", "--kappa", "3.0", "--warmup_ticks", "6", "--tau_Q_high", "0.5", "--tau_Q_low", "0.1"]  # fmt: skip
BENCH_GATES = ["--tau_h", "0.35", "--tau_d", "0.3", "--T_stag", "5", "--lambda_cost", "0.0", "--B_reserve", "1",
               "--T_max", "2.0", "--kappa", "3.0", "--warmup_ticks", "8", "--tau_Q_high", "0.5", "--tau_Q_low", "0.1"]  # fmt: skip
ADAPTIVE = {"AdaIS", "AdaS-FixedT8", "AdaS-UniCh", "AdaS-NoQ", "AdaS-NoQpre", "AdaS-NoQpost", "AdaS-RandomT",
            "VoI-Timing-MC"}  # fmt: skip
BASELINES = ("Baseline-F2", "Baseline-F", "Baseline-F8", "Baseline-F16", "Baseline-R")


def nl_args() -> list:
    cal = json.loads(NL_CAL.read_text(encoding="utf-8"))
    return ["--p3_nl_model", "pipeline",
            "--p3_nl_fail_w", str(cal["weight"]["p_fail_w"]),
            "--p3_nl_wrong_w", str(cal["weight"]["p_wrong_w"]),
            "--p3_nl_fail_r", str(cal["reference"]["p_fail_r"])]  # fmt: skip


def scheduler_args(gates: list, prefix: str | None, lambda_b: float) -> list:
    return gates + (["--bootstrap_channels", PREFIXES[prefix]] if PREFIXES[prefix] else []) + [
        "--p3_lambda_B", str(lambda_b)]  # fmt: skip


def selected(gates: list) -> list:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    return scheduler_args(gates, cfg["prefix"], cfg["lambda_B"])


def run(
    tag, method, problem, profile, seed, budget, fatigue, extra, sigma=0.1, env=None
) -> dict:
    parent = OUT / tag / problem / f"fat{fatigue:.1f}_B{budget}"
    cmd = [PY, str(RUNNER), "--method", method, "--problem", problem, "--profile", profile, "--seed", str(seed),
           "--T", "80", "--B_total", str(budget), "--fatigue_rate", str(fatigue), "--sigma_noise", str(sigma),
           "--output_dir", str(parent), "--p3", "--pref_model", "v2", *nl_args(), *extra]  # fmt: skip
    r = {"run_id": f"{tag}|{method}|{problem}|{profile}|fat{fatigue:.1f}|B{budget}|s{seed}", "cmd": cmd,
         "run_dir": str(parent / f"{method}_{problem}_{profile}_s{seed}")}  # fmt: skip
    if env:
        r["env"] = env
    return r


def grid(tag, methods, problems=("SAV-AERO",), profile="P-balanced", seeds=HELDOUT, fatigues=FATIGUE,
         budgets=BUDGETS, sigma=0.1, env=None) -> list:  # fmt: skip
    return [run(tag, m, pb, profile, s, b, f, extra, sigma, env)
            for pb in problems for f in fatigues for b in budgets for m, extra in methods for s in seeds]  # fmt: skip


def write(name: str, runs: list) -> Path:
    SPECS.mkdir(parents=True, exist_ok=True)
    ids = [r["run_id"] for r in runs]
    assert len(ids) == len(set(ids)), f"duplicate run_id in {name}"
    path = SPECS / f"{name}.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in runs),
        encoding="utf-8",
    )
    print(f"{name}: {len(runs)} runs -> {path}", flush=True)
    return path


# ------------------------------------------------------------------ matrices
def p3_calib() -> Path:
    runs = []
    for prefix in PREFIXES:
        for lb in LAMBDA_B:
            cand = f"{prefix}_lB{lb:g}"
            runs += grid(
                f"p3_calib/{cand}",
                [("AdaIS", scheduler_args(U2, prefix, lb))],
                seeds=CALIB,
            )
    return write("p3_calib", runs)


def _arms(gates):
    adais = selected(gates)
    return [("AdaIS", adais), ("VoI-Timing-MC", adais)] + [(m, []) for m in BASELINES]


def p3_core() -> Path:
    return write("p3_core", grid("p3_core", _arms(U2)))


def p3_edge() -> Path:
    return write("p3_edge", grid("p3_edge", _arms(U2), profile="P-edge"))


def p3_ablation() -> Path:
    a = selected(U2)
    return write(
        "p3_ablation",
        grid(
            "p3_ablation", [(m, a) for m in ("AdaS-FixedT8", "AdaS-UniCh", "AdaS-NoQ")]
        ),
    )


def p3_corruption() -> Path:
    gates = list(U2)
    gates[gates.index("--tau_Q_high") + 1], gates[gates.index("--tau_Q_low") + 1] = (
        "0.55",
        "0.25",
    )
    a = selected(gates) + ["--corrupt_feedback"]
    methods = [
        (m, a)
        for m in (
            "AdaIS",
            "AdaS-FixedT8",
            "AdaS-UniCh",
            "AdaS-NoQ",
            "AdaS-NoQpre",
            "AdaS-NoQpost",
        )
    ]
    methods += [(m, ["--corrupt_feedback"]) for m in ("Baseline-F8", "Baseline-R")]
    return write(
        "p3_corruption",
        grid("p3_corruption", methods, fatigues=CORR_FATIGUE, sigma=0.35),
    )


def p3_benchmark() -> Path:
    a = selected(BENCH_GATES)
    methods = [(m, a) for m in ("AdaIS", "AdaS-NoQ", "AdaS-FixedT8", "AdaS-UniCh")] + [
        (m, []) for m in BASELINES
    ]
    return write(
        "p3_benchmark", grid("p3_benchmark", methods, problems=BENCH, seeds=BENCH_SEEDS)
    )


def p3_host() -> Path:
    host = {"JCDE_HOST": "nsga2"}
    b = selected(BENCH_GATES)
    runs = grid("p3_host/bench", [("AdaIS", b), ("AdaS-NoQ", b), ("Baseline-F8", []), ("Baseline-R", [])],
                problems=BENCH, seeds=range(11, 16), env=host)  # fmt: skip
    runs += grid(
        "p3_host/aero",
        [("AdaIS", selected(U2)), ("VoI-Timing-MC", selected(U2))],
        env=host,
    )
    return write("p3_host", runs)


# 2026-10-23, before any aero host run: JCDE_HOST=nsga2 is the A1 path, which optimises with a neutral UPV, so
# its search ignores the elicited preferences and only the recommendation can differ between methods. A second
# single host keeps the default host's preference coupling (objective shaping with the UPV ASF, ROI-biased
# reference directions) and drops only the solver mixture (Abl-3, one NSGA-II generator); it is added on the
# aero case. (The A3 "UPV direct" NSGA-III path was tried first and dropped: its ROI bias reads only w, which
# the runs barely move, so AdaIS and VoI-MC gave identical populations in a 30-tick smoke test.)
def p3_host_single() -> Path:
    a = selected(U2)
    return write("p3_host_single", grid("p3_host_single/aero", [("AdaIS", a), ("VoI-Timing-MC", a)],
                                        env={"JCDE_HOST": "single_nsga2"}))  # fmt: skip


def p3_scans() -> Path:
    """NL respondent (exact vs pipeline), NL burden kappa, quality-score weights and tau_Q on the P3 design."""
    a = selected(U2)
    runs = []
    for r in grid(
        "p3_scans/nl_exact",
        [("AdaIS", a), ("Baseline-F8", [])],
        seeds=range(51, 56),
        budgets=(10, 20),
    ):
        r["cmd"] = [c for c in r["cmd"]]
        i = r["cmd"].index("--p3_nl_model")
        r["cmd"][i + 1] = "exact"
        runs.append(r)
    for kappa in (0.40, 0.25, 0.15, 0.08):
        env = {"JCDE_OVERRIDES": json.dumps({"cost": {"nl_clarify": kappa}})}
        runs += grid(
            f"p3_scans/kappa_nl{kappa:.2f}",
            [("AdaIS", a)],
            seeds=range(51, 56),
            budgets=(10, 20),
            env=env,
        )
    gates = list(U2)
    gates[gates.index("--tau_Q_high") + 1], gates[gates.index("--tau_Q_low") + 1] = (
        "0.55",
        "0.25",
    )
    corr = selected(gates) + ["--corrupt_feedback"]
    for a1 in (0.4, 0.8, 1.2):
        for a3 in (0.0, 0.8):
            if (a1, a3) == (0.8, 0.8):
                continue
            env = {
                "JCDE_OVERRIDES": json.dumps({"sched": {"alpha1": a1, "alpha3": a3}})
            }
            runs += grid(f"p3_scans/quality_a1{a1}_a3{a3}", [("AdaIS", corr)], fatigues=CORR_FATIGUE, budgets=(10,),
                         sigma=0.35, env=env)  # fmt: skip
    for hi, lo in ((0.45, 0.15), (0.65, 0.35)):
        g = list(gates)
        g[g.index("--tau_Q_high") + 1], g[g.index("--tau_Q_low") + 1] = str(hi), str(lo)
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        args = scheduler_args(g, cfg["prefix"], cfg["lambda_B"]) + [
            "--corrupt_feedback"
        ]
        runs += grid(
            f"p3_scans/tauQ{hi}-{lo}",
            [("AdaIS", args)],
            fatigues=CORR_FATIGUE,
            budgets=(10,),
            sigma=0.35,
        )
    return write("p3_scans", runs)


def p3_e10_twin() -> Path:
    """Simulated counterpart of E10': the same three personas (ground truth via --sdm_profile_key), the same
    session settings (fatigue 0.3, B 10) and both E10' policies, answered by the parametric SDM with the
    calibrated NL model. Seeds 51-53 match the agent sessions block by block; 54-60 widen the reference."""
    personas = json.loads(
        (E10_CONFIG.parent / "personas.json").read_text(encoding="utf-8")
    )
    runs = []
    for name in ("P1", "P2", "P3"):
        key = ["--sdm_profile_key", personas[name]["profile_key"]]
        runs += grid(f"p3_e10_twin/{name}", [("Baseline-F8", key), ("AdaIS", selected(U2) + key)],
                     fatigues=(0.3,), budgets=(10,))  # fmt: skip
    return write("p3_e10_twin", runs)


def p3_e9a() -> Path:
    """E9a (R4-7): VoI-Timing-MC whose timing posterior also absorbs ranking/threshold/ROI/NL answers."""
    return write("p3_e9a", grid("p3_e9a", [("VoI-Timing-MC-AllCh", selected(U2))]))


def _corruption_args() -> list:
    gates = list(U2)
    gates[gates.index("--tau_Q_high") + 1], gates[gates.index("--tau_Q_low") + 1] = (
        "0.55",
        "0.25",
    )
    return selected(gates) + ["--corrupt_feedback"]


def p3_fatigue() -> Path:
    """E6 (R3-4e): two alternative fatigue forms on the common-corruption design (linear = p3_corruption)."""
    a = _corruption_args()
    runs = []
    for form in ("saturating", "step"):
        env = {"JCDE_OVERRIDES": json.dumps({"fatigue_form": form})}
        runs += grid(f"p3_fatigue/{form}", [("AdaIS", a), ("AdaS-NoQ", a), ("Baseline-R", ["--corrupt_feedback"])],
                     fatigues=CORR_FATIGUE, budgets=(10,), sigma=0.35, env=env)  # fmt: skip
    return write("p3_fatigue", runs)


# Uncertainty propagation (E6', R3-4c): behavioural parameters of the parametric decision maker are set values,
# so they are sampled jointly (Latin hypercube, seed fixed) and AdaIS, VoI-Timing-MC and Fixed-8 are run on the
# same sample and seed. Reported: probability that AdaIS beats VoI-Timing-MC on regret, burden and HV.
UNCERT_N = 60
COST0 = {
    "pairwise": 0.10,
    "threshold": 0.15,
    "roi": 0.20,
    "ranking": 0.30,
    "nl_clarify": 0.60,
}
LAT0 = {
    "pairwise": 10.0,
    "ranking": 30.0,
    "threshold": 15.0,
    "roi": 20.0,
    "nl_clarify": 60.0,
}


def p3_uncert() -> Path:
    from scipy.stats import qmc

    names = [f"cost_{c}" for c in COST0] + [
        "latency_x",
        "BETA_LAT",
        "GAMMA_QUAL",
        "sigma",
        "fatigue",
    ]
    lo = np.array([0.6] * 5 + [0.6, 1.0, 0.3, 0.05, 0.0])
    hi = np.array([1.4] * 5 + [1.4, 3.0, 0.9, 0.20, 1.0])
    X = qmc.scale(
        qmc.LatinHypercube(d=len(names), seed=20261022).random(UNCERT_N), lo, hi
    )
    a = selected(U2)
    runs, design = [], []
    for k, x in enumerate(X):
        p = dict(zip(names, (float(v) for v in x)))
        ov = {"cost": {c: round(COST0[c] * p[f"cost_{c}"], 4) for c in COST0},
              "sdm_latency": {c: round(LAT0[c] * p["latency_x"], 3) for c in LAT0},
              "sdm_class": {"BETA_LAT": round(p["BETA_LAT"], 4), "GAMMA_QUAL": round(p["GAMMA_QUAL"], 4)}}  # fmt: skip
        env = {"JCDE_OVERRIDES": json.dumps(ov)}
        seed = 51 + k % 10
        for m, extra in (("AdaIS", a), ("VoI-Timing-MC", a), ("Baseline-F8", [])):
            runs.append(run(f"p3_uncert/s{k:03d}", m, "SAV-AERO", "P-balanced", seed, 10, round(p["fatigue"], 4),
                            extra, sigma=round(p["sigma"], 4), env=env))  # fmt: skip
        design.append({"sample": k, "seed": seed, **p})
    (SPECS / "p3_uncert_design.json").write_text(
        json.dumps(design, indent=1), encoding="utf-8"
    )
    return write("p3_uncert", runs)


# Morris screening (E6, R3-4b) on ZDT1 with the known-utility decision maker: 15 scheduler / valuation /
# quality / decision-maker parameters, r = 10 trajectories, 4 levels (SALib, seed fixed), seeds 1 and 2.
MORRIS = [  # name, low, high, kind (cli / cost / sched)
    ("tau_h", 0.25, 0.45, "cli"), ("tau_d", 0.20, 0.40, "cli"), ("T_stag", 3, 7, "cli_int"),
    ("B_reserve", 0, 2, "cli_int"), ("T_max", 1.5, 3.0, "cli"), ("warmup_ticks", 4, 12, "cli_int"),
    ("tau_Q_high", 0.40, 0.60, "cli"), ("tau_Q_low", 0.05, 0.25, "cli"), ("p3_lambda_B", 0.5, 4.0, "cli"),
    ("p3_lambda_min", 0.0, 0.5, "cli"), ("p3_omega", 0.0, 0.3, "cli"), ("sigma_noise", 0.05, 0.20, "cli"),
    ("nl_clarify", 0.08, 0.60, "cost"), ("alpha1", 0.4, 1.2, "sched"), ("alpha3", 0.0, 1.2, "sched"),
]  # fmt: skip


def p3_morris() -> Path:
    from SALib.sample import morris as morris_sample

    problem = {
        "num_vars": len(MORRIS),
        "names": [m[0] for m in MORRIS],
        "bounds": [[m[1], m[2]] for m in MORRIS],
    }
    X = morris_sample.sample(problem, N=10, num_levels=4, seed=20261022)
    base = selected(BENCH_GATES)
    runs, design = [], []
    for k, x in enumerate(X):
        cli, cost, sched = list(base), {}, {}
        vals = {}
        for (name, _, _, kind), v in zip(MORRIS, x):
            v = int(round(v)) if kind == "cli_int" else round(float(v), 5)
            vals[name] = v
            if kind.startswith("cli") and name != "sigma_noise":
                flag = f"--{name}"
                if flag in cli:
                    cli[cli.index(flag) + 1] = str(v)
                else:
                    cli += [flag, str(v)]
            elif kind == "cost":
                cost[name] = v
            elif kind == "sched":
                sched[name] = v
        env = {"JCDE_OVERRIDES": json.dumps({"cost": cost, "sched": sched})}
        for seed in (1, 2):
            runs.append(run(f"p3_morris/x{k:03d}", "AdaIS", "ZDT1", "P-balanced", seed, 10, 0.3, cli,
                            sigma=vals["sigma_noise"], env=env))  # fmt: skip
        design.append({"point": k, **vals})
    (SPECS / "p3_morris_design.json").write_text(json.dumps({"problem": problem, "points": design}, indent=1),
                                                 encoding="utf-8")  # fmt: skip
    return write("p3_morris", runs)


def random_t_from(source: str, tag: str, warmup: int, env: dict | None = None) -> Path:
    """Count-matched Random-T: per block, K = realized interaction count of the P3 AdaIS run; K ticks are drawn
    uniformly without replacement from [warmup, 80) by an RNG keyed on SHA-256 of the block identity."""
    import pandas as pd

    runs = []
    for line in (SPECS / f"{source}.jsonl").read_text(encoding="utf-8").splitlines():
        spec = json.loads(line)
        cmd = spec["cmd"]
        if (
            cmd[cmd.index("--method") + 1] != "AdaIS"
            or cmd[cmd.index("--problem") + 1] != "SAV-AERO"
        ):
            continue
        frame = pd.read_csv(Path(spec["run_dir"]) / "metrics.csv")
        K = int(frame["interaction_count"].iloc[-1])
        key = spec["run_id"].replace("|AdaIS|", "|AdaS-RandomT|")
        seed = int.from_bytes(
            hashlib.sha256(f"Random-T-P3|{key}".encode()).digest()[:8], "little"
        )
        ticks = sorted(
            int(t)
            for t in np.random.default_rng(seed).choice(
                np.arange(warmup, 80), K, replace=False
            )
        )
        parent = Path(cmd[cmd.index("--output_dir") + 1].replace(source, tag, 1))
        sched = (
            parent
            / "schedules"
            / f"{Path(spec['run_dir']).name.replace('AdaIS_', 'AdaS-RandomT_')}.json"
        )
        sched.parent.mkdir(parents=True, exist_ok=True)
        sched.write_text(json.dumps({"scenario_key": key, "planned_ticks": ticks, "planned_K": K,
                                     "source_metrics": str(Path(spec["run_dir"]) / "metrics.csv")}), encoding="utf-8")  # fmt: skip
        new = list(cmd)
        new[new.index("--method") + 1] = "AdaS-RandomT"
        new[new.index("--output_dir") + 1] = str(parent)
        new += ["--timing_schedule_path", str(sched), "--timing_scenario_key", key]
        r = {"run_id": key.replace(source, tag, 1), "cmd": new,
             "run_dir": str(parent / Path(spec["run_dir"]).name.replace("AdaIS_", "AdaS-RandomT_"))}  # fmt: skip
        if env or spec.get("env"):
            r["env"] = env or spec["env"]
        runs.append(r)
    return write(tag, runs)


BUILDERS = {
    "p3_calib": p3_calib,
    "p3_core": p3_core,
    "p3_core_random_t": lambda: random_t_from("p3_core", "p3_core_random_t", warmup=6),
    "p3_edge": p3_edge,
    "p3_edge_random_t": lambda: random_t_from("p3_edge", "p3_edge_random_t", warmup=6),
    "p3_ablation": p3_ablation,
    "p3_corruption": p3_corruption,
    "p3_benchmark": p3_benchmark,
    "p3_host": p3_host,
    "p3_host_random_t": lambda: random_t_from(
        "p3_host", "p3_host_random_t", warmup=6, env={"JCDE_HOST": "nsga2"}
    ),
    "p3_host_single": p3_host_single,
    "p3_host_single_random_t": lambda: random_t_from(
        "p3_host_single", "p3_host_single_random_t", warmup=6, env={"JCDE_HOST": "single_nsga2"}
    ),
    "p3_scans": p3_scans,
    "p3_e10_twin": p3_e10_twin,
    "p3_e9a": p3_e9a,
    "p3_fatigue": p3_fatigue,
    "p3_uncert": p3_uncert,
    "p3_morris": p3_morris,
}

if __name__ == "__main__":
    for name in sys.argv[1:]:
        BUILDERS[name]()
