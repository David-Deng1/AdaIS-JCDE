# AdaIS: adaptive interaction scheduling with feedback-quality monitoring

Code for the paper *AdaIS: Adaptive Interaction Scheduling with Feedback-Quality Monitoring for Human-in-the-Loop Computational Engineering Design Optimization* (Journal of Computational Design and Engineering, manuscript JCDE-2026-572).

The original simulation and out-of-model-check archive, including this code (`AdaIS-JCDE-code.zip`), is available at https://doi.org/10.5281/zenodo.23730934. Code repository: https://github.com/David-Deng1/AdaIS-JCDE. That archive contains simulated and external-agent responses. The separate human-pilot supporting records described in the revised manuscript are supplied with the journal submission and are not included in this public repository.

## Channel-availability study added for revision

The complete channel-availability archive for manuscript JCDE-2026-572 is available in the [versioned release](https://github.com/David-Deng1/AdaIS-JCDE/releases/tag/channel-availability-r1-20261108) as `JCDE-2026-572_File_S5_data.zip`. It contains full AdaIS and five single-channel removals on 90 matched blocks each (540 planned simulations), the frozen protocol, a self-contained code snapshot, portable specifications, retained run outputs and recovery records, integrity checks, full-precision analysis, Figure 16 source data and reproduction scripts. `SHA256SUMS.txt` is provided with the release. Extract the archive and follow its own `README.md`.

This archive contains simulated responses. Its code snapshot is separate from the main-branch implementation described below, and its 540-run study is separate from the earlier 9,410-run Zenodo archive. The smaller supporting datasets and the consolidated Supplementary Material are supplied with the revised journal submission.

## Contents

| Path | Contents |
|---|---|
| `hitlo/` | Host optimizer (generator portfolio and rule-based budget split), scheduler, simulated decision maker (SDM), benchmark problems (ZDT1–ZDT3, BNH, TNK) and the SAV-AERO evaluator |
| `runners/run_adais_jcde.py` | Single runner for every scheduler in the paper; `--p3` selects the implementation described in Section 3 |
| `runners/` (other files) | Value-of-information baselines (VoI-Timing-MC, VoI-AllCh), preference model, ground-truth labels, file-relay respondent and rule-based parser of the out-of-model check |
| `jcde_trace/` | Start-up hook that records the per-step state used by the analysis |
| `scripts/build_specs_p3.py`, `scripts/run_chain_p3.py`, `scripts/orchestrator.py` | Build the run matrices and run them (resumable) |
| `scripts/run_e10_sessions.py` | Sessions answered by an external agent through files (out-of-model check, Section 4.7) |
| `scripts/analysis/` | All analyses; `scripts/reproduce_analysis.py` runs them in dependency order |
| `scripts/jcde_paths.py` | Data locations, overridable by environment variables |

## Requirements

Python 3.14.3 with the packages in `requirements.txt` (the versions used for the paper). Runs pin the BLAS libraries to one thread; the orchestrator sets this.

## Data

Download the archive from Zenodo and unpack it next to this repository, so that the folder containing the repository also contains `jcde_p3_runs/`, `jcde_r1_runs/` and `e10_materials/`. Alternatively, point `ADAIS_DATA_ROOT` at the unpacked archive. `DATA.md` in the archive describes its layout.

## Reproduce the tables and figures

```
python scripts/reproduce_analysis.py
```

This rewrites `jcde_p3_runs/analysis/` from the run outputs. To check the archived outputs without overwriting them, write to another folder and compare:

```
ADAIS_ANALYSIS_DIR=/tmp/analysis python scripts/reproduce_analysis.py --compare <data root>/jcde_p3_runs/analysis
```

On the authors' machine every non-image output was identical byte for byte.

## Re-run the simulations

One run, for example AdaIS on SAV-AERO with seed 51:

```
python runners/run_adais_jcde.py --method AdaIS --problem SAV-AERO --profile P-balanced --seed 51 --T 80 \
    --B_total 10 --fatigue_rate 0.3 --sigma_noise 0.1 --output_dir out --p3 --pref_model v2 \
    --p3_nl_model pipeline --p3_nl_fail_w 0.4 --p3_nl_wrong_w 0.5 --p3_nl_fail_r 0.0 \
    --tau_h 0.35 --tau_d 0.25 --T_stag 4 --lambda_cost 0.0 --B_reserve 2 --T_max 2.5 --kappa 3.0 \
    --warmup_ticks 6 --tau_Q_high 0.5 --tau_Q_low 0.1 --bootstrap_channels threshold,ranking,threshold \
    --p3_lambda_B 1.0
```

All matrices, in the order used for the paper (calibration on seeds 41–45, configuration selection, then the held-out matrices):

```
python scripts/run_chain_p3.py
```

The 9,410 simulation runs took about 212 hours of single-threaded run time in total. Every run is deterministic given its seed. Re-running a spec in the environment set by the orchestrator reproduces its outputs byte for byte, apart from the recorded wall-clock time and command line; this was checked on runs from the held-out, corruption and benchmark matrices. Specs record absolute paths, so rebuild them with `scripts/build_specs_p3.py <matrix>` instead of reusing the archived ones.

## Where each result comes from

| Paper | Script (`scripts/analysis/`) | Output (`jcde_p3_runs/analysis/`) |
|---|---|---|
| Section 5.1, Fig. 3 | `bench_eval.py`, `fig_paper.py` | `benchmark/`, `figures_p3/fig_bench_effects` |
| Section 5.2, Figs. 4–6 | `g2_direction.py`, `e_process.py`, `fig_paper.py`, `fig_p3.py` | `g2/`, `process/`, `figures_p3/` |
| Section 5.3, Fig. 7 | `g2_direction.py` (`p3_core` and `p3_ablation`), `fig_paper.py` | `e_ablation/` |
| Section 5.4, Fig. 8 | `e7_corruption.py`, `fig_p3.py` | `e7_corruption/` |
| Section 5.5, Fig. 9 | `e6_uncert.py`, `e6_morris.py`, `e6_scans.py`, `e6_fatigue.py`, `fig_paper.py` | `e6_*/` |
| Section 5.6, Fig. 10, Table S3 | `e10_analysis.py`, `e10_grounding.py`, `e10_paper.py` | `e10/` |
| Section 5.7 | `e5_host.py`, `e9a_voi.py`, `g2_direction.py` (`p3_edge`) | `e5_host/`, `e9a_voi_allch/`, `e2_edge/` |
| Table 5, Table S2 | `evidence_map.py`, `evidence_claims.py` | `evidence/` |
| Table S4 | `seed_cluster_check.py` | `seed_cluster/` |
| Run integrity | `audit_runs.py` | `audit/` |

## Notes

- **SAV-AERO.** The evaluator is a closed-form composition of semi-empirical component relations (Appendix B of the paper). It reads no data file. In this release its two files (`hitlo/problems/sav_aero.py`, `sav_common.py`) carry no comments; the code is otherwise identical to the version used for the runs.
- **Out-of-model check.** The agents ran in OpenAI Codex (model `gpt-6.1-sol`, 22 October 2026). The persona briefs, operating prompt and relay transcripts are in `e10_materials/` of the archive. The persona briefs are role descriptions written by the authors and do not describe the domain experts.
- **Host controller.** No trained budget controller is loaded; the rule-based controller in `hitlo/engine/opt_state.py` splits the evaluation budget (Section 4.3).
