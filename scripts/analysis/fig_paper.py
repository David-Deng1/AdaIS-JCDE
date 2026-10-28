"""Figures of the revised paper that replace the submitted Figs 3-17 (D8-3), drawn from existing analysis outputs.

Same style as fig_p3.py (imported: rcParams, method colour slots and markers, forest panel, save as vector PDF +
600 dpi PNG). No numbers are typed into a figure: every value, count and interval is read from the analysis CSVs
or from the run files they summarise, and intervals are those of the analysis scripts (seed clusters for paired
effects; LHS samples for E6'; Morris bootstrap for mu*). Caption text belongs to the figure list
(JCDE_大修准备_20261020/改稿草稿/图表清单_20261026.md), not to the figures.

  fig_bench_effects      benchmark sign matrix: AdaIS against each baseline and ablation per problem and metric,
                         coloured by whether the seed-cluster CI excludes zero, with seeds favouring AdaIS
                         (replaces Figs 3-5)
  fig_aero_convergence   SAV-AERO mean HV per tick, interactions per five-tick bin and channel mix
                         (P-balanced held-out blocks; replaces Figs 6 and 15)
  fig_aero_ablation      nominal SAV-AERO ablation, paired effects against NoT, NoC and NoQ (replaces Fig 9)
  fig_sensitivity        Morris screening, E6' parameter-uncertainty propagation and fatigue forms
                         (replaces Figs 16-17)

Usage: python fig_paper.py [--out DIR] [--only NAME ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_p3 as fp  # noqa: E402
import truth_eval as te  # noqa: E402
from build_results import PUBLIC  # noqa: E402

plt = fp.plt
A, RUNS = fp.A, fp.RUNS
PROBLEMS = ["ZDT1", "ZDT2", "ZDT3", "BNH", "TNK"]
BENCH_COMP = [
    "Fixed-2",
    "Fixed-4",
    "Fixed-8",
    "Fixed-16",
    "Random-C",
    "NoT",
    "NoC",
    "NoQ",
]
BENCH_METRICS = [("regret_rec", "(a) Recommendation regret"), ("terminal_hv", "(b) Terminal HV"),
                 ("hv_auc", "(c) HV-AUC"), ("C_cog", "(d) Interaction burden")]  # fmt: skip
# diverging status fills: two validated poles and a neutral midpoint (validate_palette.js: poles pass CVD and
# normal-vision separation; contrast relief comes from the text in every cell)
FAVOUR, NEUTRAL, AGAINST = "#5d9ae0", "#e3e2de", "#f08c5f"
CHANNELS = [("ch_threshold", "Threshold"), ("ch_roi", "ROI"), ("ch_ranking", "Ranking"),
            ("ch_pairwise", "Pairwise"), ("ch_nl_clarify", "NL")]  # fmt: skip
TIMING_METHODS = ["AdaIS", "Random-T", "VoI-MC"]
FORMS = {
    "linear": ("#0b0b0b", "o"),
    "saturating": ("#52514e", "s"),
    "step": ("#8d8c88", "D"),
}
# readable labels for the 15 Morris factors (names as in the parameter provenance table)
MORRIS_LABEL = {"alpha1": r"$\alpha_1$ latency-excess weight", "alpha3": r"$\alpha_3$ reversal weight",
                "tau_h": r"$\tau_h$ uncertainty trigger", "tau_d": r"$\tau_d$ conflict trigger",
                "T_stag": r"$T_{stag}$ stagnation ticks", "warmup_ticks": "warm-up ticks",
                "B_reserve": r"$B_{reserve}$ reserved budget", "T_max": r"$T_{max}$ cooldown bound",
                "tau_Q_high": r"$\tau_{Q,high}$ quality boundary", "tau_Q_low": r"$\tau_{Q,low}$ quality boundary",
                "p3_lambda_B": r"$\lambda_B$ burden multiplier", "p3_lambda_min": r"$\lambda_{min}$ multiplier floor",
                "p3_omega": r"$\omega$ phase priority", "nl_clarify": r"$\kappa_{NL}$ NL burden",
                "sigma_noise": r"$\sigma$ answer noise"}  # fmt: skip


def public(df: pd.DataFrame, cols=("a", "b", "method")) -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = df[c].map(lambda m: PUBLIC.get(m, m))
    return df


def fig_bench_effects(out: Path) -> None:
    e = public(pd.read_csv(A / "benchmark" / "bench_paired_effects.csv"))
    e = e[(e["a"] == "AdaIS") & e["b"].isin(BENCH_COMP)]
    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 5.6), gridspec_kw={"hspace": 0.38, "wspace": 0.18}
    )
    for ax, (metric, title) in zip(axes.flat, BENCH_METRICS):
        d = e[e["metric"] == metric].set_index(["b", "problem"])
        for i, comp in enumerate(BENCH_COMP):
            for j, prob in enumerate(PROBLEMS):
                r = d.loc[(comp, prob)]
                fill, sym = (
                    (FAVOUR, "▲ ")
                    if r["ci_low"] > 0
                    else (AGAINST, "▼ ")
                    if r["ci_high"] < 0
                    else (NEUTRAL, "")
                )
                ax.add_patch(
                    plt.Rectangle(
                        (j, i), 1, 1, facecolor=fill, edgecolor="white", linewidth=1.5
                    )
                )
                ax.text(j + 0.5, i + 0.5, f"{sym}{int(r['clusters_favour'])}/{int(r['n_clusters'])}", ha="center",
                        va="center", fontsize=6.5, color=fp.INK)  # fmt: skip
        ax.set_xlim(0, len(PROBLEMS))
        ax.set_ylim(len(BENCH_COMP), 0)
        ax.set_xticks(np.arange(len(PROBLEMS)) + 0.5)
        ax.set_xticklabels(PROBLEMS)
        ax.set_yticks(np.arange(len(BENCH_COMP)) + 0.5)
        ax.set_yticklabels(BENCH_COMP if ax in axes[:, 0] else [])
        ax.tick_params(length=0)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.xaxis.tick_top()
        ax.set_title(title, fontsize=8, loc="left", pad=16)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=FAVOUR, label="▲ CI favours AdaIS"),
               plt.Rectangle((0, 0), 1, 1, facecolor=NEUTRAL, label="CI includes zero"),
               plt.Rectangle((0, 0), 1, 1, facecolor=AGAINST, label="▼ CI favours the comparator")]  # fmt: skip
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.2, columnspacing=1.5)  # fmt: skip
    fp.save(fig, out, "fig_bench_effects")


def aero_ticks() -> pd.DataFrame:
    """Per run and tick on the P-balanced held-out blocks: hypervolume and cumulative interactions."""
    frames = []
    for run in te.iter_runs(RUNS, ["p3_core", "p3_core_random_t"]):
        name = PUBLIC.get(run.method, run.method)
        if name not in fp.TRAJ_METHODS:
            continue
        t = run.ticks
        frames.append(pd.DataFrame({"method": name, "run": str(run.run_dir), "tick": t["tick"], "hv": t["hv"],
                                    "n_int": t["n_int"]}))  # fmt: skip
    return pd.concat(frames, ignore_index=True)


def fig_aero_convergence(out: Path) -> None:
    df = aero_ticks()
    proc = public(pd.read_csv(A / "process" / "process_means.csv"))
    proc = proc[proc["matrix"].isin(["p3_core", "p3_core_random_t"])].set_index(
        "method"
    )
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.5)
    ax = fig.add_subplot(gs[0, 0])
    for m in fp.TRAJ_METHODS:
        g = df[df["method"] == m].groupby("tick")["hv"].mean()
        ax.plot(g.index, g.to_numpy(), color=fp.SLOT[m], linewidth=1.3)
        k = g.iloc[9::10]
        ax.plot(
            k.index,
            k.to_numpy(),
            linestyle="none",
            color=fp.SLOT[m],
            marker=fp.MARK[m],
            markersize=3.5,
        )
    ax.set_xlabel("Tick")
    ax.set_ylabel("Mean hypervolume")
    ax.set_title("(a) Convergence", fontsize=8, loc="left")
    ax = fig.add_subplot(gs[0, 1])
    df = df.sort_values(["run", "tick"])
    df["q"] = df.groupby("run")["n_int"].diff().fillna(df["n_int"])
    df["bin"] = df["tick"] // 5
    for m in TIMING_METHODS:
        g = df[df["method"] == m]
        per = g.groupby(["run", "bin"])["q"].sum().groupby("bin").mean()
        ax.plot(
            per.index * 5 + 2,
            per.to_numpy(),
            color=fp.SLOT[m],
            marker=fp.MARK[m],
            markersize=3.5,
            linewidth=1.2,
        )
    ax.set_xlabel("Tick (five-tick bins)")
    ax.set_ylabel("Mean interactions per bin")
    ax.set_title("(b) Timing of interactions", fontsize=8, loc="left")
    ax = fig.add_subplot(gs[0, 2])
    y = np.arange(len(CHANNELS))[::-1]
    offs = np.linspace(0.2, -0.2, len(TIMING_METHODS) + 1)
    for m, dy in zip(TIMING_METHODS + ["Fixed-8"], offs):
        v = [proc.loc[m, c] for c, _ in CHANNELS]
        ax.plot(
            v,
            y + dy,
            linestyle="none",
            color=fp.SLOT[m],
            marker=fp.MARK[m],
            markersize=4,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in CHANNELS])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean interactions per run")
    ax.set_title("(c) Channel mix", fontsize=8, loc="left")
    names = fp.TRAJ_METHODS
    handles = [
        plt.Line2D([], [], color=fp.SLOT[m], marker=fp.MARK[m], markersize=4, label=m)
        for m in names
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(names),
        fontsize=7,
        bbox_to_anchor=(0.5, 1.08),
    )
    fp.save(fig, out, "fig_aero_convergence")


def fig_aero_ablation(out: Path) -> None:
    e = public(pd.read_csv(A / "e_ablation" / "g2_paired_effects.csv"))
    e = e[(e["a"] == "AdaIS") & e["b"].isin(["NoT", "NoC", "NoQ"])].assign(
        series="AdaIS"
    )
    metrics = [("regret_rec", "Recommendation regret"), ("terminal_hv", "Terminal HV"), ("hv_auc", "HV-AUC"),
               ("C_cog", "Interaction burden"), ("N_int", "Interactions")]  # fmt: skip
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2, 1.9), sharey=True)
    for ax, (metric, label) in zip(axes, metrics):
        fp.forest_panel(
            ax,
            e[e["metric"] == metric],
            label,
            ["NoT", "NoC", "NoQ"],
            {"AdaIS": (fp.INK, "o")},
        )
    axes[0].set_ylabel("AdaIS vs")
    fp.save(fig, out, "fig_aero_ablation")


def fig_sensitivity(out: Path) -> None:
    """(a)-(c) Morris mu* with its bootstrap CI for all 15 screened factors (ZDT1, benchmark configuration), one
    shared factor axis ordered by the mean of mu* scaled to each output's maximum; (d)-(g) E6' paired advantage of
    AdaIS over the 60 LHS samples of the set SDM parameters; (h)-(k) the three fatigue forms under the common
    corruption model. Filled marker = CI excludes zero (positive favours AdaIS)."""
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(3, 12, height_ratios=[2.5, 0.62, 1.0], hspace=0.62, wspace=2.2)
    mor = pd.read_csv(A / "e6_morris" / "morris_indices.csv")
    outs = [("regret_rec", "Recommendation regret"), ("terminal_hv", "Terminal HV"), ("C_cog", "Interaction burden")]
    wide = mor.pivot_table(index="param", columns="output", values="mu_star")[[o for o, _ in outs]]
    order = (wide / wide.max()).mean(axis=1).sort_values().index.tolist()  # least influential at the bottom
    y = np.arange(len(order))
    for i, (out_name, label) in enumerate(outs):
        ax = fig.add_subplot(gs[0, 3 + 3 * i: 6 + 3 * i])
        d = mor[mor["output"] == out_name].set_index("param").reindex(order)
        ax.errorbar(d["mu_star"], y, xerr=d["mu_star_conf"], fmt="o", color=fp.INK, markersize=3.2, capsize=1.6,
                    linewidth=0.9)  # fmt: skip
        ax.set_yticks(y)
        ax.set_yticklabels([MORRIS_LABEL.get(p, p) for p in order] if i == 0 else [], fontsize=6.5)
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(r"Morris $\mu^*$", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(f"({'abc'[i]}) {label}", fontsize=8, loc="left")
    uncert = public(pd.read_csv(A / "e6_uncert" / "e6u_paired_effects.csv")).assign(series="E6'")
    fat = public(pd.read_csv(A / "e6_fatigue" / "e6_fatigue_effects.csv"))
    rows = [("regret_rec", "Regret"), ("terminal_hv", "Terminal HV"), ("hv_auc", "HV-AUC"), ("C_cog", "Burden")]
    for i, (metric, label) in enumerate(rows):
        ax = fig.add_subplot(gs[1, 3 * i: 3 * i + 3])
        fp.forest_panel(ax, uncert[uncert["metric"] == metric], f"({'defg'[i]}) {label}", ["Fixed-8", "VoI-MC"],
                        {"E6'": (fp.INK, "o")}, loc="left")  # fmt: skip
        ax.set_ylim(-0.5, 1.5)
        if i:
            ax.set_yticklabels([])
    rows = [("regret_rec", "Regret"), ("w_err_l1", "Weight error"), ("terminal_hv", "Terminal HV"),
            ("C_cog", "Burden")]  # fmt: skip
    for i, (metric, label) in enumerate(rows):
        ax = fig.add_subplot(gs[2, 3 * i: 3 * i + 3])
        fp.forest_panel(ax, fat[fat["metric"] == metric].rename(columns={"form": "series"}), f"({'hijk'[i]}) {label}",
                        ["NoQ", "Random-C"], FORMS, loc="left")  # fmt: skip
        ax.set_ylim(-0.5, 1.5)
        if i:
            ax.set_yticklabels([])
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="none", markersize=4, label=f"{f} fatigue")
               for f, (c, m) in FORMS.items()]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fp.save(fig, out, "fig_sensitivity")


FIGS = {"fig_bench_effects": fig_bench_effects, "fig_aero_convergence": fig_aero_convergence,
        "fig_aero_ablation": fig_aero_ablation, "fig_sensitivity": fig_sensitivity}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(A / "figures_p3"))
    ap.add_argument("--only", nargs="*", choices=list(FIGS))
    args = ap.parse_args()
    for name in args.only or FIGS:
        FIGS[name](Path(args.out))


if __name__ == "__main__":
    main()
