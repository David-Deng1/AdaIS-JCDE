"""Figures for the revised results (route P3, preference model v2): E1/E2 regret trajectories and paired effects,
E7 monitor discrimination and regret distributions. Static, journal style, 600 dpi PNG + vector PDF.

Every plotted number comes from the analysis outputs (g2/, e2_edge/, e7_corruption/) or from the same truth_eval
references they use, so the figures and the tables cannot disagree. Colour follows the method (fixed slot per
method, validated categorical palette) and every series also carries its own marker, so identity never relies on
colour alone.

  fig_regret_trajectories  normalised recommendation regret against interactions issued and against cumulative
                           burden, P-balanced and P-edge, budget 20 (R3-2)
  fig_e1e2_forest          paired AdaIS advantage per comparator, both profiles (R1-10, R3-2, R4-7)
  fig_e7_monitor           beta gap against the simulator's corruption labels, paired effects against the
                           quality-monitor ablations, regret distribution per method (R4-5, R4-6, R2-2)

Usage: python fig_p3.py [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth_eval as te  # noqa: E402
from build_results import PUBLIC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jcde_paths  # noqa: E402
RUNS = jcde_paths.P3_RUNS
A = jcde_paths.ANALYSIS
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SLOT = {"AdaIS": "#2a78d6", "Random-T": "#eb6834", "VoI-MC": "#1baf7a", "Fixed-2": "#eda100", "Fixed-4": "#e87ba4",
        "Fixed-8": "#008300", "Fixed-16": "#6250d6", "Random-C": "#e34948", "NoT": "#eb6834", "NoC": "#1baf7a",
        "NoQ": "#e34948", "NoQ-pre": "#eda100", "NoQ-post": "#6250d6"}  # fmt: skip
MARK = {"AdaIS": "o", "Random-T": "s", "VoI-MC": "D", "Fixed-2": "^", "Fixed-4": "v", "Fixed-8": "<",
        "Fixed-16": ">", "Random-C": "X", "NoT": "s", "NoC": "D", "NoQ": "X", "NoQ-pre": "^", "NoQ-post": "v"}  # fmt: skip
PROFILE = {"P-balanced": ("#2a78d6", "o"), "P-edge": ("#eb6834", "s")}
COMPARATORS = [
    "VoI-MC",
    "Random-T",
    "Fixed-2",
    "Fixed-4",
    "Fixed-8",
    "Fixed-16",
    "Random-C",
]
# In-figure notes repeat the caption and carry typed counts; the captions now live in the figure list (D8-3).
NOTES = False
TRAJ_METHODS = ["AdaIS", "VoI-MC", "Random-T", "Fixed-8", "Fixed-16", "Random-C"]

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 600, "lines.linewidth": 1.6,
})  # fmt: skip


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


def trajectories(matrices: list) -> pd.DataFrame:
    """Per run and tick: cumulative interactions, cumulative burden and normalised recommendation regret, with the
    references (U*, U0) pooled over the given matrices exactly as g2_direction.py pools them."""
    runs = list(te.iter_runs(RUNS, matrices))
    ref = {
        (r.problem, tuple(r.w_star), tuple(r.r_star)): (r.u_star, r.u0)
        for r in te.references(runs).itertuples()
    }
    frames = []
    for run in runs:
        name = PUBLIC.get(run.method, run.method)
        if name not in TRAJ_METHODS:
            continue
        u_star, u0 = ref[te.truth_key(run)]
        t = run.ticks
        frames.append(pd.DataFrame({"method": name, "tick": t["tick"], "n_int": t["n_int"], "cost": t["cost"],
                                    "regret": (t["u_rec"] - u_star) / max(u0 - u_star, 1e-12)}))  # fmt: skip
    return pd.concat(frames, ignore_index=True)


def fig_regret_trajectories(out: Path) -> None:
    """One curve per method from tick 0 to the last tick: mean cumulative interactions (a) or burden (b) against
    mean regret at that tick. A curve starts at the first tick at which every run has a feasible design, so all
    points average the same runs and the curves end at the table means."""
    profiles = [
        ("P-balanced", ["p3_core", "p3_core_random_t"]),
        ("P-edge", ["p3_edge", "p3_edge_random_t"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharey="row")
    for row, (profile, mats) in enumerate(profiles):
        df = trajectories(mats)
        for m in TRAJ_METHODS:
            g = df[df["method"] == m].groupby("tick").agg(n_int=("n_int", "mean"), cost=("cost", "mean"),
                                                         regret=("regret", "mean"), n=("regret", "count"))  # fmt: skip
            # start where every run has a feasible design, so each point averages the same runs (earlier ticks
            # average a changing subset and draw spurious jumps)
            g = g[g["n"] == g["n"].max()]
            marks = g[g.index % 10 == 9]  # every 10th tick, plus the last one below
            for col, ax in (("n_int", axes[row, 0]), ("cost", axes[row, 1])):
                ax.plot(g[col], g["regret"], color=SLOT[m], linewidth=1.4, label=m)
                ax.plot(
                    marks[col],
                    marks["regret"],
                    linestyle="none",
                    color=SLOT[m],
                    marker=MARK[m],
                    markersize=3.5,
                )
                ax.plot(g[col].iloc[-1], g["regret"].iloc[-1], linestyle="none", color=SLOT[m], marker=MARK[m],
                        markersize=6, markeredgecolor="white", markeredgewidth=0.8)  # fmt: skip
        axes[row, 0].set_ylabel(f"{profile}\nmean normalised regret")
        for ax in axes[row]:
            ax.axhline(1.0, color=INK2, linewidth=0.6, linestyle=(0, (3, 3)))
    axes[1, 0].set_xlabel("Mean interactions issued")
    axes[1, 1].set_xlabel("Mean cumulative interaction burden")
    axes[0, 0].set_title("(a) against interactions", loc="left", fontsize=8)
    axes[0, 1].set_title("(b) against burden", loc="left", fontsize=8)
    handles = [
        plt.Line2D([], [], color=SLOT[m], marker=MARK[m], markersize=4, label=m)
        for m in TRAJ_METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(TRAJ_METHODS),
        fontsize=7,
        bbox_to_anchor=(0.5, 1.03),
    )
    if NOTES:
        fig.text(0.5, -0.02, "Each curve follows the run from tick 0 to tick 79 (small markers every 10 ticks, large marker "
                 "= end of run); 90 runs per method (3 fatigue levels x 3 budgets x 10 held-out seeds); ticks before a run's first feasible design are left out.\nRegret of the "
                 "design the method would recommend: 1 = no better than the initial best design, 0 = best design known.",
                 ha="center", fontsize=7, color=INK2)  # fmt: skip
    save(fig, out, "fig_regret_trajectories")


def forest_panel(
    ax, rows: pd.DataFrame, label: str, order: list, series: dict, loc: str = "center"
) -> None:
    """One metric: a point and CI per comparator and series (first series on top, as in the legend); filled
    marker = CI excludes zero."""
    y0 = np.arange(len(order))[::-1]
    offsets = np.linspace(0.18, -0.18, len(series)) if len(series) > 1 else [0.0]
    for (name, (color, marker)), dy in zip(series.items(), offsets):
        d = rows[rows["series"] == name].set_index("b")
        for yi, comp in zip(y0, order):
            if comp not in d.index:
                continue
            r = d.loc[comp]
            sig = r["ci_low"] > 0 or r["ci_high"] < 0
            ax.errorbar(r["advantage"], yi + dy, xerr=[[r["advantage"] - r["ci_low"]], [r["ci_high"] - r["advantage"]]],
                        fmt=marker, color=color, markerfacecolor=color if sig else "white", markersize=4.5,
                        capsize=1.8, linewidth=1.0, markeredgewidth=1.0)  # fmt: skip
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_title(label, fontsize=8, color=INK, loc=loc)
    ax.set_yticks(y0)
    ax.set_yticklabels(order)
    ax.grid(axis="y", visible=False)


METRICS_E12 = [("regret_rec", "Recommendation regret"), ("terminal_hv", "Terminal HV"), ("hv_auc", "HV-AUC"),
               ("C_cog", "Interaction burden")]  # fmt: skip


def fig_e1e2_forest(out: Path) -> None:
    rows = []
    for profile, rel in (
        ("P-balanced", "g2/g2_paired_effects.csv"),
        ("P-edge", "e2_edge/g2_paired_effects.csv"),
    ):
        e = pd.read_csv(A / rel)
        rows.append(e.assign(series=profile, b=e["b"].map(lambda m: PUBLIC.get(m, m))))
    e = pd.concat(rows)
    fig, axes = plt.subplots(1, len(METRICS_E12), figsize=(7.2, 2.9), sharey=True)
    for ax, (metric, label) in zip(axes, METRICS_E12):
        forest_panel(ax, e[e["metric"] == metric], label, COMPARATORS, PROFILE)
    axes[0].set_ylabel("AdaIS vs")
    handles = [plt.Line2D([], [], color=c, marker=mk, linestyle="none", markersize=4.5, label=p)
               for p, (c, mk) in PROFILE.items()]  # fmt: skip
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        fontsize=7,
        bbox_to_anchor=(0.5, 1.06),
    )
    if NOTES:
        fig.text(0.5, -0.05, "Paired AdaIS advantage (positive favours AdaIS) over 90 matched blocks, 95% bootstrap CI "
                 "with the 10 seeds as clusters; filled marker = CI excludes zero.", ha="center", fontsize=7,
                 color=INK2)  # fmt: skip
    save(fig, out, "fig_e1e2_forest")


def fig_e7_monitor(out: Path) -> None:
    gap = pd.read_csv(A / "e7_corruption/e7_beta_gap.csv")
    eff = pd.read_csv(A / "e7_corruption/e7_paired_effects.csv")
    runs = pd.read_csv(A / "e7_corruption/e7_runs.csv")
    for d in (gap, eff, runs):
        for col in ("method", "b"):
            if col in d:
                d[col] = d[col].map(lambda m: PUBLIC.get(m, m))
    monitor = [
        "AdaIS",
        "NoT",
        "NoQ-pre",
        "NoC",
    ]  # variants in which beta can differ from 1
    abl = ["NoQ", "NoQ-pre", "NoQ-post", "NoT", "NoC", "Fixed-8", "Random-C"]
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.55, wspace=0.9)
    ax = fig.add_subplot(gs[0, 0])
    g = gap.set_index("method").reindex(monitor)
    y = np.arange(len(monitor))[::-1]
    for yi, m in zip(y, monitor):
        r = g.loc[m]
        sig = r["ci_low"] > 0
        ax.errorbar(r["beta_gap"], yi, xerr=[[r["beta_gap"] - r["ci_low"]], [r["ci_high"] - r["beta_gap"]]],
                    fmt=MARK[m], color=SLOT[m], markerfacecolor=SLOT[m] if sig else "white", markersize=4.5,
                    capsize=1.8, linewidth=1.0)  # fmt: skip
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(monitor)
    ax.grid(axis="y", visible=False)
    ax.set_title("(a) β gap (clean − corrupted)", fontsize=8, loc="left")
    for i, (metric, label) in enumerate([("w_err_l1", "(b) Weight error"), ("regret_rec", "(c) Regret"),
                                         ("C_cog", "(d) Burden")]):  # fmt: skip
        ax = fig.add_subplot(gs[0, i + 1])
        forest_panel(
            ax,
            eff[eff["metric"] == metric].assign(series="AdaIS"),
            label,
            abl,
            {"AdaIS": (INK, "o")},
            loc="left",
        )
        if i:
            ax.set_yticklabels([])
    ax = fig.add_subplot(gs[1, :])
    order = ["AdaIS", "NoQ", "NoQ-pre", "NoQ-post", "NoT", "NoC", "Fixed-8", "Random-C"]
    data = [runs.loc[runs["method"] == m, "regret_rec"].to_numpy() for m in order]
    bp = ax.boxplot(data, positions=np.arange(len(order)), widths=0.5, whis=(10, 90), showfliers=False,
                    patch_artist=True, medianprops={"color": INK, "linewidth": 1.2})  # fmt: skip
    # the x labels identify the methods; colour only marks AdaIS (ablations would otherwise reuse comparator slots)
    tone = {m: SLOT["AdaIS"] if m == "AdaIS" else "#8d8c88" for m in order}
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor("white")
        patch.set_edgecolor(tone[m])
    rng = np.random.default_rng(20261024)
    for i, (m, d) in enumerate(zip(order, data)):
        ax.scatter(
            i + rng.uniform(-0.18, 0.18, len(d)),
            d,
            s=6,
            color=tone[m],
            alpha=0.55,
            linewidths=0,
        )
        k = int((d > 2).sum())
        ax.text(
            i,
            np.max(d) + 0.15,
            f"{k} run{'s' if k != 1 else ''} > 2",
            ha="center",
            fontsize=6.5,
            color=INK2,
        )
    ax.axhline(1.0, color=INK2, linewidth=0.6, linestyle=(0, (3, 3)))
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylim(-0.2, max(np.max(d) for d in data) + 0.7)
    ax.set_ylabel("Normalised recommendation regret")
    ax.set_title("(e) Regret per run under common corruption (box: median, IQR, whiskers p10-p90)", fontsize=8,
                 loc="left")  # fmt: skip
    if NOTES:
        fig.text(0.5, 0.0, "(a) runs with at least one corrupted answer; (b)-(d) paired AdaIS advantage over 90 matched "
                 "blocks. 95% bootstrap CIs with seeds as clusters; filled = CI excludes zero.", ha="center",
                 fontsize=7, color=INK2)  # fmt: skip
    save(fig, out, "fig_e7_monitor")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(A / "figures_p3"))
    out = Path(ap.parse_args().out)
    fig_e1e2_forest(out)
    fig_e7_monitor(out)
    fig_regret_trajectories(out)


if __name__ == "__main__":
    main()
