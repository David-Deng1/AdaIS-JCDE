"""E10' tables and figure for the paper (D8-1): LLM-based decision-maker agents and their SDM twins on SAV-AERO.

Reads only existing outputs: analysis/e10/ (written by e10_analysis.py), the NL grounding check of the agents'
Fixed-8 sessions (nl_grounding_v2_phaseA_check.json) and the persona definitions the sessions used. Descriptive
only: three seeds per persona and policy, no significance tests, no bootstrap. For the paper the personas are
renamed P1/P2/P3 -> Persona A/B/C (PERSONA below) so that they cannot be confused with the P-balanced and P-edge
profiles of the simulation matrices; Persona C uses the same profile as P-edge. Every count that the evidence map
reports for E10' (evidence_claims.py, rows L1-L4) is recomputed here and asserted equal to evidence_map_rows.csv.

Sign convention: effects are AdaIS minus Fixed-8 within one persona and seed; lower regret and burden and higher
terminal HV favour AdaIS.

Outputs: analysis/e10/paper/e10_paper_paired.csv, e10_paper_channels.csv, e10_paper_nl.csv, e10_paper_tables.md;
analysis/figures_p3/fig_e10_agents.{pdf,png}.

Usage: python e10_paper.py [--out DIR] [--fig_out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_p3 as fp  # noqa: E402  (shared style: rcParams, palette, save)

plt = fp.plt
A = fp.A
E10 = A / "e10"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jcde_paths  # noqa: E402
PERSONAS_JSON = jcde_paths.E10_MATERIALS / "personas.json"
PERSONA = {"P1": "Persona A", "P2": "Persona B", "P3": "Persona C"}
# colour and marker per persona, from the validated categorical slots; Persona C shares P-edge's slot and marker
STYLE = {
    "Persona A": ("#2a78d6", "o"),
    "Persona B": ("#1baf7a", "D"),
    "Persona C": ("#eb6834", "s"),
}
EFFECTS = [("d_regret_rec", "regret", "Recommendation regret", -1), ("d_terminal_hv", "hv", "Terminal HV", 1),
           ("d_C_cog", "burden", "Interaction burden", -1)]  # fmt: skip
SHORT = {"regret": "Regret", "hv": "Terminal HV", "burden": "Burden"}  # panel titles
CHANNELS = [
    ("pairwise", "Pairwise"),
    ("ranking", "Ranking"),
    ("threshold", "Threshold"),
    ("roi", "ROI box"),
]


def paired() -> pd.DataFrame:
    p = pd.read_csv(E10 / "e10_paired.csv")
    p["persona"] = p["persona"].map(PERSONA)
    wide = p.pivot_table(index=["persona", "seed"], columns="source",
                         values=[k for k, *_ in EFFECTS] + ["d_N_int"])  # fmt: skip
    out = pd.DataFrame(index=wide.index)
    for k, short, _, good in EFFECTS + [("d_N_int", "interactions", "", -1)]:
        out[f"{short}_agent"] = wide[(k, "agent")]
        out[f"{short}_sdm"] = wide[(k, "sdm")]
        out[f"{short}_same_sign"] = np.sign(out[f"{short}_agent"]) == np.sign(
            out[f"{short}_sdm"]
        )
        out[f"{short}_agent_favours_adais"] = np.sign(out[f"{short}_agent"]) == good
        out[f"{short}_sdm_favours_adais"] = np.sign(out[f"{short}_sdm"]) == good
    return out.reset_index()


def channels() -> pd.DataFrame:
    c = pd.read_csv(E10 / "e10_channel_agreement.csv")
    rows = []
    for ch, label in CHANNELS:
        row = {"channel": label}
        for src in ("agent", "sdm"):
            r = c[(c["source"] == src) & (c["channel"] == ch)].iloc[0]
            row[f"n_{src}"] = int(r["n_answers"])
            # ROI: share of boxes that overlap the current population (its noise-free label is a strict IoU)
            row[f"share_{src}"] = float(
                r["roi_overlaps_population"] if ch == "roi" else r["agreement_or_iou"]
            )
            if ch == "roi":
                row[f"mean_iou_{src}"] = float(r["agreement_or_iou"])
        rows.append(row)
    return pd.DataFrame(rows)


def nl() -> pd.DataFrame:
    """Agents' NL texts (all asked in Fixed-8 sessions; AdaIS selected NL in none of the 18 sessions), by persona:
    weight statements grounded by the frozen rule-based parser, and reference (numeric) statements grounded."""
    j = json.loads(
        (E10 / "nl_grounding_v2_phaseA_check.json").read_text(encoding="utf-8")
    )
    rec = pd.DataFrame(j["records"])
    rec["persona"] = rec["persona"].map(PERSONA)
    rec["kind"] = np.where(rec["target"].str.startswith("w_"), "weight", "reference")
    rows = []
    for persona, g in rec.groupby("persona"):
        w, r = g[g["kind"] == "weight"], g[g["kind"] == "reference"]
        rows.append({"persona": persona, "nl_texts": len(g), "weight_statements": len(w),
                     "weight_grounded": int((w["outcome"] != "ungrounded").sum()),
                     "weight_correct": int((w["outcome"] == "correct").sum()),
                     "weight_wrong": int((w["outcome"] == "wrong").sum()),
                     "reference_statements": len(r), "reference_grounded": int((r["outcome"] != "ungrounded").sum())})  # fmt: skip
    t = pd.DataFrame(rows)
    total = t.drop(columns="persona").sum().to_dict()
    t = pd.concat([t, pd.DataFrame([{"persona": "All", **total}])], ignore_index=True)
    assert (
        int(total["nl_texts"]) == j["n_nl"]
        and int(total["weight_statements"]) == j["weight"]["n"]
    )
    assert (
        int(total["weight_statements"] - total["weight_grounded"])
        == j["weight"]["ungrounded"]
    )
    assert (
        int(total["reference_statements"] - total["reference_grounded"])
        == j["reference"]["ungrounded"]
    )
    runs = pd.read_csv(E10 / "e10_runs.csv")
    m = runs[runs["matched"]]
    nl_by = m.groupby(["source", "policy"])["n_nl"].sum()
    assert nl_by[("agent", "Fixed-8")] == j["n_nl"], (
        "NL texts in the check differ from the agents' Fixed-8 runs"
    )
    t.attrs["nl_by_policy"] = {f"{s} / {p}": int(v) for (s, p), v in nl_by.items()}
    return t


def assert_evidence_map(pr: pd.DataFrame, ch: pd.DataFrame) -> None:
    """The evidence map (rows L1-L4) and these tables must report the same numbers."""
    ev = pd.read_csv(A / "evidence" / "evidence_map_rows.csv")
    ev = (
        ev[ev["id"].isin(["L1", "L2", "L3", "L4"])]
        .set_index("metric")["adv"]
        .astype(float)
    )
    c = ch.set_index("channel")
    mine = {"same sign, regret": pr["regret_same_sign"].sum(), "same sign, terminal HV": pr["hv_same_sign"].sum(),
            "same sign, burden": pr["burden_same_sign"].sum(), "persona-seed pairs": len(pr),
            "Persona C agent seeds with higher AdaIS regret": (pr.loc[pr.persona == "Persona C", "regret_agent"] > 0).sum(),
            "Persona C agent seeds": (pr.persona == "Persona C").sum(),
            "Persona C SDM-twin seeds with higher AdaIS regret": (pr.loc[pr.persona == "Persona C", "regret_sdm"] > 0).sum(),
            "Persona C SDM-twin seeds": (pr.persona == "Persona C").sum(),
            "agents: share of ROI boxes overlapping the population": c.loc["ROI box", "share_agent"],
            "agents: ROI answers": c.loc["ROI box", "n_agent"],
            "SDM twins: share of ROI boxes overlapping the population": c.loc["ROI box", "share_sdm"],
            "agents: pairwise agreement": c.loc["Pairwise", "share_agent"],
            "SDM twins: pairwise agreement": c.loc["Pairwise", "share_sdm"],
            "agents: ranking agreement": c.loc["Ranking", "share_agent"],
            "SDM twins: ranking agreement": c.loc["Ranking", "share_sdm"]}  # fmt: skip
    for label, v in mine.items():
        assert label in ev.index, f"evidence map has no row {label!r}"
        assert abs(float(v) - float(ev[label])) < 1e-12, (
            f"{label}: tables {v} vs evidence map {ev[label]}"
        )
    print(f"evidence map rows L1-L4: {len(mine)} values agree", flush=True)


def fmt(x: float, nd: int = 3) -> str:
    """Signed value; magnitudes that would round to zero are shown in scientific notation so that the sign used in
    the counts stays visible."""
    if not isinstance(x, float):
        return str(x)
    return f"{x:+.1e}" if 0 < abs(x) < 0.5 * 10**-nd else f"{x:+.{nd}f}"


def tables_md(
    pr: pd.DataFrame, ch: pd.DataFrame, nlt: pd.DataFrame, personas: dict
) -> str:
    L = ["# E10′ tables for the paper (generated by e10_paper.py; do not edit)", "",
         "LLM-based decision-maker agents and their SDM twins on SAV-AERO (fatigue 0.3, budget 10, seeds 51–53). "
         "Descriptive only: one session per persona, policy and seed; no significance tests.", "",
         "## Table E10-1. Personas", "", "| Persona | Session label | True weights (Cd0, Cn) | Target point (Cd0, Cn) |",
         "|---|---|---|---|"]  # fmt: skip
    for key, name in PERSONA.items():
        p = personas[key]
        L.append(f"| {name} | {key} | ({p['w_star'][0]}, {p['w_star'][1]}) | "
                 f"({p['r_star_display']['Cd0']}, {p['r_star_display']['Cn']}) |")  # fmt: skip
    L += ["", "Persona C uses the same preference profile as the P-edge simulation matrix.", "",
          "## Table E10-2. Paired AdaIS minus Fixed-8 effects per persona and seed", "",
          "Lower regret and burden and higher terminal HV favour AdaIS. ‘Same sign’ compares the agent's effect with its "
          "SDM twin's.", "",
          "| Persona | Seed | Regret: agent | Regret: SDM twin | Terminal HV: agent | Terminal HV: SDM twin | "
          "Burden: agent | Burden: SDM twin |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in pr.itertuples():
        L.append(f"| {r.persona} | {r.seed} | {fmt(r.regret_agent)} | {fmt(r.regret_sdm)} | {fmt(r.hv_agent)} | "
                 f"{fmt(r.hv_sdm)} | {fmt(r.burden_agent, 2)} | {fmt(r.burden_sdm, 2)} |")  # fmt: skip
    n = len(pr)
    L.append(f"| Mean | | {fmt(pr.regret_agent.mean())} | {fmt(pr.regret_sdm.mean())} | {fmt(pr.hv_agent.mean())} | "
             f"{fmt(pr.hv_sdm.mean())} | {fmt(pr.burden_agent.mean(), 2)} | {fmt(pr.burden_sdm.mean(), 2)} |")  # fmt: skip
    L.append(f"| Favours AdaIS | | {pr.regret_agent_favours_adais.sum()}/{n} | {pr.regret_sdm_favours_adais.sum()}/{n} | "
             f"{pr.hv_agent_favours_adais.sum()}/{n} | {pr.hv_sdm_favours_adais.sum()}/{n} | "
             f"{pr.burden_agent_favours_adais.sum()}/{n} | {pr.burden_sdm_favours_adais.sum()}/{n} |")  # fmt: skip
    L.append(f"| Same sign as SDM twin | | {pr.regret_same_sign.sum()}/{n} | | {pr.hv_same_sign.sum()}/{n} | | "
             f"{pr.burden_same_sign.sum()}/{n} | |")  # fmt: skip
    mism = []
    for short, label in (("regret", "regret"), ("hv", "terminal HV"), ("burden", "burden")):
        for r in pr[~pr[f"{short}_same_sign"]].itertuples():
            mism.append(f"{r.persona} seed {r.seed}, {label}: agent {fmt(getattr(r, short + '_agent'))}, "
                        f"SDM twin {fmt(getattr(r, short + '_sdm'))}")
    L += ["", "Sign mismatches: " + ("; ".join(mism) if mism else "none") + "."]
    c = pr[pr.persona == "Persona C"]
    L += ["", f"Persona C: AdaIS's regret was higher than under Fixed-8 in {int((c.regret_agent > 0).sum())} of "
          f"{len(c)} seeds for the agents and in {int((c.regret_sdm > 0).sum())} of {len(c)} for the SDM twins.", "",
          "## Table E10-3. Answers that agree with the persona's noise-free answer, by channel", "",
          "| Channel | Agents: answers | Agents: agreement | SDM twins: answers | SDM twins: agreement |",
          "|---|---|---|---|---|"]  # fmt: skip
    for r in ch.itertuples():
        L.append(f"| {r.channel}{' (overlaps the current population)' if r.channel == 'ROI box' else ''} | {r.n_agent} | "
                 f"{r.share_agent:.3f} | {r.n_sdm} | {r.share_sdm:.3f} |")  # fmt: skip
    roi = ch.set_index("channel").loc["ROI box"]
    L += ["", f"ROI boxes: mean IoU with the noise-free box {roi.mean_iou_agent:.3f} (agents) and {roi.mean_iou_sdm:.3f} "
          "(SDM twins).", "", "## Table E10-4. Natural-language answers of the agents", "",
          "All NL answers came from Fixed-8 sessions; " + "; ".join(f"{k}: {v}" for k, v in nlt.attrs["nl_by_policy"].items())
          + " NL answers. Weight statements and reference statements as read by the frozen rule-based parser.", "",
          "| Persona | NL answers | Weight statements | grounded | correct | wrong | Reference statements | grounded |",
          "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in nlt.itertuples():
        L.append(f"| {r.persona} | {r.nl_texts} | {r.weight_statements} | {r.weight_grounded} | {r.weight_correct} | "
                 f"{r.weight_wrong} | {r.reference_statements} | {r.reference_grounded} |")  # fmt: skip
    return "\n".join(L) + "\n"


def figure(pr: pd.DataFrame, ch: pd.DataFrame, out: Path) -> None:
    """(a)-(c) one point per persona and seed: the agent's paired effect (vertical) against its SDM twin's
    (horizontal), dashed line = equal effects; (d) per channel, the share of answers agreeing with the persona's
    noise-free answer (ROI: share of boxes overlapping the current population). Caption text lives in the figure
    list, not in the figure."""
    fig = plt.figure(figsize=(7.2, 2.35))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.42, 1], wspace=0.55)  # column 4 leaves room for (d) labels
    axes = [fig.add_subplot(gs[0, j]) for j in (0, 1, 2, 4)]
    for i, (k, short, label, _) in enumerate(EFFECTS):
        ax = axes[i]
        x, y = pr[f"{short}_sdm"], pr[f"{short}_agent"]
        lim = 1.1 * float(np.nanmax(np.abs(np.r_[x, y])))
        ax.plot([-lim, lim], [-lim, lim], color=fp.INK2, linewidth=0.6, linestyle=(0, (3, 3)))
        ax.axhline(0, color=fp.INK2, linewidth=0.7)
        ax.axvline(0, color=fp.INK2, linewidth=0.7)
        for persona, (color, marker) in STYLE.items():
            g = pr[pr["persona"] == persona]
            ax.scatter(g[f"{short}_sdm"], g[f"{short}_agent"], s=24, color=color, marker=marker,
                       edgecolors="white", linewidths=0.6, zorder=3)  # fmt: skip
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_box_aspect(1)
        ax.set_title(f"({'abc'[i]}) {SHORT[short]}\nsame sign {int(pr[f'{short}_same_sign'].sum())}/{len(pr)}",
                     fontsize=7.5, loc="left")  # fmt: skip
        ax.set_xlabel("SDM twin", fontsize=7.5)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("LLM-based agent", fontsize=7.5)
    ax = axes[3]
    ax.set_box_aspect(1)
    y = np.arange(len(ch))[::-1]
    for yi, r in zip(y, ch.itertuples()):
        ax.plot([r.share_agent, r.share_sdm], [yi, yi], color=fp.GRID, linewidth=2.0, zorder=1)
    for src, mk, face in (("agent", "o", fp.INK), ("sdm", "s", "white")):
        ax.scatter(ch[f"share_{src}"], y, marker=mk, s=24, edgecolors=fp.INK, facecolors=face, linewidths=1.0,
                   zorder=3)  # fmt: skip
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}\n(n = {a}, {s})" for c, a, s in zip(ch["channel"], ch["n_agent"], ch["n_sdm"])],
                       fontsize=6.5)  # fmt: skip
    ax.set_ylim(-0.6, len(ch) - 0.4)
    ax.set_xlim(-0.06, 1.06)
    ax.set_xticks([0, 0.5, 1])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Share agreeing", fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_title("(d) Answers by channel\n ", fontsize=7.5, loc="left")
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="none", markersize=5, label=p)
               for p, (c, m) in STYLE.items()]  # fmt: skip
    handles += [plt.Line2D([], [], color=fp.INK, marker="o", linestyle="none", markersize=5,
                           label="(d) LLM-based agents"),
                plt.Line2D([], [], color=fp.INK, marker="s", markerfacecolor="white", linestyle="none",
                           markersize=5, label="(d) SDM twins")]  # fmt: skip
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 0.99),
               columnspacing=1.2, handletextpad=0.3)  # fmt: skip
    fp.save(fig, out, "fig_e10_agents")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(E10 / "paper"))
    ap.add_argument("--fig_out", default=str(A / "figures_p3"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    personas = json.loads(PERSONAS_JSON.read_text(encoding="utf-8"))
    pr, ch, nlt = paired(), channels(), nl()
    assert_evidence_map(pr, ch)
    pr.to_csv(out / "e10_paper_paired.csv", index=False)
    ch.to_csv(out / "e10_paper_channels.csv", index=False)
    nlt.to_csv(out / "e10_paper_nl.csv", index=False)
    (out / "e10_paper_tables.md").write_text(
        tables_md(pr, ch, nlt, personas), encoding="utf-8"
    )
    figure(pr, ch, Path(args.fig_out))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
