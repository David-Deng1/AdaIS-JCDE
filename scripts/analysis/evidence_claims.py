"""Evidence map for the paper (D8-2; R1-16, R3-10): each revised claim -> the rows of the analysis outputs that
support it, with its evidence layer, decision-maker model, comparators, matching unit, inference type and status.

Claims are listed in evidence_claims.json (next to this script). No number is typed there: every number in the
outputs is read from the analysis CSVs, mainly analysis/evidence/evidence_table.csv (evidence_map.py), plus the
beta gap, the corruption-by-fatigue slices, the regret tail, the process statistics, the Morris indices and the
E10' tables. The status of a claim is computed from its rows by the fixed rule below and must equal the status the
claim declares, so a claim whose wording no longer matches the evidence stops the script.

Row rule (paired effects; CI = 95% bootstrap with seeds as clusters, decision of 10-24; E6' resamples parameter
samples; share = clusters favouring AdaIS / clusters, see truth_eval.paired_effect):
    expect "+"   CI above 0 and share >= 0.5 -> robust; CI above 0 and share < 0.5 -> directional;
                 CI spans 0 and advantage > 0 -> directional; CI spans 0 and advantage <= 0 -> not supported;
                 CI below 0 -> contradiction
    expect "-"   mirror image: CI below 0 and share <= 0.5 -> adverse; ... CI above 0 -> contradiction
    expect "0"   CI spans 0 -> null; otherwise contradiction
    expect ">=0" (not worse) / "<=0" (not better): null unless the CI excludes 0 on the forbidden side (contradiction)
Claim rule: no evidence -> not supported. Descriptive evidence only -> descriptive if all its checks hold, otherwise
not supported (a failed check stops the script for a claim kept in the paper). Otherwise the weakest primary row in
the claim's direction: favourable: not supported < directional < robust; adverse: not supported < directional <
adverse; none: every primary row must be null. Context rows are shown and must not contradict (a "+" or "-" context row needs at least its point estimate in
that direction); they do not set the
status. Wording checks on paper claims: directional claims say "directional", null claims say "no detectable" or
"no worse", E10' claims say "LLM-based decision-maker agents", and a short list of phrases is refused.

Outputs (analysis/evidence/): evidence_map_paper.csv (one row per claim), evidence_map_rows.csv (one row per
evidence row, all numbers), evidence_map_paper.md (paper tables, English), 证据地图_审阅版.md (review version for
the authors, in Chinese, with the submitted wording and the type of change).

Usage: python evidence_claims.py [--out DIR]
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import operator
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_results import PUBLIC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jcde_paths  # noqa: E402
RUNS = jcde_paths.P3_RUNS
A = jcde_paths.ANALYSIS
CLAIMS = HERE / "evidence_claims.json"
NAMES = {**PUBLIC, "VoI-Timing-MC-AllCh": "VoI-AllCh"}

SECTIONS = {
    "B": "Benchmark",
    "E1": "E1 aero P-balanced",
    "E2": "E2 aero P-edge",
    "E8": "E8 aero nominal ablation",
    "E7": "E7/E8 common corruption",
    "E9a": "E9a VoI all-channel posterior",
    "E5": "E5 host",
    "E6u": "E6' uncertainty propagation",
    "E6f": "E6 fatigue form",
    "E6s": "E6 scans",
}
SECTION_LAYER = {"B": "benchmark", "E1": "aero", "E8": "aero", "E9a": "aero", "E2": "edge", "E7": "corruption",
                 "E5": "host", "E6u": "uncertainty", "E6f": "uncertainty", "E6s": "uncertainty"}  # fmt: skip
# source: csv, kind, name of the estimate column (effect sources), layer (None: decided per row)
SOURCES = {
    "beta_gap": ("e7_corruption/e7_beta_gap.csv", "effect", "beta_gap", "corruption"),
    "fatigue_slice": (
        "e7_corruption/e7_by_fatigue.csv",
        "effect",
        "advantage",
        "corruption",
    ),
    "tail": ("e7_corruption/e7_regret_tail.csv", "table", None, "corruption"),
    "tail_profile": ("e7_corruption/e7_tail_profile.csv", "table", None, "corruption"),
    "process": ("process/process_means.csv", "table", None, None),
    "morris": ("e6_morris/morris_indices.csv", "table", None, "uncertainty"),
    "e10_dir": ("e10/e10_direction_agreement.csv", "table", None, "e10"),
    "e10_ch": ("e10/e10_channel_agreement.csv", "table", None, "e10"),
    "e10_pair": ("e10/e10_paired.csv", "table", None, "e10"),
}
PROCESS_LAYER = {
    "p3_core": "aero",
    "p3_core_random_t": "aero",
    "p3_corruption": "corruption",
}
LAYERS = {  # layer: (paper label, decision-maker model)
    "benchmark": ("Benchmark sanity check (ZDT1-3, BNH, TNK)", "SDM with known utility"),
    "aero": ("Held-out SAV-AERO simulation (P-balanced)", "Parametric SDM, balanced persona, ground-truth utility"),
    "edge": ("P-edge boundary (SAV-AERO)", "Parametric SDM, drag-priority persona"),
    "corruption": ("Common-corruption stress (SAV-AERO)", "Parametric SDM with common corruption model"),
    "host": ("Host variants (preference-blind NSGA-II; single generator)", "Parametric SDM (known utility)"),
    "uncertainty": ("Parameter uncertainty and sensitivity (E6', Morris, scans, fatigue forms)",
                    "Parametric SDM with sampled or scanned set parameters"),
    "e10": ("E10' LLM-based decision-maker agents (SAV-AERO)",
            "LLM-based decision-maker agents (persona-conditioned) and their SDM twins"),
}  # fmt: skip
LAYER_ZH = {"benchmark": "基准检验", "aero": "SAV-AERO 留出仿真（P-balanced）", "edge": "P-edge 边界条件",
            "corruption": "共同污染压力测试", "host": "宿主变体", "uncertainty": "参数不确定性与敏感性",
            "e10": "E10′ LLM 智能体决策者"}  # fmt: skip
STATUS_ZH = {"robust": "稳健", "directional": "方向性", "adverse": "不利（显著）", "null": "无可检测差异",
             "descriptive": "描述性", "not supported": "不成立"}  # fmt: skip
METRIC = {"regret_rec": "regret", "regret_best": "best-design regret", "terminal_hv": "terminal HV",
          "hv_auc": "HV-AUC", "w_err_l1": "weight error", "C_cog": "burden", "N_int": "interactions",
          "T_resp": "latency", "absorbed_corrupted": "corrupted evidence absorbed",
          "corrupt_share_absorbed": "corrupted share absorbed", "n_nl": "NL queries", "beta_gap": "β gap"}  # fmt: skip
REL_METRICS = {"C_cog", "N_int", "terminal_hv", "hv_auc", "T_resp"}
BENCH = {"BNH", "TNK", "ZDT1", "ZDT2", "ZDT3"}
REFUSED = ["equivalent", "comparable quality", "engineering-scale", "validat", "human", "participant", "pilot",
           "quality control"]  # fmt: skip
RANK = {
    "favourable": ["not supported", "directional", "robust"],
    "adverse": ["not supported", "directional", "adverse"],
}


class ClaimError(RuntimeError):
    pass


def public(name) -> str:
    method, _, variant = str(name).partition("|")
    return NAMES.get(method, method) + (f"|{variant}" if variant else "")


def load_sources() -> dict[str, pd.DataFrame]:
    t = pd.read_csv(A / "evidence" / "evidence_table.csv")
    t["scope"] = t["scope"].fillna("")
    out = {"evidence": t}
    for name, (rel, _, _, _) in SOURCES.items():
        d = pd.read_csv(A / rel)
        for col in ("method", "a", "b"):
            if col in d:
                d[col] = d[col].map(public)
        out[name] = d
    return out


def expand(where: dict) -> list[dict]:
    keys = list(where)
    vals = [v if isinstance(v, list) else [v] for v in where.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def select(
    df: pd.DataFrame, where: dict, what: str, single: bool = True
) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    for k, v in where.items():
        if k not in df:
            raise ClaimError(f"{what}: no column {k!r}")
        m &= df[k] == v
    rows = df[m]
    if rows.empty or (single and len(rows) != 1):
        raise ClaimError(f"{what}: {where} matches {len(rows)} rows")
    return rows


def spec_where(spec: dict) -> tuple[str, dict]:
    """Evidence spec -> (source, where). Short form for evidence_table: s, scope, a, b, m."""
    if "s" in spec:
        where = {"section": SECTIONS[spec["s"]], "scope": spec.get("scope", ""), "a": spec.get("a", "AdaIS"),
                 "b": spec["b"], "metric": spec["m"]}  # fmt: skip
        return "evidence", where
    return spec["src"], dict(spec["where"])


def effect_row(src: dict, source: str, where: dict, spec: dict, cid: str) -> dict:
    df = src[source]
    r = select(df, where, f"{cid} {source}").iloc[0]
    est = SOURCES[source][2] if source in SOURCES else "advantage"
    if source == "evidence":
        layer = SECTION_LAYER[spec["s"]]
        section, scope, a, b, metric = (
            r["section"],
            r["scope"],
            r["a"],
            r["b"],
            r["metric"],
        )
        n = int(r["n"])
    else:
        layer = SOURCES[source][3]
        section, scope = source, f"fatigue {r['fatigue']:g}" if "fatigue" in r else ""
        a = (
            r.get("a", r.get("method", "AdaIS"))
            if source != "beta_gap"
            else r["method"]
        )
        b = r.get("b", "") if source != "beta_gap" else "corruption labels"
        metric = r.get("metric", "beta_gap")
        n = int(r.get("n", r.get("n_runs", 0)))
    mean_b = r.get("mean_b", float("nan"))
    rel = (100 * r[est] / abs(mean_b), 100 * r["ci_low"] / abs(mean_b), 100 * r["ci_high"] / abs(mean_b)) if (
        metric in REL_METRICS and pd.notna(mean_b) and abs(mean_b) > 1e-12) else None  # fmt: skip
    if source == "evidence" and spec["s"] == "E6u":
        unit, inference = (
            f"{int(r['n_clusters'])} parameter samples",
            "inferential (bootstrap over parameter samples)",
        )
    else:
        per = " per problem" if str(scope).split("/")[-1] in BENCH else ""
        unit = f"{n} {'runs' if source == 'beta_gap' else 'blocks'} / {int(r['n_clusters'])} seed clusters{per}"
        inference = "inferential (seed-cluster bootstrap)"
    return {"kind": "effect", "source": source, "section": section, "scope": scope, "a": a, "b": b,
            "metric": metric, "n": n, "mean_a": r.get("mean_a", float("nan")), "mean_b": mean_b,
            "adv": float(r[est]), "lo": float(r["ci_low"]), "hi": float(r["ci_high"]),
            "fav": int(r["clusters_favour"]), "nc": int(r["n_clusters"]), "rel": rel, "layer": layer,
            "unit": unit, "inference": inference}  # fmt: skip


def row_status(r: dict, expect: str, cid: str) -> str:
    lo, hi, adv, share = r["lo"], r["hi"], r["adv"], r["fav"] / r["nc"]
    what = f"{cid}: {r['metric']} {r['a']} vs {r['b']} {r['scope']} expects {expect!r} but CI [{lo:.3g}, {hi:.3g}]"
    if expect == "+":
        if hi < 0:
            raise ClaimError(what)
        if lo > 0:
            return "robust" if share >= 0.5 else "directional"
        return "directional" if adv > 0 else "not supported"
    if expect == "-":
        if lo > 0:
            raise ClaimError(what)
        if hi < 0:
            return "adverse" if share <= 0.5 else "directional"
        return "directional" if adv < 0 else "not supported"
    if expect == "0":
        if lo > 0 or hi < 0:
            raise ClaimError(what)
        return "null"
    if expect == ">=0":
        if hi < 0:
            raise ClaimError(what)
        return "null"
    if expect == "<=0":
        if lo > 0:
            raise ClaimError(what)
        return "null"
    raise ClaimError(f"{cid}: unknown expectation {expect!r}")


def table_value(src: dict, spec: dict, cid: str):
    source, where = spec["src"], spec["where"]
    rows = select(
        src[source],
        where,
        f"{cid} {source}",
        single=spec.get("agg", "value") == "value",
    )
    col = rows[spec["field"]]
    agg = spec.get("agg", "value")
    if agg == "value":
        v = col.iloc[0]
    elif agg == "count":
        v = int(col.count())
    elif agg == "count_pos":
        v = int((col > 0).sum())
    elif agg == "min":
        v = col.min()
    elif agg == "mean":
        v = col.mean()
    elif agg == "n_items":
        v = len(str(col.iloc[0]).split())
    else:
        raise ClaimError(f"{cid}: unknown aggregation {agg!r}")
    if source == "process":
        layer = PROCESS_LAYER[where["matrix"]]
        unit = f"{int(rows['n_runs'].iloc[0])} runs"
    elif source in ("tail", "tail_profile"):
        layer, unit = (
            SOURCES[source][3],
            f"{int(rows[{'tail': 'n', 'tail_profile': 'n_runs'}[source]].iloc[0])} runs",
        )
    elif source == "e10_dir":
        layer, unit = "e10", f"{int(rows['n_pairs'].iloc[0])} persona-seed pairs"
    elif source == "e10_ch":
        layer, unit = "e10", f"{int(rows['n_answers'].iloc[0])} answers"
    elif source == "e10_pair":
        layer, unit = "e10", f"{len(rows)} sessions"
    else:
        layer, unit = SOURCES[source][3], "Morris elementary effects"
    v = v.item() if hasattr(v, "item") else v
    return v, layer, unit


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Eq: operator.eq,
        ast.NotEq: operator.ne}  # fmt: skip


def safe_eval(expr: str, names: dict):
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ClaimError(f"check {expr!r}: unknown name {node.id!r}")
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and type(node.ops[0]) in _OPS
        ):
            return _OPS[type(node.ops[0])](ev(node.left), ev(node.comparators[0]))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "abs"
        ):
            return abs(ev(node.args[0]))
        raise ClaimError(f"check {expr!r}: unsupported expression")

    return ev(ast.parse(expr, mode="eval"))


def disp(name: str) -> str:
    """Scan arms are 'method|variant'; shown as 'method (variant)' so table cells keep their bars."""
    method, _, variant = str(name).partition("|")
    return f"{method} ({variant})" if variant else method


def fmt(x: float) -> str:
    return f"{x:.3g}"


def effect_text(rows: list[dict]) -> list[str]:
    """One line per (a, b, metric); rows over several scopes are summarised by scope sign."""
    out = []
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["a"], r["b"], r["metric"], r["source"]), []).append(r)
    for (a, b, metric, _), g in groups.items():
        vs = f"{disp(a)} vs {disp(b)}" if a != "AdaIS" else f"vs {disp(b)}"
        label = METRIC.get(metric, metric)
        if len(g) == 1:
            r = g[0]
            scope = f"{r['scope']}: " if r["scope"] else ""
            rel = (
                f" ({r['rel'][0]:.1f}% [{r['rel'][1]:.1f}, {r['rel'][2]:.1f}])"
                if r["rel"]
                else ""
            )
            out.append(f"{scope}{label} {vs}: {fmt(r['adv'])} [{fmt(r['lo'])}, {fmt(r['hi'])}]{rel}, "
                       f"{r['fav']}/{r['nc']} {r['sign']}")  # fmt: skip
        else:
            signs = ", ".join(f"{r['scope'].split('/')[-1]} {r['sign']}" for r in g)
            advs = [r["adv"] for r in g]
            favs = [r["fav"] for r in g]
            fav = (
                f"{min(favs)}" if min(favs) == max(favs) else f"{min(favs)}-{max(favs)}"
            )
            out.append(f"{label} {vs}: {signs}; range {fmt(min(advs))} to {fmt(max(advs))}, "
                       f"{fav}/{g[0]['nc']}")  # fmt: skip
    return out


def value_text(v) -> str:
    return fmt(v) if isinstance(v, float) else str(v)


def build(src: dict, claims: list[dict]) -> tuple[list[dict], list[dict]]:
    out, long = [], []
    for c in claims:
        cid, paper = c["id"], c.get("paper", True)
        if c["layer"] not in LAYERS:
            raise ClaimError(f"{cid}: unknown layer {c['layer']!r}")
        effects, values, names, layers, units, inferences = [], [], {}, set(), [], []
        for spec in c["evidence"]:
            source, where = spec_where(spec)
            if source == "evidence" or SOURCES[source][1] == "effect":
                expanded = expand(where)
                if spec.get("key") and len(expanded) > 1:
                    raise ClaimError(
                        f"{cid}: key on a spec that expands to {len(expanded)} rows"
                    )
                for w in expanded:
                    r = effect_row(src, source, w, spec, cid)
                    r["expect"], r["role"] = spec["x"], spec.get("role", "primary")
                    r["status"] = row_status(r, r["expect"], cid)
                    if r["role"] == "context" and r["status"] == "not supported":
                        raise ClaimError(
                            f"{cid}: context row {r['metric']} vs {r['b']} {r['scope']} expects "
                            f"{r['expect']!r} but the point estimate goes the other way"
                        )
                    r["sign"] = "+" if r["lo"] > 0 else ("-" if r["hi"] < 0 else "0")
                    effects.append(r)
                    if spec.get("key"):
                        for f in ("adv", "lo", "hi", "fav", "nc"):
                            names[f"{spec['key']}_{f}"] = r[f]
            else:
                v, layer, unit = table_value(src, spec, cid)
                values.append(
                    {
                        "label": spec["label"],
                        "value": v,
                        "layer": layer,
                        "unit": unit,
                        "source": source,
                    }
                )
                if spec.get("key"):
                    names[spec["key"]] = v
        for key, spec in c.get("keys", {}).items():
            source, where = spec_where(spec)
            r = effect_row(src, source, where, spec, cid)
            for f in ("adv", "lo", "hi", "fav", "nc"):
                names[f"{key}_{f}"] = r[f]
        for r in effects:
            layers.add(r["layer"])
            units.append(r["unit"])
            inferences.append(r["inference"])
        for v in values:
            layers.add(v["layer"])
            units.append(v["unit"])
            inferences.append("descriptive")
        if layers - {c["layer"]}:
            raise ClaimError(
                f"{cid}: layer {c['layer']!r} but evidence from {sorted(layers)}"
            )
        checks = [(chk, bool(safe_eval(chk, names))) for chk in c.get("checks", [])]
        failed = [chk for chk, ok in checks if not ok]
        if failed and paper:
            raise ClaimError(f"{cid}: check failed: {failed}")

        primary = [r for r in effects if r["role"] == "primary"]
        if not effects and not values:
            status = "not supported"
        elif not primary:
            status = "not supported" if failed else "descriptive"
        elif c["direction"] == "none":
            if any(r["status"] != "null" for r in primary):
                raise ClaimError(
                    f"{cid}: direction 'none' but primary rows {[r['status'] for r in primary]}"
                )
            status = "null"
        else:
            rank = RANK[c["direction"]]
            bad = [r["status"] for r in primary if r["status"] not in rank]
            if bad:
                raise ClaimError(
                    f"{cid}: primary rows {bad} do not fit direction {c['direction']!r}"
                )
            status = min((r["status"] for r in primary), key=rank.index)
        if status != c["status"]:
            raise ClaimError(
                f"{cid}: wording claims {c['status']!r}, rule gives {status!r}"
            )
        if paper:
            text = c["claim_en"].lower()
            refused = [p for p in REFUSED if p in text]
            if refused:
                raise ClaimError(f"{cid}: refused phrase {refused}")
            if status == "directional" and "directional" not in text:
                raise ClaimError(f"{cid}: directional claim must say so")
            if status == "null" and not ("no detectable" in text or "no worse" in text):
                raise ClaimError(
                    f"{cid}: null claim must say 'no detectable' or 'no worse'"
                )
            if c["layer"] == "e10" and "llm-based decision-maker agents" not in text:
                raise ClaimError(
                    f"{cid}: E10' claims must say 'LLM-based decision-maker agents'"
                )

        comps = []
        for r in effects:
            label = (
                disp(r["b"])
                if r["a"] == "AdaIS"
                else f"{disp(r['a'])} vs {disp(r['b'])}"
            )
            if label not in comps:
                comps.append(label)
        lines = effect_text([r for r in effects if r["role"] == "primary"])
        ctx = effect_text([r for r in effects if r["role"] == "context"])
        vals = [f"{v['label']}: {value_text(v['value'])}" for v in values]
        out.append({
            "id": cid, "paper": paper, "main": bool(c.get("main")), "layer": c["layer"],
            "layer_label": LAYERS[c["layer"]][0], "decision_maker": c.get("dm", LAYERS[c["layer"]][1]),
            "comparators": c.get("comparators", ", ".join(comps) or "-"),
            "unit": "; ".join(dict.fromkeys(units)) or "-",
            "inference": " + ".join(dict.fromkeys(inferences)) or "-", "direction": c["direction"],
            "status": status, "claim_en": c["claim_en"], "claim_zh": c["claim_zh"],
            "effects": " | ".join(lines), "context": " | ".join(ctx), "values": " | ".join(vals),
            "checks": "; ".join(f"{chk} ({'ok' if ok else 'fails'})" for chk, ok in checks),
            "original": c["original"], "review3": c["review3"], "change": c["change"], "note": c.get("note", ""),
            "comments": ", ".join(c.get("comments", [])),
            "vs_review3": c.get("vs_review3", ""),
        })  # fmt: skip
        for r in effects:
            long.append({"id": cid, **{k: r[k] for k in ("role", "expect", "status", "sign", "section", "scope", "a",
                         "b", "metric", "n", "mean_a", "mean_b", "adv", "lo", "hi", "fav", "nc", "unit")}})  # fmt: skip
        for v in values:
            long.append({"id": cid, "role": "descriptive", "section": v["source"], "metric": v["label"],
                         "adv": v["value"], "unit": v["unit"]})  # fmt: skip
    return out, long


def write_md(out: list[dict], path: Path) -> None:
    paper = [c for c in out if c["paper"]]
    head = ("| ID | Claim | Evidence layer | Decision maker | Comparators | Matching unit | Inference | "
            "Effect of AdaIS [95% CI], seeds favouring AdaIS | Status |")  # fmt: skip
    rule = "|---|---|---|---|---|---|---|---|---|"

    def rows(cs):
        lines = [head, rule]
        for c in cs:
            eff = "<br>".join(x for x in (c["effects"].replace(" | ", "<br>"),
                                          ("context: " + c["context"].replace(" | ", "<br>")) if c["context"] else "",
                                          c["values"].replace(" | ", "<br>")) if x)  # fmt: skip
            lines.append(f"| {c['id']} | {c['claim_en']} | {c['layer_label']} | {c['decision_maker']} | "
                         f"{c['comparators']} | {c['unit']} | {c['inference']} | {eff} | {c['status']} |")  # fmt: skip
        return lines

    counts = pd.Series([c["status"] for c in paper]).value_counts()
    text = [
        "# Evidence map (generated by evidence_claims.py from evidence_claims.json; do not edit)", "",
        "Effect: advantage of AdaIS (positive favours AdaIS; for regret, weight error, burden, interactions and "
        "latency it is the reduction), with the 95% bootstrap CI that resamples seeds as clusters (parameter samples "
        "for E6'), the relative advantage in brackets where meaningful, and the number of seed clusters (samples) "
        "whose summed advantage favours AdaIS. Sign: + CI above 0, - below 0, 0 spans 0. Rows over several "
        "benchmark problems are summarised by problem sign and the range of the advantage. Under the balanced "
        "persona the true weights equal the initial weights, so weight recovery is only reported where it carries "
        "information.", "",
        "Status (fixed rule, evidence_claims.py): robust = CI excludes 0 in the stated direction with at least half "
        "of the clusters agreeing; adverse = the same against AdaIS; directional = point estimate in the stated "
        "direction, CI includes 0; null = no detectable difference; descriptive = counts or rankings without "
        "inference. Simulation-level precision only: the blocks are simulated decision-maker runs, not people.", "",
        "Claims in this revision: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + ".", "",
        "## Table: evidence map (main text)", "", *rows([c for c in paper if c["main"]]), "",
        "## Table S: evidence map (all claims)", "", *rows(paper), "",
    ]  # fmt: skip
    path.write_text("\n".join(text), encoding="utf-8")


def write_review(out: list[dict], path: Path) -> None:
    paper = [c for c in out if c["paper"]]
    withdrawn = [c for c in out if not c["paper"]]
    counts = pd.Series([STATUS_ZH[c["status"]] for c in paper]).value_counts()
    lines = [
        "# 证据地图 · 审阅版（evidence_claims.py 生成，不要手改）", "",
        "供作者审定结论用。每条结论的数字都由脚本从分析输出读取，状态由固定规则判定（规则写在脚本文件头）。"
        "效应为 AdaIS 的优势（正值对 AdaIS 有利），方括号内是按种子聚类的 95% CI（E6′ 按参数样本），最后是 AdaIS 占优的种子数。", "",
        "**本次修订的结论：** " + "，".join(f"{k} {v} 条" for k, v in counts.items()) + f"；共 {len(paper)} 条。"
        f"另有 {len(withdrawn)} 条原稿说法撤回或改写，见最后一节。", "",
    ]  # fmt: skip
    for layer in LAYERS:
        cs = [c for c in paper if c["layer"] == layer]
        if not cs:
            continue
        lines += [f"## {LAYER_ZH[layer]}", "", "| 编号 | 结论（中文转述） | 状态 | 数字 | 原稿说法（位置） | 改动 | 第三节判定 | 说明 |",
                  "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for c in cs:
            nums = "<br>".join(x for x in (c["effects"].replace(" | ", "<br>"),
                                           ("对照：" + c["context"].replace(" | ", "<br>")) if c["context"] else "",
                                           c["values"].replace(" | ", "<br>")) if x)  # fmt: skip
            lines.append(f"| {c['id']}{'（正文）' if c['main'] else ''} | {c['claim_zh']} | {STATUS_ZH[c['status']]} | "
                         f"{nums} | {c['original']} | {c['change']} | {c['review3']} | {c['note']} |")  # fmt: skip
        lines.append("")
    lines += ["## 原稿说法的撤回与改写（不进论文表）", "", "| 编号 | 原稿说法 | 按规则的状态 | 数字 | 改动 | 第三节判定 | 说明 |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for c in withdrawn:
        nums = "<br>".join(x for x in (c["effects"].replace(" | ", "<br>"), c["values"].replace(" | ", "<br>"),
                                       c["checks"]) if x) or "没有对应证据"  # fmt: skip
        lines.append(f"| {c['id']} | {c['original']} | {STATUS_ZH[c['status']]} | {nums} | {c['change']} | "
                     f"{c['review3']} | {c['note']} |")  # fmt: skip
    lines += ["", "## 与 10-24 审阅第三节判定不同的条目", "",
              "第三节写于按单元格计算 CI 的时候；第八节（按种子聚类，决定 A）在后，以第八节为准。下面是本表与第三节的判定或证据描述不同的条目。", ""]  # fmt: skip
    for c in out:
        if c["vs_review3"]:
            lines.append(
                f"- {c['id']}（本表：{c['change']}，{STATUS_ZH[c['status']]}）：{c['vs_review3']}"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(A / "evidence"))
    out_dir = Path(ap.parse_args().out)
    out_dir.mkdir(parents=True, exist_ok=True)
    claims = json.loads(CLAIMS.read_text(encoding="utf-8"))["claims"]
    ids = [c["id"] for c in claims]
    if len(ids) != len(set(ids)):
        raise ClaimError("duplicate claim ids")
    try:
        out, long = build(load_sources(), claims)
    except ClaimError as e:
        sys.exit(f"evidence_claims: {e}")
    pd.DataFrame(out).to_csv(out_dir / "evidence_map_paper.csv", index=False)
    pd.DataFrame(long).to_csv(out_dir / "evidence_map_rows.csv", index=False)
    write_md(out, out_dir / "evidence_map_paper.md")
    write_review(out, out_dir / "证据地图_审阅版.md")
    paper = pd.DataFrame([c for c in out if c["paper"]])
    print(paper.groupby(["layer", "status"]).size().unstack(fill_value=0).to_string())
    print(
        f"{len(paper)} paper claims ({int(paper['main'].sum())} main), {len(out) - len(paper)} withdrawn -> {out_dir}"
    )


if __name__ == "__main__":
    main()
