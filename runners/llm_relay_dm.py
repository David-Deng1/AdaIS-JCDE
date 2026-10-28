"""File-relay decision maker for the out-of-model check (E10'): an external LLM agent answers the queries.

The runner writes each query as JSON to <relay_dir>/q_XXXX.json and blocks until the agent writes
<relay_dir>/a_XXXX.json. Ground truth (w*, r*) is kept only for evaluation and is never shown in the
query files; the agent sees its persona brief separately. Latency and quality signals for the
scheduler still come from the SimulatedDM fatigue model (an LLM's response time has no human
meaning), so this is a hybrid setting and is reported as such.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hitlo.core.simulated_dm import SDMProfile, SDM_PROFILES, SimulatedDM

# Third persona for E10' (between the built-in balanced and edge profiles); used only in E10'.
SDM_PROFILES.setdefault(
    "SAV-AERO_mid",
    SDMProfile(
        name="SAV-AERO_mid",
        problem_id="SAV-AERO",
        n_obj=2,
        w_star=np.array([0.65, 0.35]),
        r_star=np.array([0.08, -9.0]),
        C_h_star=[
            {"slot": "r_0", "value": 0.10, "tolerance": 0.04},
            {"slot": "r_1", "value": -7.0, "tolerance": 2.0},
        ],
        description="SAV-AERO intermediate: drag first, normal force close second",
    ),
)

OBJ_TEXT = {  # objective presentation for SAV-AERO (f2 is stored as -Cn)
    "SAV-AERO": [
        ("Cd0", "zero-lift drag coefficient, lower is better", 1.0),
        ("Cn", "normal-force coefficient, higher is better", -1.0),
    ],
}


def _design(label: str, f: np.ndarray, problem_id: str) -> Dict[str, Any]:
    spec = OBJ_TEXT.get(problem_id)
    if not spec:
        return {
            "id": label,
            **{f"f{k + 1}": round(float(v), 6) for k, v in enumerate(f)},
        }
    return {
        "id": label,
        **{name: round(sign * float(v), 6) for (name, _, sign), v in zip(spec, f)},
    }


class RelayDM(SimulatedDM):
    """SimulatedDM-compatible respondent whose answers come from files written by an external agent."""

    def __init__(
        self,
        *args: Any,
        relay_dir: str,
        problem_id: str,
        poll_s: float = 2.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.relay = Path(relay_dir)
        self.relay.mkdir(parents=True, exist_ok=True)
        self.problem_id = problem_id
        self.poll_s = poll_s
        self._k = 0
        self.transcript: List[Dict[str, Any]] = []

    # ---- relay plumbing ----
    def _ask(self, payload: Dict[str, Any], validate) -> Dict[str, Any]:
        self._k += 1
        q = self.relay / f"q_{self._k:04d}.json"
        a = self.relay / f"a_{self._k:04d}.json"
        payload = {
            "query_id": self._k,
            **payload,
            "answer_file": a.name,
            "answer_format": payload.get("answer_format"),
        }
        q.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        while True:
            if (self.relay / "ABORT").exists():
                raise RuntimeError("relay aborted by operator")
            if a.exists():
                try:
                    ans = json.loads(a.read_text(encoding="utf-8"))
                    err = validate(ans)
                except (json.JSONDecodeError, OSError) as exc:
                    err = f"invalid JSON: {exc}"
                    ans = None
                if not err:
                    self.transcript.append({"query": payload, "answer": ans})
                    return ans
                (self.relay / f"e_{self._k:04d}.txt").write_text(err, encoding="utf-8")
                a.rename(
                    self.relay / f"a_{self._k:04d}_rejected_{int(time.time())}.json"
                )
            time.sleep(self.poll_s)

    # ---- channels ----
    def respond_pairwise(self, x_i, x_j, f_i, f_j) -> int:
        ans = self._ask(
            {
                "channel": "pairwise",
                "task": "Which of the two designs do you prefer?",
                "designs": [
                    _design("A", f_i, self.problem_id),
                    _design("B", f_j, self.problem_id),
                ],
                "answer_format": {"choice": "A or B", "reason": "one short sentence"},
            },
            lambda x: (
                None if x.get("choice") in ("A", "B") else "choice must be 'A' or 'B'"
            ),
        )
        return 0 if ans["choice"] == "A" else 1

    def respond_ranking(self, candidates_F, k: int = 5) -> List[int]:
        F = np.asarray(candidates_F, dtype=float)
        labels = [f"D{i + 1}" for i in range(len(F))]
        k = min(k, len(F))

        def ok(x):
            r = x.get("ranking")
            if (
                not isinstance(r, list)
                or len(r) < k
                or len(set(r[:k])) != k
                or not set(r[:k]) <= set(labels)
            ):
                return f"ranking must list {k} distinct ids from {labels}, best first"
            return None

        ans = self._ask(
            {
                "channel": "ranking",
                "task": f"Rank the designs from most to least preferred (top {k}).",
                "designs": [_design(l, f, self.problem_id) for l, f in zip(labels, F)],
                "answer_format": {
                    "ranking": f"list of {k} design ids, best first",
                    "reason": "short",
                },
            },
            ok,
        )
        return [labels.index(d) for d in ans["ranking"][:k]]

    def respond_threshold(
        self, slot_name: str, proposed_value: float
    ) -> Tuple[bool, float]:
        idx = int(slot_name.split("_")[1]) if "_" in slot_name else 0
        spec = OBJ_TEXT.get(self.problem_id)
        if slot_name.startswith("r_") and spec:
            name, desc, sign = spec[idx]
            shown = sign * float(proposed_value)
            task = (
                f"The system proposes an aspiration level for {name} ({desc}): {shown:.6g}. "
                "Accept it, or give the value you actually aim for."
            )
        else:
            sign = 1.0
            shown = float(proposed_value)
            task = (
                f"The system proposes relative importance {shown:.3f} (0-1) for objective {idx + 1}. "
                "Accept it, or give your value."
            )

        def ok(x):
            if not isinstance(x.get("accept"), bool):
                return "accept must be true or false"
            if not x["accept"] and not isinstance(x.get("value"), (int, float)):
                return "when accept is false, give a numeric value"
            return None

        ans = self._ask(
            {
                "channel": "threshold",
                "slot": slot_name,
                "task": task,
                "answer_format": {
                    "accept": "true/false",
                    "value": "number if not accepted",
                },
            },
            ok,
        )
        if ans["accept"]:
            return True, float(proposed_value)
        return False, float(ans["value"]) * sign

    def respond_roi(self, pop_F, projection_dims=(0, 1)):
        F = np.asarray(pop_F, dtype=float)
        idx = np.linspace(0, len(F) - 1, num=min(25, len(F))).astype(int)
        order = np.argsort(F[:, 0])
        shown = [
            _design(f"P{n + 1}", F[order[i]], self.problem_id)
            for n, i in enumerate(idx)
        ]
        spec = OBJ_TEXT.get(self.problem_id)
        names = [s[0] for s in spec] if spec else ["f1", "f2"]
        fmt = {
            f"{names[0]}_min": "number",
            f"{names[0]}_max": "number",
            f"{names[1]}_min": "number",
            f"{names[1]}_max": "number",
        }

        def ok(x):
            return (
                None
                if all(isinstance(x.get(key), (int, float)) for key in fmt)
                else f"give numbers for {list(fmt)}"
            )

        ans = self._ask(
            {
                "channel": "roi",
                "task": "Mark the region of objective space you are most interested in.",
                "designs": shown,
                "answer_format": fmt,
            },
            ok,
        )
        lo0, hi0 = sorted([ans[f"{names[0]}_min"], ans[f"{names[0]}_max"]])
        lo1, hi1 = sorted([ans[f"{names[1]}_min"], ans[f"{names[1]}_max"]])
        if spec:  # convert back to stored objective signs (f2 = -Cn)
            lo1, hi1 = sorted([spec[1][2] * lo1, spec[1][2] * hi1])
        return (float(lo0), float(hi0), float(lo1), float(hi1))

    def respond_nl_clarify(self, slot_name: str) -> Dict[str, Any]:
        ans = self._ask(
            {
                "channel": "nl_clarify",
                "slot": slot_name,
                "task": "In one or two sentences, clarify your preference about the trade-off between "
                "the objectives (state numbers if you have them).",
                "answer_format": {"text": "free text"},
            },
            lambda x: (
                None
                if isinstance(x.get("text"), str) and x["text"].strip()
                else "text required"
            ),
        )
        return {
            "slot": slot_name,
            "value": None,
            "text": ans["text"],
            "source": "llm_text_ungrounded",
        }


def make_relay_dm(
    profile_key: str,
    problem: Any,
    relay_dir: str,
    problem_id: str,
    sigma_noise: float,
    fatigue_rate: float,
    rng_seed: Optional[int],
) -> RelayDM:
    p = SDM_PROFILES[profile_key]
    return RelayDM(
        w_star=p.w_star.copy(),
        r_star=p.r_star.copy(),
        C_h_star=list(p.C_h_star),
        problem=problem,
        sigma_noise=sigma_noise,
        fatigue_rate=fatigue_rate,
        rng_seed=rng_seed,
        relay_dir=relay_dir,
        problem_id=problem_id,
    )
