from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONSTANTS_PATH = ROOT / "voi_adapter_constants.json"
CONSTANTS = json.loads(CONSTANTS_PATH.read_text(encoding="utf-8"))
PARTICLE_COUNT = int(CONSTANTS["particle_count"])
ARCHIVE_CAP = int(CONSTANTS["archive_cap"])
EWMA_WINDOW = int(CONSTANTS["ewma_window"])
QUERY_COST = float(CONSTANTS["query_cost"])
CONTINUATION_COST = float(CONSTANTS["continuation_cost"])
RHO = float(CONSTANTS["augmented_tchebycheff_rho"])
LOGISTIC_SCALE = float(CONSTANTS["noise_mapping"]["logistic_scale"])


def dirichlet_particles(
    dimension: int,
    *,
    count: int = PARTICLE_COUNT,
    seed: int,
) -> np.ndarray:
    if dimension < 2:
        raise ValueError("preference particles require at least two objectives")
    if count <= 0:
        raise ValueError("particle count must be positive")
    rng = np.random.default_rng(int(seed))
    particles = rng.dirichlet(np.ones(dimension, dtype=float), size=int(count))
    if not np.isfinite(particles).all():
        raise RuntimeError("Dirichlet prior produced non-finite particles")
    return particles


def augmented_tchebycheff_utility(
    normalized_minimization_point: np.ndarray,
    particles: np.ndarray,
    *,
    rho: float = RHO,
) -> np.ndarray:
    point = np.asarray(normalized_minimization_point, dtype=float)
    theta = np.asarray(particles, dtype=float)
    if point.ndim != 1 or theta.ndim != 2 or theta.shape[1] != point.shape[0]:
        raise ValueError("point and particle dimensions are inconsistent")
    if not np.isfinite(point).all() or not np.isfinite(theta).all():
        raise ValueError("utility inputs must be finite")
    weighted_loss = theta * point[None, :]
    # Higher is better: the negative augmented achievement loss is a utility.
    return -(np.max(weighted_loss, axis=1) + float(rho) * np.sum(weighted_loss, axis=1))


def _utility_matrix(
    normalized_archive: np.ndarray,
    particles: np.ndarray,
    *,
    rho: float = RHO,
) -> np.ndarray:
    archive = np.asarray(normalized_archive, dtype=float)
    theta = np.asarray(particles, dtype=float)
    if archive.ndim != 2 or theta.ndim != 2 or archive.shape[1] != theta.shape[1]:
        raise ValueError("archive and particle dimensions are inconsistent")
    weighted_loss = theta[:, None, :] * archive[None, :, :]
    return -(
        np.max(weighted_loss, axis=2) + float(rho) * np.sum(weighted_loss, axis=2)
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


@dataclass
class ParticlePosterior:
    particles: np.ndarray
    weights: np.ndarray
    logistic_scale: float = LOGISTIC_SCALE
    rho: float = RHO

    def __post_init__(self) -> None:
        self.particles = np.asarray(self.particles, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        if self.particles.ndim != 2 or self.weights.shape != (len(self.particles),):
            raise ValueError("particle posterior shapes are inconsistent")
        if self.logistic_scale <= 0:
            raise ValueError("logistic scale must be positive")
        self._normalize_weights()

    @classmethod
    def from_prior(
        cls,
        dimension: int,
        *,
        seed: int,
        count: int = PARTICLE_COUNT,
        logistic_scale: float = LOGISTIC_SCALE,
        rho: float = RHO,
    ) -> "ParticlePosterior":
        particles = dirichlet_particles(dimension, count=count, seed=seed)
        weights = np.full(count, 1.0 / count, dtype=float)
        return cls(particles, weights, logistic_scale=logistic_scale, rho=rho)

    def copy(self) -> "ParticlePosterior":
        return ParticlePosterior(
            self.particles.copy(),
            self.weights.copy(),
            logistic_scale=self.logistic_scale,
            rho=self.rho,
        )

    def _normalize_weights(self) -> None:
        if not np.isfinite(self.weights).all() or np.any(self.weights < 0):
            raise ValueError("posterior weights must be finite and nonnegative")
        total = float(self.weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("posterior weight mass collapsed")
        self.weights /= total

    @property
    def effective_sample_size(self) -> float:
        denominator = float(np.sum(np.square(self.weights)))
        return float(1.0 / denominator)

    def pair_likelihood(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        first_utility = augmented_tchebycheff_utility(first, self.particles, rho=self.rho)
        second_utility = augmented_tchebycheff_utility(second, self.particles, rho=self.rho)
        likelihood = _sigmoid((first_utility - second_utility) / self.logistic_scale)
        return np.clip(likelihood, 1e-12, 1.0 - 1e-12)

    def preference_probability(self, first: np.ndarray, second: np.ndarray) -> float:
        return float(np.dot(self.weights, self.pair_likelihood(first, second)))

    def update(self, first: np.ndarray, second: np.ndarray, *, preferred: int) -> None:
        if preferred not in {0, 1}:
            raise ValueError("preferred must be 0 (first) or 1 (second)")
        first_likelihood = self.pair_likelihood(first, second)
        likelihood = first_likelihood if preferred == 0 else 1.0 - first_likelihood
        self.weights *= likelihood
        self._normalize_weights()

    def expected_utilities(self, normalized_archive: np.ndarray) -> np.ndarray:
        utility = _utility_matrix(normalized_archive, self.particles, rho=self.rho)
        return self.weights @ utility

    def expected_recommendation_utility(self, normalized_archive: np.ndarray) -> float:
        expected = self.expected_utilities(normalized_archive)
        if len(expected) == 0:
            raise ValueError("recommendation utility requires a nonempty archive")
        return float(np.max(expected))


def pair_query_voi(
    posterior: ParticlePosterior,
    normalized_archive: np.ndarray,
    pair: tuple[int, int],
) -> float:
    archive = np.asarray(normalized_archive, dtype=float)
    first_index, second_index = pair
    if first_index == second_index or min(pair) < 0 or max(pair) >= len(archive):
        raise ValueError("invalid archive pair")
    utility = _utility_matrix(archive, posterior.particles, rho=posterior.rho)
    current_best = float(np.max(posterior.weights @ utility))
    likelihood = posterior.pair_likelihood(archive[first_index], archive[second_index])
    probability_first = float(np.dot(posterior.weights, likelihood))
    probability_first = float(np.clip(probability_first, 1e-12, 1.0 - 1e-12))

    weights_first = posterior.weights * likelihood
    weights_first /= weights_first.sum()
    weights_second = posterior.weights * (1.0 - likelihood)
    weights_second /= weights_second.sum()
    best_first = float(np.max(weights_first @ utility))
    best_second = float(np.max(weights_second @ utility))
    expected_after = probability_first * best_first + (1.0 - probability_first) * best_second
    # Convexity makes this nonnegative in exact arithmetic; clip round-off only.
    return float(max(0.0, expected_after - current_best))


def select_best_pair(
    posterior: ParticlePosterior,
    normalized_archive: np.ndarray,
) -> tuple[tuple[int, int] | None, float]:
    archive = np.asarray(normalized_archive, dtype=float)
    if len(archive) < 2:
        return None, 0.0
    best_pair: tuple[int, int] | None = None
    best_value = -np.inf
    for pair in combinations(range(len(archive)), 2):
        value = pair_query_voi(posterior, archive, pair)
        if value > best_value:
            best_pair = pair
            best_value = value
    if best_pair is None or not np.isfinite(best_value):
        raise RuntimeError("could not select a finite VoI pair")
    return best_pair, float(max(0.0, best_value))


def _nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    n = len(objectives)
    keep = np.ones(n, dtype=bool)
    for index in range(n):
        dominated = np.all(objectives <= objectives[index], axis=1) & np.any(
            objectives < objectives[index], axis=1
        )
        dominated[index] = False
        if dominated.any():
            keep[index] = False
    return keep


def _crowding_distance(objectives: np.ndarray) -> np.ndarray:
    n, dimension = objectives.shape
    if n <= 2:
        return np.full(n, np.inf)
    distance = np.zeros(n, dtype=float)
    for objective in range(dimension):
        order = np.argsort(objectives[:, objective], kind="stable")
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = float(objectives[order[-1], objective] - objectives[order[0], objective])
        if span > 0:
            interior = order[1:-1]
            distance[interior] += (
                objectives[order[2:], objective] - objectives[order[:-2], objective]
            ) / span
    return distance


def feasible_nondominated_archive(
    objectives: np.ndarray,
    *,
    feasible: np.ndarray | None = None,
    cap: int = ARCHIVE_CAP,
) -> np.ndarray:
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2:
        raise ValueError("objectives must be a two-dimensional array")
    if feasible is None:
        mask = np.ones(len(values), dtype=bool)
    else:
        mask = np.asarray(feasible, dtype=bool)
        if mask.shape != (len(values),):
            raise ValueError("feasible mask length does not match objectives")
    mask &= np.isfinite(values).all(axis=1)
    values = values[mask]
    if len(values) == 0:
        return np.empty((0, objectives.shape[1]), dtype=float)
    # Stable de-duplication keeps the first observed representative.
    seen: set[tuple[float, ...]] = set()
    unique = []
    for row in values:
        key = tuple(float(value) for value in row)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    archive = np.asarray(unique, dtype=float)
    archive = archive[_nondominated_mask(archive)]
    if len(archive) > cap:
        distance = _crowding_distance(archive)
        selected = sorted(range(len(archive)), key=lambda i: (-distance[i], i))[:cap]
        archive = archive[np.sort(selected)]
    return archive


def normalize_archive(archive: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(archive, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("normalization requires a nonempty two-dimensional archive")
    if not np.isfinite(values).all():
        raise ValueError("archive contains non-finite objectives")
    ideal = np.min(values, axis=0)
    nadir = np.max(values, axis=0)
    span = nadir - ideal
    normalized = np.divide(
        values - ideal,
        span,
        out=np.zeros_like(values, dtype=float),
        where=span > 0,
    )
    return np.clip(normalized, 0.0, 1.0), ideal, nadir


class ContinueValueTracker:
    def __init__(self, window: int = EWMA_WINDOW) -> None:
        if window <= 0:
            raise ValueError("EWMA window must be positive")
        self.window = int(window)
        self.increments: deque[float] = deque(maxlen=self.window)
        self._previous: float | None = None

    def observe(self, expected_best_utility: float) -> float:
        value = float(expected_best_utility)
        if not np.isfinite(value):
            raise ValueError("continue-value observation must be finite")
        increment = 0.0 if self._previous is None else max(0.0, value - self._previous)
        self._previous = value
        self.increments.append(increment)
        data = np.asarray(self.increments, dtype=float)
        alpha = 2.0 / (self.window + 1.0)
        ages = np.arange(len(data) - 1, -1, -1, dtype=float)
        weights = np.power(1.0 - alpha, ages)
        return float(np.dot(data, weights) / weights.sum())


@dataclass(frozen=True)
class QueryDecision:
    query: bool
    reason: str
    query_rate: float
    continue_rate: float


def should_query(
    query_value: float,
    continue_value: float,
    *,
    tick: int,
    warmup_ticks: int,
    budget_remaining: float,
    paused: bool = False,
    already_queried: bool = False,
    query_cost: float = QUERY_COST,
    continuation_cost: float = CONTINUATION_COST,
) -> QueryDecision:
    query_value = float(query_value)
    continue_value = float(continue_value)
    if (
        not np.isfinite(query_value)
        or not np.isfinite(continue_value)
        or query_value < 0
        or continue_value < 0
    ):
        return QueryDecision(False, "invalid_value", np.nan, np.nan)
    if query_cost <= 0 or continuation_cost <= 0:
        raise ValueError("query and continuation costs must be positive")
    query_rate = query_value / query_cost
    continue_rate = continue_value / continuation_cost
    if tick < warmup_ticks:
        return QueryDecision(False, "warmup", query_rate, continue_rate)
    if budget_remaining <= 0:
        return QueryDecision(False, "budget", query_rate, continue_rate)
    if paused:
        return QueryDecision(False, "quality_pause", query_rate, continue_rate)
    if already_queried:
        return QueryDecision(False, "one_query_per_tick", query_rate, continue_rate)
    if query_rate > continue_rate:
        return QueryDecision(True, "query_value_exceeds_continue_value", query_rate, continue_rate)
    return QueryDecision(False, "continue_value_not_lower", query_rate, continue_rate)


def make_decision_record(
    *,
    tick: int,
    decision: QueryDecision,
    query_value: float,
    continue_value: float,
    selected_pair: tuple[int, int] | None,
    posterior: ParticlePosterior,
    archive_size: int,
) -> dict[str, object]:
    values = [query_value, continue_value, posterior.effective_sample_size]
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("decision log values must be finite")
    return {
        "tick": int(tick),
        "decision": "query" if decision.query else "continue",
        "decision_reason": decision.reason,
        "query_value": float(query_value),
        "continue_value": float(continue_value),
        "query_rate": float(decision.query_rate),
        "continue_rate": float(decision.continue_rate),
        "selected_pair": list(selected_pair) if selected_pair is not None else None,
        "posterior_ess": float(posterior.effective_sample_size),
        "archive_size": int(archive_size),
    }
