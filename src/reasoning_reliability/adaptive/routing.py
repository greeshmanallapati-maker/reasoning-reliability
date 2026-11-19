from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Route = Literal["direct", "cot", "verification"]


@dataclass
class RoutingDecision:
    route: Route
    combined_score: float
    explanation: str


class RoutingPolicy:
    def __init__(
        self, low_threshold: float = 0.35, high_threshold: float = 0.70, uncertainty_weight: float = 0.5
    ) -> None:
        if not 0 <= low_threshold <= high_threshold <= 1:
            raise ValueError("Thresholds must satisfy 0 <= low <= high <= 1")
        if not 0 <= uncertainty_weight <= 1:
            raise ValueError("uncertainty_weight must be in [0, 1]")
        self.low = low_threshold
        self.high = high_threshold
        self.weight = uncertainty_weight

    def decide(self, difficulty: float, uncertainty: float, mode: str = "combined") -> RoutingDecision:
        if mode == "difficulty_only":
            score = difficulty
        elif mode == "uncertainty_only":
            score = uncertainty
        elif mode == "combined":
            score = (1 - self.weight) * difficulty + self.weight * uncertainty
        else:
            raise ValueError(f"Unknown routing mode: {mode}")
        route: Route = "direct" if score < self.low else "cot" if score < self.high else "verification"
        return RoutingDecision(
            route=route,
            combined_score=score,
            explanation=(
                f"mode={mode}; difficulty={difficulty:.3f}; uncertainty={uncertainty:.3f}; "
                f"combined={score:.3f}; thresholds=({self.low:.3f}, {self.high:.3f})"
            ),
        )


def select_thresholds_on_development(
    difficulty: list[float],
    uncertainty: list[float],
    route_correctness: dict[Route, list[bool]],
    route_costs: dict[Route, list[float]],
    cost_lambda: float = 0.1,
    uncertainty_weight: float = 0.5,
    grid: tuple[float, ...] = tuple(np.linspace(0.1, 0.9, 9)),
) -> tuple[RoutingPolicy, dict[str, float]]:
    """Select thresholds using only development predictions and min-max normalized cost."""
    n = len(difficulty)
    if len(uncertainty) != n or any(len(values) != n for values in route_correctness.values()):
        raise ValueError("All development arrays must have equal length")
    if any(len(values) != n for values in route_costs.values()):
        raise ValueError("All development cost arrays must have equal length")
    all_costs = np.asarray([value for values in route_costs.values() for value in values], float)
    span = float(all_costs.max() - all_costs.min()) if all_costs.size else 0.0
    best: tuple[float, float, float, float, float] | None = None
    for low in grid:
        for high in grid:
            if high < low:
                continue
            policy = RoutingPolicy(low, high, uncertainty_weight)
            correctness, costs = [], []
            for i, (d, u) in enumerate(zip(difficulty, uncertainty, strict=True)):
                route = policy.decide(d, u).route
                correctness.append(float(route_correctness[route][i]))
                costs.append(float(route_costs[route][i]))
            normalized = [(cost - all_costs.min()) / span if span else 0.0 for cost in costs]
            accuracy = float(np.mean(correctness))
            mean_cost = float(np.mean(normalized))
            utility = accuracy - cost_lambda * mean_cost
            candidate = (utility, accuracy, -mean_cost, low, high)
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    utility, accuracy, negative_cost, low, high = best
    return RoutingPolicy(low, high, uncertainty_weight), {
        "utility": utility,
        "development_accuracy": accuracy,
        "normalized_cost": -negative_cost,
        "low_threshold": low,
        "high_threshold": high,
    }

