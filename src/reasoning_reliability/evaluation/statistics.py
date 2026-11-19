from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import binomtest


def bootstrap_accuracy_ci(
    correctness: Iterable[bool], confidence: float = 0.95, samples: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    values = np.asarray(list(correctness), dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    estimates = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return tuple(float(x) for x in np.quantile(estimates, [alpha, 1 - alpha]))


def paired_bootstrap_difference(
    a: Iterable[bool], b: Iterable[bool], samples: int = 10_000, seed: int = 42
) -> tuple[float, float, float]:
    av, bv = np.asarray(list(a), float), np.asarray(list(b), float)
    if len(av) != len(bv) or not len(av):
        raise ValueError("Paired samples must be non-empty and equal length")
    delta = av - bv
    rng = np.random.default_rng(seed)
    estimates = rng.choice(delta, size=(samples, len(delta)), replace=True).mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(delta.mean()), float(low), float(high)


def mcnemar_exact(a: Iterable[bool], b: Iterable[bool]) -> dict[str, float | int]:
    pairs = list(zip(a, b, strict=True))
    a_only = sum(x and not y for x, y in pairs)
    b_only = sum(y and not x for x, y in pairs)
    discordant = a_only + b_only
    p_value = 1.0 if discordant == 0 else binomtest(min(a_only, b_only), discordant, 0.5).pvalue
    return {"a_only_correct": a_only, "b_only_correct": b_only, "p_value": float(p_value)}

