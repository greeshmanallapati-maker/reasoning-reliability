from __future__ import annotations

import math

from reasoning_reliability.model import GenerationResult


class UncertaintyEstimator:
    """Likelihood proxy; explicitly not a calibrated probability."""

    def estimate(self, result: GenerationResult) -> tuple[float, str]:
        if result.error or result.timed_out:
            return 1.0, "generation_failure"
        if result.mean_logprob is None:
            return 0.5, "fallback_no_token_logprobs"
        confidence_proxy = math.exp(max(-20.0, min(0.0, result.mean_logprob)))
        return 1.0 - confidence_proxy, "mean_generated_token_likelihood_proxy"


def disagreement_uncertainty(answers: list[str | None]) -> float:
    usable = [answer for answer in answers if answer is not None]
    if not usable:
        return 1.0
    most_common = max(usable.count(answer) for answer in set(usable))
    return 1.0 - most_common / len(answers)

