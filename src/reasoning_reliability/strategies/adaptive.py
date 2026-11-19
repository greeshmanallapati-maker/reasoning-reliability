from __future__ import annotations

from reasoning_reliability.adaptive import DifficultyEstimator, RoutingPolicy, UncertaintyEstimator
from reasoning_reliability.data.schemas import BenchmarkExample

from .base import BaseStrategy, StrategyResult
from .cot import ChainOfThoughtStrategy
from .direct import DirectAnsweringStrategy
from .verification import CoTVerificationStrategy


class AdaptiveReasoningStrategy(BaseStrategy):
    name = "adaptive"

    def __init__(self, engine, policy: RoutingPolicy | None = None, mode: str = "combined") -> None:
        super().__init__(engine)
        self.direct = DirectAnsweringStrategy(engine)
        self.cot = ChainOfThoughtStrategy(engine)
        self.verification = CoTVerificationStrategy(engine)
        self.difficulty = DifficultyEstimator()
        self.uncertainty = UncertaintyEstimator()
        self.policy = policy or RoutingPolicy()
        self.mode = mode

    def run(self, example: BenchmarkExample) -> StrategyResult:
        preliminary = self.direct.run(example)
        return self.run_with_precomputed(example, preliminary)

    def run_with_precomputed(
        self,
        example: BenchmarkExample,
        preliminary: StrategyResult,
        cot_result: StrategyResult | None = None,
        verification_result: StrategyResult | None = None,
    ) -> StrategyResult:
        """Route using existing results when a UI compares all strategies.

        Reusing results preserves the logical token/latency accounting while avoiding
        duplicate model generations that can exhaust Apple MPS memory.
        """
        difficulty, features = self.difficulty.estimate(example)
        uncertainty, method = self.uncertainty.estimate(preliminary.calls[0])
        decision = self.policy.decide(difficulty, uncertainty, self.mode)
        if decision.route == "direct":
            routed = preliminary
        elif decision.route == "cot":
            routed = cot_result or self.cot.run(example)
        else:
            routed = verification_result or self.verification.run(example)
        extra_calls = [] if routed is preliminary else preliminary.calls
        return StrategyResult(
            strategy=self.name,
            final_answer=routed.final_answer,
            initial_answer=routed.initial_answer,
            reasoning=routed.reasoning,
            reasoning_tokens=routed.reasoning_tokens,
            selected_route=decision.route,
            difficulty_score=difficulty,
            difficulty_features=features,
            uncertainty_score=uncertainty,
            uncertainty_method=method,
            routing_explanation=decision.explanation,
            verification_claim=routed.verification_claim,
            verification_explanation=routed.verification_explanation,
            calls=extra_calls + routed.calls,
            failure=routed.failure,
        )

