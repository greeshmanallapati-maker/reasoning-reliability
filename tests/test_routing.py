from conftest import MockEngine

from reasoning_reliability.adaptive import DifficultyEstimator, RoutingPolicy, UncertaintyEstimator
from reasoning_reliability.adaptive.routing import select_thresholds_on_development
from reasoning_reliability.model import GenerationResult
from reasoning_reliability.strategies import (
    AdaptiveReasoningStrategy,
    ChainOfThoughtStrategy,
    CoTVerificationStrategy,
    DirectAnsweringStrategy,
)


def test_threshold_routes():
    policy = RoutingPolicy(0.3, 0.7, 0.5)
    assert policy.decide(0.1, 0.1).route == "direct"
    assert policy.decide(0.5, 0.5).route == "cot"
    assert policy.decide(0.9, 0.9).route == "verification"


def test_difficulty_is_bounded(math_example):
    score, features = DifficultyEstimator().estimate(math_example)
    assert 0 <= score <= 1 and features["numeric_count"] == 2


def test_uncertainty_fallback():
    score, method = UncertaintyEstimator().estimate(GenerationResult(prompt="x"))
    assert score == 0.5 and method == "fallback_no_token_logprobs"


def test_adaptive_reuses_direct_call(math_example):
    engine = MockEngine(["FINAL_ANSWER: 5"], logprob=-0.001)
    strategy = AdaptiveReasoningStrategy(engine, RoutingPolicy(0.9, 0.95))
    result = strategy.run(math_example)
    assert result.selected_route == "direct" and len(result.calls) == 1


def test_adaptive_counts_preliminary_overhead(math_example):
    engine = MockEngine(["FINAL_ANSWER: 5", "work\nFINAL_ANSWER: 5"], logprob=-2)
    strategy = AdaptiveReasoningStrategy(engine, RoutingPolicy(0.0, 1.0))
    result = strategy.run(math_example)
    assert result.selected_route == "cot" and result.total_tokens == 30


def test_adaptive_compare_reuses_precomputed_results(math_example):
    engine = MockEngine([
        "FINAL_ANSWER: 5",
        "work\nFINAL_ANSWER: 5",
        "VERIFICATION: PASS\nEXPLANATION: sound\nCORRECTION_NEEDED: NO\nFINAL_ANSWER: 5",
    ])
    direct = DirectAnsweringStrategy(engine).run(math_example)
    cot = ChainOfThoughtStrategy(engine).run(math_example)
    verification = CoTVerificationStrategy(engine).run_from_initial(math_example, cot)
    strategy = AdaptiveReasoningStrategy(engine, RoutingPolicy(0.0, 0.0))
    result = strategy.run_with_precomputed(math_example, direct, cot, verification)
    assert result.selected_route == "verification"
    assert result.final_answer == "5"
    assert len(engine.prompts) == 3


def test_cost_sensitive_threshold_selection():
    policy, report = select_thresholds_on_development(
        [0.1, 0.5, 0.9], [0.1, 0.5, 0.9],
        {"direct": [True, False, False], "cot": [True, True, False], "verification": [True, True, True]},
        {"direct": [1, 1, 1], "cot": [2, 2, 2], "verification": [3, 3, 3]},
        cost_lambda=0.1,
    )
    assert 0 <= policy.low <= policy.high <= 1
    assert "utility" in report

