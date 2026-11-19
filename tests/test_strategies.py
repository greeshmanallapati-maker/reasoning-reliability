from conftest import MockEngine

from reasoning_reliability.strategies import ChainOfThoughtStrategy, CoTVerificationStrategy, DirectAnsweringStrategy


def test_direct_one_call_and_malformed(math_example):
    engine = MockEngine(["the answer is 5"])
    result = DirectAnsweringStrategy(engine).run(math_example)
    assert result.final_answer is None
    assert result.failure == "missing_or_malformed_final_answer"
    assert len(result.calls) == 1


def test_cot_separates_reasoning(math_example):
    result = ChainOfThoughtStrategy(MockEngine(["2+3=5.\nFINAL_ANSWER: 5"])).run(math_example)
    assert result.reasoning == "2+3=5."
    assert result.reasoning_tokens == 1
    assert result.final_answer == "5"


def test_verification_preserves_correct(math_example):
    engine = MockEngine(["2+3=5\nFINAL_ANSWER: 5", "VERIFICATION: PASS\nEXPLANATION: sound\nCORRECTION_NEEDED: NO\nFINAL_ANSWER: 5"])
    result = CoTVerificationStrategy(engine).run(math_example)
    assert result.initial_answer == result.final_answer == "5"
    assert result.total_tokens == 30 and len(engine.prompts) == 2


def test_verification_corrects_wrong(math_example):
    engine = MockEngine(["bad math\nFINAL_ANSWER: 6", "VERIFICATION: FAIL\nEXPLANATION: arithmetic\nCORRECTION_NEEDED: YES\nFINAL_ANSWER: 5"])
    result = CoTVerificationStrategy(engine).run(math_example)
    assert result.initial_answer == "6" and result.final_answer == "5"


def test_verification_falls_back_to_initial(math_example):
    engine = MockEngine(["work\nFINAL_ANSWER: 5", "unstructured verifier response"])
    result = CoTVerificationStrategy(engine).run(math_example)
    assert result.final_answer == "5" and result.verification_claim == "UNPARSEABLE"


def test_verification_can_reuse_existing_cot(math_example):
    engine = MockEngine([
        "work\nFINAL_ANSWER: 5",
        "VERIFICATION: PASS\nEXPLANATION: sound\nCORRECTION_NEEDED: NO\nFINAL_ANSWER: 5",
    ])
    cot = ChainOfThoughtStrategy(engine).run(math_example)
    result = CoTVerificationStrategy(engine).run_from_initial(math_example, cot)
    assert result.final_answer == "5"
    assert len(engine.prompts) == 2
    assert len(result.calls) == 2

