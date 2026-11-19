import json

from conftest import MockEngine

from reasoning_reliability.evaluation import BenchmarkRunner
from reasoning_reliability.strategies import CoTVerificationStrategy, DirectAnsweringStrategy


def test_exports_resume_and_verification_outcome(tmp_path, math_example):
    strategies = {
        "direct": DirectAnsweringStrategy(MockEngine(["FINAL_ANSWER: 5"])),
        "verification": CoTVerificationStrategy(MockEngine(["work\nFINAL_ANSWER: 6", "VERIFICATION: FAIL\nEXPLANATION: fix\nCORRECTION_NEEDED: YES\nFINAL_ANSWER: 5"])),
    }
    runner = BenchmarkRunner(strategies, str(tmp_path), run_id="test")
    frame = runner.run([math_example])
    assert len(frame) == 2
    assert (tmp_path / "test" / "question_results.csv").exists()
    outcome = frame.loc[frame.strategy == "verification", "verification_outcome"].item()
    assert outcome == "wrong_corrected_to_right"
    # A second runner with empty mocks must skip completed keys.
    runner2 = BenchmarkRunner({"direct": DirectAnsweringStrategy(MockEngine([]))}, str(tmp_path), run_id="test")
    assert len(runner2.run([math_example])) == 2


def test_missing_initial_answer_counts_as_wrong_preserved(tmp_path, math_example):
    strategy = CoTVerificationStrategy(MockEngine([
        "unparseable initial",
        "VERIFICATION: FAIL\nEXPLANATION: still wrong\nCORRECTION_NEEDED: NO\nFINAL_ANSWER: 6",
    ]))
    runner = BenchmarkRunner({"verification": strategy}, str(tmp_path), run_id="missing-initial")
    frame = runner.run([math_example])
    assert frame.verification_outcome.item() == "wrong_preserved"


def test_configurable_batch_runner(tmp_path, math_example):
    second = math_example.__class__("gsm8k:2", "gsm8k", "test", "What is 3 + 4?", "7")
    strategy = DirectAnsweringStrategy(MockEngine(["FINAL_ANSWER: 5", "FINAL_ANSWER: 7"]))
    runner = BenchmarkRunner(
        {"direct": strategy}, str(tmp_path), run_id="batch", batch_size=2
    )
    frame = runner.run([math_example, second])
    assert len(frame) == 2
    assert frame.correct.all()

