import pandas as pd
import pytest

from reasoning_reliability.evaluation.metrics import brier_score, expected_calibration_error, pareto_frontier, summarize_results
from reasoning_reliability.evaluation.statistics import bootstrap_accuracy_ci, mcnemar_exact


def test_bootstrap_ci_and_mcnemar():
    low, high = bootstrap_accuracy_ci([True, True, False], samples=500)
    assert 0 <= low <= high <= 1
    assert 0 <= mcnemar_exact([True, False], [False, False])["p_value"] <= 1


def test_calibration_validation():
    assert expected_calibration_error([True, False], [0.8, 0.2], bins=2) >= 0
    assert brier_score([True, False], [0.8, 0.2]) == pytest.approx(0.04)


def test_summary_and_pareto():
    frame = pd.DataFrame([
        {"strategy": "a", "question_id": "1", "final_answer": "x", "correct": True, "failure": None, "timed_out": False, "total_latency_seconds": 1, "input_tokens": 5, "output_tokens": 5, "total_tokens": 10, "peak_gpu_memory_bytes": 0},
        {"strategy": "b", "question_id": "1", "final_answer": "y", "correct": False, "failure": None, "timed_out": False, "total_latency_seconds": 2, "input_tokens": 10, "output_tokens": 10, "total_tokens": 20, "peak_gpu_memory_bytes": 0},
    ])
    summary = pareto_frontier(summarize_results(frame))
    assert summary.loc[summary.strategy == "a", "pareto_efficient"].item()
    assert not summary.loc[summary.strategy == "b", "pareto_efficient"].item()

