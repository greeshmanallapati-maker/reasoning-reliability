from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from .statistics import bootstrap_accuracy_ci


def expected_calibration_error(
    correctness: Iterable[bool], probabilities: Iterable[float], bins: int = 10
) -> float:
    y, p = np.asarray(list(correctness), float), np.asarray(list(probabilities), float)
    if len(y) != len(p) or np.any((p < 0) | (p > 1)):
        raise ValueError("Calibration requires matched probabilities in [0, 1]")
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1])
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def brier_score(correctness: Iterable[bool], probabilities: Iterable[float]) -> float:
    y, p = np.asarray(list(correctness), float), np.asarray(list(probabilities), float)
    if len(y) != len(p) or np.any((p < 0) | (p > 1)):
        raise ValueError("Brier score requires matched probabilities in [0, 1]")
    return float(np.mean((p - y) ** 2))


def summarize_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for strategy, group in frame.groupby("strategy"):
        correct = group["correct"].astype(bool)
        ci_low, ci_high = bootstrap_accuracy_ci(correct, samples=2_000)
        trial_accuracy = (
            group.groupby("trial")["correct"].mean()
            if "trial" in group.columns
            else pd.Series([correct.mean()])
        )
        answers_per_question = group.groupby("question_id")["final_answer"].apply(list)
        repeated = answers_per_question.apply(len).max() > 1
        agreement = (
            answers_per_question.apply(
                lambda xs: max(Counter(str(x) for x in xs).values()) / len(xs)
            ).mean()
            if repeated
            else np.nan
        )
        verification_rows = group[group["verification_outcome"].notna()] if "verification_outcome" in group else pd.DataFrame()
        initial_accuracy = (
            (verification_rows["initial_answer"] == verification_rows["ground_truth"]).mean()
            if not verification_rows.empty
            else np.nan
        )
        rows.append(
            {
                "strategy": strategy,
                "n": len(group),
                "accuracy": correct.mean(),
                "trials": int(group["trial"].nunique()) if "trial" in group else 1,
                "accuracy_std_across_trials": trial_accuracy.std(ddof=0) if len(trial_accuracy) > 1 else np.nan,
                "error_rate": 1 - correct.mean(),
                "accuracy_ci_low": ci_low,
                "accuracy_ci_high": ci_high,
                "parse_failure_rate": group["failure"].fillna("").str.contains("parse|malformed").mean(),
                "timeout_rate": group["timed_out"].astype(bool).mean(),
                "mean_latency_seconds": group["total_latency_seconds"].mean(),
                "median_latency_seconds": group["total_latency_seconds"].median(),
                "p95_latency_seconds": group["total_latency_seconds"].quantile(0.95),
                "mean_input_tokens": group["input_tokens"].mean(),
                "mean_output_tokens": group["output_tokens"].mean(),
                "mean_total_tokens": group["total_tokens"].mean(),
                "mean_reasoning_tokens": group.get("reasoning_tokens", pd.Series([0] * len(group), index=group.index)).mean(),
                "total_generated_tokens": group["output_tokens"].sum(),
                "total_evaluation_seconds": group["total_latency_seconds"].sum(),
                "tokens_per_correct": group["total_tokens"].sum() / max(correct.sum(), 1),
                "tokens_per_second": group["output_tokens"].sum() / max(group["model_call_latency_seconds"].sum(), 1e-9)
                if "model_call_latency_seconds" in group
                else group["output_tokens"].sum() / max(group["total_latency_seconds"].sum(), 1e-9),
                "questions_per_second": len(group) / max(group["total_latency_seconds"].sum(), 1e-9),
                "answer_agreement": agreement,
                "correctness_stability": (
                    group.groupby("question_id")["correct"].mean().apply(
                        lambda rate: max(rate, 1 - rate)
                    ).mean()
                    if repeated
                    else np.nan
                ),
                "peak_gpu_memory_bytes": group["peak_gpu_memory_bytes"].max(),
                "initial_accuracy": initial_accuracy,
                "verification_correction_rate": (
                    (verification_rows["verification_outcome"] == "wrong_corrected_to_right").mean()
                    if not verification_rows.empty
                    else np.nan
                ),
                "harmful_correction_rate": (
                    (verification_rows["verification_outcome"] == "right_changed_to_wrong").mean()
                    if not verification_rows.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def pareto_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    dominated = []
    for _, row in summary.iterrows():
        is_dominated = any(
            (other.accuracy >= row.accuracy and other.mean_total_tokens <= row.mean_total_tokens)
            and (other.accuracy > row.accuracy or other.mean_total_tokens < row.mean_total_tokens)
            for _, other in summary.iterrows()
        )
        dominated.append(is_dominated)
    result = summary.copy()
    result["pareto_efficient"] = np.logical_not(dominated)
    return result

