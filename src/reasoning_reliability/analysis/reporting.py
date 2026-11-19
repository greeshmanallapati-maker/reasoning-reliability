from __future__ import annotations

import pandas as pd

from reasoning_reliability.evaluation.metrics import summarize_results


def computed_insights(frame: pd.DataFrame) -> list[str]:
    summary = summarize_results(frame)
    if summary.empty:
        return ["No completed benchmark results are available."]
    accurate = summary.loc[summary.accuracy.idxmax()]
    fastest = summary.loc[summary.mean_latency_seconds.idxmin()]
    cheapest = summary.loc[summary.mean_total_tokens.idxmin()]
    insights = [
        f"Highest measured accuracy: {accurate.strategy} ({accurate.accuracy:.1%}).",
        f"Lowest mean latency: {fastest.strategy} ({fastest.mean_latency_seconds:.3f}s).",
        f"Fewest mean tokens: {cheapest.strategy} ({cheapest.mean_total_tokens:.1f}).",
    ]
    verification = frame[frame.strategy == "verification"]
    if not verification.empty:
        outcomes = verification.verification_outcome.value_counts()
        insights.append(
            "Verification corrected "
            f"{outcomes.get('wrong_corrected_to_right', 0)} wrong answers and harmed "
            f"{outcomes.get('right_changed_to_wrong', 0)} initially correct answers."
        )
    return insights

