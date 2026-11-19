#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from itertools import combinations

from reasoning_reliability.analysis import computed_insights, generate_plots
from reasoning_reliability.evaluation.metrics import pareto_frontier, summarize_results
from reasoning_reliability.evaluation.statistics import mcnemar_exact, paired_bootstrap_difference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Result run directory containing question_results.csv")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    source = run_dir / "question_results.csv"
    if not source.exists():
        raise SystemExit(f"No benchmark results at {source}")
    frame = pd.read_csv(source)
    paths = generate_plots(frame, run_dir / "figures")
    summary = pareto_frontier(summarize_results(frame))
    if "direct" in set(summary.strategy):
        direct = summary.set_index("strategy").loc["direct"]
        summary["relative_latency_vs_direct"] = summary.mean_latency_seconds / max(direct.mean_latency_seconds, 1e-9)
        summary["relative_tokens_vs_direct"] = summary.mean_total_tokens / max(direct.mean_total_tokens, 1e-9)
        summary["accuracy_change_vs_direct"] = summary.accuracy - direct.accuracy
    summary.to_csv(run_dir / "strategy_comparison.csv", index=False)
    paired_rows = []
    collapsed = frame.groupby(["strategy", "question_id"], as_index=False)["correct"].mean()
    collapsed["correct"] = collapsed["correct"] >= 0.5
    for first, second in combinations(sorted(collapsed.strategy.unique()), 2):
        a = collapsed[collapsed.strategy == first].set_index("question_id")["correct"]
        b = collapsed[collapsed.strategy == second].set_index("question_id")["correct"]
        common = a.index.intersection(b.index)
        if len(common):
            delta, low, high = paired_bootstrap_difference(a.loc[common], b.loc[common])
            paired_rows.append({"strategy_a": first, "strategy_b": second, "n": len(common),
                                "accuracy_delta_a_minus_b": delta, "ci_low": low, "ci_high": high,
                                **mcnemar_exact(a.loc[common], b.loc[common])})
    pd.DataFrame(paired_rows).to_csv(run_dir / "paired_comparisons.csv", index=False)
    (run_dir / "insights.md").write_text(
        "# Computed research insights\n\n" + "\n".join(f"- {x}" for x in computed_insights(frame)) + "\n"
    )
    calibration_note = (
        "Probability calibration was not computed because the recorded uncertainty signal is a "
        "generated-token likelihood proxy, not a validated probability."
        if "confidence_probability" not in frame or frame["confidence_probability"].isna().all()
        else "Probability calibration inputs were available; see figure 10."
    )
    report_lines = [
        "# Run-specific results report",
        "",
        f"Records: {len(frame)}",
        f"Questions: {frame.question_id.nunique()}",
        f"Datasets: {', '.join(sorted(frame.dataset.unique()))}",
        "",
        "## Computed insights",
        "",
        *[f"- {item}" for item in computed_insights(frame)],
        "",
        "## Calibration status",
        "",
        calibration_note,
        "",
        "## Interpretation guardrail",
        "",
        "These findings apply only to this recorded model configuration and question sample. "
        "A smoke run is an engineering validation, not a definitive hypothesis test.",
    ]
    (run_dir / "results_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Generated {len(paths)} figures in {run_dir / 'figures'}")


if __name__ == "__main__":
    main()

