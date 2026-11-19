from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reasoning_reliability.evaluation.metrics import pareto_frontier, summarize_results

COLORS = {
    "direct": "#4C78A8",
    "cot": "#F58518",
    "verification": "#E45756",
    "adaptive": "#72B7B2",
    "adaptive_difficulty": "#54A24B",
    "adaptive_uncertainty": "#B279A2",
}


def _bar(summary: pd.DataFrame, column: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    error = None
    if column == "accuracy" and {"accuracy_ci_low", "accuracy_ci_high"}.issubset(summary.columns):
        error = np.vstack([
            summary.accuracy - summary.accuracy_ci_low,
            summary.accuracy_ci_high - summary.accuracy,
        ])
    ax.bar(
        summary.strategy,
        summary[column],
        yerr=error,
        capsize=4 if error is not None else 0,
        color=[COLORS.get(x, "#999999") for x in summary.strategy],
    )
    ax.set(ylabel=ylabel, title=title)
    if column in {"accuracy", "error_rate", "answer_agreement"}:
        ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_plots(frame: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = pareto_frontier(summarize_results(frame))
    paths: list[Path] = []
    specs = [
        ("accuracy", "Accuracy", "Accuracy by reasoning strategy", "01_accuracy.png"),
        ("error_rate", "Error rate", "Error rate by reasoning strategy", "02_error_rate.png"),
        ("mean_latency_seconds", "Seconds", "Mean end-to-end latency", "03_latency.png"),
        ("mean_total_tokens", "Tokens", "Mean total token consumption", "04_tokens.png"),
        ("answer_agreement", "Agreement", "Repeated-answer consistency", "09_consistency.png"),
    ]
    for column, ylabel, title, filename in specs:
        path = output / filename
        _bar(summary, column, ylabel, title, path)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in summary.iterrows():
        ax.scatter(row.mean_total_tokens, row.accuracy, s=90, color=COLORS.get(row.strategy, "#999"))
        ax.annotate(row.strategy, (row.mean_total_tokens, row.accuracy), xytext=(5, 5), textcoords="offset points")
    ax.set(xlabel="Mean total tokens", ylabel="Accuracy", title="Accuracy versus computational proxy")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = output / "05_accuracy_cost.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    grouped = frame.groupby(["dataset", "difficulty_category", "strategy"])["correct"].mean().unstack(fill_value=0)
    ax = grouped.plot(kind="bar", figsize=(11, 5), color=[COLORS.get(x, "#999") for x in grouped.columns])
    ax.set(ylabel="Accuracy", title="Performance by dataset and difficulty", ylim=(0, 1))
    ax.figure.tight_layout()
    path = output / "06_dataset_difficulty.png"
    ax.figure.savefig(path, dpi=180)
    plt.close(ax.figure)
    paths.append(path)

    adaptive = frame[frame.strategy.str.startswith("adaptive")]
    if not adaptive.empty:
        routing = adaptive["selected_adaptive_route"].value_counts(normalize=True)
        ax = routing.plot(kind="bar", figsize=(7, 5), color="#72B7B2")
        ax.set(ylabel="Fraction of questions", title="Adaptive routing distribution", ylim=(0, 1))
        ax.figure.tight_layout()
        path = output / "07_routing.png"
        ax.figure.savefig(path, dpi=180)
        plt.close(ax.figure)
        paths.append(path)

    verification = frame[frame.verification_outcome.notna()]
    if not verification.empty:
        counts = verification.verification_outcome.value_counts()
        ax = counts.plot(kind="bar", figsize=(8, 5), color="#E45756")
        ax.set(ylabel="Questions", title="Verification benefits and harms")
        ax.figure.tight_layout()
        path = output / "08_verification.png"
        ax.figure.savefig(path, dpi=180)
        plt.close(ax.figure)
        paths.append(path)

    if "confidence_probability" in frame.columns and frame["confidence_probability"].notna().any():
        calibrated = frame.dropna(subset=["confidence_probability"]).copy()
        calibrated["confidence_bin"] = pd.cut(
            calibrated["confidence_probability"], bins=np.linspace(0, 1, 11), include_lowest=True
        )
        reliability = calibrated.groupby("confidence_bin", observed=False).agg(
            confidence=("confidence_probability", "mean"), accuracy=("correct", "mean")
        ).dropna()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")
        ax.plot(reliability.confidence, reliability.accuracy, marker="o", label="Model")
        ax.set(xlabel="Mean confidence", ylabel="Empirical accuracy", title="Confidence calibration", xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        fig.tight_layout()
        path = output / "10_confidence_calibration.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    elif "uncertainty_score" in frame.columns and frame["uncertainty_score"].notna().any():
        proxy = frame.dropna(subset=["uncertainty_score"])
        fig, ax = plt.subplots(figsize=(7, 5))
        for correct, group in proxy.groupby("correct"):
            ax.hist(
                group.uncertainty_score,
                bins=np.linspace(0, 1, 11),
                alpha=0.6,
                label="Correct" if bool(correct) else "Incorrect",
            )
        ax.set(
            xlabel="Uncertainty proxy (not a calibrated probability)",
            ylabel="Records",
            title="Uncertainty-proxy diagnostic",
            xlim=(0, 1),
        )
        ax.legend()
        fig.tight_layout()
        path = output / "10_uncertainty_proxy_diagnostic.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    ablations = summary[summary.strategy.str.startswith("adaptive")]
    if not ablations.empty:
        path = output / "11_adaptive_ablation.png"
        _bar(ablations, "accuracy", "Accuracy", "Adaptive routing ablation", path)
        paths.append(path)

    baseline = summary.set_index("strategy")
    if "direct" in baseline.index:
        delta = summary.copy()
        delta["accuracy_gain"] = delta.accuracy - baseline.loc["direct", "accuracy"]
        delta["extra_tokens"] = delta.mean_total_tokens - baseline.loc["direct", "mean_total_tokens"]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(delta.extra_tokens, delta.accuracy_gain, color=[COLORS.get(x, "#999") for x in delta.strategy])
        for _, row in delta.iterrows():
            ax.annotate(row.strategy, (row.extra_tokens, row.accuracy_gain))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(xlabel="Additional mean tokens vs direct", ylabel="Accuracy change", title="Accuracy improvement versus token cost")
        fig.tight_layout()
        path = output / "12_accuracy_improvement_cost.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    summary.to_csv(output / "summary_with_pareto.csv", index=False)
    return paths

