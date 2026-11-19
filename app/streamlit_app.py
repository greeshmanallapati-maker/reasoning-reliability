from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

# Set before importing Streamlit or any package that may import huggingface_hub.
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasoning_reliability.adaptive import DifficultyEstimator, RoutingPolicy, UncertaintyEstimator  # noqa: E402
from reasoning_reliability.analysis import computed_insights  # noqa: E402
from reasoning_reliability.data.schemas import BenchmarkExample  # noqa: E402
from reasoning_reliability.evaluation import summarize_results  # noqa: E402
from reasoning_reliability.model import HuggingFaceEngine  # noqa: E402
from reasoning_reliability.strategies import (  # noqa: E402
    AdaptiveReasoningStrategy,
    ChainOfThoughtStrategy,
    CoTVerificationStrategy,
    DirectAnsweringStrategy,
)
from reasoning_reliability.utils.config import load_config  # noqa: E402
from reasoning_reliability.utils.presentation import chatbot_response, raw_model_response, technical_payload  # noqa: E402

st.set_page_config(page_title="Reasoning Reliability Lab", layout="wide")
st.title("Does Chain-of-Thought Actually Improve Decision Reliability?")


@st.cache_resource
def load_engine(config_path: str) -> HuggingFaceEngine:
    config = load_config(config_path)
    return HuggingFaceEngine(config["model"], config["generation"])


def engine_or_stop(config_path: str) -> HuggingFaceEngine:
    try:
        return load_engine(config_path)
    except Exception as exc:
        st.error(
            "The Hugging Face model could not be loaded. Public models use anonymous "
            "Hub access by default, so an expired cached login should not be sent. "
            "If you configured `hub_auth: authenticated`, refresh it with `hf auth login`."
        )
        st.code(f"{type(exc).__name__}: {exc}")
        st.stop()


def make_example(task: str, question: str, choices: list[str]) -> BenchmarkExample:
    if task == "GSM8K-style math":
        return BenchmarkExample("interactive", "gsm8k", "interactive", question, "unknown")
    mapping = {chr(65 + i): value for i, value in enumerate(choices) if value.strip()}
    return BenchmarkExample("interactive", "arc_challenge", "interactive", question, "unknown", mapping)


profiles = {
    "Cloud · Qwen3-0.6B (CPU)": ROOT / "configs" / "demo_cloud.yaml",
    "Local · Qwen3-4B": ROOT / "configs" / "demo.yaml",
    "Local · Qwen3-1.7B": ROOT / "configs" / "demo_light.yaml",
}
default_profile = "Local · Qwen3-4B" if platform.system() == "Darwin" else "Cloud · Qwen3-0.6B (CPU)"
profile = st.sidebar.selectbox(
    "Model profile",
    list(profiles),
    index=list(profiles).index(default_profile),
    help="Community Cloud uses the compact CPU model; Apple Silicon defaults to the stronger local 4B model.",
)
config_path = str(profiles[profile])
st.sidebar.caption(f"Configuration: `{profiles[profile].relative_to(ROOT)}`")
tabs = st.tabs(["Ask a Question", "Compare All", "Benchmark Dashboard", "Research Insights"])

with tabs[0]:
    task = st.selectbox("Task type", ["GSM8K-style math", "ARC-style multiple choice"])
    if task == "GSM8K-style math":
        st.caption("Enter a math word problem whose final answer is a number.")
        question = st.text_area("Question", placeholder="Example: A box has 6 rows of 17 pencils. How many pencils are there?")
    else:
        st.caption("Enter a science or commonsense question and provide at least two answer choices.")
        question = st.text_area("Question", placeholder="Example: Which planet is known as the Red Planet?")
    choices = [st.text_input(f"Choice {chr(65+i)}") for i in range(4)] if task.startswith("ARC") else []
    selected = st.selectbox("Strategy", ["adaptive", "direct", "cot", "verification"])
    if st.button("Run", type="primary") and question.strip():
        if task.startswith("ARC") and sum(bool(choice.strip()) for choice in choices) < 2:
            st.error("ARC questions require at least two non-empty answer choices.")
            st.stop()
        engine = engine_or_stop(config_path)
        mapping = {
            "direct": DirectAnsweringStrategy(engine),
            "cot": ChainOfThoughtStrategy(engine),
            "verification": CoTVerificationStrategy(engine),
            "adaptive": AdaptiveReasoningStrategy(engine, RoutingPolicy()),
        }
        result = mapping[selected].run(make_example(task, question, choices))
        with st.chat_message("assistant"):
            st.markdown(chatbot_response(result))
            st.caption(
                f"Route: {result.selected_route or selected} · "
                f"{result.latency_seconds:.2f}s · {result.total_tokens} tokens"
            )
        if result.final_answer is None and result.calls and result.calls[-1].text.strip():
            st.info(
                "The response is shown exactly as generated. It did not use the strict "
                "`FINAL_ANSWER:` research delimiter, so it would count as a parse failure "
                "in benchmark scoring."
            )
        if st.toggle("Show JSON output", value=False, key="ask_json_output"):
            st.json(technical_payload(result))
        st.caption("The app does not claim correctness without an independent reference answer.")

with tabs[1]:
    compare_task = st.selectbox(
        "Comparison task type",
        ["GSM8K-style math", "ARC-style multiple choice"],
        key="compare_task",
    )
    compare_question = st.text_area("Question to compare", key="compare")
    compare_choices = (
        [st.text_input(f"Comparison choice {chr(65+i)}", key=f"compare_choice_{i}") for i in range(4)]
        if compare_task.startswith("ARC")
        else []
    )
    if st.button("Compare strategies") and compare_question.strip():
        if compare_task.startswith("ARC") and sum(bool(choice.strip()) for choice in compare_choices) < 2:
            st.error("ARC comparisons require at least two non-empty choices.")
            st.stop()
        engine = engine_or_stop(config_path)
        example = make_example(compare_task, compare_question, compare_choices)
        direct_strategy = DirectAnsweringStrategy(engine)
        cot_strategy = ChainOfThoughtStrategy(engine)
        verification_strategy = CoTVerificationStrategy(engine)
        adaptive_strategy = AdaptiveReasoningStrategy(engine)

        # Compute each distinct model response once. Adaptive routing then reuses
        # those results, which prevents duplicate MPS generations during comparison.
        with st.status("Running four reasoning strategies…", expanded=False) as status:
            direct_result = direct_strategy.run(example)
            cot_result = cot_strategy.run(example)
            verification_result = verification_strategy.run_from_initial(example, cot_result)
            adaptive_result = adaptive_strategy.run_with_precomputed(
                example,
                direct_result,
                cot_result=cot_result,
                verification_result=verification_result,
            )
            status.update(label="Comparison complete", state="complete")

        results = [
            (direct_strategy.name, direct_result),
            (cot_strategy.name, cot_result),
            (verification_strategy.name, verification_result),
            (adaptive_strategy.name, adaptive_result),
        ]
        rows = []
        details = {}
        difficulty, _ = DifficultyEstimator().estimate(example)
        uncertainty_estimator = UncertaintyEstimator()
        for strategy_name, result in results:
            uncertainty = result.uncertainty_score
            if uncertainty is None and result.calls:
                uncertainty, _ = uncertainty_estimator.estimate(result.calls[0])
            rows.append({
                "strategy": strategy_name,
                "answer": result.final_answer or raw_model_response(result),
                "latency_seconds": round(result.latency_seconds, 3),
                "total_tokens": result.total_tokens,
                "difficulty": round(result.difficulty_score or difficulty, 3),
                "uncertainty_proxy": round(uncertainty, 3) if uncertainty is not None else None,
                "route": result.selected_route or strategy_name,
                "verification": result.verification_claim,
            })
            details[strategy_name] = technical_payload(result)
        st.dataframe(pd.DataFrame(rows), width="stretch")
        with st.expander("Strategy details and reasoning"):
            st.json(details)

with tabs[2]:
    result_files = sorted((ROOT / "results").glob("*/question_results.csv"), reverse=True)
    if not result_files:
        st.info("No real benchmark results are available yet.")
    else:
        selected_file = st.selectbox("Benchmark run", result_files, format_func=lambda p: p.parent.name)
        frame = pd.read_csv(selected_file)
        dataset_filter = st.multiselect("Datasets", sorted(frame.dataset.unique()), default=sorted(frame.dataset.unique()))
        strategy_filter = st.multiselect(
            "Strategies", sorted(frame.strategy.unique()), default=sorted(frame.strategy.unique())
        )
        filtered = frame[frame.dataset.isin(dataset_filter) & frame.strategy.isin(strategy_filter)]
        if filtered.empty:
            st.warning("No rows match the selected filters.")
        else:
            summary = summarize_results(filtered)
            best = summary.loc[summary.accuracy.idxmax()]
            fastest = summary.loc[summary.mean_latency_seconds.idxmin()]
            cheapest = summary.loc[summary.mean_total_tokens.idxmin()]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Evaluations", len(filtered))
            c2.metric("Best accuracy", f"{best.accuracy:.1%}", best.strategy)
            c3.metric("Fastest", f"{fastest.mean_latency_seconds:.2f}s", fastest.strategy)
            c4.metric("Fewest tokens", f"{cheapest.mean_total_tokens:.1f}", cheapest.strategy)
            chart_data = summary.set_index("strategy")
            left, right = st.columns(2)
            left.subheader("Accuracy")
            left.bar_chart(chart_data[["accuracy"]])
            right.subheader("Latency and tokens")
            right.bar_chart(chart_data[["mean_latency_seconds", "mean_total_tokens"]])
            adaptive_rows = filtered[filtered.selected_adaptive_route.notna()]
            if not adaptive_rows.empty:
                st.subheader("Adaptive routing distribution")
                st.bar_chart(adaptive_rows.selected_adaptive_route.value_counts())
            verification_rows = filtered[filtered.verification_outcome.notna()]
            if not verification_rows.empty:
                st.subheader("Verification outcomes")
                st.bar_chart(verification_rows.verification_outcome.value_counts())
            st.subheader("Summary table")
            st.dataframe(summary, width="stretch")
            with st.expander("Question-level records"):
                st.dataframe(filtered, width="stretch")

with tabs[3]:
    result_files = sorted((ROOT / "results").glob("*/question_results.csv"), reverse=True)
    if not result_files:
        st.info("Insights are unavailable until a benchmark has completed; no mock results are shown.")
    else:
        for insight in computed_insights(pd.read_csv(result_files[0])):
            st.write(f"- {insight}")
