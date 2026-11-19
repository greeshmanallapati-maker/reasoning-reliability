from __future__ import annotations

from typing import Any

from reasoning_reliability.strategies.base import StrategyResult


def raw_model_response(result: StrategyResult) -> str:
    """Return the final model call's visible text without changing its meaning."""
    for call in reversed(result.calls):
        if call.text.strip():
            return call.text.strip()
    return ""


def chatbot_response(result: StrategyResult) -> str:
    """Present a strategy result as readable assistant text.

    Benchmark parsing intentionally remains strict. In the interactive demo, a
    useful raw answer is still shown when a small model ignores the requested
    FINAL_ANSWER delimiter.
    """
    raw = raw_model_response(result)
    if result.final_answer is not None:
        if result.reasoning:
            return f"{result.reasoning}\n\n**Final answer:** {result.final_answer}"
        return f"**Final answer:** {result.final_answer}"
    if raw:
        return raw
    return "I couldn't generate a response for that question."


def technical_payload(result: StrategyResult) -> dict[str, Any]:
    """Compact JSON-friendly diagnostics, omitting irrelevant null fields."""
    raw = raw_model_response(result)
    values: dict[str, Any] = {
        "strategy": result.strategy,
        "selected_route": result.selected_route or result.strategy,
        "parsed_answer": result.final_answer,
        "raw_response": raw or None,
        "difficulty": result.difficulty_score,
        "difficulty_features": result.difficulty_features or None,
        "uncertainty_proxy": result.uncertainty_score,
        "uncertainty_method": result.uncertainty_method,
        "latency_seconds": result.latency_seconds,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "verification": result.verification_claim,
        "verification_explanation": result.verification_explanation,
        "parser_status": "parsed" if result.final_answer is not None else "strict_delimiter_not_found",
        "failure": result.failure,
    }
    return {key: value for key, value in values.items() if value is not None}

