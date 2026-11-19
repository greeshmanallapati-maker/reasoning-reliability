from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from reasoning_reliability.data.answer_parser import parse_arc_prediction, parse_gsm8k_prediction
from reasoning_reliability.data.schemas import BenchmarkExample
from reasoning_reliability.model import GenerationResult, InferenceEngine


@dataclass
class StrategyResult:
    strategy: str
    final_answer: str | None
    reasoning: str = ""
    reasoning_tokens: int = 0
    initial_answer: str | None = None
    selected_route: str | None = None
    difficulty_score: float | None = None
    difficulty_features: dict[str, Any] = field(default_factory=dict)
    uncertainty_score: float | None = None
    uncertainty_method: str | None = None
    routing_explanation: str | None = None
    verification_claim: str | None = None
    verification_explanation: str | None = None
    calls: list[GenerationResult] = field(default_factory=list)
    failure: str | None = None

    @property
    def input_tokens(self) -> int:
        return sum(call.input_tokens for call in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def latency_seconds(self) -> float:
        return sum(call.latency_seconds for call in self.calls)

    @property
    def peak_gpu_memory_bytes(self) -> int:
        return max((call.peak_gpu_memory_bytes for call in self.calls), default=0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["calls"] = [call.to_dict() for call in self.calls]
        data.update(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            latency_seconds=self.latency_seconds,
            reasoning_tokens=self.reasoning_tokens,
            peak_gpu_memory_bytes=self.peak_gpu_memory_bytes,
        )
        return data


def format_question(example: BenchmarkExample) -> str:
    if not example.choices:
        return example.question
    options = "\n".join(f"{label}. {text}" for label, text in example.choices.items())
    return f"{example.question}\n\nOptions:\n{options}"


def parse_prediction(example: BenchmarkExample, text: str) -> str | None:
    if example.dataset == "gsm8k":
        return parse_gsm8k_prediction(text)
    return parse_arc_prediction(text, list(example.choices))


class BaseStrategy:
    name = "base"
    supports_batch = False

    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine

    def run(self, example: BenchmarkExample) -> StrategyResult:
        raise NotImplementedError

    def run_batch(self, examples: list[BenchmarkExample]) -> list[StrategyResult]:
        return [self.run(example) for example in examples]

