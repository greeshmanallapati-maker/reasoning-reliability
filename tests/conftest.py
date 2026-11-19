from __future__ import annotations

from collections import deque

import pytest

from reasoning_reliability.data.schemas import BenchmarkExample
from reasoning_reliability.model import GenerationResult


class MockEngine:
    model_id = "mock/model"

    def __init__(self, outputs: list[str], logprob: float = -0.1) -> None:
        self.outputs = deque(outputs)
        self.logprob = logprob
        self.prompts: list[str] = []

    def generate(self, prompt: str, **overrides) -> GenerationResult:
        self.prompts.append(prompt)
        text = self.outputs.popleft()
        return GenerationResult(
            prompt=prompt,
            text=text,
            input_tokens=10,
            output_tokens=5,
            latency_seconds=0.02,
            model_id=self.model_id,
            token_logprobs=[self.logprob] * 5,
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate_batch(self, prompts: list[str], **overrides) -> list[GenerationResult]:
        return [self.generate(prompt, **overrides) for prompt in prompts]


@pytest.fixture
def math_example() -> BenchmarkExample:
    return BenchmarkExample("gsm8k:1", "gsm8k", "test", "What is 2 + 3?", "5")


@pytest.fixture
def arc_example() -> BenchmarkExample:
    return BenchmarkExample(
        "arc:1", "arc_easy", "test", "Which is a gas?", "B", {"A": "Iron", "B": "Oxygen", "C": "Ice"}
    )

