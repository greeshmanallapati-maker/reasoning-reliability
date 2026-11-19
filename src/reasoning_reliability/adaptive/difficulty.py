from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from reasoning_reliability.data.schemas import BenchmarkExample


@dataclass(frozen=True)
class DifficultyFeatures:
    word_count: int
    sentence_count: int
    numeric_count: int
    operation_count: int
    constraint_count: int
    reasoning_indicator_count: int
    option_count: int
    is_math: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class DifficultyEstimator:
    OPERATIONS = re.compile(r"[+×*÷/=]|\b(?:sum|difference|product|percent|total|each|per)\b", re.I)
    CONSTRAINTS = re.compile(r"\b(?:if|only|except|at least|at most|before|after|must|cannot)\b", re.I)
    INDICATORS = re.compile(r"\b(?:then|therefore|because|first|next|remaining|altogether|how many)\b", re.I)

    def features(self, example: BenchmarkExample) -> DifficultyFeatures:
        text = example.question
        return DifficultyFeatures(
            word_count=len(text.split()),
            sentence_count=max(1, len(re.findall(r"[?.!]", text))),
            numeric_count=len(re.findall(r"[-+]?\d+(?:\.\d+)?", text)),
            operation_count=len(self.OPERATIONS.findall(text)),
            constraint_count=len(self.CONSTRAINTS.findall(text)),
            reasoning_indicator_count=len(self.INDICATORS.findall(text)),
            option_count=len(example.choices),
            is_math=int(example.dataset == "gsm8k"),
        )

    def estimate(self, example: BenchmarkExample) -> tuple[float, dict[str, int]]:
        f = self.features(example)
        score = (
            min(f.word_count / 80, 1) * 0.15
            + min(f.sentence_count / 5, 1) * 0.10
            + min(f.numeric_count / 5, 1) * 0.15
            + min(f.operation_count / 4, 1) * 0.20
            + min(f.constraint_count / 3, 1) * 0.15
            + min(f.reasoning_indicator_count / 3, 1) * 0.15
            + f.is_math * 0.10
        )
        return min(max(score, 0.0), 1.0), f.as_dict()

