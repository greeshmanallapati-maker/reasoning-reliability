from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    dataset: Literal["gsm8k", "arc_easy", "arc_challenge"]
    split: str
    question: str
    answer: str
    choices: dict[str, str] = field(default_factory=dict)
    difficulty: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

