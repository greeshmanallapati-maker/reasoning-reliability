from __future__ import annotations

from reasoning_reliability.data.schemas import BenchmarkExample
from reasoning_reliability.model import GenerationResult

from .base import BaseStrategy, StrategyResult, format_question, parse_prediction


class DirectAnsweringStrategy(BaseStrategy):
    name = "direct"
    supports_batch = True

    def prompt(self, example: BenchmarkExample) -> str:
        format_example = "FINAL_ANSWER: 42" if example.dataset == "gsm8k" else "FINAL_ANSWER: B"
        return (
            "You are answering a controlled research benchmark. Compute the answer, but do not "
            "show reasoning or repeat the question. Your entire response must be exactly one line "
            f"in the same format as `{format_example}`. Replace the example value with the actual "
            f"answer. Do not output XML tags or placeholders.\n\n{format_question(example)}"
        )

    def from_generation(self, example: BenchmarkExample, call: GenerationResult) -> StrategyResult:
        answer = parse_prediction(example, call.text)
        return StrategyResult(
            strategy=self.name,
            final_answer=answer,
            calls=[call],
            failure=call.error or ("missing_or_malformed_final_answer" if answer is None else None),
        )

    def run(self, example: BenchmarkExample) -> StrategyResult:
        return self.from_generation(example, self.engine.generate(self.prompt(example)))

    def run_batch(self, examples: list[BenchmarkExample]) -> list[StrategyResult]:
        calls = self.engine.generate_batch([self.prompt(example) for example in examples])
        return [self.from_generation(example, call) for example, call in zip(examples, calls, strict=True)]

