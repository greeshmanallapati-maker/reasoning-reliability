from __future__ import annotations

from reasoning_reliability.data.answer_parser import split_reasoning
from reasoning_reliability.data.schemas import BenchmarkExample

from .base import BaseStrategy, StrategyResult, format_question, parse_prediction


class ChainOfThoughtStrategy(BaseStrategy):
    name = "cot"
    supports_batch = True

    def prompt(self, example: BenchmarkExample) -> str:
        format_example = "FINAL_ANSWER: 42" if example.dataset == "gsm8k" else "FINAL_ANSWER: B"
        return (
            "Solve the question carefully using a concise logical sequence of intermediate steps. "
            "Show only calculations or reasoning necessary to support the conclusion. "
            f"End with a separate line in the same format as `{format_example}`, replacing the "
            "example value with the actual answer. Do not output XML tags or placeholders.\n\n"
            f"{format_question(example)}"
        )

    def run(self, example: BenchmarkExample) -> StrategyResult:
        call = self.engine.generate(self.prompt(example), collect_token_scores=False)
        return self.from_generation(example, call)

    def from_generation(self, example: BenchmarkExample, call) -> StrategyResult:
        reasoning, _ = split_reasoning(call.text)
        answer = parse_prediction(example, call.text)
        reasoning_tokens = self.engine.count_tokens(reasoning) if reasoning else 0
        return StrategyResult(
            strategy=self.name,
            final_answer=answer,
            reasoning=reasoning,
            reasoning_tokens=reasoning_tokens,
            calls=[call],
            failure=call.error or ("missing_or_malformed_final_answer" if answer is None else None),
        )

    def run_batch(self, examples: list[BenchmarkExample]) -> list[StrategyResult]:
        calls = self.engine.generate_batch(
            [self.prompt(example) for example in examples], collect_token_scores=False
        )
        return [self.from_generation(example, call) for example, call in zip(examples, calls, strict=True)]

