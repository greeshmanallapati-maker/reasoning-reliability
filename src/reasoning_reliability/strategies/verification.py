from __future__ import annotations

import re

from reasoning_reliability.data.schemas import BenchmarkExample

from .base import BaseStrategy, StrategyResult, format_question, parse_prediction
from .cot import ChainOfThoughtStrategy


class CoTVerificationStrategy(BaseStrategy):
    name = "verification"

    def __init__(self, engine) -> None:
        super().__init__(engine)
        self.cot = ChainOfThoughtStrategy(engine)

    def verification_prompt(self, example: BenchmarkExample, solution: str, answer: str | None) -> str:
        checklist = (
            "arithmetic errors, assumptions, missing constraints, units, and logical inconsistencies"
            if example.dataset == "gsm8k"
            else "question interpretation, scientific concepts, contradictions with options, and unsupported conclusions"
        )
        return (
            "Independently verify the proposed solution below. The reference answer is not available. "
            f"Check {checklist}. Preserve the answer when it is sound; do not change it merely to be useful.\n\n"
            f"QUESTION:\n{format_question(example)}\n\nPROPOSED_SOLUTION:\n{solution}\n"
            f"PROPOSED_ANSWER: {answer or 'INVALID'}\n\n"
            "Return exactly these fields:\nVERIFICATION: PASS or FAIL\n"
            "EXPLANATION: <brief explanation>\nCORRECTION_NEEDED: YES or NO\n"
            "FINAL_ANSWER: <number or valid choice label>"
        )

    def run(self, example: BenchmarkExample) -> StrategyResult:
        initial = self.cot.run(example)
        return self.run_from_initial(example, initial)

    def run_from_initial(self, example: BenchmarkExample, initial: StrategyResult) -> StrategyResult:
        verify_call = self.engine.generate(
            self.verification_prompt(example, initial.reasoning, initial.final_answer),
            collect_token_scores=False,
        )
        corrected = parse_prediction(example, verify_call.text)
        final = corrected if corrected is not None else initial.final_answer
        claim_match = re.search(r"VERIFICATION\s*:\s*(PASS|FAIL)", verify_call.text, re.I)
        explanation_match = re.search(r"EXPLANATION\s*:\s*(.+)", verify_call.text, re.I)
        failure = initial.failure if not verify_call.error else verify_call.error
        if corrected is None and initial.final_answer is None:
            failure = failure or "verification_and_initial_parse_failure"
        return StrategyResult(
            strategy=self.name,
            initial_answer=initial.final_answer,
            final_answer=final,
            reasoning=initial.reasoning,
            reasoning_tokens=initial.reasoning_tokens,
            verification_claim=claim_match.group(1).upper() if claim_match else "UNPARSEABLE",
            verification_explanation=explanation_match.group(1).strip() if explanation_match else None,
            calls=initial.calls + [verify_call],
            failure=failure,
        )

