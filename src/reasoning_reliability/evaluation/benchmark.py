from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from reasoning_reliability.data.schemas import BenchmarkExample
from reasoning_reliability.strategies.base import BaseStrategy, StrategyResult
from reasoning_reliability.utils.reproducibility import seed_everything

from .metrics import pareto_frontier, summarize_results
from .storage import ResultStore


def verification_outcome(result: StrategyResult, reference: str) -> str | None:
    initial_right = result.initial_answer == reference
    final_right = result.final_answer == reference
    return {
        (False, True): "wrong_corrected_to_right",
        (True, False): "right_changed_to_wrong",
        (True, True): "right_preserved",
        (False, False): "wrong_preserved",
    }[(initial_right, final_right)]


class BenchmarkRunner:
    def __init__(
        self,
        strategies: dict[str, BaseStrategy],
        output_dir: str,
        run_id: str | None = None,
        seed: int = 42,
        model_metadata: dict[str, Any] | None = None,
        experiment_partition: str = "development",
        batch_size: int = 1,
    ) -> None:
        self.strategies = strategies
        self.seed = seed
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        self.store = ResultStore(output_dir, self.run_id)
        self.model_metadata = model_metadata or {}
        self.experiment_partition = experiment_partition
        self.batch_size = max(1, int(batch_size))

    def run(self, examples: list[BenchmarkExample], trials: int = 1) -> pd.DataFrame:
        completed = self.store.completed_keys()
        self.store.write_json(
            "run_metadata.json",
            {"run_id": self.run_id, "seed": self.seed, "model": self.model_metadata,
             "trials": trials, "experiment_partition": self.experiment_partition,
             "batch_size": self.batch_size},
        )
        for trial in range(trials):
            seed_everything(self.seed + trial)
            for offset in range(0, len(examples), self.batch_size):
                batch = examples[offset : offset + self.batch_size]
                for name, strategy in self.strategies.items():
                    pending = [example for example in batch if (example.id, name, trial) not in completed]
                    if not pending:
                        continue
                    if strategy.supports_batch and len(pending) > 1:
                        started = time.perf_counter()
                        results = strategy.run_batch(pending)
                        end_to_end = (time.perf_counter() - started) / len(pending)
                        for example, result in zip(pending, results, strict=True):
                            self.store.append(self._record(example, result, name, trial, end_to_end))
                    else:
                        for example in pending:
                            started = time.perf_counter()
                            result = strategy.run(example)
                            end_to_end = time.perf_counter() - started
                            self.store.append(self._record(example, result, name, trial, end_to_end))
        frame = self.store.finalize()
        summary = pareto_frontier(summarize_results(frame))
        summary.to_csv(self.store.run_dir / "summary.csv", index=False)
        return frame

    def _record(
        self,
        example: BenchmarkExample,
        result: StrategyResult,
        strategy_name: str,
        trial: int,
        end_to_end: float,
    ) -> dict[str, Any]:
        timed_out = any(call.timed_out for call in result.calls)
        return {
            "experiment_id": self.run_id,
            "dataset": example.dataset,
            "dataset_split": example.split,
            "experiment_partition": self.experiment_partition,
            "question_id": example.id,
            "question": example.question,
            "ground_truth": example.answer,
            "strategy": strategy_name,
            "difficulty_category": example.difficulty,
            "difficulty_score": result.difficulty_score,
            "difficulty_features": result.difficulty_features,
            "uncertainty_score": result.uncertainty_score,
            "uncertainty_method": result.uncertainty_method,
            "selected_adaptive_route": result.selected_route,
            "routing_explanation": result.routing_explanation,
            "initial_answer": result.initial_answer,
            "final_answer": result.final_answer,
            "correct": result.final_answer == example.answer,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            "model_call_latency_seconds": result.latency_seconds,
            "total_latency_seconds": end_to_end,
            "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
            "verification_outcome": verification_outcome(result, example.answer),
            "verification_claim": result.verification_claim,
            "trial": trial,
            "random_seed": self.seed + trial,
            "model_configuration": self.model_metadata,
            "timed_out": timed_out,
            "failure": result.failure,
            "reasoning": result.reasoning,
            "reasoning_tokens": result.reasoning_tokens,
            "calls": [call.to_dict() for call in result.calls],
        }

