#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from reasoning_reliability.adaptive import RoutingPolicy
from reasoning_reliability.data import BenchmarkLoader
from reasoning_reliability.evaluation import BenchmarkRunner
from reasoning_reliability.model import HuggingFaceEngine
from reasoning_reliability.strategies import (
    AdaptiveReasoningStrategy,
    ChainOfThoughtStrategy,
    CoTVerificationStrategy,
    DirectAnsweringStrategy,
)
from reasoning_reliability.utils.config import load_config
from reasoning_reliability.utils.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true", help="Reuse --run-id and skip completed records")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["benchmark"]["seed"])
    seed_everything(seed)
    engine = HuggingFaceEngine(config["model"], config["generation"])
    if not args.no_warmup:
        engine.warmup()
    adaptive_config = config["adaptive"]
    policy = RoutingPolicy(
        adaptive_config["low_threshold"],
        adaptive_config["high_threshold"],
        adaptive_config["uncertainty_weight"],
    )
    strategies = {
        "direct": DirectAnsweringStrategy(engine),
        "cot": ChainOfThoughtStrategy(engine),
        "verification": CoTVerificationStrategy(engine),
        "adaptive": AdaptiveReasoningStrategy(engine, policy, "combined"),
    }
    if args.ablations:
        strategies.update(
            adaptive_difficulty=AdaptiveReasoningStrategy(engine, policy, "difficulty_only"),
            adaptive_uncertainty=AdaptiveReasoningStrategy(engine, policy, "uncertainty_only"),
        )
    benchmark = config["benchmark"]
    loader = BenchmarkLoader(benchmark["cache_dir"])
    examples = []
    for name in ("gsm8k", "arc_easy", "arc_challenge"):
        examples.extend(loader.load(name, benchmark[name], seed, benchmark["partition"]))
    if args.resume and not args.run_id:
        parser.error("--resume requires --run-id")
    runner = BenchmarkRunner(
        strategies,
        config["output_dir"],
        run_id=args.run_id,
        seed=seed,
        model_metadata=engine.load_metadata | {"generation": config["generation"]},
        experiment_partition=benchmark["partition"],
        batch_size=benchmark.get("batch_size", 1),
    )
    frame = runner.run(examples, int(benchmark["trials"]))
    print(f"Completed {len(frame)} records in {runner.store.run_dir}")


if __name__ == "__main__":
    main()

