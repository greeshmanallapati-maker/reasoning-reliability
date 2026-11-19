#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from reasoning_reliability.adaptive.routing import select_thresholds_on_development


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit adaptive thresholds from a development run only")
    parser.add_argument("run_dir")
    parser.add_argument("--cost-lambda", type=float, default=0.1)
    parser.add_argument("--uncertainty-weight", type=float, default=0.5)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    frame = pd.read_csv(run_dir / "question_results.csv")
    if "experiment_partition" not in frame or set(frame["experiment_partition"]) != {"development"}:
        raise SystemExit("Router fitting is allowed only on records labeled experiment_partition=development")
    adaptive = frame[frame.strategy == "adaptive"].drop_duplicates("question_id").set_index("question_id")
    if adaptive.empty or adaptive.difficulty_score.isna().any() or adaptive.uncertainty_score.isna().any():
        raise SystemExit("Run must contain adaptive development records with difficulty and uncertainty")
    question_ids = adaptive.index.tolist()
    correctness, costs = {}, {}
    for route in ("direct", "cot", "verification"):
        fixed = frame[frame.strategy == route].drop_duplicates("question_id").set_index("question_id")
        if not set(question_ids).issubset(fixed.index):
            raise SystemExit(f"Missing fixed-strategy records for {route}")
        correctness[route] = fixed.loc[question_ids, "correct"].astype(bool).tolist()
        costs[route] = fixed.loc[question_ids, "total_tokens"].astype(float).tolist()
    policy, report = select_thresholds_on_development(
        adaptive.loc[question_ids, "difficulty_score"].astype(float).tolist(),
        adaptive.loc[question_ids, "uncertainty_score"].astype(float).tolist(),
        correctness,
        costs,
        cost_lambda=args.cost_lambda,
        uncertainty_weight=args.uncertainty_weight,
    )
    payload = report | {
        "source_run": run_dir.name,
        "partition": "development",
        "cost_unit": "min-max normalized total tokens across candidate routes",
        "cost_lambda": args.cost_lambda,
        "uncertainty_weight": policy.weight,
    }
    target = run_dir / "routing_policy.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
