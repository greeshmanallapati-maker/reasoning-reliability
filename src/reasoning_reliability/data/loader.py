from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

from .answer_parser import normalize_arc_label, parse_gsm8k_reference
from .schemas import BenchmarkExample


def _stable_id(prefix: str, raw_id: str, question: str) -> str:
    identity = raw_id or hashlib.sha256(question.encode()).hexdigest()[:16]
    return f"{prefix}:{identity}"


def preprocess_gsm8k(row: dict[str, Any], split: str, index: int) -> BenchmarkExample:
    answer = parse_gsm8k_reference(row["answer"])
    if answer is None:
        raise ValueError(f"Invalid GSM8K reference answer at index {index}")
    return BenchmarkExample(
        id=_stable_id("gsm8k", str(row.get("id", index)), row["question"]),
        dataset="gsm8k",
        split=split,
        question=row["question"].strip(),
        answer=answer,
        difficulty="multi_step_math",
    )


def preprocess_arc(
    row: dict[str, Any], subset: str, split: str, index: int
) -> BenchmarkExample:
    choice_data = row["choices"]
    labels = [str(x).upper() for x in choice_data["label"]]
    texts = [str(x) for x in choice_data["text"]]
    choices = dict(zip(labels, texts, strict=True))
    answer = normalize_arc_label(row["answerKey"], labels)
    if answer is None:
        raise ValueError(f"Invalid ARC reference label at index {index}")
    dataset = "arc_easy" if subset == "ARC-Easy" else "arc_challenge"
    return BenchmarkExample(
        id=_stable_id(dataset, str(row.get("id", index)), row["question"]),
        dataset=dataset,
        split=split,
        question=row["question"].strip(),
        choices=choices,
        answer=answer,
        difficulty="easy" if dataset == "arc_easy" else "challenge",
    )


def deterministic_sample(
    examples: Iterable[BenchmarkExample], size: int | None, seed: int
) -> list[BenchmarkExample]:
    values = list(examples)
    if len({example.id for example in values}) != len(values):
        raise ValueError("Duplicate example IDs detected")
    if size is None or size >= len(values):
        return values
    if size < 0:
        raise ValueError("Sample size cannot be negative")
    return random.Random(seed).sample(values, size)


class BenchmarkLoader:
    """Lazy Hugging Face dataset loader with cached, deterministic selections."""

    def __init__(self, cache_dir: str | Path = "results/selections") -> None:
        self.cache_dir = Path(cache_dir)

    def load(self, name: str, size: int | None, seed: int, partition: str) -> list[BenchmarkExample]:
        if partition not in {"development", "evaluation"}:
            raise ValueError("partition must be development or evaluation")
        cache = self.cache_dir / f"{name}-{partition}-seed{seed}-n{size or 'all'}.json"
        if cache.exists():
            return [BenchmarkExample(**item) for item in json.loads(cache.read_text())]
        examples = self._download(name)
        # Disjoint deterministic halves prevent prompt/routing tuning on final answers.
        ordered = sorted(examples, key=lambda e: hashlib.sha256(f"{seed}:{e.id}".encode()).hexdigest())
        midpoint = len(ordered) // 2
        pool = ordered[:midpoint] if partition == "development" else ordered[midpoint:]
        selected = deterministic_sample(pool, size, seed)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([e.to_dict() for e in selected], indent=2), encoding="utf-8")
        return selected

    @staticmethod
    def _download(name: str) -> list[BenchmarkExample]:
        # All bundled benchmark datasets are public. Avoid an expired ambient
        # Hugging Face login breaking their download with a misleading 401.
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        from datasets import load_dataset

        if name == "gsm8k":
            split = "test"
            data = load_dataset("openai/gsm8k", "main", split=split)
            return [preprocess_gsm8k(dict(row), split, i) for i, row in enumerate(data)]
        mapping = {"arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge"}
        if name not in mapping:
            raise ValueError(f"Unsupported dataset: {name}")
        subset, split = mapping[name], "test"
        data = load_dataset("allenai/ai2_arc", subset, split=split)
        return [preprocess_arc(dict(row), subset, split, i) for i, row in enumerate(data)]

