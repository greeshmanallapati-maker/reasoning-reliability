from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


class ResultStore:
    def __init__(self, output_dir: str | Path, run_id: str) -> None:
        self.run_dir = Path(output_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "question_results.jsonl"

    def completed_keys(self) -> set[tuple[str, str, int]]:
        if not self.jsonl_path.exists():
            return set()
        keys = set()
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            keys.add((row["question_id"], row["strategy"], int(row["trial"])))
        return keys

    def append(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()

    def finalize(self) -> pd.DataFrame:
        rows = [json.loads(line) for line in self.jsonl_path.read_text().splitlines() if line.strip()]
        frame = pd.DataFrame(rows)
        frame.to_csv(self.run_dir / "question_results.csv", index=False)
        return frame

    def write_json(self, name: str, value: dict[str, Any] | list[Any]) -> None:
        path = self.run_dir / name
        path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

