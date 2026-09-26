from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from aimzo.tasks.math_scoring import (
    is_math_correct,
    score_math_response,
    scoring_metadata,
)


@dataclass(frozen=True)
class MathTaskExample:
    question: str
    answer: str
    index: int
    solution: str | None = None
    metadata: dict[str, Any] | None = None


class BaseMathTask:
    benchmark = "math"

    def __init__(
        self,
        *,
        max_train_samples: int | None = None,
        max_eval_samples: int | None = None,
    ) -> None:
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples

    def load_train(self) -> list[MathTaskExample]:
        raise NotImplementedError

    def load_eval(self) -> list[MathTaskExample]:
        raise NotImplementedError

    def sample_train_batch(
        self,
        *,
        batch_size: int,
        seed: int,
        step: int,
    ) -> list[MathTaskExample]:
        rows = self.load_train()
        if not rows:
            return []
        rng = random.Random(int(seed) + int(step))
        if batch_size <= len(rows):
            return rng.sample(rows, int(batch_size))
        return [rng.choice(rows) for _ in range(int(batch_size))]

    def score_response(
        self,
        response: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
        *,
        format_reward: float = 0.1,
    ) -> float:
        question = None if metadata is None else metadata.get("question")
        return score_math_response(
            response,
            answer,
            benchmark=self.benchmark,
            format_reward=format_reward,
            question=question,
        )

    def is_correct(
        self,
        response: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        question = None if metadata is None else metadata.get("question")
        return is_math_correct(
            response,
            answer,
            benchmark=self.benchmark,
            question=question,
        )

    def scoring_metadata(self) -> dict[str, Any]:
        return scoring_metadata(self.benchmark)


class MathTask(BaseMathTask):
    benchmark = "math"

    def __init__(
        self,
        *,
        train_path: str | Path,
        eval_path: str | Path,
        max_train_samples: int | None = None,
        max_eval_samples: int | None = None,
    ) -> None:
        super().__init__(
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
        )
        self.train_path = Path(train_path).expanduser()
        self.eval_path = Path(eval_path).expanduser()

    def load_train(self) -> list[MathTaskExample]:
        return _read_math_rows(
            self.train_path,
            max_samples=self.max_train_samples,
            task="math",
        )

    def load_eval(self) -> list[MathTaskExample]:
        return _read_math_rows(
            self.eval_path,
            max_samples=self.max_eval_samples,
            task="math",
        )


class DeepScalerTask(BaseMathTask):
    benchmark = "deepscaler"

    def __init__(
        self,
        *,
        data_root: str | Path | None = None,
        max_train_samples: int | None = None,
        max_eval_samples: int | None = None,
    ) -> None:
        super().__init__(
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
        )
        self.data_root = (
            _default_deepscaler_root()
            if data_root is None
            else Path(data_root).expanduser()
        )

    @property
    def path(self) -> Path:
        return self.data_root / "dataset" / "deepscaler" / "deepscaler.json"

    def load_train(self) -> list[MathTaskExample]:
        return _read_math_rows(
            self.path,
            max_samples=self.max_train_samples,
            task="deepscaler",
        )

    def load_eval(self) -> list[MathTaskExample]:
        return _read_math_rows(
            self.path,
            max_samples=self.max_eval_samples,
            task="deepscaler",
        )


def _default_deepscaler_root() -> Path:
    return Path("data")


def _read_math_rows(
    path: Path,
    *,
    max_samples: int | None,
    task: str,
) -> list[MathTaskExample]:
    if not path.exists():
        raise FileNotFoundError(f"Missing math task data file: {path}")

    if path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Math JSON data must be a list of records: {path}")
        records = payload
    elif path.suffix == ".parquet":
        records = pd.read_parquet(path).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported math task data extension for {path}")

    if max_samples is not None:
        records = records[: max(0, int(max_samples))]

    examples: list[MathTaskExample] = []
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"Math row {index} in {path} must be a mapping")
        problem = row.get("problem")
        answer = row.get("answer")
        if (
            not isinstance(problem, str)
            or not problem.strip()
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            raise ValueError(
                f"Math row {index} in {path} must contain non-empty problem and answer fields"
            )
        solution = row.get("solution")
        examples.append(
            MathTaskExample(
                question=problem,
                answer=answer.strip(),
                index=index,
                solution=solution if isinstance(solution, str) else None,
                metadata={"task": task},
            )
        )
    return examples
