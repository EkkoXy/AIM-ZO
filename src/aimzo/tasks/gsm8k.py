from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from aimzo.paths import dataset_path
from aimzo.rewards import extract_gsm8k_final_answer, normalize_answer


@dataclass(frozen=True)
class GSM8KExample:
    question: str
    answer: str
    index: int


def _split_path(split: str, *, data_root: Path | str | None = None) -> Path:
    filename = {
        "train": "train-00000-of-00001.parquet",
        "test": "test-00000-of-00001.parquet",
    }[split]
    if data_root is not None:
        return Path(data_root).expanduser() / "dataset" / "gsm8k" / "main" / filename
    return dataset_path("gsm8k", "main", filename)


def _cap_rows(rows: list[GSM8KExample], max_samples: int | None) -> list[GSM8KExample]:
    if max_samples is None:
        return rows
    return rows[: max(0, int(max_samples))]


def _read_examples(path: Path, *, max_samples: int | None) -> list[GSM8KExample]:
    if not path.exists():
        raise FileNotFoundError(f"Missing GSM8K parquet file: {path}")
    frame = pd.read_parquet(path)
    missing = {"question", "answer"}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing GSM8K columns in {path}: {', '.join(sorted(missing))}")

    rows: list[GSM8KExample] = []
    for index, row in enumerate(frame.to_dict(orient="records")):
        rows.append(
            GSM8KExample(
                question=str(row["question"]),
                answer=normalize_answer(extract_gsm8k_final_answer(str(row["answer"]))),
                index=index,
            )
        )
    return _cap_rows(rows, max_samples)


@dataclass
class GSM8KTask:
    benchmark = "gsm8k"

    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    data_root: Path | str | None = None

    def load_train(self) -> list[GSM8KExample]:
        return _read_examples(
            _split_path("train", data_root=self.data_root),
            max_samples=self.max_train_samples,
        )

    def load_eval(self) -> list[GSM8KExample]:
        return _read_examples(
            _split_path("test", data_root=self.data_root),
            max_samples=self.max_eval_samples,
        )

    def sample_train_batch(self, *, batch_size: int, seed: int, step: int) -> list[GSM8KExample]:
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
        metadata: dict | None = None,
        *,
        format_reward: float = 0.1,
    ) -> float:
        from aimzo.tasks.math_scoring import score_math_response

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
        metadata: dict | None = None,
    ) -> bool:
        from aimzo.tasks.math_scoring import is_math_correct

        question = None if metadata is None else metadata.get("question")
        return is_math_correct(
            response,
            answer,
            benchmark=self.benchmark,
            question=question,
        )

    def scoring_metadata(self) -> dict:
        from aimzo.tasks.math_scoring import scoring_metadata

        return scoring_metadata(self.benchmark)


def write_gsm8k_parquet(
    *,
    train_rows: Iterable[dict],
    test_rows: Iterable[dict],
    output_root: Path,
) -> tuple[Path, Path]:
    main = Path(output_root) / "dataset" / "gsm8k" / "main"
    main.mkdir(parents=True, exist_ok=True)
    train_path = main / "train-00000-of-00001.parquet"
    test_path = main / "test-00000-of-00001.parquet"
    pd.DataFrame(list(train_rows)).to_parquet(train_path)
    pd.DataFrame(list(test_rows)).to_parquet(test_path)
    return train_path, test_path
