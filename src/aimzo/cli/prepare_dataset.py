from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from aimzo.paths import data_root, dataset_path
from aimzo.tasks.gsm8k import write_gsm8k_parquet


_HF_ZOREGULAR_DATASETS = {
    "boolq": ("super_glue", "boolq"),
    "cb": ("super_glue", "cb"),
    "copa": ("super_glue", "copa"),
    "drop": ("ucinlp/drop", None),
    "multirc": ("super_glue", "multirc"),
    "record": ("super_glue", "record"),
    "rte": ("glue", "rte"),
    "squad": ("squad", None),
    "sst2": ("glue", "sst2"),
    "wic": ("super_glue", "wic"),
    "wsc": ("super_glue", "wsc.fixed"),
}


def prepare_gsm8k(*, force: bool = False) -> tuple[str, str]:
    train_path = dataset_path("gsm8k", "main", "train-00000-of-00001.parquet")
    test_path = dataset_path("gsm8k", "main", "test-00000-of-00001.parquet")
    if train_path.exists() and test_path.exists() and not force:
        return str(train_path), str(test_path)

    from datasets import load_dataset

    train_rows = list(load_dataset("openai/gsm8k", "main", split="train"))
    test_rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    written_train, written_test = write_gsm8k_parquet(
        train_rows=train_rows,
        test_rows=test_rows,
        output_root=data_root(),
    )
    return str(written_train), str(written_test)


def prepare_hf_zoregular(task: str, *, force: bool = False) -> Path:
    dataset_name, config_name = _HF_ZOREGULAR_DATASETS[task]
    output_path = dataset_path(task)
    if (output_path / "dataset_dict.json").exists() and not force:
        return output_path

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, config_name)
    if output_path.exists():
        if not force:
            raise FileExistsError(
                f"dataset path exists but is incomplete: {output_path}; "
                "pass --force to replace it"
            )
        shutil.rmtree(output_path)
    dataset.save_to_disk(str(output_path))
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare AIMZO datasets")
    parser.add_argument(
        "--task",
        choices=["gsm8k", *_HF_ZOREGULAR_DATASETS],
        required=True,
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.task == "gsm8k":
        train_path, test_path = prepare_gsm8k(force=args.force)
        print(f"train={train_path}")
        print(f"test={test_path}")
        return 0
    if args.task in _HF_ZOREGULAR_DATASETS:
        output_path = prepare_hf_zoregular(args.task, force=args.force)
        print(f"dataset={output_path}")
        return 0
    raise ValueError(f"Unsupported task {args.task!r}")


if __name__ == "__main__":
    raise SystemExit(main())
