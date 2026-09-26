from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from aimzo.artifacts import RunArtifactWriter
from aimzo.cli.hf_zoregular_train import build_backend, validate_cli_config
from aimzo.config import load_config
from aimzo.tasks.registry import build_task
from aimzo.tasks.zoregular import load_zoregular_split
from aimzo.trainers.hf_zoregular import HFZORegularTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a full-parameter HF ZORegular checkpoint"
    )
    parser.add_argument("--config", required=True, help="Training YAML config")
    parser.add_argument("--checkpoint", help="Checkpoint directory; omit for zero-shot")
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap; the default evaluates the full official validation split.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path; defaults to <checkpoint>/official_eval.json.",
    )
    return parser


def official_eval_config(config: Any, *, max_eval_samples: int | None) -> Any:
    """Switch a train-derived dev config to the official validation split."""
    return replace(
        config,
        data=replace(
            config.data,
            max_eval_samples=max_eval_samples,
            zoregular_train_dev_samples=None,
            zoregular_sampling_policy="split",
        ),
    )


def official_evaluation_rows(config: Any, *, max_eval_samples: int | None):
    rows = load_zoregular_split(config.data.task, config.data.data_root, "validation")
    if max_eval_samples is None:
        return rows, None
    if max_eval_samples < 0:
        raise ValueError("max_eval_samples must be nonnegative")
    indices = np.random.RandomState(0).permutation(len(rows))[:max_eval_samples]
    return [rows[int(index)] for index in indices], [int(index) for index in indices]


def evaluate_checkpoint(
    *,
    config_path: str | Path,
    checkpoint: str | Path | None,
    max_eval_samples: int | None,
) -> dict[str, Any]:
    config = official_eval_config(
        load_config(config_path),
        max_eval_samples=max_eval_samples,
    )
    validate_cli_config(config)
    task = build_task(config.data)
    if bool(getattr(task, "is_generation", False)):
        raise ValueError(
            "generation tasks require the task-metric evaluator; "
            "this command currently reports classification accuracy only"
        )

    backend = build_backend(config)
    if checkpoint is not None:
        backend.load_checkpoint(checkpoint)
    writer = RunArtifactWriter(config.output_path)
    trainer = HFZORegularTrainer(
        config=config,
        task=task,
        backend=backend,
        artifact_writer=writer,
    )
    try:
        rows, source_indices = official_evaluation_rows(
            config, max_eval_samples=max_eval_samples
        )
        metrics = trainer.evaluate_rows(rows, step=0)
        grouped_metrics = None
        if config.data.task == "multirc":
            full_rows = load_zoregular_split(
                "multirc", config.data.data_root, "validation"
            )
            grouped_metrics = trainer.evaluate_rows(
                full_rows, step=0, include_multirc_grouped=True
            )
            if "error" in grouped_metrics:
                raise RuntimeError(
                    f"MultiRC grouped evaluation failed: {grouped_metrics['error']}"
                )
        if "error" in metrics:
            raise RuntimeError(f"official evaluation failed: {metrics['error']}")
    finally:
        writer.close()
    return {
        "config": str(Path(config_path)),
        "checkpoint": str(Path(checkpoint)) if checkpoint is not None else None,
        "split": "official_validation",
        "max_eval_samples": max_eval_samples,
        "eval_seed": 0 if max_eval_samples is not None else None,
        "source_indices": source_indices,
        "metrics": metrics,
        "official_grouped_metrics": grouped_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_checkpoint(
        config_path=args.config,
        checkpoint=args.checkpoint,
        max_eval_samples=args.max_eval_samples,
    )
    output = (
        Path(args.output).expanduser()
        if args.output is not None
        else (
            Path(args.checkpoint).expanduser() / "official_eval.json"
            if args.checkpoint is not None
            else load_config(args.config).output_path / "zero_shot_eval.json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
