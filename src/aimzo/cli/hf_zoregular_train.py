from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import sys
import traceback
from typing import Any

from aimzo.artifacts import RunArtifactWriter
from aimzo.backends.hf import HFBackend
from aimzo.config import ExperimentConfig, load_config, validate_config
from aimzo.tasks.registry import build_task
from aimzo.tasks.zoregular import is_zoregular_task
from aimzo.trainers.hf_zoregular import HFZORegularTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AIM-ZO and supported ZO baselines with Hugging Face")
    parser.add_argument(
        "config_path",
        nargs="?",
        help="Path to an AIM-ZO YAML config",
    )
    parser.add_argument(
        "--config",
        dest="config_option",
        help="Path to an AIM-ZO YAML config",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override trainer.output_dir from the config.",
    )
    return parser


def validate_cli_config(config: ExperimentConfig) -> None:
    validate_config(config)
    if config.backend.name != "hf":
        raise ValueError("aimzo-hf-zoregular-train requires backend.name='hf'")
    if not is_zoregular_task(config.data.task):
        raise ValueError("aimzo-hf-zoregular-train requires a ZORegular data.task")


def build_backend(config: ExperimentConfig) -> HFBackend:
    return HFBackend(
        model_name=config.backend.model_name,
        dtype=config.backend.dtype,
        lora_rank=config.backend.lora_rank,
        lora_alpha=config.backend.lora_alpha,
        lora_target_modules=config.backend.lora_target_modules,
        parameter_scope=config.zo.parameter_scope,
        trust_remote_code=config.backend.trust_remote_code,
        source_compat_tokenizer=(
            config.objective.name == "source_sst2_candidate_scoring"
        ),
        source_compat_tokenizer_profile=config.zo.source_compatibility_profile,
    )


def build_artifact_writer(config: ExperimentConfig) -> RunArtifactWriter:
    interval = config.logging.history_snapshot_interval
    if interval is None:
        return RunArtifactWriter(config.output_path)
    return RunArtifactWriter(
        config.output_path,
        history_snapshot_interval=interval,
    )


def _seed_process_for_training(config: ExperimentConfig) -> None:
    seed = int(config.zo.seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = _resolve_config_path(
        parser,
        args.config_path,
        args.config_option,
    )
    config: ExperimentConfig | None = None
    try:
        config = load_config(config_path)
        if args.output_dir is not None:
            config = replace(
                config,
                trainer=replace(config.trainer, output_dir=args.output_dir),
            )
        validate_cli_config(config)

        _seed_process_for_training(config)
        backend = build_backend(config)
        task = build_task(config.data)
        artifact_writer = build_artifact_writer(config)
        trainer = HFZORegularTrainer(
            config=config,
            task=task,
            backend=backend,
            artifact_writer=artifact_writer,
        )
        result = trainer.train()
        print(_compact_json(_success_summary(config, result)))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI reports compact JSON errors.
        if os.environ.get("AIMZO_DEBUG_TRACEBACK") == "1":
            traceback.print_exc()
        print(_compact_json(_failure_summary(config, exc)), file=sys.stderr)
        return 1


def _resolve_config_path(
    parser: argparse.ArgumentParser,
    config_path: str | None,
    config_option: str | None,
) -> str:
    if config_path is not None and config_option is not None:
        parser.error("pass config either positionally or with --config, not both")
    resolved = config_path or config_option
    if resolved is None:
        parser.error("config path is required")
    return resolved


def _success_summary(config: ExperimentConfig, result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ok": bool(result.get("ok", True)),
        "steps": result.get("steps"),
        "completed_steps": result.get("completed_steps"),
        "final_eval": result.get("final_eval"),
        "output_dir": str(config.output_path),
    }
    for key in ("final_checkpoint", "final_adapter"):
        if result.get(key) is not None:
            summary[key] = result[key]
    if result.get("resume") is not None:
        summary["resume"] = result["resume"]
    return summary


def _failure_summary(
    config: ExperimentConfig | None,
    exc: Exception,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ok": False,
        "error": str(exc),
        "error_type": type(exc).__name__,
    }
    if config is None:
        return summary

    results_path = config.output_path / "results.json"
    summary["output_dir"] = str(config.output_path)
    summary["results_path"] = str(results_path)
    if results_path.exists():
        trainer_results = _load_trainer_results(results_path)
        summary["trainer_results"] = trainer_results
        if isinstance(trainer_results, dict) and "error" in trainer_results:
            summary["trainer_error"] = trainer_results["error"]
    return summary


def _load_trainer_results(results_path: Path) -> Any:
    try:
        return json.loads(results_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - preserve original failure path.
        return {
            "ok": False,
            "error": f"failed to read trainer results: {exc}",
            "error_type": type(exc).__name__,
        }


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    raise SystemExit(main())
