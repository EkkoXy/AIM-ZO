from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aimzo.cli.hf_zoregular_train import build_backend, validate_cli_config
from aimzo.config import load_config
from aimzo.tasks.qa_metrics import (
    postprocess_qa_prediction,
    qa_exact_match,
    qa_f1,
)
from aimzo.tasks.registry import build_task
from aimzo.tasks.zoregular import ZORegularSample, load_zoregular_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate SQuAD/DROP checkpoints with generation F1 and EM"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Omit for zero-shot")
    parser.add_argument("--scope", choices=("dev", "official"), default="official")
    parser.add_argument("--max-eval-samples", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-examples", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _evaluation_rows(
    config: Any,
    *,
    scope: str,
    max_eval_samples: int,
    eval_seed: int,
) -> tuple[list[ZORegularSample], list[int] | None]:
    task = build_task(config.data)
    if scope == "dev":
        return list(task.load_train_dev()), None
    rows = load_zoregular_split(config.data.task, config.data.data_root, "validation")
    count = min(max(0, int(max_eval_samples)), len(rows))
    indices = np.random.RandomState(int(eval_seed)).permutation(len(rows))[:count]
    return [rows[int(index)] for index in indices], [int(index) for index in indices]


def _payload_checksum(rows: list[ZORegularSample]) -> str:
    payload = [
        {"data": row.data, "correct_candidate": row.correct_candidate} for row in rows
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    validate_cli_config(config)
    if config.data.task not in {"squad", "drop"}:
        raise ValueError("QA metric evaluation supports only squad and drop")
    task = build_task(config.data)
    rows, source_indices = _evaluation_rows(
        config,
        scope=args.scope,
        max_eval_samples=args.max_eval_samples,
        eval_seed=args.eval_seed,
    )

    backend = build_backend(config)
    if args.checkpoint is not None:
        backend.load_checkpoint(args.checkpoint)
    tokenizer = backend.tokenizer
    model = backend.model
    tokenizer.padding_side = "left"
    max_model_len = int(
        args.max_model_len or getattr(config.backend, "max_model_len", None) or 1024
    )

    total_f1 = 0.0
    total_em = 0.0
    examples: list[dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [task.template.encode(row) for row in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=max_model_len,
            padding=True,
        )
        input_ids = encoded["input_ids"].to(backend.device)
        attention_mask = encoded["attention_mask"].to(backend.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=int(args.max_new_tokens),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = int(input_ids.shape[1])
        for offset, row in enumerate(batch):
            raw = tokenizer.decode(
                output_ids[offset, prompt_length:],
                skip_special_tokens=True,
            ).strip()
            prediction = postprocess_qa_prediction(raw)
            answers = [str(answer) for answer in list(row.correct_candidate)]
            f1 = qa_f1(prediction, answers)
            em = qa_exact_match(prediction, answers)
            total_f1 += f1
            total_em += em
            evaluation_index = start + offset
            if int(args.save_examples) < 0 or evaluation_index < int(
                args.save_examples
            ):
                examples.append(
                    {
                        "evaluation_index": evaluation_index,
                        "source_index": (
                            source_indices[evaluation_index]
                            if source_indices is not None
                            else None
                        ),
                        "prediction": prediction,
                        "raw_prediction": raw,
                        "answers": answers,
                        "f1": f1,
                        "em": em,
                    }
                )

    count = len(rows)
    return {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "scope": args.scope,
        "eval_seed": int(args.eval_seed) if args.scope == "official" else None,
        "num_examples": count,
        "payload_checksum": _payload_checksum(rows),
        "source_indices": source_indices if args.scope == "official" else None,
        "f1": total_f1 / count if count else 0.0,
        "em": total_em / count if count else 0.0,
        "examples": examples,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_checkpoint(args)
    output = args.output or (
        args.checkpoint / "official_qa_eval.json"
        if args.checkpoint is not None
        else load_config(args.config).output_path / "zero_shot_qa_eval.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("scope", "num_examples", "f1", "em", "payload_checksum")
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
