#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from aimzo.config import load_config
from aimzo.reproduction import iter_experiments, load_experiment_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve and run one paper experiment")
    parser.add_argument("--registry", type=Path, default=Path("configs/main/experiments.yaml"))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model-path", help="override the public model placeholder")
    parser.add_argument("--data-root", help="override data/dataset")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    registry = load_experiment_registry(args.registry)
    matches = list(iter_experiments(registry, protocol_ids=[args.protocol], tasks=[args.task], seeds=[args.seed]))
    if len(matches) != 1:
        parser.error(f"expected one experiment, found {len(matches)}")
    protocol, config = matches[0]
    if args.model_path:
        config["backend"]["model_name"] = args.model_path
    if args.data_root:
        config["data"]["data_root"] = args.data_root
    if args.output_dir:
        config["trainer"]["output_dir"] = args.output_dir
    summary = {
        "protocol": protocol["id"],
        "model": protocol["model"],
        "method": protocol["method"],
        "task": args.task,
        "seed": args.seed,
        "dtype": config["backend"]["dtype"],
        "learning_rate": config["zo"]["learning_rate"],
        "epsilon": config["zo"]["eps"],
        "steps": config["trainer"]["max_steps"],
        "calls_per_step": protocol["calls_per_step"],
        "total_calls": int(config["trainer"]["max_steps"]) * int(protocol["calls_per_step"]),
        "checkpoints": config["trainer"]["checkpoint_milestones"],
        "output_dir": config["trainer"]["output_dir"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    with tempfile.TemporaryDirectory(prefix="aimzo-main-") as directory:
        config_path = Path(directory) / "resolved.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        load_config(config_path)
        return subprocess.call([sys.executable, "-m", "aimzo.cli.hf_zoregular_train", str(config_path)])


if __name__ == "__main__":
    raise SystemExit(main())
