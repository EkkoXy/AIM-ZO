#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from aimzo.reproduction import (
    experiment_relative_path,
    iter_experiments,
    load_experiment_registry,
)


def _csv(value: str | None) -> list[str]:
    return [] if not value else [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate resolved paper experiment YAML files")
    parser.add_argument("--registry", type=Path, default=Path("configs/main/experiments.yaml"))
    parser.add_argument("--output", type=Path, default=Path("configs/main/generated"))
    parser.add_argument("--protocol")
    parser.add_argument("--model")
    parser.add_argument("--method")
    parser.add_argument("--task")
    parser.add_argument("--seed")
    parser.add_argument("--check", action="store_true", help="validate and list without writing")
    args = parser.parse_args()
    registry = load_experiment_registry(args.registry)
    experiments = list(iter_experiments(
        registry,
        protocol_ids=_csv(args.protocol),
        models=_csv(args.model),
        methods=_csv(args.method),
        tasks=_csv(args.task),
        seeds=[int(value) for value in _csv(args.seed)],
    ))
    if not experiments:
        parser.error("filters matched no experiments")
    for protocol, config in experiments:
        relative = experiment_relative_path(protocol, config)
        if not args.check:
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(relative)
    print(f"validated={len(experiments)} written={0 if args.check else len(experiments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
