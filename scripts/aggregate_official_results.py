#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _metrics(payload: dict[str, Any]) -> dict[str, float]:
    if "f1" in payload and "em" in payload:
        return {"f1": float(payload["f1"]), "em": float(payload["em"])}
    grouped = payload.get("official_grouped_metrics")
    if isinstance(grouped, dict):
        found = {key: float(value) for key, value in grouped.items() if key.lower() in {"f1", "f1a", "em", "exact_match"} and isinstance(value, (int, float))}
        if found:
            return found
    metrics = payload.get("metrics", payload)
    return {key: float(value) for key, value in metrics.items() if key in {"accuracy", "macro_f1", "f1", "f1a", "em", "exact_match"} and isinstance(value, (int, float))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate official evaluation JSON files across seeds")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [_metrics(json.loads(path.read_text(encoding="utf-8"))) for path in args.inputs]
    names = sorted(set.intersection(*(set(row) for row in rows))) if rows else []
    if not names:
        parser.error("input files have no common official metric")
    result = {"num_seeds": len(rows), "metrics": {}}
    for name in names:
        values = [row[name] for row in rows]
        result["metrics"][name] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
