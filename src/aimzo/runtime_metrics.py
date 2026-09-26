from __future__ import annotations

import warnings
from typing import Any


def collect_cuda_memory() -> dict[str, int]:
    try:
        import torch
    except Exception:  # noqa: BLE001 - CUDA metrics are optional diagnostics.
        return {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not torch.cuda.is_available():
                return {}
            return {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "cuda_current_allocated_bytes": int(torch.cuda.memory_allocated()),
                "cuda_current_reserved_bytes": int(torch.cuda.memory_reserved()),
            }
    except Exception:  # noqa: BLE001 - never fail training on metrics collection.
        return {}


def merge_runtime_metrics(*sources: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    seen: set[int] = set()
    for source in sources:
        if source is None:
            continue
        identity = id(source)
        if identity in seen:
            continue
        seen.add(identity)
        runtime_metrics = getattr(source, "runtime_metrics", None)
        if not callable(runtime_metrics):
            continue
        payload = runtime_metrics()
        for key, value in dict(payload).items():
            if isinstance(value, int | float):
                merged[key] = merged.get(key, 0) + value
            else:
                merged.setdefault(key, value)
    merged.update(collect_cuda_memory())
    return merged
