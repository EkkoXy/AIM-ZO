from __future__ import annotations

from typing import Any

__all__ = ["RealVLLMZOTrainer", "ZOReasoningTrainer"]

_LAZY_EXPORTS = {
    "RealVLLMZOTrainer": ("aimzo.trainers.real_vllm", "RealVLLMZOTrainer"),
    "ZOReasoningTrainer": ("aimzo.trainers.reasoning", "ZOReasoningTrainer"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    attribute = getattr(import_module(module_name), attribute_name)
    globals()[name] = attribute
    return attribute
