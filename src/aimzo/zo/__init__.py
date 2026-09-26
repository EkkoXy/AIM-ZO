from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "MeZOTwoPointEstimator",
    "ParameterStateStore",
    "ZOEstimatorConfig",
    "apply_device_seeded_perturbation",
    "apply_device_seeded_update",
    "apply_seeded_perturbation",
    "lora_trainable_parameters",
    "parameter_selection_metadata",
    "select_trainable_parameters",
]


def __getattr__(name: str) -> Any:
    if name == "MeZOTwoPointEstimator":
        from aimzo.zo.estimator import MeZOTwoPointEstimator

        return MeZOTwoPointEstimator
    if name == "ZOEstimatorConfig":
        from aimzo.zo.config import ZOEstimatorConfig

        return ZOEstimatorConfig
    if name == "ParameterStateStore":
        from aimzo.zo.state import ParameterStateStore

        return ParameterStateStore
    if name in {
        "apply_device_seeded_perturbation",
        "apply_device_seeded_update",
        "apply_seeded_perturbation",
        "lora_trainable_parameters",
        "parameter_selection_metadata",
        "select_trainable_parameters",
    }:
        params = import_module("aimzo.zo.params")
        return getattr(params, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
