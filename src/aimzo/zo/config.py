from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZOEstimatorConfig:
    eps: float = 1e-3
    learning_rate: float = 1e-3
    seed: int = 42
    weight_decay: float = 0.0
    parameter_scope: str = "full_parameters"
