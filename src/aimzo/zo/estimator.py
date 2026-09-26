from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math

import torch

from aimzo.zo.config import ZOEstimatorConfig
from aimzo.zo.params import (
    apply_seeded_perturbation,
    apply_seeded_update,
    lora_trainable_parameters,
    select_trainable_parameters,
)


@dataclass
class MeZOTwoPointEstimator:
    config: ZOEstimatorConfig
    _step_count: int = field(default=0, init=False, repr=False)

    def step(
        self,
        model: torch.nn.Module,
        objective_fn: Callable[[], float | torch.Tensor],
        *,
        step_seed: int | None = None,
    ) -> dict[str, float]:
        _validate_config(self.config)
        parameters = _selected_parameters(model, self.config.parameter_scope)

        seed = (
            int(step_seed)
            if step_seed is not None
            else int(self.config.seed) + self._step_count
        )
        eps = float(self.config.eps)

        center_parameters = [
            (param, param.detach().clone())
            for _, param in parameters
        ]
        try:
            apply_seeded_perturbation(
                model,
                seed=seed,
                eps=eps,
                scaling=1.0,
                parameter_scope=self.config.parameter_scope,
            )
            objective_plus = _objective_to_float(objective_fn())

            apply_seeded_perturbation(
                model,
                seed=seed,
                eps=eps,
                scaling=-2.0,
                parameter_scope=self.config.parameter_scope,
            )
            objective_minus = _objective_to_float(objective_fn())
        finally:
            _restore_parameters(center_parameters)

        _validate_finite("objective_plus", objective_plus)
        _validate_finite("objective_minus", objective_minus)
        projected_grad = (objective_plus - objective_minus) / (2.0 * eps)
        _validate_finite("projected_grad", projected_grad)
        self.apply_update(model, projected_grad=projected_grad, seed=seed)
        if step_seed is None:
            self._step_count += 1
        return {
            "seed": float(seed),
            "objective_plus": objective_plus,
            "objective_minus": objective_minus,
            "projected_grad": float(projected_grad),
        }

    def step_with_worker_evaluator(
        self,
        model: torch.nn.Module,
        evaluate_candidate: Callable[..., float | torch.Tensor],
        *,
        step_seed: int | None = None,
    ) -> dict[str, float]:
        _validate_config(self.config)
        _selected_parameters(model, self.config.parameter_scope)

        seed = (
            int(step_seed)
            if step_seed is not None
            else int(self.config.seed) + self._step_count
        )
        eps = float(self.config.eps)

        objective_plus = _objective_to_float(
            evaluate_candidate(
                seed=seed,
                eps=eps,
                scaling=1.0,
                candidate_id="plus",
            )
        )
        objective_minus = _objective_to_float(
            evaluate_candidate(
                seed=seed,
                eps=eps,
                scaling=-1.0,
                candidate_id="minus",
            )
        )

        _validate_finite("objective_plus", objective_plus)
        _validate_finite("objective_minus", objective_minus)
        projected_grad = (objective_plus - objective_minus) / (2.0 * eps)
        _validate_finite("projected_grad", projected_grad)
        self.apply_update(model, projected_grad=projected_grad, seed=seed)
        if step_seed is None:
            self._step_count += 1
        return {
            "seed": float(seed),
            "objective_plus": objective_plus,
            "objective_minus": objective_minus,
            "projected_grad": float(projected_grad),
        }

    def apply_update(
        self,
        model: torch.nn.Module,
        *,
        projected_grad: float,
        seed: int,
    ) -> None:
        _validate_config(self.config)
        _selected_parameters(model, self.config.parameter_scope)
        _validate_finite("projected_grad", float(projected_grad))
        apply_seeded_update(
            model,
            seed=seed,
            projected_grad=projected_grad,
            learning_rate=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
            parameter_scope=self.config.parameter_scope,
        )


def _objective_to_float(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


@torch.no_grad()
def _restore_parameters(
    center_parameters: list[tuple[torch.nn.Parameter, torch.Tensor]],
) -> None:
    for param, center in center_parameters:
        param.copy_(center)


def _validate_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _selected_parameters(
    model: torch.nn.Module,
    parameter_scope: str,
) -> list[tuple[str, torch.nn.Parameter]]:
    if parameter_scope == "lora":
        parameters = lora_trainable_parameters(model)
        if not parameters:
            raise ValueError("No trainable LoRA parameters found")
        return parameters
    parameters = select_trainable_parameters(model, parameter_scope)
    if not parameters:
        raise ValueError(f"No trainable parameters found for scope {parameter_scope!r}")
    return parameters


def _validate_config(config: ZOEstimatorConfig) -> None:
    eps = float(config.eps)
    learning_rate = float(config.learning_rate)
    weight_decay = float(config.weight_decay)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and > 0")
    if not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite")
    if not math.isfinite(weight_decay):
        raise ValueError("weight_decay must be finite")
