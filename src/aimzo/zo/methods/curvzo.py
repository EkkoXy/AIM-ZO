from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import torch

from aimzo.config import CurvZOConfig, ZOConfig
from aimzo.zo.params import ParameterList, select_trainable_parameters

from .base import (
    HFZOMethodStepResult,
    METHOD_SOURCE_COMMITS,
    ObjectiveFn,
    UpdateTraceAccumulator,
    evaluate_objective,
    method_config_diagnostics,
)


def _source_objective_delta(f_plus: float, f_minus: float) -> float:
    device = _source_scalar_device()
    plus = torch.tensor(float(f_plus), dtype=torch.float16, device=device)
    minus = torch.tensor(float(f_minus), dtype=torch.float16, device=device)
    return float((plus - minus).item())


def _source_projected_grad(f_plus: float, f_minus: float, eps: float) -> float:
    device = _source_scalar_device()
    plus = torch.tensor(float(f_plus), dtype=torch.float16, device=device)
    minus = torch.tensor(float(f_minus), dtype=torch.float16, device=device)
    return float(((plus - minus) / float(2.0 * eps)).item())


def _source_scalar_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class HFCurvZOMethod:
    name = "curvzo"

    def __init__(self, config: ZOConfig) -> None:
        self.config = config
        self.curvzo_config = self._normalize_curvzo_config(config.curvzo)
        self.sensitivity_list: list[float] = []
        self._sens_global_ema = 1e-8
        self._need_init_sens = True
        self.frac = float(self.curvzo_config.sample_ratio)
        self.alpha = float(self.curvzo_config.sgs_alpha)
        self._last_inclusion_probabilities: np.ndarray | None = None
        self._last_sampled_indices: list[int] = []

    def step(
        self,
        model: torch.nn.Module,
        objective_fn: ObjectiveFn,
        *,
        seed: int,
        step: int,
    ) -> HFZOMethodStepResult:
        eps = float(self.config.eps)
        learning_rate = float(self.config.learning_rate)
        parameter_scope = self.config.parameter_scope
        parameters = select_trainable_parameters(model, parameter_scope)
        if not parameters:
            raise ValueError(
                f"No trainable parameters found for scope {parameter_scope!r}"
            )
        self._ensure_sensitivity_state(parameters)

        used_ratio = float(self.frac)
        sampled_indices = self._sample_tensors(
            ratio=used_ratio,
            mode=str(self.curvzo_config.sampling_mode),
        )
        if self.config.source_compatibility_profile == "curvzo_source":
            zo_random_seed = int(np.random.randint(1000000000))
        else:
            zo_random_seed = int(seed)

        self._apply_sparse_perturbation(
            parameters,
            sampled_indices,
            seed=zo_random_seed,
            eps=eps,
            scaling=1.0,
        )
        plus = evaluate_objective("plus", objective_fn)
        self._apply_sparse_perturbation(
            parameters,
            sampled_indices,
            seed=zo_random_seed,
            eps=eps,
            scaling=-2.0,
        )
        minus = evaluate_objective("minus", objective_fn)
        self._apply_sparse_perturbation(
            parameters,
            sampled_indices,
            seed=zo_random_seed,
            eps=eps,
            scaling=1.0,
        )

        f_plus = plus.objective_value
        f_minus = minus.objective_value
        update_skipped = not (
            plus.ok and minus.ok and f_plus is not None and f_minus is not None
        )
        objective_delta = 0.0
        projected_grad = 0.0
        update_trace = UpdateTraceAccumulator()
        if not update_skipped:
            if self.config.source_compatibility_profile == "curvzo_source":
                objective_delta = _source_objective_delta(f_plus, f_minus)
                projected_grad = _source_projected_grad(f_plus, f_minus, eps)
            else:
                objective_delta = float(f_plus) - float(f_minus)
                projected_grad = objective_delta / (2.0 * eps)
            self._update_sensitivity_list(
                parameters,
                sampled_indices,
                seed=zo_random_seed,
                projected_grad_abs=abs(projected_grad),
                beta=float(self.curvzo_config.sensitivity_beta),
                normalize_by_numel=bool(
                    self.curvzo_config.normalize_energy_by_numel
                ),
            )
            adaptive_every = int(self.curvzo_config.adaptive_every)
            adaptive_index = (
                int(step) - 1
                if self.config.source_compatibility_profile == "curvzo_source"
                else int(step)
            )
            if adaptive_every > 0 and adaptive_index % adaptive_every == 0:
                self._adaptive_k_alpha()
            self._apply_sparse_update(
                parameters,
                sampled_indices,
                seed=zo_random_seed,
                projected_grad=projected_grad,
                learning_rate=learning_rate,
                weight_decay=float(self.config.weight_decay),
                update_trace=update_trace,
            )
        update_norm, parameter_delta_checksum = update_trace.finish()

        candidate_results = [
            plus.to_dict(seed=zo_random_seed, eps=eps, scaling=1.0),
            minus.to_dict(seed=zo_random_seed, eps=eps, scaling=-1.0),
        ]
        return HFZOMethodStepResult(
            zo_method=self.name,
            seed=zo_random_seed,
            eps=eps,
            learning_rate=learning_rate,
            f_plus=f_plus,
            f_minus=f_minus,
            objective_delta=objective_delta,
            projected_grad=projected_grad,
            candidate_results=candidate_results,
            update_norm=update_norm,
            parameter_delta_checksum=parameter_delta_checksum,
            method_diagnostics={
                "authority_source": "paper+source",
                "method_source_commit": METHOD_SOURCE_COMMITS[self.name],
                "estimator_mode": "curvzo_sparse_two_side",
                "method_config": method_config_diagnostics(
                    eps=eps,
                    learning_rate=learning_rate,
                    parameter_scope=parameter_scope,
                    weight_decay=float(self.config.weight_decay),
                    curvzo=self.curvzo_config,
                ),
                "objective_calls": 2,
                "num_objective_calls": 2,
                "seed_replay": True,
                "seed_source": (
                    "source_numpy_after_sampling"
                    if self.config.source_compatibility_profile == "curvzo_source"
                    else "trainer"
                ),
                "sampling_mode": str(self.curvzo_config.sampling_mode),
                "sample_ratio": used_ratio,
                "current_sample_ratio": float(self.frac),
                "sgs_alpha": float(self.alpha),
                "selected_tensor_count": len(parameters),
                "sampled_tensor_count": len(sampled_indices),
                "sampled_tensor_indices": list(sampled_indices),
                "sampled_tensor_checksum": _json_checksum(sampled_indices),
                "inclusion_probability_checksum": _json_checksum(
                    self._inclusion_probability_payload(parameters)
                ),
                "sensitivity_checksum": _json_checksum(self.sensitivity_list),
                "sensitivity_min": min(self.sensitivity_list),
                "sensitivity_max": max(self.sensitivity_list),
                "sensitivity_mean": sum(self.sensitivity_list)
                / len(self.sensitivity_list),
            },
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "sensitivity_list": list(self.sensitivity_list),
            "sens_global_ema": float(self._sens_global_ema),
            "need_init_sens": bool(self._need_init_sens),
            "frac": float(self.frac),
            "alpha": float(self.alpha),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.sensitivity_list = [
            float(value) for value in state_dict.get("sensitivity_list", [])
        ]
        self._sens_global_ema = float(state_dict.get("sens_global_ema", 1e-8))
        self._need_init_sens = bool(state_dict.get("need_init_sens", False))
        self.frac = float(state_dict.get("frac", self.curvzo_config.sample_ratio))
        self.alpha = float(state_dict.get("alpha", self.curvzo_config.sgs_alpha))

    def _ensure_sensitivity_state(self, parameters: ParameterList) -> None:
        if (
            not self._need_init_sens
            and len(self.sensitivity_list) == len(parameters)
        ):
            return
        init_val = float(self.curvzo_config.sensitivity_init)
        self.sensitivity_list = [init_val for _ in parameters]
        self._sens_global_ema = 1e-8
        self.frac = float(self.curvzo_config.sample_ratio)
        self.alpha = float(self.curvzo_config.sgs_alpha)
        self._need_init_sens = False

    def _sample_tensors(self, *, ratio: float, mode: str) -> list[int]:
        sens = np.asarray(self.sensitivity_list, dtype=np.float64)
        sens = np.clip(sens, 1e-12, None)
        count = int(sens.shape[0])
        if count <= 0:
            raise ValueError("CurvZO requires at least one trainable tensor")

        probabilities = sens / sens.sum()
        sample_count = int(np.floor(count * float(ratio)))
        sample_count = max(1, min(sample_count, count))

        if mode == "multinomial":
            sampled = np.random.choice(
                count, size=sample_count, replace=False, p=probabilities
            )
            inclusion = np.minimum(1.0, sample_count * probabilities)
        elif mode == "poisson_pps":
            lower = 0.0
            upper = 1.0 / (float(probabilities.max()) + 1e-18)
            for _ in range(40):
                mid = 0.5 * (lower + upper)
                proposed = np.minimum(1.0, mid * probabilities)
                if float(proposed.sum()) > sample_count:
                    upper = mid
                else:
                    lower = mid
            inclusion = np.minimum(1.0, lower * probabilities)
            mask = np.random.rand(count) < inclusion
            sampled = np.flatnonzero(mask)
            if sampled.size == 0:
                sampled = np.array([int(np.argmax(inclusion))], dtype=np.int64)
        elif mode == "uniform":
            sampled = np.random.choice(count, size=sample_count, replace=False)
            inclusion = np.full(count, sample_count / count, dtype=np.float64)
        else:
            raise ValueError(f"unknown CurvZO sampling mode: {mode}")

        sampled_indices = [int(index) for index in sampled.tolist()]
        inclusion_dtype = (
            np.float32
            if self.config.source_compatibility_profile == "curvzo_source"
            else np.float64
        )
        self._last_inclusion_probabilities = inclusion.astype(inclusion_dtype)
        self._last_sampled_indices = sampled_indices
        return sampled_indices

    @torch.no_grad()
    def _apply_sparse_perturbation(
        self,
        parameters: ParameterList,
        sampled_indices: list[int],
        *,
        seed: int,
        eps: float,
        scaling: float,
    ) -> None:
        torch.manual_seed(int(seed))
        for index in sampled_indices:
            _, param = parameters[index]
            z = torch.normal(
                mean=0.0,
                std=1.0,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            if self.config.source_compatibility_profile == "curvzo_source":
                param.data = param.data + float(scaling) * z * float(eps)
            else:
                param.add_(z, alpha=float(eps) * float(scaling))

    @torch.no_grad()
    def _update_sensitivity_list(
        self,
        parameters: ParameterList,
        sampled_indices: list[int],
        *,
        seed: int,
        projected_grad_abs: float,
        beta: float,
        normalize_by_numel: bool,
    ) -> None:
        grad_abs = float(abs(projected_grad_abs))
        global_beta = float(self.curvzo_config.sensitivity_global_beta)
        self._sens_global_ema = (
            (1.0 - global_beta) * self._sens_global_ema + global_beta * grad_abs
        )
        scale = grad_abs / (self._sens_global_ema + 1e-12)
        weights = self._compute_energy_weights(
            parameters,
            sampled_indices,
            seed=seed,
            normalize_by_numel=normalize_by_numel,
        )
        for index in sampled_indices:
            old_value = float(self.sensitivity_list[index])
            self.sensitivity_list[index] = float(
                (1.0 - beta) * old_value + beta * (scale * weights[index])
            )

    @torch.no_grad()
    def _compute_energy_weights(
        self,
        parameters: ParameterList,
        sampled_indices: list[int],
        *,
        seed: int,
        normalize_by_numel: bool,
    ) -> dict[int, float]:
        torch.manual_seed(int(seed))
        energies: list[float] = []
        indices: list[int] = []
        for index in sampled_indices:
            _, param = parameters[index]
            z = torch.normal(
                mean=0.0,
                std=1.0,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            ).to(torch.float64)
            pi = max(1e-8, self._inclusion_probability(index))
            energy = float(torch.sum(z * z).item()) / (pi * pi)
            if normalize_by_numel:
                energy /= max(1, int(param.numel()))
            energies.append(energy)
            indices.append(index)
        total = sum(energies) + 1e-12
        return {index: energy / total for index, energy in zip(indices, energies)}

    @torch.no_grad()
    def _apply_sparse_update(
        self,
        parameters: ParameterList,
        sampled_indices: list[int],
        *,
        seed: int,
        projected_grad: float,
        learning_rate: float,
        weight_decay: float,
        update_trace: UpdateTraceAccumulator,
    ) -> None:
        torch.manual_seed(int(seed))
        for index in sampled_indices:
            name, param = parameters[index]
            z = torch.normal(
                mean=0.0,
                std=1.0,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            pi = max(1e-8, self._inclusion_probability(index))
            if self.config.source_compatibility_profile == "curvzo_source":
                before = param.data.detach().clone()
                if _applies_weight_decay(name):
                    param.data = param.data - float(learning_rate) * ((float(projected_grad) / pi) * z + float(weight_decay) * param.data)
                else:
                    param.data = param.data - float(learning_rate) * ((float(projected_grad) / pi) * z)
                update_trace.add_delta(name, param.data - before, scale=1.0)
            else:
                update = z.mul(float(projected_grad) / pi)
                if weight_decay and _applies_weight_decay(name):
                    update = update.add(param.data, alpha=float(weight_decay))
                update_trace.add_delta(name, update, scale=-float(learning_rate))
                param.add_(update, alpha=-float(learning_rate))

    def _adaptive_k_alpha(self) -> None:
        sens = np.asarray(self.sensitivity_list, dtype=np.float64)
        sens = np.clip(sens, 1e-12, None)
        count = int(len(sens))
        eps = 1e-12
        values = sens / (float(sens.mean()) + eps)
        total = float(np.sum(values))
        sqrt_total = float(np.sum(np.sqrt(values)))
        effective_count = (sqrt_total**2) / (total + eps)
        q = values / (total + eps)
        entropy = -float(np.sum(q * np.log(q + eps)))
        normalized_entropy = entropy / math.log(count + eps)
        raw_fraction = 0.1 + (0.8 - 0.1) * (
            0.7 * (effective_count / count) + (1.0 - 0.7) * (1.0 - normalized_entropy)
        )
        raw_k = count * raw_fraction
        previous_k = float(self.frac) * count
        k = (1.0 - 0.1) * previous_k + 0.1 * raw_k
        k = float(np.clip(k, 0.1 * count, 0.8 * count))
        fraction = k / count
        alpha_entropy = 1.0 - normalized_entropy
        alpha_floor = 1.0 - count * 1e-4 / max(k, 1e-8)
        alpha_star = min(alpha_entropy, alpha_floor)
        self.frac = float(fraction)
        self.alpha = float(np.clip(alpha_star, 0.05, 0.5))

    def _inclusion_probability(self, index: int) -> float:
        if self._last_inclusion_probabilities is None:
            return 1.0
        return float(self._last_inclusion_probabilities[int(index)])

    def _inclusion_probability_payload(
        self,
        parameters: ParameterList,
    ) -> list[dict[str, float | int | str]]:
        return [
            {
                "index": int(index),
                "name": parameters[index][0],
                "pi": self._inclusion_probability(index),
            }
            for index in self._last_sampled_indices
        ]

    def _normalize_curvzo_config(
        self,
        value: CurvZOConfig | dict[str, Any],
    ) -> CurvZOConfig:
        if isinstance(value, CurvZOConfig):
            return value
        if isinstance(value, dict):
            return CurvZOConfig(**value)
        raise ValueError("zo.curvzo must be a mapping")


def _applies_weight_decay(name: str) -> bool:
    lowered = str(name).lower()
    return (
        "bias" not in lowered
        and "layer_norm" not in lowered
        and "layernorm" not in lowered
    )


def _json_checksum(payload: Any) -> str:
    safe_payload = _json_ready(payload)
    return hashlib.sha256(
        json.dumps(safe_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int | str | bool) or value is None:
        return value
    return str(value)
