from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch

from aimzo.config import HiZOOConfig, ZOConfig, normalize_hizoo_config
from aimzo.zo.params import ParameterList, select_trainable_parameters

from .base import (
    CandidateEvaluation,
    HFZOMethodStepResult,
    METHOD_SOURCE_COMMITS,
    ObjectiveFn,
    UpdateTraceAccumulator,
    bounded_tensor_fingerprint,
    evaluate_objective,
    method_config_diagnostics,
)


_MIN_HESSIAN = 1e-12


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


class HFHiZOOMethod:
    name = "hizoo"

    def __init__(self, config: ZOConfig) -> None:
        self.config = config
        self.hizoo_config = self._normalize_hizoo_config(config.hizoo)
        self._hessian: dict[str, torch.Tensor] = {}
        self._hessian_clamp_count = 0

    def step(
        self,
        model: torch.nn.Module,
        objective_fn: ObjectiveFn,
        *,
        seed: int,
        step: int,
    ) -> HFZOMethodStepResult:
        del step
        eps = float(self.config.eps)
        learning_rate = float(self.config.learning_rate)
        parameter_scope = self.config.parameter_scope
        parameters = select_trainable_parameters(model, parameter_scope)
        if not parameters:
            raise ValueError(
                f"No trainable parameters found for scope {parameter_scope!r}"
            )
        self._ensure_hessian_state(parameters)

        center = evaluate_objective("center", objective_fn)
        if self.config.source_compatibility_profile == "hizoo_source":
            self._apply_source_hessian_scaled_perturbation(
                parameters,
                seed=seed,
                eps=eps,
                scaling=1.0,
            )
            plus = evaluate_objective("plus", objective_fn)
            self._apply_source_hessian_scaled_perturbation(
                parameters,
                seed=seed,
                eps=eps,
                scaling=-2.0,
            )
            minus = evaluate_objective("minus", objective_fn)
            self._apply_source_hessian_scaled_perturbation(
                parameters,
                seed=seed,
                eps=eps,
                scaling=1.0,
            )
        else:
            plus = self._evaluate_candidate(
                parameters,
                objective_fn,
                candidate_id="plus",
                seed=seed,
                eps=eps,
                scaling=1.0,
            )
            minus = self._evaluate_candidate(
                parameters,
                objective_fn,
                candidate_id="minus",
                seed=seed,
                eps=eps,
                scaling=-1.0,
            )

        f_plus = plus.objective_value
        f_minus = minus.objective_value
        loss0 = center.objective_value
        update_skipped = not (
            center.ok
            and plus.ok
            and minus.ok
            and loss0 is not None
            and f_plus is not None
            and f_minus is not None
        )
        objective_delta = 0.0
        projected_grad = 0.0
        second_diff = 0.0
        update_trace = UpdateTraceAccumulator()
        if not update_skipped:
            if self.config.source_compatibility_profile == "hizoo_source":
                objective_delta = _source_objective_delta(f_plus, f_minus)
                projected_grad = _source_projected_grad(f_plus, f_minus, eps)
            else:
                objective_delta = float(f_plus) - float(f_minus)
                projected_grad = objective_delta / (2.0 * eps)
            second_diff = abs(float(f_plus) + float(f_minus) - 2.0 * float(loss0))
            self._apply_hizoo_update(
                parameters,
                seed=seed,
                eps=eps,
                loss0=float(loss0),
                f_plus=float(f_plus),
                f_minus=float(f_minus),
                projected_grad=projected_grad,
                second_diff=second_diff,
                learning_rate=learning_rate,
                weight_decay=float(self.config.weight_decay),
                update_trace=update_trace,
            )
        update_norm, parameter_delta_checksum = update_trace.finish()

        candidate_results = [
            center.to_dict(seed=seed, eps=eps, scaling=0.0),
            plus.to_dict(seed=seed, eps=eps, scaling=1.0),
            minus.to_dict(seed=seed, eps=eps, scaling=-1.0),
        ]
        return HFZOMethodStepResult(
            zo_method=self.name,
            seed=int(seed),
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
                "authority_source": "paper+official_source+zookit",
                "method_source_commit": METHOD_SOURCE_COMMITS[self.name],
                "estimator_mode": "hizoo_three_point",
                "method_config": method_config_diagnostics(
                    eps=eps,
                    learning_rate=learning_rate,
                    parameter_scope=parameter_scope,
                    weight_decay=float(self.config.weight_decay),
                    hizoo=self.hizoo_config,
                ),
                "objective_calls": 3,
                "num_objective_calls": 3,
                "loss0": None if loss0 is None else float(loss0),
                "second_diff": second_diff,
                "hessian_smooth": float(self.hizoo_config.hessian_smooth),
                "hessian_init": float(self.hizoo_config.hessian_init),
                "hessian_smooth_type": self.hizoo_config.hessian_smooth_type,
                **self._hessian_diagnostics(),
            },
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "hessian": {
                name: hessian.detach().clone()
                for name, hessian in self._hessian.items()
            },
            "hessian_clamp_count": int(self._hessian_clamp_count),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        hessian = state_dict.get("hessian", {})
        self._hessian = {
            str(name): tensor.detach().clone()
            for name, tensor in dict(hessian).items()
        }
        self._hessian_clamp_count = int(state_dict.get("hessian_clamp_count", 0))

    def _evaluate_candidate(
        self,
        parameters: ParameterList,
        objective_fn: ObjectiveFn,
        *,
        candidate_id: str,
        seed: int,
        eps: float,
        scaling: float,
    ) -> CandidateEvaluation:
        self._apply_hessian_scaled_perturbation(
            parameters,
            seed=seed,
            eps=eps,
            scaling=scaling,
        )
        try:
            return evaluate_objective(candidate_id, objective_fn)
        finally:
            self._apply_hessian_scaled_perturbation(
                parameters,
                seed=seed,
                eps=eps,
                scaling=-float(scaling),
            )

    @torch.no_grad()
    def _apply_hessian_scaled_perturbation(
        self,
        parameters: ParameterList,
        *,
        seed: int,
        eps: float,
        scaling: float,
    ) -> None:
        torch.manual_seed(int(seed))
        for name, param in parameters:
            z = torch.randn_like(param.data)
            hessian = self._hessian[name].to(device=param.device)
            scaled_noise = z.to(dtype=hessian.dtype).div(torch.sqrt(hessian))
            param.add_(
                scaled_noise.to(dtype=param.dtype),
                alpha=float(eps) * float(scaling),
            )

    @torch.no_grad()
    def _apply_hizoo_update(
        self,
        parameters: ParameterList,
        *,
        seed: int,
        eps: float,
        loss0: float,
        f_plus: float,
        f_minus: float,
        projected_grad: float,
        second_diff: float,
        learning_rate: float,
        weight_decay: float,
        update_trace: UpdateTraceAccumulator,
    ) -> None:
        smooth = float(self.hizoo_config.hessian_smooth)
        torch.manual_seed(int(seed))
        for name, param in parameters:
            z = torch.randn_like(param.data)
            hessian = self._hessian[name].to(device=param.device)
            if self.config.source_compatibility_profile == "hizoo_source":
                before = param.data.clone()
                state_dtype = hessian.dtype
                loss0_t = torch.tensor(float(loss0), dtype=state_dtype, device=param.device)
                loss1_t = torch.tensor(float(f_plus), dtype=state_dtype, device=param.device)
                loss2_t = torch.tensor(float(f_minus), dtype=state_dtype, device=param.device)
                z_state = z.to(dtype=state_dtype)
                h_temp = hessian * z_state * z_state
                h_est = (
                    torch.abs(loss1_t + loss2_t - 2 * loss0_t)
                    * h_temp
                    * float(smooth)
                    / (2 * float(eps) * float(eps))
                )
                hessian = ((1 - float(smooth)) * hessian + h_est)
                self._hessian[name] = hessian.detach().clone()
                grad = (
                    (loss1_t - loss2_t)
                    / (2 * float(eps))
                    * z_state
                    / torch.sqrt(hessian)
                )
                if weight_decay:
                    grad = grad.add(
                        param.data.to(dtype=state_dtype), alpha=float(weight_decay)
                    )
                param.add_(
                    grad.to(dtype=param.dtype), alpha=-float(learning_rate)
                )
                update_trace.add_delta(name, param.data - before, scale=1.0)
            else:
                hessian = self._clamp_hessian(hessian)
                h_temp = hessian * z * z
                h_est = h_temp.mul(
                    float(second_diff) * smooth / (2.0 * float(eps) * float(eps))
                )
                hessian = hessian.mul(1.0 - smooth).add(h_est)
                hessian = self._clamp_hessian(hessian)
                self._hessian[name] = hessian.detach().clone()

                update = (
                    z.to(dtype=hessian.dtype)
                    .div(torch.sqrt(hessian))
                    .mul(float(projected_grad))
                )
                if weight_decay:
                    update = update.add(param.data, alpha=float(weight_decay))
                scale = -float(learning_rate)
                update_trace.add_delta(name, update, scale=scale)
                param.add_(update.to(dtype=param.dtype), alpha=scale)

    @torch.no_grad()
    def _apply_source_hessian_scaled_perturbation(
        self,
        parameters: ParameterList,
        *,
        seed: int,
        eps: float,
        scaling: float,
    ) -> None:
        torch.manual_seed(int(seed))
        for name, param in parameters:
            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            hessian = self._hessian[name].to(device=param.device)
            scaled_noise = z.to(dtype=hessian.dtype).div(torch.sqrt(hessian))
            param.add_(
                scaled_noise.to(dtype=param.dtype),
                alpha=float(scaling) * float(eps),
            )

    def _ensure_hessian_state(self, parameters: ParameterList) -> None:
        hessian_init = float(self.hizoo_config.hessian_init)
        current_names = {name for name, _ in parameters}
        for name, param in parameters:
            state_dtype = (
                torch.float32
                if self.hizoo_config.hessian_state_dtype == "float32"
                else param.dtype
            )
            existing = self._hessian.get(name)
            if existing is None or tuple(existing.shape) != tuple(param.shape):
                self._hessian[name] = torch.full(
                    param.shape,
                    hessian_init,
                    device=param.device,
                    dtype=state_dtype,
                )
                continue
            self._hessian[name] = self._clamp_hessian(
                existing.to(device=param.device, dtype=state_dtype)
            )
        stale_names = set(self._hessian) - current_names
        for name in stale_names:
            del self._hessian[name]

    def _clamp_hessian(self, hessian: torch.Tensor) -> torch.Tensor:
        invalid = ~torch.isfinite(hessian) | (hessian <= 0)
        if invalid.any():
            self._hessian_clamp_count += int(invalid.sum().item())
            hessian = hessian.masked_fill(invalid, _MIN_HESSIAN)
        return hessian.clamp_min(_MIN_HESSIAN)

    def _hessian_state_checksum(self, summary: dict[str, Any]) -> str:
        encoded = json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _hessian_diagnostics(self) -> dict[str, float | int | str]:
        if not self._hessian:
            return {
                "hessian_min": math.nan,
                "hessian_max": math.nan,
                "hessian_mean": math.nan,
                "hessian_sample_count": 0,
                "hessian_summary_kind": "bounded_sample",
                "hessian_clamp_count": int(self._hessian_clamp_count),
                "hessian_state_checksum": hashlib.sha256(b"").hexdigest(),
            }
        sample_min = math.inf
        sample_max = -math.inf
        sample_sum = 0.0
        sample_count = 0
        hessian_count = 0
        parameter_summaries = []
        for name in sorted(self._hessian):
            hessian = self._hessian[name]
            count = int(hessian.numel())
            hessian_count += count
            if count == 0:
                parameter_summaries.append(
                    {
                        "name": name,
                        "shape": [int(dim) for dim in hessian.shape],
                        "dtype": str(hessian.dtype),
                        "device": str(hessian.device),
                        "count": 0,
                        "min": None,
                        "max": None,
                        "mean": None,
                    }
                )
                continue
            fingerprint, _ = bounded_tensor_fingerprint(
                name,
                hessian,
                error_prefix="hizoo hessian state",
            )
            sample = list(fingerprint["sample"])
            if sample:
                sample_min = min(sample_min, min(sample))
                sample_max = max(sample_max, max(sample))
                sample_sum += sum(sample)
                sample_count += len(sample)
            parameter_summaries.append(fingerprint)
        hessian_mean = sample_sum / sample_count if sample_count else math.nan
        hessian_min = sample_min if sample_count else math.nan
        hessian_max = sample_max if sample_count else math.nan
        summary = {
            "parameters": parameter_summaries,
            "hessian_min": hessian_min if sample_count else None,
            "hessian_max": hessian_max if sample_count else None,
            "hessian_mean": hessian_mean if sample_count else None,
            "hessian_count": hessian_count,
            "hessian_sample_count": sample_count,
            "hessian_clamp_count": int(self._hessian_clamp_count),
            "summary_kind": "bounded_sample",
        }
        return {
            "hessian_min": hessian_min,
            "hessian_max": hessian_max,
            "hessian_mean": hessian_mean,
            "hessian_sample_count": sample_count,
            "hessian_summary_kind": "bounded_sample",
            "hessian_clamp_count": int(self._hessian_clamp_count),
            "hessian_state_checksum": self._hessian_state_checksum(summary),
        }

    def _normalize_hizoo_config(
        self,
        hizoo: HiZOOConfig | dict[str, Any],
    ) -> HiZOOConfig:
        return normalize_hizoo_config(hizoo)
