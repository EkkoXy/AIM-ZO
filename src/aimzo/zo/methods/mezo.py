from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any

import torch

from aimzo.config import ZOConfig
from aimzo.zo.params import (
    apply_device_seeded_perturbation,
    select_trainable_parameters,
)

from .base import (
    CandidateEvaluation,
    HFZOMethodStepResult,
    METHOD_SOURCE_COMMITS,
    ObjectiveFn,
    UpdateTraceAccumulator,
    evaluate_objective,
    method_config_diagnostics,
)


def _parameter_selection_checksum(
    parameters: list[tuple[str, torch.nn.Parameter]],
) -> str:
    payload = [
        {
            "name": name,
            "shape": tuple(int(dim) for dim in param.shape),
            "numel": int(param.numel()),
        }
        for name, param in parameters
    ]
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _parameter_value_sample_checksum(
    parameters: list[tuple[str, torch.nn.Parameter]],
) -> str:
    digest = hashlib.sha256()
    for name, param in parameters:
        flat = param.detach().reshape(-1)
        numel = int(flat.numel())
        if numel == 0:
            sample_values: list[float] = []
        else:
            sample_count = min(8, numel)
            if sample_count == 1:
                indices = [0]
            else:
                indices = [
                    round(index * (numel - 1) / (sample_count - 1))
                    for index in range(sample_count)
                ]
            sample_values = [
                float(flat[index].detach().cpu().item()) for index in indices
            ]
        payload = {
            "name": name,
            "shape": [int(dim) for dim in param.shape],
            "dtype": str(param.dtype),
            "numel": numel,
            "sample": sample_values,
        }
        digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _dict_mean(values: dict[str, float]) -> float | None:
    finite_values = [
        float(value)
        for value in values.values()
        if isinstance(value, (float, int)) and math.isfinite(float(value))
    ]
    if not finite_values:
        return None
    return sum(finite_values) / float(len(finite_values))


def _source_objective_delta(f_plus: float, f_minus: float) -> float:
    device = _source_scalar_device()
    plus = torch.tensor(float(f_plus), dtype=torch.float16, device=device)
    minus = torch.tensor(float(f_minus), dtype=torch.float16, device=device)
    return float((plus - minus).item())


def _trace_parameter_values_enabled() -> bool:
    return os.environ.get("AIMZO_TRACE_PARAMETER_VALUES", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _source_projected_grad(f_plus: float, f_minus: float, eps: float) -> float:
    device = _source_scalar_device()
    plus = torch.tensor(float(f_plus), dtype=torch.float16, device=device)
    minus = torch.tensor(float(f_minus), dtype=torch.float16, device=device)
    return float(((plus - minus) / float(2.0 * eps)).item())


def _source_scalar_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class HFMeZOMethod:
    name = "mezo"

    def __init__(self, config: ZOConfig) -> None:
        self.config = config
        self._loren_a: dict[str, torch.Tensor] = {}
        self._loren_momentum: dict[str, torch.Tensor] = {}
        self._loren_a_norm: dict[str, float] = {}
        self._loren_a_grad_norm: dict[str, float] = {}
        self._loren_a_delta_ratio: dict[str, float] = {}
        self._loren_noise_scale: dict[str, float] = {}

    def step(
        self,
        model: torch.nn.Module,
        objective_fn: ObjectiveFn,
        *,
        seed: int,
        step: int,
    ) -> HFZOMethodStepResult:
        original_seed = int(seed)
        fixed_pool = self._fixed_seed_pool()
        fixed_seed_pool_mode = None
        if fixed_pool:
            seed = self._fixed_pool_seed(base_seed=original_seed, step=step, pool=fixed_pool)
            fixed_seed_pool_mode = "offline_fixed_seed_pool"
        eps = float(self.config.eps)
        learning_rate = float(self.config.learning_rate)
        parameter_scope = self.config.parameter_scope
        parameters = select_trainable_parameters(model, parameter_scope)
        if not parameters:
            raise ValueError(
                f"No trainable parameters found for scope {parameter_scope!r}"
            )

        if self.config.source_compatibility_profile == "mezo_source":
            self._apply_source_perturbation(parameters, seed=seed, eps=eps, scaling=1.0)
            plus = evaluate_objective("plus", objective_fn)
            self._apply_source_perturbation(parameters, seed=seed, eps=eps, scaling=-2.0)
            minus = evaluate_objective("minus", objective_fn)
            self._apply_source_perturbation(parameters, seed=seed, eps=eps, scaling=1.0)
        else:
            plus = self._evaluate_candidate(
                model,
                objective_fn,
                candidate_id="plus",
                seed=seed,
                eps=eps,
                scaling=1.0,
                parameter_scope=parameter_scope,
            )
            minus = self._evaluate_candidate(
                model,
                objective_fn,
                candidate_id="minus",
                seed=seed,
                eps=eps,
                scaling=-1.0,
                parameter_scope=parameter_scope,
            )

        f_plus = plus.objective_value
        f_minus = minus.objective_value
        update_skipped = not (
            plus.ok and minus.ok and f_plus is not None and f_minus is not None
        )
        objective_delta = 0.0
        projected_grad = 0.0
        update_trace = UpdateTraceAccumulator()
        trace_parameter_values = _trace_parameter_values_enabled()
        selected_parameter_value_checksum_before_update = None
        if trace_parameter_values:
            selected_parameter_value_checksum_before_update = (
                _parameter_value_sample_checksum(parameters)
            )
        if str(self.config.estimator) == "mezo_population":
            if self.config.source_compatibility_profile == "mezo_source":
                raise ValueError("mezo_population does not support mezo_source profile")
            num_noise = max(2, int(getattr(self.config, "num_noise", 1)))
            candidates: list[CandidateEvaluation] = []
            objective_values: list[float] = []
            probe_seeds = [
                int(seed) + 1_000_003 * int(index)
                for index in range(num_noise)
            ]
            for index, probe_seed in enumerate(probe_seeds):
                if bool(getattr(self.config, "mezo_loren_enabled", False)):
                    candidate = self._evaluate_loren_candidate(
                        parameters,
                        objective_fn,
                        candidate_id=f"population_{index}",
                        seed=probe_seed,
                        eps=eps,
                        scaling=1.0,
                    )
                else:
                    candidate = self._evaluate_candidate(
                        model,
                        objective_fn,
                        candidate_id=f"population_{index}",
                        seed=probe_seed,
                        eps=eps,
                        scaling=1.0,
                        parameter_scope=parameter_scope,
                    )
                candidates.append(candidate)
                if candidate.ok and candidate.objective_value is not None:
                    objective_values.append(float(candidate.objective_value))
            update_skipped = len(objective_values) != num_noise
            f_mean = (
                sum(objective_values) / float(len(objective_values))
                if objective_values
                else None
            )
            projected_grads: list[tuple[int, float]] = []
            if not update_skipped and f_mean is not None:
                eps_denom = (
                    float(eps)
                    if bool(getattr(self.config, "mezo_population_divide_by_eps", True))
                    else 1.0
                )
                denom = eps_denom * float(max(1, num_noise - 1))
                for probe_seed, value in zip(
                    probe_seeds,
                    objective_values,
                    strict=False,
                ):
                    projected = (float(value) - float(f_mean)) / denom
                    projected_grads.append((int(probe_seed), float(projected)))
                    if bool(getattr(self.config, "mezo_loren_enabled", False)):
                        self._apply_loren_update(
                            parameters,
                            seed=int(probe_seed),
                            projected_grad=float(projected),
                            fweight=float(value) - float(f_mean),
                            eps=eps,
                            num_noise=num_noise,
                            learning_rate=learning_rate,
                            weight_decay=float(self.config.weight_decay),
                            update_trace=update_trace,
                        )
                    else:
                        self._apply_mezo_update(
                            parameters,
                            seed=int(probe_seed),
                            projected_grad=float(projected),
                            learning_rate=learning_rate,
                            weight_decay=float(self.config.weight_decay),
                            update_trace=update_trace,
                        )
                objective_delta = 0.0
                projected_grad = (
                    sum(value for _, value in projected_grads)
                    / float(len(projected_grads))
                )
            update_norm, parameter_delta_checksum = update_trace.finish()
            candidate_results = [
                candidate.to_dict(seed=probe_seed, eps=eps, scaling=1.0)
                for candidate, probe_seed in zip(
                    candidates,
                    probe_seeds,
                    strict=False,
                )
            ]
            return HFZOMethodStepResult(
                zo_method=self.name,
                seed=int(seed),
                eps=eps,
                learning_rate=learning_rate,
                f_plus=f_mean,
                f_minus=None,
                objective_delta=objective_delta,
                projected_grad=projected_grad,
                candidate_results=candidate_results,
                update_norm=update_norm,
                parameter_delta_checksum=parameter_delta_checksum,
                method_diagnostics={
                    "authority_source": "paper+source+zookit+es_population_ablation",
                    "method_source_commit": METHOD_SOURCE_COMMITS[self.name],
                    "estimator_mode": "population_rloo",
                    "method_config": method_config_diagnostics(
                        eps=eps,
                        learning_rate=learning_rate,
                        parameter_scope=parameter_scope,
                        weight_decay=float(self.config.weight_decay),
                        population={"num_noise": int(num_noise)},
                        mezo_population={
                            "divide_by_eps": bool(
                                getattr(
                                    self.config,
                                    "mezo_population_divide_by_eps",
                                    True,
                                )
                            ),
                            "loren_enabled": bool(
                                getattr(self.config, "mezo_loren_enabled", False)
                            ),
                            "loren_damping": float(
                                getattr(self.config, "mezo_loren_damping", 0.1)
                            ),
                            "loren_lr_cov": float(
                                getattr(self.config, "mezo_loren_lr_cov", 1e-3)
                            ),
                            "loren_beta1": float(
                                getattr(self.config, "mezo_loren_beta1", 0.9)
                            ),
                        },
                    ),
                    "objective_calls": int(num_noise),
                    "num_objective_calls": int(num_noise),
                    "seed_replay": True,
                    "original_seed": int(original_seed),
                    "population_mean_loss": f_mean,
                    "population_projected_grads": [
                        float(value) for _, value in projected_grads
                    ],
                    "population_projected_grad_mean": (
                        float(projected_grad) if projected_grads else None
                    ),
                    "selected_parameter_checksum": _parameter_selection_checksum(
                        parameters
                    ),
                    "selected_parameter_value_checksum_before_update": (
                        selected_parameter_value_checksum_before_update
                    ),
                    "selected_parameter_value_checksum_after_update": (
                        _parameter_value_sample_checksum(parameters)
                        if trace_parameter_values
                        else None
                    ),
                    "selected_tensor_count": len(parameters),
                    "selected_parameter_count": sum(
                        int(param.numel()) for _, param in parameters
                    ),
                    "loren_a_parameters": len(self._loren_a),
                    "loren_momentum_parameters": len(self._loren_momentum),
                    "loren_a_norm_mean": _dict_mean(self._loren_a_norm),
                    "loren_a_grad_norm_mean": _dict_mean(self._loren_a_grad_norm),
                    "loren_a_delta_ratio_mean": _dict_mean(
                        self._loren_a_delta_ratio
                    ),
                    "loren_noise_scale_mean": _dict_mean(self._loren_noise_scale),
                },
            )
        if not update_skipped:
            if self.config.source_compatibility_profile == "mezo_source":
                objective_delta = _source_objective_delta(f_plus, f_minus)
                projected_grad = _source_projected_grad(f_plus, f_minus, eps)
            else:
                objective_delta = float(f_plus) - float(f_minus)
                denom = 2.0 * (
                    float(eps)
                    if bool(getattr(self.config, "mezo_two_point_divide_by_eps", True))
                    else 1.0
                )
                projected_grad = objective_delta / denom
            self._apply_mezo_update(
                parameters,
                seed=seed,
                projected_grad=projected_grad,
                learning_rate=learning_rate,
                weight_decay=float(self.config.weight_decay),
                update_trace=update_trace,
            )
        update_norm, parameter_delta_checksum = update_trace.finish()

        candidate_results = [
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
                "authority_source": "paper+source+zookit",
                "method_source_commit": METHOD_SOURCE_COMMITS[self.name],
                "estimator_mode": "two_side",
                "method_config": method_config_diagnostics(
                    eps=eps,
                    learning_rate=learning_rate,
                    parameter_scope=parameter_scope,
                    weight_decay=float(self.config.weight_decay),
                    mezo_two_point={
                        "divide_by_eps": bool(
                            getattr(
                                self.config,
                                "mezo_two_point_divide_by_eps",
                                True,
                            )
                        )
                    },
                ),
                "objective_calls": 2,
                "num_objective_calls": 2,
                "seed_replay": True,
                "original_seed": int(original_seed),
                "fixed_seed_pool_mode": fixed_seed_pool_mode,
                "fixed_seed_pool_size": len(fixed_pool),
                "fixed_seed_pool_selected_seed": (
                    int(seed) if fixed_seed_pool_mode is not None else None
                ),
                "selected_parameter_checksum": _parameter_selection_checksum(
                    parameters
                ),
                "selected_parameter_value_checksum_before_update": (
                    selected_parameter_value_checksum_before_update
                ),
                "selected_parameter_value_checksum_after_update": (
                    _parameter_value_sample_checksum(parameters)
                    if trace_parameter_values
                    else None
                ),
                "selected_tensor_count": len(parameters),
                "selected_parameter_count": sum(
                    int(param.numel()) for _, param in parameters
                ),
            },
        )

    @torch.no_grad()
    def _apply_mezo_update(
        self,
        parameters: list[tuple[str, torch.nn.Parameter]],
        *,
        seed: int,
        projected_grad: float,
        learning_rate: float,
        weight_decay: float,
        update_trace: UpdateTraceAccumulator,
    ) -> None:
        torch.manual_seed(int(seed))
        for name, param in parameters:
            z = torch.randn_like(param.data)
            if self.config.source_compatibility_profile == "mezo_source":
                before = param.data.clone()
                if weight_decay:
                    param.data = param.data - float(learning_rate) * (float(projected_grad) * z + float(weight_decay) * param.data)
                else:
                    param.data = param.data - float(learning_rate) * (float(projected_grad) * z)
                update_trace.add_delta(name, param.data - before, scale=1.0)
            else:
                update = z.mul(float(projected_grad))
                if weight_decay:
                    update = update.add(param.data, alpha=float(weight_decay))
                scale = -float(learning_rate)
                update_trace.add_delta(name, update, scale=scale)
                param.add_(update, alpha=scale)

    @torch.no_grad()
    def _evaluate_loren_candidate(
        self,
        parameters: list[tuple[str, torch.nn.Parameter]],
        objective_fn: ObjectiveFn,
        *,
        candidate_id: str,
        seed: int,
        eps: float,
        scaling: float,
    ) -> CandidateEvaluation:
        self._apply_loren_perturbation(
            parameters,
            seed=seed,
            eps=eps,
            scaling=scaling,
        )
        try:
            return evaluate_objective(candidate_id, objective_fn)
        finally:
            self._apply_loren_perturbation(
                parameters,
                seed=seed,
                eps=eps,
                scaling=-float(scaling),
            )

    def _loren_a_key(self, name: str, param: torch.nn.Parameter) -> str:
        return ":".join(
            [
                str(name),
                "in" if param.data.ndim >= 2 else "flat",
                str(int(param.data.shape[1]) if param.data.ndim >= 2 else int(param.data.numel())),
                str(param.device),
            ]
        )

    def _loren_a_vector(self, name: str, param: torch.nn.Parameter) -> torch.Tensor:
        key = self._loren_a_key(name, param)
        dim = int(param.data.shape[1]) if param.data.ndim >= 2 else int(param.data.numel())
        cached = self._loren_a.get(key)
        if cached is not None and tuple(cached.shape) == (dim,):
            return cached.to(device=param.device, dtype=torch.float32)
        std = float(getattr(self.config, "mezo_loren_a_init_std", 1.0))
        seed_payload = {
            "method": "mezo_loren_a",
            "name": str(name),
            "dim": int(dim),
            "device": str(param.device),
        }
        seed = int(
            hashlib.sha256(json.dumps(seed_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16],
            16,
        ) % (2**63 - 1)
        generator = torch.Generator(device=param.device)
        generator.manual_seed(seed)
        vector = torch.randn(
            (dim,),
            device=param.device,
            dtype=torch.float32,
            generator=generator,
        ).mul(std)
        self._loren_a[key] = vector.detach().float().contiguous()
        norm = float(vector.norm().item())
        self._loren_a_norm[str(name)] = norm
        return vector

    def _loren_noise_and_stats(
        self,
        name: str,
        param: torch.nn.Parameter,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, int]:
        u = torch.randn_like(param.data).to(dtype=torch.float32)
        a = self._loren_a_vector(name, param)
        damping = float(getattr(self.config, "mezo_loren_damping", 0.1))
        if param.data.ndim >= 2:
            rows = u.reshape(-1, int(param.data.shape[1]))
        else:
            rows = u.reshape(1, -1)
        row_count = max(1, int(rows.shape[0]))
        sq_norm_a = torch.dot(a, a)
        flat_sq_norm_a = sq_norm_a * float(row_count)
        if float(flat_sq_norm_a.item()) <= 1e-12:
            z = u
            dot_au = torch.tensor(0.0, device=param.device, dtype=torch.float32)
            damping_sq_norm_a = math.sqrt(max(damping, 1e-12))
        else:
            dot_au = rows.matmul(a).sum()
            damping_sq_norm_a = math.sqrt(
                max(damping + float(flat_sq_norm_a.item()), 1e-12)
            )
            alpha = (
                (math.sqrt(max(damping, 1e-12)) + damping_sq_norm_a)
                * dot_au
                / (flat_sq_norm_a * damping_sq_norm_a)
            )
            z = rows.sub(alpha.mul(a).unsqueeze(0)).reshape_as(u)
        u_norm = float(u.norm().item())
        z_norm = float(z.norm().item())
        if u_norm > 0.0:
            self._loren_noise_scale[str(name)] = z_norm / u_norm
        self._loren_a_norm[str(name)] = float(a.norm().item())
        return z.to(device=param.device, dtype=param.dtype), u, dot_au, float(flat_sq_norm_a.item()), damping_sq_norm_a, row_count

    @torch.no_grad()
    def _apply_loren_perturbation(
        self,
        parameters: list[tuple[str, torch.nn.Parameter]],
        *,
        seed: int,
        eps: float,
        scaling: float,
    ) -> None:
        torch.manual_seed(int(seed))
        for name, param in parameters:
            z, *_ = self._loren_noise_and_stats(name, param)
            param.add_(z, alpha=float(eps) * float(scaling))

    @torch.no_grad()
    def _apply_loren_update(
        self,
        parameters: list[tuple[str, torch.nn.Parameter]],
        *,
        seed: int,
        projected_grad: float,
        fweight: float,
        eps: float,
        num_noise: int,
        learning_rate: float,
        weight_decay: float,
        update_trace: UpdateTraceAccumulator,
    ) -> None:
        torch.manual_seed(int(seed))
        denom = float(max(1, int(num_noise) - 1))
        lr_cov = float(getattr(self.config, "mezo_loren_lr_cov", 1e-3))
        beta1 = float(getattr(self.config, "mezo_loren_beta1", 0.9))
        damping = float(getattr(self.config, "mezo_loren_damping", 0.1))
        damping_sqrt = math.sqrt(max(damping, 1e-12))
        for name, param in parameters:
            z, u, dot_au, flat_sq_norm_a, damping_sq_norm_a, row_count = (
                self._loren_noise_and_stats(name, param)
            )
            gx = z.to(dtype=torch.float32).mul(float(projected_grad))
            if param.data.ndim >= 2:
                key = str(name)
                previous = self._loren_momentum.get(key)
                if previous is None or tuple(previous.shape) != tuple(param.data.shape):
                    momentum = torch.zeros_like(param.data, dtype=torch.float32)
                else:
                    momentum = previous.to(device=param.device, dtype=torch.float32)
                momentum.mul_(beta1).add_(gx)
                self._loren_momentum[key] = momentum.detach().float().contiguous()
                gx = momentum
            update = gx.to(device=param.device, dtype=param.dtype)
            if weight_decay:
                update = update.add(param.data, alpha=float(weight_decay))
            update_trace.add_delta(name, update, scale=-float(learning_rate))
            param.add_(update, alpha=-float(learning_rate))

            if lr_cov <= 0.0:
                continue
            a = self._loren_a_vector(name, param)
            if flat_sq_norm_a <= 1e-12:
                continue
            if param.data.ndim >= 2:
                rows = u.reshape(-1, int(param.data.shape[1]))
            else:
                rows = u.reshape(1, -1)
            c1_sum = dot_au.mul(rows.sum(dim=0)).sub(a.mul(float(row_count)))
            c2_scale = (
                (damping_sqrt + damping_sq_norm_a)
                * (dot_au.square() - float(flat_sq_norm_a))
                / (float(flat_sq_norm_a) * float(damping_sq_norm_a))
            )
            c2_sum = a.mul(float(row_count)).mul(c2_scale)
            a_grad = (
                float(fweight)
                * (float(eps) ** float(getattr(self.config, "mezo_loren_a_eps_power", 2.0)))
                * (c1_sum - c2_sum)
                / float(damping_sq_norm_a)
                / denom
            )
            old_norm = float(a.norm().item())
            delta = a_grad.mul(-lr_cov)
            key = self._loren_a_key(name, param)
            self._loren_a[key] = a.add(delta).detach().float().contiguous()
            self._loren_a_grad_norm[str(name)] = float(a_grad.norm().item())
            self._loren_a_delta_ratio[str(name)] = float(delta.norm().item()) / max(
                old_norm,
                1e-12,
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "loren_a": {
                key: value.detach().cpu()
                for key, value in self._loren_a.items()
            },
            "loren_momentum": {
                key: value.detach().cpu()
                for key, value in self._loren_momentum.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        loren_a = state_dict.get("loren_a", {})
        if isinstance(loren_a, dict):
            self._loren_a = {
                str(key): value.detach().float().contiguous()
                for key, value in loren_a.items()
                if isinstance(value, torch.Tensor)
            }
        loren_momentum = state_dict.get("loren_momentum", {})
        if isinstance(loren_momentum, dict):
            self._loren_momentum = {
                str(key): value.detach().float().contiguous()
                for key, value in loren_momentum.items()
                if isinstance(value, torch.Tensor)
            }

    def _fixed_seed_pool(self) -> list[int]:
        raw = str(getattr(self.config, "fixed_seed_pool", "") or "")
        seeds: list[int] = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            seeds.append(int(item))
        return seeds

    def _fixed_pool_seed(self, *, base_seed: int, step: int, pool: list[int]) -> int:
        payload = {
            "method": "mezo_fixed_seed_pool_offset",
            "base_seed": int(base_seed),
            "step": int(step),
            "pool_size": int(len(pool)),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
        offset = int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)
        return int(pool[int(offset % len(pool))])

    def _evaluate_candidate(
        self,
        model: torch.nn.Module,
        objective_fn: ObjectiveFn,
        *,
        candidate_id: str,
        seed: int,
        eps: float,
        scaling: float,
        parameter_scope: str,
    ) -> CandidateEvaluation:
        apply_device_seeded_perturbation(
            model,
            seed=seed,
            eps=eps,
            scaling=scaling,
            parameter_scope=parameter_scope,
        )
        try:
            return evaluate_objective(candidate_id, objective_fn)
        finally:
            apply_device_seeded_perturbation(
                model,
                seed=seed,
                eps=eps,
                scaling=-float(scaling),
                parameter_scope=parameter_scope,
            )

    @torch.no_grad()
    def _apply_source_perturbation(
        self,
        parameters: list[tuple[str, torch.nn.Parameter]],
        *,
        seed: int,
        eps: float,
        scaling: float,
    ) -> None:
        torch.manual_seed(int(seed))
        for _, param in parameters:
            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            param.data = param.data + float(scaling) * z * float(eps)
