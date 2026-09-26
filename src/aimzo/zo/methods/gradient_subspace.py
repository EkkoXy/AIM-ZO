from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from typing import Any

import torch

from aimzo.config import PGAPConfig, SVD0Config, ZOConfig
from aimzo.zo.params import ParameterList, select_trainable_parameters

from .base import (
    HFZOMethodStepResult,
    METHOD_SOURCE_COMMITS,
    ObjectiveFn,
    UpdateTraceAccumulator,
    evaluate_objective,
    method_config_diagnostics,
)


class HFGradientSubspaceMethod:
    """Paper-derived SVD-0/P-GAP implementation for HF ZORegular training."""

    def __init__(self, config: ZOConfig, *, name: str) -> None:
        if name not in {"svd0", "pgap"}:
            raise ValueError("gradient subspace method name must be svd0 or pgap")
        self.config = config
        self.name = name
        self.method_config = self._normalize_method_config(config, name=name)
        self._u_cache: dict[str, torch.Tensor] = {}
        self._s_cache: dict[str, torch.Tensor] = {}
        self._v_cache: dict[str, torch.Tensor] = {}
        self._basis_refresh_counts: Counter[str] = Counter()

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

        refresh = self._should_refresh_basis(step=step, parameters=parameters)
        refresh_objective_calls = 0
        refresh_projected_grads: list[float] = []
        if refresh:
            refresh_projected_grads = self._refresh_bases(
                parameters,
                objective_fn,
                seed=seed,
                eps=eps,
            )
            refresh_objective_calls = 2 * int(self._probe_count())

        self._apply_subspace_perturbation(
            parameters,
            seed=seed,
            step=step,
            eps=eps,
            scaling=1.0,
        )
        try:
            plus = evaluate_objective("plus", objective_fn)
            self._apply_subspace_perturbation(
                parameters,
                seed=seed,
                step=step,
                eps=eps,
                scaling=-2.0,
            )
            minus = evaluate_objective("minus", objective_fn)
        finally:
            self._apply_subspace_perturbation(
                parameters,
                seed=seed,
                step=step,
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
            objective_delta = float(f_plus) - float(f_minus)
            projected_grad = objective_delta / (2.0 * eps)
            self._apply_subspace_update(
                parameters,
                seed=seed,
                step=step,
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
        diagnostics = {
            "authority_source": "paper-derived",
            "method_source_commit": METHOD_SOURCE_COMMITS[self.name],
            "method_config": method_config_diagnostics(
                eps=eps,
                learning_rate=learning_rate,
                parameter_scope=parameter_scope,
                weight_decay=float(self.config.weight_decay),
                **{self.name: self.method_config},
            ),
            "objective_calls": 2 + refresh_objective_calls,
            "num_objective_calls": 2 + refresh_objective_calls,
            "basis_refreshed": bool(refresh),
            "basis_refresh_counts": dict(self._basis_refresh_counts),
            "basis_cache_size": len(self._u_cache),
            "rank": int(self._rank()),
            "update_interval": int(self._update_interval()),
            "probe_count": int(self._probe_count()),
            "target_ndim": int(self._target_ndim()),
            "gradient_subspace_parameter_count": int(
                sum(
                    param.numel()
                    for _, param in parameters
                    if param.data.ndim == int(self._target_ndim())
                )
            ),
            "fallback_parameter_count": int(
                sum(
                    param.numel()
                    for _, param in parameters
                    if param.data.ndim != int(self._target_ndim())
                )
            ),
            "parameter_selection_checksum": self._parameter_selection_checksum(
                parameters
            ),
            "refresh_projected_grad_mean": (
                sum(refresh_projected_grads) / len(refresh_projected_grads)
                if refresh_projected_grads
                else None
            ),
            "refresh_projected_grad_abs_mean": (
                sum(abs(value) for value in refresh_projected_grads)
                / len(refresh_projected_grads)
                if refresh_projected_grads
                else None
            ),
        }
        if self.name == "pgap":
            diagnostics["pgap_delta"] = float(self._pgap_delta(step=step))
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
            method_diagnostics=diagnostics,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "u_cache": {
                name: tensor.detach().cpu() for name, tensor in self._u_cache.items()
            },
            "s_cache": {
                name: tensor.detach().cpu() for name, tensor in self._s_cache.items()
            },
            "v_cache": {
                name: tensor.detach().cpu() for name, tensor in self._v_cache.items()
            },
            "basis_refresh_counts": dict(self._basis_refresh_counts),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._u_cache = self._tensor_cache_from_state(state_dict.get("u_cache", {}))
        self._s_cache = self._tensor_cache_from_state(state_dict.get("s_cache", {}))
        self._v_cache = self._tensor_cache_from_state(state_dict.get("v_cache", {}))
        counts = state_dict.get("basis_refresh_counts", {})
        if isinstance(counts, dict):
            self._basis_refresh_counts = Counter(
                {str(name): int(value) for name, value in counts.items()}
            )

    def _should_refresh_basis(
        self,
        *,
        step: int,
        parameters: ParameterList,
    ) -> bool:
        if not self._has_all_bases(parameters):
            return True
        return int(step) % int(self._update_interval()) == 0

    def _has_all_bases(self, parameters: ParameterList) -> bool:
        target_ndim = int(self._target_ndim())
        for name, param in parameters:
            if param.data.ndim != target_ndim:
                continue
            if name not in self._u_cache or name not in self._v_cache:
                return False
            if self.name == "pgap" and name not in self._s_cache:
                return False
        return True

    @torch.no_grad()
    def _refresh_bases(
        self,
        parameters: ParameterList,
        objective_fn: ObjectiveFn,
        *,
        seed: int,
        eps: float,
    ) -> list[float]:
        target_ndim = int(self._target_ndim())
        accumulators: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.data, dtype=torch.float32)
            for name, param in parameters
            if param.data.ndim == target_ndim
        }
        projected_grads: list[float] = []
        probe_count = int(self._probe_count())
        for probe_idx in range(probe_count):
            probe_seed = int(seed) + 1_000_003 * (probe_idx + 1)
            self._apply_dense_perturbation(
                parameters,
                seed=probe_seed,
                eps=eps,
                scaling=1.0,
            )
            try:
                plus = evaluate_objective(f"basis_probe_{probe_idx}_plus", objective_fn)
                self._apply_dense_perturbation(
                    parameters,
                    seed=probe_seed,
                    eps=eps,
                    scaling=-2.0,
                )
                minus = evaluate_objective(
                    f"basis_probe_{probe_idx}_minus", objective_fn
                )
            finally:
                self._apply_dense_perturbation(
                    parameters,
                    seed=probe_seed,
                    eps=eps,
                    scaling=1.0,
                )
            if not (
                plus.ok
                and minus.ok
                and plus.objective_value is not None
                and minus.objective_value is not None
            ):
                continue
            projected_grad = (
                float(plus.objective_value) - float(minus.objective_value)
            ) / (2.0 * float(eps))
            projected_grads.append(projected_grad)
            torch.manual_seed(probe_seed)
            for name, param in parameters:
                z = torch.randn_like(param.data)
                if name not in accumulators:
                    continue
                accumulators[name].add_(
                    z.to(dtype=torch.float32),
                    alpha=float(projected_grad) / float(probe_count),
                )
        for name, gradient_estimate in accumulators.items():
            if gradient_estimate.numel() == 0:
                continue
            self._set_basis_from_gradient(name, gradient_estimate)
        return projected_grads

    @torch.no_grad()
    def _set_basis_from_gradient(self, name: str, gradient: torch.Tensor) -> None:
        rows, cols = int(gradient.shape[0]), int(gradient.shape[1])
        rank = min(int(self._rank()), rows, cols)
        if rank <= 0:
            return
        u, s, vh = torch.linalg.svd(gradient.float(), full_matrices=False)
        self._u_cache[name] = u[:, :rank].contiguous()
        self._s_cache[name] = s[:rank].contiguous()
        self._v_cache[name] = vh[:rank, :].t().contiguous()
        self._basis_refresh_counts[name] += 1

    @torch.no_grad()
    def _apply_dense_perturbation(
        self,
        parameters: ParameterList,
        *,
        seed: int,
        eps: float,
        scaling: float,
    ) -> None:
        torch.manual_seed(int(seed))
        for _, param in parameters:
            param.add_(torch.randn_like(param.data), alpha=float(eps) * float(scaling))

    @torch.no_grad()
    def _apply_subspace_perturbation(
        self,
        parameters: ParameterList,
        *,
        seed: int,
        step: int,
        eps: float,
        scaling: float,
    ) -> None:
        torch.manual_seed(int(seed))
        for name, param in parameters:
            z = self._noise_for_param(name, param, step=step)
            param.add_(z, alpha=float(eps) * float(scaling))

    @torch.no_grad()
    def _apply_subspace_update(
        self,
        parameters: ParameterList,
        *,
        seed: int,
        step: int,
        projected_grad: float,
        learning_rate: float,
        weight_decay: float,
        update_trace: UpdateTraceAccumulator,
    ) -> None:
        torch.manual_seed(int(seed))
        for name, param in parameters:
            z = self._noise_for_param(name, param, step=step)
            update = z.mul(float(projected_grad))
            if weight_decay:
                update = update.add(param.data, alpha=float(weight_decay))
            update_trace.add_delta(name, update, scale=-float(learning_rate))
            param.add_(update, alpha=-float(learning_rate))

    def _noise_for_param(
        self,
        name: str,
        param: torch.nn.Parameter,
        *,
        step: int,
    ) -> torch.Tensor:
        if param.data.ndim != int(self._target_ndim()):
            return torch.randn_like(param.data)
        u = self._u_cache.get(name)
        v = self._v_cache.get(name)
        if u is None or v is None:
            return torch.randn_like(param.data)
        u = u.to(device=param.device, dtype=param.dtype)
        v = v.to(device=param.device, dtype=param.dtype)
        rank = min(int(u.shape[1]), int(v.shape[1]))
        if rank <= 0:
            return torch.randn_like(param.data)
        z = torch.randn((rank, rank), device=param.device, dtype=param.dtype)
        if self.name == "pgap":
            z = self._project_pgap_noise(name, z, param=param, step=step)
        return u[:, :rank].matmul(z).matmul(v[:, :rank].t())

    def _project_pgap_noise(
        self,
        name: str,
        z_init: torch.Tensor,
        *,
        param: torch.nn.Parameter,
        step: int,
    ) -> torch.Tensor:
        singular_values = self._s_cache.get(name)
        if singular_values is None:
            return z_init
        s = singular_values.to(device=param.device, dtype=param.dtype)
        rank = min(int(z_init.shape[0]), int(s.numel()))
        if rank <= 0:
            return z_init
        s = s[:rank]
        z = z_init.clone()
        diag = torch.diagonal(z[:rank, :rank])
        inner = torch.sum(diag * s)
        norm = torch.linalg.vector_norm(s)
        if float(norm.item()) == 0.0:
            return z_init
        xi = -1.0 if torch.rand((), device=param.device).item() < 0.5 else 1.0
        delta = float(self._pgap_delta(step=step))
        alpha = (inner - float(xi) * math.sqrt(delta) * norm) / (
            norm.square() + 1e-12
        )
        diag.sub_(alpha * s)
        return z

    def _pgap_delta(self, *, step: int) -> float:
        if not isinstance(self.method_config, PGAPConfig):
            return 0.0
        decay_steps = int(self.method_config.delta_decay_steps)
        if decay_steps <= 0:
            return float(self.method_config.delta_init)
        ratio = min(max(float(step) / float(decay_steps), 0.0), 1.0)
        return float(self.method_config.delta_init) + ratio * (
            float(self.method_config.delta_final) - float(self.method_config.delta_init)
        )

    def _rank(self) -> int:
        return int(self.method_config.rank)

    def _update_interval(self) -> int:
        return int(self.method_config.update_interval)

    def _probe_count(self) -> int:
        return int(self.method_config.probe_count)

    def _target_ndim(self) -> int:
        return int(self.method_config.target_ndim)

    @staticmethod
    def _parameter_selection_checksum(parameters: ParameterList) -> str:
        payload = [
            {
                "name": name,
                "shape": tuple(int(dim) for dim in param.shape),
                "numel": int(param.numel()),
            }
            for name, param in parameters
        ]
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    @staticmethod
    def _tensor_cache_from_state(value: Any) -> dict[str, torch.Tensor]:
        if not isinstance(value, dict):
            return {}
        return {
            str(name): tensor.detach().contiguous()
            for name, tensor in value.items()
            if isinstance(tensor, torch.Tensor)
        }

    @staticmethod
    def _normalize_method_config(
        config: ZOConfig,
        *,
        name: str,
    ) -> SVD0Config | PGAPConfig:
        value = config.svd0 if name == "svd0" else config.pgap
        cls = SVD0Config if name == "svd0" else PGAPConfig
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise ValueError(f"zo.{name} must be a mapping")


def gradient_subspace_config_to_dict(value: SVD0Config | PGAPConfig) -> dict[str, Any]:
    return asdict(value)
