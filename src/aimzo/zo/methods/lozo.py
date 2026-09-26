from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

import torch

from aimzo.config import LOZOConfig, ZOConfig
from aimzo.zo.params import ParameterList, select_trainable_parameters

from .base import (
    CandidateEvaluation,
    HFZOMethodStepResult,
    METHOD_SOURCE_COMMITS,
    ObjectiveFn,
    UpdateTraceAccumulator,
    evaluate_objective,
    method_config_diagnostics,
)


class HFLOZOMethod:
    name = "lozo"

    def __init__(self, config: ZOConfig) -> None:
        self.config = config
        self.lozo_config = self._normalize_lozo_config(config.lozo)
        self._v_cache: dict[tuple[str, tuple[int, ...], str, str], torch.Tensor] = {}
        self._v_refresh_counts: Counter[str] = Counter()

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

        self._apply_lozo_perturbation(
            parameters,
            seed=seed,
            step=step,
            eps=eps,
            scaling=1.0,
        )
        try:
            plus = evaluate_objective("plus", objective_fn)
            self._apply_lozo_perturbation(
                parameters,
                seed=seed,
                step=step,
                eps=eps,
                scaling=-2.0,
            )
            minus = evaluate_objective("minus", objective_fn)
        finally:
            self._apply_lozo_perturbation(
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
            self._apply_lozo_update(
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
            "authority_source": "paper+author_source",
            "method_source_commit": METHOD_SOURCE_COMMITS[self.name],
            "method_config": method_config_diagnostics(
                eps=eps,
                learning_rate=learning_rate,
                parameter_scope=parameter_scope,
                weight_decay=float(self.config.weight_decay),
                lozo=self.lozo_config,
            ),
            "objective_calls": 2,
            "num_objective_calls": 2,
            "lozo_rank": int(self.lozo_config.rank),
            "rank": int(self.lozo_config.rank),
            "lozo_step_interval": int(self.lozo_config.step_interval),
            "lozo_normalization": str(self.lozo_config.normalization),
            "lozo_v_cache_size": len(self._v_cache),
            "lozo_v_refresh_counts": dict(self._v_refresh_counts),
            "lozo_parameter_count": int(
                sum(param.numel() for _, param in parameters if param.data.ndim == 2)
            ),
            "fallback_parameter_count": int(
                sum(param.numel() for _, param in parameters if param.data.ndim != 2)
            ),
            "parameter_selection_checksum": self._parameter_selection_checksum(
                parameters
            ),
        }
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
            "v_cache": {
                json.dumps(key): tensor.detach().cpu()
                for key, tensor in self._v_cache.items()
            },
            "v_refresh_counts": dict(self._v_refresh_counts),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        cache = state_dict.get("v_cache", {})
        self._v_cache = {}
        if isinstance(cache, dict):
            for key_payload, tensor in cache.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                key = json.loads(key_payload)
                self._v_cache[
                    (str(key[0]), tuple(int(dim) for dim in key[1]), str(key[2]), str(key[3]))
                ] = tensor.detach().contiguous()
        counts = state_dict.get("v_refresh_counts", {})
        if isinstance(counts, dict):
            self._v_refresh_counts = Counter(
                {str(name): int(value) for name, value in counts.items()}
            )

    @torch.no_grad()
    def _apply_lozo_perturbation(
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
    def _apply_lozo_update(
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
        if param.data.ndim != 2:
            return torch.randn_like(param.data)
        rows, cols = int(param.data.shape[0]), int(param.data.shape[1])
        rank = min(int(self.lozo_config.rank), rows, cols)
        if rank <= 0:
            return torch.randn_like(param.data)
        v = self._v_matrix(name, param, rank=rank, step=step)
        u = torch.randn(
            (rows, rank),
            device=param.device,
            dtype=param.dtype,
        )
        noise = u.matmul(v.t())
        if str(self.lozo_config.normalization) == "rank":
            noise = noise.div(float(rank))
        return noise

    def _v_matrix(
        self,
        name: str,
        param: torch.nn.Parameter,
        *,
        rank: int,
        step: int,
    ) -> torch.Tensor:
        interval = int(self.lozo_config.step_interval)
        key = (
            str(name),
            tuple(int(dim) for dim in param.data.shape),
            str(param.device),
            str(param.dtype),
        )
        cached = self._v_cache.get(key)
        refresh = int(step) % interval == 0
        if cached is not None and not refresh:
            return cached.to(device=param.device, dtype=param.dtype)
        v = torch.randn(
            (int(param.data.shape[1]), int(rank)),
            device=param.device,
            dtype=param.dtype,
        )
        self._v_cache[key] = v.detach().contiguous()
        self._v_refresh_counts[str(name)] += 1
        return v

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
    def _normalize_lozo_config(value: LOZOConfig | dict[str, Any]) -> LOZOConfig:
        if isinstance(value, LOZOConfig):
            return value
        if isinstance(value, dict):
            return LOZOConfig(**value)
        raise ValueError("zo.lozo must be a mapping")
