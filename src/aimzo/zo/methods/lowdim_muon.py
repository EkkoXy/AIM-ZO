from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import replace
from typing import Any

import torch

from aimzo.config import LowDimMuonConfig, ZOConfig
from aimzo.zo.params import ParameterList, select_trainable_parameters

from .base import (
    HFZOMethodStepResult,
    METHOD_SOURCE_COMMITS,
    ObjectiveFn,
    UpdateTraceAccumulator,
    bounded_tensor_fingerprint,
    evaluate_objective,
    method_config_diagnostics,
)


def _lowdim_defaults_for_method(method_name: str) -> LowDimMuonConfig:
    method = str(method_name).lower()
    if method == "zomuon":
        return replace(LowDimMuonConfig(), step_interval=100, num_samples=4)
    if method == "zomopi":
        return replace(LowDimMuonConfig(), step_interval=500, num_samples=8)
    raise ValueError(f"Unsupported low-dimensional Muon method: {method_name!r}")


def lowdim_config_for_method(config: ZOConfig, method_name: str) -> LowDimMuonConfig:
    value = config.lowdim_muon
    defaults = _lowdim_defaults_for_method(method_name)
    if isinstance(value, dict):
        return replace(defaults, **dict(value))
    if isinstance(value, LowDimMuonConfig):
        if value == LowDimMuonConfig():
            return defaults
        return value
    raise ValueError("zo.lowdim_muon must be a mapping or LowDimMuonConfig")


def _state_checksum(state: dict[str, torch.Tensor]) -> str:
    if not state:
        return "0" * 64
    digest = hashlib.sha256()
    for name in sorted(state):
        fingerprint, _ = bounded_tensor_fingerprint(
            name,
            state[name],
            error_prefix="lowdim state checksum",
        )
        digest.update(
            json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _source_style_state_checksum(state: dict[str, torch.Tensor]) -> str:
    if not state:
        return "0" * 64
    digest = hashlib.sha256()
    for name in sorted(state):
        source_tensor = state[name].detach().float().cpu().reshape(-1)
        metadata = {
            "name": name,
            "shape": [int(dim) for dim in state[name].shape],
            "dtype": str(state[name].dtype),
            "numel": int(state[name].numel()),
            "sum": float(source_tensor.sum().item()),
            "sample": source_tensor[:1024].tolist(),
        }
        digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _zeropower_via_svd(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("SVD Muon transform expects a 2D matrix")
    original_dtype = matrix.dtype
    matrix_for_svd = (
        matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
    )
    if float(torch.linalg.vector_norm(matrix_for_svd).item()) == 0.0:
        return torch.zeros_like(matrix)
    u, singular_values, vh = torch.linalg.svd(matrix_for_svd, full_matrices=False)
    singular_floor = torch.finfo(singular_values.dtype).tiny
    if float(singular_values.max().item()) <= singular_floor:
        return torch.zeros_like(matrix)
    return u.matmul(vh).to(dtype=original_dtype)


def _zeropower_via_newton_schulz(
    matrix: torch.Tensor, *, steps: int = 5
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz Muon transform expects a 2D matrix")
    if float(torch.linalg.vector_norm(matrix.float()).item()) == 0.0:
        return torch.zeros_like(matrix)
    original_dtype = matrix.dtype
    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (torch.linalg.vector_norm(x) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        xx_t = x.matmul(x.T)
        x = a * x + (b * xx_t + c * xx_t.matmul(xx_t)).matmul(x)
    if transposed:
        x = x.T
    return x.to(dtype=original_dtype)


def _zeropower_via_source_newton_schulz(
    matrix: torch.Tensor, *, steps: int = 5
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz Muon transform expects a 2D matrix")
    if float(torch.linalg.vector_norm(matrix.float()).item()) == 0.0:
        return torch.zeros_like(matrix)
    x = matrix
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        xx_t = x @ x.mT
        x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x
    if transposed:
        x = x.mT
    return x


def _highpass_via_pion_newton_schulz(
    matrix: torch.Tensor,
    *,
    steps: int = 5,
    promotion_steps: int = 2,
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("Pion Newton-Schulz transform expects a 2D matrix")
    if float(torch.linalg.vector_norm(matrix.float()).item()) == 0.0:
        return torch.zeros_like(matrix)
    original_dtype = matrix.dtype
    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (torch.linalg.vector_norm(x) + 1e-7)

    for _ in range(int(promotion_steps)):
        xx_t = x.matmul(x.T)
        x = 1.875 * x + (-1.25 * xx_t + 0.375 * xx_t.matmul(xx_t)).matmul(x)

    for _ in range(int(steps) - int(promotion_steps)):
        xx_t = x.matmul(x.T)
        x = (2.5 * xx_t - 1.5 * xx_t.matmul(xx_t)).matmul(x)

    if transposed:
        x = x.T
    return x.to(dtype=original_dtype)


class HFLowDimMuonMethod:
    def __init__(self, config: ZOConfig, *, name: str) -> None:
        self.config = config
        self.name = str(name).lower()
        self.lowdim_config = lowdim_config_for_method(config, self.name)
        self.P_matrices: dict[str, torch.Tensor] = {}
        self.V_matrices: dict[str, torch.Tensor] = {}
        self.u_momentum: dict[str, torch.Tensor] = {}
        self.step_index = 0
        self._projection_refresh_count = 0
        self._last_gradient_change = 0.0
        self._pre_transform_update_squared_norm = 0.0
        self._post_transform_update_squared_norm = 0.0
        self._perturbation_trace: list[dict[str, Any]] = []
        self._perturbation_call_trace: list[dict[str, Any]] = []
        self._lowdim_update_trace: list[dict[str, Any]] = []
        self._perturbation_trace_truncated = False

    def step(
        self,
        model: torch.nn.Module,
        objective_fn: ObjectiveFn,
        *,
        seed: int,
        step: int,
    ) -> HFZOMethodStepResult:
        parameters = self._selected_parameters(model)
        self.step_index = int(step)
        self._projection_refresh_count = 0
        self._last_gradient_change = 0.0
        self._pre_transform_update_squared_norm = 0.0
        self._post_transform_update_squared_norm = 0.0
        self._perturbation_trace = []
        self._perturbation_call_trace = []
        self._lowdim_update_trace = []
        self._perturbation_trace_truncated = False
        if bool(self.lowdim_config.multiple_sample):
            return self._one_sided_multi_sample_step(
                parameters, objective_fn, seed=int(seed), step=int(step)
            )
        return self._two_sided_single_sample_step(
            parameters, objective_fn, seed=int(seed), step=int(step)
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "lowdim_config": self.lowdim_config.__dict__.copy(),
            "P_matrices": {
                name: value.detach().clone() for name, value in self.P_matrices.items()
            },
            "V_matrices": {
                name: value.detach().clone() for name, value in self.V_matrices.items()
            },
            "u_momentum": {
                name: value.detach().clone() for name, value in self.u_momentum.items()
            },
            "step_index": int(self.step_index),
            "projection_refresh_count": int(self._projection_refresh_count),
            "last_gradient_change": float(self._last_gradient_change),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.P_matrices = {
            str(name): tensor.detach().clone()
            for name, tensor in dict(state_dict.get("P_matrices", {})).items()
        }
        self.V_matrices = {
            str(name): tensor.detach().clone()
            for name, tensor in dict(state_dict.get("V_matrices", {})).items()
        }
        self.u_momentum = {
            str(name): tensor.detach().clone()
            for name, tensor in dict(state_dict.get("u_momentum", {})).items()
        }
        self.step_index = int(state_dict.get("step_index", 0))
        self._projection_refresh_count = int(
            state_dict.get("projection_refresh_count", 0)
        )
        self._last_gradient_change = float(state_dict.get("last_gradient_change", 0.0))
        self._pre_transform_update_squared_norm = 0.0
        self._post_transform_update_squared_norm = 0.0

    def _selected_parameters(self, model: torch.nn.Module) -> ParameterList:
        parameter_scope = self.config.parameter_scope
        parameters = select_trainable_parameters(model, parameter_scope)
        if not parameters:
            raise ValueError(
                f"No trainable parameters found for scope {parameter_scope!r}"
            )
        return parameters

    def _one_sided_multi_sample_step(
        self,
        parameters: ParameterList,
        objective_fn: ObjectiveFn,
        *,
        seed: int,
        step: int,
    ) -> HFZOMethodStepResult:
        eps = float(self.config.eps)
        baseline = evaluate_objective("center", objective_fn)
        saved_noise: dict[str, dict[int, torch.Tensor]] = {}
        projected_grads: list[float] = []
        candidates = [baseline.to_dict(seed=seed, eps=eps, scaling=0.0)]
        update_trace = UpdateTraceAccumulator()

        if baseline.ok and baseline.objective_value is not None:
            for sample_idx in range(int(self.lowdim_config.num_samples)):
                current_seed = int(seed) + sample_idx
                self._apply_saved_perturbation(
                    parameters,
                    seed=current_seed,
                    sample_idx=sample_idx,
                    saved_noise=saved_noise,
                    step=step,
                    eps=eps,
                    scaling=1.0,
                    resample_projection=(sample_idx == 0),
                )
                try:
                    plus = evaluate_objective(f"plus_{sample_idx}", objective_fn)
                finally:
                    self._apply_saved_perturbation(
                        parameters,
                        seed=current_seed,
                        sample_idx=sample_idx,
                        saved_noise=saved_noise,
                        step=step,
                        eps=eps,
                        scaling=-1.0,
                        resample_projection=False,
                    )
                candidates.append(
                    plus.to_dict(seed=current_seed, eps=eps, scaling=1.0)
                )
                if plus.ok and plus.objective_value is not None:
                    projected_grads.append(
                        self._projected_grad_from_losses(
                            plus_loss=float(plus.objective_value),
                            baseline_loss=float(baseline.objective_value),
                            eps=eps,
                        )
                    )

        if len(projected_grads) == int(self.lowdim_config.num_samples):
            self._apply_lowdim_update(
                parameters,
                saved_noise,
                projected_grads,
                update_trace,
            )
        update_norm, parameter_delta_checksum = update_trace.finish()

        return self._build_result(
            seed=seed,
            eps=eps,
            f_plus=None,
            f_minus=None,
            objective_delta=None,
            projected_grads=projected_grads,
            candidate_results=candidates,
            parameters=parameters,
            num_objective_calls=len(candidates),
            loss0=baseline.objective_value,
            update_norm=update_norm,
            parameter_delta_checksum=parameter_delta_checksum,
        )

    def _two_sided_single_sample_step(
        self,
        parameters: ParameterList,
        objective_fn: ObjectiveFn,
        *,
        seed: int,
        step: int,
    ) -> HFZOMethodStepResult:
        eps = float(self.config.eps)
        saved_noise: dict[str, dict[int, torch.Tensor]] = {}
        self._apply_saved_perturbation(
            parameters,
            seed=seed,
            sample_idx=0,
            saved_noise=saved_noise,
            step=step,
            eps=eps,
            scaling=1.0,
            resample_projection=True,
        )
        try:
            plus = evaluate_objective("plus", objective_fn)
        finally:
            self._apply_saved_perturbation(
                parameters,
                seed=seed,
                sample_idx=0,
                saved_noise=saved_noise,
                step=step,
                eps=eps,
                scaling=-1.0,
                resample_projection=False,
            )

        self._apply_saved_perturbation(
            parameters,
            seed=seed,
            sample_idx=0,
            saved_noise=saved_noise,
            step=step,
            eps=eps,
            scaling=-1.0,
            resample_projection=False,
        )
        try:
            minus = evaluate_objective("minus", objective_fn)
        finally:
            self._apply_saved_perturbation(
                parameters,
                seed=seed,
                sample_idx=0,
                saved_noise=saved_noise,
                step=step,
                eps=eps,
                scaling=1.0,
                resample_projection=False,
            )

        projected_grads: list[float] = []
        objective_delta = None
        update_trace = UpdateTraceAccumulator()
        if (
            plus.ok
            and minus.ok
            and plus.objective_value is not None
            and minus.objective_value is not None
        ):
            objective_delta = float(plus.objective_value) - float(
                minus.objective_value
            )
            projected_grads.append(objective_delta / (2.0 * eps))
            self._apply_lowdim_update(
                parameters,
                saved_noise,
                projected_grads,
                update_trace,
            )
        update_norm, parameter_delta_checksum = update_trace.finish()

        return self._build_result(
            seed=seed,
            eps=eps,
            f_plus=plus.objective_value,
            f_minus=minus.objective_value,
            objective_delta=objective_delta,
            projected_grads=projected_grads,
            candidate_results=[
                plus.to_dict(seed=seed, eps=eps, scaling=1.0),
                minus.to_dict(seed=seed, eps=eps, scaling=-1.0),
            ],
            parameters=parameters,
            num_objective_calls=2,
            loss0=None,
            update_norm=update_norm,
            parameter_delta_checksum=parameter_delta_checksum,
        )

    def _effective_rank(self, param: torch.Tensor) -> int:
        return min(int(self.lowdim_config.rank), int(param.shape[0]))

    def _sample_orthogonal_p(
        self,
        rows: int,
        rank: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        basis = torch.randn(rows, rank, device=device, dtype=torch.float32)
        q, r = torch.linalg.qr(basis, mode="reduced")
        signs = torch.sign(torch.diagonal(r))
        signs[signs == 0] = 1.0
        q = q * signs
        return q.to(dtype=dtype)

    def _is_matrix_parameter(self, param: torch.nn.Parameter) -> bool:
        return param.data.ndim == 2

    def _maybe_refresh_projection(
        self,
        name: str,
        param: torch.nn.Parameter,
        *,
        step: int,
        resample: bool,
    ) -> bool:
        if not self._is_matrix_parameter(param):
            return False
        rank = self._effective_rank(param.data)
        current = self.P_matrices.get(name)
        refresh_step = int(step)
        if str(self.lowdim_config.refresh_indexing) == "source":
            refresh_step = max(0, refresh_step - 1)
        should_refresh = (
            current is None
            or tuple(current.shape) != (int(param.data.shape[0]), rank)
            or (
                bool(resample)
                and refresh_step % int(self.lowdim_config.step_interval) == 0
            )
        )
        if not should_refresh:
            return False
        if bool(self.lowdim_config.reset_v_on_p_refresh):
            self.V_matrices.pop(name, None)
        self.P_matrices[name] = self._sample_orthogonal_p(
            int(param.data.shape[0]),
            rank,
            device=param.device,
            dtype=param.dtype,
        )
        self._projection_refresh_count += 1
        return True

    def _source_zomuon_refreshes_projection_every_call(self, *, step: int) -> bool:
        if self.name != "zomuon":
            return False
        if str(self.config.source_compatibility_profile) != "zomuon_source":
            return False
        refresh_step = int(step)
        if str(self.lowdim_config.refresh_indexing) == "source":
            refresh_step = max(0, refresh_step - 1)
        return refresh_step % int(self.lowdim_config.step_interval) == 0

    def _records_source_debug_trace(self) -> bool:
        return bool(self.lowdim_config.source_debug_trace) and str(
            self.config.source_compatibility_profile
        ) in {
            "zomuon_source",
            "zomopi_source",
        }

    def _records_expensive_source_trace(self) -> bool:
        return os.getenv("AIMZO_LOW_DIM_SOURCE_TRACE") == "1"

    def _noise_for_param(
        self,
        name: str,
        param: torch.nn.Parameter,
        *,
        sample_idx: int,
        saved_noise: dict[str, dict[int, torch.Tensor]],
        step: int,
        resample_projection: bool,
        source_refresh_call: bool = False,
    ) -> torch.Tensor:
        param_noises = saved_noise.setdefault(name, {})
        if sample_idx in param_noises and not source_refresh_call:
            if self.name != "zomopi":
                return param_noises[sample_idx]
            # Official ZO-MOPI restores regenerate the perturbation noise
            # after resetting the seed, which also preserves the downstream RNG
            # stream used by its streaming power-iteration state.
            noise = torch.randn_like(param_noises[sample_idx])
            param_noises[sample_idx] = noise
            return noise
        if self._is_matrix_parameter(param):
            self._maybe_refresh_projection(
                name,
                param,
                step=step,
                resample=(resample_projection or source_refresh_call),
            )
            rank = self._effective_rank(param.data)
            noise = torch.randn(
                (rank, int(param.data.shape[1])),
                device=param.device,
                dtype=param.dtype,
            )
        else:
            noise = torch.randn_like(param.data)
        param_noises[sample_idx] = noise
        return noise

    def _full_noise_from_saved(
        self,
        name: str,
        param: torch.nn.Parameter,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        if self._is_matrix_parameter(param):
            return self.P_matrices[name].to(
                device=param.device, dtype=param.dtype
            ).matmul(noise)
        return noise

    def _bounded_tensor_trace(self, name: str, tensor: torch.Tensor) -> dict[str, Any]:
        fingerprint, _ = bounded_tensor_fingerprint(
            name,
            tensor,
            error_prefix="lowdim perturbation trace",
        )
        flat = tensor.detach().reshape(-1)
        first_count = min(16, int(flat.numel()))
        fingerprint["first_values"] = [
            float(flat[index].detach().cpu().item()) for index in range(first_count)
        ]
        return fingerprint

    def _retained_projection_samples(self, *, limit: int = 8) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for name in sorted(self.P_matrices)[:limit]:
            samples.append(
                {
                    "name": name,
                    "p": self._bounded_tensor_trace(name, self.P_matrices[name]),
                }
            )
        return samples

    def _source_style_model_state_samples(
        self, parameters: ParameterList
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
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
            samples.append(
                {
                    "name": name,
                    "shape": [int(dim) for dim in param.shape],
                    "dtype": str(param.dtype),
                    "sample_values": sample_values,
                }
            )
        return samples

    def _source_style_model_state_checksum(
        self, samples: list[dict[str, Any]]
    ) -> str:
        digest = hashlib.sha256()
        for payload in samples:
            digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _update_policy_name(self) -> str:
        if self.name == "zomuon":
            return "zomuon_direct_transform"
        if self.name == "zomopi":
            return "zomopi_streaming_pi_then_phase2"
        raise ValueError(f"Unsupported low-dimensional Muon method: {self.name!r}")

    def _effective_update_transform_name(self) -> str:
        optimizer = str(self.lowdim_config.phase2_optimizer)
        if self.name == "zomopi" and int(self.step_index) < int(
            self.lowdim_config.phase2_steps
        ):
            return "streaming_power_iteration"
        if optimizer == "sgd":
            return "sgd"
        if optimizer == "muon_svd":
            return "muon_svd"
        if optimizer in {"muon", "muon_ns", "ns"}:
            return "muon_ns"
        if optimizer in {"pion", "pion_ns"}:
            return "pion_ns"
        raise ValueError(f"Unsupported phase2 optimizer: {optimizer}")

    def _uses_lowdim_momentum(self) -> bool:
        return self.name == "zomopi"

    def _transform_matrix_update(self, name: str, matrix: torch.Tensor) -> torch.Tensor:
        transform = self._effective_update_transform_name()
        if transform == "sgd":
            return matrix
        if transform == "streaming_power_iteration":
            transformed, v_new = self._zeropower_via_streaming_pi(name, matrix)
            self.V_matrices[name] = v_new.detach().clone()
            return transformed
        if transform == "muon_svd":
            return _zeropower_via_svd(matrix)
        if transform == "muon_ns":
            if (
                self.name == "zomuon"
                and str(self.config.source_compatibility_profile) == "zomuon_source"
            ):
                return _zeropower_via_source_newton_schulz(matrix)
            return _zeropower_via_newton_schulz(matrix)
        if transform == "pion_ns":
            return _highpass_via_pion_newton_schulz(
                matrix,
                steps=int(self.lowdim_config.pion_steps),
                promotion_steps=int(self.lowdim_config.pion_promotion_steps),
            )
        raise ValueError(f"Unsupported effective update transform: {transform}")

    def _zeropower_via_streaming_pi(
        self, name: str, matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if matrix.ndim != 2:
            raise ValueError("streaming power iteration expects a 2D matrix")
        _, cols = matrix.shape
        k = min(int(self.lowdim_config.k_start), int(matrix.shape[0]), int(cols))
        matrix32 = matrix.float()
        v_state = self.V_matrices.get(name)
        if v_state is None or tuple(v_state.shape) != (cols, k):
            v_state = torch.randn(cols, k, device=matrix.device, dtype=torch.float32)
        else:
            v_state = v_state.to(device=matrix.device, dtype=torch.float32)
        v_state, _ = torch.linalg.qr(v_state, mode="reduced")
        gv = matrix32.matmul(v_state)
        z = matrix32.T.matmul(gv)
        v_new, _ = torch.linalg.qr(z, mode="reduced")
        u_unnorm = matrix32.matmul(v_new)
        u_norm = torch.nn.functional.normalize(u_unnorm, p=2, dim=0)
        transformed = u_norm.matmul(v_new.T).to(dtype=matrix.dtype)
        return transformed, v_new.to(dtype=matrix.dtype)

    def _accumulate_lowdim_noise(
        self,
        accumulator: torch.Tensor,
        noise: torch.Tensor,
        *,
        projected_grad: float,
    ) -> None:
        if str(self.config.source_compatibility_profile) in {
            "zomuon_source",
            "zomopi_source",
        }:
            accumulator.add_(noise * float(projected_grad))
        else:
            accumulator.add_(noise, alpha=float(projected_grad))

    def _projected_grad_from_losses(
        self,
        *,
        plus_loss: float,
        baseline_loss: float,
        eps: float,
    ) -> float:
        if str(self.config.source_compatibility_profile) in {
            "zomuon_source",
            "zomopi_source",
        }:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            plus_tensor = torch.tensor(
                float(plus_loss), device=device, dtype=torch.float16
            )
            baseline_tensor = torch.tensor(
                float(baseline_loss), device=device, dtype=torch.float16
            )
            return float(((plus_tensor - baseline_tensor) / float(eps)).item())
        return (float(plus_loss) - float(baseline_loss)) / float(eps)

    @torch.no_grad()
    def _apply_lowdim_parameter_update(
        self,
        param: torch.nn.Parameter,
        update: torch.Tensor,
        *,
        learning_rate: float,
    ) -> None:
        scale = -float(learning_rate)
        if str(self.config.source_compatibility_profile) in {
            "zomuon_source",
            "zomopi_source",
        }:
            param.add_(update * scale)
        else:
            param.add_(update, alpha=scale)

    @torch.no_grad()
    def _apply_saved_perturbation(
        self,
        parameters: ParameterList,
        *,
        seed: int,
        sample_idx: int,
        saved_noise: dict[str, dict[int, torch.Tensor]],
        step: int,
        eps: float,
        scaling: float,
        resample_projection: bool,
    ) -> None:
        torch.manual_seed(int(seed))
        source_refresh_call = self._source_zomuon_refreshes_projection_every_call(
            step=step
        )
        for name, param in parameters:
            trace_enabled = self._records_source_debug_trace()
            param_before = (
                self._bounded_tensor_trace(name, param.data) if trace_enabled else None
            )
            refresh_count_before = int(self._projection_refresh_count)
            noise = self._noise_for_param(
                name,
                param,
                sample_idx=sample_idx,
                saved_noise=saved_noise,
                step=step,
                resample_projection=resample_projection,
                source_refresh_call=source_refresh_call,
            )
            projection_refreshed = (
                int(self._projection_refresh_count) > refresh_count_before
            )
            full_noise = self._full_noise_from_saved(name, param, noise)
            if str(self.config.source_compatibility_profile) in {
                "zomuon_source",
                "zomopi_source",
            }:
                param.add_(full_noise * (float(eps) * float(scaling)))
            else:
                param.add_(full_noise, alpha=float(eps) * float(scaling))
            if trace_enabled:
                if len(self._perturbation_trace) < 16:
                    entry: dict[str, Any] = {
                        "name": name,
                        "sample_idx": int(sample_idx),
                        "seed": int(seed),
                        "step": int(step),
                        "eps": float(eps),
                        "scaling": float(scaling),
                        "resample_projection": bool(resample_projection),
                        "source_refresh_call": bool(source_refresh_call),
                        "projection_refreshed": bool(projection_refreshed),
                        "noise": self._bounded_tensor_trace(name, noise),
                        "full_noise": self._bounded_tensor_trace(name, full_noise),
                        "param_before": param_before,
                        "param_after": self._bounded_tensor_trace(name, param.data),
                    }
                    if self._is_matrix_parameter(param):
                        entry["p"] = self._bounded_tensor_trace(
                            name, self.P_matrices[name]
                        )
                    self._perturbation_trace.append(entry)
                else:
                    self._perturbation_trace_truncated = True
        if self._records_source_debug_trace():
            if len(self._perturbation_call_trace) < 16:
                call_trace = {
                    "sample_idx": int(sample_idx),
                    "seed": int(seed),
                    "step": int(step),
                    "eps": float(eps),
                    "scaling": float(scaling),
                    "resample_projection": bool(resample_projection),
                    "source_refresh_call": bool(source_refresh_call),
                    "p_state_checksum": _state_checksum(self.P_matrices),
                }
                if self._records_expensive_source_trace():
                    model_state_samples = self._source_style_model_state_samples(
                        parameters
                    )
                    call_trace.update(
                        {
                            "model_state_checksum": (
                                self._source_style_model_state_checksum(
                                    model_state_samples
                                )
                            ),
                            "model_state_samples": model_state_samples,
                        }
                    )
                self._perturbation_call_trace.append(call_trace)
            else:
                self._perturbation_trace_truncated = True

    @torch.no_grad()
    def _apply_lowdim_update(
        self,
        parameters: ParameterList,
        saved_noise: dict[str, dict[int, torch.Tensor]],
        projected_grads: list[float],
        update_trace: UpdateTraceAccumulator,
    ) -> None:
        if not projected_grads:
            return
        for name, param in parameters:
            noises = saved_noise.get(name, {})
            if not noises:
                continue
            first_noise = next(iter(noises.values()))
            grad_est = torch.zeros_like(first_noise)
            for sample_idx, projected_grad in enumerate(projected_grads):
                noise = noises.get(sample_idx)
                if noise is None:
                    continue
                self._accumulate_lowdim_noise(
                    grad_est,
                    noise,
                    projected_grad=float(projected_grad),
                )
            grad_est.div_(float(len(projected_grads)))

            if self._uses_lowdim_momentum():
                old_momentum = self.u_momentum.get(name)
                if old_momentum is None or tuple(old_momentum.shape) != tuple(
                    grad_est.shape
                ):
                    update_source = grad_est.clone()
                else:
                    update_source = old_momentum.to(
                        device=grad_est.device, dtype=grad_est.dtype
                    )
                    update_source.mul_(float(self.lowdim_config.beta)).add_(
                        grad_est, alpha=1.0 - float(self.lowdim_config.beta)
                    )
                self.u_momentum[name] = update_source.detach().clone()
            else:
                update_source = grad_est

            if self._is_matrix_parameter(param):
                lowdim_rge = update_source
                self._pre_transform_update_squared_norm += _bounded_squared_norm(
                    name,
                    lowdim_rge,
                    error_prefix="lowdim pre-transform update",
                )
                transformed = self._transform_matrix_update(name, lowdim_rge)
                self._post_transform_update_squared_norm += _bounded_squared_norm(
                    name,
                    transformed,
                    error_prefix="lowdim post-transform update",
                )
                self._last_gradient_change = _bounded_mean_abs_difference(
                    update_source,
                    transformed,
                    error_prefix="lowdim gradient change",
                )
                update = self.P_matrices[name].to(
                    device=param.device, dtype=param.dtype
                ).matmul(transformed)
                learning_rate = float(self.config.learning_rate)
            else:
                lowdim_rge = update_source
                transformed = update_source
                update = update_source
                update_norm = _bounded_squared_norm(
                    name,
                    lowdim_rge,
                    error_prefix="lowdim fallback update",
                )
                self._pre_transform_update_squared_norm += update_norm
                self._post_transform_update_squared_norm += update_norm
                learning_rate = float(self.lowdim_config.one_d_lr)
            if _should_weight_decay(name) and float(self.config.weight_decay) != 0.0:
                update = update.add(param.data, alpha=float(self.config.weight_decay))
            if self._records_source_debug_trace():
                if len(self._lowdim_update_trace) < 16:
                    update_entry = {
                        "name": name,
                        "lowdim_rge": self._bounded_tensor_trace(name, lowdim_rge),
                        "transformed_update": self._bounded_tensor_trace(
                            name, transformed
                        ),
                        "full_update": self._bounded_tensor_trace(name, update),
                        "param_before": self._bounded_tensor_trace(name, param.data),
                    }
                    self._lowdim_update_trace.append(update_entry)
                else:
                    self._perturbation_trace_truncated = True
            scale = -float(learning_rate)
            update_trace.add_delta(name, update, scale=scale)
            self._apply_lowdim_parameter_update(
                param,
                update,
                learning_rate=learning_rate,
            )
            if self._records_source_debug_trace() and self._lowdim_update_trace:
                last_trace = self._lowdim_update_trace[-1]
                if last_trace.get("name") == name and "param_after" not in last_trace:
                    last_trace["param_after"] = self._bounded_tensor_trace(
                        name, param.data
                    )

    def _build_result(
        self,
        *,
        seed: int,
        eps: float,
        f_plus: float | None,
        f_minus: float | None,
        objective_delta: float | None,
        projected_grads: list[float],
        candidate_results: list[dict[str, Any]],
        parameters: ParameterList,
        num_objective_calls: int,
        loss0: float | None,
        update_norm: float,
        parameter_delta_checksum: str,
    ) -> HFZOMethodStepResult:
        projected_grad_mean = (
            sum(projected_grads) / len(projected_grads) if projected_grads else 0.0
        )
        if len(projected_grads) > 1:
            projected_grad_std = float(
                torch.tensor(projected_grads).std(unbiased=False).item()
            )
        else:
            projected_grad_std = 0.0
        diagnostics = self._diagnostics(
            parameters,
            num_objective_calls=num_objective_calls,
            loss0=loss0,
            projected_grads=projected_grads,
            projected_grad_mean=projected_grad_mean,
            projected_grad_std=projected_grad_std,
        )
        return HFZOMethodStepResult(
            zo_method=self.name,
            seed=int(seed),
            eps=float(eps),
            learning_rate=float(self.config.learning_rate),
            f_plus=f_plus,
            f_minus=f_minus,
            objective_delta=objective_delta,
            projected_grad=float(projected_grad_mean),
            candidate_results=candidate_results,
            update_norm=update_norm,
            parameter_delta_checksum=parameter_delta_checksum,
            method_diagnostics=diagnostics,
        )

    def _diagnostics(
        self,
        parameters: ParameterList,
        *,
        num_objective_calls: int,
        loss0: float | None,
        projected_grads: list[float],
        projected_grad_mean: float,
        projected_grad_std: float,
    ) -> dict[str, Any]:
        matrix_count = 0
        fallback_count = 0
        effective_ranks: list[int] = []
        streaming_pi_ks: list[int] = []
        for _, param in parameters:
            if self._is_matrix_parameter(param):
                matrix_count += int(param.numel())
                effective_rank = self._effective_rank(param.data)
                effective_ranks.append(effective_rank)
                streaming_pi_ks.append(
                    min(
                        int(self.lowdim_config.k_start),
                        int(effective_rank),
                        int(param.data.shape[1]),
                    )
                )
            else:
                fallback_count += int(param.numel())
        update_policy = self._update_policy_name()
        effective_update_transform = self._effective_update_transform_name()
        streaming_pi_k = (
            max(streaming_pi_ks)
            if effective_update_transform == "streaming_power_iteration"
            and streaming_pi_ks
            else 0
        )
        diagnostics = {
            "authority_source": "paper+official_source",
            "method_source_commit": METHOD_SOURCE_COMMITS[self.name],
            "estimator_mode": (
                "one_side_multi_sample"
                if bool(self.lowdim_config.multiple_sample)
                else "two_side_single_sample"
            ),
            "method_config": method_config_diagnostics(
                eps=float(self.config.eps),
                learning_rate=float(self.config.learning_rate),
                parameter_scope=str(self.config.parameter_scope),
                weight_decay=float(self.config.weight_decay),
                lowdim_muon=self.lowdim_config,
            ),
            "seed_stream": str(self.config.seed_mode),
            "update_policy": update_policy,
            "effective_update_transform": effective_update_transform,
            "uses_lowdim_momentum": self._uses_lowdim_momentum(),
            "effective_estimator_mode": (
                "one_side_multi_sample"
                if bool(self.lowdim_config.multiple_sample)
                else "two_side_single_sample"
            ),
            "script_compatible_perturbation_mode": str(
                self.lowdim_config.perturbation_mode
            ),
            "objective_calls": int(num_objective_calls),
            "num_objective_calls": int(num_objective_calls),
            "multiple_sample": bool(self.lowdim_config.multiple_sample),
            "num_samples": int(self.lowdim_config.num_samples),
            "rank": int(self.lowdim_config.rank),
            "effective_rank_min": min(effective_ranks) if effective_ranks else 0,
            "effective_rank_max": max(effective_ranks) if effective_ranks else 0,
            "step_interval": int(self.lowdim_config.step_interval),
            "phase2_optimizer": str(self.lowdim_config.phase2_optimizer),
            "phase2_steps": int(self.lowdim_config.phase2_steps),
            "pion_steps": int(self.lowdim_config.pion_steps),
            "pion_promotion_steps": int(self.lowdim_config.pion_promotion_steps),
            "beta": float(self.lowdim_config.beta),
            "one_d_lr": float(self.lowdim_config.one_d_lr),
            "matrix_parameter_count": int(matrix_count),
            "fallback_parameter_count": int(fallback_count),
            "projection_refresh_count": int(self._projection_refresh_count),
            "projection_refresh_indexing": str(self.lowdim_config.refresh_indexing),
            "p_state_checksum": _state_checksum(self.P_matrices),
            "v_state_checksum": _state_checksum(self.V_matrices),
            "pre_transform_update_norm": math.sqrt(
                self._pre_transform_update_squared_norm
            ),
            "post_transform_update_norm": math.sqrt(
                self._post_transform_update_squared_norm
            ),
            "streaming_pi_k": streaming_pi_k,
            "gradient_change": float(self._last_gradient_change),
            "loss0": None if loss0 is None else float(loss0),
            "projected_grads": [float(value) for value in projected_grads],
            "projected_grad_mean": float(projected_grad_mean),
            "projected_grad_std": float(projected_grad_std),
        }
        if self._records_source_debug_trace():
            diagnostics.update(
                {
                    "perturbation_trace_policy": "bounded_source_debug",
                    "perturbation_trace": list(self._perturbation_trace),
                    "perturbation_call_trace": list(self._perturbation_call_trace),
                    "update_trace": list(self._lowdim_update_trace),
                    "perturbation_trace_truncated": bool(
                        self._perturbation_trace_truncated
                    ),
                    "retained_projection_samples": self._retained_projection_samples(),
                }
            )
            if self._records_expensive_source_trace():
                diagnostics.update(
                    {
                        "source_style_p_state_checksum": _source_style_state_checksum(
                            self.P_matrices
                        ),
                        "source_style_v_state_checksum": _source_style_state_checksum(
                            self.V_matrices
                        ),
                    }
                )
        return diagnostics


def _should_weight_decay(name: str) -> bool:
    lowered = name.lower()
    return (
        "bias" not in lowered
        and "layer_norm" not in lowered
        and "layernorm" not in lowered
    )


def _bounded_squared_norm(
    name: str,
    tensor: torch.Tensor,
    *,
    error_prefix: str,
) -> float:
    _, squared_norm = bounded_tensor_fingerprint(
        name,
        tensor,
        error_prefix=error_prefix,
    )
    if not math.isfinite(squared_norm):
        raise ValueError(f"{error_prefix} norm must be finite")
    return squared_norm


def _bounded_mean_abs_difference(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    error_prefix: str,
    max_values: int = 8,
) -> float:
    if tuple(before.shape) != tuple(after.shape):
        raise ValueError(f"{error_prefix} tensors must have matching shapes")
    numel = int(before.numel())
    if numel == 0:
        return 0.0
    sample_count = min(max_values, numel)
    if sample_count == 1:
        flat_indices = [0]
    else:
        flat_indices = [
            round(index * (numel - 1) / (sample_count - 1))
            for index in range(sample_count)
        ]
    total = 0.0
    for flat_index in flat_indices:
        value = abs(
            _scalar_at_flat_index(after, flat_index)
            - _scalar_at_flat_index(before, flat_index)
        )
        if not math.isfinite(value):
            raise ValueError(f"{error_prefix} sample values must be finite")
        total += value
    return total / sample_count


def _scalar_at_flat_index(tensor: torch.Tensor, flat_index: int) -> float:
    if tensor.ndim == 0:
        return float(tensor.item())
    remaining = int(flat_index)
    indices: list[int] = []
    for size in reversed(tensor.shape):
        size_int = int(size)
        indices.append(remaining % size_int)
        remaining //= size_int
    return float(tensor[tuple(reversed(indices))].item())
