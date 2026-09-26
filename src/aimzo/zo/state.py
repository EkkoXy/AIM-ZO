from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from aimzo.logging import write_json_atomic
from aimzo.zo.params import (
    canonical_seeded_noise_like,
    parameter_selection_metadata,
    select_trainable_parameters,
)
from aimzo.zo.lora_state import LoRAStateStore

PARAMETER_CONFIG_FILENAME = "parameter_state.json"
PARAMETER_SAFETENSORS_FILENAME = "parameter_state.safetensors"
PARAMETER_TORCH_FILENAME = "parameter_state.bin"
ZO_STATE_FILENAME = "zo_state.json"


class ParameterStateStore:
    """Owns the main-process center tensors for a selected parameter scope."""

    def __init__(
        self,
        model: torch.nn.Module,
        parameter_scope: str = "full_parameters",
    ) -> None:
        self.model = model
        self.parameter_scope = parameter_scope
        self._lora_store: LoRAStateStore | None = None
        if parameter_scope == "lora":
            self._lora_store = LoRAStateStore(model)
            return
        self._parameters = select_trainable_parameters(model, parameter_scope)
        if not self._parameters:
            raise ValueError(
                f"No trainable parameters found for scope {parameter_scope!r}"
            )
        self._center_state = self._capture_model_state()

    def checksum(self) -> str:
        if self._lora_store is not None:
            return self._lora_store.checksum()
        return _checksum_state(self._center_state)

    @property
    def current_state_metadata(self) -> dict[str, Any]:
        if self._lora_store is not None:
            metadata = parameter_selection_metadata(self.model, self.parameter_scope)
            metadata.update(self._lora_store.current_state_metadata)
            return metadata
        metadata = parameter_selection_metadata(self.model, self.parameter_scope)
        metadata.update(
            {
                "checksum": self.checksum(),
                "num_tensors": len(self._center_state),
                "numel": sum(
                    int(tensor.numel()) for tensor in self._center_state.values()
                ),
                "parameters": [
                    {
                        "name": name,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                    }
                    for name, tensor in self._center_state.items()
                ],
            }
        )
        return metadata

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self._lora_store is not None:
            return self._lora_store.state_dict()
        return OrderedDict(
            (name, tensor.detach().clone())
            for name, tensor in self._center_state.items()
        )

    @torch.no_grad()
    def perturb(
        self,
        *,
        seed: int,
        eps: float,
        scaling: float,
    ) -> dict[str, torch.Tensor]:
        if self._lora_store is not None:
            return self._lora_store.perturb(seed=seed, eps=eps, scaling=scaling)
        perturbed: dict[str, torch.Tensor] = OrderedDict()
        for index, (name, tensor) in enumerate(self._center_state.items()):
            z = canonical_seeded_noise_like(tensor, seed=seed, index=index)
            perturbed[name] = tensor.add(z, alpha=float(eps) * float(scaling))
        return perturbed

    def materialize_candidate(
        self,
        *,
        seed: int,
        eps: float,
        scaling: float,
        output_dir: str | Path,
    ) -> dict[str, torch.Tensor]:
        if self._lora_store is not None:
            return self._lora_store.materialize_candidate(
                seed=seed,
                eps=eps,
                scaling=scaling,
                output_dir=output_dir,
            )
        state = self.perturb(seed=seed, eps=eps, scaling=scaling)
        self._save_state(
            output_dir,
            state,
            config_extra={
                "candidate": {
                    "seed": int(seed),
                    "eps": float(eps),
                    "scaling": float(scaling),
                    "center_checksum": self.checksum(),
                }
            },
        )
        return OrderedDict((name, tensor.clone()) for name, tensor in state.items())

    @torch.no_grad()
    def apply_update(
        self,
        *,
        projected_grad: float,
        seed: int,
        learning_rate: float,
        weight_decay: float = 0.0,
    ) -> None:
        if self._lora_store is not None:
            self._lora_store.apply_update(
                projected_grad=projected_grad,
                seed=seed,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            return
        updated: dict[str, torch.Tensor] = OrderedDict()
        for index, (name, tensor) in enumerate(self._center_state.items()):
            z = canonical_seeded_noise_like(tensor, seed=seed, index=index)
            update = z.mul(float(projected_grad))
            if weight_decay:
                update = update.add(tensor, alpha=float(weight_decay))
            updated[name] = tensor.add(update, alpha=-float(learning_rate))
        self._center_state = OrderedDict(
            (name, tensor.detach().clone()) for name, tensor in updated.items()
        )
        self._copy_center_to_model()

    def save(
        self,
        output_dir: str | Path,
        *,
        zo_state: dict[str, Any] | None = None,
    ) -> None:
        if self._lora_store is not None:
            self._lora_store.save_adapter(output_dir, zo_state=zo_state)
            return
        self._save_state(output_dir, self._center_state, zo_state=zo_state)

    def state_payload(self) -> dict[str, Any]:
        if self._lora_store is not None:
            raise ValueError("state_payload is only available for full_parameters")
        state = OrderedDict(
            (name, tensor.detach().cpu().contiguous().clone())
            for name, tensor in self._center_state.items()
        )
        payload = _parameter_config_payload(self.parameter_scope, state)
        payload["checksum"] = payload["state_checksum"]
        payload["state"] = state
        return payload

    @torch.no_grad()
    def load(self, input_dir: str | Path) -> dict[str, Any] | None:
        if self._lora_store is not None:
            return self._lora_store.load_adapter(input_dir)
        input_path = Path(input_dir)
        config = _load_parameter_config(input_path)
        self._validate_checkpoint_config(config)
        state = _load_state(input_path, config)
        self._validate_loaded_state_checksum(state, config)
        self._validate_state_matches_model(state)
        self._center_state = OrderedDict(
            (name, tensor.detach().clone()) for name, tensor in state.items()
        )
        self._copy_center_to_model()
        return self.read_zo_state(input_path)

    @staticmethod
    def read_zo_state(input_dir: str | Path) -> dict[str, Any] | None:
        zo_state_path = Path(input_dir) / ZO_STATE_FILENAME
        if not zo_state_path.exists():
            return None
        return json.loads(zo_state_path.read_text(encoding="utf-8"))

    def _capture_model_state(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (name, param.detach().clone()) for name, param in self._parameters
        )

    def _validate_state_matches_model(self, state: dict[str, torch.Tensor]) -> None:
        expected = OrderedDict((name, param) for name, param in self._parameters)
        if list(state) != list(expected):
            raise ValueError(
                "Parameter state names do not match selected model parameters"
            )
        for name, tensor in state.items():
            param = expected[name]
            if tuple(tensor.shape) != tuple(param.shape):
                raise ValueError(
                    f"Parameter tensor shape for {name!r} is {tuple(tensor.shape)}, "
                    f"expected {tuple(param.shape)}"
                )
            if tensor.dtype != param.dtype:
                raise ValueError(
                    f"Parameter tensor dtype for {name!r} is {tensor.dtype}, "
                    f"expected {param.dtype}"
                )

    def _validate_checkpoint_config(self, config: dict[str, Any]) -> None:
        checkpoint_format = config.get("checkpoint_format")
        if checkpoint_format != "zo_parameter_state":
            raise ValueError(
                "Parameter checkpoint format must be 'zo_parameter_state', "
                f"got {checkpoint_format!r}"
            )
        parameter_scope = config.get("parameter_scope")
        if parameter_scope != self.parameter_scope:
            raise ValueError(
                f"Parameter checkpoint scope is {parameter_scope!r}, "
                f"expected {self.parameter_scope!r}"
            )
        parameters = config.get("parameters")
        if not isinstance(parameters, list):
            raise ValueError(
                "Parameter checkpoint manifest is missing parameters metadata"
            )
        expected = OrderedDict((name, param) for name, param in self._parameters)
        manifest_names: list[str] = []
        for entry in parameters:
            if not isinstance(entry, dict):
                raise ValueError(
                    "Parameter checkpoint manifest contains invalid metadata"
                )
            name = entry.get("name")
            if not isinstance(name, str):
                raise ValueError(
                    "Parameter checkpoint manifest contains an unnamed tensor"
                )
            manifest_names.append(name)
            if name not in expected:
                raise ValueError(f"Unexpected parameter tensor {name!r} in checkpoint")
            dtype = entry.get("dtype")
            expected_dtype = str(expected[name].dtype)
            if dtype != expected_dtype:
                raise ValueError(
                    f"Parameter tensor dtype metadata for {name!r} is {dtype!r}, "
                    f"expected {expected_dtype!r}"
                )
            shape = entry.get("shape")
            expected_shape = list(expected[name].shape)
            if shape != expected_shape:
                raise ValueError(
                    f"Parameter tensor shape metadata for {name!r} is {shape!r}, "
                    f"expected {expected_shape!r}"
                )
        parameter_names = config.get("parameter_names")
        if parameter_names != manifest_names:
            raise ValueError(
                "Parameter checkpoint names do not match parameters metadata"
            )
        if manifest_names != list(expected):
            raise ValueError(
                "Parameter checkpoint names do not match selected model parameters"
            )

    def _validate_loaded_state_checksum(
        self,
        state: dict[str, torch.Tensor],
        config: dict[str, Any],
    ) -> None:
        expected_checksum = config.get("state_checksum")
        if not isinstance(expected_checksum, str) or not expected_checksum:
            raise ValueError("Parameter checkpoint manifest is missing state_checksum")
        actual_checksum = _checksum_state(state)
        if actual_checksum != expected_checksum:
            raise ValueError(
                "Parameter checkpoint tensor checksum does not match manifest"
            )

    @torch.no_grad()
    def _copy_center_to_model(self) -> None:
        expected = OrderedDict((name, param) for name, param in self._parameters)
        for name, tensor in self._center_state.items():
            expected[name].copy_(
                tensor.to(device=expected[name].device, dtype=expected[name].dtype)
            )

    def _save_state(
        self,
        output_dir: str | Path,
        state: dict[str, torch.Tensor],
        *,
        config_extra: dict[str, Any] | None = None,
        zo_state: dict[str, Any] | None = None,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _save_state_tensors(output_path, state)
        config = _parameter_config_payload(self.parameter_scope, state)
        if config_extra:
            config.update(config_extra)
        write_json_atomic(output_path / PARAMETER_CONFIG_FILENAME, config)
        if zo_state is not None:
            write_json_atomic(output_path / ZO_STATE_FILENAME, zo_state)


def _parameter_config_payload(
    parameter_scope: str,
    state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return {
        "checkpoint_format": "zo_parameter_state",
        "parameter_scope": parameter_scope,
        "state_checksum": _checksum_state(state),
        "parameter_names": list(state),
        "parameters": [
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            for name, tensor in state.items()
        ],
    }


def _save_state_tensors(output_path: Path, state: dict[str, torch.Tensor]) -> None:
    for filename in (PARAMETER_SAFETENSORS_FILENAME, PARAMETER_TORCH_FILENAME):
        stale_path = output_path / filename
        if stale_path.exists():
            stale_path.unlink()
    cpu_state = OrderedDict(
        (name, tensor.detach().cpu().contiguous()) for name, tensor in state.items()
    )
    try:
        from safetensors.torch import save_file
    except ImportError:
        torch.save(cpu_state, output_path / PARAMETER_TORCH_FILENAME)
    else:
        save_file(cpu_state, output_path / PARAMETER_SAFETENSORS_FILENAME)


def _load_parameter_config(input_path: Path) -> dict[str, Any]:
    config_path = input_path / PARAMETER_CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing parameter checkpoint manifest {PARAMETER_CONFIG_FILENAME}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Parameter checkpoint manifest must be a JSON object")
    return config


def _load_state(
    input_path: Path,
    config: dict[str, Any],
) -> OrderedDict[str, torch.Tensor]:
    safetensors_path = input_path / PARAMETER_SAFETENSORS_FILENAME
    torch_path = input_path / PARAMETER_TORCH_FILENAME
    has_safetensors = safetensors_path.exists()
    has_torch = torch_path.exists()
    if has_safetensors and has_torch:
        raise ValueError(
            f"Parameter directory {input_path} contains both "
            f"{PARAMETER_SAFETENSORS_FILENAME} and {PARAMETER_TORCH_FILENAME}; "
            "remove one tensor file before loading."
        )
    if has_safetensors:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                f"{PARAMETER_SAFETENSORS_FILENAME} exists but safetensors is not installed"
            ) from exc
        loaded = load_file(safetensors_path)
    elif has_torch:
        loaded = torch.load(torch_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No parameter tensor file found in {input_path}: expected "
            f"{PARAMETER_SAFETENSORS_FILENAME} or {PARAMETER_TORCH_FILENAME}"
        )

    names = config.get("parameter_names")
    if not isinstance(names, list):
        raise ValueError("Parameter checkpoint manifest is missing parameter_names")
    try:
        return OrderedDict((str(name), loaded[str(name)]) for name in names)
    except KeyError as exc:
        raise ValueError(
            f"Parameter tensor file is missing tensor {exc.args[0]!r}"
        ) from exc


def _checksum_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
