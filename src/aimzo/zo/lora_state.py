from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from aimzo.logging import write_json_atomic
from aimzo.zo.params import lora_trainable_parameters

ADAPTER_CONFIG_FILENAME = "adapter_config.json"
SAFETENSORS_FILENAME = "adapter_model.safetensors"
TORCH_FILENAME = "adapter_model.bin"
ZO_STATE_FILENAME = "zo_state.json"
_PEFT_CONFIG_FIELDS = (
    "base_model_name_or_path",
    "revision",
    "task_type",
    "inference_mode",
    "r",
    "target_modules",
    "lora_alpha",
    "lora_dropout",
    "fan_in_fan_out",
    "bias",
    "use_rslora",
    "modules_to_save",
    "init_lora_weights",
    "layers_to_transform",
    "layers_pattern",
    "rank_pattern",
    "alpha_pattern",
    "megatron_config",
    "megatron_core",
    "loftq_config",
    "use_dora",
    "layer_replication",
    "auto_mapping",
)


class LoRAStateStore:
    """Owns the main-process center LoRA adapter tensors."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self._parameters = lora_trainable_parameters(model)
        if not self._parameters:
            raise ValueError("No trainable LoRA parameters found")
        self._center_state = self._capture_model_state()

    def checksum(self) -> str:
        return _checksum_state(self._center_state)

    @property
    def current_state_metadata(self) -> dict[str, Any]:
        return {
            "checksum": self.checksum(),
            "num_tensors": len(self._center_state),
            "numel": sum(int(tensor.numel()) for tensor in self._center_state.values()),
            "parameters": [
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
                for name, tensor in self._center_state.items()
            ],
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return OrderedDict(
            (name, tensor.detach().clone()) for name, tensor in self._center_state.items()
        )

    @torch.no_grad()
    def perturb(
        self,
        *,
        seed: int,
        eps: float,
        scaling: float,
    ) -> dict[str, torch.Tensor]:
        perturbed: dict[str, torch.Tensor] = OrderedDict()
        for index, (name, tensor) in enumerate(self._center_state.items()):
            generator = _generator_for_tensor(tensor, seed=seed, index=index)
            z = _randn_like_with_generator(tensor, generator)
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
        state = self.perturb(seed=seed, eps=eps, scaling=scaling)
        self._save_state(
            output_dir,
            state,
            adapter_config_extra={
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
        updated: dict[str, torch.Tensor] = OrderedDict()
        for index, (name, tensor) in enumerate(self._center_state.items()):
            generator = _generator_for_tensor(tensor, seed=seed, index=index)
            z = _randn_like_with_generator(tensor, generator)
            update = z.mul(float(projected_grad))
            if weight_decay:
                update = update.add(tensor, alpha=float(weight_decay))
            updated[name] = tensor.add(update, alpha=-float(learning_rate))
        self._center_state = OrderedDict(
            (name, tensor.detach().clone()) for name, tensor in updated.items()
        )
        self._copy_center_to_model()

    def save_adapter(
        self,
        output_dir: str | Path,
        *,
        zo_state: dict[str, Any] | None = None,
    ) -> None:
        self._save_state(output_dir, self._center_state, zo_state=zo_state)

    @torch.no_grad()
    def load_adapter(self, input_dir: str | Path) -> dict[str, Any] | None:
        input_path = Path(input_dir)
        state = _load_state(input_path)
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
                "Adapter state parameter names do not match model LoRA parameters"
            )
        for name, tensor in state.items():
            param = expected[name]
            if tuple(tensor.shape) != tuple(param.shape):
                raise ValueError(
                    f"Adapter tensor shape for {name!r} is {tuple(tensor.shape)}, "
                    f"expected {tuple(param.shape)}"
                )

    @torch.no_grad()
    def _copy_center_to_model(self) -> None:
        expected = OrderedDict((name, param) for name, param in self._parameters)
        for name, tensor in self._center_state.items():
            expected[name].copy_(tensor.to(device=expected[name].device, dtype=expected[name].dtype))

    def _save_state(
        self,
        output_dir: str | Path,
        state: dict[str, torch.Tensor],
        *,
        adapter_config_extra: dict[str, Any] | None = None,
        zo_state: dict[str, Any] | None = None,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        config = _adapter_config_payload(self.model, state)
        tensor_name_mapping = config.get("tensor_name_mapping")
        if isinstance(tensor_name_mapping, dict):
            state_to_save = _remap_state_keys(state, tensor_name_mapping)
        else:
            state_to_save = state
        _save_state_tensors(output_path, state_to_save)
        if adapter_config_extra:
            config.update(adapter_config_extra)
        write_json_atomic(output_path / ADAPTER_CONFIG_FILENAME, config)
        if zo_state is not None:
            write_json_atomic(output_path / ZO_STATE_FILENAME, zo_state)


def _adapter_config_payload(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    peft_config = _peft_adapter_config_payload(model)
    if peft_config is not None:
        peft_config["parameter_names"] = list(state)
        peft_config["tensor_name_mapping"] = {
            name: _peft_saved_tensor_name(name) for name in state
        }
        return peft_config

    return {
        "format": "aimzo_local_lora_state",
        "compatibility": "test_or_local_only",
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


def _peft_adapter_config_payload(model: torch.nn.Module) -> dict[str, Any] | None:
    peft_configs = getattr(model, "peft_config", None)
    if not peft_configs:
        return None

    config = None
    active_adapter = getattr(model, "active_adapter", None)
    if callable(active_adapter):
        active_adapter = active_adapter()
    if isinstance(peft_configs, dict):
        if active_adapter is not None and active_adapter in peft_configs:
            config = peft_configs[active_adapter]
        elif len(peft_configs) == 1:
            config = next(iter(peft_configs.values()))
    else:
        config = peft_configs
    if config is None:
        return None

    raw = _config_to_dict(config)
    payload = {
        key: _jsonable_config_value(raw[key])
        for key in _PEFT_CONFIG_FIELDS
        if key in raw
    }
    if "peft_type" in raw:
        payload["peft_type"] = _jsonable_config_value(raw["peft_type"])
    else:
        payload["peft_type"] = "LORA"
    return payload


def _config_to_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    return {
        key: value
        for key, value in vars(config).items()
        if not key.startswith("_")
    }


def _jsonable_config_value(value: Any) -> Any:
    if isinstance(value, set | tuple):
        return list(value)
    if isinstance(value, dict):
        return {
            str(key): _jsonable_config_value(child_value)
            for key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_jsonable_config_value(child_value) for child_value in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _remap_state_keys(
    state: dict[str, torch.Tensor],
    tensor_name_mapping: dict[str, str],
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (tensor_name_mapping.get(name, name), tensor)
        for name, tensor in state.items()
    )


def _peft_saved_tensor_name(name: str) -> str:
    parts = name.split(".")
    for lora_token in ("lora_embedding_A", "lora_embedding_B"):
        if len(parts) >= 2 and parts[-2] == lora_token:
            return ".".join(parts[:-1])
    for lora_token in (
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
    ):
        for index, part in enumerate(parts[:-2]):
            if part == lora_token:
                return ".".join(parts[: index + 1] + parts[index + 2 :])
    return name


def _save_state_tensors(output_path: Path, state: dict[str, torch.Tensor]) -> None:
    for filename in (SAFETENSORS_FILENAME, TORCH_FILENAME):
        stale_path = output_path / filename
        if stale_path.exists():
            stale_path.unlink()
    cpu_state = OrderedDict(
        (name, tensor.detach().cpu().contiguous()) for name, tensor in state.items()
    )
    try:
        from safetensors.torch import save_file
    except ImportError:
        torch.save(cpu_state, output_path / TORCH_FILENAME)
    else:
        save_file(cpu_state, output_path / SAFETENSORS_FILENAME)


def _load_state(input_path: Path) -> OrderedDict[str, torch.Tensor]:
    safetensors_path = input_path / SAFETENSORS_FILENAME
    torch_path = input_path / TORCH_FILENAME
    has_safetensors = safetensors_path.exists()
    has_torch = torch_path.exists()
    if has_safetensors and has_torch:
        raise ValueError(
            f"Adapter directory {input_path} contains both {SAFETENSORS_FILENAME} "
            f"and {TORCH_FILENAME}; remove one tensor file before loading."
        )
    if has_safetensors:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                f"{SAFETENSORS_FILENAME} exists but safetensors is not installed"
            ) from exc
        loaded = load_file(safetensors_path)
    elif has_torch:
        loaded = torch.load(torch_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No adapter tensor file found in {input_path}: expected "
            f"{SAFETENSORS_FILENAME} or {TORCH_FILENAME}"
        )

    config_path = input_path / ADAPTER_CONFIG_FILENAME
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        names = config.get("parameter_names")
        tensor_name_mapping = config.get("tensor_name_mapping")
        if isinstance(names, list):
            if isinstance(tensor_name_mapping, dict):
                return OrderedDict(
                    (str(name), loaded[str(tensor_name_mapping.get(str(name), str(name)))])
                    for name in names
                )
            return OrderedDict((str(name), loaded[str(name)]) for name in names)

    return OrderedDict((name, loaded[name]) for name in sorted(loaded))


def _checksum_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _randn_like_with_generator(
    tensor: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randn(
        tensor.shape,
        generator=generator,
        device=tensor.device,
        dtype=tensor.dtype,
    )


def _generator_for_tensor(
    tensor: torch.Tensor,
    *,
    seed: int,
    index: int,
) -> torch.Generator:
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(int(seed) + int(index))
    return generator
