from __future__ import annotations

import torch

ParameterList = list[tuple[str, torch.nn.Parameter]]


def lora_trainable_parameters(
    model: torch.nn.Module,
) -> ParameterList:
    return [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and "lora" in name.lower()
    ]


def select_trainable_parameters(
    model: torch.nn.Module,
    parameter_scope: str = "full_parameters",
) -> ParameterList:
    if parameter_scope == "lora":
        return lora_trainable_parameters(model)
    if parameter_scope == "full_parameters":
        return [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and torch.is_floating_point(param)
        ]
    raise ValueError("parameter_scope must be one of: full_parameters, lora")


def parameter_selection_metadata(
    model: torch.nn.Module,
    parameter_scope: str,
    *,
    preview_limit: int = 20,
) -> dict[str, object]:
    selected = select_trainable_parameters(model, parameter_scope)
    rule = (
        "requires_grad_lora_name"
        if parameter_scope == "lora"
        else "requires_grad_floating_parameters"
    )
    return {
        "parameter_scope": parameter_scope,
        "parameter_selection_rule": rule,
        "num_selected_tensors": len(selected),
        "num_selected_parameters": sum(int(param.numel()) for _, param in selected),
        "selected_tensor_names_preview": [
            name for name, _ in selected[: int(preview_limit)]
        ],
    }


def _randn_like_with_generator(
    param: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randn(
        param.shape,
        generator=generator,
        device=param.device,
        dtype=param.dtype,
    )


def canonical_seeded_noise_like(
    tensor: torch.Tensor,
    *,
    seed: int,
    index: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(index))
    noise = torch.randn(
        tensor.shape,
        generator=generator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return noise.to(device=tensor.device, dtype=tensor.dtype)


def _generator_for_param(param: torch.Tensor, *, seed: int, index: int) -> torch.Generator:
    generator = torch.Generator(device=param.device)
    generator.manual_seed(int(seed) + int(index))
    return generator


@torch.no_grad()
def apply_device_seeded_perturbation(
    model: torch.nn.Module,
    *,
    seed: int,
    eps: float,
    scaling: float,
    parameter_scope: str = "full_parameters",
) -> None:
    torch.manual_seed(int(seed))
    for _, param in select_trainable_parameters(model, parameter_scope):
        z = torch.randn_like(param.data)
        param.add_(z, alpha=float(eps) * float(scaling))


@torch.no_grad()
def apply_device_seeded_update(
    model: torch.nn.Module,
    *,
    seed: int,
    projected_grad: float,
    learning_rate: float,
    weight_decay: float = 0.0,
    parameter_scope: str = "full_parameters",
) -> None:
    torch.manual_seed(int(seed))
    for _, param in select_trainable_parameters(model, parameter_scope):
        z = torch.randn_like(param.data)
        update = z.mul(float(projected_grad))
        if weight_decay:
            update = update.add(param.data, alpha=float(weight_decay))
        param.add_(update, alpha=-float(learning_rate))


@torch.no_grad()
def apply_seeded_perturbation(
    model: torch.nn.Module,
    *,
    seed: int,
    eps: float,
    scaling: float,
    parameter_scope: str = "full_parameters",
) -> None:
    for index, (_, param) in enumerate(
        select_trainable_parameters(model, parameter_scope)
    ):
        if parameter_scope == "full_parameters":
            z = canonical_seeded_noise_like(param, seed=seed, index=index)
        else:
            generator = _generator_for_param(param, seed=seed, index=index)
            z = _randn_like_with_generator(param, generator)
        param.add_(z, alpha=float(eps) * float(scaling))


@torch.no_grad()
def apply_seeded_update(
    model: torch.nn.Module,
    *,
    seed: int,
    projected_grad: float,
    learning_rate: float,
    weight_decay: float = 0.0,
    parameter_scope: str = "full_parameters",
) -> None:
    for index, (_, param) in enumerate(
        select_trainable_parameters(model, parameter_scope)
    ):
        if parameter_scope == "full_parameters":
            z = canonical_seeded_noise_like(param, seed=seed, index=index)
        else:
            generator = _generator_for_param(param, seed=seed, index=index)
            z = _randn_like_with_generator(param, generator)
        update = z.mul(float(projected_grad))
        if weight_decay:
            update = update.add(param, alpha=float(weight_decay))
        param.add_(update, alpha=-float(learning_rate))
