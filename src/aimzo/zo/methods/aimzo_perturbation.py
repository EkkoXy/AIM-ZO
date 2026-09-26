from __future__ import annotations

import math

import torch

from .aimzo_utils import stable_int_seed


def sample_probe_a(
    *,
    name: str,
    probe_seed: int,
    out_dim: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    orthogonal: bool,
) -> torch.Tensor:
    """Sample the left factor used by one structured AIMZO probe."""
    seed = stable_int_seed(
        {
            "method": "agzo_abh_probe_a",
            "name": str(name),
            "probe_seed": int(probe_seed),
            "rank": int(rank),
            "out_dim": int(out_dim),
        }
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    sample_dtype = (
        torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    )
    factor = torch.randn(
        (int(out_dim), int(rank)),
        device=device,
        dtype=sample_dtype,
        generator=generator,
    )
    if orthogonal:
        factor, _ = torch.linalg.qr(factor, mode="reduced")
    return factor.to(dtype=dtype).detach().contiguous()


def sample_probe_b(
    *,
    name: str,
    probe_seed: int,
    block: int,
    rank: int,
    span_rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample the small coefficient factor between A and Q_active."""
    seed = stable_int_seed(
        {
            "method": "agzo_abh_probe_b",
            "name": str(name),
            "probe_seed": int(probe_seed),
            "rank": int(rank),
            "span_rank": int(span_rank),
            "block": int(block),
        }
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        (int(rank), int(span_rank)),
        device=device,
        dtype=dtype,
        generator=generator,
    ).detach().contiguous()


def low_rank_frobenius_norm(
    a_matrix: torch.Tensor,
    b_matrix: torch.Tensor,
    q_active: torch.Tensor,
) -> float | None:
    """Compute ||A B Q||_F without materializing the dense perturbation."""
    a_float = a_matrix.float()
    right_float = b_matrix.float().matmul(q_active.float())
    a_gram = a_float.transpose(0, 1).matmul(a_float)
    norm_sq = a_gram.matmul(right_float).mul(right_float).sum()
    if not bool(torch.isfinite(norm_sq).item()):
        return None
    return math.sqrt(max(float(norm_sq.item()), 0.0))


def perturbation_normalization_scale(
    *,
    normalization: str,
    a_matrix: torch.Tensor,
    b_matrix: torch.Tensor,
    q_active: torch.Tensor,
    parameter_numel: int,
    layer_fro_scale: float,
) -> tuple[float, float | None, float]:
    """Return scale, pre-normalization norm, and the layer-Fro target norm."""
    target_norm = math.sqrt(float(parameter_numel)) * float(layer_fro_scale)
    if normalization == "none":
        return 1.0, None, target_norm

    pre_norm = low_rank_frobenius_norm(a_matrix, b_matrix, q_active)
    if normalization == "layer_fro":
        if target_norm > 0.0 and pre_norm is not None and pre_norm > 0.0:
            return target_norm / (pre_norm + 1e-12), pre_norm, target_norm
        return 1.0, pre_norm, target_norm

    normalizer = math.sqrt(float(b_matrix.shape[0] * q_active.shape[0]))
    scale = 1.0 / normalizer if normalizer > 0.0 else 1.0
    return scale, pre_norm, target_norm
