from __future__ import annotations

import torch

from .aimzo_utils import stable_int_seed


def initialize_oja_basis(
    *,
    module_name: str,
    feature_dim: int,
    right_rank: int,
    device: torch.device,
    seed: int | None,
) -> torch.Tensor:
    init_seed = stable_int_seed(
        {
            "method": "agzo_abh_oja_q",
            "name": str(module_name),
            "feature_dim": int(feature_dim),
            "right_rank": int(right_rank),
            "seed": int(seed or 0),
        }
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(init_seed)
    raw = torch.randn(
        (int(feature_dim), int(right_rank)),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q[:, :right_rank].detach().contiguous()


def oja_covariance_action(tokens: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute H^T H Q / T without forming the feature covariance matrix."""
    denominator = max(1, int(tokens.shape[0]))
    return tokens.t().matmul(tokens.matmul(q)).div(float(denominator))


def orthonormalize_columns(q: torch.Tensor) -> torch.Tensor:
    orthogonal, _ = torch.linalg.qr(q.float(), mode="reduced")
    return orthogonal


def select_top_tail_basis(
    q: torch.Tensor,
    *,
    module_name: str,
    step: int,
    noise_seed: int | None,
    active_rank: int,
    top_count: int,
    tail_count: int,
) -> torch.Tensor:
    """Select the deterministic head and a reproducible random spectral tail.

    ``q`` stores the wide basis column-wise. The returned active basis is
    row-wise because AIMZO constructs right-structured perturbations as
    ``A @ B @ Q_active``.
    """
    active_rank = min(int(active_rank), int(q.shape[1]))
    if active_rank <= 0:
        return q[:, :0].t().detach().contiguous()
    if top_count <= 0 and tail_count <= 0:
        return q[:, :active_rank].t().detach().contiguous()
    if top_count + tail_count != active_rank:
        return q[:, :active_rank].t().detach().contiguous()
    if top_count < 0 or tail_count < 0 or top_count > active_rank:
        return q[:, :active_rank].t().detach().contiguous()

    tail_start = int(top_count)
    tail_available = max(0, int(q.shape[1]) - tail_start)
    if tail_count == 0:
        active = q[:, :top_count]
    elif tail_available < tail_count:
        active = q[:, :active_rank]
    else:
        seed = stable_int_seed(
            {
                "method": "agzo_oja_active_tail",
                "module": str(module_name),
                "step": int(step),
                "top_count": int(top_count),
                "tail_count": int(tail_count),
                "q_shape": [int(dim) for dim in q.shape],
                "noise_seed": noise_seed,
            }
        )
        generator = torch.Generator(device=q.device)
        generator.manual_seed(int(seed))
        tail_indices = (
            torch.randperm(tail_available, device=q.device, generator=generator)[
                :tail_count
            ]
            + tail_start
        )
        active = torch.cat([q[:, :top_count], q[:, tail_indices]], dim=1)
    return active.t().detach().contiguous()
