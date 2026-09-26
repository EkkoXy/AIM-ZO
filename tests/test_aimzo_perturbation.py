from __future__ import annotations

import pytest
import torch

from aimzo.zo.methods.aimzo_perturbation import (
    low_rank_frobenius_norm,
    perturbation_normalization_scale,
    sample_probe_a,
    sample_probe_b,
)


def test_low_rank_frobenius_norm_matches_materialized_perturbation():
    torch.manual_seed(11)
    a_matrix = torch.randn(7, 2)
    b_matrix = torch.randn(2, 4)
    q_active, _ = torch.linalg.qr(torch.randn(9, 4), mode="reduced")
    q_active = q_active.t().contiguous()

    actual = low_rank_frobenius_norm(a_matrix, b_matrix, q_active)
    expected = torch.linalg.vector_norm(a_matrix @ b_matrix @ q_active).item()

    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_layer_fro_scale_targets_sqrt_parameter_numel():
    a_matrix = torch.tensor([[1.0], [2.0]])
    b_matrix = torch.tensor([[3.0, 4.0]])
    q_active = torch.eye(2)

    scale, pre_norm, target = perturbation_normalization_scale(
        normalization="layer_fro",
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        q_active=q_active,
        parameter_numel=20,
        layer_fro_scale=1.0,
    )

    dense = a_matrix @ b_matrix @ q_active
    assert pre_norm == pytest.approx(torch.linalg.vector_norm(dense).item())
    assert torch.linalg.vector_norm(dense * scale).item() == pytest.approx(target)
    assert target == pytest.approx(20**0.5)


def test_probe_factors_are_reproducible_and_shape_checked():
    kwargs = {
        "name": "layer.proj.weight",
        "probe_seed": 142,
        "rank": 2,
        "device": torch.device("cpu"),
        "dtype": torch.float64,
    }
    a_first = sample_probe_a(out_dim=5, orthogonal=False, **kwargs)
    a_second = sample_probe_a(out_dim=5, orthogonal=False, **kwargs)
    b_first = sample_probe_b(block=3, span_rank=4, **kwargs)
    b_second = sample_probe_b(block=3, span_rank=4, **kwargs)

    assert torch.equal(a_first, a_second)
    assert torch.equal(b_first, b_second)
    assert a_first.shape == (5, 2)
    assert b_first.shape == (2, 4)
