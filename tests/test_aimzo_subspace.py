from __future__ import annotations

import torch

from aimzo.zo.methods.aimzo_subspace import (
    initialize_oja_basis,
    oja_covariance_action,
    select_top_tail_basis,
)


def test_oja_covariance_action_matches_explicit_gram_matrix():
    torch.manual_seed(3)
    tokens = torch.randn(7, 5)
    q, _ = torch.linalg.qr(torch.randn(5, 3), mode="reduced")

    actual = oja_covariance_action(tokens, q)
    expected = tokens.t().matmul(tokens).div(7.0).matmul(q)

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_initialize_oja_basis_is_reproducible_and_orthonormal():
    kwargs = {
        "module_name": "layer.proj",
        "feature_dim": 8,
        "right_rank": 4,
        "device": torch.device("cpu"),
        "seed": 142,
    }
    first = initialize_oja_basis(**kwargs)
    second = initialize_oja_basis(**kwargs)

    assert torch.equal(first, second)
    assert torch.allclose(first.t() @ first, torch.eye(4), atol=1e-6)


def test_select_top_tail_basis_preserves_head_and_is_reproducible():
    q = torch.eye(8)

    first = select_top_tail_basis(
        q,
        module_name="layer.proj",
        step=7,
        noise_seed=142,
        active_rank=4,
        top_count=2,
        tail_count=2,
    )
    second = select_top_tail_basis(
        q,
        module_name="layer.proj",
        step=7,
        noise_seed=142,
        active_rank=4,
        top_count=2,
        tail_count=2,
    )

    assert torch.equal(first, second)
    assert torch.equal(first[:2], q[:, :2].t())
    assert first.shape == (4, 8)
    assert torch.equal(first @ first.t(), torch.eye(4))


def test_select_top_tail_basis_falls_back_to_leading_basis_for_bad_counts():
    q = torch.eye(6)

    selected = select_top_tail_basis(
        q,
        module_name="layer.proj",
        step=1,
        noise_seed=None,
        active_rank=4,
        top_count=3,
        tail_count=2,
    )

    assert torch.equal(selected, q[:, :4].t())
