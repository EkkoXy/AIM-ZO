from __future__ import annotations

import pytest

from aimzo.zo.methods.aimzo_population import centered_population_estimate


def test_centered_population_main_path_uses_rloo_sample_denominator():
    estimate = centered_population_estimate(
        [10, 20, 30],
        [1.0, 2.0, 4.0],
        center_loss=2.5,
        baseline_mode="population_mean",
        eps=1e-3,
        divide_by_eps=False,
        normalize_std=False,
        std_eps=1e-8,
        clip_std=0.0,
        trim_extremes=0,
    )

    mean = 7.0 / 3.0
    assert estimate.baseline == pytest.approx(mean)
    assert estimate.mean_minus_center == pytest.approx(mean - 2.5)
    assert [value for _, value in estimate.projected_grads] == pytest.approx(
        [(1.0 - mean) / 2.0, (2.0 - mean) / 2.0, (4.0 - mean) / 2.0]
    )
    assert sum(value for _, value in estimate.projected_grads) == pytest.approx(0.0)


def test_center_baseline_uses_population_size_denominator():
    estimate = centered_population_estimate(
        [1, 2],
        [3.0, 5.0],
        center_loss=2.0,
        baseline_mode="center",
        eps=0.5,
        divide_by_eps=True,
        normalize_std=False,
        std_eps=1e-8,
        clip_std=0.0,
        trim_extremes=0,
    )

    assert estimate.baseline == 2.0
    assert estimate.projected_grads == ((1, 1.0), (2, 3.0))
