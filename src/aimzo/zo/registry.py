from __future__ import annotations

from typing import TYPE_CHECKING

from aimzo.config import ZOConfig

if TYPE_CHECKING:
    from aimzo.zo.estimator import MeZOTwoPointEstimator


def supported_estimators() -> tuple[str, ...]:
    return ("mezo_two_point",)


def build_estimator(config: ZOConfig) -> MeZOTwoPointEstimator:
    if config.estimator != "mezo_two_point":
        raise ValueError(
            f"unknown ZO estimator {config.estimator!r}; "
            f"expected one of: {', '.join(supported_estimators())}"
        )
    from aimzo.zo.config import ZOEstimatorConfig
    from aimzo.zo.estimator import MeZOTwoPointEstimator

    return MeZOTwoPointEstimator(
        ZOEstimatorConfig(
            eps=config.eps,
            learning_rate=config.learning_rate,
            seed=config.seed,
            weight_decay=config.weight_decay,
            parameter_scope=config.parameter_scope,
        )
    )
