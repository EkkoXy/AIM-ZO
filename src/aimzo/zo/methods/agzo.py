from __future__ import annotations

from .aimzo import (
    HFAIMZOMethod,
    _ActivationCapture,
    _safe_tensor_norm,
    _source_projected_grad,
    _stable_int_seed,
    _two_side_residual_projected_grads,
)
from .aimzo_population import (
    population_sample_denominator as _loren_population_sample_denom,
)


class HFAGZOMethod(HFAIMZOMethod):
    """Compatibility entry point for legacy experiment configurations."""

    name = "agzo"


__all__ = ["HFAGZOMethod"]
