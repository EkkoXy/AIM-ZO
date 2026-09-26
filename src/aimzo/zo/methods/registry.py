from __future__ import annotations

from aimzo.config import ZOConfig

from .base import HFZOMethod
from .agzo import HFAGZOMethod
from .curvzo import HFCurvZOMethod
from .hizoo import HFHiZOOMethod
from .gradient_subspace import HFGradientSubspaceMethod
from .lowdim_muon import HFLowDimMuonMethod
from .lozo import HFLOZOMethod
from .mezo import HFMeZOMethod
from .aimzo import HFAIMZOMethod
from .oja_abq import HFOjaABQMethod


def build_hf_zo_method(config: ZOConfig) -> HFZOMethod:
    method = str(config.method).lower()
    if method == "mezo":
        return HFMeZOMethod(config)
    if method == "hizoo":
        return HFHiZOOMethod(config)
    if method == "agzo":
        return HFAGZOMethod(config)
    if method == "aimzo":
        return HFAIMZOMethod(config)
    if method == "zomuon":
        return HFLowDimMuonMethod(config, name="zomuon")
    if method == "zomopi":
        return HFLowDimMuonMethod(config, name="zomopi")
    if method == "curvzo":
        return HFCurvZOMethod(config)
    if method == "lozo":
        return HFLOZOMethod(config)
    if method == "oja_abq":
        return HFOjaABQMethod(config)
    if method == "svd0":
        return HFGradientSubspaceMethod(config, name="svd0")
    if method == "pgap":
        return HFGradientSubspaceMethod(config, name="pgap")
    raise ValueError(f"Unsupported HF ZO method: {config.method!r}")
