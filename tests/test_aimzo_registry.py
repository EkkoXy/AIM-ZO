from aimzo.config import AIMZOConfig, ZOConfig
from aimzo.zo.methods.aimzo import HFAIMZOMethod
from aimzo.zo.methods.registry import build_hf_zo_method


def test_registry_builds_aimzo_method():
    config = ZOConfig(
        method="aimzo",
        eps=8e-5,
        learning_rate=1e-3,
        aimzo=AIMZOConfig(),
    )
    method = build_hf_zo_method(config)
    assert isinstance(method, HFAIMZOMethod)
    assert method.name == "aimzo"
