from __future__ import annotations

from typing import Any

from aimzo.config import AGZOConfig, OjaABQConfig, ZOConfig

from .agzo import HFAGZOMethod


class HFOjaABQMethod(HFAGZOMethod):
    name = "oja_abq"

    def _normalize_method_config(
        self,
        value: AGZOConfig | OjaABQConfig | dict[str, Any],
    ) -> AGZOConfig:
        raw = self.config.oja_abq
        if isinstance(raw, OjaABQConfig):
            config = raw
        elif isinstance(raw, AGZOConfig):
            config = OjaABQConfig(**dict(raw.__dict__))
        else:
            config = OjaABQConfig(**dict(raw))
        if config.perturbation_form == "abh_oja":
            return config
        return OjaABQConfig(
            **{
                **dict(config.__dict__),
                "perturbation_form": "abh_oja",
            }
        )


def oja_abq_config_to_agzo_dict(config: ZOConfig) -> dict[str, Any]:
    raw = config.oja_abq
    if isinstance(raw, OjaABQConfig):
        payload = dict(raw.__dict__)
    elif isinstance(raw, AGZOConfig):
        payload = dict(raw.__dict__)
    else:
        payload = dict(raw)
    payload["perturbation_form"] = "abh_oja"
    return payload
