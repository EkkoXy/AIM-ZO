from __future__ import annotations

import math
from typing import Protocol, TypeAlias


TokenLogprobBatch: TypeAlias = list[list[float]]


def validate_generation_args(
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    if not isinstance(num_samples, int) or isinstance(num_samples, bool):
        raise ValueError("num_samples must be an int")
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise ValueError("max_new_tokens must be an int")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if not math.isfinite(float(temperature)) or temperature < 0:
        raise ValueError("temperature must be finite and >= 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must satisfy 0 < top_p <= 1")


class Backend(Protocol):
    model: object
    tokenizer: object

    def generate(
        self,
        prompts: list[str],
        *,
        num_samples: int,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[list[str]]:
        raise NotImplementedError

    def logprobs(self, prompts: list[str], responses: list[str]) -> list[float]:
        raise NotImplementedError

    def token_logprobs(
        self,
        prompts: list[str],
        responses: list[str],
        *,
        adapter_path: str | None = None,
    ) -> TokenLogprobBatch:
        raise NotImplementedError

    def save_checkpoint(self, output_dir: str) -> None:
        raise NotImplementedError
