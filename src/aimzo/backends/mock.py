from __future__ import annotations

from pathlib import Path

from aimzo.backends.base import TokenLogprobBatch, validate_generation_args


def _build_mock_lora_model(parameter_scope: str = "full_parameters"):
    import torch

    parameter_scope = str(parameter_scope)
    if parameter_scope not in {"lora", "full_parameters"}:
        raise ValueError("parameter_scope must be one of: full_parameters, lora")

    class _MockLoRAModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.tensor([[1.0]]))
            self.base_weight = torch.nn.Parameter(
                torch.tensor([[0.0]]),
                requires_grad=parameter_scope == "full_parameters",
            )

        def forward(self, values):
            return values @ self.lora_weight + self.base_weight

    return _MockLoRAModel()


class MockBackend:
    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        parameter_scope: str = "full_parameters",
    ):
        self.parameter_scope = str(parameter_scope)
        if self.parameter_scope not in {"lora", "full_parameters"}:
            raise ValueError("parameter_scope must be one of: full_parameters, lora")
        self.responses = responses or [
            "The answer is \\boxed{1}.",
            "This is a longer wrong answer: \\boxed{0}.",
        ]
        self._model = None
        self.tokenizer = None

    @property
    def model(self):
        if self._model is None:
            self._model = _build_mock_lora_model(self.parameter_scope)
        return self._model

    @model.setter
    def model(self, value) -> None:
        self._model = value

    def generate(
        self,
        prompts: list[str],
        *,
        num_samples: int,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[list[str]]:
        validate_generation_args(
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        result: list[list[str]] = []
        for _ in prompts:
            group = [
                self.responses[index % len(self.responses)] for index in range(num_samples)
            ]
            result.append(group)
        return result

    def logprobs(self, prompts: list[str], responses: list[str]) -> list[float]:
        if len(prompts) != len(responses):
            raise ValueError(
                "prompts and responses must have the same length; "
                f"got prompts={len(prompts)}, responses={len(responses)}"
            )
        weight = self.model.lora_weight.detach().reshape(-1)[0]
        import torch.nn.functional as F

        scale = float(F.softplus(weight).item()) + 1e-8
        return [-float(len(response)) * scale for response in responses]

    def token_logprobs(
        self,
        prompts: list[str],
        responses: list[str],
        *,
        adapter_path: str | None = None,
    ) -> TokenLogprobBatch:
        if len(prompts) != len(responses):
            raise ValueError(
                "prompts and responses must have the same length; "
                f"got prompts={len(prompts)}, responses={len(responses)}"
            )
        rows: TokenLogprobBatch = []
        for response in responses:
            token_count = max(1, len(str(response).split()))
            rows.append([-1.0 for _ in range(token_count)])
        return rows

    def save_checkpoint(self, output_dir: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
