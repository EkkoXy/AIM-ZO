from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import shutil

import torch
import torch.nn.functional as F

from aimzo.backends.base import TokenLogprobBatch, validate_generation_args


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_torch_dtype(dtype: str, device: torch.device) -> torch.dtype:
    try:
        torch_dtype = _DTYPE_MAP[dtype.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(_DTYPE_MAP))
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of: {supported}") from exc
    if device.type == "cpu" and torch_dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return torch_dtype


def _patch_transformers_remote_code_compat() -> None:
    import typing

    import transformers.utils as transformers_utils

    if hasattr(transformers_utils, "LossKwargs"):
        return

    class LossKwargs(typing.TypedDict, total=False):
        labels: torch.Tensor
        num_items_in_batch: torch.Tensor

    transformers_utils.LossKwargs = LossKwargs


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _fallback_token_id(tokenizer: Any) -> int:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    if pad_id is not None:
        return int(pad_id)
    raise ValueError("tokenizer must define eos_token_id or pad_token_id")


def _source_compat_should_patch_opt_bos(
    *,
    model_name: str,
    source_compat_tokenizer_profile: str | None,
) -> bool:
    if source_compat_tokenizer_profile in {"curvzo_source", "mezo_source"}:
        return False
    if source_compat_tokenizer_profile in {
        "hizoo_source",
        "zomuon_source",
        "zomopi_source",
    }:
        return "opt" in str(model_name)
    return "opt" in str(model_name).lower()


def _source_compat_should_patch_pad_token_id(
    *,
    source_compat_tokenizer_profile: str | None,
) -> bool:
    return source_compat_tokenizer_profile not in {"curvzo_source", "mezo_source"}


@contextmanager
def _temporary_eval(model: Any) -> Iterator[None]:
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        yield
    finally:
        if was_training and hasattr(model, "train"):
            model.train()


@contextmanager
def _saveable_generation_config(model: Any) -> Iterator[None]:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None or bool(getattr(generation_config, "do_sample", False)):
        yield
        return

    replacements = {"top_p": 1.0, "top_k": 50}
    old_values: dict[str, Any] = {}
    for key, value in replacements.items():
        if hasattr(generation_config, key):
            old_values[key] = getattr(generation_config, key)
            setattr(generation_config, key, value)
    try:
        yield
    finally:
        for key, value in old_values.items():
            setattr(generation_config, key, value)


@contextmanager
def _ignore_shutil_permission_metadata_errors() -> Iterator[None]:
    original_copymode = shutil.copymode
    original_copystat = shutil.copystat

    def copymode(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        try:
            original_copymode(src, dst, *args, **kwargs)
        except PermissionError:
            return

    def copystat(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        try:
            original_copystat(src, dst, *args, **kwargs)
        except PermissionError:
            return

    shutil.copymode = copymode
    shutil.copystat = copystat
    try:
        yield
    finally:
        shutil.copymode = original_copymode
        shutil.copystat = original_copystat


def sequence_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    responses: list[str],
    device: torch.device,
) -> list[float]:
    if len(prompts) != len(responses):
        raise ValueError(
            "prompts and responses must have the same length; "
            f"got prompts={len(prompts)}, responses={len(responses)}"
        )
    if not prompts:
        return []

    eos_id = _fallback_token_id(tokenizer)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id)

    encoded: list[list[int]] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    for prompt, response in zip(prompts, responses, strict=True):
        prompt_ids = _encode(tokenizer, prompt)
        if not prompt_ids:
            raise ValueError("empty prompts are not supported for sequence logprobs")

        response_ids = _encode(tokenizer, response)
        if not response_ids:
            response_ids = [eos_id]

        encoded.append(prompt_ids + response_ids)
        prompt_lengths.append(len(prompt_ids))
        response_lengths.append(len(response_ids))

    max_len = max(len(ids) for ids in encoded)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for ids in encoded:
        pad = max_len - len(ids)
        input_rows.append(ids + [pad_id] * pad)
        mask_rows.append([1] * len(ids) + [0] * pad)

    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
    with _temporary_eval(model), torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    values: list[float] = []
    for index, _ in enumerate(encoded):
        prompt_end = prompt_lengths[index]
        response_end = prompt_end + response_lengths[index]
        shift_logits = logits[index, prompt_end - 1 : response_end - 1]
        target_ids = input_ids[index, prompt_end:response_end]
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        values.append(float(token_log_probs.sum().item()))
    return values


def token_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    responses: list[str],
    device: torch.device,
) -> TokenLogprobBatch:
    if len(prompts) != len(responses):
        raise ValueError(
            "prompts and responses must have the same length; "
            f"got prompts={len(prompts)}, responses={len(responses)}"
        )
    if not prompts:
        return []

    eos_id = _fallback_token_id(tokenizer)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id)

    encoded: list[list[int]] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    for prompt, response in zip(prompts, responses, strict=True):
        prompt_ids = _encode(tokenizer, prompt)
        if not prompt_ids:
            raise ValueError("empty prompts are not supported for token logprobs")
        response_ids = _encode(tokenizer, response)
        if not response_ids:
            response_ids = [eos_id]
        encoded.append(prompt_ids + response_ids)
        prompt_lengths.append(len(prompt_ids))
        response_lengths.append(len(response_ids))

    max_len = max(len(ids) for ids in encoded)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for ids in encoded:
        pad = max_len - len(ids)
        input_rows.append(ids + [pad_id] * pad)
        mask_rows.append([1] * len(ids) + [0] * pad)

    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
    with _temporary_eval(model), torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    rows: TokenLogprobBatch = []
    for index, _ in enumerate(encoded):
        prompt_end = prompt_lengths[index]
        response_end = prompt_end + response_lengths[index]
        shift_logits = logits[index, prompt_end - 1 : response_end - 1]
        target_ids = input_ids[index, prompt_end:response_end]
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_values = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        rows.append([float(value) for value in token_values.detach().cpu().tolist()])
    return rows


class HFBackend:
    def __init__(
        self,
        *,
        model_name: str,
        dtype: str = "float32",
        lora_rank: int = 2,
        lora_alpha: int = 4,
        lora_target_modules: tuple[str, ...] = ("c_attn",),
        parameter_scope: str = "full_parameters",
        device: str | None = None,
        trust_remote_code: bool = False,
        source_compat_tokenizer: bool = False,
        source_compat_tokenizer_profile: str | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        parameter_scope = str(parameter_scope)
        if parameter_scope not in {"lora", "full_parameters"}:
            raise ValueError("parameter_scope must be one of: full_parameters, lora")

        self.dtype = str(dtype)
        self.parameter_scope = parameter_scope
        self.trust_remote_code = bool(trust_remote_code)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch_dtype = resolve_torch_dtype(dtype, self.device)
        self.torch_dtype = torch_dtype

        if self.trust_remote_code:
            _patch_transformers_remote_code_compat()

        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(trust_remote_code),
        }
        if source_compat_tokenizer:
            tokenizer_kwargs["use_fast"] = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        if source_compat_tokenizer:
            if _source_compat_should_patch_pad_token_id(
                source_compat_tokenizer_profile=source_compat_tokenizer_profile,
            ):
                self.tokenizer.pad_token_id = 0
            if _source_compat_should_patch_opt_bos(
                model_name=model_name,
                source_compat_tokenizer_profile=source_compat_tokenizer_profile,
            ):
                self.tokenizer.bos_token_id = 0
        added_pad_token = False
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                added_pad_token = True

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )
        if added_pad_token:
            base_model.resize_token_embeddings(len(self.tokenizer))

        if parameter_scope == "full_parameters":
            for parameter in base_model.parameters():
                if torch.is_floating_point(parameter):
                    parameter.requires_grad_(True)
            self.model = base_model.to(self.device)
        else:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                r=int(lora_rank),
                lora_alpha=int(lora_alpha),
                target_modules=list(lora_target_modules),
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(base_model, lora_config).to(self.device)
        self.model.eval()

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

        all_outputs: list[list[str]] = []
        with _temporary_eval(self.model):
            for prompt in prompts:
                encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                do_sample = temperature > 0
                num_return_sequences = int(num_samples) if do_sample else 1
                generation_kwargs: dict[str, Any] = {
                    **encoded,
                    "do_sample": do_sample,
                    "max_new_tokens": int(max_new_tokens),
                    "num_return_sequences": num_return_sequences,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                if do_sample:
                    generation_kwargs["temperature"] = float(temperature)
                    generation_kwargs["top_p"] = float(top_p)

                output_ids = self.model.generate(
                    **generation_kwargs,
                )
                prompt_len = encoded["input_ids"].shape[1]
                group = [
                    self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
                    for row in output_ids
                ]
                if not do_sample:
                    group = group * int(num_samples)
                all_outputs.append(group)
        return all_outputs

    def logprobs(self, prompts: list[str], responses: list[str]) -> list[float]:
        return sequence_logprobs(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            responses=responses,
            device=self.device,
        )

    def token_logprobs(
        self,
        prompts: list[str],
        responses: list[str],
        *,
        adapter_path: str | None = None,
    ) -> TokenLogprobBatch:
        del adapter_path
        return token_logprobs(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            responses=responses,
            device=self.device,
        )

    def save_checkpoint(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        with _ignore_shutil_permission_metadata_errors(), _saveable_generation_config(
            self.model
        ):
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

    def load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(checkpoint_dir).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {path}")
        if self.parameter_scope != "full_parameters":
            raise ValueError("HFBackend.load_checkpoint supports full_parameters only")
        if (path / "model.safetensors.index.json").is_file() or (
            path / "pytorch_model.bin.index.json"
        ).is_file():
            # Load into the existing model one shard at a time. Constructing a
            # second model doubles peak GPU memory and OOMs OPT-13B on 48GB.
            from transformers.modeling_utils import load_sharded_checkpoint

            incompatible = load_sharded_checkpoint(
                self.model,
                path,
                strict=False,
                prefer_safe=True,
            )
            missing = set(incompatible.missing_keys)
            unexpected = set(incompatible.unexpected_keys)
            allowed_missing = {"lm_head.weight"}
            if unexpected or not missing.issubset(allowed_missing):
                raise RuntimeError(
                    "incomplete sharded checkpoint load: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            if missing and hasattr(self.model, "tie_weights"):
                self.model.tie_weights()
            return
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=self.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )
        for parameter in model.parameters():
            if torch.is_floating_point(parameter):
                parameter.requires_grad_(True)
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
