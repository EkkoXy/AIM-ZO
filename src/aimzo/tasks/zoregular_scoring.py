from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ZORegularClassificationFeature:
    input_ids_by_candidate: list[list[int]]
    option_lens: list[int]
    label: int
    gold_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        gold_indices = self.gold_indices or (self.label,)
        object.__setattr__(
            self,
            "gold_indices",
            tuple(int(index) for index in gold_indices),
        )

    @property
    def num_options(self) -> int:
        return len(self.input_ids_by_candidate)


@dataclass(frozen=True)
class ZORegularGenerationFeature:
    input_ids: list[int]
    labels_start: int


class TinyTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        for token in str(text).split():
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab) + 1
            ids.append(self._vocab[token])
        return ids


def _correct_candidate_indices(
    candidates: list[Any],
    correct_candidate: Any,
) -> tuple[int, ...]:
    if isinstance(correct_candidate, list):
        matches = tuple(
            index for index, candidate in enumerate(candidates)
            if candidate in correct_candidate
        )
        return matches or (0,)

    if correct_candidate in candidates:
        return (candidates.index(correct_candidate),)
    if isinstance(correct_candidate, int) and 0 <= correct_candidate < len(candidates):
        return (int(correct_candidate),)
    return (0,)


def _tokenize(
    tokenizer: Any,
    text: str,
    *,
    source_style_tokenization: bool = False,
) -> list[int]:
    if source_style_tokenization:
        return list(tokenizer.encode(str(text).strip(" ")))
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def encode_zoregular_classification_sample(
    template: Any,
    sample: Any,
    tokenizer: Any,
    max_length: int = 2048,
    source_style_tokenization: bool = False,
) -> ZORegularClassificationFeature:
    candidates = list(sample.candidates or [])
    if not candidates:
        raise ValueError("ZORegular classification sample requires non-empty candidates")

    prompt_ids = _tokenize(
        tokenizer,
        template.encode(sample),
        source_style_tokenization=source_style_tokenization,
    )
    input_ids_by_candidate: list[list[int]] = []
    option_lens: list[int] = []
    for candidate in candidates:
        full_ids = _tokenize(
            tokenizer,
            template.verbalize(sample, candidate),
            source_style_tokenization=source_style_tokenization,
        )
        raw_option_len = max(1, len(full_ids) - len(prompt_ids))
        if len(full_ids) > max_length:
            full_ids = full_ids[-max_length:]
        option_lens.append(min(raw_option_len, len(full_ids)))
        input_ids_by_candidate.append(full_ids)

    gold_indices = _correct_candidate_indices(candidates, sample.correct_candidate)
    return ZORegularClassificationFeature(
        input_ids_by_candidate=input_ids_by_candidate,
        option_lens=option_lens,
        label=gold_indices[0],
        gold_indices=gold_indices,
    )


def _generation_answer(correct_candidate: Any) -> Any:
    if isinstance(correct_candidate, list):
        if not correct_candidate:
            raise ValueError("ZORegular generation sample requires non-empty answers")
        return correct_candidate[0]
    return correct_candidate


def encode_zoregular_generation_sample(
    template: Any,
    sample: Any,
    tokenizer: Any,
    max_length: int = 1024,
) -> ZORegularGenerationFeature:
    answer = _generation_answer(sample.correct_candidate)
    prompt_ids = _tokenize(tokenizer, template.encode(sample))
    full_ids = _tokenize(tokenizer, template.verbalize(sample, answer))
    prompt_len = len(prompt_ids)
    full_len = len(full_ids)
    if full_len > max_length:
        overflow = full_len - max_length
        full_ids = full_ids[overflow:]
        adjusted_prompt_len = max(0, prompt_len - overflow)
    else:
        adjusted_prompt_len = prompt_len
    labels_start = max(0, min(adjusted_prompt_len, len(full_ids) - 1))
    return ZORegularGenerationFeature(input_ids=full_ids, labels_start=labels_start)


def _padded_length(length: int, pad_to_multiple_of: int) -> int:
    if pad_to_multiple_of <= 1:
        return length
    return ((length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of


def collate_zoregular_classification(
    features: list[ZORegularClassificationFeature],
    pad_token_id: int,
    pad_to_multiple_of: int = 8,
    padding_side: str = "right",
    source_style_labels: bool = False,
) -> dict[str, torch.Tensor]:
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")
    rows = [ids for feature in features for ids in feature.input_ids_by_candidate]
    max_row_len = max((len(row) for row in rows), default=0)
    padded_len = _padded_length(max_row_len, pad_to_multiple_of)

    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for row in rows:
        pad_len = padded_len - len(row)
        pad_ids = [int(pad_token_id)] * pad_len
        pad_mask = [0] * pad_len
        if padding_side == "left":
            input_rows.append(pad_ids + list(row))
            mask_rows.append(pad_mask + [1] * len(row))
        else:
            input_rows.append(list(row) + pad_ids)
            mask_rows.append([1] * len(row) + pad_mask)

    if source_style_labels:
        num_options = torch.tensor(
            [
                feature.num_options
                for feature in features
                for _ in range(feature.num_options)
            ],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [
                feature.label
                for feature in features
                for _ in range(feature.num_options)
            ],
            dtype=torch.long,
        )
    else:
        num_options = torch.tensor(
            [feature.num_options for feature in features],
            dtype=torch.long,
        )
        labels = torch.tensor([feature.label for feature in features], dtype=torch.long)

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        "option_len": torch.tensor(
            [length for feature in features for length in feature.option_lens],
            dtype=torch.long,
        ),
        "num_options": num_options,
        "labels": labels,
        "gold_mask": torch.tensor(
            [
                candidate_index in feature.gold_indices
                for feature in features
                for candidate_index in range(feature.num_options)
            ],
            dtype=torch.bool,
        ),
    }


def collate_zoregular_generation(
    features: list[ZORegularGenerationFeature],
    pad_token_id: int,
    pad_to_multiple_of: int = 8,
) -> dict[str, torch.Tensor]:
    max_row_len = max((len(feature.input_ids) for feature in features), default=0)
    padded_len = _padded_length(max_row_len, pad_to_multiple_of)

    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for feature in features:
        input_ids = list(feature.input_ids)
        pad_len = padded_len - len(input_ids)
        input_rows.append(input_ids + [int(pad_token_id)] * pad_len)
        mask_rows.append([1] * len(input_ids) + [0] * pad_len)
        label_rows.append(
            [-100] * feature.labels_start
            + input_ids[feature.labels_start:]
            + [-100] * pad_len
        )

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }


def _group_candidate_scores(
    candidate_scores: torch.Tensor,
    labels: torch.Tensor,
    num_options: torch.Tensor,
    gold_mask: torch.Tensor | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    offset = 0
    for label, count in zip(labels.tolist(), num_options.tolist(), strict=True):
        next_offset = offset + int(count)
        group_gold_mask = (
            None if gold_mask is None else gold_mask[offset:next_offset]
        )
        groups.append(
            (
                candidate_scores[offset:next_offset],
                torch.tensor([label], device=candidate_scores.device),
                group_gold_mask,
            )
        )
        offset = next_offset
    if offset != int(candidate_scores.numel()):
        raise ValueError("candidate_scores length must equal sum(num_options)")
    if gold_mask is not None and offset != int(gold_mask.numel()):
        raise ValueError("gold_mask length must equal sum(num_options)")
    return groups


def zoregular_classification_loss(
    candidate_scores: torch.Tensor,
    labels: torch.Tensor,
    num_options: torch.Tensor,
    gold_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    source_expanded_loss = _source_expanded_fixed_option_cross_entropy(
        candidate_scores,
        labels,
        num_options,
        gold_mask,
    )
    if source_expanded_loss is not None:
        return source_expanded_loss

    source_style_loss = _single_gold_fixed_option_cross_entropy(
        candidate_scores,
        labels,
        num_options,
        gold_mask,
    )
    if source_style_loss is not None:
        return source_style_loss

    losses = []
    for scores, label, group_gold_mask in _group_candidate_scores(
        candidate_scores,
        labels,
        num_options,
        gold_mask,
    ):
        if group_gold_mask is None:
            losses.append(F.cross_entropy(scores.unsqueeze(0), label))
            continue
        gold_scores = scores[group_gold_mask.to(device=scores.device, dtype=torch.bool)]
        if gold_scores.numel() == 0:
            losses.append(F.cross_entropy(scores.unsqueeze(0), label))
        else:
            losses.append(
                torch.logsumexp(scores, dim=0) - torch.logsumexp(gold_scores, dim=0)
            )
    if not losses:
        return candidate_scores.sum() * 0.0
    return torch.stack(losses).mean()


def _source_expanded_fixed_option_cross_entropy(
    candidate_scores: torch.Tensor,
    labels: torch.Tensor,
    num_options: torch.Tensor,
    gold_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if num_options.numel() == 0:
        return None
    option_count = int(num_options[0].item())
    if option_count <= 0:
        return None
    if int(candidate_scores.numel()) != int(num_options.numel()):
        return None
    if candidate_scores.numel() % option_count != 0:
        return None
    if not torch.all(num_options == option_count).item():
        return None

    labels_for_loss = labels.to(device=candidate_scores.device, dtype=torch.long)
    labels_for_loss = labels_for_loss.reshape(-1, option_count)[:, 0]
    if gold_mask is not None:
        reshaped_gold = gold_mask.to(device=candidate_scores.device, dtype=torch.bool)
        reshaped_gold = reshaped_gold.reshape(-1, option_count)
        if not torch.all(reshaped_gold.sum(dim=1) == 1).item():
            return None
        gold_labels = reshaped_gold.to(torch.long).argmax(dim=1)
        if not torch.equal(gold_labels, labels_for_loss):
            return None

    return F.cross_entropy(
        candidate_scores.reshape(-1, option_count),
        labels_for_loss,
    )


def _single_gold_fixed_option_cross_entropy(
    candidate_scores: torch.Tensor,
    labels: torch.Tensor,
    num_options: torch.Tensor,
    gold_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if num_options.numel() == 0:
        return None
    option_count = int(num_options[0].item())
    if option_count <= 0:
        return None
    if not torch.all(num_options == option_count).item():
        return None
    if int(candidate_scores.numel()) != int(num_options.numel()) * option_count:
        return None

    labels_for_loss = labels.to(device=candidate_scores.device, dtype=torch.long)
    if gold_mask is not None:
        reshaped_gold = gold_mask.to(device=candidate_scores.device, dtype=torch.bool)
        reshaped_gold = reshaped_gold.reshape(-1, option_count)
        if not torch.all(reshaped_gold.sum(dim=1) == 1).item():
            return None
        gold_labels = reshaped_gold.to(torch.long).argmax(dim=1)
        if not torch.equal(gold_labels, labels_for_loss):
            return None

    return F.cross_entropy(
        candidate_scores.reshape(-1, option_count),
        labels_for_loss,
    )


def zoregular_generation_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    shifted_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shifted_logits, shifted_labels, ignore_index=-100)


def classify_from_candidate_scores(
    candidate_scores: torch.Tensor,
    labels: torch.Tensor,
    num_options: torch.Tensor,
    gold_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    source_expanded = _classify_source_expanded_fixed_options(
        candidate_scores,
        labels,
        num_options,
        gold_mask,
    )
    if source_expanded is not None:
        return source_expanded

    predictions: list[int] = []
    gold_labels: list[int] = []
    correct = 0
    for scores, label, group_gold_mask in _group_candidate_scores(
        candidate_scores,
        labels,
        num_options,
        gold_mask,
    ):
        prediction = int(torch.argmax(scores).item())
        predictions.append(prediction)
        gold_labels.append(int(label.item()))
        if group_gold_mask is None:
            correct += int(prediction == int(label.item()))
        else:
            correct += int(bool(group_gold_mask[prediction].item()))

    accuracy = correct / len(gold_labels) if gold_labels else 0.0
    return {"accuracy": accuracy, "predictions": predictions, "labels": gold_labels}


def _classify_source_expanded_fixed_options(
    candidate_scores: torch.Tensor,
    labels: torch.Tensor,
    num_options: torch.Tensor,
    gold_mask: torch.Tensor | None,
) -> dict[str, Any] | None:
    if num_options.numel() == 0:
        return None
    option_count = int(num_options[0].item())
    if option_count <= 0:
        return None
    if int(candidate_scores.numel()) != int(num_options.numel()):
        return None
    if candidate_scores.numel() % option_count != 0:
        return None
    if not torch.all(num_options == option_count).item():
        return None

    grouped_scores = candidate_scores.reshape(-1, option_count)
    predictions = [
        int(value)
        for value in torch.argmax(grouped_scores, dim=1).detach().cpu().tolist()
    ]
    labels_for_metric = labels.to(device=candidate_scores.device, dtype=torch.long)
    labels_for_metric = labels_for_metric.reshape(-1, option_count)[:, 0]
    gold_labels = [int(value) for value in labels_for_metric.detach().cpu().tolist()]
    if gold_mask is None:
        correct = sum(
            int(prediction == label)
            for prediction, label in zip(predictions, gold_labels, strict=True)
        )
    else:
        grouped_gold = gold_mask.to(device=candidate_scores.device, dtype=torch.bool)
        grouped_gold = grouped_gold.reshape(-1, option_count)
        correct = sum(
            int(bool(row[prediction].item()))
            for prediction, row in zip(predictions, grouped_gold, strict=True)
        )
    accuracy = correct / len(gold_labels) if gold_labels else 0.0
    return {"accuracy": accuracy, "predictions": predictions, "labels": gold_labels}
