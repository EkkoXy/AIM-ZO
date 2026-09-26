from __future__ import annotations

from typing import Any

from aimzo.tasks.zoregular import ZORegularSample


def build_zoregular_classification_payload(
    *,
    task_name: str,
    template: Any,
    samples: list[ZORegularSample],
    split: str,
) -> dict[str, Any]:
    prompts: list[str] = []
    candidate_responses: list[list[str]] = []
    labels: list[int] = []
    gold_indices: list[list[int]] = []
    num_options: list[int] = []

    for index, sample in enumerate(samples):
        candidates = list(sample.candidates or [])
        if not candidates:
            raise ValueError(
                f"{task_name} classification sample {index} requires candidates"
            )
        prompt = str(template.encode(sample))
        responses = [
            _suffix(
                prompt=prompt,
                full_text=str(template.verbalize(sample, candidate)),
                task_name=task_name,
                sample_index=index,
            )
            for candidate in candidates
        ]
        gold = _gold_indices(candidates, sample.correct_candidate)
        prompts.append(prompt)
        candidate_responses.append(responses)
        labels.append(gold[0])
        gold_indices.append(gold)
        num_options.append(len(candidates))

    return {
        "objective_kind": "classification",
        "task_name": str(task_name),
        "prompts": prompts,
        "candidate_responses": candidate_responses,
        "labels": labels,
        "gold_indices": gold_indices,
        "num_options": num_options,
        "metadata": {"split": str(split)},
    }


def build_zoregular_generation_payload(
    *,
    task_name: str,
    template: Any,
    samples: list[ZORegularSample],
    split: str,
) -> dict[str, Any]:
    prompts: list[str] = []
    responses: list[str] = []
    for index, sample in enumerate(samples):
        answer = _first_answer(sample.correct_candidate, task_name, index)
        prompt = str(template.encode(sample))
        full_text = str(template.verbalize(sample, answer))
        prompts.append(prompt)
        responses.append(
            _suffix(
                prompt=prompt,
                full_text=full_text,
                task_name=task_name,
                sample_index=index,
            )
        )
    return {
        "objective_kind": "generation",
        "task_name": str(task_name),
        "prompts": prompts,
        "responses": responses,
        "answer_policy": "first_gold_answer",
        "metadata": {"split": str(split)},
    }


def _suffix(
    *,
    prompt: str,
    full_text: str,
    task_name: str,
    sample_index: int,
) -> str:
    if not full_text.startswith(prompt):
        raise ValueError(
            f"{task_name} sample {sample_index} verbalized text does not start "
            "with prompt"
        )
    suffix = full_text[len(prompt) :]
    if suffix == "" or suffix.strip() == "":
        raise ValueError(
            f"{task_name} sample {sample_index} has empty response suffix"
        )
    return suffix


def _gold_indices(candidates: list[Any], correct_candidate: Any) -> list[int]:
    if isinstance(correct_candidate, list):
        matches = [
            index
            for index, candidate in enumerate(candidates)
            if candidate in correct_candidate
        ]
        return matches or [0]
    if correct_candidate in candidates:
        return [candidates.index(correct_candidate)]
    if isinstance(correct_candidate, int) and 0 <= correct_candidate < len(candidates):
        return [int(correct_candidate)]
    return [0]


def _first_answer(candidate: Any, task_name: str, sample_index: int) -> Any:
    if isinstance(candidate, list):
        if not candidate:
            raise ValueError(f"{task_name} generation sample {sample_index} has no answer")
        candidate = candidate[0]
    if candidate is None or str(candidate) == "":
        raise ValueError(f"{task_name} generation sample {sample_index} has no answer")
    return candidate
