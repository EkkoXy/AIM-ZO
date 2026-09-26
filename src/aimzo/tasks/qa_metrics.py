from __future__ import annotations

import collections
import re
import string


def normalize_qa_answer(text: str) -> str:
    value = str(text).lower()
    value = "".join(
        character for character in value if character not in string.punctuation
    )
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def qa_exact_match(prediction: str, answers: list[str]) -> float:
    normalized = normalize_qa_answer(prediction)
    return float(any(normalized == normalize_qa_answer(answer) for answer in answers))


def qa_f1(prediction: str, answers: list[str]) -> float:
    prediction_tokens = normalize_qa_answer(prediction).split()
    scores: list[float] = []
    for answer in answers:
        answer_tokens = normalize_qa_answer(answer).split()
        common = collections.Counter(prediction_tokens) & collections.Counter(
            answer_tokens
        )
        same = sum(common.values())
        if not prediction_tokens or not answer_tokens:
            scores.append(float(prediction_tokens == answer_tokens))
        elif same == 0:
            scores.append(0.0)
        else:
            precision = same / len(prediction_tokens)
            recall = same / len(answer_tokens)
            scores.append(2.0 * precision * recall / (precision + recall))
    return max(scores) if scores else 0.0


def postprocess_qa_prediction(text: str) -> str:
    value = str(text).strip()
    for marker in (
        "\nTitle:",
        "\nContext:",
        "\nQuestion:",
        "\nAnswer:",
        "Title:",
        "Context:",
        "Question:",
    ):
        if marker in value:
            value = value.split(marker, 1)[0]
    value = value.splitlines()[0] if value.splitlines() else value
    value = re.sub(r"^\s*Answer:\s*", "", value).strip()
    return value.strip(" \t\n\r\"'")


def multirc_grouped_metrics(rows, predictions):
    """SuperGLUE answer-level positive F1 and question-level exact match."""
    if len(rows) != len(predictions):
        raise ValueError("MultiRC prediction count must match rows")
    groups = {}
    tp = fp = fn = 0
    for row, prediction in zip(rows, predictions, strict=True):
        gold = int(row.correct_candidate)
        prediction = int(row.candidates[prediction])
        idx = row.data["idx"]
        key = (int(idx["paragraph"]), int(idx["question"]))
        groups[key] = groups.get(key, True) and prediction == gold
        tp += int(prediction == gold == 1)
        fp += int(prediction == 1 and gold == 0)
        fn += int(prediction == 0 and gold == 1)
    denominator = 2 * tp + fp + fn
    return {
        "f1a": 2 * tp / denominator if denominator else 0.0,
        "em": sum(groups.values()) / len(groups) if groups else 0.0,
        "num_questions": len(groups),
    }
