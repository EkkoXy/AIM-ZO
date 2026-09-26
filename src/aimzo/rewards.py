from __future__ import annotations

import math
import re
from typing import Any


DEFAULT_FORMAT_REWARD = 0.1
_GSM8K_FINAL_ANSWER_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


def extract_gsm8k_final_answer(answer_text: str) -> str:
    match = _GSM8K_FINAL_ANSWER_RE.search(str(answer_text))
    if match is None:
        raise ValueError(f"Could not extract GSM8K final answer from: {answer_text!r}")
    return match.group(1).replace(",", "")


def normalize_answer(answer: str) -> str:
    normalized = str(answer).strip().replace(",", "").replace(" ", "")
    try:
        value = float(normalized)
    except (ValueError, OverflowError):
        return normalized.lower()
    if not math.isfinite(value):
        return normalized.lower()
    if value == int(value):
        return str(int(value))
    return str(value)


def _extract_last_braced_boxed(text: str) -> tuple[bool, str | None]:
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return False, None

    match = matches[-1]
    start_brace = match.end() - 1
    depth = 0
    content_start = start_brace + 1
    for index in range(start_brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                content = text[content_start:index].strip()
                return True, content or None
    return True, None


def extract_math_answer(text: str) -> str | None:
    has_boxed, boxed = _extract_last_braced_boxed(str(text))
    if has_boxed:
        return boxed
    matches = re.findall(r"-?[\d,]+(?:\.\d+)?", str(text))
    if matches:
        return matches[-1].replace(",", "")
    return None


def _has_boxed_answer(text: str) -> bool:
    has_boxed, _ = _extract_last_braced_boxed(str(text))
    return has_boxed


def score_training_answer(
    text: str,
    gold: str,
    *,
    format_reward: float = DEFAULT_FORMAT_REWARD,
) -> float:
    _, boxed = _extract_last_braced_boxed(str(text))
    if boxed is None:
        return 0.0
    return 1.0 if normalize_answer(boxed) == normalize_answer(gold) else float(format_reward)


def is_validation_correct(response: str, gold: str) -> bool:
    pred = extract_math_answer(response)
    if pred is None:
        return False
    return normalize_answer(pred) == normalize_answer(gold)


def grpo_reward_function(
    completions: list[Any],
    answer: list[Any],
    *,
    format_reward: float = DEFAULT_FORMAT_REWARD,
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for completion, gold in zip(completions, answer):
        if isinstance(completion, list) and completion and isinstance(completion[0], dict):
            text = str(completion[0].get("content", ""))
        else:
            text = str(completion)
        rewards.append(
            score_training_answer(text, str(gold), format_reward=float(format_reward))
        )
    return rewards
