from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

DEFAULT_FORMAT_REWARD = 0.1


def validation_benchmark(benchmark: str) -> str:
    normalized = str(benchmark).strip().lower()
    aliases = {
        "math": "math500",
        "math-500": "math500",
        "math_500": "math500",
        "deepscaler": "math500",
        "gsm8k": "gsm8k",
    }
    mapped = aliases.get(normalized, normalized)
    if mapped not in {"gsm8k", "math500"}:
        raise ValueError(f"unsupported math benchmark {benchmark!r}")
    return mapped


def math_verify_available() -> bool:
    try:
        import math_verify  # noqa: F401
    except Exception:
        return False
    return True


def extract_boxed_answer(text: str) -> str | None:
    """Extract the stripped content of the last well-formed braced ``\\boxed{...}``."""
    text = str(text)
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None

    index = matches[-1].end() - 1
    depth = 0
    start = index + 1
    for cursor in range(index, len(text)):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                content = text[start:cursor].strip()
                return content
    return None


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


def _extract_last_number(text: str) -> str | None:
    matches = re.findall(r"-?[\d,]+(?:\.\d+)?", str(text))
    if not matches:
        return None
    return matches[-1].replace(",", "")


def _contains_boxed_answer_marker(text: str) -> bool:
    return re.search(r"\\boxed\s*\{", str(text)) is not None


def _answers_match(pred: str, gold: str) -> bool:
    try:
        return float(pred) == float(gold)
    except ValueError:
        return normalize_answer(pred) == normalize_answer(gold)


def score_math_response(
    response: str,
    gold: str,
    *,
    benchmark: str,
    format_reward: float = DEFAULT_FORMAT_REWARD,
    question: str | None = None,
) -> float:
    pred = extract_boxed_answer(response)
    if pred is None:
        return 0.0
    if _answers_match(pred, gold):
        return 1.0
    try:
        if check_math_verify_by_benchmark(
            rf"\boxed{{{pred}}}",
            gold,
            validation_benchmark(benchmark),
            question=question,
        ):
            return 1.0
    except Exception:
        pass
    return float(format_reward)


def is_math_correct(
    response: str | None,
    gold: str,
    *,
    benchmark: str,
    question: str | None = None,
) -> bool:
    if response is None:
        return False
    mapped = validation_benchmark(benchmark)
    if math_verify_available():
        return bool(
            check_math_verify_by_benchmark(
                response,
                gold,
                mapped,
                question=question,
            )
        )
    pred = extract_boxed_answer(response)
    if pred is None and mapped == "gsm8k" and not _contains_boxed_answer_marker(response):
        pred = _extract_last_number(response)
    if pred is None:
        return False
    return _answers_match(pred, gold)


def scoring_metadata(benchmark: str) -> dict[str, Any]:
    available = math_verify_available()
    return {
        "benchmark": benchmark,
        "validation_benchmark": validation_benchmark(benchmark),
        "answer_style": "math",
        "math_verify_available": available,
        "scoring_fallback": None
        if available
        else (
            "boxed_or_last_number"
            if validation_benchmark(benchmark) == "gsm8k"
            else "normalized_exact"
        ),
    }


@dataclass(frozen=True)
class ValidationPolicy:
    benchmark: str
    family: str
    parse_mode: str
    prepare_gold: Callable[[str], str]
    build_gold_extraction: Callable[[str], tuple[Any, ...]]
    build_pred_extraction: Callable[[], tuple[Any, ...]]


def _identity_gold(text: str) -> str:
    return str(text)


def _wrap_inline_latex_gold(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("$") and stripped.endswith("$"):
        return stripped
    return f"${stripped}$"


def _soft_grpo_normalization_config():
    try:
        from math_verify import NormalizationConfig as normalization_config_cls
    except ImportError:
        from math_verify import LatexNormalizationConfig as normalization_config_cls

    return normalization_config_cls(
        nits=False,
        malformed_operators=False,
        basic_latex=True,
        boxed="all",
        units=True,
    )


def _soft_grpo_latex_pred_config():
    from math_verify import LatexExtractionConfig

    return LatexExtractionConfig(
        normalization_config=_soft_grpo_normalization_config(),
        boxed_match_priority=0,
        try_extract_without_anchor=False,
    )


def _expr_only_gold(_: str) -> tuple[Any, ...]:
    from math_verify import ExprExtractionConfig

    return (ExprExtractionConfig(),)


def _latex_only_gold(_: str) -> tuple[Any, ...]:
    from math_verify import LatexExtractionConfig

    return (LatexExtractionConfig(),)


def _latex_then_expr_pred() -> tuple[Any, ...]:
    from math_verify import ExprExtractionConfig

    return (_soft_grpo_latex_pred_config(), ExprExtractionConfig())


_VALIDATION_POLICIES = {
    "gsm8k": ValidationPolicy(
        benchmark="gsm8k",
        family="math",
        parse_mode="first_match",
        prepare_gold=_identity_gold,
        build_gold_extraction=_expr_only_gold,
        build_pred_extraction=_latex_then_expr_pred,
    ),
    "math500": ValidationPolicy(
        benchmark="math500",
        family="math",
        parse_mode="first_match",
        prepare_gold=_wrap_inline_latex_gold,
        build_gold_extraction=_latex_only_gold,
        build_pred_extraction=_latex_then_expr_pred,
    ),
}


def get_validation_policy(benchmark: str) -> ValidationPolicy:
    return _VALIDATION_POLICIES[validation_benchmark(benchmark)]


def check_math_verify_by_benchmark(
    response: str | None,
    gold: str,
    benchmark: str,
    *,
    question: str | None = None,
) -> int:
    del question
    if response is None:
        return 0

    from math_verify import parse, verify

    policy = get_validation_policy(benchmark)
    gold_text = policy.prepare_gold(gold)
    try:
        gold_parsed = parse(
            gold_text,
            extraction_config=policy.build_gold_extraction(gold_text),
            extraction_mode=policy.parse_mode,
        )
        pred_parsed = parse(
            response,
            extraction_config=policy.build_pred_extraction(),
            extraction_mode=policy.parse_mode,
        )
        if not gold_parsed or not pred_parsed:
            return 0
        return int(verify(gold_parsed, pred_parsed))
    except Exception:
        return 0
