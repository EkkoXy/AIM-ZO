from __future__ import annotations

import json
import math
import numbers
import os
from pathlib import Path
from typing import Any


def make_json_safe(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, numbers.Real):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def write_json_atomic(path: Path | str, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(
        json.dumps(
            make_json_safe(payload),
            indent=2,
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, target)
