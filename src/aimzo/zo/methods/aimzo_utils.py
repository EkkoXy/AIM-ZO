from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_json_checksum(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_int_seed(payload: Any) -> int:
    digest = stable_json_checksum(payload)
    return int(digest[:16], 16) % (2**63 - 1)
