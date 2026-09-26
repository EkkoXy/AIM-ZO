from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    configured = os.environ.get("AIMZO_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    return project_root() / "data"


def dataset_root() -> Path:
    return data_root() / "dataset"


def model_root() -> Path:
    return data_root() / "model"


def dataset_path(*parts: str) -> Path:
    return dataset_root().joinpath(*parts)
