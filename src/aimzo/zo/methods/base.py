from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Protocol

import torch

ObjectiveFn = Callable[[], tuple[float, dict[str, float]]]
EMPTY_UPDATE_DELTA_CHECKSUM = hashlib.sha256().hexdigest()

METHOD_SOURCE_COMMITS: dict[str, str | dict[str, str]] = {
    "mezo": {"mezo": "552cb1b", "zookit": "03f2061"},
    "hizoo": {"hizoo": "4f99055", "zookit": "03f2061"},
    "agzo": "623e09a",
    "aimzo": "aimzo-public-v1",
    "zomuon": "166a8f3",
    "zomopi": "2ba4f71",
    "curvzo": "anonymous-4open-CurvZO-9F35",
    "lozo": "5d1ade5",
    "oja_abq": "aimzo-paper-derived-oja-abq",
    "svd0": "paper-derived-openreview-nlgQsugmGw",
    "pgap": "paper-derived-arxiv-2510.18228",
}


@dataclass
class CandidateEvaluation:
    candidate_id: str
    objective_value: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    error_state: dict[str, Any] | None = None
    elapsed_time: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error_state is None and self.objective_value is not None

    def to_dict(
        self,
        *,
        seed: int,
        eps: float,
        scaling: float,
    ) -> dict[str, Any]:
        finite = self.objective_value is not None and math.isfinite(
            float(self.objective_value)
        )
        return {
            "candidate_id": self.candidate_id,
            "objective_value": self.objective_value,
            "ok": self.ok,
            "finite": finite,
            "error_state": self.error_state,
            "metrics": dict(self.metrics),
            "perturbation_seed": int(seed),
            "eps": float(eps),
            "scaling": float(scaling),
            "elapsed_time": float(self.elapsed_time),
        }


@dataclass
class HFZOMethodStepResult:
    zo_method: str
    seed: int
    eps: float
    learning_rate: float
    f_plus: float | None
    f_minus: float | None
    objective_delta: float | None
    projected_grad: float
    candidate_results: list[dict[str, Any]]
    method_diagnostics: dict[str, Any] = field(default_factory=dict)
    update_norm: float = 0.0
    parameter_delta_checksum: str = EMPTY_UPDATE_DELTA_CHECKSUM


class UpdateTraceAccumulator:
    def __init__(self, *, defer_samples: bool = False) -> None:
        self._digest = hashlib.sha256()
        self._squared_norm = 0.0
        self._defer_samples = bool(defer_samples)
        self._deferred_records: list[dict[str, Any]] = []

    def add_delta(
        self,
        name: str,
        source: torch.Tensor,
        *,
        scale: float = 1.0,
    ) -> None:
        if self._defer_samples:
            scale_value = float(scale)
            if not math.isfinite(scale_value):
                raise ValueError("update trace delta scale must be finite")
            detached = source.detach()
            self._deferred_records.append(
                {
                    "name": str(name),
                    "shape": [int(dim) for dim in detached.shape],
                    "dtype": str(detached.dtype),
                    "device": str(detached.device),
                    "numel": int(detached.numel()),
                    "scale": scale_value,
                    "sample_tensor": _bounded_tensor_sample_tensor(detached),
                }
            )
            return
        fingerprint, squared_norm = bounded_tensor_fingerprint(
            name,
            source,
            scale=scale,
            error_prefix="update trace delta",
        )
        self._squared_norm += squared_norm
        self._digest.update(
            json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        self._digest.update(b"\0")

    def add_factorized_delta(
        self,
        name: str,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        scale: float = 1.0,
    ) -> None:
        """Trace a 2-D ``left @ right`` update without materializing it."""
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError("factorized update trace requires compatible matrices")
        scale_value = float(scale)
        if not math.isfinite(scale_value):
            raise ValueError("update trace delta scale must be finite")
        shape = (int(left.shape[0]), int(right.shape[1]))
        numel = int(shape[0] * shape[1])
        sample_count = min(8, numel)
        flat_indices = (
            []
            if sample_count == 0
            else [0]
            if sample_count == 1
            else [
                round(index * (numel - 1) / (sample_count - 1))
                for index in range(sample_count)
            ]
        )
        values = []
        for flat_index in flat_indices:
            row, column = divmod(int(flat_index), shape[1])
            values.append((left[row] * right[:, column]).sum())
        sample_tensor = (
            torch.stack(values).detach().clone()
            if values
            else torch.empty(0, device=left.device, dtype=left.dtype)
        )
        if self._defer_samples:
            self._deferred_records.append(
                {
                    "name": str(name),
                    "shape": [*shape],
                    "dtype": str(left.dtype),
                    "device": str(left.device),
                    "numel": numel,
                    "scale": scale_value,
                    "sample_tensor": sample_tensor,
                }
            )
            return
        sample = [float(value) * scale_value for value in sample_tensor.cpu().tolist()]
        squared_norm = float(sum(value * value for value in sample))
        fingerprint = {
            "name": str(name),
            "shape": [*shape],
            "dtype": str(left.dtype),
            "device": str(left.device),
            "numel": numel,
            "scale": scale_value,
            "fingerprint_kind": "bounded_sample",
            "sample": sample,
            "summary": _bounded_sample_summary(
                sample,
                numel=numel,
                error_prefix="factorized update trace delta",
            ),
        }
        self._squared_norm += squared_norm
        self._digest.update(
            json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        self._digest.update(b"\0")

    def finish(self) -> tuple[float, str]:
        if self._deferred_records:
            self._finish_deferred_records()
        if not math.isfinite(self._squared_norm):
            raise ValueError("update trace accumulated norm must be finite")
        return math.sqrt(self._squared_norm), self._digest.hexdigest()

    def _finish_deferred_records(self) -> None:
        values_by_record: list[list[float] | None] = [
            None for _ in self._deferred_records
        ]
        records_by_device: dict[str, list[tuple[int, torch.Tensor]]] = {}
        for index, record in enumerate(self._deferred_records):
            sample_tensor = record["sample_tensor"]
            records_by_device.setdefault(str(sample_tensor.device), []).append(
                (index, sample_tensor)
            )
        for records in records_by_device.values():
            sizes = [int(sample.numel()) for _, sample in records]
            nonempty = [sample for _, sample in records if int(sample.numel()) > 0]
            merged_values = (
                torch.cat(nonempty).detach().cpu().tolist() if nonempty else []
            )
            offset = 0
            for (record_index, _), size in zip(records, sizes, strict=True):
                values_by_record[record_index] = [
                    float(value) for value in merged_values[offset : offset + size]
                ]
                offset += size
        for record, raw_values in zip(
            self._deferred_records,
            values_by_record,
            strict=True,
        ):
            scale = float(record["scale"])
            sample = [float(value) * scale for value in (raw_values or [])]
            squared_norm = float(sum(value * value for value in sample))
            fingerprint = {
                "name": str(record["name"]),
                "shape": list(record["shape"]),
                "dtype": str(record["dtype"]),
                "device": str(record["device"]),
                "numel": int(record["numel"]),
                "scale": scale,
                "fingerprint_kind": "bounded_sample",
                "sample": sample,
                "summary": _bounded_sample_summary(
                    sample,
                    numel=int(record["numel"]),
                    error_prefix="update trace delta",
                ),
            }
            self._squared_norm += squared_norm
            self._digest.update(
                json.dumps(
                    fingerprint,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            self._digest.update(b"\0")
        self._deferred_records.clear()


def bounded_tensor_fingerprint(
    name: str,
    tensor: torch.Tensor,
    *,
    scale: float = 1.0,
    max_values: int = 8,
    error_prefix: str = "tensor fingerprint",
) -> tuple[dict[str, Any], float]:
    scale_value = float(scale)
    if not math.isfinite(scale_value):
        raise ValueError(f"{error_prefix} scale must be finite")
    detached = tensor.detach()
    sample = _bounded_scaled_tensor_sample(
        detached,
        scale=scale_value,
        max_values=max_values,
        error_prefix=error_prefix,
    )
    squared_l2 = float(sum(value * value for value in sample))
    if not math.isfinite(squared_l2):
        raise ValueError(f"{error_prefix} norm must be finite")
    fingerprint = {
        "name": str(name),
        "shape": [int(dim) for dim in detached.shape],
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "scale": scale_value,
        "fingerprint_kind": "bounded_sample",
        "sample": sample,
        "summary": _bounded_sample_summary(
            sample,
            numel=int(detached.numel()),
            error_prefix=error_prefix,
        ),
    }
    return fingerprint, squared_l2


def _bounded_sample_summary(
    sample: list[float],
    *,
    numel: int,
    error_prefix: str,
) -> dict[str, float | int | None]:
    if not sample:
        return {
            "numel": int(numel),
            "sample_count": 0,
            "l2": 0.0,
            "mean": None,
            "min": None,
            "max": None,
        }
    l2 = math.sqrt(sum(value * value for value in sample))
    mean = sum(sample) / len(sample)
    minimum = min(sample)
    maximum = max(sample)
    for label, value in (
        ("l2", l2),
        ("mean", mean),
        ("min", minimum),
        ("max", maximum),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{error_prefix} {label} must be finite")
    return {
        "numel": int(numel),
        "sample_count": len(sample),
        "l2": l2,
        "mean": mean,
        "min": minimum,
        "max": maximum,
    }


def _bounded_scaled_tensor_sample(
    tensor: torch.Tensor,
    *,
    scale: float,
    max_values: int = 8,
    error_prefix: str,
) -> list[float]:
    numel = int(tensor.numel())
    if numel == 0:
        return []
    sample_count = min(max_values, numel)
    if sample_count == 1:
        indices = [0]
    else:
        indices = [
            round(index * (numel - 1) / (sample_count - 1))
            for index in range(sample_count)
        ]
    sample = [
        _scaled_scalar_at_flat_index(tensor, flat_index, scale=scale)
        for flat_index in indices
    ]
    for value in sample:
        if not math.isfinite(value):
            raise ValueError(f"{error_prefix} sample values must be finite")
    return sample


def _bounded_tensor_sample_tensor(
    tensor: torch.Tensor,
    *,
    max_values: int = 8,
) -> torch.Tensor:
    numel = int(tensor.numel())
    if numel == 0:
        return torch.empty(0, device=tensor.device, dtype=tensor.dtype)
    sample_count = min(max_values, numel)
    if sample_count == 1:
        flat_indices = [0]
    else:
        flat_indices = [
            round(index * (numel - 1) / (sample_count - 1))
            for index in range(sample_count)
        ]
    values = []
    for flat_index in flat_indices:
        remaining = int(flat_index)
        indices: list[int] = []
        for size in reversed(tensor.shape):
            size_int = int(size)
            indices.append(remaining % size_int)
            remaining //= size_int
        values.append(tensor[tuple(reversed(indices))] if tensor.ndim else tensor)
    return torch.stack(values).detach().clone()


def _scaled_scalar_at_flat_index(
    tensor: torch.Tensor,
    flat_index: int,
    *,
    scale: float,
) -> float:
    if tensor.ndim == 0:
        scalar = tensor
    else:
        remaining = int(flat_index)
        indices: list[int] = []
        for size in reversed(tensor.shape):
            size_int = int(size)
            indices.append(remaining % size_int)
            remaining //= size_int
        scalar = tensor[tuple(reversed(indices))]
    return float(scalar.item()) * float(scale)


class HFZOMethod(Protocol):
    name: str

    def step(
        self,
        model: torch.nn.Module,
        objective_fn: ObjectiveFn,
        *,
        seed: int,
        step: int,
    ) -> HFZOMethodStepResult: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


def method_config_diagnostics(
    *,
    eps: float,
    learning_rate: float,
    parameter_scope: str,
    weight_decay: float,
    **method_configs: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "eps": float(eps),
        "learning_rate": float(learning_rate),
        "parameter_scope": str(parameter_scope),
        "weight_decay": float(weight_decay),
    }
    for name, value in method_configs.items():
        if value is None:
            continue
        payload[name] = asdict(value) if is_dataclass(value) else dict(value)
    return payload


def objective_to_float(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("objective value tensor must contain exactly one element")
        numeric = float(value.detach().item())
    else:
        numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("objective value must be finite")
    return numeric


def evaluate_objective(
    candidate_id: str,
    objective_fn: ObjectiveFn,
) -> CandidateEvaluation:
    started_at = time.perf_counter()
    try:
        value, metrics = objective_fn()
        objective_value = objective_to_float(value)
    except Exception as exc:  # noqa: BLE001 - surfaced in candidate diagnostics.
        return CandidateEvaluation(
            candidate_id=candidate_id,
            objective_value=None,
            metrics={},
            error_state={"type": type(exc).__name__, "message": str(exc)},
            elapsed_time=time.perf_counter() - started_at,
        )
    return CandidateEvaluation(
        candidate_id=candidate_id,
        objective_value=objective_value,
        metrics={key: float(metric) for key, metric in dict(metrics).items()},
        elapsed_time=time.perf_counter() - started_at,
    )
