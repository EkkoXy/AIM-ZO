from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aimzo.logging import make_json_safe, write_json_atomic


class RunArtifactWriter:
    def __init__(
        self,
        output_dir: str | Path,
        tensorboard_writer: Any | None = None,
        history_snapshot_interval: int | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.tensorboard_writer = tensorboard_writer
        self.history_snapshot_interval = history_snapshot_interval
        self.warnings: list[str] = []
        self._history_jsonl_keys: set[tuple[str, str]] | None = None

    @property
    def has_tensorboard_writer(self) -> bool:
        return self.tensorboard_writer is not None and bool(
            getattr(self.tensorboard_writer, "enabled", True)
        )

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, payload: dict[str, Any]) -> None:
        self._write("config.json", payload)
        self._tensorboard_text("config", payload, step=0)
        self._persist_warnings()

    def write_manifest(self, payload: dict[str, Any]) -> None:
        self._write("manifest.json", payload)
        self._tensorboard_text("manifest", payload, step=0)
        self._persist_warnings()

    def write_history(
        self,
        payload: list[dict[str, Any]],
        *,
        force_snapshot: bool = False,
    ) -> None:
        self._append_new_history_rows(payload)
        if self._should_snapshot_history(payload, force_snapshot=force_snapshot):
            self._write("history.json", payload)
        if payload:
            self._tensorboard_step(payload[-1])
        self._persist_warnings()

    def write_eval_history(self, payload: list[dict[str, Any]]) -> None:
        self._write("eval_history.json", payload)
        self._persist_warnings()

    def write_results(self, payload: dict[str, Any]) -> None:
        history = payload.get("history") if isinstance(payload, dict) else None
        if isinstance(history, list):
            self._write("history.json", history)
        self._write("results.json", payload)
        self._tensorboard_text("results", payload, step=0)
        self._persist_warnings()

    def write_runtime_summary(self, payload: dict[str, Any]) -> None:
        self._write("runtime_summary.json", payload)
        self._persist_warnings()

    def append_candidate_result(self, payload: dict[str, Any]) -> None:
        self._append_jsonl("candidate_results.jsonl", payload)
        self._persist_warnings()

    def append_rollout_batch(self, payload: dict[str, Any]) -> None:
        self._append_jsonl("rollout_batches.jsonl", payload)
        self._persist_warnings()

    def append_tensorboard_samples(
        self,
        tag: str,
        samples: list[dict[str, Any]],
        *,
        step: int,
    ) -> None:
        if self.tensorboard_writer is None:
            return
        try:
            self.tensorboard_writer.log_samples(tag, samples, step=step)
        except Exception as exc:  # noqa: BLE001 - TensorBoard is best-effort.
            self.warnings.append(f"failed to write tensorboard samples {tag}: {exc}")
        self._persist_warnings()

    def close(self) -> None:
        if self.tensorboard_writer is None:
            return
        try:
            self.tensorboard_writer.close()
        except Exception as exc:  # noqa: BLE001 - TensorBoard is best-effort.
            self.warnings.append(f"failed to close tensorboard writer: {exc}")
        self._persist_warnings()

    def checkpoint_path(self, *, step: int) -> Path:
        return self.output_dir / "checkpoints" / f"step_{int(step)}"

    def final_checkpoint_path(self) -> Path:
        return self.output_dir / "checkpoints" / "final"

    def phase3_checkpoint_path(self, *, step: int) -> Path:
        return self.output_dir / "checkpoints" / f"step_{int(step):06d}"

    def phase3_final_adapter_path(self) -> Path:
        return self.output_dir / "final"

    def _write(self, filename: str, payload: Any) -> None:
        self.ensure_output_dir()
        write_json_atomic(self.output_dir / filename, payload)

    def _append_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
        self.ensure_output_dir()
        path = self.output_dir / filename
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    make_json_safe(payload),
                    ensure_ascii=False,
                    default=str,
                    allow_nan=False,
                )
                + "\n"
            )

    def _append_new_history_rows(self, payload: list[dict[str, Any]]) -> None:
        self._ensure_history_jsonl_keys_loaded()
        assert self._history_jsonl_keys is not None
        for index, row in enumerate(payload):
            key = self._history_row_key(row, index=index)
            if key in self._history_jsonl_keys:
                continue
            self._append_jsonl("history.jsonl", row)
            self._history_jsonl_keys.add(key)

    def _ensure_history_jsonl_keys_loaded(self) -> None:
        if self._history_jsonl_keys is not None:
            return
        self._history_jsonl_keys = set()
        path = self.output_dir / "history.jsonl"
        if not path.is_file():
            return
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                self.warnings.append(
                    f"failed to read history.jsonl row {index + 1}: {exc}"
                )
                continue
            if isinstance(row, dict):
                self._history_jsonl_keys.add(self._history_row_key(row, index=index))

    def _history_row_key(self, row: dict[str, Any], *, index: int) -> tuple[str, str]:
        if "step" in row:
            return ("step", str(row.get("step")))
        return ("index", str(index))

    def _should_snapshot_history(
        self,
        payload: list[dict[str, Any]],
        *,
        force_snapshot: bool,
    ) -> bool:
        if force_snapshot:
            return True
        if not payload:
            return True
        if not (self.output_dir / "history.json").exists():
            return True
        interval = self.history_snapshot_interval
        if interval is None or int(interval) <= 0:
            return False
        return len(payload) % int(interval) == 0

    def _tensorboard_text(self, tag: str, payload: Any, *, step: int) -> None:
        if self.tensorboard_writer is None:
            return
        try:
            self.tensorboard_writer.log_text(tag, payload, step=step)
        except Exception as exc:  # noqa: BLE001 - TensorBoard is best-effort.
            self.warnings.append(f"failed to write tensorboard text {tag}: {exc}")

    def _tensorboard_step(self, payload: dict[str, Any]) -> None:
        if self.tensorboard_writer is None:
            return
        try:
            self.tensorboard_writer.log_step(payload)
        except Exception as exc:  # noqa: BLE001 - TensorBoard is best-effort.
            self.warnings.append(f"failed to write tensorboard step: {exc}")

    def _persist_warnings(self) -> None:
        warnings = self._all_warnings()
        if not warnings:
            return
        self.ensure_output_dir()
        write_json_atomic(self.output_dir / "warnings.json", {"warnings": warnings})

    def _all_warnings(self) -> list[str]:
        collected: list[str] = []
        for warning in self.warnings:
            collected.append(str(warning))
        if self.tensorboard_writer is not None:
            for warning in getattr(self.tensorboard_writer, "warnings", []) or []:
                collected.append(str(warning))
        deduped: list[str] = []
        seen: set[str] = set()
        for warning in collected:
            if warning in seen:
                continue
            seen.add(warning)
            deduped.append(warning)
        return deduped
