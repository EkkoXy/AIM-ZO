from __future__ import annotations

from pathlib import Path

from aimzo.config import DataConfig
from aimzo.tasks.gsm8k import GSM8KTask
from aimzo.tasks.math_tasks import DeepScalerTask, MathTask
from aimzo.tasks.zoregular import ZO_REGULAR_TASKS, ZORegularTask, is_zoregular_task


def supported_tasks() -> tuple[str, ...]:
    return ("gsm8k", "math", "deepscaler", *ZO_REGULAR_TASKS)


def build_task(config: DataConfig):
    if config.task == "gsm8k":
        return GSM8KTask(
            max_train_samples=config.max_train_samples,
            max_eval_samples=config.max_eval_samples,
            data_root=Path(config.data_root).expanduser()
            if config.data_root is not None
            else None,
        )
    if config.task == "math":
        if config.train_file is None or config.eval_file is None:
            raise ValueError("math task requires train_file and eval_file")
        root = Path(config.data_root).expanduser() if config.data_root else Path(".")
        return MathTask(
            train_path=root / config.train_file,
            eval_path=root / config.eval_file,
            max_train_samples=config.max_train_samples,
            max_eval_samples=config.max_eval_samples,
        )
    if config.task == "deepscaler":
        return DeepScalerTask(
            data_root=Path(config.data_root).expanduser()
            if config.data_root is not None
            else None,
            max_train_samples=config.max_train_samples,
            max_eval_samples=config.max_eval_samples,
        )
    if is_zoregular_task(config.task):
        if config.data_root is None:
            raise ValueError(f"{config.task} task requires data_root")
        return ZORegularTask(
            task_name=config.task,
            data_root=Path(config.data_root).expanduser(),
            max_train_samples=config.max_train_samples,
            max_eval_samples=config.max_eval_samples,
            zoregular_train_dev_samples=config.zoregular_train_dev_samples,
            seed=config.seed,
            sampling_policy=config.zoregular_sampling_policy,
        )
    raise ValueError(_unknown("task", config.task, supported_tasks()))


def _unknown(kind: str, value: str, supported: tuple[str, ...]) -> str:
    return f"unknown {kind} {value!r}; expected one of: {', '.join(supported)}"
