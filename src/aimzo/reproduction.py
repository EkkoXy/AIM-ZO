from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

GENERATION_TASKS = frozenset({"squad", "record", "drop"})
EXPECTED_CALL_BUDGETS = frozenset({39999, 40000})


def load_experiment_registry(path: str | Path) -> dict[str, Any]:
    registry = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise TypeError("experiment registry must be a mapping")
    for key in ("models", "tasks", "protocols"):
        if key not in registry:
            raise ValueError(f"experiment registry is missing {key!r}")
    validate_experiment_registry(registry)
    return registry


def validate_experiment_registry(registry: dict[str, Any]) -> None:
    models = registry["models"]
    tasks = registry["tasks"]
    seen: set[str] = set()
    for protocol in registry["protocols"]:
        protocol_id = str(protocol["id"])
        if protocol_id in seen:
            raise ValueError(f"duplicate protocol id: {protocol_id}")
        seen.add(protocol_id)
        if protocol["model"] not in models:
            raise ValueError(f"{protocol_id}: unknown model {protocol['model']}")
        unknown_tasks = set(protocol["tasks"]) - set(tasks)
        if unknown_tasks:
            raise ValueError(f"{protocol_id}: unknown tasks {sorted(unknown_tasks)}")
        budget = int(protocol["steps"]) * int(protocol["calls_per_step"])
        if protocol.get("budget_policy") != "short_runtime" and budget not in EXPECTED_CALL_BUDGETS:
            raise ValueError(
                f"{protocol_id}: objective-call budget is {budget}, expected 39999 or 40000"
            )
        milestones = protocol.get("checkpoints")
        if milestones and (len(milestones) != 5 or int(milestones[-1]) != int(protocol["steps"])):
            raise ValueError(f"{protocol_id}: checkpoints must contain five points ending at max steps")


def iter_experiments(
    registry: dict[str, Any],
    *,
    protocol_ids: Iterable[str] | None = None,
    models: Iterable[str] | None = None,
    methods: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    filters = {
        "protocol": set(protocol_ids or ()),
        "model": set(models or ()),
        "method": set(methods or ()),
        "task": set(tasks or ()),
        "seed": {int(seed) for seed in seeds or ()},
    }
    for source in registry["protocols"]:
        if filters["protocol"] and source["id"] not in filters["protocol"]:
            continue
        if filters["model"] and source["model"] not in filters["model"]:
            continue
        if filters["method"] and source["method"] not in filters["method"]:
            continue
        for task in source["tasks"]:
            if filters["task"] and task not in filters["task"]:
                continue
            task_seeds = source.get("seeds_by_task", {}).get(task, source["seeds"])
            for seed in task_seeds:
                if filters["seed"] and int(seed) not in filters["seed"]:
                    continue
                resolved = deepcopy(source)
                resolved.update(source.get("per_task", {}).get(task, {}))
                yield resolved, build_training_config(registry, resolved, task, int(seed))


def build_training_config(
    registry: dict[str, Any], protocol: dict[str, Any], task: str, seed: int
) -> dict[str, Any]:
    task_spec = registry["tasks"][task]
    method = str(protocol["method"])
    steps = int(protocol["steps"])
    checkpoints = (
        []
        if protocol.get("checkpoint_policy") in {"interval", "disabled"}
        else list(protocol.get("checkpoints") or _uniform_milestones(steps))
    )
    config: dict[str, Any] = {
        "data": {
            "task": task,
            "data_root": "data/dataset",
            "max_train_samples": int(task_spec["train"]),
            "max_eval_samples": int(task_spec["dev"]),
            "seed": seed,
            "zoregular_sampling_policy": "mezo_train_dev",
            "zoregular_batch_sampler_policy": "accelerate_seedable",
        },
        "backend": {
            "name": "hf",
            "model_name": registry["models"][protocol["model"]],
            "dtype": protocol["dtype"],
            "max_model_len": int(protocol.get("context", task_spec["context"])),
        },
        "objective": {
            "name": (
                "zoregular_generation_ce"
                if task in GENERATION_TASKS
                else "zoregular_classification_ce"
            ),
            "rollout_protocol": "fixed_rollout_per_step",
            "loss_aggregation": "sequence_sum_then_sample_mean",
        },
        "zo": {
            "method": method,
            "estimator": "mezo_two_point",
            "eps": float(protocol["epsilon"]),
            "eps_schedule": protocol.get("epsilon_schedule", "constant"),
            "eps_schedule_min_ratio": float(protocol.get("epsilon_min_ratio", 0.0)),
            "eps_schedule_total_steps": int(protocol.get("epsilon_total_steps", steps)),
            "learning_rate": float(protocol["learning_rate"]),
            "lr_schedule": "constant",
            "parameter_scope": "full_parameters",
            "seed": seed,
            "seed_mode": "aimzo_step" if method == "aimzo" else "source_numpy_randint",
            "noise_backend": "gpu_seeded",
            "restore_strategy": "seed_replay",
            "weight_decay": 0.0,
            "source_update_order": bool(protocol.get("source_update_order", False)),
            "source_compatibility_profile": protocol.get("source_compatibility_profile"),
        },
        "trainer": {
            "output_dir": f"outputs/main/{protocol['model']}/{method}/{task}/seed{seed}",
            "train_batch_size": 16,
            "eval_batch_size": 16,
            "max_steps": steps,
            "eval_every": int(protocol["eval_every"]),
            "checkpoint_milestones": checkpoints,
            "save_every": 0,
            "max_periodic_checkpoints": 0 if protocol.get("checkpoint_policy") == "disabled" else 5,
            "save_checkpoints": protocol.get("checkpoint_policy") != "disabled",
            "best_checkpoint_metric": task_spec["metric"],
            "best_checkpoint_mode": task_spec["mode"],
        },
        "logging": {
            "tensorboard": False,
            "history_snapshot_interval": (
                1 if protocol.get("budget_policy") == "short_runtime" else None
            ),
        },
    }
    _add_method_config(config["zo"], method, protocol)
    return config


def experiment_relative_path(protocol: dict[str, Any], config: dict[str, Any]) -> Path:
    return Path(protocol["model"]) / protocol["method"] / config["data"]["task"] / f"seed{config['data']['seed']}.yaml"


def _uniform_milestones(steps: int) -> list[int]:
    return [round(steps * fraction / 5) for fraction in range(1, 6)]


def _add_method_config(zo: dict[str, Any], method: str, protocol: dict[str, Any]) -> None:
    if method == "aimzo":
        zo["aimzo"] = {
            "rank": 1,
            "target_module_regex": ".*",
            "fallback": "gaussian",
            "max_activation_tokens": 128,
            "estimator_mode": "loren_population",
            "subspace_backend": "power_iteration",
            "power_iterations": 5,
            "basis_seed_mode": "ambient",
            "perturbation_form": "abh_oja",
            "abh_normalization": "layer_fro",
            "abh_right_rank": 64,
            "abh_oja_wide_right_rank": 128,
            "abh_oja_active_top_count": 48,
            "abh_oja_active_tail_count": 16,
            "abh_oja_active_resample_per_probe": True,
            "abh_oja_eta": 0.3,
            "abh_oja_update_interval": int(protocol["aimzo_refresh"]),
            "abh_oja_q_update_rule": "qr_oja",
            "abh_noise_cache": "none",
            "abh_num_noise": 15,
            "abh_left_factor": "gaussian_ab_random",
            "abh_a_seed_mode": "block",
            "abh_loren_population_divide_by_eps": False,
            "abh_multi_update_weighting": "uniform",
            "abh_population_summary_last_only": True,
            "abh_population_deferred_update_trace": True,
            "abh_population_cache_restore_factors": True,
            "abh_population_fused_varied_q_update": bool(
                protocol.get("aimzo_fused_varied_q", False)
            ),
        }
    elif method == "agzo":
        zo["agzo"] = {
            "rank": 1,
            "target_module_regex": ".*",
            "fallback": "gaussian",
            "max_activation_tokens": int(
                protocol.get("agzo_max_activation_tokens", 128)
            ),
            "source_scalar_dtype": "bfloat16",
            "stream_activation_basis": True,
            "estimator_mode": "two_side",
            "subspace_backend": "power_iteration",
            "power_iterations": 3,
            "basis_seed_mode": protocol.get(
                "agzo_basis_seed_mode", "perturbation_seed"
            ),
            "perturbation_form": "basis",
        }
    elif method == "hizoo":
        zo["hizoo"] = {
            "hessian_smooth_type": "constant1e-8",
            "hessian_init": 1.0,
            "hessian_state_dtype": "float32" if protocol["dtype"] == "float16" else "parameter",
        }
    elif method == "lozo":
        zo["lozo"] = {"rank": 2 if protocol["model"] == "qwen3-0.6b-base" else 1, "step_interval": 50, "normalization": "source", "fallback": "gaussian"}
    elif method == "zomuon":
        zo["lowdim_muon"] = {"rank": 64, "step_interval": 100, "k_start": 32, "num_samples": 4, "multiple_sample": True, "perturbation_mode": "two_side", "phase2_optimizer": "muon", "beta": 0.0}
    elif method == "curvzo":
        zo["curvzo"] = {"sample_ratio": 0.4, "sgs_alpha": 0.1, "sensitivity_init": 0.01, "sensitivity_beta": 0.1, "sensitivity_global_beta": 0.1, "adaptive_every": 20, "sampling_mode": "poisson_pps"}
