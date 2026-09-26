from pathlib import Path

import yaml

from aimzo.config import load_config
from aimzo.reproduction import (
    experiment_relative_path,
    iter_experiments,
    load_experiment_registry,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "main" / "experiments.yaml"


def test_registry_expands_and_matches_call_budgets(tmp_path: Path) -> None:
    registry = load_experiment_registry(REGISTRY)
    experiments = list(iter_experiments(registry))
    assert len(experiments) > 200
    for protocol, config in experiments:
        budget = protocol["steps"] * protocol["calls_per_step"]
        if protocol.get("budget_policy") == "short_runtime":
            assert protocol["steps"] == 10
        else:
            assert budget in {39999, 40000}
        milestones = config["trainer"]["checkpoint_milestones"]
        assert len(milestones) in {0, 5}
        path = tmp_path / experiment_relative_path(protocol, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        load_config(path)


def test_task_specific_parameters_and_seed_counts() -> None:
    registry = load_experiment_registry(REGISTRY)
    qwen = list(iter_experiments(registry, protocol_ids=["qwen06-mezo"]))
    counts = {}
    for _, config in qwen:
        task = config["data"]["task"]
        counts[task] = counts.get(task, 0) + 1
    assert counts["rte"] == 5
    assert counts["boolq"] == 3
    assert counts["squad"] == 3
    assert all(not config["trainer"]["checkpoint_milestones"] for _, config in qwen)

    opt = list(iter_experiments(registry, protocol_ids=["opt27-aimzo"], tasks=["wic", "wsc"], seeds=[42]))
    by_task = {config["data"]["task"]: config for _, config in opt}
    assert by_task["wic"]["zo"]["eps_schedule_min_ratio"] == 0.5
    assert by_task["wsc"]["zo"]["learning_rate"] == 2.0e-3


def test_generation_uses_loss_checkpoint_selection() -> None:
    registry = load_experiment_registry(REGISTRY)
    _, squad = next(iter(iter_experiments(registry, protocol_ids=["opt13-aimzo"], tasks=["squad"], seeds=[42])))
    assert squad["objective"]["name"] == "zoregular_generation_ce"
    assert squad["trainer"]["best_checkpoint_metric"] == "loss"
    assert squad["trainer"]["best_checkpoint_mode"] == "min"

def test_opt13_mezo_and_aimzo_cover_six_paper_tasks() -> None:
    registry = load_experiment_registry(REGISTRY)
    expected = {"rte", "boolq", "sst2", "wic", "wsc", "squad"}
    by_id = {row["id"]: row for row in registry["protocols"]}
    assert set(by_id["opt13-mezo"]["tasks"]) == expected
    assert set(by_id["opt13-aimzo"]["tasks"]) == expected

def test_large_model_runtime_protocols_are_exact_short_runs() -> None:
    registry = load_experiment_registry(REGISTRY)
    runtime = list(
        iter_experiments(
            registry,
            protocol_ids=[
                "qwen8-runtime-mezo",
                "qwen8-runtime-aimzo",
                "qwen8-runtime-agzo",
                "opt30-runtime-mezo",
                "opt30-runtime-aimzo",
                "opt30-runtime-agzo",
            ],
        )
    )
    assert len(runtime) == 6
    assert {protocol["model"] for protocol, _ in runtime} == {
        "qwen3-8b-base",
        "opt-30b",
    }
    assert {protocol["method"] for protocol, _ in runtime} == {
        "mezo",
        "aimzo",
        "agzo",
    }
    for protocol, config in runtime:
        assert protocol["scope"] == "paper_runtime"
        assert config["backend"]["dtype"] == "bfloat16"
        assert config["backend"]["max_model_len"] == 2048
        assert config["trainer"]["max_steps"] == 10
        assert config["trainer"]["eval_every"] == 0
        assert config["trainer"]["save_checkpoints"] is False
        assert config["trainer"]["checkpoint_milestones"] == []
        assert config["logging"]["history_snapshot_interval"] == 1
        if protocol["method"] == "agzo":
            assert config["zo"]["agzo"]["basis_seed_mode"] == "ambient"
            assert config["zo"]["agzo"]["max_activation_tokens"] == 0
        if protocol["method"] == "aimzo":
            assert config["zo"]["aimzo"][
                "abh_population_fused_varied_q_update"
            ] is True
