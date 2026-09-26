import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "main"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_release_manifest_checksums_and_revision() -> None:
    manifest = _rows(RESULTS / "manifest.csv")
    assert len(manifest) == 5
    for row in manifest:
        path = ROOT / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert len(_rows(path)) == int(row["rows"])
        assert row["release_code_revision"] in {"50779c6", "d8c80d1"}


def test_result_rows_have_public_config_provenance() -> None:
    seed_files = [
        RESULTS / "opt13b" / "official-by-seed.csv",
        RESULTS / "qwen3-0.6b" / "official-by-seed.csv",
    ]
    for path in seed_files:
        for row in _rows(path):
            assert len(row["release_config_sha256"]) == 64
            assert row["release_code_revision"] == "50779c6"
            assert not any(token in " ".join(row.values()).lower() for token in ("/home/", "/media/", "/public_data/", "myzo", "oszo", "zo4llm"))


def test_paper_metric_coverage_and_disclosures() -> None:
    opt = _rows(RESULTS / "opt13b" / "paper-main.csv")
    qwen = _rows(RESULTS / "qwen3-0.6b" / "paper-main.csv")
    assert len(opt) == 49
    assert len([row for row in opt if row["provenance_status"] == "historical_aggregate_without_seed_payload"]) == 8
    grouped = {(row["method"], row["metric"]) for row in qwen if row["dataset"] == "MultiRC"}
    assert grouped == {("MeZO", "answer_accuracy"), ("MeZO", "f1a"), ("MeZO", "em"), ("AIM-ZO", "answer_accuracy"), ("AIM-ZO", "f1a"), ("AIM-ZO", "em")}

def test_large_model_runtime_rows_are_individual_runs() -> None:
    rows = _rows(RESULTS / "large-model-runtime" / "by-run.csv")
    assert len(rows) == 6
    assert {(row["model"], row["method"]) for row in rows} == {
        (model, method)
        for model in ("Qwen3-8B-Base", "OPT-30B")
        for method in ("MeZO", "AIM-ZO", "AGZO")
    }
    for row in rows:
        assert row["seed"] == "42"
        assert row["dtype"] == "bfloat16"
        assert row["steps"] == "10"
        assert int(row["total_objective_calls"]) == 10 * int(
            row["objective_calls_per_step"]
        )
        assert len(row["executed_config_sha256"]) == 64
        assert len(row["release_config_sha256"]) == 64
        assert row["release_code_revision"] == "d8c80d1"
