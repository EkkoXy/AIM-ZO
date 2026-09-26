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
    assert len(manifest) == 12
    for row in manifest:
        path = ROOT / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert len(_rows(path)) == int(row["rows"])
        assert row["release_code_revision"] in {
            "50779c6",
            "dd3d001",
            "83b5b59",
            "a9ac64d",
        }


def test_result_rows_have_public_config_provenance() -> None:
    seed_files = [
        RESULTS / "opt-2.7b" / "official-by-seed.csv",
        RESULTS / "opt13b" / "official-by-seed.csv",
        RESULTS / "qwen3-0.6b" / "official-by-seed.csv",
    ]
    for path in seed_files:
        for row in _rows(path):
            assert len(row["release_config_sha256"]) == 64
            expected_revision = (
                "a9ac64d" if path.parent.name == "opt-2.7b" else "50779c6"
            )
            assert row["release_code_revision"] == expected_revision
            assert not any(
                token in " ".join(row.values()).lower()
                for token in (
                    "/home/",
                    "/media/",
                    "/public_data/",
                    "myzo",
                    "oszo",
                    "zo4llm",
                )
            )


def test_paper_metric_coverage_and_disclosures() -> None:
    opt27 = _rows(RESULTS / "opt-2.7b" / "paper-main.csv")
    opt = _rows(RESULTS / "opt13b" / "paper-main.csv")
    qwen = _rows(RESULTS / "qwen3-0.6b" / "paper-main.csv")
    assert len(opt27) == 14
    assert {row["method"] for row in opt27} == {"MeZO", "AIM-ZO"}
    assert all(row["num_seeds"] == "5" for row in opt27)
    assert all(
        row["provenance_status"] == "aggregate_from_verified_seed_rows"
        for row in opt27
    )
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
        assert row["release_code_revision"] == "dd3d001"


def test_large_model_result_and_gap_coverage() -> None:
    qwen_seed = _rows(RESULTS / "qwen3-8b" / "official-by-seed.csv")
    qwen_paper = _rows(RESULTS / "qwen3-8b" / "paper-main.csv")
    opt_paper = _rows(RESULTS / "opt30b" / "paper-main.csv")
    coverage = _rows(RESULTS / "large-model-coverage.csv")
    zero_shot = _rows(RESULTS / "large-model-zero-shot.csv")
    assert len(qwen_seed) == 27
    assert len(qwen_paper) == 24
    assert len(opt_paper) == 18
    assert len(coverage) == 42
    assert len(zero_shot) == 16
    assert {row["method"] for row in qwen_seed} == {"AGZO"}
    assert all(row["release_code_revision"] == "83b5b59" for row in qwen_seed)
    assert len([row for row in coverage if row["reproduction_status"] == "complete"]) == 18
    assert len([row for row in coverage if row["reproduction_status"] != "complete"]) == 24
    multirc = [row for row in qwen_seed if row["dataset"] == "MultiRC"]
    assert {row["metric"] for row in multirc} == {"answer_accuracy", "f1a", "em"}
    assert {row["evaluation_scope"] for row in multirc} == {"full_official"}
