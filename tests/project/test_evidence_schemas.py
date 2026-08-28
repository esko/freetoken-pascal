from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate_evidence.py"
SPEC = importlib.util.spec_from_file_location("validate_evidence", SCRIPT)
assert SPEC and SPEC.loader
VALIDATE_EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATE_EVIDENCE)
SCHEMA_DIR = ROOT / "schemas"
RESULT_DIR = ROOT / "tests" / "fixtures" / "results"


def load(name: str) -> dict:
    return json.loads((RESULT_DIR / name).read_text(encoding="utf-8"))


def test_all_example_evidence_is_schema_valid() -> None:
    paths = sorted(RESULT_DIR.glob("*.json"))

    assert len(paths) == 6
    assert VALIDATE_EVIDENCE.validate_paths(paths, schema_dir=SCHEMA_DIR) == []


def test_all_evidence_schemas_are_valid_draft_2020_12() -> None:
    schemas = sorted(SCHEMA_DIR.glob("*.schema.json"))

    assert len(schemas) == 5
    for path in schemas:
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_benchmark_requires_selected_runtime_behavior() -> None:
    invalid = copy.deepcopy(load("benchmark.json"))
    del invalid["selected_behavior"]

    errors = VALIDATE_EVIDENCE.validate_document(invalid, schema_dir=SCHEMA_DIR)

    assert any("selected_behavior" in error for error in errors)


def test_benchmark_rejects_unpinned_commit_and_missing_repeats() -> None:
    invalid = copy.deepcopy(load("benchmark.json"))
    invalid["commit"] = "main"
    invalid["runs"] = []

    errors = VALIDATE_EVIDENCE.validate_document(invalid, schema_dir=SCHEMA_DIR)

    assert any("commit" in error for error in errors)
    assert any("runs" in error for error in errors)


def test_benchmark_summary_must_match_raw_run_count() -> None:
    invalid = copy.deepcopy(load("benchmark.json"))
    invalid["summary"]["sample_count"] = 2

    errors = VALIDATE_EVIDENCE.validate_document(invalid, schema_dir=SCHEMA_DIR)

    assert errors == ["summary.sample_count must equal the number of raw runs"]


def test_correctness_evidence_requires_an_independent_reference() -> None:
    invalid = copy.deepcopy(load("correctness.json"))
    del invalid["reference"]

    errors = VALIDATE_EVIDENCE.validate_document(invalid, schema_dir=SCHEMA_DIR)

    assert any("reference" in error for error in errors)


def test_correctness_summary_cannot_disagree_with_comparisons() -> None:
    invalid = copy.deepcopy(load("correctness.json"))
    invalid["comparisons"][0]["passed"] = False

    errors = VALIDATE_EVIDENCE.validate_document(invalid, schema_dir=SCHEMA_DIR)

    assert errors == ["passed must equal the conjunction of comparison results"]


def test_unknown_schema_cannot_escape_schema_directory() -> None:
    invalid = copy.deepcopy(load("correctness.json"))
    invalid["schema_name"] = "../outside.json"

    errors = VALIDATE_EVIDENCE.validate_document(invalid, schema_dir=SCHEMA_DIR)

    assert errors == ["schema_name must name a schema in the repository schema directory"]
