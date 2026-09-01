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

    assert len(paths) == 8
    assert VALIDATE_EVIDENCE.validate_paths(paths, schema_dir=SCHEMA_DIR) == []


def test_all_evidence_schemas_are_valid_draft_2020_12() -> None:
    schemas = sorted(SCHEMA_DIR.glob("*.schema.json"))

    assert len(schemas) == 9
    for path in schemas:
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_warm_cache_evidence_reuses_full_h2_identity_without_rehash() -> None:
    evidence = load("qwen38-gguf-cache-zero-warm-h2.json")

    assert VALIDATE_EVIDENCE.validate_document(evidence, schema_dir=SCHEMA_DIR) == []
    assert evidence["identity"]["hash_reuse"]["model_shard_hashes_recomputed"] is False
    assert evidence["identity"]["hash_reuse"]["runtime_ple_integrity_hash"] == "performed"
    assert evidence["identity"]["hash_reuse"]["source_identity"] == "full-h2-canonical"
    assert evidence["performance"]["decode_tokens_per_second"] is None
    assert evidence["thermal"]["qualification"] == "unqualified"


def test_warm_cache_evidence_rejects_rehash_and_unbounded_claims() -> None:
    cases = (
        (
            "identity.hash_reuse.model_shard_hashes_recomputed",
            lambda value: value["identity"]["hash_reuse"].update(
                model_shard_hashes_recomputed=True
            ),
        ),
        (
            "identity.source_full_h2_evidence_sha256",
            lambda value: value["identity"].update(
                source_full_h2_evidence_sha256="not-a-sha256"
            ),
        ),
        (
            "performance.decode_tokens_per_second",
            lambda value: value["performance"].update(decode_tokens_per_second=1.0),
        ),
        (
            "thermal.qualification",
            lambda value: value["thermal"].update(qualification="qualified"),
        ),
        (
            "timing.total_seconds",
            lambda value: value["timing"].update(total_seconds=301),
        ),
        (
            "request.max_new_tokens",
            lambda value: value["request"].update(max_new_tokens=9),
        ),
    )
    for expected_path, mutate in cases:
        invalid = copy.deepcopy(load("qwen38-gguf-cache-zero-warm-h2.json"))
        mutate(invalid)

        errors = VALIDATE_EVIDENCE.validate_document(invalid, schema_dir=SCHEMA_DIR)

        assert any(expected_path.rsplit(".", 1)[-1] in error for error in errors), (
            expected_path,
            errors,
        )


def test_dual_p4_device_evidence_is_explicitly_non_serving() -> None:
    evidence = load("qwen38-dual-p4-device.json")

    assert VALIDATE_EVIDENCE.validate_document(evidence, schema_dir=SCHEMA_DIR) == []
    assert evidence["serving"] == {
        "classification": "non-serving",
        "model_loaded": False,
        "model_forward": False,
        "tps_claimed": False,
        "thermal_qualified": False,
        "dual_gpu_policy": "not-selected",
    }
    assert [device["index"] for device in evidence["devices"]] == [0, 1]


def test_dual_p4_device_evidence_rejects_serving_tps_and_thermal_claims() -> None:
    cases = (
        (
            "serving.classification",
            lambda value: value["serving"].update(classification="serving"),
        ),
        (
            "serving.tps_claimed",
            lambda value: value["serving"].update(tps_claimed=True),
        ),
        (
            "performance.tokens_per_second",
            lambda value: value["performance"].update(tokens_per_second=1.0),
        ),
        (
            "thermal.qualification",
            lambda value: value["thermal"].update(qualification="qualified"),
        ),
        (
            "devices[0].allocation_bytes",
            lambda value: value["devices"][0].update(allocation_bytes=4194305),
        ),
        (
            "devices",
            lambda value: value["devices"].__setitem__(
                1, copy.deepcopy(value["devices"][0])
            ),
        ),
    )
    for expected_path, mutate in cases:
        invalid = copy.deepcopy(load("qwen38-dual-p4-device.json"))
        mutate(invalid)

        errors = VALIDATE_EVIDENCE.validate_document(invalid, schema_dir=SCHEMA_DIR)

        assert errors, expected_path


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
