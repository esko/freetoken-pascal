from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, SchemaError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_DIR = ROOT / "schemas"
DEFAULT_EXAMPLE_DIR = ROOT / "tests" / "fixtures" / "results"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r} is forbidden")


def _strict_json_loads(data: str) -> Any:
    return json.loads(data, parse_constant=_reject_json_constant)


def _nonfinite_paths(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [path]
    if isinstance(value, dict):
        return [
            nested
            for key, item in value.items()
            for nested in _nonfinite_paths(item, f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            nested
            for index, item in enumerate(value)
            for nested in _nonfinite_paths(item, f"{path}[{index}]")
        ]
    return []


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _comparison_semantic_errors(comparisons: list[dict[str, Any]]) -> list[str]:
    """Reject well-typed metric records whose predicate is nevertheless forged."""
    errors: list[str] = []
    for index, comparison in enumerate(comparisons):
        metric = comparison["metric"]
        observed = comparison["observed"]
        limit = comparison["limit"]
        passed = comparison["passed"]
        if metric in {"max_abs", "relative_rms"}:
            if not _finite_number(observed) or not _finite_number(limit):
                errors.append(f"comparisons[{index}] {metric} values must be finite numbers")
                continue
            expected = observed <= limit
        elif metric == "cosine":
            if not _finite_number(observed) or not _finite_number(limit):
                errors.append(f"comparisons[{index}] cosine values must be finite numbers")
                continue
            if not -1 <= observed <= 1 or not -1 <= limit <= 1:
                errors.append(f"comparisons[{index}] cosine values must be in [-1, 1]")
                continue
            expected = observed >= limit
        elif metric == "array_exact":
            expected = observed
        else:
            # String comparisons are exact contracts: a producer cannot mark two
            # different syntax/status or state-hash values as passing.
            expected = observed == limit
        if passed is not expected:
            errors.append(f"comparisons[{index}] {metric}.passed must equal its metric predicate")
    return errors


def validate_document(document: Any, *, schema_dir: Path) -> list[str]:
    if not isinstance(document, dict):
        return ["document root must be an object"]
    nonfinite = _nonfinite_paths(document)
    if nonfinite:
        return [f"{path}: non-finite numeric values are forbidden" for path in nonfinite]
    schema_name = document.get("schema_name")
    if not isinstance(schema_name, str) or Path(schema_name).name != schema_name:
        return ["schema_name must name a schema in the repository schema directory"]
    schema_path = schema_dir / schema_name
    try:
        schema = _strict_json_loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return [f"unable to read schema {schema_name}: {error}"]
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        return [f"invalid schema {schema_name}: {error.message}"]
    errors = [
        f"{'.'.join(str(part) for part in error.absolute_path) or '$'}: {error.message}"
        for error in sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document),
            key=lambda item: (list(item.absolute_path), item.message),
        )
    ]
    if errors:
        return errors
    if schema_name == "benchmark-result.schema.json":
        if document["summary"]["sample_count"] != len(document["runs"]):
            errors.append("summary.sample_count must equal the number of raw runs")
    elif schema_name == "quant-census.schema.json":
        quant_types = document["by_quant_type"].values()
        if sum(entry["tensors"] for entry in quant_types) != document["tensor_count"]:
            errors.append("tensor_count must equal the by_quant_type tensor sum")
        if (
            sum(entry["bytes"] for entry in document["by_quant_type"].values())
            != document["total_bytes"]
        ):
            errors.append("total_bytes must equal the by_quant_type byte sum")
        if document["shard_count"] != len(document["shards"]):
            errors.append("shard_count must equal the number of shard identities")
        if document["tensor_count"] != len(document["tensors"]):
            errors.append("tensor_count must equal the number of tensor records")
        if sum(entry["nbytes"] for entry in document["tensors"]) != document["total_bytes"]:
            errors.append("total_bytes must equal the tensor-record byte sum")
        measured = document["evidence_status"] == "measured"
        verified = all(shard["sha256_status"] == "verified" for shard in document["shards"])
        if measured != verified:
            errors.append("measured census status must exactly match verified shard hashes")
    elif schema_name == "correctness-evidence.schema.json":
        comparison_passed = all(comparison["passed"] for comparison in document["comparisons"])
        if document["passed"] != comparison_passed:
            errors.append("passed must equal the conjunction of comparison results")
        else:
            errors.extend(_comparison_semantic_errors(document["comparisons"]))
        if document["commit"] != document["subject"]["commit"]:
            errors.append("commit must identify the subject implementation commit")
        for key in (
            "artifact_sha256",
            "quant_census_sha256",
            "quantization",
            "cache_mode",
        ):
            if document["subject"][key] != document["reference"][key]:
                errors.append(f"subject/reference {key} must match")
        pairs = [
            (comparison["observation"], comparison["metric"])
            for comparison in document["comparisons"]
        ]
        if len(pairs) != len(set(pairs)):
            errors.append("comparison observation/metric pairs must be unique")
        if (
            document["evidence_status"] == "measured"
            and document["subject"]["implementation"] == document["reference"]["implementation"]
        ):
            errors.append("measured correctness evidence requires an independent reference")
        if document["evidence_status"] == "measured" and all(
            document["subject"][key] == document["reference"][key] for key in ("revision", "commit")
        ):
            errors.append("measured correctness evidence requires distinct immutable revisions")
        for party in ("subject", "reference"):
            for key in (
                "tokenizer_repository",
                "tokenizer_revision",
                "corpus_sha256",
                "prompt_id",
                "prompt_sha256",
                "context_tokens",
            ):
                if document[party][key] != document["workload"][key]:
                    errors.append(f"{party}.{key} must match workload.{key}")
    if document.get("evidence_status") == "measured" and document.get("commit") == "0" * 40:
        errors.append("measured evidence cannot use the placeholder commit")
    return errors


def validate_paths(paths: list[Path], *, schema_dir: Path) -> list[str]:
    errors: list[str] = []
    for path in paths:
        try:
            document = _strict_json_loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"{path}: unable to read JSON: {error}")
            continue
        errors.extend(
            f"{path}: {error}" for error in validate_document(document, schema_dir=schema_dir)
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate FreeToken-Pascal evidence bundles")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    args = parser.parse_args(argv)
    paths = args.paths or sorted(DEFAULT_EXAMPLE_DIR.glob("*.json"))
    if not paths:
        print("ERROR: no evidence documents selected", file=sys.stderr)
        return 1
    errors = validate_paths(paths, schema_dir=args.schema_dir)
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"validated {len(paths)} evidence documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
