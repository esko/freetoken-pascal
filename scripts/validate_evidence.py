from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, SchemaError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_DIR = ROOT / "schemas"
DEFAULT_EXAMPLE_DIR = ROOT / "tests" / "fixtures" / "results"


def validate_document(document: Any, *, schema_dir: Path) -> list[str]:
    if not isinstance(document, dict):
        return ["document root must be an object"]
    schema_name = document.get("schema_name")
    if not isinstance(schema_name, str) or Path(schema_name).name != schema_name:
        return ["schema_name must name a schema in the repository schema directory"]
    schema_path = schema_dir / schema_name
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
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
    if document.get("evidence_status") == "measured" and document.get("commit") == "0" * 40:
        errors.append("measured evidence cannot use the placeholder commit")
    return errors


def validate_paths(paths: list[Path], *, schema_dir: Path) -> list[str]:
    errors: list[str] = []
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
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
