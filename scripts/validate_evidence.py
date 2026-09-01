from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping
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
    if isinstance(value, bool) or not isinstance(value, int | float):
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


def _target_cpu_benchmark_semantic_errors(document: dict[str, Any]) -> list[str]:
    """Check invariants that JSON Schema cannot express across report fields."""
    errors: list[str] = []
    warmups = document["warmups"]
    warmup_count = warmups["count"]
    for name in ("reference_cold_dequant_dense", "reference_dense_resident", "native"):
        if len(warmups[name]) != warmup_count:
            errors.append(f"warmups.{name} length must equal warmups.count")

    samples = document["samples"]
    sample_names = ("reference_cold_dequant_dense", "reference_dense_resident", "native")
    sample_count = len(samples[sample_names[0]])
    for name in sample_names:
        if len(samples[name]) != sample_count:
            errors.append(f"samples.{name} length must match the other raw sample arrays")
    for group_name in ("dense_resident", "cold_dequant_dense"):
        group = document["statistics"][group_name]
        if group["reference"]["sample_count"] != sample_count:
            errors.append(f"statistics.{group_name}.reference.sample_count must match raw samples")
        if group["native"]["sample_count"] != sample_count:
            errors.append(f"statistics.{group_name}.native.sample_count must match raw samples")
        expected_ratio = (
            group["reference"]["median_elapsed_ns"] / group["native"]["median_elapsed_ns"]
        )
        if not math.isclose(
            group["reference_to_native_median_ratio"], expected_ratio, rel_tol=1e-12
        ):
            errors.append(
                f"statistics.{group_name}.reference_to_native_median_ratio must match medians"
            )

    required_cold_components = {
        "range/source validation",
        "packed bytes/view setup",
        "sha256 hashing",
        "gguf.dequantize",
        "dense_fp32_swiglu",
    }
    for index, sample in enumerate(samples["reference_cold_dequant_dense"]):
        missing = required_cold_components - set(sample["timed_components"])
        if missing:
            errors.append(f"cold sample {index} omits timed components: {sorted(missing)}")
    for index, sample in enumerate(samples["reference_dense_resident"]):
        if "gguf.dequantize" in sample["timed_components"]:
            errors.append(f"dense-resident sample {index} includes out-of-scope dequantization")
        if "dense_fp32_swiglu" not in sample["timed_components"]:
            errors.append(f"dense-resident sample {index} must include dense_fp32_swiglu")
    selected = document["selected_behavior"]
    native_observation_groups = (
        ("warmups.native", warmups["native"]),
        ("samples.native", samples["native"]),
    )
    for group_name, observations in native_observation_groups:
        for index, sample in enumerate(observations):
            if sample["timed_operation"] != "executor.execute":
                errors.append(f"{group_name}[{index}] must time executor.execute")
            telemetry = sample["telemetry"]
            if telemetry.get("fallback_reason") is not None:
                errors.append(f"{group_name}[{index}] reports fallback telemetry")
            if telemetry.get("backend") != selected["backend"]:
                errors.append(f"{group_name}[{index}] backend must match selected behavior")
            if telemetry.get("kernel_census") != selected["kernel_census"]:
                errors.append(f"{group_name}[{index}] kernels must match selected behavior")

    for name in ("cold_dequant_dense", "dense_resident"):
        group = document["correctness"][name]
        if group["comparison_count"] != len(group["comparisons"]):
            errors.append(f"correctness.{name}.comparison_count must match comparisons")
        if not all(comparison["correct"] for comparison in group["comparisons"]):
            errors.append(f"correctness.{name} contains a failed comparison")
    comparisons_correct = all(
        all(comparison["correct"] for comparison in document["correctness"][name]["comparisons"])
        for name in ("cold_dequant_dense", "dense_resident")
    )
    if document["correctness"]["correct"] != comparisons_correct:
        errors.append("correctness.correct must equal all per-sample comparisons")

    if "avx2" not in selected["backend"]:
        errors.append("selected_behavior.backend must identify avx2")
    if not selected["kernel_census"] or any(
        "avx2" not in kernel for kernel in selected["kernel_census"]
    ):
        errors.append("selected_behavior.kernel_census must identify only avx2 kernels")
    if any(selected["fallbacks"].values()):
        errors.append("selected_behavior.fallbacks must be empty for measured evidence")
    identity_scope = document["metadata"]["identity_scope"].lower()
    if "selected expert ranges" not in identity_scope or "full-model" not in identity_scope:
        errors.append("metadata.identity_scope must state full-model identity and partial ranges")
    cold_scope = document["statistics"]["cold_dequant_dense"]["scope"].lower()
    for required in ("validation", "setup", "sha-256", "dequantization", "dense"):
        if required not in cold_scope:
            errors.append(f"cold_dequant_dense scope must mention {required}")
    if document["warnings"]:
        errors.append("measured target-CPU evidence cannot carry warning-only BLAS state")
    native = document["metadata"]["native"]
    build = native["build"]
    if build["commit"] != document["metadata"]["commit"]:
        errors.append("native build commit must match benchmark commit")
    for name in ("q4_k", "mixed_gemv"):
        measured_library = native["libraries"][name]
        built_library = build["libraries"][name]
        if built_library["sha256"] != measured_library["sha256"]:
            errors.append(f"native build hash for {name} must match measured library")
        if Path(built_library["path"]).resolve() != Path(measured_library["path"]).resolve():
            errors.append(f"native build path for {name} must match measured library")
    return errors


_Q3_CONTENT_RANGE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)


def _q3_read_json(path: Path, *, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        errors.append(f"unable to read repository {label} {path}: {error}")
        return None
    if not isinstance(value, dict):
        errors.append(f"repository {label} {path} must contain an object")
        return None
    return value


def _qwen38_q3_triad_semantic_errors(document: dict[str, Any], *, schema_dir: Path) -> list[str]:
    """Validate all cross-field and on-disk invariants for Q3 triad evidence."""
    errors: list[str] = []
    metadata = document["metadata"]
    artifact = metadata["artifact"]
    census_identity = metadata["census"]
    source = document["source"]
    repo_root = Path(schema_dir).resolve().parent
    expected_manifest_path = (repo_root / "manifests/qwen38-gguf.json").resolve()
    expected_census_path = (
        repo_root / "tests/fixtures/results/qwen38-q3-census.metadata.json"
    ).resolve()

    def _repository_path(value: str, expected: Path, label: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        if path != expected:
            errors.append(f"metadata.{label} must identify repository file {expected}")
        return path

    manifest_path = _repository_path(
        artifact["manifest_path"], expected_manifest_path, "artifact.manifest_path"
    )
    census_path = _repository_path(census_identity["path"], expected_census_path, "census.path")
    manifest = _q3_read_json(manifest_path, label="manifest", errors=errors)
    actual_census = _q3_read_json(census_path, label="census", errors=errors)
    if manifest is not None:
        try:
            manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        except OSError as error:
            errors.append(f"unable to hash repository manifest {manifest_path}: {error}")
        else:
            if artifact["manifest_sha256"] != manifest_sha256:
                errors.append(
                    "metadata.artifact.manifest_sha256 disagrees with repository manifest"
                )
        try:
            manifest_variant = manifest["variants"]["UD-Q3_K_XL"]
        except (KeyError, TypeError) as error:
            manifest_variant = None
            errors.append(f"repository manifest lacks UD-Q3_K_XL: {error}")
        if manifest_variant is not None:
            if artifact["repository"] != manifest["repository"]:
                errors.append("metadata.artifact.repository disagrees with repository manifest")
            if artifact["revision"] != manifest["revision"]:
                errors.append("metadata.artifact.revision disagrees with repository manifest")
            if artifact["shards"] != manifest_variant["shards"]:
                errors.append("metadata.artifact.shards disagrees with repository manifest")
    if actual_census is not None:
        try:
            census_sha256 = hashlib.sha256(census_path.read_bytes()).hexdigest()
        except OSError as error:
            errors.append(f"unable to hash repository census {census_path}: {error}")
        else:
            if census_identity["sha256"] != census_sha256:
                errors.append("metadata.census.sha256 disagrees with repository census")
        try:
            actual_counts = {
                str(name): int(values["tensors"])
                for name, values in actual_census["by_quant_type"].items()
            }
            actual_bytes = {
                str(name): int(values["bytes"])
                for name, values in actual_census["by_quant_type"].items()
            }
            actual_tensor_count = int(actual_census["tensor_count"])
            actual_total_bytes = int(actual_census["total_bytes"])
            actual_model_sha256 = str(actual_census["model_sha256"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"repository census totals/counts are malformed: {error}")
        else:
            if census_identity["quant_type_counts"] != actual_counts:
                errors.append("metadata.census.quant_type_counts disagrees with repository census")
            if census_identity["quant_type_bytes"] != actual_bytes:
                errors.append("metadata.census.quant_type_bytes disagrees with repository census")
            if census_identity["tensor_count"] != actual_tensor_count:
                errors.append("metadata.census.tensor_count disagrees with repository census")
            if census_identity["total_bytes"] != actual_total_bytes:
                errors.append("metadata.census.total_bytes disagrees with repository census")
            if census_identity["model_sha256"] != actual_model_sha256:
                errors.append("metadata.census.model_sha256 disagrees with repository census")
            if artifact["model_sha256"] != actual_model_sha256:
                errors.append("metadata.artifact.model_sha256 disagrees with repository census")
        artifact_shards = [
            (item["name"], item["size"], item["sha256"]) for item in artifact["shards"]
        ]
        census_shards = [
            (item["name"], item["size"], item["sha256"]) for item in actual_census.get("shards", [])
        ]
        if artifact_shards != census_shards:
            errors.append("metadata.artifact.shards disagrees with repository census shards")

    if artifact["repository"] != source["repository"]:
        errors.append("metadata.artifact.repository must match source.repository")
    if artifact["revision"] != source["revision"]:
        errors.append("metadata.artifact.revision must match source.revision")
    if artifact["variant"] != source["variant"]:
        errors.append("metadata.artifact.variant must match source.variant")
    expected_base_url = (
        f"https://huggingface.co/{source['repository']}/resolve/"
        f"{source['revision']}/{source['variant']}"
    )
    if source["base_url"] != expected_base_url:
        errors.append("source.base_url does not match repository/revision/variant")

    try:
        current_commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .lower()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        errors.append(f"cannot resolve current repository commit: {error}")
    else:
        if metadata["commit"] != current_commit:
            errors.append("metadata.commit must identify the current repository commit")

    audit = metadata["audit"]
    audit_current = audit["current"]
    audit_history = audit["history"]
    expected_current_audit = {
        "command": metadata["command"],
        "host": metadata["host"],
        "offline": document["fetch"]["offline"],
        "cache_dir": document["fetch"]["cache_dir"],
    }
    if audit_current != expected_current_audit:
        errors.append("metadata.audit.current must match the current report invocation")
    if not audit_history or audit_history[-1] != expected_current_audit:
        errors.append("metadata.audit.history must end with the current report invocation")
    checkpoint = metadata["checkpoint"]
    checkpoint_path = checkpoint["path"]
    if checkpoint["enabled"] != (checkpoint_path is not None):
        errors.append("metadata.checkpoint.enabled must match metadata.checkpoint.path")
    if not checkpoint["enabled"] and checkpoint["resumed"]:
        errors.append("metadata.checkpoint.resumed requires an enabled checkpoint")

    if metadata["probe_count"] != len(document["probes"]):
        errors.append("metadata.probe_count must match probes length")
    probes = document["probes"]
    identities: list[tuple[int, int]] = []
    all_ranges: list[dict[str, Any]] = []
    declared_shards = {item["name"]: item for item in artifact["shards"]}
    for index, probe in enumerate(probes):
        identity = (probe["layer"], probe["expert"])
        identities.append(identity)
        expected_id = f"layer-{probe['layer']:02d}-expert-{probe['expert']:03d}"
        if probe["probe_id"] != expected_id:
            errors.append(f"probes[{index}].probe_id does not match layer/expert")
        if probe["seed"] != metadata["seed"] + index:
            errors.append(f"probes[{index}].seed must be metadata.seed + index")
        if probe["commit"] != metadata["commit"]:
            errors.append(f"probes[{index}].commit must match metadata")
        raw = probe["raw"]
        raw_metadata = raw.get("triad_metadata", {})
        if not isinstance(raw_metadata, dict):
            errors.append(f"probes[{index}].raw.triad_metadata must be an object")
        else:
            if raw_metadata.get("commit") != probe["commit"]:
                errors.append(f"probes[{index}].raw.triad_metadata.commit must match summary")
            for identity_key in ("command", "host"):
                if raw_metadata.get(identity_key) != probe[identity_key]:
                    errors.append(
                        f"probes[{index}].raw.triad_metadata.{identity_key} must match summary"
                    )
        raw_selection = raw.get("selection", {})
        if (
            raw_selection.get("variant") != source["variant"]
            or raw_selection.get("layer") != probe["layer"]
            or raw_selection.get("expert") != probe["expert"]
            or raw_selection.get("seed") != probe["seed"]
        ):
            errors.append(f"probes[{index}].raw.selection identity disagrees with summary")
        raw_fetch = raw.get("fetch", {})
        if raw.get("source") != source:
            errors.append(f"probes[{index}].raw.source must match aggregate source")
        for raw_key, expected in (
            ("schema_version", 1),
            ("evidence_status", "artifact-metadata"),
            ("range_evidence", "measured/artifact-byte"),
            ("validation_class", "H0/no-P4"),
        ):
            if raw.get(raw_key) != expected:
                errors.append(f"probes[{index}].raw.{raw_key} is not bound to the probe contract")
        if raw_fetch.get("transport") != "http-range":
            errors.append(f"probes[{index}].raw.fetch.transport must be http-range")
        if raw_fetch.get("range_count") != 3:
            errors.append(f"probes[{index}].raw.fetch.range_count must be three")
        if raw_fetch.get("full_shard_bytes") != 0:
            errors.append(f"probes[{index}] reports a full-shard read")
        invocation_matches = [
            item
            for item in audit_history
            if item["command"] == probe["command"] and item["host"] == probe["host"]
        ]
        if not any(
            item["offline"] == raw_fetch.get("offline")
            and item["cache_dir"] == raw_fetch.get("cache_dir")
            for item in invocation_matches
        ):
            errors.append(f"probes[{index}] command/host/fetch mode is absent from metadata.audit")
        if raw.get("repeats") != metadata["repeats"]:
            errors.append(f"probes[{index}].raw.repeats must match metadata.repeats")
        if raw.get("warmup") != metadata["warmup"]:
            errors.append(f"probes[{index}].raw.warmup must match metadata.warmup")
        if raw_fetch.get("fetched_bytes") != sum(item["length"] for item in probe["ranges"]):
            errors.append(f"probes[{index}].raw.fetch.fetched_bytes disagrees with ranges")
        if probe["correctness"]["passed"] is not True or probe["correctness"]["finite"] is not True:
            errors.append(f"probes[{index}] correctness did not pass")
        raw_ab = raw.get("ab", {})
        raw_oracle = raw_ab.get("oracle", {})
        if raw_ab.get("independent_oracle") is not True:
            errors.append(f"probes[{index}].raw.ab.independent_oracle must be true")
        for oracle_key, expected in (
            ("name", "gguf-py"),
            ("package", "gguf"),
            ("version", "0.19.0"),
            ("operation", "dequantize + FP32 dense SwiGLU"),
        ):
            if raw_oracle.get(oracle_key) != expected:
                errors.append(f"probes[{index}].raw.ab.oracle.{oracle_key} is not pinned")
        if raw_ab.get("correct") is not True:
            errors.append(f"probes[{index}].raw.ab.correct must be true")
        if probe["correctness"]["passed"] != raw_ab.get("correct"):
            errors.append(f"probes[{index}].correctness.passed must match raw AB")
        if probe["correctness"]["finite"] != raw_ab.get("finite"):
            errors.append(f"probes[{index}].correctness.finite must match raw AB")
        expected_outputs = {
            "oracle": raw_ab.get("oracle", {}).get("output_sha256"),
            "scalar": raw_ab.get("scalar", {}).get("output_sha256"),
            "native": raw_ab.get("avx2", {}).get("output_sha256"),
        }
        if probe["output_hashes"] != expected_outputs:
            errors.append(f"probes[{index}].output_hashes must match raw probe outputs")
        expected_workload_hashes = {
            key: raw_selection.get(key)
            for key in ("hidden_sha256", "expert_ids_sha256", "routing_weights_sha256")
        }
        if probe["workload_hashes"] != expected_workload_hashes:
            errors.append(f"probes[{index}].workload_hashes must match raw selection")
        expected_timing = {
            "scalar_elapsed_ns": raw_ab.get("scalar", {}).get("raw_elapsed_ns"),
            "native_elapsed_ns": raw_ab.get("avx2", {}).get("raw_elapsed_ns"),
        }
        for mode_name, mode in (
            ("scalar", raw_ab.get("scalar", {})),
            ("native", raw_ab.get("avx2", {})),
        ):
            if mode.get("repeats") != metadata["repeats"]:
                errors.append(f"probes[{index}].raw.ab.{mode_name}.repeats must match metadata")
            if mode.get("warmup") != metadata["warmup"]:
                errors.append(f"probes[{index}].raw.ab.{mode_name}.warmup must match metadata")
            if len(mode.get("raw_elapsed_ns", [])) != metadata["repeats"]:
                errors.append(f"probes[{index}].raw.ab.{mode_name} samples must match repeats")
            if len(mode.get("telemetry", [])) != metadata["repeats"]:
                errors.append(f"probes[{index}].raw.ab.{mode_name} telemetry must match repeats")
        for timing_name, expected in expected_timing.items():
            if probe["timing"][timing_name] != expected:
                errors.append(f"probes[{index}].timing.{timing_name} must match raw timing")
        raw_oracle_timing = raw_ab.get("oracle", {}).get("raw_elapsed_ns", {})
        if probe["timing"]["oracle_elapsed_ns"] != {
            "dequantize": raw_oracle_timing.get("dequantize"),
            "dense_expert": raw_oracle_timing.get("dense_expert"),
        }:
            errors.append(f"probes[{index}].timing.oracle_elapsed_ns must match raw timing")
        expected_range_hashes = {item["projection"]: item["sha256"] for item in probe["ranges"]}
        if probe["range_hashes"] != expected_range_hashes:
            errors.append(f"probes[{index}].range_hashes must match range records")
        if tuple(item["projection"] for item in probe["ranges"]) != ("gate", "up", "down"):
            errors.append(f"probes[{index}].ranges must be ordered gate, up, down")
        packed_hashes = raw_oracle.get("packed_source_sha256", {})
        if not isinstance(packed_hashes, dict):
            errors.append(f"probes[{index}].raw.ab.oracle.packed_source_sha256 must be an object")
            packed_hashes = {}
        for item in probe["ranges"]:
            if packed_hashes.get(item["projection"]) != item["sha256"]:
                errors.append(
                    f"probes[{index}] {item['projection']} range SHA must match oracle packed bytes"
                )

        raw_ranges = raw_fetch.get("ranges", [])
        if not isinstance(raw_ranges, list) or len(raw_ranges) != 3:
            errors.append(f"probes[{index}].raw.fetch.ranges must contain three ranges")
            raw_ranges = []
        raw_cache_hits = sum(
            1 for item in raw_ranges if isinstance(item, dict) and item.get("cache") == "hit"
        )
        raw_cache_misses = sum(
            1 for item in raw_ranges if isinstance(item, dict) and item.get("cache") == "miss"
        )
        if raw_fetch.get("cache_hits") != raw_cache_hits:
            errors.append(f"probes[{index}].raw.fetch.cache_hits must match range cache states")
        if raw_fetch.get("cache_misses") != raw_cache_misses:
            errors.append(f"probes[{index}].raw.fetch.cache_misses must match range cache states")
        raw_range_keys = (
            "projection",
            "url",
            "shard",
            "shard_size",
            "declared_shard_sha256",
            "artifact_offset",
            "length",
            "sha256",
            "content_range",
            "cache",
        )
        for range_index, item in enumerate(probe["ranges"]):
            path = f"probes[{index}].ranges[{range_index}]"
            if range_index < len(raw_ranges):
                raw_item = raw_ranges[range_index]
                if any(item.get(key) != raw_item.get(key) for key in raw_range_keys):
                    errors.append(f"{path} disagrees with raw probe range")
            if item["start"] != item["artifact_offset"]:
                errors.append(f"{path}.start must equal artifact_offset")
            if item["end"] != item["start"] + item["length"] - 1:
                errors.append(f"{path}.end must be inclusive start + length - 1")
            if item["start"] + item["length"] > item["shard_size"]:
                errors.append(f"{path} exceeds shard size")
            if item["url"] != f"{source['base_url']}/{item['shard']}":
                errors.append(f"{path}.url must derive from pinned source and shard")
            content_range = _Q3_CONTENT_RANGE.fullmatch(item["content_range"].strip())
            if content_range is None:
                errors.append(f"{path}.content_range is malformed")
            else:
                actual_start, actual_end, actual_total = (
                    int(value) for value in content_range.groups()
                )
                if (actual_start, actual_end, actual_total) != (
                    item["start"],
                    item["end"],
                    item["shard_size"],
                ):
                    errors.append(f"{path}.content_range disagrees with inclusive range")
            shard = declared_shards.get(item["shard"])
            if shard is None:
                errors.append(f"{path} references undeclared shard {item['shard']!r}")
            elif (
                item["shard_size"] != shard["size"]
                or item["declared_shard_sha256"] != shard["sha256"]
            ):
                errors.append(f"{path} shard identity disagrees with artifact manifest")
            all_ranges.append(item)

        raw_layout = raw.get("layout", {})
        if raw_layout.get("top_k") != 2:
            errors.append(f"probes[{index}].raw.layout.top_k must be Qwen top-k 2")
        raw_descriptors = raw_layout.get("descriptors", [])
        if not isinstance(raw_descriptors, list) or len(raw_descriptors) != 3:
            errors.append(f"probes[{index}].raw.layout.descriptors must contain three entries")
            raw_descriptors = []
        descriptor_by_projection = {str(item.get("projection")): item for item in raw_descriptors}
        for descriptor in raw_descriptors:
            if descriptor.get("layer") != probe["layer"]:
                errors.append(f"probes[{index}] raw layout layer disagrees with selection")
            if descriptor.get("selected_expert") != probe["expert"]:
                errors.append(
                    f"probes[{index}] raw layout selected_expert disagrees with selection"
                )
        if probe["quant_names"] != {
            projection: descriptor_by_projection.get(projection, {}).get("quant_name")
            for projection in ("gate", "up", "down")
        }:
            errors.append(f"probes[{index}].quant_names must match raw layout")
        expected_promoted = False
        for projection, item in zip(("gate", "up", "down"), probe["ranges"], strict=True):
            if actual_census is None:
                continue
            expected_name = f"blk.{probe['layer']}.ffn_{projection}_exps.weight"
            records = [
                record
                for record in actual_census.get("tensors", [])
                if record.get("name") == expected_name
            ]
            if len(records) != 1:
                errors.append(f"probes[{index}] census must contain one {expected_name}")
                continue
            record = records[0]
            expected_promoted = expected_promoted or record.get("quant_name") in {"Q5_K", "Q8_0"}
            try:
                experts, output_dim, input_dim = (int(value) for value in record["shape"])
                row_bytes = int(record["row_bytes"])
                tensor_offset = int(record["offset"])
                shard_index = int(record["shard_index"])
                expected_bytes = output_dim * row_bytes
                expected_start = tensor_offset + probe["expert"] * expected_bytes
                expected_shard = actual_census["shards"][shard_index]
            except (KeyError, IndexError, TypeError, ValueError) as error:
                errors.append(f"probes[{index}] census geometry is malformed: {error}")
                continue
            if not 0 <= probe["expert"] < experts:
                errors.append(f"probes[{index}] expert is outside census range")
            expected_end = expected_start + expected_bytes - 1
            for key, expected in (
                ("quant_names", record["quant_name"]),
                ("start", expected_start),
                ("end", expected_end),
                ("length", expected_bytes),
                ("artifact_offset", expected_start),
                ("shard", expected_shard["name"]),
                ("shard_size", declared_shards[expected_shard["name"]]["size"]),
            ):
                if (
                    probe["quant_names"].get(projection) if key == "quant_names" else item[key]
                ) != expected:
                    errors.append(
                        f"probes[{index}] {projection} {key} disagrees with census geometry"
                    )
            descriptor = descriptor_by_projection.get(projection)
            if descriptor is None:
                continue
            descriptor_checks = {
                "quant_name": record["quant_name"],
                "quant_type": int(record["quant_type"]),
                "output_dim": output_dim,
                "input_dim": input_dim,
                "row_stride_bytes": row_bytes,
                "expert_stride_bytes": expected_bytes,
                "tensor_bytes": expected_bytes,
                "artifact_offset": expected_start,
                "artifact_end": expected_start + expected_bytes,
                "num_experts_remapped": 1,
                "rows_per_expert": output_dim,
            }
            for key, expected in descriptor_checks.items():
                if descriptor.get(key) != expected:
                    errors.append(
                        f"probes[{index}] raw layout {projection}.{key} disagrees with census"
                    )
        if raw_selection.get("promoted") is not expected_promoted:
            errors.append(
                f"probes[{index}].raw.selection.promoted flag disagrees with quant family"
            )

        comparison_expectations = {
            "oracle_vs_scalar": ("oracle", "scalar", "gguf-py oracle", "scalar"),
            "oracle_vs_native": ("oracle", "native", "gguf-py oracle", "native executor"),
            "scalar_vs_native": ("scalar", "native", "scalar", "native executor"),
        }
        comparison_names = tuple(comparison_expectations)
        raw_output_hashes = {
            "oracle": raw_oracle.get("output_sha256"),
            "scalar": raw_ab.get("scalar", {}).get("output_sha256"),
            "native": raw_ab.get("avx2", {}).get("output_sha256"),
        }
        for comparison_name, (
            expected_key,
            actual_key,
            expected_name,
            actual_name,
        ) in comparison_expectations.items():
            raw_comparison = raw_ab.get(comparison_name)
            if probe["correctness"].get(comparison_name) != raw_comparison:
                errors.append(
                    f"probes[{index}].correctness.{comparison_name} must match raw comparison"
                )
            if not isinstance(raw_comparison, dict) or raw_comparison.get("correct") is not True:
                errors.append(f"probes[{index}].{comparison_name} must pass")
                continue
            if raw_comparison.get("error") is not None:
                errors.append(f"probes[{index}].{comparison_name} must have a null error")
            if raw_comparison.get("expected") != expected_name:
                errors.append(f"probes[{index}].{comparison_name}.expected is not bound")
            if raw_comparison.get("actual") != actual_name:
                errors.append(f"probes[{index}].{comparison_name}.actual is not bound")
            if raw_comparison.get("expected_output_sha256") != raw_output_hashes[expected_key]:
                errors.append(
                    f"probes[{index}].{comparison_name}.expected_output_sha256 is not bound"
                )
            if raw_comparison.get("actual_output_sha256") != raw_output_hashes[actual_key]:
                errors.append(
                    f"probes[{index}].{comparison_name}.actual_output_sha256 is not bound"
                )
            if (
                raw_comparison.get("shape_match") is not True
                or raw_comparison.get("finite") is not True
            ):
                errors.append(
                    f"probes[{index}].{comparison_name} must have finite matching outputs"
                )
            max_abs = raw_comparison.get("max_abs_error")
            relative_rms = raw_comparison.get("relative_rms_error")
            violation = raw_comparison.get("max_tolerance_violation")
            rtol = raw_comparison.get("rtol")
            atol = raw_comparison.get("atol")
            if not all(
                _finite_number(value) for value in (max_abs, relative_rms, violation, rtol, atol)
            ):
                errors.append(f"probes[{index}].{comparison_name} numeric metrics must be finite")
            elif (
                max_abs < 0 or relative_rms < 0 or raw_comparison.get("correct") != (violation <= 0)
            ):
                errors.append(
                    f"probes[{index}].{comparison_name} numeric predicate is inconsistent"
                )
        if probe["correctness"]["passed"] is not all(
            raw_ab.get(name, {}).get("correct") is True for name in comparison_names
        ):
            errors.append(f"probes[{index}].correctness.passed disagrees with subcomparisons")
        if probe["correctness"]["rtol"] != raw_ab.get("rtol") or probe["correctness"][
            "atol"
        ] != raw_ab.get("atol"):
            errors.append(f"probes[{index}].correctness tolerances must match raw AB")
        if raw.get("oracle") != raw_oracle:
            errors.append(f"probes[{index}].raw.oracle must match raw AB oracle")
        if raw_ab.get("native") != raw_ab.get("avx2"):
            errors.append(f"probes[{index}].raw.ab.native must match raw.ab.avx2")
        if raw_oracle.get("dense_projection_sha256") != raw.get("oracle", {}).get(
            "dense_projection_sha256"
        ):
            errors.append(
                f"probes[{index}].raw oracle dense projection hashes are not self-consistent"
            )
        expected_ab_timing = {
            "scalar_raw_elapsed_ns": raw_ab.get("scalar", {}).get("raw_elapsed_ns"),
            "native_raw_elapsed_ns": raw_ab.get("avx2", {}).get("raw_elapsed_ns"),
            "oracle_raw_elapsed_ns": raw_oracle_timing,
            "comparison_claim": False,
        }
        if raw_ab.get("timing") != expected_ab_timing:
            errors.append(f"probes[{index}].raw.ab.timing must match raw mode timings")
        if raw_ab.get("max_abs_error") != raw_ab.get("scalar_vs_native", {}).get("max_abs_error"):
            errors.append(f"probes[{index}].raw.ab.max_abs_error must match scalar/native error")

        def _raw_kernels(mode: Mapping[str, Any]) -> list[str]:
            return sorted(
                {
                    str(kernel)
                    for telemetry in mode.get("telemetry", [])
                    if isinstance(telemetry, dict)
                    for kernel in telemetry.get("kernel_census", [])
                }
            )

        expected_kernels_by_mode = {
            "scalar": _raw_kernels(raw_ab.get("scalar", {})),
            "native": _raw_kernels(raw_ab.get("avx2", {})),
        }
        for mode_name, mode, expected_request in (
            ("scalar", raw_ab.get("scalar", {}), "forced_scalar"),
            ("native", raw_ab.get("avx2", {}), "forced_avx2"),
        ):
            mode_path = f"probes[{index}].raw.ab.{mode_name}"
            if mode.get("requested_mode") != expected_request:
                errors.append(f"{mode_path}.requested_mode is not bound to the probe mode")
            selected_backend = mode.get("selected_backend")
            telemetry = mode.get("telemetry", [])
            telemetry_backends = {
                item.get("backend")
                for item in telemetry
                if isinstance(item, dict) and isinstance(item.get("backend"), str)
            }
            if not isinstance(selected_backend, str) or not telemetry_backends:
                errors.append(f"{mode_path} is missing selected backend telemetry")
            elif telemetry_backends != {selected_backend}:
                errors.append(f"{mode_path}.selected_backend must match telemetry backends")
            isa_values = (mode.get("q4k_isa"), mode.get("mixed_isa"))
            if any(value not in {"scalar", "avx2"} for value in isa_values):
                errors.append(f"{mode_path} ISA selection is malformed")
            fallback_values = (
                mode.get("q4k_fallback_reason"),
                mode.get("mixed_fallback_reason"),
            )
            if any(value is not None and not isinstance(value, str) for value in fallback_values):
                errors.append(f"{mode_path} fallback selection is malformed")
            allowed_fallbacks = {*fallback_values, None}
            if any(
                isinstance(item, dict)
                and item.get("fallback_reason") not in allowed_fallbacks
                and not (
                    isinstance(item.get("fallback_reason"), str)
                    and item["fallback_reason"].startswith("reference_")
                )
                for item in telemetry
            ):
                errors.append(f"{mode_path} telemetry fallback is not bound to selected fallback")
            observed_kernels = expected_kernels_by_mode[mode_name]
            actual_avx2 = bool(
                (isinstance(selected_backend, str) and "avx2" in selected_backend)
                or any("avx2" in kernel for kernel in observed_kernels)
            )
            if mode.get("actual_avx2") != actual_avx2:
                errors.append(f"{mode_path}.actual_avx2 must match backend/kernel census")
            if selected_backend != "reference" or actual_avx2:
                errors.append(f"{mode_path} violates the reference-only backend contract")
            if any("avx2" in kernel for kernel in observed_kernels):
                errors.append(f"{mode_path} reference-only kernel census contains AVX2")
            if selected_backend == "reference" and any(
                not kernel.startswith("reference") for kernel in observed_kernels
            ):
                errors.append(f"{mode_path} reference backend has a non-reference kernel census")
        if probe["kernel_census_by_mode"] != expected_kernels_by_mode:
            errors.append(f"probes[{index}].kernel_census_by_mode must match raw telemetry")
        if probe["kernel_census"] != sorted(set().union(*expected_kernels_by_mode.values())):
            errors.append(f"probes[{index}].kernel_census must match raw telemetry")
        if not all(expected_kernels_by_mode.values()):
            errors.append(f"probes[{index}] raw telemetry is missing kernel census")
    if len(set(identities)) != len(identities):
        errors.append("Q3 triad probe layer/expert identities must be unique")
    aggregate_ranges = document["fetch"]["ranges"]
    if aggregate_ranges != all_ranges:
        errors.append("fetch.ranges must equal the ordered per-probe ranges")
    for index, item in enumerate(aggregate_ranges):
        if item["start"] != item["artifact_offset"]:
            errors.append(f"fetch.ranges[{index}].start must equal artifact_offset")
        if item["end"] != item["start"] + item["length"] - 1:
            errors.append(f"fetch.ranges[{index}].end must be inclusive start + length - 1")
        content_range = _Q3_CONTENT_RANGE.fullmatch(item["content_range"].strip())
        if content_range is not None:
            actual_start, actual_end, actual_total = (
                int(value) for value in content_range.groups()
            )
            if (actual_start, actual_end, actual_total) != (
                item["start"],
                item["end"],
                item["shard_size"],
            ):
                errors.append(f"fetch.ranges[{index}].content_range disagrees with range")
    if document["fetch"]["range_count"] != len(aggregate_ranges):
        errors.append("fetch.range_count must match fetch.ranges length")
    if document["fetch"]["fetched_bytes"] != sum(item["length"] for item in aggregate_ranges):
        errors.append("fetch.fetched_bytes must match range lengths")
    if document["fetch"]["full_shard_bytes"] != 0:
        errors.append("fetch.full_shard_bytes must remain zero")
    expected_cache_hits = sum(int(probe["raw"]["fetch"].get("cache_hits", 0)) for probe in probes)
    expected_cache_misses = sum(
        int(probe["raw"]["fetch"].get("cache_misses", 0)) for probe in probes
    )
    if document["fetch"]["cache_hits"] != expected_cache_hits:
        errors.append("fetch.cache_hits must equal raw probe cache hits")
    if document["fetch"]["cache_misses"] != expected_cache_misses:
        errors.append("fetch.cache_misses must equal raw probe cache misses")
    return errors


def _qwen38_dual_p4_semantic_errors(document: dict[str, Any]) -> list[str]:
    """Validate identity bindings that JSON Schema cannot express across arrays."""
    errors: list[str] = []
    inventory = document["hardware_inventory"]
    inventory_gpus = inventory["gpu_identities"]
    devices = document["devices"]
    profile = inventory["profile_id"]

    for label, records in (
        ("hardware_inventory.gpu_identities", inventory_gpus),
        ("devices", devices),
    ):
        uuids = [record["uuid"] for record in records]
        buses = [record["pci_bus_id"] for record in records]
        if len(uuids) != len(set(uuids)):
            errors.append(f"{label} UUIDs must be unique")
        if len(buses) != len(set(buses)):
            errors.append(f"{label} PCI bus IDs must be unique")

    inventory_by_index = {record["index"]: record for record in inventory_gpus}
    for index, record in enumerate(inventory_gpus):
        if record["ecc_profile"] != profile:
            errors.append(
                f"hardware_inventory.gpu_identities[{index}].ecc_profile must match profile_id"
            )
    for index, device in enumerate(devices):
        expected = inventory_by_index.get(device["index"])
        if expected is None:
            errors.append(f"devices[{index}].index must identify a bound inventory GPU")
            continue
        for field in ("uuid", "pci_bus_id", "pci_root", "numa_node"):
            if device[field] != expected[field]:
                errors.append(f"devices[{index}].{field} must match the bound inventory GPU")
        if device["ecc_profile"] != profile:
            errors.append(f"devices[{index}].ecc_profile must match hardware_inventory.profile_id")
        if device["ecc_profile"] != expected["ecc_profile"]:
            errors.append(f"devices[{index}].ecc_profile must match the inventory GPU profile")
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
    elif schema_name == "qwen38-real-artifact-target-cpu-benchmark.schema.json":
        errors.extend(_target_cpu_benchmark_semantic_errors(document))
    elif schema_name == "qwen38-q3-triad.schema.json":
        errors.extend(_qwen38_q3_triad_semantic_errors(document, schema_dir=schema_dir))
    elif schema_name == "qwen38-dual-p4-device-evidence.schema.json":
        errors.extend(_qwen38_dual_p4_semantic_errors(document))
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
