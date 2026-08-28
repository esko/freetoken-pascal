"""Torch-free Qwen3.8 reference-correctness evidence contracts.

Observation bundles are deliberately small, semantic snapshots rather than model
weights. Arrays use NumPy's non-pickle format inside a validated ZIP container so
independent runtimes can exchange router, state, PLE, and logit observations.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTITY_FIELDS = frozenset(
    {
        "implementation",
        "revision",
        "commit",
        "artifact_sha256",
        "quant_census_sha256",
        "corpus_sha256",
        "prompt_id",
        "prompt_sha256",
        "quantization",
        "dtype",
        "cache_mode",
        "execution_mode",
        "context_tokens",
    }
)
_WORKLOAD_FIELDS = (
    "artifact_sha256",
    "quant_census_sha256",
    "corpus_sha256",
    "prompt_id",
    "prompt_sha256",
    "quantization",
    "cache_mode",
    "context_tokens",
)
_REQUIRED_CATEGORIES = frozenset(
    {"factual", "code", "tool-json", "repetitive-edit", "long-retrieval"}
)
_REQUIRED_CONTEXT_TARGETS = frozenset({32768, 128000, 262000})
_TYPE_NAMES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


@dataclass(frozen=True)
class Tolerance:
    max_abs: float
    relative_rms: float
    min_cosine: float

    def __post_init__(self) -> None:
        values = (self.max_abs, self.relative_rms, self.min_cosine)
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in values
        ):
            raise ValueError("numeric tolerances must be finite numbers")
        if self.max_abs < 0 or self.relative_rms < 0:
            raise ValueError("error tolerances must be non-negative")
        if not -1 <= self.min_cosine <= 1:
            raise ValueError("min_cosine must be in [-1, 1]")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(document: Any) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r} is forbidden")


def _strict_json_loads(data: str | bytes) -> Any:
    return json.loads(data, parse_constant=_reject_json_constant)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _validate_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError("observation identity must be an object")
    missing = _IDENTITY_FIELDS - set(identity)
    extra = set(identity) - _IDENTITY_FIELDS
    if missing or extra:
        raise ValueError(
            "observation identity fields disagree: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for key in (
        "implementation",
        "revision",
        "prompt_id",
        "quantization",
        "dtype",
        "cache_mode",
        "execution_mode",
    ):
        if not isinstance(identity[key], str) or not identity[key]:
            raise ValueError(f"identity {key} must be a non-empty string")
    if not isinstance(identity["commit"], str) or not _COMMIT_RE.fullmatch(identity["commit"]):
        raise ValueError("identity commit must be a 40-character lowercase hex revision")
    if not isinstance(identity["revision"], str) or not (
        _COMMIT_RE.fullmatch(identity["revision"])
        or re.fullmatch(r"fixture-[a-z0-9-]+", identity["revision"])
    ):
        raise ValueError("identity revision must be an immutable commit or fixture revision")
    for key in (
        "artifact_sha256",
        "quant_census_sha256",
        "corpus_sha256",
        "prompt_sha256",
    ):
        if not isinstance(identity[key], str) or not _SHA256_RE.fullmatch(identity[key]):
            raise ValueError(f"identity {key} must be a SHA-256 digest")
    if (
        not isinstance(identity["context_tokens"], int)
        or isinstance(identity["context_tokens"], bool)
        or identity["context_tokens"] < 0
    ):
        raise ValueError("identity context_tokens must be a non-negative integer")
    return dict(identity)


def _array_payload(name: str, value: Any) -> tuple[np.ndarray, bytes]:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"invalid observation name {name!r}")
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"observation {name} cannot use object/pickle dtype")
    if array.size == 0:
        raise ValueError(f"observation {name} cannot be empty")
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        raise ValueError(f"observation {name} must contain only finite values")
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return array, buffer.getvalue()


def write_observation_bundle(
    path: str | Path,
    identity: dict[str, Any],
    observations: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Write a deterministic semantic observation bundle without pickle data."""
    checked_identity = _validate_identity(identity)
    if not isinstance(observations, dict) or not observations:
        raise ValueError("observation bundle must contain at least one array")
    payloads: dict[str, bytes] = {}
    records: dict[str, dict[str, Any]] = {}
    for name in sorted(observations):
        array, payload = _array_payload(name, observations[name])
        member = f"arrays/{name}.npy"
        payloads[member] = payload
        records[name] = {
            "member": member,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
            "payload_bytes": len(payload),
            "sha256": _sha256(payload),
        }
    manifest = {
        "schema_name": "qwen38-observation-bundle",
        "schema_version": 1,
        "identity": checked_identity,
        "observations": records,
    }
    target = Path(path)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(_zip_info("manifest.json"), _canonical_json(manifest))
        for member, payload in payloads.items():
            archive.writestr(_zip_info(member), payload)
    return manifest


def read_observation_bundle(
    path: str | Path,
    *,
    maximum_payload_bytes: int = 256 << 20,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read and fully validate a semantic observation bundle."""
    if (
        not isinstance(maximum_payload_bytes, int)
        or isinstance(maximum_payload_bytes, bool)
        or maximum_payload_bytes <= 0
    ):
        raise ValueError("maximum_payload_bytes must be a positive integer")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if names.count("manifest.json") != 1 or len(names) != len(set(names)):
                raise ValueError("corrupt observation bundle member list")
            manifest_info = archive.getinfo("manifest.json")
            if (
                manifest_info.compress_type != zipfile.ZIP_STORED
                or manifest_info.file_size <= 0
                or manifest_info.file_size > 1 << 20
            ):
                raise ValueError("corrupt observation bundle manifest metadata")
            manifest = _strict_json_loads(archive.read("manifest.json"))
            if not isinstance(manifest, dict):
                raise ValueError("corrupt observation bundle manifest")
            if (
                manifest.get("schema_name") != "qwen38-observation-bundle"
                or manifest.get("schema_version") != 1
            ):
                raise ValueError("unsupported observation bundle schema")
            if set(manifest) != {"schema_name", "schema_version", "identity", "observations"}:
                raise ValueError("observation bundle manifest has unknown fields")
            identity = _validate_identity(manifest["identity"])
            records = manifest["observations"]
            if not isinstance(records, dict) or not records:
                raise ValueError("observation bundle has no observation records")
            expected_members = {"manifest.json"}
            arrays: dict[str, np.ndarray] = {}
            total_payload = 0
            for name, record in sorted(records.items()):
                if not isinstance(record, dict) or set(record) != {
                    "member",
                    "dtype",
                    "shape",
                    "nbytes",
                    "payload_bytes",
                    "sha256",
                }:
                    raise ValueError(f"corrupt observation record {name!r}")
                member = f"arrays/{name}.npy"
                if record["member"] != member or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                    raise ValueError(f"corrupt observation member for {name!r}")
                if (
                    not isinstance(record["member"], str)
                    or not isinstance(record["dtype"], str)
                    or not isinstance(record["sha256"], str)
                    or not _SHA256_RE.fullmatch(record["sha256"])
                    or not isinstance(record["shape"], list)
                    or any(
                        not isinstance(dimension, int)
                        or isinstance(dimension, bool)
                        or dimension < 0
                        for dimension in record["shape"]
                    )
                    or any(
                        not isinstance(record[key], int)
                        or isinstance(record[key], bool)
                        or record[key] <= 0
                        for key in ("nbytes", "payload_bytes")
                    )
                ):
                    raise ValueError(f"corrupt observation metadata for {name!r}")
                expected_members.add(member)
                payload_size = record["payload_bytes"]
                total_payload += payload_size
                if payload_size <= 0 or total_payload > maximum_payload_bytes:
                    raise ValueError("observation bundle payload exceeds configured bound")
                member_info = archive.getinfo(member)
                if (
                    member_info.compress_type != zipfile.ZIP_STORED
                    or member_info.file_size != payload_size
                ):
                    raise ValueError(f"observation {name} archive metadata is invalid")
                payload = archive.read(member)
                if len(payload) != payload_size or _sha256(payload) != record["sha256"]:
                    raise ValueError(f"observation {name} payload digest mismatch")
                array = np.load(io.BytesIO(payload), allow_pickle=False)
                if array.dtype.hasobject:
                    raise ValueError(f"observation {name} contains forbidden pickle dtype")
                if array.size == 0:
                    raise ValueError(f"observation {name} cannot be empty")
                if array.dtype.str != record["dtype"] or list(array.shape) != record["shape"]:
                    raise ValueError(f"observation {name} dtype/shape disagrees with manifest")
                if int(array.nbytes) != record["nbytes"]:
                    raise ValueError(f"observation {name} byte count disagrees with manifest")
                if array.dtype.kind in "fc" and not np.isfinite(array).all():
                    raise ValueError(f"observation {name} must contain only finite values")
                array.flags.writeable = False
                arrays[name] = array
            if set(names) != expected_members:
                raise ValueError("corrupt observation bundle contains undeclared members")
            return identity, arrays
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        zipfile.BadZipFile,
    ) as error:
        raise ValueError(f"corrupt observation bundle: {error}") from error


def _numeric_metrics(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float, float]:
    lhs = actual.astype(np.float64, copy=False).reshape(-1)
    rhs = expected.astype(np.float64, copy=False).reshape(-1)
    delta = lhs - rhs
    max_abs = float(np.max(np.abs(delta)))
    reference_rms = float(np.sqrt(np.mean(rhs * rhs)))
    relative_rms = float(np.sqrt(np.mean(delta * delta)) / max(reference_rms, 1e-30))
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm == 0 or rhs_norm == 0:
        cosine = 1.0 if np.array_equal(lhs, rhs) else 0.0
    else:
        cosine = float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm))
        cosine = min(1.0, max(-1.0, cosine))
    return max_abs, relative_rms, cosine


def compare_observation_bundles(
    subject_path: str | Path,
    reference_path: str | Path,
    *,
    tolerances: dict[str, Tolerance],
    exact_observations: set[str],
    require_independent: bool = True,
    evidence_status: str = "synthetic",
) -> dict[str, Any]:
    """Compare the same model/quant/workload without permitting substitution."""
    subject_identity, subject = read_observation_bundle(subject_path)
    reference_identity, reference = read_observation_bundle(reference_path)
    for key in _WORKLOAD_FIELDS:
        if subject_identity[key] != reference_identity[key]:
            raise ValueError(
                f"subject/reference {key} mismatch: "
                f"{subject_identity[key]!r} != {reference_identity[key]!r}"
            )
    if (
        require_independent
        and subject_identity["implementation"] == reference_identity["implementation"]
    ):
        raise ValueError("correctness evidence requires an independent reference implementation")
    if require_independent and all(
        subject_identity[key] == reference_identity[key] for key in ("revision", "commit")
    ):
        raise ValueError("independent references cannot use the same revision and commit")
    if evidence_status not in {"synthetic", "measured"}:
        raise ValueError("evidence_status must be synthetic or measured")
    if evidence_status == "measured" and subject_identity["commit"] == "0" * 40:
        raise ValueError("measured evidence cannot use the placeholder commit")
    if set(subject) != set(reference):
        raise ValueError(
            f"subject/reference observation sets differ: {sorted(subject)} != {sorted(reference)}"
        )
    contracted = set(tolerances) | set(exact_observations)
    if set(subject) != contracted:
        raise ValueError(
            "comparison contract must classify every observation exactly once: "
            f"observations={sorted(subject)}, contracted={sorted(contracted)}"
        )
    overlap = set(tolerances) & set(exact_observations)
    if overlap:
        raise ValueError(f"observations cannot be both numeric and exact: {sorted(overlap)}")

    comparisons: list[dict[str, Any]] = []
    for name in sorted(subject):
        actual = subject[name]
        expected = reference[name]
        if actual.shape != expected.shape:
            raise ValueError(
                f"observation {name} shape mismatch: {actual.shape} != {expected.shape}"
            )
        if name in exact_observations:
            passed = bool(np.array_equal(actual, expected))
            comparisons.append(
                {
                    "observation": name,
                    "metric": "array_exact",
                    "observed": passed,
                    "limit": True,
                    "passed": passed,
                }
            )
            continue
        if actual.dtype.kind not in "fiu" or expected.dtype.kind not in "fiu":
            raise ValueError(f"numeric observation {name} has unsupported dtype")
        tolerance = tolerances[name]
        max_abs, relative_rms, cosine = _numeric_metrics(actual, expected)
        metric_values = (
            ("max_abs", max_abs, tolerance.max_abs, max_abs <= tolerance.max_abs),
            (
                "relative_rms",
                relative_rms,
                tolerance.relative_rms,
                relative_rms <= tolerance.relative_rms,
            ),
            ("cosine", cosine, tolerance.min_cosine, cosine >= tolerance.min_cosine),
        )
        for metric, observed, limit, passed in metric_values:
            comparisons.append(
                {
                    "observation": name,
                    "metric": metric,
                    "observed": observed,
                    "limit": limit,
                    "passed": bool(passed),
                }
            )

    subject_summary = {
        key: subject_identity[key]
        for key in (
            "implementation",
            "revision",
            "commit",
            "artifact_sha256",
            "quant_census_sha256",
            "quantization",
            "dtype",
            "cache_mode",
            "execution_mode",
            "corpus_sha256",
            "prompt_id",
            "prompt_sha256",
            "context_tokens",
        )
    }
    reference_summary = {
        key: reference_identity[key]
        for key in (
            "implementation",
            "revision",
            "commit",
            "artifact_sha256",
            "quant_census_sha256",
            "quantization",
            "dtype",
            "cache_mode",
            "execution_mode",
            "corpus_sha256",
            "prompt_id",
            "prompt_sha256",
            "context_tokens",
        )
    }
    return {
        "schema_name": "correctness-evidence.schema.json",
        "schema_version": 2,
        "evidence_status": evidence_status,
        "commit": subject_identity["commit"],
        "subject": subject_summary,
        "reference": reference_summary,
        "workload": {
            "corpus_sha256": subject_identity["corpus_sha256"],
            "prompt_id": subject_identity["prompt_id"],
            "prompt_sha256": subject_identity["prompt_sha256"],
            "context_tokens": subject_identity["context_tokens"],
        },
        "comparisons": comparisons,
        "passed": all(comparison["passed"] for comparison in comparisons),
    }


def _validate_expectation(expectation: Any, *, label: str) -> list[str]:
    if not isinstance(expectation, dict):
        return [f"{label} expectation must be an object"]
    kind = expectation.get("kind")
    if kind in {"contains", "exact"}:
        if set(expectation) != {"kind", "value"} or not isinstance(expectation.get("value"), str):
            return [f"{label} {kind} expectation must contain one string value"]
        return []
    if kind == "json-object":
        required = expectation.get("required")
        if set(expectation) != {"kind", "required"} or not isinstance(required, dict):
            return [f"{label} json-object expectation must contain required fields"]
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(type_name, str)
            or type_name not in _TYPE_NAMES
            for key, type_name in required.items()
        ):
            return [f"{label} json-object expectation has invalid field types"]
        return []
    return [f"{label} has unsupported expectation kind {kind!r}"]


def _validate_corpus(document: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "corpus_id",
        "tokenizer",
        "sampling",
        "cases",
    }:
        return ["prompt corpus root fields are invalid"]
    if (
        not isinstance(document["schema_version"], int)
        or isinstance(document["schema_version"], bool)
        or document["schema_version"] != 1
    ):
        errors.append("prompt corpus schema_version must be 1")
    if not isinstance(document["corpus_id"], str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]*", document["corpus_id"]
    ):
        errors.append("prompt corpus corpus_id is invalid")
    tokenizer = document["tokenizer"]
    if not isinstance(tokenizer, dict) or set(tokenizer) != {"repository", "revision"}:
        errors.append("prompt corpus tokenizer identity is invalid")
    elif (
        not isinstance(tokenizer["repository"], str)
        or not tokenizer["repository"]
        or not isinstance(tokenizer["revision"], str)
        or not _COMMIT_RE.fullmatch(tokenizer["revision"])
    ):
        errors.append("prompt corpus tokenizer must have a repository and immutable commit")
    sampling = document["sampling"]
    if (
        not isinstance(sampling, dict)
        or set(sampling) != {"temperature", "top_p", "seed"}
        or not isinstance(sampling.get("temperature"), (int, float))
        or isinstance(sampling.get("temperature"), bool)
        or not math.isfinite(sampling.get("temperature"))
        or sampling.get("temperature") != 0.0
        or not isinstance(sampling.get("top_p"), (int, float))
        or isinstance(sampling.get("top_p"), bool)
        or not math.isfinite(sampling.get("top_p"))
        or sampling.get("top_p") != 1.0
        or not isinstance(sampling.get("seed"), int)
        or isinstance(sampling.get("seed"), bool)
        or sampling.get("seed") < 0
    ):
        errors.append("prompt corpus sampling must use temperature=0, top_p=1, and a seed")
    cases = document["cases"]
    if not isinstance(cases, list) or not cases:
        return [*errors, "prompt corpus cases must be a non-empty array"]
    ids: set[str] = set()
    categories: set[str] = set()
    context_targets: set[int] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"case {index} must be an object")
            continue
        if not {"id", "category", "messages", "expectation"} <= set(case) or not set(case) <= {
            "id",
            "category",
            "messages",
            "expectation",
            "context",
        }:
            errors.append(f"case {index} fields are invalid")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", case_id):
            errors.append(f"case {index} has invalid id")
        elif case_id in ids:
            errors.append(f"duplicate case id {case_id}")
        else:
            ids.add(case_id)
        category = case.get("category")
        if isinstance(category, str) and category in _REQUIRED_CATEGORIES:
            categories.add(category)
        else:
            errors.append(f"case {case_id} has invalid category")
        messages = case.get("messages")
        if not isinstance(messages, list) or not messages:
            errors.append(f"case {case_id} has no messages")
        elif any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in {"system", "user", "assistant"}
            or not isinstance(message["content"], str)
            for message in messages
        ):
            errors.append(f"case {case_id} has invalid messages")
        expectation = case.get("expectation")
        errors.extend(_validate_expectation(expectation, label=f"case {case_id}"))
        context = case.get("context")
        if context is not None:
            if (
                not isinstance(context, dict)
                or set(context) != {"generator", "target_tokens", "needle_fraction", "needle"}
                or context.get("generator") != "numbered-records-v1"
                or not isinstance(context.get("target_tokens"), int)
                or isinstance(context.get("target_tokens"), bool)
                or context.get("target_tokens") <= 0
                or not isinstance(context.get("needle_fraction"), (int, float))
                or isinstance(context.get("needle_fraction"), bool)
                or not math.isfinite(context.get("needle_fraction"))
                or not 0 < context.get("needle_fraction") < 1
                or not isinstance(context.get("needle"), str)
                or not context.get("needle")
            ):
                errors.append(f"case {case_id} has invalid context generator")
            else:
                context_targets.add(context["target_tokens"])
            if isinstance(messages, list) and not any(
                "{{CONTEXT}}" in message.get("content", "")
                for message in messages
                if isinstance(message, dict)
            ):
                errors.append(f"case {case_id} context is not inserted into a message")
    if not _REQUIRED_CATEGORIES <= categories:
        errors.append(
            f"prompt corpus misses categories {sorted(_REQUIRED_CATEGORIES - categories)}"
        )
    if not _REQUIRED_CONTEXT_TARGETS <= context_targets:
        missing_targets = sorted(_REQUIRED_CONTEXT_TARGETS - context_targets)
        errors.append(f"prompt corpus misses context targets {missing_targets}")
    return errors


def load_prompt_corpus(path: str | Path) -> dict[str, Any]:
    try:
        document = _strict_json_loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"unable to load prompt corpus: {error}") from error
    errors = _validate_corpus(document)
    if errors:
        raise ValueError("invalid prompt corpus: " + "; ".join(errors))
    return document


def validate_probe_output(case: dict[str, Any], output: str) -> list[str]:
    """Validate syntax-level corpus expectations without claiming model correctness."""
    expectation = case.get("expectation", {})
    contract_errors = _validate_expectation(expectation, label="probe")
    if contract_errors:
        raise ValueError("; ".join(contract_errors))
    kind = expectation.get("kind")
    if kind == "contains":
        return [] if expectation.get("value") in output else ["output misses required value"]
    if kind == "exact":
        return [] if expectation.get("value") == output else ["output is not exact"]
    if kind != "json-object":
        return [f"unsupported probe expectation {kind!r}"]
    try:
        value = _strict_json_loads(output)
    except (json.JSONDecodeError, ValueError) as error:
        return [f"output is not valid strict JSON: {error}"]
    if not isinstance(value, dict):
        return ["JSON probe output must be one object"]
    errors = []
    for key, type_name in expectation.get("required", {}).items():
        expected_type = _TYPE_NAMES.get(type_name)
        if expected_type is None:
            errors.append(f"probe declares unknown type {type_name!r}")
            continue
        candidate = value.get(key)
        if (
            key not in value
            or not isinstance(candidate, expected_type)
            or (type_name in {"integer", "number"} and isinstance(candidate, bool))
        ):
            errors.append(f"JSON key {key!r} must have type {type_name}")
    return errors


def numbered_record_context(
    record_count: int,
    *,
    needle_fraction: float,
    needle: str,
) -> str:
    """Build deterministic numbered records with one explicit retrieval needle."""
    if not isinstance(record_count, int) or isinstance(record_count, bool) or record_count <= 0:
        raise ValueError("record_count must be a positive integer")
    if (
        not isinstance(needle_fraction, (int, float))
        or isinstance(needle_fraction, bool)
        or not math.isfinite(needle_fraction)
        or not 0 < needle_fraction < 1
    ):
        raise ValueError("needle_fraction must be finite and in (0, 1)")
    if not isinstance(needle, str) or not needle:
        raise ValueError("needle must be a non-empty string")
    needle_index = min(record_count - 1, int(record_count * needle_fraction))
    records = []
    for index in range(record_count):
        payload = needle if index == needle_index else f"filler-{index:08d}"
        records.append(f"record {index:08d}: {payload}")
    return "\n".join(records)


def prompt_corpus_sha256(document: dict[str, Any]) -> str:
    errors = _validate_corpus(document)
    if errors:
        raise ValueError("invalid prompt corpus: " + "; ".join(errors))
    return _sha256(_canonical_json(document))


def materialize_prompt_case(
    corpus: dict[str, Any],
    case_id: str,
    *,
    context_text: str | None = None,
    token_counter: Any | None = None,
) -> dict[str, Any]:
    """Materialize a case and prove the tokenizer-specific target token count."""
    errors = _validate_corpus(corpus)
    if errors:
        raise ValueError("invalid prompt corpus: " + "; ".join(errors))
    matches = [case for case in corpus["cases"] if case["id"] == case_id]
    if not matches:
        raise ValueError(f"unknown prompt case {case_id!r}")
    case = matches[0]
    context = case.get("context")
    if context is None and context_text is not None:
        raise ValueError("context_text was supplied for a case without a context generator")
    if context is not None:
        if context_text is None or token_counter is None or not callable(token_counter):
            raise ValueError("context cases require context_text and a callable token_counter")
        if not isinstance(context_text, str) or not context_text:
            raise ValueError("context_text must be a non-empty string")
    messages = [dict(message) for message in case["messages"]]
    if context is not None:
        messages = [
            {
                "role": message["role"],
                "content": message["content"].replace("{{CONTEXT}}", context_text),
            }
            for message in messages
        ]
        observed_tokens = token_counter(messages)
        if (
            not isinstance(observed_tokens, int)
            or isinstance(observed_tokens, bool)
            or observed_tokens != context["target_tokens"]
        ):
            raise ValueError(
                "materialized prompt token count mismatch: "
                f"expected {context['target_tokens']}, observed {observed_tokens!r}"
            )
    else:
        observed_tokens = token_counter(messages) if callable(token_counter) else 0
        if not isinstance(observed_tokens, int) or isinstance(observed_tokens, bool):
            raise ValueError("token_counter must return an integer")
    return {
        "case_id": case_id,
        "messages": messages,
        "prompt_sha256": _sha256(_canonical_json(messages)),
        "context_tokens": observed_tokens,
    }


__all__ = [
    "Tolerance",
    "compare_observation_bundles",
    "load_prompt_corpus",
    "materialize_prompt_case",
    "numbered_record_context",
    "prompt_corpus_sha256",
    "read_observation_bundle",
    "validate_probe_output",
    "write_observation_bundle",
]
