"""H0 contracts for long-horizon semantic and sensitive-state evidence.

This module only validates deterministic evidence supplied by a runtime.  It does not
load model weights, run CUDA, or turn synthetic fixtures into model-quality claims.
The contract keeps semantic probes and per-step internal observations together so a
small state-control drift cannot be hidden by fluent short-prompt output.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from freetoken.reference_correctness import (
    Tolerance,
    compare_observation_bundles,
    read_observation_bundle,
    validate_probe_output,
)
from freetoken.sensitive_census import evaluate_sensitive_tensor_fixture

_PROBE_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_CONTRACT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_PROBE_KINDS = frozenset(
    {
        "multi-turn-coding",
        "repeated-tool-calls",
        "state-dependent-reasoning",
        "structured-transform",
        "long-generation",
    }
)
_REQUIRED_PROBE_KINDS = frozenset(_PROBE_KINDS)
_OBSERVATION_NAMES = frozenset(
    {"continuation_tokens", "router_ids", "semantic_output_tokens", "gdn_state"}
)
_COMPARISONS = frozenset({"exact", "numeric"})
_PERTURBATION_FIELDS = frozenset({"perturbation", "scale_mutation"})
_TYPE_NAMES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r} is forbidden")


def _strict_json_loads(data: str) -> Any:
    return json.loads(data, parse_constant=_reject_json_constant)


def _expectation_errors(expectation: Any, *, label: str) -> list[str]:
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


def _validate_contract(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("long-horizon contract must be an object")
    expected_root = {
        "schema_name",
        "schema_version",
        "contract_id",
        "minimum_steps",
        "probes",
        "observations",
        "sensitive_control",
    }
    if set(document) != expected_root:
        raise ValueError(
            "long-horizon contract fields disagree: "
            f"expected={sorted(expected_root)}, actual={sorted(document)}"
        )
    if document["schema_name"] != "qwen38-long-horizon-contract" or document["schema_version"] != 1:
        raise ValueError("unsupported long-horizon contract schema")
    if not isinstance(document["contract_id"], str) or not _CONTRACT_ID_RE.fullmatch(
        document["contract_id"]
    ):
        raise ValueError("long-horizon contract_id is invalid")
    minimum_steps = document["minimum_steps"]
    if not isinstance(minimum_steps, int) or isinstance(minimum_steps, bool) or minimum_steps < 2:
        raise ValueError("long-horizon minimum_steps must be an integer >= 2")

    probes = document["probes"]
    if not isinstance(probes, list) or not probes:
        raise ValueError("long-horizon probes must be a non-empty array")
    probe_ids: set[str] = set()
    probe_kinds: set[str] = set()
    for index, probe in enumerate(probes):
        if not isinstance(probe, dict) or set(probe) != {"id", "kind", "steps"}:
            raise ValueError(f"long-horizon probe {index} fields are invalid")
        probe_id = probe["id"]
        kind = probe["kind"]
        if not isinstance(probe_id, str) or not _PROBE_ID_RE.fullmatch(probe_id):
            raise ValueError(f"long-horizon probe {index} id is invalid")
        if probe_id in probe_ids:
            raise ValueError(f"long-horizon probe id is duplicated: {probe_id}")
        probe_ids.add(probe_id)
        if kind not in _PROBE_KINDS:
            raise ValueError(f"long-horizon probe {probe_id} kind is unsupported: {kind!r}")
        probe_kinds.add(kind)
        steps = probe["steps"]
        if not isinstance(steps, list) or len(steps) < minimum_steps:
            raise ValueError(
                f"long-horizon probe {probe_id} needs at least the minimum horizon "
                f"of {minimum_steps} steps"
            )
        step_ids: set[str] = set()
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict) or set(step) != {"id", "expectation"}:
                raise ValueError(f"long-horizon probe {probe_id} step {step_index} is invalid")
            step_id = step["id"]
            if not isinstance(step_id, str) or not _PROBE_ID_RE.fullmatch(step_id):
                raise ValueError(f"long-horizon probe {probe_id} step id is invalid")
            if step_id in step_ids:
                raise ValueError(f"long-horizon probe {probe_id} step id is duplicated")
            step_ids.add(step_id)
            errors = _expectation_errors(
                step["expectation"], label=f"probe {probe_id} step {step_id}"
            )
            if errors:
                raise ValueError("; ".join(errors))
    if not _REQUIRED_PROBE_KINDS <= probe_kinds:
        missing = sorted(_REQUIRED_PROBE_KINDS - probe_kinds)
        raise ValueError(f"long-horizon contract misses probe kinds {missing}")

    observations = document["observations"]
    if not isinstance(observations, dict) or set(observations) != _OBSERVATION_NAMES:
        raise ValueError(
            f"long-horizon observations must declare exactly {sorted(_OBSERVATION_NAMES)}"
        )
    for name, descriptor in observations.items():
        if not isinstance(descriptor, dict):
            raise ValueError(f"long-horizon observation {name} must be an object")
        comparison = descriptor.get("comparison")
        minimum = descriptor.get("minimum_steps")
        if comparison not in _COMPARISONS:
            raise ValueError(f"long-horizon observation {name} comparison is invalid")
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < minimum_steps:
            raise ValueError(
                f"long-horizon observation {name} minimum_steps must be >= contract minimum"
            )
        if comparison == "exact":
            if set(descriptor) != {"comparison", "minimum_steps"}:
                raise ValueError(f"long-horizon exact observation {name} has extra fields")
            continue
        if set(descriptor) != {"comparison", "minimum_steps", "tolerance"}:
            raise ValueError(f"long-horizon numeric observation {name} lacks tolerance")
        tolerance = descriptor["tolerance"]
        if not isinstance(tolerance, dict) or set(tolerance) != {
            "max_abs",
            "relative_rms",
            "min_cosine",
        }:
            raise ValueError(f"long-horizon observation {name} tolerance is incomplete")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in tolerance.values()
        ):
            raise ValueError(f"long-horizon observation {name} tolerance is non-finite")
        if (
            tolerance["max_abs"] < 0
            or tolerance["relative_rms"] < 0
            or not -1 <= tolerance["min_cosine"] <= 1
        ):
            raise ValueError(f"long-horizon observation {name} tolerance is out of range")

    sensitive_control = document["sensitive_control"]
    if not isinstance(sensitive_control, dict) or set(sensitive_control) != {
        "tensor_class",
        "fixture",
        "tensor_identity",
        "candidate_field",
        "required_observation",
    }:
        raise ValueError("long-horizon sensitive_control fields are invalid")
    if not isinstance(sensitive_control["tensor_class"], str) or not sensitive_control[
        "tensor_class"
    ].startswith("gdn_"):
        raise ValueError("long-horizon sensitive_control tensor_class must be a GDN class")
    if not isinstance(sensitive_control["fixture"], str) or not sensitive_control["fixture"]:
        raise ValueError("long-horizon sensitive_control fixture must be non-empty")
    if (
        not isinstance(sensitive_control["tensor_identity"], str)
        or not sensitive_control["tensor_identity"]
    ):
        raise ValueError("long-horizon sensitive_control tensor_identity must be non-empty")
    if (
        not isinstance(sensitive_control["candidate_field"], str)
        or not sensitive_control["candidate_field"]
    ):
        raise ValueError("long-horizon sensitive_control candidate_field must be non-empty")
    if sensitive_control["candidate_field"] not in _PERTURBATION_FIELDS:
        raise ValueError(
            "long-horizon sensitive_control candidate_field must identify a declared perturbation"
        )
    required_observation = sensitive_control["required_observation"]
    if (
        required_observation not in observations
        or observations[required_observation]["comparison"] != "numeric"
    ):
        raise ValueError(
            "long-horizon sensitive_control required_observation must be a numeric observation"
        )
    return document


def _sensitive_fixture_candidates(
    fixture: str, *, contract_path: str | Path | None = None
) -> tuple[Path, ...]:
    declared = Path(fixture)
    candidates: list[Path] = []
    if declared.is_absolute():
        candidates.append(declared)
    else:
        if contract_path is not None:
            candidates.append(Path(contract_path).resolve().parent / declared)
        repository_root = Path(__file__).resolve().parents[2]
        candidates.extend(
            [
                Path.cwd() / declared,
                repository_root / declared,
                repository_root / "tests" / "fixtures" / declared,
            ]
        )
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def _load_sensitive_fixture(
    document: Mapping[str, Any], *, contract_path: str | Path | None = None
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    control = document["sensitive_control"]
    candidates = _sensitive_fixture_candidates(control["fixture"], contract_path=contract_path)
    for path in candidates:
        if not path.is_file():
            continue
        try:
            fixture = _strict_json_loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"unable to load long-horizon sensitive fixture {path}: {error}"
            ) from error
        if not isinstance(fixture, dict):
            raise ValueError(f"long-horizon sensitive fixture {path} must be an object")
        try:
            result = evaluate_sensitive_tensor_fixture(fixture)
        except ValueError as error:
            raise ValueError(f"invalid long-horizon sensitive fixture {path}: {error}") from error
        if result["tensor_identity"] != control["tensor_identity"]:
            raise ValueError(
                "long-horizon sensitive fixture tensor identity disagrees: "
                f"{result['tensor_identity']!r} != {control['tensor_identity']!r}"
            )
        if result["tensor_class"] != control["tensor_class"]:
            raise ValueError(
                "long-horizon sensitive fixture tensor class disagrees: "
                f"{result['tensor_class']!r} != {control['tensor_class']!r}"
            )
        candidate = fixture.get("candidate")
        field = control["candidate_field"]
        if (
            not isinstance(candidate, Mapping)
            or field not in candidate
            or candidate[field] is None
            or candidate[field] == ""
        ):
            raise ValueError(
                "long-horizon sensitive fixture lacks declared candidate perturbation field "
                f"{field!r}"
            )
        if result["expected_rejected"] is not True or result["rejected"] is not True:
            raise ValueError("long-horizon sensitive fixture must be a rejected positive control")
        return fixture, result, path
    raise ValueError(
        f"long-horizon sensitive fixture could not be resolved: {control['fixture']!r}"
    )


def load_long_horizon_contract(path: str | Path) -> dict[str, Any]:
    """Load and fail closed on a versioned long-horizon H0 contract."""
    try:
        document = _strict_json_loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"unable to load long-horizon contract: {error}") from error
    checked = _validate_contract(document)
    _load_sensitive_fixture(checked, contract_path=path)
    return checked


def _contract_document(contract: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(contract, Mapping):
        document = _validate_contract(dict(contract))
        _load_sensitive_fixture(document)
        return document
    return load_long_horizon_contract(contract)


def validate_long_horizon_outputs(
    contract: Mapping[str, Any], outputs: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """Validate every declared semantic turn in a trajectory.

    Missing or extra probe IDs and wrong turn counts are contract errors.  A malformed
    or incorrect output is represented as a failed probe so a caller can retain the
    complete report alongside numeric state comparisons.
    """
    document = _contract_document(contract)
    if not isinstance(outputs, Mapping):
        raise ValueError("long-horizon outputs must be an object keyed by probe ID")
    expected_ids = {probe["id"] for probe in document["probes"]}
    if set(outputs) != expected_ids:
        raise ValueError(
            "long-horizon output probe IDs disagree: "
            f"expected={sorted(expected_ids)}, actual={sorted(outputs)}"
        )
    reports: list[dict[str, Any]] = []
    for probe in document["probes"]:
        probe_id = probe["id"]
        values = outputs[probe_id]
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
            raise ValueError(f"long-horizon output {probe_id} must be a sequence")
        errors: list[str] = []
        if len(values) != len(probe["steps"]):
            errors.append(f"expected {len(probe['steps'])} outputs, observed {len(values)}")
        for step, output in zip(probe["steps"], values, strict=False):
            if not isinstance(output, str):
                errors.append(f"step {step['id']} output must be a string")
                continue
            step_errors = validate_probe_output({"expectation": step["expectation"]}, output)
            errors.extend(f"step {step['id']}: {error}" for error in step_errors)
        reports.append(
            {
                "id": probe_id,
                "kind": probe["kind"],
                "step_count": len(values),
                "errors": errors,
                "passed": not errors,
            }
        )
    return {
        "schema_name": document["schema_name"],
        "schema_version": document["schema_version"],
        "contract_id": document["contract_id"],
        "probes": reports,
        "passed": all(report["passed"] for report in reports),
    }


def _validate_observation_horizon(
    observations: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    *,
    label: str,
) -> None:
    expected_steps: int | None = None
    for name, descriptor in contract["observations"].items():
        if name not in observations:
            raise ValueError(f"{label} is missing long-horizon observation {name}")
        array = observations[name]
        if array.ndim == 0 or array.shape[0] < descriptor["minimum_steps"]:
            raise ValueError(
                f"{label} observation {name} is below the minimum horizon "
                f"of {descriptor['minimum_steps']} steps"
            )
        if expected_steps is None:
            expected_steps = array.shape[0]
        elif array.shape[0] != expected_steps:
            raise ValueError(
                f"{label} long-horizon observation step counts disagree: "
                f"{name} has {array.shape[0]}, expected {expected_steps}"
            )
        if descriptor["comparison"] == "exact" and array.dtype.kind not in "iu":
            raise ValueError(f"{label} exact observation {name} must use integer values")
        if descriptor["comparison"] == "numeric" and array.dtype.kind not in "fiu":
            raise ValueError(f"{label} numeric observation {name} has unsupported dtype")


def compare_long_horizon_bundles(
    subject_path: str | Path,
    reference_path: str | Path,
    *,
    contract: str | Path | Mapping[str, Any],
    subject_outputs: Mapping[str, Sequence[str]],
    reference_outputs: Mapping[str, Sequence[str]],
    require_independent: bool = True,
    evidence_status: str = "synthetic",
) -> dict[str, Any]:
    """Compare long-horizon state/route evidence under a strict H0 contract."""
    document = _contract_document(contract)
    _subject_identity, subject = read_observation_bundle(subject_path)
    _reference_identity, reference = read_observation_bundle(reference_path)
    _validate_observation_horizon(subject, document, label="subject")
    _validate_observation_horizon(reference, document, label="reference")
    subject_semantic = validate_long_horizon_outputs(document, subject_outputs)
    reference_semantic = validate_long_horizon_outputs(document, reference_outputs)
    semantic_comparisons: list[dict[str, Any]] = []
    for probe in document["probes"]:
        subject_steps = subject_outputs[probe["id"]]
        reference_steps = reference_outputs[probe["id"]]
        for step, actual, expected in zip(
            probe["steps"], subject_steps, reference_steps, strict=False
        ):
            passed = isinstance(actual, str) and actual == expected
            semantic_comparisons.append(
                {
                    "probe": probe["id"],
                    "step": step["id"],
                    "metric": "text_exact",
                    "observed": passed,
                    "limit": True,
                    "passed": passed,
                }
            )
    semantic_passed = (
        subject_semantic["passed"]
        and reference_semantic["passed"]
        and all(item["passed"] for item in semantic_comparisons)
    )
    exact_observations = {
        name
        for name, descriptor in document["observations"].items()
        if descriptor["comparison"] == "exact"
    }
    tolerances = {
        name: Tolerance(**descriptor["tolerance"])
        for name, descriptor in document["observations"].items()
        if descriptor["comparison"] == "numeric"
    }
    evidence = compare_observation_bundles(
        subject_path,
        reference_path,
        tolerances=tolerances,
        exact_observations=exact_observations,
        require_independent=require_independent,
        evidence_status=evidence_status,
    )
    sensitive_fixture, sensitive_result, sensitive_path = _load_sensitive_fixture(document)
    evidence["long_horizon"] = {
        "contract_id": document["contract_id"],
        "minimum_steps": document["minimum_steps"],
        "probe_kinds": sorted({probe["kind"] for probe in document["probes"]}),
        "semantic": {
            "subject": subject_semantic,
            "reference": reference_semantic,
            "comparisons": semantic_comparisons,
            "passed": semantic_passed,
        },
        "sensitive_control": {
            **dict(document["sensitive_control"]),
            "resolved_fixture": str(sensitive_path),
            "fixture_tensor_identity": sensitive_fixture["tensor_identity"],
            "fixture_tensor_class": sensitive_fixture["tensor_class"],
            "declared_perturbation": sensitive_fixture["candidate"][
                document["sensitive_control"]["candidate_field"]
            ],
            "fixture_rejected": sensitive_result["rejected"],
            "fixture_expected_rejected": sensitive_result["expected_rejected"],
        },
    }
    evidence["passed"] = bool(evidence["passed"] and semantic_passed)
    return evidence


__all__ = [
    "compare_long_horizon_bundles",
    "load_long_horizon_contract",
    "validate_long_horizon_outputs",
]
