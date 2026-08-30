"""Sensitive-tensor precision-island census and fail-closed validation.

The ordinary GGUF census deliberately treats routed expert banks as a family.  That
is unsafe for the small tensors which drive every token, so this module keeps an
explicit, exact-identity policy boundary for routers, shared-expert gates, GDN
controls, hyperconnection writes, norms, and reference-identified controls.

This is an H0 tensor-level contract.  It records the selected representation and
scale metadata; it does not claim the later long-horizon model-quality gate.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_EXPERT_RE = re.compile(r"(?:^|\.)(?:ffn|mlp)_(?:gate|up|down)_exps\.weight(?:$|\.)")
_SCALED_FORMATS = frozenset(
    {
        "Q8_0",
        "Q8_K",
        "Q8_K_XL",
        "Q6_K",
        "Q5_K",
        "Q5_1",
        "Q5_0",
        "Q4_K",
        "Q4_0",
        "Q4_1",
        "Q3_K",
        "Q2_K",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ3_XXS",
        "IQ3_XS",
        "IQ3_S",
        "IQ4_XS",
        "IQ4_NL",
        "F8_E4M3",
        "F8_E5M2",
    }
)
_LOSSLESS_FORMATS = frozenset({"F32", "F16", "BF16", "FLOAT32", "FLOAT16", "BFLOAT16"})
_ALLOWED_SENSITIVE_FORMATS = frozenset({"F32", "F16", "BF16", "Q8_0", "Q8_K", "Q8_K_XL"})
_CLASSES = frozenset(
    {
        "moe_router",
        "shared_expert_gate",
        "shared_expert_gate_scale",
        "gdn_in_proj_a",
        "gdn_in_proj_b",
        "gdn_state_projection",
        "gdn_control",
        "gdn_output_gate",
        "hyperconnection_control",
        "residual_write_gate",
        "norm",
        "reference_control",
    }
)


def _normalise_format(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"sensitive tensor quant format must be a non-empty string, got {value!r}")
    raw = value.strip().upper().replace(" ", "_").replace(".", "_")
    aliases = {
        "FLOAT32": "F32",
        "FLOAT16": "F16",
        "BFLOAT16": "BF16",
        "TORCH_FLOAT32": "F32",
        "TORCH_FLOAT16": "F16",
        "TORCH_BFLOAT16": "BF16",
    }
    return aliases.get(raw, raw)


def _is_expert_tensor(name: str) -> bool:
    # Keep the exclusion exact: a broad ``blk.*`` rule must never make a routed
    # expert look like a sensitive control, and a control must never enter a bank.
    return bool(_EXPERT_RE.search(name))


def _ends_with_any(name: str, suffixes: tuple[str, ...]) -> bool:
    return name.endswith(suffixes)


def classify_sensitive_tensor(
    name: str,
    *,
    reference_identified_controls: Mapping[str, str] | None = None,
) -> str | None:
    """Return the class for an exact tensor identity, or ``None`` for ordinary tensors.

    Patterns cover the reconciled Qwen GGUF names and their source checkpoint names.
    Reference-identified controls are an exact-name extension point, never a wildcard.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"tensor identity must be a non-empty string, got {name!r}")
    if _is_expert_tensor(name):
        return None
    controls = reference_identified_controls or {}
    if name in controls:
        value = controls[name]
        if value not in _CLASSES:
            raise ValueError(f"unsupported reference-sensitive tensor class {value!r} for {name}")
        return value

    # MoE router and shared-expert gate names in the GGUF census, followed by the
    # source checkpoint spellings used before converter fusion.
    if re.fullmatch(r"blk\.\d+\.ffn_gate_inp\.weight", name) or re.fullmatch(
        r"model\.layers\.\d+\.mlp\.gate\.weight", name
    ):
        return "moe_router"
    if re.fullmatch(
        r"(?:blk\.\d+\.ffn_gate_inp_shexp|model\.layers\.\d+\.mlp\.shared_expert_gate)\.weight",
        name,
    ):
        return "shared_expert_gate"
    if re.fullmatch(
        r"(?:blk\.\d+\.ffn_gate_inp_shexp|model\.layers\.\d+\.mlp\.shared_expert_gate)\.(?:weight_scale|weight_scale_2|input_scale)",
        name,
    ):
        return "shared_expert_gate_scale"

    # GDN's reconciled in_proj_a/in_proj_b are represented as ssm_alpha/beta by
    # GGUF.  Keep both spellings explicit so fusion cannot erase their identity.
    if _ends_with_any(name, (".ssm_alpha.weight", ".in_proj_a.weight")):
        return "gdn_in_proj_a"
    if _ends_with_any(name, (".ssm_beta.weight", ".in_proj_b.weight")):
        return "gdn_in_proj_b"
    if _ends_with_any(name, (".attn_qkv.weight", ".linear_attn.in_proj_qkv.weight")):
        return "gdn_state_projection"
    if _ends_with_any(
        name,
        (
            ".ssm_a",
            ".ssm_a.weight",
            ".ssm_dt.bias",
            ".A_log",
            ".A_log.weight",
            ".dt_bias",
            ".dt_bias.weight",
        ),
    ):
        return "gdn_control"
    if _ends_with_any(name, (".attn_gate.weight", ".linear_attn.gate.weight")):
        return "gdn_output_gate"

    # Converter names for the write gate are block_inject and hc_*_inject.  A
    # fused source tensor is retained as sensitive because it contains that gate.
    if _ends_with_any(
        name,
        (
            ".hc_attn_inject.weight",
            ".hc_ffn_inject.weight",
            ".block_inject_weight.weight",
            ".input_mix_weight_down_block_inject.weight",
            ".input_mix_weight_down_block_inject.qweight",
        ),
    ):
        return "residual_write_gate"
    if re.fullmatch(r"output_hc_(?:up|down)\.weight", name) or re.fullmatch(
        r"blk\.\d+\.hc_(?:attn|ffn)_(?:up|down)\.weight", name
    ) or _ends_with_any(
        name,
        (
            ".input_mix_weight_up.weight",
            ".input_mix_weight_down.weight",
            ".input_mix_weight_down_block_inject.weight",
        ),
    ):
        return "hyperconnection_control"

    # Norms are small, continuously active tensors.  This suffix set is shared by
    # GGUF (ssm_norm, attn_q_norm, indexer.q_norm) and source names (hc_norm,
    # q_layernorm, norm_key, ...), while remaining narrower than a wildcard tensor rule.
    if _ends_with_any(
        name,
        (
            ".norm.weight",
            "_norm.weight",
            ".layernorm.weight",
            "_layernorm.weight",
        ),
    ):
        return "norm"
    return None


def default_scale_representation(quant_format: str) -> dict[str, Any]:
    if quant_format in _LOSSLESS_FORMATS or quant_format in {"F32", "F16", "BF16"}:
        return {"kind": "none", "location": "authoritative-tensor", "required": False}
    if quant_format == "Q8_0":
        return {
            "kind": "per-block",
            "location": "inline-ggml-block",
            "dtype": "F16",
            "required": True,
        }
    if quant_format in _SCALED_FORMATS:
        return {
            "kind": "format-defined",
            "location": "inline-quant-block",
            "dtype": "format-defined",
            "required": True,
        }
    return {"kind": "unknown", "location": "unknown", "required": True}


def _validate_scale_representation(value: Any, quant_format: str, name: str) -> dict[str, Any]:
    if value is None:
        if quant_format in _SCALED_FORMATS:
            raise ValueError(f"{name}: missing required scale metadata for {quant_format}")
        raise ValueError(f"{name}: scale representation cannot be null")
    if isinstance(value, str):
        # Permit concise metadata from callers while storing one canonical object.
        value = {"kind": value, "location": "declared", "required": quant_format in _SCALED_FORMATS}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}: scale representation must be an object")
    result = dict(value)
    required = {"kind", "location", "required"}
    if not required <= set(result):
        raise ValueError(f"{name}: incomplete scale representation; required={sorted(required)}")
    if not isinstance(result["kind"], str) or not result["kind"]:
        raise ValueError(f"{name}: scale representation kind must be non-empty")
    if not isinstance(result["location"], str) or not result["location"]:
        raise ValueError(f"{name}: scale representation location must be non-empty")
    if not isinstance(result["required"], bool):
        raise ValueError(f"{name}: scale representation required must be boolean")
    if quant_format in _SCALED_FORMATS and not result["required"]:
        raise ValueError(f"{name}: scaled sensitive tensor must declare required scale metadata")
    if quant_format in _LOSSLESS_FORMATS and result["required"]:
        raise ValueError(f"{name}: lossless tensor cannot require quantization scale metadata")
    return result


def _rationale(tensor_class: str, quant_format: str) -> str:
    if tensor_class == "moe_router":
        return (
            "Router logits select experts for every token; keep an explicit precision decision "
            "outside expert-bank rules."
        )
    if tensor_class in {"shared_expert_gate", "shared_expert_gate_scale"}:
        return (
            "The shared-expert gate contributes on every layer/token and its scale must be "
            "validated independently."
        )
    if tensor_class in {
        "gdn_in_proj_a",
        "gdn_in_proj_b",
        "gdn_state_projection",
        "gdn_control",
    }:
        return (
            "GDN state-driving/control values recur across decode steps; scale and precision "
            "errors can accumulate."
        )
    if tensor_class == "gdn_output_gate":
        return (
            "The GDN output gate is continuously active and cannot inherit a routed-expert "
            "quantization rule."
        )
    if tensor_class == "residual_write_gate":
        return (
            "Hyperconnection/residual writes affect the active stream every layer and require "
            "an explicit precision island."
        )
    if tensor_class == "hyperconnection_control":
        return (
            "Hyperconnection mix projections control the active residual streams on every layer; "
            "they require an explicit precision decision outside expert-bank rules."
        )
    if tensor_class == "norm":
        return (
            "Normalization is continuously active; preserve its authoritative source "
            "representation pending evidence."
        )
    if quant_format in _SCALED_FORMATS:
        return (
            "Reference-identified control with format-defined scale; independent parity is "
            "required."
        )
    return "Reference comparison identified this continuously active control as sensitive."


def build_sensitive_tensor_census(
    tensors: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    conversion_provenance: str = "gguf_census:direct-tensor-metadata",
    reference_identified_controls: Mapping[str, str] | None = None,
    required_identities: Sequence[str] = (),
    promotion_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build and validate deterministic records for sensitive tensor identities.

    ``tensors`` is the existing ``inspect_gguf``/quant-census record shape.  Unknown
    tensors are ordinary by default; callers can require additional exact identities
    through ``reference_identified_controls`` and ``required_identities``.
    """
    if not isinstance(profile, str) or not profile:
        raise ValueError("sensitive census profile must be a non-empty string")
    if not isinstance(conversion_provenance, str) or not conversion_provenance:
        raise ValueError("conversion provenance must be a non-empty string")
    controls = dict(reference_identified_controls or {})
    evidence_by_class = dict(promotion_evidence or {})
    required = tuple(required_identities)
    if any(not isinstance(name, str) or not name for name in required):
        raise ValueError("required sensitive identities must be non-empty strings")
    by_name: dict[str, Mapping[str, Any]] = {}
    result: list[dict[str, Any]] = []
    for tensor in tensors:
        if not isinstance(tensor, Mapping):
            raise ValueError("sensitive census tensor records must be objects")
        name = tensor.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"sensitive census tensor has invalid identity {name!r}")
        by_name[name] = tensor
        tensor_class = classify_sensitive_tensor(name, reference_identified_controls=controls)
        if tensor_class is None:
            continue
        quant_value = tensor.get("quant_name", tensor.get("quant_format", tensor.get("dtype")))
        quant_format = _normalise_format(quant_value)
        if quant_format not in _ALLOWED_SENSITIVE_FORMATS:
            raise ValueError(
                f"{name}: unsupported sensitive quantization {quant_format}; "
                "sensitive tensors allow source precision or explicit Q8 only"
            )
        scale_metadata = tensor.get("scale_representation")
        if scale_metadata is None and "scale_representation" in tensor:
            _validate_scale_representation(None, quant_format, name)
        if scale_metadata is None:
            if quant_format in _SCALED_FORMATS:
                raise ValueError(f"{name}: missing required scale metadata for {quant_format}")
            scale_metadata = default_scale_representation(quant_format)
        scale = _validate_scale_representation(scale_metadata, quant_format, name)
        # A quantized source cannot be reconstructed as the original F32 values.
        # Keep its authoritative source format visible; selecting Q8 (including
        # retaining an already-Q8 source) still requires an explicit per-class
        # status/evidence rather than a broad expert-bank default.
        selected_value = tensor.get("selected_precision")
        selected = quant_format if selected_value is None else selected_value
        selected = _normalise_format(selected)
        if selected not in _ALLOWED_SENSITIVE_FORMATS:
            raise ValueError(f"{name}: unsupported selected sensitive precision {selected}")
        if selected != quant_format and quant_format not in _LOSSLESS_FORMATS:
            raise ValueError(
                f"{name}: lossy source {quant_format} cannot be relabeled or promoted to "
                f"{selected} without authoritative source bytes"
            )
        promotion_status = "baseline"
        promotion_id: str | None = None
        needs_evidence = selected != quant_format or selected in {"Q8_0", "Q8_K", "Q8_K_XL"}
        if needs_evidence:
            evidence = tensor.get("promotion_evidence", evidence_by_class.get(tensor_class))
            if evidence is None:
                # Census generation may describe an existing or proposed reduced
                # representation, but serving qualification remains blocked until
                # real tensor/model evidence exists.
                promotion_status = "unqualified"
            elif not isinstance(evidence, Mapping):
                raise ValueError(f"{name}: Q8 promotion evidence must be an object")
            else:
                promotion_status = evidence.get("status")
                promotion_id = evidence.get("evidence_id")
                if (
                    promotion_status not in {"qualified", "passed"}
                    or not isinstance(promotion_id, str)
                    or not promotion_id
                ):
                    raise ValueError(
                        f"{name}: Q8 promotion evidence must have status qualified/passed "
                        "and a non-empty evidence_id"
                    )
        source_dtype = quant_format if quant_format in {"F32", "F16", "BF16"} else None
        record: dict[str, Any] = {
            "identity": name,
            "name": name,
            "class": tensor_class,
            "dtype": source_dtype,
            "quant_format": quant_format,
            "dtype_or_quant": quant_format,
            "scale_representation": scale,
            "conversion_provenance": tensor.get("conversion_provenance", conversion_provenance),
            "selected_precision": selected,
            "promotion_status": promotion_status,
            "promotion_evidence": promotion_id,
            "rationale": tensor.get("rationale", _rationale(tensor_class, quant_format)),
        }
        if "shape" in tensor:
            record["shape"] = list(tensor["shape"])
        result.append(record)

    missing = [name for name in required if name not in by_name]
    if missing:
        raise ValueError(f"missing required sensitive tensor metadata: {sorted(missing)}")
    result.sort(key=lambda item: str(item["identity"]))
    validate_sensitive_tensor_census(result)
    return result


def build_sensitive_precision_policy(profile: str) -> dict[str, Any]:
    """Return the explicit policy metadata embedded in a quant census."""
    if not isinstance(profile, str) or not profile:
        raise ValueError("sensitive policy profile must be a non-empty string")
    return {
        "schema_name": "sensitive-tensor-precision-policy",
        "schema_version": 1,
        "profile": profile,
        "expert_rule": {
            "scope": "routed-expert-tensors-only",
            "match": "explicit ffn_*_exps.weight identities",
            "sensitive_exclusion": True,
        },
        "sensitive_rule": "exact-identity-only",
        "baseline": "authoritative-source-or-nearest-lossless-runtime-representation",
        "allowed_formats": sorted(_ALLOWED_SENSITIVE_FORMATS),
        "sub8_default": "reject",
        "q8_promotion": "per-class-evidence-required",
        "serving_qualification": "blocked-until-sensitive-evidence",
        "long_horizon_semantic_gate": "deferred",
    }


def validate_sensitive_tensor_census(
    records: Sequence[Mapping[str, Any]],
    *,
    expert_rule: Mapping[str, Any] | None = None,
) -> None:
    """Validate record identity, scale metadata and the expert exclusion boundary."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("sensitive tensor census must be an array")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("sensitive tensor census records must be objects")
        required = {
            "identity",
            "name",
            "class",
            "dtype",
            "quant_format",
            "dtype_or_quant",
            "scale_representation",
            "conversion_provenance",
            "selected_precision",
            "promotion_status",
            "promotion_evidence",
            "rationale",
        }
        missing = required - set(record)
        if missing:
            raise ValueError(f"sensitive tensor record is missing fields: {sorted(missing)}")
        name = record["identity"]
        if not isinstance(name, str) or not name or record["name"] != name:
            raise ValueError(f"sensitive tensor identity/name mismatch for {name!r}")
        if name in seen:
            raise ValueError(f"duplicate sensitive tensor identity {name!r}")
        seen.add(name)
        if _is_expert_tensor(name):
            raise ValueError(f"sensitive tensor {name} overlaps the routed expert bank")
        tensor_class = record["class"]
        if tensor_class not in _CLASSES:
            raise ValueError(f"unsupported sensitive tensor class {tensor_class!r} for {name}")
        classified = classify_sensitive_tensor(name)
        if classified is not None and classified != tensor_class:
            raise ValueError(
                f"sensitive tensor {name} class {tensor_class!r} disagrees with {classified!r}"
            )
        quant_format = _normalise_format(record["quant_format"])
        if record["dtype_or_quant"] != quant_format:
            raise ValueError(f"{name}: dtype_or_quant disagrees with quant_format")
        if quant_format not in _ALLOWED_SENSITIVE_FORMATS:
            raise ValueError(f"{name}: unsupported sensitive quantization {quant_format}")
        _validate_scale_representation(record["scale_representation"], quant_format, name)
        selected = _normalise_format(record["selected_precision"])
        quant_changed = selected != quant_format
        needs_evidence = quant_changed or selected in {"Q8_0", "Q8_K", "Q8_K_XL"}
        if needs_evidence:
            if (
                record["promotion_status"]
                not in {
                    "unqualified",
                    "qualified",
                    "passed",
                }
                or (
                    record["promotion_status"] in {"qualified", "passed"}
                    and (
                        not isinstance(record["promotion_evidence"], str)
                        or not record["promotion_evidence"]
                    )
                )
                or (
                    record["promotion_status"] == "unqualified"
                    and record["promotion_evidence"] is not None
                )
            ):
                raise ValueError(
                    f"{name}: selected precision requires explicit per-class promotion evidence "
                    "before serving qualification"
                )
        elif record["promotion_status"] != "baseline" or record["promotion_evidence"] is not None:
            raise ValueError(
                f"{name}: non-Q8 sensitive precision must use baseline promotion metadata"
            )
        if (
            not isinstance(record["conversion_provenance"], str)
            or not record["conversion_provenance"]
        ):
            raise ValueError(f"{name}: conversion provenance must be non-empty")
        if not isinstance(record["rationale"], str) or not record["rationale"]:
            raise ValueError(f"{name}: rationale must be non-empty")
    if expert_rule is not None:
        if not isinstance(expert_rule, Mapping):
            raise ValueError("expert rule must be an object")
        if expert_rule.get("scope") != "routed-expert-tensors-only":
            raise ValueError("expert rule must be scoped to routed-expert-tensors-only")
        if expert_rule.get("sensitive_exclusion") is not True:
            raise ValueError("expert rule must explicitly exclude sensitive tensors")


def validate_sensitive_profile_qualification(records: Sequence[Mapping[str, Any]]) -> None:
    """Reject serving selection while any sensitive class lacks real evidence."""
    validate_sensitive_tensor_census(records)
    blocked = [
        str(record["identity"]) for record in records if record["promotion_status"] == "unqualified"
    ]
    if blocked:
        raise ValueError(
            "profile qualification is blocked by unqualified sensitive tensors: "
            f"{blocked[:8]}" + (" ..." if len(blocked) > 8 else "")
        )


def _fixture_document(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        result = json.loads(Path(value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"invalid sensitive tensor fixture: {error}") from error
    if not isinstance(result, dict):
        raise ValueError("sensitive tensor fixture must be an object")
    return result


def _fixture_values(section: Any, label: str) -> np.ndarray:
    if not isinstance(section, Mapping) or not isinstance(section.get("values"), list):
        raise ValueError(f"sensitive tensor fixture {label} must contain a values array")
    values = np.asarray(section["values"], dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"sensitive tensor fixture {label} values must be finite and non-empty")
    dtype = section.get("dtype", "float64")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(f"sensitive tensor fixture {label} dtype must be non-empty")
    return values


def _fixture_metrics(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float, float]:
    delta = actual - expected
    max_abs = float(np.max(np.abs(delta)))
    relative_rms = float(
        np.sqrt(np.mean(delta * delta)) / max(np.sqrt(np.mean(expected * expected)), 1e-30)
    )
    actual_norm = float(np.linalg.norm(actual))
    expected_norm = float(np.linalg.norm(expected))
    if actual_norm == 0.0 or expected_norm == 0.0:
        cosine = 1.0 if np.array_equal(actual, expected) else 0.0
    else:
        cosine = float(np.dot(actual, expected) / (actual_norm * expected_norm))
    return max_abs, relative_rms, min(1.0, max(-1.0, cosine))


def evaluate_sensitive_tensor_fixture(
    fixture: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one deterministic tensor-level ablation fixture.

    This intentionally reports a rejection rather than claiming the later semantic
    gate: it proves that a bad gate scale/control tensor is visible to the H0 harness.
    """
    document = _fixture_document(fixture)
    expected_keys = {
        "schema_name",
        "schema_version",
        "tensor_identity",
        "tensor_class",
        "seed",
        "reference",
        "candidate",
        "tolerance",
        "expected_rejected",
    }
    if set(document) != expected_keys:
        raise ValueError(
            f"sensitive tensor fixture fields disagree: expected={sorted(expected_keys)}, "
            f"actual={sorted(document)}"
        )
    if document["schema_name"] != "sensitive-tensor-ablation" or document["schema_version"] != 1:
        raise ValueError("unsupported sensitive tensor fixture schema")
    if classify_sensitive_tensor(document["tensor_identity"]) != document["tensor_class"]:
        raise ValueError(
            "sensitive tensor fixture identity/class is not a reconciled sensitive tensor"
        )
    if (
        not isinstance(document["seed"], int)
        or isinstance(document["seed"], bool)
        or document["seed"] < 0
    ):
        raise ValueError("sensitive tensor fixture seed must be a non-negative integer")
    reference = _fixture_values(document["reference"], "reference")
    candidate = _fixture_values(document["candidate"], "candidate")
    if reference.shape != candidate.shape:
        raise ValueError("sensitive tensor fixture reference/candidate shapes disagree")
    tolerance = document["tolerance"]
    if not isinstance(tolerance, Mapping) or set(tolerance) != {
        "max_abs",
        "relative_rms",
        "min_cosine",
    }:
        raise ValueError("sensitive tensor fixture tolerance is incomplete")
    limits = {key: tolerance[key] for key in tolerance}
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for value in limits.values()
    ):
        raise ValueError("sensitive tensor fixture tolerances must be finite numbers")
    max_abs, relative_rms, cosine = _fixture_metrics(candidate, reference)
    passed = (
        max_abs <= limits["max_abs"]
        and relative_rms <= limits["relative_rms"]
        and cosine >= limits["min_cosine"]
    )
    rejected = not passed
    if not isinstance(document["expected_rejected"], bool):
        raise ValueError("sensitive tensor fixture expected_rejected must be boolean")
    return {
        "schema_name": "sensitive-tensor-ablation-result",
        "schema_version": 1,
        "tensor_identity": document["tensor_identity"],
        "tensor_class": document["tensor_class"],
        "max_abs": max_abs,
        "relative_rms": relative_rms,
        "cosine": cosine,
        "rejected": rejected,
        "expected_rejected": document["expected_rejected"],
        "passed": passed,
    }


def assert_sensitive_tensor_fixture_rejected(
    fixture: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    report = evaluate_sensitive_tensor_fixture(fixture)
    if not report["rejected"]:
        raise ValueError(f"sensitive tensor fixture was not rejected: {report['tensor_identity']}")
    if not report["expected_rejected"]:
        raise ValueError("sensitive tensor fixture does not declare an expected rejection")
    return report


__all__ = [
    "assert_sensitive_tensor_fixture_rejected",
    "build_sensitive_precision_policy",
    "build_sensitive_tensor_census",
    "classify_sensitive_tensor",
    "default_scale_representation",
    "evaluate_sensitive_tensor_fixture",
    "validate_sensitive_profile_qualification",
    "validate_sensitive_tensor_census",
]
