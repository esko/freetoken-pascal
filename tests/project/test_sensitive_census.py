from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from freetoken.gguf_census import build_sensitive_tensor_census
from freetoken.sensitive_census import (
    assert_q8_0_packed_scale_parity,
    assert_sensitive_tensor_fixture_rejected,
    classify_sensitive_tensor,
    evaluate_q8_0_packed_scale_parity,
    evaluate_sensitive_tensor_fixture,
    validate_sensitive_profile_qualification,
    validate_sensitive_tensor_census,
)
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "sensitive"


def _records() -> list[dict[str, object]]:
    return [
        {
            "name": "blk.0.ffn_gate_inp.weight",
            "shape": [512, 2560],
            "quant_name": "F32",
            "quant_type": 0,
        },
        {
            "name": "blk.0.ffn_gate_inp_shexp.weight",
            "shape": [2560],
            "quant_name": "F32",
            "quant_type": 0,
        },
        {
            "name": "blk.0.ssm_alpha.weight",
            "shape": [48, 2560],
            "quant_name": "F32",
            "quant_type": 0,
        },
        {
            "name": "blk.0.ssm_beta.weight",
            "shape": [48, 2560],
            "quant_name": "F32",
            "quant_type": 0,
        },
        {
            "name": "blk.0.attn_qkv.weight",
            "shape": [5632, 2560],
            "quant_name": "F32",
            "quant_type": 0,
        },
        {
            "name": "blk.0.ssm_a",
            "shape": [48],
            "quant_name": "F32",
            "quant_type": 0,
        },
        {
            "name": "blk.0.ssm_dt.bias",
            "shape": [48],
            "quant_name": "F32",
            "quant_type": 0,
        },
        {
            "name": "blk.0.hc_attn_inject.weight",
            "shape": [4, 10240],
            "quant_name": "F32",
            "quant_type": 0,
        },
        {
            "name": "blk.0.hc_attn_up.weight",
            "shape": [10240, 512],
            "quant_name": "F32",
            "quant_type": 0,
        },
        {
            "name": "blk.0.ssm_norm.weight",
            "shape": [128],
            "quant_name": "F32",
            "quant_type": 0,
        },
    ]


def test_sensitive_census_classifies_reconciled_tensor_identities() -> None:
    assert classify_sensitive_tensor("blk.0.ffn_gate_inp.weight") == "moe_router"
    assert classify_sensitive_tensor("blk.0.ffn_gate_inp_shexp.weight") == "shared_expert_gate"
    assert classify_sensitive_tensor("blk.0.ssm_alpha.weight") == "gdn_in_proj_a"
    assert classify_sensitive_tensor("blk.0.ssm_beta.weight") == "gdn_in_proj_b"
    assert classify_sensitive_tensor("blk.0.attn_qkv.weight") == "gdn_state_projection"
    assert (
        classify_sensitive_tensor("model.layers.0.linear_attn.in_proj_z.weight")
        == "gdn_output_gate"
    )
    assert classify_sensitive_tensor("blk.0.in_proj_z.weight") == "gdn_output_gate"
    assert (
        classify_sensitive_tensor("model.layers.0.linear_attn.in_proj.weight")
        == "gdn_state_projection"
    )
    assert (
        classify_sensitive_tensor("model.layers.0.linear_attn.in_proj_qkvz.weight")
        == "gdn_state_projection"
    )
    assert classify_sensitive_tensor("blk.0.in_proj_qkvz.weight") == "gdn_state_projection"
    assert classify_sensitive_tensor("blk.0.ssm_a") == "gdn_control"
    assert classify_sensitive_tensor("blk.0.ssm_dt.bias") == "gdn_control"
    assert classify_sensitive_tensor("blk.0.hc_attn_inject.weight") == "residual_write_gate"
    assert classify_sensitive_tensor("blk.0.hc_attn_up.weight") == "hyperconnection_control"
    assert classify_sensitive_tensor("output_hc_down.weight") == "hyperconnection_control"
    assert classify_sensitive_tensor("blk.0.ssm_norm.weight") == "norm"
    assert classify_sensitive_tensor("blk.0.ffn_gate_exps.weight") is None


def test_sensitive_census_is_explicit_and_machine_readable() -> None:
    records = build_sensitive_tensor_census(
        _records(),
        profile="throughput-q3",
        conversion_provenance="fixture-qwen-gguf-direct",
    )

    assert [record["identity"] for record in records] == sorted(
        record["identity"] for record in records
    )
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
        "rationale",
    }
    assert all(required <= set(record) for record in records)
    assert all(record["identity"] == record["name"] for record in records)
    assert all(record["selected_precision"] == "F32" for record in records)
    validate_sensitive_tensor_census(records)


def test_sensitive_policy_rejects_low_bit_controls_before_expert_dispatch() -> None:
    bad = _records()
    bad[1] = {**bad[1], "quant_name": "Q4_K", "quant_type": 12}

    with pytest.raises(ValueError, match="unsupported sensitive quantization"):
        build_sensitive_tensor_census(bad, profile="throughput-q3")


def test_sensitive_policy_rejects_missing_scale_metadata() -> None:
    bad = [
        {
            "name": "blk.0.ssm_alpha.weight",
            "shape": [48, 2560],
            "quant_name": "Q8_0",
            "quant_type": 8,
            "scale_representation": None,
        }
    ]

    with pytest.raises(ValueError, match="missing required scale metadata"):
        build_sensitive_tensor_census(bad, profile="candidate-ap-q4")


def test_sensitive_policy_rejects_implicit_scale_metadata() -> None:
    bad = [
        {
            "name": "blk.0.ssm_alpha.weight",
            "shape": [48, 2560],
            "quant_name": "Q8_0",
            "quant_type": 8,
        }
    ]

    with pytest.raises(ValueError, match="missing required scale metadata"):
        build_sensitive_tensor_census(bad, profile="candidate-ap-q4")


def test_sensitive_policy_blocks_unqualified_q8_promotion_from_serving() -> None:
    record = {
        "name": "blk.0.ssm_alpha.weight",
        "shape": [48, 2560],
        "quant_name": "F32",
        "quant_type": 0,
        "selected_precision": "Q8_0",
    }

    records = build_sensitive_tensor_census([record], profile="candidate-ap-q4")
    assert records[0]["promotion_status"] == "unqualified"
    with pytest.raises(ValueError, match="blocked by unqualified"):
        validate_sensitive_profile_qualification(records)


def test_sensitive_policy_rejects_untrusted_q8_qualification_evidence() -> None:
    record = {
        "name": "blk.0.ssm_alpha.weight",
        "shape": [48, 2560],
        "quant_name": "F32",
        "quant_type": 0,
        "selected_precision": "Q8_0",
    }
    with pytest.raises(ValueError, match="only baseline/unqualified are allowed"):
        build_sensitive_tensor_census(
            [record],
            profile="candidate-ap-q4",
            promotion_evidence={
                "gdn_in_proj_a": {"status": "qualified", "evidence_id": "fixture-gdn-q8-v1"}
            },
        )


def test_lossless_source_precision_change_requires_class_evidence() -> None:
    record = {
        "name": "blk.0.ssm_alpha.weight",
        "shape": [48, 2560],
        "quant_name": "F32",
        "quant_type": 0,
        "selected_precision": "F16",
    }

    records = build_sensitive_tensor_census([record], profile="candidate-ap-q4")

    assert records[0]["promotion_status"] == "unqualified"
    with pytest.raises(ValueError, match="blocked by unqualified"):
        validate_sensitive_profile_qualification(records)


def test_lossy_source_cannot_be_relabelled_as_higher_precision() -> None:
    record = {
        "name": "blk.0.ssm_alpha.weight",
        "shape": [48, 2560],
        "quant_name": "Q8_0",
        "quant_type": 8,
        "selected_precision": "F32",
        "scale_representation": {
            "kind": "per-block",
            "location": "inline-ggml-block",
            "dtype": "F16",
            "required": True,
        },
    }

    with pytest.raises(ValueError, match="cannot be relabeled or promoted"):
        build_sensitive_tensor_census([record], profile="reference-q4")


def test_sensitive_validator_rejects_lossy_source_relabel() -> None:
    source = {
        "name": "blk.0.ssm_alpha.weight",
        "shape": [48, 2560],
        "quant_name": "Q8_0",
        "quant_type": 8,
        "scale_representation": {
            "kind": "per-block",
            "location": "inline-ggml-block",
            "dtype": "F16",
            "required": True,
        },
    }
    records = build_sensitive_tensor_census([source], profile="reference-q4")
    records[0]["selected_precision"] = "F32"

    with pytest.raises(ValueError, match="cannot be relabeled or promoted"):
        validate_sensitive_tensor_census(records)


def test_q8_source_is_census_visible_but_unqualified_for_serving() -> None:
    record = {
        "name": "blk.0.ssm_alpha.weight",
        "shape": [48, 2560],
        "quant_name": "Q8_0",
        "quant_type": 8,
        "scale_representation": {
            "kind": "per-block",
            "location": "inline-ggml-block",
            "dtype": "F16",
            "required": True,
        },
    }

    records = build_sensitive_tensor_census([record], profile="reference-q4")

    assert records[0]["selected_precision"] == "Q8_0"
    assert records[0]["promotion_status"] == "unqualified"
    with pytest.raises(ValueError, match="blocked by unqualified"):
        validate_sensitive_profile_qualification(records)


def test_q8_promotion_rejects_arbitrary_qualification_evidence() -> None:
    record = {
        "name": "blk.0.ssm_alpha.weight",
        "shape": [48, 2560],
        "quant_name": "Q8_0",
        "quant_type": 8,
        "scale_representation": {
            "kind": "per-block",
            "location": "inline-ggml-block",
            "dtype": "F16",
            "required": True,
        },
    }

    with pytest.raises(ValueError, match="only baseline/unqualified are allowed"):
        build_sensitive_tensor_census(
            [record],
            profile="reference-q4",
            promotion_evidence={
                "gdn_in_proj_a": {
                    "status": "passed",
                    "evidence_id": "fixture-qualified-q8-v1",
                }
            },
        )


@pytest.mark.parametrize("status", ["qualified", "passed"])
def test_sensitive_validator_rejects_future_qualification_statuses(status: str) -> None:
    records = build_sensitive_tensor_census(_records(), profile="reference-q4")
    records[0]["promotion_status"] = status
    records[0]["promotion_evidence"] = "untrusted-fixture-id"

    with pytest.raises(ValueError, match="promotion metadata"):
        validate_sensitive_tensor_census(records)


def test_sensitive_policy_does_not_allow_expert_wildcards_to_capture_controls() -> None:
    records = build_sensitive_tensor_census(_records(), profile="reference-q4")
    assert not any("_exps." in str(record["identity"]) for record in records)
    validate_sensitive_tensor_census(
        records,
        expert_rule={"scope": "routed-expert-tensors-only", "sensitive_exclusion": True},
    )


@pytest.mark.parametrize(
    ("filename", "profile"),
    [
        ("qwen38-q4-census.metadata.json", "reference-q4"),
        ("qwen38-q3-census.metadata.json", "throughput-q3"),
    ],
)
def test_checked_in_qwen_profiles_carry_complete_sensitive_census(
    filename: str, profile: str
) -> None:
    census = json.loads((ROOT / "tests" / "fixtures" / "results" / filename).read_text())
    schema = json.loads((ROOT / "schemas" / "quant-census.schema.json").read_text())

    Draft202012Validator(schema).validate(census)
    records = census["sensitive_tensors"]
    validate_sensitive_tensor_census(records, expert_rule=census["sensitive_policy"]["expert_rule"])
    assert census["sensitive_policy"]["profile"] == profile
    assert len(records) == 783
    assert {record["class"] for record in records} >= {
        "moe_router",
        "shared_expert_gate",
        "gdn_in_proj_a",
        "gdn_in_proj_b",
        "gdn_state_projection",
        "gdn_control",
        "hyperconnection_control",
        "residual_write_gate",
        "norm",
    }
    assert all(
        "source=unsloth/Qwen3.8-Flash-Next-GGUF@c8b5954a88c2775c546b92593eda40ea041d3176"
        in record["conversion_provenance"]
        for record in records
    )
    assert {record["selected_precision"] for record in records} == {"F32", "Q8_0"}
    q8_records = [record for record in records if record["selected_precision"] == "Q8_0"]
    assert len(q8_records) == 266
    assert all(record["promotion_status"] == "unqualified" for record in q8_records)
    assert all(record["promotion_evidence"] is None for record in q8_records)
    with pytest.raises(ValueError, match="blocked by unqualified"):
        validate_sensitive_profile_qualification(records)


def test_schema_requires_sensitive_policy_and_records_together() -> None:
    census = json.loads(
        (ROOT / "tests" / "fixtures" / "results" / "qwen38-q4-census.metadata.json").read_text()
    )
    schema = json.loads((ROOT / "schemas" / "quant-census.schema.json").read_text())
    census.pop("sensitive_policy")

    errors = list(Draft202012Validator(schema).iter_errors(census))

    assert any("sensitive_policy" in error.message for error in errors)


def test_schema_rejects_future_qualification_statuses() -> None:
    census = json.loads(
        (ROOT / "tests" / "fixtures" / "results" / "qwen38-q4-census.metadata.json").read_text()
    )
    schema = json.loads((ROOT / "schemas" / "quant-census.schema.json").read_text())
    record = census["sensitive_tensors"][0]
    record["promotion_status"] = "qualified"
    record["promotion_evidence"] = "untrusted-fixture-id"

    errors = list(Draft202012Validator(schema).iter_errors(census))

    assert any(list(error.absolute_path)[-1:] == ["promotion_status"] for error in errors)


def test_q8_0_packed_scale_and_dequant_match_ggml_reference() -> None:
    fixture = json.loads((FIXTURES / "q8-0-packed-scale-parity.json").read_text(encoding="utf-8"))

    report = evaluate_q8_0_packed_scale_parity(fixture)

    assert report["passed"] is True
    assert report["max_abs"] == 0.0
    assert report["packed_sha256"] == fixture["packed_sha256"]
    assert report["decoded_f32_sha256"] == fixture["reference_f32_sha256"]
    assert_q8_0_packed_scale_parity(fixture)


def test_q8_0_packed_scale_parity_rejects_misscaled_payload() -> None:
    fixture = json.loads((FIXTURES / "q8-0-packed-scale-parity.json").read_text(encoding="utf-8"))
    packed = bytearray(base64.b64decode(fixture["packed_base64"], validate=True))
    # Replace the inline little-endian FP16 scale (0.125) with 0.25 while
    # preserving the signed payload codes and a valid artifact checksum.
    packed[:2] = bytes.fromhex("0034")
    fixture["packed_base64"] = base64.b64encode(packed).decode("ascii")
    fixture["packed_sha256"] = hashlib.sha256(packed).hexdigest()

    report = evaluate_q8_0_packed_scale_parity(fixture)

    assert report["passed"] is False
    with pytest.raises(ValueError, match="scale/dequant parity failed"):
        assert_q8_0_packed_scale_parity(fixture)


@pytest.mark.parametrize("name", ["shared-gate-mis-scaled.json", "gdn-control-perturbed.json"])
def test_controlled_sensitive_tensor_fixtures_are_rejected(name: str) -> None:
    fixture = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    report = evaluate_sensitive_tensor_fixture(fixture)
    assert report["rejected"] is True
    assert_sensitive_tensor_fixture_rejected(fixture)
