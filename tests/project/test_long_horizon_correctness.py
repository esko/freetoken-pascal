from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from freetoken.long_horizon_correctness import (
    compare_long_horizon_bundles,
    load_long_horizon_contract,
    validate_long_horizon_outputs,
)
from freetoken.reference_correctness import write_observation_bundle
from freetoken.sensitive_census import assert_sensitive_tensor_fixture_rejected

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "tests/fixtures/qwen38-long-horizon-contract.json"
CONTROL_FIXTURE = ROOT / "tests/fixtures/sensitive/gdn-control-perturbed.json"


def _identity(implementation: str) -> dict[str, object]:
    return {
        "implementation": implementation,
        "revision": "1" * 40 if implementation == "freetoken" else "2" * 40,
        "commit": "3" * 40,
        "tokenizer_repository": "unsloth/Qwen3.8-Flash-Next-GGUF",
        "tokenizer_revision": "c8b5954a88c2775c546b92593eda40ea041d3176",
        "artifact_sha256": "4" * 64,
        "quant_census_sha256": "5" * 64,
        "corpus_sha256": "6" * 64,
        "prompt_id": "long-horizon-fixture",
        "prompt_sha256": "7" * 64,
        "quantization": "UD-Q4_K_XL",
        "dtype": "float32",
        "cache_mode": "disabled",
        "execution_mode": "fixture-reference",
        "context_tokens": 32,
    }


def _observations(*, degraded: bool = False, control: str = "candidate") -> dict[str, np.ndarray]:
    steps = 16
    continuation = np.arange(100, 100 + steps, dtype=np.int32)
    router_ids = np.stack(
        [np.array([step % 4, (step + 1) % 4], dtype=np.int32) for step in range(steps)]
    )
    semantic = np.stack(
        [np.array([step, step + 1, step + 2], dtype=np.int32) for step in range(steps)]
    )
    current = np.zeros(2, dtype=np.float32)
    states = []
    for step in range(steps):
        current = current * np.float32(0.875) + np.array([1.0, -0.5], dtype=np.float32)
        if degraded and step >= 7:
            current = current + np.array([0.002, 0.0], dtype=np.float32)
        states.append(current.copy())
    control_values = json.loads(CONTROL_FIXTURE.read_text(encoding="utf-8"))[control]["values"]
    control_input = np.asarray(control_values, dtype=np.float32)
    return {
        "continuation_tokens": continuation,
        "router_ids": router_ids,
        "semantic_output_tokens": semantic,
        "gdn_state": np.stack(states),
        "sensitive_control_input": np.repeat(control_input[None, :], steps, axis=0),
    }


def _outputs(contract: dict[str, object]) -> dict[str, list[str]]:
    return {
        probe["id"]: [
            step["expectation"]["value"]
            if step["expectation"]["kind"] in {"contains", "exact"}
            else '{"path":"a.py","line":7}'
            for step in probe["steps"]
        ]
        for probe in contract["probes"]
    }


def test_long_horizon_contract_declares_required_probe_families() -> None:
    contract = load_long_horizon_contract(CONTRACT)

    assert contract["minimum_steps"] == 16
    assert all(len(probe["steps"]) >= contract["minimum_steps"] for probe in contract["probes"])
    assert {probe["kind"] for probe in contract["probes"]} >= {
        "multi-turn-coding",
        "repeated-tool-calls",
        "state-dependent-reasoning",
        "structured-transform",
        "long-generation",
    }
    assert contract["sensitive_control"]["required_observation"] == "gdn_state"
    assert contract["sensitive_control"]["input_observation"] == "sensitive_control_input"

    sensitive_fixture = CONTRACT.parent / contract["sensitive_control"]["fixture"]
    report = assert_sensitive_tensor_fixture_rejected(sensitive_fixture)
    assert report["tensor_class"] == contract["sensitive_control"]["tensor_class"]


def test_long_horizon_output_validator_checks_every_turn() -> None:
    contract = load_long_horizon_contract(CONTRACT)
    outputs = _outputs(contract)

    report = validate_long_horizon_outputs(contract, outputs)

    assert report["passed"] is True
    assert all(item["passed"] for item in report["probes"])

    broken = copy.deepcopy(outputs)
    broken["repeated-tool-calls"][2] = "wrong-result"
    report = validate_long_horizon_outputs(contract, broken)

    assert report["passed"] is False
    assert any(item["errors"] for item in report["probes"] if item["id"] == "repeated-tool-calls")


def test_long_horizon_bundle_requires_minimum_state_horizon(tmp_path: Path) -> None:
    short = {name: value[:8] for name, value in _observations().items()}
    subject = tmp_path / "subject.ftobs"
    reference = tmp_path / "reference.ftobs"
    write_observation_bundle(subject, _identity("freetoken"), short)
    write_observation_bundle(reference, _identity("independent"), short)

    with pytest.raises(ValueError, match="minimum horizon"):
        compare_long_horizon_bundles(
            subject,
            reference,
            contract=CONTRACT,
            subject_outputs=_outputs(load_long_horizon_contract(CONTRACT)),
            reference_outputs=_outputs(load_long_horizon_contract(CONTRACT)),
        )


def test_degraded_gdn_control_fails_long_horizon_state_gate(tmp_path: Path) -> None:
    subject = tmp_path / "subject.ftobs"
    reference = tmp_path / "reference.ftobs"
    write_observation_bundle(subject, _identity("freetoken"), _observations(degraded=True))
    write_observation_bundle(
        reference, _identity("independent"), _observations(control="reference")
    )
    contract = load_long_horizon_contract(CONTRACT)
    outputs = _outputs(contract)

    evidence = compare_long_horizon_bundles(
        subject,
        reference,
        contract=contract,
        subject_outputs=outputs,
        reference_outputs=outputs,
    )

    assert evidence["passed"] is False
    assert evidence["long_horizon"]["minimum_steps"] == 16
    assert evidence["long_horizon"]["sensitive_control"]["tensor_class"] == "gdn_in_proj_a"
    assert evidence["long_horizon"]["sensitive_control"]["declared_perturbation"] == {
        "index": 1,
        "delta": 0.07,
    }
    assert evidence["long_horizon"]["sensitive_control"]["control_input"]["passed"] is True
    assert (
        evidence["long_horizon"]["sensitive_control"]["control_input"]["subject_matches_candidate"]
        is True
    )
    assert evidence["long_horizon"]["semantic"]["passed"] is True
    state_comparisons = [
        item for item in evidence["comparisons"] if item["observation"] == "gdn_state"
    ]
    assert state_comparisons
    assert any(item["passed"] is False for item in state_comparisons)
    # Semantic outputs, routes, and continuation IDs remain unchanged in this
    # positive control; only accumulated state drift should fail the gate.
    assert all(
        item["passed"] for item in evidence["comparisons"] if item["observation"] != "gdn_state"
    )
    control_comparisons = [
        item for item in evidence["comparisons"] if item["observation"] == "sensitive_control_input"
    ]
    assert control_comparisons == [
        {
            "observation": "sensitive_control_input",
            "metric": "expected_control_difference",
            "observed": True,
            "limit": True,
            "passed": True,
        }
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sensitive_control", None, "sensitive_control"),
        ("minimum_steps", True, "minimum_steps"),
    ],
)
def test_long_horizon_contract_rejects_malformed_documents(
    field: str, value: object, message: str, tmp_path: Path
) -> None:
    document = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if value is None:
        document.pop(field)
    else:
        document[field] = value
    path = tmp_path / "invalid-contract.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_long_horizon_contract(path)


def test_long_horizon_contract_rejects_short_semantic_probe(tmp_path: Path) -> None:
    document = json.loads(CONTRACT.read_text(encoding="utf-8"))
    document["probes"][0]["steps"] = document["probes"][0]["steps"][:8]
    path = tmp_path / "short-probe.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="minimum horizon"):
        load_long_horizon_contract(path)


def test_long_horizon_contract_requires_declared_perturbation_field(tmp_path: Path) -> None:
    document = json.loads(CONTRACT.read_text(encoding="utf-8"))
    document["sensitive_control"]["candidate_field"] = "dtype"
    path = tmp_path / "wrong-perturbation-field.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=r"candidate_field.*perturbation"):
        load_long_horizon_contract(path)


def test_long_horizon_gate_fails_semantic_output_even_when_state_matches(tmp_path: Path) -> None:
    subject = tmp_path / "subject.ftobs"
    reference = tmp_path / "reference.ftobs"
    observations = _observations()
    write_observation_bundle(subject, _identity("freetoken"), observations)
    write_observation_bundle(
        reference, _identity("independent"), _observations(control="reference")
    )
    contract = load_long_horizon_contract(CONTRACT)
    subject_outputs = _outputs(contract)
    reference_outputs = _outputs(contract)
    # Both strings satisfy the declared contains probe, but deterministic A/B
    # evidence still requires the subject/reference semantic trajectory to match.
    subject_outputs["multi-turn-coding"][15] = "tests pass; subject variant"

    evidence = compare_long_horizon_bundles(
        subject,
        reference,
        contract=contract,
        subject_outputs=subject_outputs,
        reference_outputs=reference_outputs,
    )

    assert evidence["passed"] is False
    assert evidence["long_horizon"]["semantic"]["passed"] is False
    assert evidence["long_horizon"]["semantic"]["subject"]["passed"] is True
    assert evidence["long_horizon"]["semantic"]["reference"]["passed"] is True
    assert any(
        item["passed"] is False
        for item in evidence["long_horizon"]["semantic"]["comparisons"]
        if item["probe"] == "multi-turn-coding" and item["step"] == "verify"
    )


def test_long_horizon_gate_rejects_fixture_class_substitution(tmp_path: Path) -> None:
    document = json.loads(CONTRACT.read_text(encoding="utf-8"))
    document["sensitive_control"]["tensor_class"] = "gdn_control"
    path = tmp_path / "wrong-class.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="tensor class disagrees"):
        load_long_horizon_contract(path)


def test_long_horizon_gate_rejects_clean_identical_control_inputs(tmp_path: Path) -> None:
    subject = tmp_path / "subject.ftobs"
    reference = tmp_path / "reference.ftobs"
    write_observation_bundle(subject, _identity("freetoken"), _observations(control="reference"))
    write_observation_bundle(
        reference, _identity("independent"), _observations(control="reference")
    )
    contract = load_long_horizon_contract(CONTRACT)
    outputs = _outputs(contract)

    evidence = compare_long_horizon_bundles(
        subject,
        reference,
        contract=contract,
        subject_outputs=outputs,
        reference_outputs=outputs,
    )

    assert evidence["passed"] is False
    control_report = evidence["long_horizon"]["sensitive_control"]["control_input"]
    assert control_report["passed"] is False
    assert control_report["subject_matches_candidate"] is False
    assert control_report["reference_matches_reference"] is True


def test_long_horizon_gate_rejects_wrong_control_input_row(tmp_path: Path) -> None:
    subject = tmp_path / "subject.ftobs"
    reference = tmp_path / "reference.ftobs"
    subject_observations = _observations()
    subject_observations["sensitive_control_input"][4, 1] += np.float32(0.01)
    write_observation_bundle(subject, _identity("freetoken"), subject_observations)
    write_observation_bundle(
        reference, _identity("independent"), _observations(control="reference")
    )
    contract = load_long_horizon_contract(CONTRACT)
    outputs = _outputs(contract)

    evidence = compare_long_horizon_bundles(
        subject,
        reference,
        contract=contract,
        subject_outputs=outputs,
        reference_outputs=outputs,
    )

    assert evidence["passed"] is False
    control_report = evidence["long_horizon"]["sensitive_control"]["control_input"]
    assert control_report["passed"] is False
    assert control_report["subject_matches_candidate"] is False
    assert control_report["reference_matches_reference"] is True
