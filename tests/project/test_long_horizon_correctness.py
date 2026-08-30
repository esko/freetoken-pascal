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


def _observations(*, degraded: bool = False) -> dict[str, np.ndarray]:
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
    return {
        "continuation_tokens": continuation,
        "router_ids": router_ids,
        "semantic_output_tokens": semantic,
        "gdn_state": np.stack(states),
    }


def test_long_horizon_contract_declares_required_probe_families() -> None:
    contract = load_long_horizon_contract(CONTRACT)

    assert contract["minimum_steps"] == 16
    assert {probe["kind"] for probe in contract["probes"]} >= {
        "multi-turn-coding",
        "repeated-tool-calls",
        "state-dependent-reasoning",
        "structured-transform",
        "long-generation",
    }
    assert contract["sensitive_control"]["required_observation"] == "gdn_state"

    sensitive_fixture = ROOT / contract["sensitive_control"]["fixture"]
    report = assert_sensitive_tensor_fixture_rejected(sensitive_fixture)
    assert report["tensor_class"] == contract["sensitive_control"]["tensor_class"]


def test_long_horizon_output_validator_checks_every_turn() -> None:
    contract = load_long_horizon_contract(CONTRACT)
    outputs = {
        probe["id"]: [
            expectation["value"]
            if expectation["kind"] in {"contains", "exact"}
            else '{"path":"a.py","line":7}'
            for expectation in (step["expectation"] for step in probe["steps"])
        ]
        for probe in contract["probes"]
    }

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
        compare_long_horizon_bundles(subject, reference, contract=CONTRACT)


def test_degraded_gdn_control_fails_long_horizon_state_gate(tmp_path: Path) -> None:
    subject = tmp_path / "subject.ftobs"
    reference = tmp_path / "reference.ftobs"
    write_observation_bundle(subject, _identity("freetoken"), _observations(degraded=True))
    write_observation_bundle(reference, _identity("independent"), _observations())

    evidence = compare_long_horizon_bundles(subject, reference, contract=CONTRACT)

    assert evidence["passed"] is False
    assert evidence["long_horizon"]["minimum_steps"] == 16
    assert evidence["long_horizon"]["sensitive_control"]["tensor_class"] == "gdn_in_proj_a"
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
