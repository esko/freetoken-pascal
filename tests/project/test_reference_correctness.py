from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from freetoken.reference_correctness import (
    Tolerance,
    compare_observation_bundles,
    load_prompt_corpus,
    materialize_prompt_case,
    numbered_record_context,
    prompt_corpus_sha256,
    read_observation_bundle,
    validate_probe_output,
    write_observation_bundle,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests/fixtures/qwen38-reference-corpus.json"
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_evidence", ROOT / "scripts/validate_evidence.py"
)
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
VALIDATE_EVIDENCE = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(VALIDATE_EVIDENCE)


def _identity(implementation: str, *, mode: str = "incremental") -> dict[str, object]:
    return {
        "implementation": implementation,
        "revision": "1" * 40 if implementation == "freetoken" else "2" * 40,
        "commit": "3" * 40,
        "artifact_sha256": "4" * 64,
        "quant_census_sha256": "5" * 64,
        "corpus_sha256": "6" * 64,
        "prompt_id": "factual-short",
        "prompt_sha256": "7" * 64,
        "quantization": "UD-Q4_K_XL",
        "dtype": "float32",
        "cache_mode": "disabled",
        "execution_mode": mode,
        "context_tokens": 8,
    }


def _observations(delta: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "router_ids": np.array([[2, 7]], dtype=np.int32),
        "router_weights": np.array([[0.75 + delta, 0.25 - delta]], dtype=np.float32),
        "gdn_state": np.array([[1.0 + delta, -2.0]], dtype=np.float32),
        "qsa_blocks": np.array([[0, 4, 9]], dtype=np.int32),
        "qsa_state": np.array([[0.5, 1.5 + delta]], dtype=np.float32),
        "ple_contribution": np.array([[0.125, -0.25 + delta]], dtype=np.float32),
        "logits": np.array([[2.0, 1.0 + delta, -3.0]], dtype=np.float32),
        "continuation_tokens": np.array([11, 12, 13], dtype=np.int32),
    }


def test_reference_corpus_covers_required_workloads_and_targets() -> None:
    corpus = load_prompt_corpus(CORPUS)

    assert {case["category"] for case in corpus["cases"]} >= {
        "factual",
        "code",
        "tool-json",
        "repetitive-edit",
        "long-retrieval",
    }
    assert {case["context"]["target_tokens"] for case in corpus["cases"] if "context" in case} == {
        32768,
        128000,
        262000,
    }


@pytest.mark.parametrize(
    ("case_id", "target_tokens"),
    [
        ("retrieval-32k", 32768),
        ("retrieval-128k", 128000),
        ("retrieval-262k-qualification", 262000),
    ],
)
def test_materialized_context_has_exact_declared_token_count(
    case_id: str, target_tokens: int
) -> None:
    corpus = load_prompt_corpus(CORPUS)
    case = next(case for case in corpus["cases"] if case["id"] == case_id)
    needle = case["context"]["needle"]
    # This deliberately uses a tiny deterministic fixture tokenizer. The rendered
    # prompt has eight fixed words outside CONTEXT, so the context contributes the
    # remaining words and the materializer must prove the exact total.
    context_text = " ".join(["fixture-token"] * (target_tokens - 9) + [needle])

    def fixture_token_counter(messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"].split()) for message in messages)

    materialized = materialize_prompt_case(
        corpus,
        case_id,
        context_text=context_text,
        token_counter=fixture_token_counter,
    )

    assert materialized["context_tokens"] == target_tokens
    assert needle in materialized["messages"][0]["content"]
    assert len(materialized["prompt_sha256"]) == 64


def test_materializer_rejects_a_context_token_count_mismatch() -> None:
    corpus = load_prompt_corpus(CORPUS)

    with pytest.raises(ValueError, match="token count mismatch"):
        materialize_prompt_case(
            corpus,
            "retrieval-32k",
            context_text="too short",
            token_counter=lambda messages: 1,
        )


def test_numbered_context_generation_is_deterministic() -> None:
    first = numbered_record_context(8, needle_fraction=0.5, needle="FT-PASCAL")
    second = numbered_record_context(8, needle_fraction=0.5, needle="FT-PASCAL")

    assert first == second
    assert "record 00000004: FT-PASCAL" in first


def test_prompt_corpus_digest_is_canonical_and_content_bound() -> None:
    corpus = load_prompt_corpus(CORPUS)
    original_digest = prompt_corpus_sha256(corpus)
    changed = copy.deepcopy(corpus)
    changed["cases"][0]["messages"][0]["content"] += " Be precise."

    assert original_digest == prompt_corpus_sha256(corpus)
    assert original_digest != prompt_corpus_sha256(changed)


def test_tool_json_probe_requires_valid_json_and_declared_types() -> None:
    case = next(
        case for case in load_prompt_corpus(CORPUS)["cases"] if case["id"] == "tool-json-short"
    )

    assert validate_probe_output(case, '{"path":"a.py","line":7}') == []
    assert "JSON" in validate_probe_output(case, "```json\n{}\n```")[0]
    assert "integer" in validate_probe_output(case, '{"path":"a.py","line":"7"}')[0]


def test_observation_bundle_roundtrips_without_pickle(tmp_path: Path) -> None:
    path = tmp_path / "subject.ftobs"
    write_observation_bundle(path, _identity("freetoken"), _observations())

    identity, observations = read_observation_bundle(path)

    assert identity == _identity("freetoken")
    assert set(observations) == set(_observations())
    for name, expected in _observations().items():
        np.testing.assert_array_equal(observations[name], expected)


def test_observation_bundle_is_byte_for_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.ftobs"
    second = tmp_path / "second.ftobs"

    write_observation_bundle(first, _identity("freetoken"), _observations())
    write_observation_bundle(second, _identity("freetoken"), _observations())

    assert first.read_bytes() == second.read_bytes()
    assert (
        hashlib.sha256(first.read_bytes()).hexdigest()
        == hashlib.sha256(second.read_bytes()).hexdigest()
    )


def test_corrupt_observation_payload_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "subject.ftobs"
    write_observation_bundle(path, _identity("freetoken"), _observations())
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x80
    path.write_bytes(data)

    with pytest.raises(ValueError, match=r"corrupt|digest|CRC"):
        read_observation_bundle(path)


def test_reader_rejects_empty_observation_arrays(tmp_path: Path) -> None:
    payload_buffer = io.BytesIO()
    np.save(payload_buffer, np.empty((0,), dtype=np.float32), allow_pickle=False)
    payload = payload_buffer.getvalue()
    manifest = {
        "schema_name": "qwen38-observation-bundle",
        "schema_version": 1,
        "identity": _identity("freetoken"),
        "observations": {
            "empty": {
                "member": "arrays/empty.npy",
                "dtype": "<f4",
                "shape": [0],
                "nbytes": 0,
                "payload_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    with zipfile.ZipFile(tmp_path / "empty.ftobs", "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        archive.writestr("arrays/empty.npy", payload)

    with pytest.raises(ValueError, match="empty"):
        read_observation_bundle(tmp_path / "empty.ftobs")


@pytest.mark.parametrize("key", ["commit", "revision", "prompt_id"])
def test_bundle_rejects_wrong_identity_types(tmp_path: Path, key: str) -> None:
    identity = _identity("freetoken")
    identity[key] = 7

    with pytest.raises(ValueError, match="identity"):
        write_observation_bundle(tmp_path / f"bad-{key}.ftobs", identity, _observations())


def test_comparison_checks_exact_routing_and_numeric_tolerances(tmp_path: Path) -> None:
    subject = tmp_path / "subject.ftobs"
    reference = tmp_path / "reference.ftobs"
    write_observation_bundle(subject, _identity("freetoken"), _observations(1e-5))
    write_observation_bundle(reference, _identity("vllm"), _observations())

    evidence = compare_observation_bundles(
        subject,
        reference,
        tolerances={
            name: Tolerance(max_abs=2e-5, relative_rms=5e-5, min_cosine=0.99999)
            for name in (
                "router_weights",
                "gdn_state",
                "qsa_state",
                "ple_contribution",
                "logits",
            )
        },
        exact_observations={"router_ids", "qsa_blocks", "continuation_tokens"},
    )

    assert evidence["passed"] is True
    assert {comparison["observation"] for comparison in evidence["comparisons"]} == set(
        _observations()
    )
    assert all(comparison["passed"] for comparison in evidence["comparisons"])


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("artifact_sha256", "8" * 64, "artifact_sha256"),
        ("quantization", "UD-Q3_K_XL", "quantization"),
        ("prompt_sha256", "9" * 64, "prompt_sha256"),
        ("context_tokens", 9, "context_tokens"),
    ],
)
def test_comparison_rejects_different_model_quant_or_workload(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    subject = tmp_path / "subject.ftobs"
    reference = tmp_path / "reference.ftobs"
    changed = copy.deepcopy(_identity("vllm"))
    changed[key] = value
    write_observation_bundle(subject, _identity("freetoken"), _observations())
    write_observation_bundle(reference, changed, _observations())

    with pytest.raises(ValueError, match=message):
        compare_observation_bundles(subject, reference, tolerances={}, exact_observations=set())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("corpus_sha256", "8" * 64),
        ("prompt_id", "different-prompt"),
        ("cache_mode", "static"),
        ("quant_census_sha256", "9" * 64),
    ],
)
def test_comparison_binds_every_workload_identity_field(
    tmp_path: Path, key: str, value: object
) -> None:
    subject = tmp_path / "subject.ftobs"
    reference = tmp_path / "reference.ftobs"
    changed = copy.deepcopy(_identity("vllm"))
    changed[key] = value
    write_observation_bundle(subject, _identity("freetoken"), _observations())
    write_observation_bundle(reference, changed, _observations())

    with pytest.raises(ValueError, match=key):
        compare_observation_bundles(
            subject,
            reference,
            tolerances={},
            exact_observations=set(_observations()),
        )


def test_comparison_requires_an_independent_reference_by_default(tmp_path: Path) -> None:
    subject = tmp_path / "subject.ftobs"
    reference = tmp_path / "reference.ftobs"
    write_observation_bundle(subject, _identity("freetoken"), _observations())
    write_observation_bundle(reference, _identity("freetoken"), _observations())

    with pytest.raises(ValueError, match="independent"):
        compare_observation_bundles(subject, reference, tolerances={}, exact_observations=set())


def test_comparison_rejects_different_labels_on_same_reference_revision(tmp_path: Path) -> None:
    subject = tmp_path / "subject.ftobs"
    reference = tmp_path / "reference.ftobs"
    subject_identity = _identity("freetoken")
    reference_identity = _identity("vllm")
    reference_identity["revision"] = subject_identity["revision"]
    reference_identity["commit"] = subject_identity["commit"]
    write_observation_bundle(subject, subject_identity, _observations())
    write_observation_bundle(reference, reference_identity, _observations())

    with pytest.raises(ValueError, match="independent"):
        compare_observation_bundles(
            subject,
            reference,
            tolerances={},
            exact_observations=set(_observations()),
        )


def test_nonfinite_observation_fails_before_comparison(tmp_path: Path) -> None:
    observations = _observations()
    observations["logits"][0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        write_observation_bundle(tmp_path / "bad.ftobs", _identity("freetoken"), observations)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document["tokenizer"].pop("repository"),
        lambda document: document["tokenizer"].__setitem__("repository", 7),
        lambda document: document["sampling"].pop("top_p"),
        lambda document: document["sampling"].__setitem__("temperature", "0.0"),
        lambda document: document["cases"][4]["context"].__setitem__("needle_fraction", "0.5"),
        lambda document: document["cases"][2]["expectation"].__setitem__("required", []),
        lambda document: document["cases"][0]["expectation"].pop("value"),
    ],
)
def test_prompt_corpus_rejects_malformed_types_and_required_fields(tmp_path: Path, mutate) -> None:
    document = copy.deepcopy(json.loads(CORPUS.read_text(encoding="utf-8")))
    mutate(document)
    path = tmp_path / "invalid-corpus.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid prompt corpus"):
        load_prompt_corpus(path)


def test_prompt_probe_rejects_malformed_expectation_payloads() -> None:
    case = {"expectation": {"kind": "contains"}}
    with pytest.raises(ValueError, match="expectation"):
        validate_probe_output(case, "anything")


def test_evidence_validator_rejects_nonfinite_json_numbers() -> None:
    evidence = json.loads(
        (ROOT / "tests/fixtures/results/correctness.json").read_text(encoding="utf-8")
    )
    evidence["comparisons"][0]["observed"] = float("nan")

    errors = VALIDATE_EVIDENCE.validate_document(evidence, schema_dir=ROOT / "schemas")

    assert any("finite" in error.lower() or "nan" in error.lower() for error in errors)


@pytest.mark.parametrize("party", ["subject", "reference"])
def test_evidence_validator_binds_each_party_to_the_declared_workload(party: str) -> None:
    evidence = json.loads(
        (ROOT / "tests/fixtures/results/correctness.json").read_text(encoding="utf-8")
    )
    evidence[party]["prompt_id"] = "different-prompt"

    errors = VALIDATE_EVIDENCE.validate_document(evidence, schema_dir=ROOT / "schemas")

    assert any(f"{party}.prompt_id" in error for error in errors)


def test_semantic_state_restore_matches_uninterrupted_and_reset_is_clean(tmp_path: Path) -> None:
    def run(values: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outputs = []
        current = state.copy()
        for value in values:
            current = current * np.float32(0.75) + value
            outputs.append(current.copy())
        return np.stack(outputs), current

    prefix = np.array([[1.0, -1.0], [0.5, 2.0]], dtype=np.float32)
    continuation = np.array([[3.0, 0.25], [-2.0, 1.5]], dtype=np.float32)
    _, prefix_state = run(prefix, np.zeros(2, dtype=np.float32))
    expected, _ = run(continuation, prefix_state)
    checkpoint = tmp_path / "state.ftobs"
    write_observation_bundle(checkpoint, _identity("freetoken"), {"gdn_state": prefix_state})

    _, restored = read_observation_bundle(checkpoint)
    actual, _ = run(continuation, restored["gdn_state"])
    reset, _ = run(continuation, np.zeros(2, dtype=np.float32))

    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(reset, expected)
    reset_again, _ = run(continuation, np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(reset_again, reset)


def test_observation_cli_emits_schema_v2_comparison(tmp_path: Path) -> None:
    identity_subject = tmp_path / "subject.json"
    identity_reference = tmp_path / "reference.json"
    arrays_subject = tmp_path / "subject.npz"
    arrays_reference = tmp_path / "reference.npz"
    subject = tmp_path / "subject.ftobs"
    reference = tmp_path / "reference.ftobs"
    output = tmp_path / "evidence.json"
    identity_subject.write_text(json.dumps(_identity("freetoken")), encoding="utf-8")
    identity_reference.write_text(json.dumps(_identity("vllm")), encoding="utf-8")
    np.savez(arrays_subject, **_observations(1e-5))
    np.savez(arrays_reference, **_observations())
    clean_env = {"PATH": str(Path(sys.executable).parent)}

    for identity, arrays, bundle in (
        (identity_subject, arrays_subject, subject),
        (identity_reference, arrays_reference, reference),
    ):
        subprocess.run(
            [
                sys.executable,
                "scripts/write_qwen38_observations.py",
                "--identity",
                str(identity),
                "--arrays",
                str(arrays),
                "--output",
                str(bundle),
            ],
            cwd=ROOT,
            env=clean_env,
            check=True,
            capture_output=True,
            text=True,
        )
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_qwen38_observations.py",
            str(subject),
            str(reference),
            "--contract",
            str(ROOT / "tests/fixtures/qwen38-reference-contract.json"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
    )

    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == 2
    assert evidence["passed"] is True
    assert evidence["workload"] == {
        key: _identity("freetoken")[key]
        for key in ("corpus_sha256", "prompt_id", "prompt_sha256", "context_tokens")
    }
