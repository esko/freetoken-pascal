from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest
from freetoken.cache_simulator import simulate_lru, validate_routing_trace
from freetoken.gguf_validation import inspect_gguf

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_qwen4_tiny_config_is_text_only_and_bounded() -> None:
    config = load_json(FIXTURES / "qwen4-tiny" / "config.json")
    text = config["text_config"]

    assert config["model_type"] == "qwen4_exp"
    assert "vision_config" not in config
    assert "mtp" not in config
    assert text["num_experts"] == 4
    assert text["num_experts_per_tok"] == 2
    assert text["num_hidden_layers"] == len(text["layer_types"]) == 2
    assert set(text["layer_types"]) == {"linear_attention", "qwen_sparse_attention"}
    rotary_dim = round(text["head_dim"] * text["rope_parameters"]["partial_rotary_factor"])
    assert sum(text["rope_parameters"]["mrope_section"]) * 2 == rotary_dim


def test_gguf_fixture_manifest_matches_bytes() -> None:
    manifest = load_json(FIXTURES / "gguf" / "manifest.json")

    assert manifest["license"] == "Apache-2.0"
    assert len(manifest["files"]) == 9
    for entry in manifest["files"]:
        data = (FIXTURES / "gguf" / entry["name"]).read_bytes()
        assert len(data) == entry["size"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]


def test_valid_gguf_fixture_has_heterogeneous_layout() -> None:
    census = inspect_gguf(FIXTURES / "gguf" / "valid-heterogeneous.gguf")

    assert census["version"] == 3
    assert census["alignment"] == 32
    assert [tensor["quant_type"] for tensor in census["tensors"]] == [2, 14, 0]
    assert [tensor["nbytes"] for tensor in census["tensors"]] == [36, 420, 32]


@pytest.mark.parametrize(
    "name",
    [
        "unknown-quant.gguf",
        "malformed-fastest-dim.gguf",
        "malformed-offset.gguf",
        "out-of-range-offset.gguf",
        "bad-magic.gguf",
        "truncated-metadata.gguf",
    ],
)
def test_malformed_gguf_fixtures_fail_closed(name: str) -> None:
    with pytest.raises(ValueError):
        inspect_gguf(FIXTURES / "gguf" / name)


def test_known_quant_outside_a_selected_capability_set_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported GGUF quant type"):
        inspect_gguf(
            FIXTURES / "gguf" / "unsupported-known-quant.gguf",
            supported_quant_types={0, 2, 14},
        )


def test_routing_trace_simulator_distinguishes_locality() -> None:
    locality = load_json(FIXTURES / "routing" / "locality-positive.json")
    adversarial = load_json(FIXTURES / "routing" / "adversarial.json")

    local_result = simulate_lru(locality, capacity=4)
    adversarial_result = simulate_lru(adversarial, capacity=4)

    assert local_result.hit_rate == pytest.approx(2 / 3)
    assert adversarial_result.hit_rate == 0
    assert local_result.hit_rate > adversarial_result.hit_rate


def test_cache_zero_is_a_stable_all_miss_control() -> None:
    trace = load_json(FIXTURES / "routing" / "locality-positive.json")

    result = simulate_lru(trace, capacity=0)

    assert result.hits == 0
    assert result.misses == result.requests == 12
    assert result.evictions == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda trace: trace.update(num_experts=0), "num_experts"),
        (lambda trace: trace["steps"][0].update(experts=[0, 0]), "duplicates"),
        (lambda trace: trace["steps"][0].update(experts=[8]), "out-of-range"),
    ],
)
def test_invalid_routing_traces_fail_closed(mutation, message: str) -> None:
    trace = load_json(FIXTURES / "routing" / "locality-positive.json")
    mutation(trace)

    with pytest.raises(ValueError, match=message):
        validate_routing_trace(trace)


def test_kernel_arithmetic_fixture_matches_independent_scalar_math() -> None:
    fixture = load_json(FIXTURES / "kernels" / "arithmetic.json")
    q4 = fixture["q4_0"]
    scale = struct.unpack("<e", bytes.fromhex(q4["scale_f16_le_hex"]))[0]
    packed = bytes.fromhex(q4["packed_nibbles_hex"])
    decoded = [(byte & 0x0F) - 8 for byte in packed]
    decoded.extend((byte >> 4) - 8 for byte in packed)

    assert [value * scale for value in decoded] == q4["expected"]

    merge = fixture["weighted_merge"]
    actual = [
        sum(
            weight * output[index]
            for weight, output in zip(merge["router_weights"], merge["expert_outputs"], strict=True)
        )
        for index in range(2)
    ]
    assert actual == merge["expected"]
