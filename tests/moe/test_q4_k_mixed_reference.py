"""Q4K adapter integration over the Qwen3.8 mixed projection census."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from freetoken.moe.cpu_abi import CpuExpertDescriptor, CpuExpertLayout, UnsupportedQuantType
from freetoken.moe.ggml_reference import (
    IQ3_XXS_BLOCK_BYTES,
    IQ4_NL_BLOCK_BYTES,
    IQ4_XS_BLOCK_BYTES,
    Q5_1_BLOCK_BYTES,
    Q5_K_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES,
)
from freetoken.moe.q4_k import Q4K_BLOCK_BYTES, Q4KExecutor


class _PackedSource:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.range_offset = 0
        self.range_size = int(values.nbytes)
        self.source_address = int(values.__array_interface__["data"][0])

    def expert_packed(self, expert: int) -> np.ndarray:
        return self.values[expert]


def _half_bytes(value: float) -> np.ndarray:
    return np.frombuffer(np.asarray(np.float16(value), dtype="<f2").tobytes(), dtype=np.uint8)


def _unit_block(block_bytes: int, *, high_bits: bool = False) -> np.ndarray:
    block = np.zeros(block_bytes, dtype=np.uint8)
    block[:2] = _half_bytes(1.0)
    if block_bytes == Q4K_BLOCK_BYTES:
        block[4:16] = 1
        block[16:] = 0x11
    elif block_bytes == Q5_K_BLOCK_BYTES:
        block[4:16] = 1
        block[48:] = 0x11
        if high_bits:
            block[16:48] = 0xFF
    elif block_bytes == Q5_1_BLOCK_BYTES:
        block[8:] = 0x11
        if high_bits:
            block[4:8] = 0xFF
    elif block_bytes == Q8_0_BLOCK_BYTES:
        block[2:] = 1
    elif block_bytes == IQ3_XXS_BLOCK_BYTES:
        # IQ3_XXS index 0 is four magnitudes of 4 with a scale of 1/4.
        pass
    elif block_bytes == IQ4_NL_BLOCK_BYTES:
        block[2:] = 0x88
    elif block_bytes == IQ4_XS_BLOCK_BYTES:
        block[2:4] = np.frombuffer(np.uint16(0xAAAA).tobytes(), dtype=np.uint8)
        block[4:8] = 0x11
        block[8:] = 0x88
    else:
        raise AssertionError(block_bytes)
    return block


def _source(experts: int, output_dim: int, input_dim: int, block_bytes: int) -> _PackedSource:
    wide_block_bytes = (
        Q4K_BLOCK_BYTES,
        Q5_K_BLOCK_BYTES,
        IQ3_XXS_BLOCK_BYTES,
        IQ4_XS_BLOCK_BYTES,
    )
    blocks = input_dim // (256 if block_bytes in wide_block_bytes else 32)
    block = _unit_block(block_bytes)
    rows = np.tile(block, blocks * output_dim * experts).reshape(
        experts, output_dim, blocks * block_bytes
    )
    return _PackedSource(np.ascontiguousarray(rows))


def _descriptor(
    layer: int,
    projection: str,
    quant_type: int,
    quant_name: str,
    source: _PackedSource,
    *,
    experts: int = 2,
    output_dim: int = 256,
    input_dim: int = 256,
) -> CpuExpertDescriptor:
    block = 256 if quant_name in {"Q4_K", "Q5_K", "IQ3_XXS", "IQ4_XS"} else 32
    block_bytes = {
        "Q4_K": Q4K_BLOCK_BYTES,
        "Q5_K": Q5_K_BLOCK_BYTES,
        "Q5_1": Q5_1_BLOCK_BYTES,
        "Q8_0": Q8_0_BLOCK_BYTES,
        "IQ3_XXS": IQ3_XXS_BLOCK_BYTES,
        "IQ4_NL": IQ4_NL_BLOCK_BYTES,
        "IQ4_XS": IQ4_XS_BLOCK_BYTES,
        "MYSTERY": Q8_0_BLOCK_BYTES,
    }[quant_name]
    row_bytes = input_dim // block * block_bytes
    return CpuExpertDescriptor(
        layer_id=layer,
        projection=projection,
        quant_type=quant_type,
        quant_name=quant_name,
        num_experts=experts,
        output_dim=output_dim,
        input_dim=input_dim,
        rows_per_expert=output_dim,
        row_stride_bytes=row_bytes,
        expert_stride_bytes=output_dim * row_bytes,
        tensor_bytes=experts * output_dim * row_bytes,
        source=source,
    )


def _pinned_promoted_layers() -> tuple[int, ...]:
    census_path = (
        Path(__file__).resolve().parents[2]
        / "tests/fixtures/results/qwen38-q4-census.metadata.json"
    )
    census = json.loads(census_path.read_text(encoding="utf-8"))
    return tuple(
        int(item["layer"]) for item in census["expert_layers"] if "Q8_0" in item["quant_types"]
    )


def test_q4k_executor_registers_builtin_decoders_for_actual_mixed_layers() -> None:
    experts = 2
    descriptors = []
    promoted_layers = _pinned_promoted_layers()
    assert promoted_layers == (2, 4, 30, 46, 47)
    layers = (0, *promoted_layers)
    for layer in layers:
        if layer == 2:
            gate_up_quant, gate_up_name = 13, "Q5_K"
        else:
            gate_up_quant, gate_up_name = 12, "Q4_K"
        if layer in promoted_layers:
            down_quant, down_name = 8, "Q8_0"
        else:
            down_quant, down_name = 7, "Q5_1"
        gate_up_block = Q4K_BLOCK_BYTES if gate_up_name == "Q4_K" else Q5_K_BLOCK_BYTES
        for projection in ("gate", "up"):
            descriptors.append(
                _descriptor(
                    layer,
                    projection,
                    gate_up_quant,
                    gate_up_name,
                    _source(experts, 256, 256, gate_up_block),
                    experts=experts,
                )
            )
        down_block = Q5_1_BLOCK_BYTES if down_name == "Q5_1" else Q8_0_BLOCK_BYTES
        descriptors.append(
            _descriptor(
                layer,
                "down",
                down_quant,
                down_name,
                _source(experts, 256, 256, down_block),
                experts=experts,
            )
        )

    for descriptor in descriptors:
        source = descriptor.source
        assert source is not None
        for expert in (0, experts // 2, experts - 1):
            actual_address = int(source.expert_packed(expert).__array_interface__["data"][0])
            expected_address = descriptor.source_address + expert * descriptor.expert_stride_bytes
            assert actual_address == expected_address

    executor = Q4KExecutor(CpuExpertLayout(tuple(descriptors), top_k=2), mode="scalar")
    executor.prepare(max_tokens=1, max_routes=2)
    hidden = np.full((1, 256), 0.01, dtype=np.float32)
    ids = np.array([[1, 1]], dtype=np.int32)
    weights = np.array([[0.25, 0.75]], dtype=np.float32)

    for layer in (0, 2, 4):
        result = executor.execute(layer, hidden, ids, weights)
        gate = np.full(256, 2.56, dtype=np.float32)
        up = np.full(256, 2.56, dtype=np.float32)
        activated = gate / (1.0 + np.exp(-gate)) * up
        expected = np.full(256, np.sum(activated), dtype=np.float32)
        np.testing.assert_allclose(result.output[0], expected, rtol=2e-6, atol=2e-6)
        expected_census = {
            0: ("q4_k_scalar", "reference_q5_1"),
            2: ("reference_q5_k", "reference_q8_0"),
            4: ("q4_k_scalar", "reference_q8_0"),
        }[layer]
        expected_backend = "reference" if layer == 2 else "mixed"
        expected_fallback = (
            "reference_dequant_packed_workspace" if layer == 2 else "mixed_reference_formats"
        )
        assert result.telemetry.backend == expected_backend
        assert result.telemetry.kernel_census == expected_census
        assert result.telemetry.fallback_reason == expected_fallback


def test_q4k_executor_dispatches_all_qwen38_q3_reference_formats() -> None:
    experts = 2
    descriptors = []
    layer_formats = {
        0: (18, "IQ3_XXS", 20, "IQ4_NL"),
        2: (23, "IQ4_XS", 8, "Q8_0"),
    }
    for layer, (gate_up_quant, gate_up_name, down_quant, down_name) in layer_formats.items():
        gate_up_bytes = {
            "IQ3_XXS": IQ3_XXS_BLOCK_BYTES,
            "IQ4_XS": IQ4_XS_BLOCK_BYTES,
        }[gate_up_name]
        for projection in ("gate", "up"):
            descriptors.append(
                _descriptor(
                    layer,
                    projection,
                    gate_up_quant,
                    gate_up_name,
                    _source(experts, 256, 256, gate_up_bytes),
                    experts=experts,
                )
            )
        down_bytes = {"IQ4_NL": IQ4_NL_BLOCK_BYTES, "Q8_0": Q8_0_BLOCK_BYTES}[down_name]
        descriptors.append(
            _descriptor(
                layer,
                "down",
                down_quant,
                down_name,
                _source(experts, 256, 256, down_bytes),
                experts=experts,
            )
        )

    executor = Q4KExecutor(CpuExpertLayout(tuple(descriptors), top_k=2), mode="scalar")
    executor.prepare(max_tokens=1, max_routes=2)
    hidden = np.full((1, 256), 0.01, dtype=np.float32)
    ids = np.array([[1, 1]], dtype=np.int32)
    weights = np.array([[0.25, 0.75]], dtype=np.float32)
    gate = np.full(256, 2.56, dtype=np.float32)
    activated = gate / (1.0 + np.exp(-gate)) * gate
    expected = np.full(256, np.sum(activated), dtype=np.float32)

    for layer, kernel_census in (
        (0, ("reference_iq3_xxs", "reference_iq4_nl")),
        (2, ("reference_iq4_xs", "reference_q8_0")),
    ):
        result = executor.execute(layer, hidden, ids, weights)
        np.testing.assert_allclose(result.output[0], expected, rtol=2e-6, atol=2e-6)
        assert result.telemetry.backend == "reference"
        assert result.telemetry.kernel_census == kernel_census
        assert result.telemetry.fallback_reason == "reference_dequant_packed_workspace"


def test_q4k_executor_keeps_unknown_packed_quant_out_of_builtin_dispatch() -> None:
    source = _source(experts=1, output_dim=256, input_dim=256, block_bytes=Q8_0_BLOCK_BYTES)
    descriptors = tuple(
        _descriptor(0, projection, 999, "MYSTERY", source, experts=1)
        for projection in ("gate", "up", "down")
    )
    executor = Q4KExecutor(CpuExpertLayout(descriptors, top_k=1), mode="scalar")
    executor.prepare(max_tokens=1, max_routes=1)
    with pytest.raises(UnsupportedQuantType, match="no reference decoder"):
        executor.execute(
            0,
            np.ones((1, 256), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int32),
            np.ones((1, 1), dtype=np.float32),
        )


@pytest.mark.parametrize(
    ("quant_type", "quant_name", "block_bytes"),
    [
        (999, "Q8_0", Q8_0_BLOCK_BYTES),
        (8, "MYSTERY", Q8_0_BLOCK_BYTES),
        (7, "Q8_0", Q8_0_BLOCK_BYTES),
        (13, "Q5_1", Q5_1_BLOCK_BYTES),
        (7, "Q5_K", Q5_K_BLOCK_BYTES),
        (13, "Q8_0", Q8_0_BLOCK_BYTES),
        (999, "IQ3_XXS", IQ3_XXS_BLOCK_BYTES),
        (18, "MYSTERY", Q8_0_BLOCK_BYTES),
        (18, "IQ4_NL", IQ4_NL_BLOCK_BYTES),
    ],
)
def test_q4k_reference_dispatch_rejects_quant_type_name_mismatches(
    quant_type: int, quant_name: str, block_bytes: int
) -> None:
    source = _source(1, 256, 256, block_bytes)
    descriptor = _descriptor(
        0,
        "gate",
        quant_type,
        quant_name,
        source,
        experts=1,
    )
    with pytest.raises(UnsupportedQuantType, match="type/name contract is inconsistent"):
        Q4KExecutor(CpuExpertLayout((descriptor,), top_k=1), mode="scalar")
