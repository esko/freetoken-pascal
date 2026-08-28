"""Q4K adapter integration over the Qwen3.8 mixed projection census."""

from __future__ import annotations

import numpy as np
from freetoken.moe.cpu_abi import CpuExpertDescriptor, CpuExpertLayout
from freetoken.moe.ggml_reference import (
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
    else:
        raise AssertionError(block_bytes)
    return block


def _source(experts: int, output_dim: int, input_dim: int, block_bytes: int) -> _PackedSource:
    blocks = input_dim // (256 if block_bytes in (Q4K_BLOCK_BYTES, Q5_K_BLOCK_BYTES) else 32)
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
    block = 256 if quant_name in {"Q4_K", "Q5_K"} else 32
    block_bytes = {
        "Q4_K": Q4K_BLOCK_BYTES,
        "Q5_K": Q5_K_BLOCK_BYTES,
        "Q5_1": Q5_1_BLOCK_BYTES,
        "Q8_0": Q8_0_BLOCK_BYTES,
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


def test_q4k_executor_registers_builtin_decoders_for_actual_mixed_layers() -> None:
    experts = 2
    descriptors = []
    for layer, gate_up_quant, gate_up_name, down_quant, down_name in (
        (0, 12, "Q4_K", 7, "Q5_1"),
        (1, 13, "Q5_K", 8, "Q8_0"),
    ):
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

    executor = Q4KExecutor(CpuExpertLayout(tuple(descriptors), top_k=2), mode="scalar")
    executor.prepare(max_tokens=1, max_routes=2)
    hidden = np.full((1, 256), 0.01, dtype=np.float32)
    ids = np.array([[1, 1]], dtype=np.int32)
    weights = np.array([[0.25, 0.75]], dtype=np.float32)

    for layer in (0, 1):
        result = executor.execute(layer, hidden, ids, weights)
        gate = np.full(256, 2.56, dtype=np.float32)
        up = np.full(256, 2.56, dtype=np.float32)
        activated = gate / (1.0 + np.exp(-gate)) * up
        expected = np.full(256, np.sum(activated), dtype=np.float32)
        np.testing.assert_allclose(result.output[0], expected, rtol=2e-6, atol=2e-6)
        assert result.telemetry.backend == "reference"
        assert result.telemetry.fallback_reason == "unsupported_quant_type"
