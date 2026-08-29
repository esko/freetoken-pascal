"""Hosted Q4_K oracle, dispatch and Issue #15 ABI adapter tests."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from freetoken.moe.cpu_abi import (
    Cancelled,
    CpuExpertDescriptor,
    CpuExpertLayout,
    UnsupportedQuantType,
    UnsupportedShape,
)
from freetoken.moe.ggml_reference import Q5_1_BLOCK_BYTES
from freetoken.moe.q4_k import (
    Q4K_BLOCK_BYTES,
    Q4K_BLOCK_ELEMENTS,
    Q4KExecutor,
    Q4KPrimitive,
    decode_q4_k_block,
    q4_k_dot,
    select_q4_k_primitive,
)


def _pack_q4_k_block(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    block = np.zeros(Q4K_BLOCK_BYTES, dtype=np.uint8)
    block[:2] = np.frombuffer(
        np.asarray(np.float16(0.125 + seed / 32), dtype="<f2").tobytes(), dtype=np.uint8
    )
    block[2:4] = np.frombuffer(
        np.asarray(np.float16(0.03125 + seed / 64), dtype="<f2").tobytes(), dtype=np.uint8
    )
    block[4:16] = rng.integers(0, 64, 12, dtype=np.uint8)
    block[16:] = rng.integers(0, 256, Q4K_BLOCK_BYTES - 16, dtype=np.uint8)
    return block


def _pack_q4_k_rows(row_count: int, input_dim: int, *, seed: int) -> np.ndarray:
    blocks_per_row = input_dim // Q4K_BLOCK_ELEMENTS
    return np.stack(
        [
            np.concatenate(
                [
                    _pack_q4_k_block(seed + row * blocks_per_row + block)
                    for block in range(blocks_per_row)
                ]
            )
            for row in range(row_count)
        ]
    )


def _independent_decode(block: np.ndarray) -> np.ndarray:
    raw = np.asarray(block, dtype=np.uint8)
    d = float(np.frombuffer(raw[:2].tobytes(), dtype="<f2")[0])
    dmin = float(np.frombuffer(raw[2:4].tobytes(), dtype="<f2")[0])
    scales = raw[4:16]
    qs = raw[16:]
    result = np.empty(Q4K_BLOCK_ELEMENTS, dtype=np.float32)
    for subblock in range(8):
        if subblock < 4:
            scale = int(scales[subblock]) & 63
            minimum = int(scales[subblock + 4]) & 63
        else:
            scale = (int(scales[subblock + 4]) & 15) | ((int(scales[subblock - 4]) >> 6) << 4)
            minimum = (int(scales[subblock + 4]) >> 4) | ((int(scales[subblock]) >> 6) << 4)
        group = subblock // 2
        high = subblock & 1
        for lane in range(32):
            packed = int(qs[group * 32 + lane])
            code = packed >> 4 if high else packed & 15
            result[subblock * 32 + lane] = d * scale * code - dmin * minimum
    return result


def test_scalar_q4_k_oracle_matches_independent_block_layout() -> None:
    for seed in (0, 1, 17, 255):
        packed = _pack_q4_k_block(seed)
        np.testing.assert_array_equal(decode_q4_k_block(packed), _independent_decode(packed))


def test_q4_k_dot_modes_match_oracle_without_torch() -> None:
    block = _pack_q4_k_block(38)
    x = np.linspace(-1.5, 1.5, Q4K_BLOCK_ELEMENTS, dtype=np.float32)
    expected = float(np.dot(_independent_decode(block), x).astype(np.float32))
    for mode in ("scalar", "avx2", "auto"):
        actual = q4_k_dot(block, x, mode=mode)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert select_q4_k_primitive("scalar").isa == "scalar"
    assert select_q4_k_primitive("forced_scalar").isa == "scalar"
    forced_avx2 = select_q4_k_primitive("forced_avx2")
    assert forced_avx2.isa in {"scalar", "avx2"}
    if forced_avx2.isa == "scalar":
        assert forced_avx2.fallback_reason in {"avx2_unavailable", "native_avx2_unavailable"}


class _PackedSource:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.range_offset = 0
        self.range_size = int(values.nbytes)
        self.source_address = int(values.__array_interface__["data"][0])

    def expert_packed(self, expert: int) -> np.ndarray:
        return self.values[expert]


def _q4_layout(*, experts: int = 3, address_offset: int = 0):
    hidden = intermediate = Q4K_BLOCK_ELEMENTS
    output_dims = {"gate": intermediate, "up": intermediate, "down": hidden}
    input_dims = {"gate": hidden, "up": hidden, "down": intermediate}
    sources = {}
    descriptors = []
    for pool_id, projection in enumerate(("gate", "up", "down")):
        output = output_dims[projection]
        input_size = input_dims[projection]
        row_bytes = input_size // Q4K_BLOCK_ELEMENTS * Q4K_BLOCK_BYTES
        values = np.stack(
            [
                np.stack([_pack_q4_k_block(11 + pool_id + e + row) for row in range(output)])
                for e in range(experts)
            ]
        )
        values = np.ascontiguousarray(values)
        if address_offset:
            storage = np.zeros(values.nbytes + address_offset, dtype=np.uint8)
            storage[address_offset:] = values.reshape(-1)
            values = storage[address_offset:].reshape(values.shape)
        source = _PackedSource(values)
        sources[projection] = source
        descriptors.append(
            CpuExpertDescriptor(
                layer_id=0,
                projection=projection,
                quant_type=12,
                quant_name="Q4_K",
                num_experts=experts,
                output_dim=output,
                input_dim=input_size,
                rows_per_expert=output,
                row_stride_bytes=row_bytes,
                expert_stride_bytes=output * row_bytes,
                tensor_bytes=experts * output * row_bytes,
                source=sources[projection],
            )
        )
    return CpuExpertLayout(tuple(descriptors), top_k=4), sources


def test_q4_k_executor_handles_first_middle_last_and_duplicate_routes() -> None:
    layout, sources = _q4_layout()
    executor = Q4KExecutor(layout, mode="scalar", activation="silu")
    executor.prepare(max_tokens=2, max_routes=4)
    hidden = np.linspace(-0.5, 0.5, 2 * Q4K_BLOCK_ELEMENTS, dtype=np.float32).reshape(2, -1)
    expert_ids = np.array([[0, 1, 1, 2], [2, 0, -1, 2]], dtype=np.int32)
    weights = np.array([[0.2, -0.3, 0.4, 0.5], [-0.25, 0.75, 0.0, 0.1]], dtype=np.float32)
    result = executor.execute(0, hidden, expert_ids, weights)

    expected = np.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for route, expert in enumerate(expert_ids[token]):
            if expert < 0:
                continue
            gate = (
                np.stack(
                    [
                        decode_q4_k_block(sources["gate"].expert_packed(expert)[row])
                        for row in range(Q4K_BLOCK_ELEMENTS)
                    ]
                )
                @ hidden[token]
            )
            up = (
                np.stack(
                    [
                        decode_q4_k_block(sources["up"].expert_packed(expert)[row])
                        for row in range(Q4K_BLOCK_ELEMENTS)
                    ]
                )
                @ hidden[token]
            )
            activated = gate / (1.0 + np.exp(-gate)) * up
            down = np.stack(
                [
                    decode_q4_k_block(sources["down"].expert_packed(expert)[row])
                    for row in range(Q4K_BLOCK_ELEMENTS)
                ]
            )
            expected[token] += (down @ activated) * weights[token, route]
    np.testing.assert_allclose(result.output, expected, rtol=3e-5, atol=3e-5)
    assert result.telemetry.backend == "q4_k_scalar"
    assert result.telemetry.kernel_census == ("q4_k_scalar",)
    assert result.telemetry.unique_experts == 3
    assert result.telemetry.routes_executed == 7


def test_q4_k_executor_cancellation_rolls_back_output_and_reports_mode() -> None:
    layout, _ = _q4_layout(experts=2)
    executor = Q4KExecutor(layout, mode="scalar")
    executor.prepare(max_tokens=1, max_routes=2)
    output = np.full((1, Q4K_BLOCK_ELEMENTS), 9, dtype=np.float32)
    checks = 0

    def cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(Cancelled) as raised:
        executor.execute(
            0,
            np.ones((1, Q4K_BLOCK_ELEMENTS), dtype=np.float32),
            np.array([[0, 1]], dtype=np.int32),
            np.ones((1, 2), dtype=np.float32),
            output=output,
            cancellation=cancel,
        )
    assert np.all(output == 0)
    assert raised.value.telemetry is not None
    assert raised.value.telemetry.backend == "q4_k_scalar"
    assert raised.value.telemetry.cancelled


def test_q4_k_direct_gemv_reuses_input_scratch_for_strided_hidden(monkeypatch) -> None:
    layout, _ = _q4_layout(experts=1)
    executor = Q4KExecutor(layout, mode="scalar")
    executor.prepare(max_tokens=1, max_routes=1)
    backing = np.ones((1, Q4K_BLOCK_ELEMENTS * 2), dtype=np.float32)
    hidden = backing[:, ::2]
    output = np.empty((1, Q4K_BLOCK_ELEMENTS), dtype=np.float32)
    monkeypatch.setattr(
        np,
        "empty",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("execute allocated")),
    )
    monkeypatch.setattr(
        np,
        "ascontiguousarray",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("execute copied with allocation")
        ),
    )
    result = executor.execute(
        0,
        hidden,
        np.zeros((1, 1), dtype=np.int32),
        np.ones((1, 1), dtype=np.float32),
        output=output,
    )
    assert result.telemetry.backend == "q4_k_scalar"
    assert result.telemetry.fallback_reason is None


def test_q4_k_gemv_rejects_partial_block_dimensions() -> None:
    primitive = select_q4_k_primitive("scalar")
    with pytest.raises(ValueError, match="positive multiple"):
        primitive.gemv(
            np.zeros((1, Q4K_BLOCK_BYTES), dtype=np.uint8),
            Q4K_BLOCK_ELEMENTS + 1,
            np.zeros(Q4K_BLOCK_ELEMENTS + 1, dtype=np.float32),
            out=np.empty(1, dtype=np.float32),
            scratch=np.empty(Q4K_BLOCK_ELEMENTS, dtype=np.float32),
        )


def test_q4_k_native_outputs_must_be_writable() -> None:
    class _FakeNative:
        def __init__(self) -> None:
            self.decode_calls = 0
            self.gemv_calls = 0

        def decode(self, block: np.ndarray, out: np.ndarray) -> None:
            del block, out
            self.decode_calls += 1

        def gemv(
            self,
            rows: np.ndarray,
            row_count: int,
            blocks_per_row: int,
            row_stride_bytes: int,
            vector: np.ndarray,
            out: np.ndarray,
        ) -> None:
            del rows, row_count, blocks_per_row, row_stride_bytes, vector, out
            self.gemv_calls += 1

    native = _FakeNative()
    primitive = Q4KPrimitive("avx2", "avx2", None, native)
    block = _pack_q4_k_block(71)
    decoded = np.empty(Q4K_BLOCK_ELEMENTS, dtype=np.float32)
    decoded.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        primitive.decode(block, out=decoded)
    assert native.decode_calls == 0

    rows = _pack_q4_k_rows(1, Q4K_BLOCK_ELEMENTS, seed=73)
    vector = np.ones(Q4K_BLOCK_ELEMENTS, dtype=np.float32)
    output = np.empty(1, dtype=np.float32)
    output.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        primitive.gemv(rows, Q4K_BLOCK_ELEMENTS, vector, out=output)
    assert native.gemv_calls == 0


@pytest.mark.parametrize("input_dim", (512, 768))
def test_q4_k_scalar_gemv_matches_block_oracle_for_multiple_blocks(input_dim: int) -> None:
    rows = _pack_q4_k_rows(2, input_dim, seed=81)
    vector = np.linspace(-1.0, 1.0, input_dim, dtype=np.float32)
    output = np.empty(rows.shape[0], dtype=np.float32)
    primitive = select_q4_k_primitive("scalar")
    primitive.gemv(rows, input_dim, vector, out=output)
    blocks_per_row = input_dim // Q4K_BLOCK_ELEMENTS
    expected = np.array(
        [
            sum(
                q4_k_dot(
                    rows[row, block * Q4K_BLOCK_BYTES : (block + 1) * Q4K_BLOCK_BYTES],
                    vector[block * Q4K_BLOCK_ELEMENTS : (block + 1) * Q4K_BLOCK_ELEMENTS],
                    mode="scalar",
                )
                for block in range(blocks_per_row)
            )
            for row in range(rows.shape[0])
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)


def test_q4_k_packed_partial_rows_fail_closed() -> None:
    layout, sources = _q4_layout(experts=1)
    gate = replace(layout.descriptor(0, "gate"), input_dim=Q4K_BLOCK_ELEMENTS + 1)
    up = replace(layout.descriptor(0, "up"), input_dim=Q4K_BLOCK_ELEMENTS + 1)
    down = replace(
        layout.descriptor(0, "down"),
        output_dim=Q4K_BLOCK_ELEMENTS + 1,
        rows_per_expert=Q4K_BLOCK_ELEMENTS + 1,
        expert_stride_bytes=(Q4K_BLOCK_ELEMENTS + 1) * Q4K_BLOCK_BYTES,
        tensor_bytes=(Q4K_BLOCK_ELEMENTS + 1) * Q4K_BLOCK_BYTES,
        source=None,
    )
    malformed = CpuExpertLayout((gate, up, down), top_k=1)
    executor = Q4KExecutor(malformed, mode="scalar")
    executor.prepare(max_tokens=1, max_routes=1)
    with pytest.raises(UnsupportedShape, match="not a multiple"):
        executor.execute(
            0,
            np.ones((1, Q4K_BLOCK_ELEMENTS + 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int32),
            np.ones((1, 1), dtype=np.float32),
        )
    assert sources["gate"].values.shape[1] == Q4K_BLOCK_ELEMENTS


def test_q4_k_inconsistent_type_name_is_not_reinterpreted() -> None:
    layout, _ = _q4_layout(experts=1)
    mismatched_gate = replace(layout.descriptor(0, "gate"), quant_name="Q5_K")
    mismatched = CpuExpertLayout(
        (mismatched_gate, layout.descriptor(0, "up"), layout.descriptor(0, "down")),
        top_k=1,
    )
    with pytest.raises(UnsupportedQuantType, match="inconsistent"):
        Q4KExecutor(mismatched, mode="scalar")


def test_q4_k_unsupported_alignment_is_a_loud_scalar_fallback() -> None:
    layout, _ = _q4_layout(address_offset=1)
    executor = Q4KExecutor(layout, mode="avx2")
    executor.prepare(max_tokens=1, max_routes=1)
    result = executor.execute(
        0,
        np.ones((1, Q4K_BLOCK_ELEMENTS), dtype=np.float32),
        np.zeros((1, 1), dtype=np.int32),
        np.ones((1, 1), dtype=np.float32),
    )
    assert result.telemetry.backend == "q4_k_scalar"
    assert result.telemetry.fallback_reason == "unsupported_alignment"


def test_q4_k_unsupported_format_falls_back_through_reference_abi() -> None:
    rng = np.random.default_rng(4)
    experts, hidden, intermediate = 2, 4, 6
    dense = {
        projection: rng.normal(size=(experts, output, input_size)).astype(np.float32)
        for projection, output, input_size in (
            ("gate", intermediate, hidden),
            ("up", intermediate, hidden),
            ("down", hidden, intermediate),
        )
    }
    descriptors = tuple(
        CpuExpertDescriptor(
            layer_id=0,
            projection=projection,
            quant_type=13,
            quant_name="Q5_K",
            num_experts=experts,
            output_dim=values.shape[1],
            input_dim=values.shape[2],
            rows_per_expert=values.shape[1],
            row_stride_bytes=values.shape[2] * 4,
            expert_stride_bytes=values.shape[1] * values.shape[2] * 4,
            tensor_bytes=values.nbytes,
            source=values,
        )
        for projection, values in dense.items()
    )
    executor = Q4KExecutor(CpuExpertLayout(descriptors, top_k=1), mode="auto")
    executor.prepare(max_tokens=1, max_routes=1)
    result = executor.execute(
        0,
        np.ones((1, hidden), dtype=np.float32),
        np.zeros((1, 1), dtype=np.int32),
        np.ones((1, 1), dtype=np.float32),
    )
    assert result.telemetry.backend == "reference"
    assert result.telemetry.kernel_census == ("reference",)
    assert result.telemetry.fallback_reason == "unsupported_quant_type"


def test_q4_k_mixed_promoted_projection_uses_supplied_reference_decoder() -> None:
    layout, _ = _q4_layout(experts=1)
    promoted_packed = np.ones(
        (1, Q4K_BLOCK_ELEMENTS, Q4K_BLOCK_ELEMENTS // 32 * Q5_1_BLOCK_BYTES),
        dtype=np.uint8,
    )
    promoted_source = _PackedSource(promoted_packed)
    up = layout.descriptor(0, "up")
    promoted_descriptor = replace(
        up,
        quant_type=7,
        quant_name="Q5_1",
        row_stride_bytes=promoted_packed.shape[-1],
        expert_stride_bytes=promoted_packed.shape[1] * promoted_packed.shape[2],
        tensor_bytes=promoted_packed.nbytes,
        source=promoted_source,
        source_address=promoted_source.source_address,
    )
    mixed = CpuExpertLayout(
        (layout.descriptor(0, "gate"), promoted_descriptor, layout.descriptor(0, "down")),
        top_k=1,
    )

    def decode_q5(packed, descriptor, *, out):
        assert packed.shape == (descriptor.output_dim, descriptor.row_stride_bytes)
        out.fill(1.0)
        return out

    executor = Q4KExecutor(mixed, mode="scalar", reference_decoders={7: decode_q5})
    executor.prepare(max_tokens=1, max_routes=1)
    result = executor.execute(
        0,
        np.ones((1, Q4K_BLOCK_ELEMENTS), dtype=np.float32),
        np.zeros((1, 1), dtype=np.int32),
        np.ones((1, 1), dtype=np.float32),
    )
    assert np.isfinite(result.output).all()
    assert result.telemetry.backend == "mixed"
    assert result.telemetry.kernel_census == ("q4_k_scalar", "reference_q5_1")
    assert result.telemetry.fallback_reason == "mixed_reference_formats"


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ unavailable")
def test_q4_k_native_source_compiles_without_forbidden_isa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "python/freetoken/moe/q4_k_native.cpp"
    scalar = root / "python/freetoken/moe/q4_k_scalar.cpp"
    avx2 = root / "python/freetoken/moe/q4_k_avx2.cpp"
    output = tmp_path / "q4_k_native.so"
    include_dir = root / "python/freetoken/moe"
    baseline_flags = ["-mno-avx", "-mno-avx2", "-mno-fma"]
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-fPIC",
            *baseline_flags,
            "-I",
            str(include_dir),
            "-c",
            str(scalar),
            "-o",
            str(tmp_path / "scalar.o"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-mavx2",
            "-mfma",
            "-I",
            str(include_dir),
            "-c",
            str(avx2),
            "-o",
            str(tmp_path / "avx2.o"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-fPIC",
            *baseline_flags,
            "-I",
            str(include_dir),
            "-c",
            str(source),
            "-o",
            str(tmp_path / "dispatch.o"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "g++",
            "-shared",
            str(tmp_path / "dispatch.o"),
            str(tmp_path / "scalar.o"),
            str(tmp_path / "avx2.o"),
            "-o",
            str(output),
        ],
        check=True,
    )
    baseline_disassembly = subprocess.run(
        ["objdump", "-d", str(tmp_path / "dispatch.o")], check=True, capture_output=True, text=True
    ).stdout.lower()
    assert "zmm" not in baseline_disassembly
    assert "0x62" not in baseline_disassembly
    disassembly = subprocess.run(
        ["objdump", "-d", str(tmp_path / "avx2.o")], check=True, capture_output=True, text=True
    ).stdout.lower()
    assert "zmm" not in disassembly
    assert "avx512" not in disassembly
    dynamic = subprocess.run(
        ["readelf", "-d", str(output)], check=True, capture_output=True, text=True
    ).stdout.lower()
    assert "python" not in dynamic
    assert "torch" not in dynamic
    assert "cudart" not in dynamic

    monkeypatch.setenv("FREETOKEN_Q4K_NATIVE_LIB", str(output))
    primitive = select_q4_k_primitive("auto")
    support = ctypes.CDLL(str(output)).freetoken_q4k_cpu_supports_avx2
    support.argtypes = []
    support.restype = ctypes.c_int
    block = _pack_q4_k_block(91)
    vector = np.linspace(-1.0, 1.0, Q4K_BLOCK_ELEMENTS, dtype=np.float32)
    if support():
        assert primitive.isa == "avx2"
        decoded = np.empty(Q4K_BLOCK_ELEMENTS, dtype=np.float32)
        primitive.decode(block, out=decoded)
        np.testing.assert_allclose(
            decoded,
            decode_q4_k_block(block),
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            primitive.dot(block, vector),
            q4_k_dot(block, vector, mode="scalar"),
            rtol=2e-5,
            atol=2e-5,
        )
        rows = np.stack([block, _pack_q4_k_block(92)])
        output_rows = np.empty(2, dtype=np.float32)
        primitive.gemv(rows, Q4K_BLOCK_ELEMENTS, vector, out=output_rows)
        expected_rows = np.array(
            [q4_k_dot(rows[row], vector, mode="scalar") for row in range(rows.shape[0])],
            dtype=np.float32,
        )
        np.testing.assert_allclose(output_rows, expected_rows, rtol=2e-5, atol=2e-5)

        for input_dim in (512, 768):
            rows = _pack_q4_k_rows(2, input_dim, seed=93 + input_dim)
            vector = np.linspace(-1.0, 1.0, input_dim, dtype=np.float32)
            output_rows = np.empty(rows.shape[0], dtype=np.float32)
            primitive.gemv(rows, input_dim, vector, out=output_rows)
            blocks_per_row = input_dim // Q4K_BLOCK_ELEMENTS
            expected_rows = np.array(
                [
                    sum(
                        q4_k_dot(
                            rows[row, block * Q4K_BLOCK_BYTES : (block + 1) * Q4K_BLOCK_BYTES],
                            vector[block * Q4K_BLOCK_ELEMENTS : (block + 1) * Q4K_BLOCK_ELEMENTS],
                            mode="scalar",
                        )
                        for block in range(blocks_per_row)
                    )
                    for row in range(rows.shape[0])
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(output_rows, expected_rows, rtol=2e-5, atol=2e-5)
    else:
        assert primitive.isa == "scalar"
        assert primitive.fallback_reason == "avx2_unavailable"
