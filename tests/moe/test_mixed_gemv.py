"""Native mixed GGML GEMV parity and fail-closed contract tests."""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from freetoken.moe.ggml_reference import (
    Q5_1_BLOCK_BYTES,
    Q5_1_BLOCK_ELEMENTS,
    Q5_K_BLOCK_BYTES,
    Q5_K_BLOCK_ELEMENTS,
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMENTS,
    decode_q5_1_block,
    decode_q5_k_block,
    decode_q8_0_block,
)
from freetoken.moe.mixed_gemv import (
    MixedGemvPrimitive,
    select_mixed_gemv_primitive,
)

_FORMATS = {
    "Q5_1": (Q5_1_BLOCK_ELEMENTS, Q5_1_BLOCK_BYTES),
    "Q8_0": (Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES),
    "Q5_K": (Q5_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES),
}
_QUANT_TYPES = {"Q5_1": 7, "Q8_0": 8, "Q5_K": 13}
_DECODERS = {
    "Q5_1": decode_q5_1_block,
    "Q8_0": decode_q8_0_block,
    "Q5_K": decode_q5_k_block,
}


def _half_bytes(value: float) -> np.ndarray:
    return np.frombuffer(np.asarray(np.float16(value), dtype="<f2").tobytes(), dtype=np.uint8)


def _pack_block(name: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    elements, block_bytes = _FORMATS[name]
    block = np.zeros(block_bytes, dtype=np.uint8)
    block[:2] = _half_bytes(0.03125 + seed / 128)
    if name == "Q5_1":
        block[2:4] = _half_bytes(-0.125 + seed / 256)
        block[4:8] = rng.integers(0, 256, 4, dtype=np.uint8)
        block[8:] = rng.integers(0, 256, 16, dtype=np.uint8)
    elif name == "Q8_0":
        block[2:] = rng.integers(0, 256, elements, dtype=np.uint8)
    else:
        block[2:4] = _half_bytes(0.015625 + seed / 256)
        block[4:] = rng.integers(0, 256, block_bytes - 4, dtype=np.uint8)
    return block


def _pack_rows(name: str, rows: int, input_dim: int, seed: int) -> np.ndarray:
    elements, _ = _FORMATS[name]
    blocks = input_dim // elements
    return np.stack(
        [
            np.concatenate(
                [_pack_block(name, seed + row * blocks + block) for block in range(blocks)]
            )
            for row in range(rows)
        ]
    ).astype(np.uint8, copy=False)


def _pack_signed_q8_0_block(scale: float, offset: int = 0) -> np.ndarray:
    """Build a Q8_0 block that exercises the complete signed-int8 range."""
    codes = np.array(
        [-128, -127, -1, 0, 1, 2, 126, 127] * 4,
        dtype=np.int8,
    )
    codes = np.roll(codes, offset)
    block = np.zeros(Q8_0_BLOCK_BYTES, dtype=np.uint8)
    block[:2] = _half_bytes(scale)
    block[2:] = codes.view(np.uint8)
    return block


def _reference_gemv(name: str, rows: np.ndarray, input_dim: int, vector: np.ndarray) -> np.ndarray:
    elements, block_bytes = _FORMATS[name]
    decoded = np.empty((rows.shape[0], input_dim), dtype=np.float32)
    decoder = _DECODERS[name]
    for row in range(rows.shape[0]):
        for block in range(input_dim // elements):
            begin = block * block_bytes
            decoder(
                rows[row, begin : begin + block_bytes],
                out=decoded[row, block * elements : (block + 1) * elements],
            )
    return np.asarray(np.sum(decoded * vector[None, :], axis=1, dtype=np.float32), dtype=np.float32)


def _independent_decode(name: str, block: np.ndarray) -> np.ndarray:
    """Decode without calling the production reference decoder."""
    elements, _ = _FORMATS[name]
    raw = np.asarray(block, dtype=np.uint8)
    d = np.float32(np.frombuffer(raw[:2].tobytes(), dtype="<f2")[0])
    result = np.empty(elements, dtype=np.float32)
    if name == "Q5_1":
        minimum = np.float32(np.frombuffer(raw[2:4].tobytes(), dtype="<f2")[0])
        qh = int.from_bytes(raw[4:8].tobytes(), "little")
        for lane in range(16):
            result[lane] = ((int(raw[8 + lane]) & 0x0F) | (((qh >> lane) & 1) << 4)) * d + minimum
            result[lane + 16] = (
                (int(raw[8 + lane]) >> 4) | (((qh >> (lane + 16)) & 1) << 4)
            ) * d + minimum
        return result
    if name == "Q8_0":
        for lane in range(32):
            result[lane] = np.int8(raw[2 + lane]) * d
        return result
    dmin = np.float32(np.frombuffer(raw[2:4].tobytes(), dtype="<f2")[0])
    scales = raw[4:16]
    for subblock in range(8):
        if subblock < 4:
            scale = int(scales[subblock]) & 63
            minimum = int(scales[subblock + 4]) & 63
        else:
            scale = (int(scales[subblock + 4]) & 15) | ((int(scales[subblock - 4]) >> 6) << 4)
            minimum = (int(scales[subblock + 4]) >> 4) | ((int(scales[subblock]) >> 6) << 4)
        for lane in range(32):
            packed = int(raw[48 + (subblock // 2) * 32 + lane])
            code = (packed & 15) if subblock % 2 == 0 else (packed >> 4)
            code |= ((int(raw[16 + lane]) >> subblock) & 1) << 4
            result[subblock * 32 + lane] = np.float32(
                np.float32(code) * d * np.float32(scale) - dmin * np.float32(minimum)
            )
    return result


def _independent_gemv(
    name: str, rows: np.ndarray, input_dim: int, vector: np.ndarray
) -> np.ndarray:
    elements, block_bytes = _FORMATS[name]
    result = np.zeros(rows.shape[0], dtype=np.float32)
    for row in range(rows.shape[0]):
        for block in range(input_dim // elements):
            begin = block * block_bytes
            values = _independent_decode(name, rows[row, begin : begin + block_bytes])
            result[row] = np.float32(
                result[row]
                + np.asarray(
                    np.sum(
                        values * vector[block * elements : (block + 1) * elements],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                )
            )
    return result


@pytest.fixture(scope="session")
def mixed_native_library(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    """Build the split baseline/AVX2 helper exactly as the package does."""
    if shutil.which("g++") is None:
        return None
    root = Path(__file__).resolve().parents[2]
    include_dir = root / "python/freetoken/moe"
    source = root / "python/freetoken/moe/mixed_gemv_native.cpp"
    scalar = root / "python/freetoken/moe/mixed_gemv_scalar.cpp"
    avx2 = root / "python/freetoken/moe/mixed_gemv_avx2.cpp"
    tmp_path = tmp_path_factory.mktemp("mixed-gemv-native")
    output = tmp_path / "mixed_gemv_native.so"
    baseline_flags = ["-mno-avx", "-mno-avx2", "-mno-fma"]
    for name, path, extra in (
        ("scalar", scalar, baseline_flags),
        ("avx2", avx2, ["-mavx2", "-mfma"]),
        ("dispatch", source, baseline_flags),
        ("avx2_native", avx2, ["-march=native"]),
        # Package builds may inherit CXXFLAGS from a host toolchain.  Verify
        # the source-level baseline fence also wins over -march=native.
        ("scalar_native", scalar, ["-march=native"]),
        ("dispatch_native", source, ["-march=native"]),
    ):
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O3",
                "-fPIC",
                *extra,
                "-I",
                str(include_dir),
                "-c",
                str(path),
                "-o",
                str(tmp_path / f"{name}.o"),
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
    return output


@pytest.mark.parametrize(
    ("name", "input_dim"),
    [("Q5_1", 640), ("Q8_0", 640), ("Q5_K", 2560)],
)
def test_scalar_mixed_gemv_matches_ggml_reference(name: str, input_dim: int) -> None:
    rows = _pack_rows(name, 3, input_dim, seed=17)
    vector = np.linspace(-1.25, 1.5, input_dim, dtype=np.float32)
    expected = _reference_gemv(name, rows, input_dim, vector)
    primitive = select_mixed_gemv_primitive("scalar")
    actual = np.empty(rows.shape[0], dtype=np.float32)
    primitive.gemv(rows, input_dim, vector, quant_name=name, out=actual)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)
    assert primitive.isa == "scalar"
    assert primitive.backend_for(name) == f"{name.lower()}_scalar"


@pytest.mark.parametrize(
    ("name", "input_dim"),
    [("Q5_1", 640), ("Q8_0", 640), ("Q5_K", 2560)],
)
def test_native_mixed_gemv_matches_ggml_reference(
    name: str,
    input_dim: int,
    mixed_native_library: Path | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if mixed_native_library is None:
        pytest.skip("g++ unavailable")
    monkeypatch.setenv("FREETOKEN_MIXED_GEMV_NATIVE_LIB", str(mixed_native_library))
    primitive = select_mixed_gemv_primitive("forced_avx2")
    if primitive.isa == "scalar":
        assert primitive.fallback_reason in {"avx2_unavailable", "native_avx2_unavailable"}
        return
    rows = _pack_rows(name, 4, input_dim, seed=53)
    vector = np.sin(np.arange(input_dim, dtype=np.float32) / 11.0)
    expected = _independent_gemv(name, rows, input_dim, vector)
    actual = np.empty(rows.shape[0], dtype=np.float32)
    primitive.gemv(rows, input_dim, vector, quant_name=name, out=actual)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-4)
    assert primitive.backend_for(name) == f"{name.lower()}_avx2"


@pytest.mark.parametrize(
    "name",
    ["Q5_1", "Q8_0", "Q5_K"],
)
def test_native_mixed_block_decode_and_dot_match_reference(
    name: str,
    mixed_native_library: Path | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if mixed_native_library is None:
        pytest.skip("g++ unavailable")
    monkeypatch.setenv("FREETOKEN_MIXED_GEMV_NATIVE_LIB", str(mixed_native_library))
    primitive = select_mixed_gemv_primitive("forced_avx2")
    if primitive.isa == "scalar":
        pytest.skip("AVX2/FMA unavailable")
    block = _pack_block(name, 127)
    elements, _ = _FORMATS[name]
    vector = np.cos(np.arange(elements, dtype=np.float32) / 7.0)
    expected = _independent_decode(name, block)
    decoded = np.empty(elements, dtype=np.float32)
    primitive.decode(block, quant_name=name, out=decoded)
    np.testing.assert_allclose(decoded, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        primitive.dot(block, vector, quant_name=_QUANT_TYPES[name]),
        np.dot(expected, vector).astype(np.float32),
        rtol=3e-5,
        atol=3e-5,
    )


@pytest.mark.parametrize("qh", [0x00000000, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555, 0x80000001])
def test_native_q5_1_expands_each_high_bit_to_its_matching_lane(
    qh: int,
    mixed_native_library: Path | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise all four eight-lane qh windows, including opposite bit patterns."""
    if mixed_native_library is None:
        pytest.skip("g++ unavailable")
    monkeypatch.setenv("FREETOKEN_MIXED_GEMV_NATIVE_LIB", str(mixed_native_library))
    primitive = select_mixed_gemv_primitive("forced_avx2")
    if primitive.isa == "scalar":
        pytest.skip("AVX2/FMA unavailable")
    block = _pack_block("Q5_1", 131)
    block[4:8] = np.frombuffer(np.uint32(qh).tobytes(), dtype=np.uint8)
    expected = _independent_decode("Q5_1", block)
    decoded = np.empty(Q5_1_BLOCK_ELEMENTS, dtype=np.float32)
    primitive.decode(block, quant_name="Q5_1", out=decoded)
    np.testing.assert_allclose(decoded, expected, rtol=2e-5, atol=2e-5)
    vector = np.linspace(-1.0, 1.0, Q5_1_BLOCK_ELEMENTS, dtype=np.float32)
    np.testing.assert_allclose(
        primitive.dot(block, vector, quant_name="Q5_1"),
        np.dot(expected, vector).astype(np.float32),
        rtol=3e-5,
        atol=3e-5,
    )


def test_native_q5_1_multi_block_gemv_preserves_block_reduction_order(
    mixed_native_library: Path | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one native float32 reduction per packed block on cancellation-heavy rows."""
    if mixed_native_library is None:
        pytest.skip("g++ unavailable")
    monkeypatch.setenv("FREETOKEN_MIXED_GEMV_NATIVE_LIB", str(mixed_native_library))
    primitive = select_mixed_gemv_primitive("forced_avx2")
    if primitive.isa == "scalar":
        pytest.skip("AVX2/FMA unavailable")
    rows = _pack_rows("Q5_1", 2, 640, seed=313)
    for row in range(rows.shape[0]):
        for block in range(rows.shape[1] // Q5_1_BLOCK_BYTES):
            qh = (0xAAAAAAAA, 0x55555555, 0xFFFFFFFF, 0x00000000)[block % 4]
            begin = block * Q5_1_BLOCK_BYTES
            rows[row, begin + 4 : begin + 8] = np.frombuffer(
                np.uint32(qh).tobytes(), dtype=np.uint8
            )
    vector = np.linspace(-500.0, 500.0, 640, dtype=np.float32)
    # The independent NumPy oracle intentionally has a different FMA/reduction
    # order from the AVX2 dot primitive.  Build the expected GEMV result from
    # that public per-block primitive so this test isolates the GEMV reduction
    # contract rather than conflating it with ordinary SIMD rounding.
    expected = np.zeros(rows.shape[0], dtype=np.float32)
    for row in range(rows.shape[0]):
        for block in range(640 // Q5_1_BLOCK_ELEMENTS):
            block_begin = block * Q5_1_BLOCK_BYTES
            input_begin = block * Q5_1_BLOCK_ELEMENTS
            expected[row] = np.float32(
                expected[row]
                + primitive.dot(
                    rows[row, block_begin : block_begin + Q5_1_BLOCK_BYTES],
                    vector[input_begin : input_begin + Q5_1_BLOCK_ELEMENTS],
                    quant_name="Q5_1",
                )
            )
    actual = np.empty(rows.shape[0], dtype=np.float32)
    primitive.gemv(rows, 640, vector, quant_name="Q5_1", out=actual)
    np.testing.assert_array_equal(actual, expected)

    canary = np.full(rows.nbytes + 64, 0xCD, dtype=np.uint8)
    bounded_rows = canary[32:-32].reshape(rows.shape)
    bounded_rows[:] = rows
    bounded = np.empty(rows.shape[0], dtype=np.float32)
    primitive.gemv(bounded_rows, 640, vector, quant_name="Q5_1", out=bounded)
    np.testing.assert_array_equal(canary[:32], 0xCD)
    np.testing.assert_array_equal(canary[-32:], 0xCD)
    np.testing.assert_array_equal(bounded, actual)


@pytest.mark.parametrize("scale", [0.03125, -0.75, 1.0])
def test_native_q8_0_vectorized_decode_preserves_signed_int8_codes(
    scale: float,
    mixed_native_library: Path | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch unsigned-byte widening, including -128 and both signed endpoints."""
    if mixed_native_library is None:
        pytest.skip("g++ unavailable")
    monkeypatch.setenv("FREETOKEN_MIXED_GEMV_NATIVE_LIB", str(mixed_native_library))
    primitive = select_mixed_gemv_primitive("forced_avx2")
    if primitive.isa == "scalar":
        pytest.skip("AVX2/FMA unavailable")
    block = _pack_signed_q8_0_block(scale)
    codes = block[2:].view(np.int8).astype(np.float32)
    expected = codes * np.float32(scale)
    decoded = np.empty(Q8_0_BLOCK_ELEMENTS, dtype=np.float32)
    primitive.decode(block, quant_name="Q8_0", out=decoded)
    np.testing.assert_array_equal(decoded, expected)

    vector = np.linspace(-3.0, 2.0, Q8_0_BLOCK_ELEMENTS, dtype=np.float32)
    expected_dot = np.asarray(np.sum(expected * vector, dtype=np.float32), dtype=np.float32)
    np.testing.assert_allclose(
        primitive.dot(block, vector, quant_name="Q8_0"),
        expected_dot,
        rtol=3e-6,
        atol=3e-6,
    )


def test_native_q8_0_multi_block_gemv_preserves_block_reduction_order_and_canaries(
    mixed_native_library: Path | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep Q8_0 GEMV block order while guarding every packed-row edge."""
    if mixed_native_library is None:
        pytest.skip("g++ unavailable")
    monkeypatch.setenv("FREETOKEN_MIXED_GEMV_NATIVE_LIB", str(mixed_native_library))
    primitive = select_mixed_gemv_primitive("forced_avx2")
    if primitive.isa == "scalar":
        pytest.skip("AVX2/FMA unavailable")

    rows = _pack_rows("Q8_0", 3, 640, seed=719)
    blocks = rows.shape[1] // Q8_0_BLOCK_BYTES
    for row in range(rows.shape[0]):
        for block in range(blocks):
            begin = block * Q8_0_BLOCK_BYTES
            rows[row, begin : begin + Q8_0_BLOCK_BYTES] = _pack_signed_q8_0_block(
                0.03125 * (1 + ((row + block) % 5)), offset=row + block
            )
    vector = np.linspace(-500.0, 500.0, 640, dtype=np.float32)

    # The established packed GEMV contract is one native float32 dot result
    # per block, accumulated in block order.  This checks that contract without
    # imposing a different FMA/reduction order on the dot primitive itself.
    expected = np.zeros(rows.shape[0], dtype=np.float32)
    for row in range(rows.shape[0]):
        for block in range(blocks):
            begin = block * Q8_0_BLOCK_BYTES
            input_begin = block * Q8_0_BLOCK_ELEMENTS
            expected[row] = np.float32(
                expected[row]
                + primitive.dot(
                    rows[row, begin : begin + Q8_0_BLOCK_BYTES],
                    vector[input_begin : input_begin + Q8_0_BLOCK_ELEMENTS],
                    quant_name="Q8_0",
                )
            )
    actual = np.empty(rows.shape[0], dtype=np.float32)
    primitive.gemv(rows, 640, vector, quant_name="Q8_0", out=actual)
    np.testing.assert_array_equal(actual, expected)

    canary = np.full(rows.nbytes + 64, 0xCD, dtype=np.uint8)
    bounded_rows = canary[32:-32].reshape(rows.shape)
    bounded_rows[:] = rows
    bounded = np.empty(rows.shape[0], dtype=np.float32)
    primitive.gemv(bounded_rows, 640, vector, quant_name="Q8_0", out=bounded)
    np.testing.assert_array_equal(canary[:32], 0xCD)
    np.testing.assert_array_equal(canary[-32:], 0xCD)
    np.testing.assert_array_equal(bounded, actual)


@pytest.mark.parametrize("name", ["Q5_1", "Q8_0", "Q5_K"])
def test_exported_scalar_symbols_match_independent_oracle(
    name: str,
    mixed_native_library: Path | None,
) -> None:
    if mixed_native_library is None:
        pytest.skip("g++ unavailable")
    library = ctypes.CDLL(str(mixed_native_library))
    byte_pointer = ctypes.POINTER(ctypes.c_uint8)
    float_pointer = ctypes.POINTER(ctypes.c_float)
    stem = name.lower()
    dot = getattr(library, f"freetoken_mixed_{stem}_dot_scalar")
    dot.argtypes = [byte_pointer, float_pointer]
    dot.restype = ctypes.c_float
    decode = getattr(library, f"freetoken_mixed_{stem}_decode_scalar")
    decode.argtypes = [byte_pointer, float_pointer]
    decode.restype = None
    elements, _ = _FORMATS[name]
    block = _pack_block(name, 223)
    vector = np.linspace(-1.0, 1.0, elements, dtype=np.float32)
    expected = _independent_decode(name, block)
    decoded = np.empty(elements, dtype=np.float32)
    decode(block.ctypes.data_as(byte_pointer), decoded.ctypes.data_as(float_pointer))
    np.testing.assert_allclose(decoded, expected, rtol=2e-5, atol=2e-5)
    actual_dot = dot(block.ctypes.data_as(byte_pointer), vector.ctypes.data_as(float_pointer))
    expected_dot = np.asarray(np.sum(expected * vector, dtype=np.float32), dtype=np.float32)
    np.testing.assert_allclose(actual_dot, expected_dot, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("name", ["Q5_1", "Q8_0", "Q5_K"])
def test_decode_rejects_input_output_overlap_for_scalar_and_native(
    name: str,
    mixed_native_library: Path | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primitives = [select_mixed_gemv_primitive("scalar")]
    if mixed_native_library is not None:
        monkeypatch.setenv("FREETOKEN_MIXED_GEMV_NATIVE_LIB", str(mixed_native_library))
        native = select_mixed_gemv_primitive("forced_avx2")
        if native.isa == "avx2":
            primitives.append(native)
    elements, block_bytes = _FORMATS[name]
    for primitive in primitives:
        storage = np.zeros(max(block_bytes, elements * 4), dtype=np.uint8)
        block = storage[:block_bytes]
        block[:] = _pack_block(name, 241)
        output = storage[: elements * 4].view(np.float32)
        before = storage.copy()
        with pytest.raises(ValueError, match="must not overlap"):
            primitive.decode(block, quant_name=name, out=output)
        np.testing.assert_array_equal(storage, before)


def test_native_gemv_rejects_null_and_bad_stride_without_writing(
    mixed_native_library: Path | None,
) -> None:
    if mixed_native_library is None:
        pytest.skip("g++ unavailable")
    library = ctypes.CDLL(str(mixed_native_library))
    byte_pointer = ctypes.POINTER(ctypes.c_uint8)
    float_pointer = ctypes.POINTER(ctypes.c_float)
    gemv = library.freetoken_mixed_q8_0_gemv
    gemv.argtypes = [
        byte_pointer,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        float_pointer,
        float_pointer,
    ]
    gemv.restype = ctypes.c_int
    rows = _pack_rows("Q8_0", 1, 640, seed=29)
    vector = np.ones(640, dtype=np.float32)
    output = np.full(1, 123.0, dtype=np.float32)
    null_bytes = byte_pointer()
    null_floats = float_pointer()
    assert (
        gemv(
            null_bytes,
            1,
            20,
            680,
            vector.ctypes.data_as(float_pointer),
            output.ctypes.data_as(float_pointer),
        )
        == -1
    )
    assert output[0] == 123.0
    assert (
        gemv(
            rows.ctypes.data_as(byte_pointer),
            1,
            20,
            679,
            vector.ctypes.data_as(float_pointer),
            output.ctypes.data_as(float_pointer),
        )
        == -1
    )
    assert output[0] == 123.0
    assert (
        gemv(
            rows.ctypes.data_as(byte_pointer),
            1,
            20,
            680,
            null_floats,
            output.ctypes.data_as(float_pointer),
        )
        == -1
    )
    assert output[0] == 123.0


def test_mixed_gemv_supports_format_specific_methods() -> None:
    primitive = MixedGemvPrimitive("scalar", "scalar", None, None)
    for name, input_dim in (("Q5_1", 640), ("Q8_0", 640), ("Q5_K", 2560)):
        rows = _pack_rows(name, 1, input_dim, seed=91)
        vector = np.ones(input_dim, dtype=np.float32)
        expected = _reference_gemv(name, rows, input_dim, vector)
        actual = np.empty(1, dtype=np.float32)
        method = getattr(primitive, f"{name.lower()}_gemv")
        method(rows, input_dim, vector, out=actual)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)


def test_mixed_gemv_rejects_invalid_layouts_and_outputs() -> None:
    primitive = select_mixed_gemv_primitive("scalar")
    rows = _pack_rows("Q8_0", 2, 640, seed=7)
    vector = np.ones(640, dtype=np.float32)
    output = np.empty(2, dtype=np.float32)
    with pytest.raises(ValueError, match="quant_name"):
        primitive.gemv(rows, 640, vector, quant_name="Q4_K", out=output)
    with pytest.raises(ValueError, match="positive multiple"):
        primitive.gemv(rows, 641, vector[:641], quant_name="Q8_0", out=output)
    with pytest.raises(ValueError, match="contiguous uint8"):
        primitive.gemv(rows[:, ::2], 640, vector, quant_name="Q8_0", out=output)
    readonly = output.copy()
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        primitive.gemv(rows, 640, vector, quant_name="Q8_0", out=readonly)
    with pytest.raises(ValueError, match="contiguous float32"):
        primitive.gemv(rows, 640, vector[::2], quant_name="Q8_0", out=output)
    with pytest.raises(ValueError, match="shape"):
        primitive.gemv(rows, 640, vector, quant_name="Q8_0", out=np.empty(1, dtype=np.float32))


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ unavailable")
def test_mixed_dispatch_baseline_contains_no_forbidden_isa(
    mixed_native_library: Path | None,
) -> None:
    assert mixed_native_library is not None
    dispatch = mixed_native_library.parent / "dispatch.o"
    scalar = mixed_native_library.parent / "scalar.o"
    dispatch_native = mixed_native_library.parent / "dispatch_native.o"
    scalar_native = mixed_native_library.parent / "scalar_native.o"
    avx2 = mixed_native_library.parent / "avx2.o"
    avx2_native = mixed_native_library.parent / "avx2_native.o"
    for baseline_object in (dispatch, scalar, dispatch_native, scalar_native):
        baseline_disassembly = subprocess.run(
            ["objdump", "-d", str(baseline_object)], check=True, capture_output=True, text=True
        ).stdout.lower()
        assert "zmm" not in baseline_disassembly
        assert "ymm" not in baseline_disassembly
        assert "0x62" not in baseline_disassembly
        assert re.search(r"\b(v[a-z][a-z0-9]*)\b", baseline_disassembly) is None
    avx_disassembly = subprocess.run(
        ["objdump", "-d", str(avx2)], check=True, capture_output=True, text=True
    ).stdout.lower()
    assert "zmm" not in avx_disassembly
    assert "avx512" not in avx_disassembly
    avx_native_disassembly = subprocess.run(
        ["objdump", "-d", str(avx2_native)], check=True, capture_output=True, text=True
    ).stdout.lower()
    assert "zmm" not in avx_native_disassembly
    assert "0x62" not in avx_native_disassembly
    assert "avx512" not in avx_native_disassembly
    loaded = ctypes.CDLL(str(mixed_native_library))
    loaded.freetoken_mixed_cpu_supports_avx2.restype = ctypes.c_int
    assert loaded.freetoken_mixed_cpu_supports_avx2() in {0, 1}
