"""Independent host-reference tests for the mixed Qwen3.8 expert formats."""

from __future__ import annotations

import numpy as np
import pytest
from freetoken.moe.ggml_reference import (
    IQ3_XXS_BLOCK_BYTES,
    IQ3_XXS_BLOCK_ELEMENTS,
    IQ4_NL_BLOCK_BYTES,
    IQ4_XS_BLOCK_BYTES,
    IQ4_XS_BLOCK_ELEMENTS,
    Q5_1_BLOCK_BYTES,
    Q5_1_BLOCK_ELEMENTS,
    Q5_K_BLOCK_BYTES,
    Q5_K_BLOCK_ELEMENTS,
    Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMENTS,
    decode_iq3_xxs_block,
    decode_iq3_xxs_rows,
    decode_iq4_nl_block,
    decode_iq4_nl_rows,
    decode_iq4_xs_block,
    decode_iq4_xs_rows,
    decode_q5_1_block,
    decode_q5_k_block,
    decode_q8_0_block,
    dequantize_iq3_xxs,
    dequantize_iq4_nl,
    dequantize_iq4_xs,
)


def _half_bytes(value: float) -> np.ndarray:
    return np.frombuffer(np.asarray(np.float16(value), dtype="<f2").tobytes(), dtype=np.uint8)


def _pack_q5_k(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    block = np.empty(Q5_K_BLOCK_BYTES, dtype=np.uint8)
    block[:2] = _half_bytes(0.03125 + seed / 64)
    block[2:4] = _half_bytes(0.015625 + seed / 128)
    block[4:16] = rng.integers(0, 256, 12, dtype=np.uint8)
    block[16:48] = rng.integers(0, 256, 32, dtype=np.uint8)
    block[48:] = rng.integers(0, 256, 128, dtype=np.uint8)
    return block


def _independent_q5_k(block: np.ndarray) -> np.ndarray:
    raw = np.asarray(block, dtype=np.uint8)
    d = float(np.frombuffer(raw[:2].tobytes(), dtype="<f2")[0])
    dmin = float(np.frombuffer(raw[2:4].tobytes(), dtype="<f2")[0])
    scales = raw[4:16]
    qh = raw[16:48]
    ql = raw[48:]
    result = np.empty(Q5_K_BLOCK_ELEMENTS, dtype=np.float32)
    for subblock in range(8):
        if subblock < 4:
            scale = int(scales[subblock]) & 63
            minimum = int(scales[subblock + 4]) & 63
        else:
            scale = (int(scales[subblock + 4]) & 15) | ((int(scales[subblock - 4]) >> 6) << 4)
            minimum = (int(scales[subblock + 4]) >> 4) | ((int(scales[subblock]) >> 6) << 4)
        for lane in range(32):
            packed = int(ql[(subblock // 2) * 32 + lane])
            code = (packed & 15) if subblock % 2 == 0 else (packed >> 4)
            code |= ((int(qh[lane]) >> subblock) & 1) << 4
            result[subblock * 32 + lane] = d * scale * code - dmin * minimum
    return result


def _pack_q5_1(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    block = np.empty(Q5_1_BLOCK_BYTES, dtype=np.uint8)
    block[:2] = _half_bytes(0.125 + seed / 32)
    block[2:4] = _half_bytes(-0.03125 + seed / 64)
    block[4:8] = rng.integers(0, 256, 4, dtype=np.uint8)
    block[8:] = rng.integers(0, 256, 16, dtype=np.uint8)
    return block


def _independent_q5_1(block: np.ndarray) -> np.ndarray:
    raw = np.asarray(block, dtype=np.uint8)
    d = float(np.frombuffer(raw[:2].tobytes(), dtype="<f2")[0])
    minimum = float(np.frombuffer(raw[2:4].tobytes(), dtype="<f2")[0])
    qh = int.from_bytes(raw[4:8].tobytes(), "little")
    qs = raw[8:]
    result = np.empty(Q5_1_BLOCK_ELEMENTS, dtype=np.float32)
    for lane in range(16):
        lo = (int(qs[lane]) & 15) | (((qh >> lane) & 1) << 4)
        hi = (int(qs[lane]) >> 4) | (((qh >> (lane + 16)) & 1) << 4)
        result[lane] = lo * d + minimum
        result[lane + 16] = hi * d + minimum
    return result


def _pack_q8_0(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    block = np.empty(Q8_0_BLOCK_BYTES, dtype=np.uint8)
    block[:2] = _half_bytes(0.0625 + seed / 64)
    block[2:] = rng.integers(0, 256, Q8_0_BLOCK_ELEMENTS, dtype=np.uint8)
    return block


def _independent_q8_0(block: np.ndarray) -> np.ndarray:
    raw = np.asarray(block, dtype=np.uint8)
    d = float(np.frombuffer(raw[:2].tobytes(), dtype="<f2")[0])
    return raw[2:].view(np.int8).astype(np.float32) * d


_IQ4_NL_VALUES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)


def _pack_iq4_nl(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    block = np.empty(IQ4_NL_BLOCK_BYTES, dtype=np.uint8)
    block[:2] = _half_bytes(-0.25 + seed / 32)
    block[2:] = rng.integers(0, 256, IQ4_NL_BLOCK_BYTES - 2, dtype=np.uint8)
    return block


def _independent_iq4_nl(block: np.ndarray) -> np.ndarray:
    raw = np.asarray(block, dtype=np.uint8)
    d = float(np.frombuffer(raw[:2].tobytes(), dtype="<f2")[0])
    codes = raw[2:]
    return np.concatenate((_IQ4_NL_VALUES[codes & 0x0F], _IQ4_NL_VALUES[codes >> 4])) * d


def _pack_iq4_xs(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    block = np.empty(IQ4_XS_BLOCK_BYTES, dtype=np.uint8)
    block[:2] = _half_bytes(-0.125 + seed / 32)
    block[2:4] = rng.integers(0, 256, 2, dtype=np.uint8)
    block[4:8] = rng.integers(0, 256, 4, dtype=np.uint8)
    block[8:] = rng.integers(0, 256, IQ4_XS_BLOCK_BYTES - 8, dtype=np.uint8)
    return block


def _independent_iq4_xs(block: np.ndarray) -> np.ndarray:
    raw = np.asarray(block, dtype=np.uint8)
    d = float(np.frombuffer(raw[:2].tobytes(), dtype="<f2")[0])
    scales_h = int.from_bytes(raw[2:4].tobytes(), "little")
    scales_l = raw[4:8]
    qs = raw[8:]
    result = np.empty(IQ4_XS_BLOCK_ELEMENTS, dtype=np.float32)
    for ib in range(8):
        scale = ((int(scales_l[ib // 2]) >> (4 * (ib % 2))) & 0x0F) | (
            ((scales_h >> (2 * ib)) & 0x03) << 4
        )
        factor = np.float32(d * (scale - 32))
        packed = qs[16 * ib : 16 * ib + 16]
        result[32 * ib : 32 * ib + 16] = _IQ4_NL_VALUES[packed & 0x0F] * factor
        result[32 * ib + 16 : 32 * ib + 32] = _IQ4_NL_VALUES[packed >> 4] * factor
    return result


def _ksigns(index: int) -> int:
    return index | ((index.bit_count() & 1) << 7)


def _independent_iq3_xxs_adversarial(block: np.ndarray) -> np.ndarray:
    """Check the extreme table entries without duplicating the full lookup table."""
    raw = np.asarray(block, dtype=np.uint8)
    d = float(np.frombuffer(raw[:2].tobytes(), dtype="<f2")[0])
    result = np.empty(IQ3_XXS_BLOCK_ELEMENTS, dtype=np.float32)
    # The pinned table maps index 0 to four 4s and 255 to (4, 28, 52, 62).
    for ib in range(8):
        q3 = raw[2 + 8 * ib : 2 + 8 * ib + 8]
        gas = int.from_bytes(raw[2 + 64 + 4 * ib : 2 + 68 + 4 * ib], "little")
        for il in range(4):
            index = int(q3[2 * il])
            values = (4,) * 8 if index == 0 else (4, 28, 52, 62) * 2 if index == 255 else None
            assert values is not None
            signs = _ksigns((gas >> (7 * il)) & 127)
            factor = d * (0.5 + (gas >> 28)) * 0.5
            begin = 32 * ib + 8 * il
            values = np.asarray(values, dtype=np.float32) * factor
            for lane in range(8):
                if signs & (1 << lane):
                    values[lane] *= -1
            result[begin : begin + 8] = values
    return result


@pytest.mark.parametrize("seed", (0, 1, 17, 255))
def test_q5_k_block_matches_independent_packed_layout(seed: int) -> None:
    block = _pack_q5_k(seed)
    np.testing.assert_array_equal(decode_q5_k_block(block), _independent_q5_k(block))


@pytest.mark.parametrize("seed", (0, 1, 17, 255))
def test_q5_1_block_matches_independent_packed_layout(seed: int) -> None:
    block = _pack_q5_1(seed)
    np.testing.assert_array_equal(decode_q5_1_block(block), _independent_q5_1(block))


@pytest.mark.parametrize("seed", (0, 1, 17, 255))
def test_q8_0_block_matches_independent_packed_layout(seed: int) -> None:
    block = _pack_q8_0(seed)
    np.testing.assert_array_equal(decode_q8_0_block(block), _independent_q8_0(block))


@pytest.mark.parametrize("seed", (0, 1, 17, 255))
def test_iq4_nl_block_matches_independent_packed_layout(seed: int) -> None:
    block = _pack_iq4_nl(seed)
    np.testing.assert_array_equal(decode_iq4_nl_block(block), _independent_iq4_nl(block))


@pytest.mark.parametrize("seed", (0, 1, 17, 255))
def test_iq4_xs_block_matches_independent_packed_layout(seed: int) -> None:
    block = _pack_iq4_xs(seed)
    np.testing.assert_array_equal(decode_iq4_xs_block(block), _independent_iq4_xs(block))


def test_iq3_xxs_block_matches_adversarial_lookup_and_sign_layout() -> None:
    block = np.zeros(IQ3_XXS_BLOCK_BYTES, dtype=np.uint8)
    block[:2] = _half_bytes(2.0)
    block[2 : 2 + 64 : 8] = 255
    block[2 + 64 : 2 + 64 + 32] = 0
    # Exercise both table extremes, high scale, and every sign-bit position.
    for ib in range(8):
        block[2 + 8 * ib : 2 + 8 * ib + 8] = 255 if ib % 2 else 0
        gas_offset = 2 + 64 + 4 * ib
        block[gas_offset : gas_offset + 4] = np.frombuffer(
            np.uint32(0xF0F0F0F0).tobytes(), dtype=np.uint8
        )
    np.testing.assert_array_equal(
        decode_iq3_xxs_block(block), _independent_iq3_xxs_adversarial(block)
    )


def test_iq_decoders_match_gguf_py_oracle_for_random_blocks() -> None:
    pytest.importorskip("gguf")
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize

    rng = np.random.default_rng(38)
    iq3 = rng.integers(0, 256, IQ3_XXS_BLOCK_BYTES, dtype=np.uint8)
    iq3[:2] = _half_bytes(0.625)
    cases = (
        (iq3, decode_iq3_xxs_block, GGMLQuantizationType.IQ3_XXS),
        (_pack_iq4_nl(17), decode_iq4_nl_block, GGMLQuantizationType.IQ4_NL),
        (_pack_iq4_xs(17), decode_iq4_xs_block, GGMLQuantizationType.IQ4_XS),
    )
    for block, decoder, quant_type in cases:
        expected = dequantize(block.reshape(1, -1), quant_type).reshape(-1)
        np.testing.assert_allclose(decoder(block), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("decoder", "block_bytes"),
    [
        (decode_iq3_xxs_block, IQ3_XXS_BLOCK_BYTES),
        (decode_iq4_nl_block, IQ4_NL_BLOCK_BYTES),
        (decode_iq4_xs_block, IQ4_XS_BLOCK_BYTES),
        (decode_q5_k_block, Q5_K_BLOCK_BYTES),
        (decode_q5_1_block, Q5_1_BLOCK_BYTES),
        (decode_q8_0_block, Q8_0_BLOCK_BYTES),
    ],
)
def test_reference_decoders_fail_closed_on_wrong_block_views(decoder, block_bytes: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        decoder(np.zeros(block_bytes - 1, dtype=np.uint8))
    with pytest.raises((TypeError, ValueError)):
        decoder(np.zeros(block_bytes, dtype=np.float32))


@pytest.mark.parametrize(
    ("block_decoder", "dequantize", "row_decoder", "block_bytes", "block_elements"),
    [
        (decode_iq3_xxs_block, dequantize_iq3_xxs, decode_iq3_xxs_rows, 98, 256),
        (decode_iq4_nl_block, dequantize_iq4_nl, decode_iq4_nl_rows, 18, 32),
        (decode_iq4_xs_block, dequantize_iq4_xs, decode_iq4_xs_rows, 136, 256),
    ],
)
def test_iq_reference_outputs_reject_aliasing_packed_storage(
    block_decoder,
    dequantize,
    row_decoder,
    block_bytes: int,
    block_elements: int,
) -> None:
    storage = np.zeros(max(block_bytes, block_elements * 4), dtype=np.uint8)
    packed = storage[:block_bytes]
    out = storage[: block_elements * 4].view(np.float32)
    assert np.shares_memory(packed, out)
    with pytest.raises(ValueError, match="shares memory"):
        block_decoder(packed, out=out)
    with pytest.raises(ValueError, match="shares memory"):
        dequantize(packed, out=out)

    class _Descriptor:
        input_dim = block_elements
        output_dim = 1
        row_stride_bytes = block_bytes

    with pytest.raises(ValueError, match="shares memory"):
        row_decoder(packed.reshape(1, block_bytes), _Descriptor(), out=out.reshape(1, -1))
