"""Independent host-reference tests for the mixed Qwen3.8 expert formats."""

from __future__ import annotations

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


@pytest.mark.parametrize(
    ("decoder", "block_bytes"),
    [
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
