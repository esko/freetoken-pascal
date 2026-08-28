"""Torch-free reference decoders for the Qwen3.8 mixed GGML expert banks.

The packed layouts and arithmetic in this module follow the pinned llama.cpp
``eaf93765572e794b8e3754fe45adbe12d381e997`` reference.  This is a bounded,
scalar correctness path for Q5_K, Q5_1 and Q8_0; it is not an AVX2 kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from freetoken.moe.cpu_abi import InvalidRequest, UnsupportedShape

Q5_K_BLOCK_ELEMENTS = 256
Q5_K_BLOCK_BYTES = 176
Q5_1_BLOCK_ELEMENTS = 32
Q5_1_BLOCK_BYTES = 24
Q8_0_BLOCK_ELEMENTS = 32
Q8_0_BLOCK_BYTES = 34


def _uint8_block(block: Any, block_bytes: int, name: str) -> np.ndarray:
    raw = np.asarray(block)
    if raw.dtype != np.dtype(np.uint8):
        raise TypeError(f"{name} block must have dtype uint8, got {raw.dtype}")
    if raw.ndim != 1 or raw.size != block_bytes:
        raise ValueError(
            f"{name} block must be a contiguous 1-D {block_bytes}-byte view, got {raw.shape}"
        )
    if not raw.flags.c_contiguous:
        raise ValueError(f"{name} block must be C-contiguous")
    return raw


def _decode_output(out: np.ndarray | None, elements: int, name: str) -> np.ndarray:
    if out is None:
        return np.empty(elements, dtype=np.float32)
    if not isinstance(out, np.ndarray) or out.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} output must be a float32 NumPy ndarray")
    if out.ndim != 1 or out.size != elements or not out.flags.c_contiguous:
        raise ValueError(f"{name} output must be a contiguous {elements}-element float32 vector")
    if not out.flags.writeable:
        raise ValueError(f"{name} output must be writable")
    return out


def _half_scalar(raw: np.ndarray, offset: int) -> float:
    return float(np.frombuffer(raw[offset : offset + 2], dtype="<f2", count=1)[0])


def _scale_min(scales: np.ndarray, index: int) -> tuple[int, int]:
    if index < 4:
        return int(scales[index] & 0x3F), int(scales[index + 4] & 0x3F)
    scale = int(scales[index + 4] & 0x0F) | ((int(scales[index - 4]) >> 6) << 4)
    minimum = int(scales[index + 4] >> 4) | ((int(scales[index]) >> 6) << 4)
    return scale, minimum


def decode_q5_k_block(block: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Decode one 176-byte GGML Q5_K block to 256 FP32 values."""
    raw = _uint8_block(block, Q5_K_BLOCK_BYTES, "Q5_K")
    result = _decode_output(out, Q5_K_BLOCK_ELEMENTS, "Q5_K")
    d = _half_scalar(raw, 0)
    dmin = _half_scalar(raw, 2)
    scales = raw[4:16]
    qh = raw[16:48]
    ql = raw[48:]
    for subblock in range(8):
        scale, minimum = _scale_min(scales, subblock)
        factor = np.float32(d * scale)
        offset = np.float32(dmin * minimum)
        packed = ql[(subblock // 2) * 32 : (subblock // 2 + 1) * 32]
        codes = packed & 0x0F if subblock % 2 == 0 else packed >> 4
        high = ((qh >> subblock) & 1).astype(np.float32) * np.float32(16.0)
        np.subtract(
            np.multiply(codes.astype(np.float32) + high, factor),
            offset,
            out=result[subblock * 32 : (subblock + 1) * 32],
        )
    return result


def decode_q5_1_block(block: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Decode one 24-byte GGML Q5_1 block to 32 FP32 values."""
    raw = _uint8_block(block, Q5_1_BLOCK_BYTES, "Q5_1")
    result = _decode_output(out, Q5_1_BLOCK_ELEMENTS, "Q5_1")
    d = np.float32(_half_scalar(raw, 0))
    minimum = np.float32(_half_scalar(raw, 2))
    qh = int.from_bytes(raw[4:8].tobytes(), "little")
    qs = raw[8:]
    high_low = np.array([(qh >> lane) & 1 for lane in range(16)], dtype=np.float32)
    high_high = np.array([(qh >> (lane + 16)) & 1 for lane in range(16)], dtype=np.float32)
    low = (qs & 0x0F).astype(np.float32) + high_low * np.float32(16.0)
    high = (qs >> 4).astype(np.float32) + high_high * np.float32(16.0)
    result[:16] = low * d + minimum
    result[16:] = high * d + minimum
    return result


def decode_q8_0_block(block: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Decode one 34-byte GGML Q8_0 block to 32 FP32 values."""
    raw = _uint8_block(block, Q8_0_BLOCK_BYTES, "Q8_0")
    result = _decode_output(out, Q8_0_BLOCK_ELEMENTS, "Q8_0")
    d = np.float32(_half_scalar(raw, 0))
    result[:] = raw[2:].view(np.int8).astype(np.float32) * d
    return result


def _dequantize_blocks(
    packed: Any,
    *,
    block_bytes: int,
    block_elements: int,
    decoder: Callable[..., np.ndarray],
    name: str,
    out: np.ndarray | None,
) -> np.ndarray:
    raw = np.asarray(packed)
    if (
        raw.dtype != np.dtype(np.uint8)
        or raw.ndim < 1
        or raw.shape[-1] <= 0
        or raw.shape[-1] % block_bytes
        or not raw.flags.c_contiguous
    ):
        raise ValueError(
            f"packed {name} rows must end in a multiple of {block_bytes} contiguous uint8 bytes, "
            f"got {raw.shape}"
        )
    shape = (*raw.shape[:-1], raw.shape[-1] // block_bytes * block_elements)
    elements = int(np.prod(shape, dtype=np.int64))
    if out is None:
        result = np.empty(elements, dtype=np.float32)
    else:
        if not isinstance(out, np.ndarray) or out.dtype != np.dtype(np.float32):
            raise TypeError(f"{name} output must be a float32 NumPy ndarray")
        if out.size != elements or not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError(f"{name} output has the wrong size, layout or writeability")
        result = out.reshape(-1)
    blocks = raw.reshape(-1, block_bytes)
    for index, block in enumerate(blocks):
        decoder(block, out=result[index * block_elements : (index + 1) * block_elements])
    return result.reshape(shape)


def dequantize_q5_k(packed: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Decode one or more packed Q5_K blocks, preserving leading dimensions."""
    return _dequantize_blocks(
        packed,
        block_bytes=Q5_K_BLOCK_BYTES,
        block_elements=Q5_K_BLOCK_ELEMENTS,
        decoder=decode_q5_k_block,
        name="Q5_K",
        out=out,
    )


def dequantize_q5_1(packed: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Decode one or more packed Q5_1 blocks, preserving leading dimensions."""
    return _dequantize_blocks(
        packed,
        block_bytes=Q5_1_BLOCK_BYTES,
        block_elements=Q5_1_BLOCK_ELEMENTS,
        decoder=decode_q5_1_block,
        name="Q5_1",
        out=out,
    )


def dequantize_q8_0(packed: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Decode one or more packed Q8_0 blocks, preserving leading dimensions."""
    return _dequantize_blocks(
        packed,
        block_bytes=Q8_0_BLOCK_BYTES,
        block_elements=Q8_0_BLOCK_ELEMENTS,
        decoder=decode_q8_0_block,
        name="Q8_0",
        out=out,
    )


def _decode_rows(
    packed: Any,
    descriptor: Any,
    *,
    out: np.ndarray,
    block_bytes: int,
    block_elements: int,
    decoder: Callable[..., np.ndarray],
    name: str,
) -> np.ndarray:
    raw = np.asarray(packed)
    input_dim = int(descriptor.input_dim)
    output_dim = int(descriptor.output_dim)
    expected_stride = input_dim // block_elements * block_bytes if input_dim > 0 else 0
    if input_dim <= 0 or input_dim % block_elements:
        raise UnsupportedShape(
            f"packed {descriptor.projection} {name} input dimension {input_dim} "
            f"is not a positive multiple of {block_elements}"
        )
    if int(descriptor.row_stride_bytes) != expected_stride:
        raise UnsupportedShape(
            f"packed {descriptor.projection} {name} row stride "
            f"{descriptor.row_stride_bytes}, expected {expected_stride}"
        )
    if (
        raw.dtype != np.dtype(np.uint8)
        or raw.ndim != 2
        or raw.shape
        != (
            output_dim,
            expected_stride,
        )
    ):
        raise UnsupportedShape(
            f"packed {descriptor.projection} {name} shape {raw.shape}, "
            f"expected {(output_dim, expected_stride)}"
        )
    if not raw.flags.c_contiguous:
        raise InvalidRequest(f"packed {descriptor.projection} {name} rows must be C-contiguous")
    if (
        not isinstance(out, np.ndarray)
        or out.dtype != np.dtype(np.float32)
        or out.shape != (output_dim, input_dim)
        or not out.flags.c_contiguous
        or not out.flags.writeable
    ):
        raise InvalidRequest(
            f"{name} workspace must be a contiguous writable float32 matrix "
            f"with shape {(output_dim, input_dim)}"
        )
    blocks_per_row = input_dim // block_elements
    for row in range(output_dim):
        row_bytes = raw[row]
        target = out[row]
        for block in range(blocks_per_row):
            begin = block * block_bytes
            decoder(
                row_bytes[begin : begin + block_bytes],
                out=target[block * block_elements : (block + 1) * block_elements],
            )
    return out


def decode_q5_k_rows(packed: Any, descriptor: Any, *, out: np.ndarray) -> np.ndarray:
    """Decode packed Q5_K expert rows into the supplied matrix workspace."""
    return _decode_rows(
        packed,
        descriptor,
        out=out,
        block_bytes=Q5_K_BLOCK_BYTES,
        block_elements=Q5_K_BLOCK_ELEMENTS,
        decoder=decode_q5_k_block,
        name="Q5_K",
    )


def decode_q5_1_rows(packed: Any, descriptor: Any, *, out: np.ndarray) -> np.ndarray:
    """Decode packed Q5_1 expert rows into the supplied matrix workspace."""
    return _decode_rows(
        packed,
        descriptor,
        out=out,
        block_bytes=Q5_1_BLOCK_BYTES,
        block_elements=Q5_1_BLOCK_ELEMENTS,
        decoder=decode_q5_1_block,
        name="Q5_1",
    )


def decode_q8_0_rows(packed: Any, descriptor: Any, *, out: np.ndarray) -> np.ndarray:
    """Decode packed Q8_0 expert rows into the supplied matrix workspace."""
    return _decode_rows(
        packed,
        descriptor,
        out=out,
        block_bytes=Q8_0_BLOCK_BYTES,
        block_elements=Q8_0_BLOCK_ELEMENTS,
        decoder=decode_q8_0_block,
        name="Q8_0",
    )


BUILTIN_REFERENCE_DECODERS: dict[int | str, Callable[..., np.ndarray]] = {
    13: decode_q5_k_rows,
    "Q5_K": decode_q5_k_rows,
    "q5_k": decode_q5_k_rows,
    7: decode_q5_1_rows,
    "Q5_1": decode_q5_1_rows,
    "q5_1": decode_q5_1_rows,
    8: decode_q8_0_rows,
    "Q8_0": decode_q8_0_rows,
    "q8_0": decode_q8_0_rows,
}


__all__ = [
    "BUILTIN_REFERENCE_DECODERS",
    "Q5_1_BLOCK_BYTES",
    "Q5_1_BLOCK_ELEMENTS",
    "Q5_K_BLOCK_BYTES",
    "Q5_K_BLOCK_ELEMENTS",
    "Q8_0_BLOCK_BYTES",
    "Q8_0_BLOCK_ELEMENTS",
    "decode_q5_1_block",
    "decode_q5_1_rows",
    "decode_q5_k_block",
    "decode_q5_k_rows",
    "decode_q8_0_block",
    "decode_q8_0_rows",
    "dequantize_q5_1",
    "dequantize_q5_k",
    "dequantize_q8_0",
]
