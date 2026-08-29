"""Torch-free reference decoders for the Qwen3.8 mixed GGML expert banks.

The packed layouts and arithmetic in this module follow the pinned llama.cpp
``eaf93765572e794b8e3754fe45adbe12d381e997`` reference.  This is a bounded,
scalar correctness path for Q5_K, Q5_1, Q8_0, IQ3_XXS, IQ4_NL and IQ4_XS; it
is not an AVX2 kernel.
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
IQ3_XXS_BLOCK_ELEMENTS = 256
IQ3_XXS_BLOCK_BYTES = 98
IQ4_NL_BLOCK_ELEMENTS = 32
IQ4_NL_BLOCK_BYTES = 18
IQ4_XS_BLOCK_ELEMENTS = 256
IQ4_XS_BLOCK_BYTES = 136

_CANONICAL_QUANT_IDS = {
    12: "Q4_K",
    7: "Q5_1",
    8: "Q8_0",
    13: "Q5_K",
    18: "IQ3_XXS",
    20: "IQ4_NL",
    23: "IQ4_XS",
}
_CANONICAL_QUANT_NAMES = frozenset(_CANONICAL_QUANT_IDS.values())
_REFERENCE_QUANT_NAMES = _CANONICAL_QUANT_NAMES - {"Q4_K"}


def canonical_quant_name(quant_type: Any, quant_name: Any) -> str | None:
    """Return a canonical packed GGML name only when type and name agree.

    Descriptor metadata is untrusted at this boundary.  A known type paired with
    an unrelated name (or the reverse) must not select a decoder by whichever key
    happens to be present, because that would make telemetry claim the wrong bytes.
    ``None`` also represents an entirely unsupported pair.
    """
    normalized_name = str(quant_name).upper()
    if normalized_name not in _CANONICAL_QUANT_NAMES:
        return None
    try:
        if isinstance(quant_type, bool):
            return None
        type_id = int(quant_type)
    except (TypeError, ValueError):
        type_id = None
    by_id = _CANONICAL_QUANT_IDS.get(type_id)
    if by_id is None and str(quant_type).upper() in _CANONICAL_QUANT_NAMES:
        by_id = str(quant_type).upper()
    return normalized_name if by_id == normalized_name else None


def canonical_reference_quant_name(quant_type: Any, quant_name: Any) -> str | None:
    """Return a canonical name for formats with a built-in reference decoder."""
    canonical = canonical_quant_name(quant_type, quant_name)
    return canonical if canonical in _REFERENCE_QUANT_NAMES else None


# The IQ3_XXS table and IQ4_NL codebook constants below are byte-for-byte
# adaptations from the MIT-licensed ggml/src/ggml-common.h at llama.cpp commit
# ``eaf93765572e794b8e3754fe45adbe12d381e997``.  The dequantization arithmetic
# follows that commit's ggml/src/ggml-quants.c; this module is an independent
# scalar implementation, not a copied upstream source file.
_IQ3_XXS_GRID = (
    0x04040404,
    0x04040414,
    0x04040424,
    0x04040C0C,
    0x04040C1C,
    0x04040C3E,
    0x04041404,
    0x04041414,
    0x04041C0C,
    0x04042414,
    0x04043E1C,
    0x04043E2C,
    0x040C040C,
    0x040C041C,
    0x040C0C04,
    0x040C0C14,
    0x040C140C,
    0x040C142C,
    0x040C1C04,
    0x040C1C14,
    0x040C240C,
    0x040C2C24,
    0x040C3E04,
    0x04140404,
    0x04140414,
    0x04140424,
    0x04140C0C,
    0x04141404,
    0x04141414,
    0x04141C0C,
    0x04141C1C,
    0x04141C3E,
    0x04142C0C,
    0x04142C3E,
    0x04143E2C,
    0x041C040C,
    0x041C043E,
    0x041C0C04,
    0x041C0C14,
    0x041C142C,
    0x041C3E04,
    0x04240C1C,
    0x04241C3E,
    0x04242424,
    0x04242C3E,
    0x04243E1C,
    0x04243E2C,
    0x042C040C,
    0x042C043E,
    0x042C1C14,
    0x042C2C14,
    0x04341C2C,
    0x04343424,
    0x043E0C04,
    0x043E0C24,
    0x043E0C34,
    0x043E241C,
    0x043E340C,
    0x0C04040C,
    0x0C04041C,
    0x0C040C04,
    0x0C040C14,
    0x0C04140C,
    0x0C04141C,
    0x0C041C04,
    0x0C041C14,
    0x0C041C24,
    0x0C04243E,
    0x0C042C04,
    0x0C0C0404,
    0x0C0C0414,
    0x0C0C0C0C,
    0x0C0C1404,
    0x0C0C1414,
    0x0C14040C,
    0x0C14041C,
    0x0C140C04,
    0x0C140C14,
    0x0C14140C,
    0x0C141C04,
    0x0C143E14,
    0x0C1C0404,
    0x0C1C0414,
    0x0C1C1404,
    0x0C1C1C0C,
    0x0C1C2434,
    0x0C1C3434,
    0x0C24040C,
    0x0C24042C,
    0x0C242C04,
    0x0C2C1404,
    0x0C2C1424,
    0x0C2C2434,
    0x0C2C3E0C,
    0x0C34042C,
    0x0C3E1414,
    0x0C3E2404,
    0x14040404,
    0x14040414,
    0x14040C0C,
    0x14040C1C,
    0x14041404,
    0x14041414,
    0x14041434,
    0x14041C0C,
    0x14042414,
    0x140C040C,
    0x140C041C,
    0x140C042C,
    0x140C0C04,
    0x140C0C14,
    0x140C140C,
    0x140C1C04,
    0x140C341C,
    0x140C343E,
    0x140C3E04,
    0x14140404,
    0x14140414,
    0x14140C0C,
    0x14140C3E,
    0x14141404,
    0x14141414,
    0x14141C3E,
    0x14142404,
    0x14142C2C,
    0x141C040C,
    0x141C0C04,
    0x141C0C24,
    0x141C3E04,
    0x141C3E24,
    0x14241C2C,
    0x14242C1C,
    0x142C041C,
    0x142C143E,
    0x142C240C,
    0x142C3E24,
    0x143E040C,
    0x143E041C,
    0x143E0C34,
    0x143E242C,
    0x1C04040C,
    0x1C040C04,
    0x1C040C14,
    0x1C04140C,
    0x1C04141C,
    0x1C042C04,
    0x1C04342C,
    0x1C043E14,
    0x1C0C0404,
    0x1C0C0414,
    0x1C0C1404,
    0x1C0C1C0C,
    0x1C0C2424,
    0x1C0C2434,
    0x1C14040C,
    0x1C14041C,
    0x1C140C04,
    0x1C14142C,
    0x1C142C14,
    0x1C143E14,
    0x1C1C0C0C,
    0x1C1C1C1C,
    0x1C241C04,
    0x1C24243E,
    0x1C243E14,
    0x1C2C0404,
    0x1C2C0434,
    0x1C2C1414,
    0x1C2C2C2C,
    0x1C340C24,
    0x1C341C34,
    0x1C34341C,
    0x1C3E1C1C,
    0x1C3E3404,
    0x24040424,
    0x24040C3E,
    0x24041C2C,
    0x24041C3E,
    0x24042C1C,
    0x24042C3E,
    0x240C3E24,
    0x24141404,
    0x24141C3E,
    0x24142404,
    0x24143404,
    0x24143434,
    0x241C043E,
    0x241C242C,
    0x24240424,
    0x24242C0C,
    0x24243424,
    0x242C142C,
    0x242C241C,
    0x242C3E04,
    0x243E042C,
    0x243E0C04,
    0x243E0C14,
    0x243E1C04,
    0x2C040C14,
    0x2C04240C,
    0x2C043E04,
    0x2C0C0404,
    0x2C0C0434,
    0x2C0C1434,
    0x2C0C2C2C,
    0x2C140C24,
    0x2C141C14,
    0x2C143E14,
    0x2C1C0414,
    0x2C1C2C1C,
    0x2C240C04,
    0x2C24141C,
    0x2C24143E,
    0x2C243E14,
    0x2C2C0414,
    0x2C2C1C0C,
    0x2C342C04,
    0x2C3E1424,
    0x2C3E2414,
    0x34041424,
    0x34042424,
    0x34042434,
    0x34043424,
    0x340C140C,
    0x340C340C,
    0x34140C3E,
    0x34143424,
    0x341C1C04,
    0x341C1C34,
    0x34242424,
    0x342C042C,
    0x342C2C14,
    0x34341C1C,
    0x343E041C,
    0x343E140C,
    0x3E04041C,
    0x3E04042C,
    0x3E04043E,
    0x3E040C04,
    0x3E041C14,
    0x3E042C14,
    0x3E0C1434,
    0x3E0C2404,
    0x3E140C14,
    0x3E14242C,
    0x3E142C14,
    0x3E1C0404,
    0x3E1C0C2C,
    0x3E1C1C1C,
    0x3E1C3404,
    0x3E24140C,
    0x3E24240C,
    0x3E2C0404,
    0x3E2C0414,
    0x3E2C1424,
    0x3E341C04,
)

# ``kvalues_iq4nl`` from the same pinned source, represented as FP32 for the
# scalar path.
_IQ4_NL_VALUES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)


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


def _reject_shared_output(raw: np.ndarray, output: np.ndarray, name: str) -> None:
    """Prevent in-place decode from overwriting packed blocks still being read."""
    if np.shares_memory(raw, output):
        raise ValueError(f"{name} output shares memory with packed input")


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
    _reject_shared_output(raw, result, "Q5_K")
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
    _reject_shared_output(raw, result, "Q5_1")
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
    _reject_shared_output(raw, result, "Q8_0")
    d = np.float32(_half_scalar(raw, 0))
    result[:] = raw[2:].view(np.int8).astype(np.float32) * d
    return result


def _iq3_xxs_grid_bytes(index: int) -> tuple[int, int, int, int]:
    packed = _IQ3_XXS_GRID[index]
    return tuple((packed >> (8 * lane)) & 0xFF for lane in range(4))  # type: ignore[return-value]


def _iq3_xxs_signs(index: int) -> int:
    # This is the compact form of the pinned ksigns_iq2xs[128] table.
    return index | ((index.bit_count() & 1) << 7)


def decode_iq3_xxs_block(block: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Decode one 98-byte GGML IQ3_XXS block to 256 FP32 values."""
    raw = _uint8_block(block, IQ3_XXS_BLOCK_BYTES, "IQ3_XXS")
    result = _decode_output(out, IQ3_XXS_BLOCK_ELEMENTS, "IQ3_XXS")
    _reject_shared_output(raw, result, "IQ3_XXS")
    d = np.float32(_half_scalar(raw, 0))
    q3_start = 2
    gas_start = q3_start + IQ3_XXS_BLOCK_ELEMENTS // 4
    for subblock in range(8):
        q3 = raw[q3_start + 8 * subblock : q3_start + 8 * subblock + 8]
        aux32 = int.from_bytes(
            raw[gas_start + 4 * subblock : gas_start + 4 * subblock + 4], "little"
        )
        factor = np.float32(d * np.float32(0.5 + (aux32 >> 28)) * np.float32(0.5))
        for group in range(4):
            signs = _iq3_xxs_signs((aux32 >> (7 * group)) & 0x7F)
            first = _iq3_xxs_grid_bytes(int(q3[2 * group]))
            second = _iq3_xxs_grid_bytes(int(q3[2 * group + 1]))
            begin = 32 * subblock + 8 * group
            for lane, magnitude in enumerate((*first, *second)):
                value = np.float32(magnitude) * factor
                result[begin + lane] = -value if signs & (1 << lane) else value
    return result


def decode_iq4_nl_block(block: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Decode one 18-byte GGML IQ4_NL block to 32 FP32 values."""
    raw = _uint8_block(block, IQ4_NL_BLOCK_BYTES, "IQ4_NL")
    result = _decode_output(out, IQ4_NL_BLOCK_ELEMENTS, "IQ4_NL")
    _reject_shared_output(raw, result, "IQ4_NL")
    d = np.float32(_half_scalar(raw, 0))
    codes = raw[2:]
    result[:16] = _IQ4_NL_VALUES[codes & 0x0F] * d
    result[16:] = _IQ4_NL_VALUES[codes >> 4] * d
    return result


def decode_iq4_xs_block(block: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Decode one 136-byte GGML IQ4_XS block to 256 FP32 values."""
    raw = _uint8_block(block, IQ4_XS_BLOCK_BYTES, "IQ4_XS")
    result = _decode_output(out, IQ4_XS_BLOCK_ELEMENTS, "IQ4_XS")
    _reject_shared_output(raw, result, "IQ4_XS")
    d = np.float32(_half_scalar(raw, 0))
    scales_h = int.from_bytes(raw[2:4].tobytes(), "little")
    scales_l = raw[4:8]
    qs = raw[8:]
    for subblock in range(8):
        scale = ((int(scales_l[subblock // 2]) >> (4 * (subblock % 2))) & 0x0F) | (
            ((scales_h >> (2 * subblock)) & 0x03) << 4
        )
        factor = np.float32(d * np.float32(scale - 32))
        codes = qs[16 * subblock : 16 * subblock + 16]
        begin = 32 * subblock
        result[begin : begin + 16] = _IQ4_NL_VALUES[codes & 0x0F] * factor
        result[begin + 16 : begin + 32] = _IQ4_NL_VALUES[codes >> 4] * factor
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
        _reject_shared_output(raw, out, name)
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


def dequantize_iq3_xxs(packed: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Decode one or more packed IQ3_XXS blocks, preserving leading dimensions."""
    return _dequantize_blocks(
        packed,
        block_bytes=IQ3_XXS_BLOCK_BYTES,
        block_elements=IQ3_XXS_BLOCK_ELEMENTS,
        decoder=decode_iq3_xxs_block,
        name="IQ3_XXS",
        out=out,
    )


def dequantize_iq4_nl(packed: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Decode one or more packed IQ4_NL blocks, preserving leading dimensions."""
    return _dequantize_blocks(
        packed,
        block_bytes=IQ4_NL_BLOCK_BYTES,
        block_elements=IQ4_NL_BLOCK_ELEMENTS,
        decoder=decode_iq4_nl_block,
        name="IQ4_NL",
        out=out,
    )


def dequantize_iq4_xs(packed: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Decode one or more packed IQ4_XS blocks, preserving leading dimensions."""
    return _dequantize_blocks(
        packed,
        block_bytes=IQ4_XS_BLOCK_BYTES,
        block_elements=IQ4_XS_BLOCK_ELEMENTS,
        decoder=decode_iq4_xs_block,
        name="IQ4_XS",
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
    _reject_shared_output(raw, out, name)
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


def decode_iq3_xxs_rows(packed: Any, descriptor: Any, *, out: np.ndarray) -> np.ndarray:
    """Decode packed IQ3_XXS expert rows into the supplied matrix workspace."""
    return _decode_rows(
        packed,
        descriptor,
        out=out,
        block_bytes=IQ3_XXS_BLOCK_BYTES,
        block_elements=IQ3_XXS_BLOCK_ELEMENTS,
        decoder=decode_iq3_xxs_block,
        name="IQ3_XXS",
    )


def decode_iq4_nl_rows(packed: Any, descriptor: Any, *, out: np.ndarray) -> np.ndarray:
    """Decode packed IQ4_NL expert rows into the supplied matrix workspace."""
    return _decode_rows(
        packed,
        descriptor,
        out=out,
        block_bytes=IQ4_NL_BLOCK_BYTES,
        block_elements=IQ4_NL_BLOCK_ELEMENTS,
        decoder=decode_iq4_nl_block,
        name="IQ4_NL",
    )


def decode_iq4_xs_rows(packed: Any, descriptor: Any, *, out: np.ndarray) -> np.ndarray:
    """Decode packed IQ4_XS expert rows into the supplied matrix workspace."""
    return _decode_rows(
        packed,
        descriptor,
        out=out,
        block_bytes=IQ4_XS_BLOCK_BYTES,
        block_elements=IQ4_XS_BLOCK_ELEMENTS,
        decoder=decode_iq4_xs_block,
        name="IQ4_XS",
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
    18: decode_iq3_xxs_rows,
    "IQ3_XXS": decode_iq3_xxs_rows,
    "iq3_xxs": decode_iq3_xxs_rows,
    20: decode_iq4_nl_rows,
    "IQ4_NL": decode_iq4_nl_rows,
    "iq4_nl": decode_iq4_nl_rows,
    23: decode_iq4_xs_rows,
    "IQ4_XS": decode_iq4_xs_rows,
    "iq4_xs": decode_iq4_xs_rows,
}


__all__ = [
    "BUILTIN_REFERENCE_DECODERS",
    "IQ3_XXS_BLOCK_BYTES",
    "IQ3_XXS_BLOCK_ELEMENTS",
    "IQ4_NL_BLOCK_BYTES",
    "IQ4_NL_BLOCK_ELEMENTS",
    "IQ4_XS_BLOCK_BYTES",
    "IQ4_XS_BLOCK_ELEMENTS",
    "Q5_1_BLOCK_BYTES",
    "Q5_1_BLOCK_ELEMENTS",
    "Q5_K_BLOCK_BYTES",
    "Q5_K_BLOCK_ELEMENTS",
    "Q8_0_BLOCK_BYTES",
    "Q8_0_BLOCK_ELEMENTS",
    "canonical_quant_name",
    "canonical_reference_quant_name",
    "decode_iq3_xxs_block",
    "decode_iq3_xxs_rows",
    "decode_iq4_nl_block",
    "decode_iq4_nl_rows",
    "decode_iq4_xs_block",
    "decode_iq4_xs_rows",
    "decode_q5_1_block",
    "decode_q5_1_rows",
    "decode_q5_k_block",
    "decode_q5_k_rows",
    "decode_q8_0_block",
    "decode_q8_0_rows",
    "dequantize_iq3_xxs",
    "dequantize_iq4_nl",
    "dequantize_iq4_xs",
    "dequantize_q5_1",
    "dequantize_q5_k",
    "dequantize_q8_0",
]
