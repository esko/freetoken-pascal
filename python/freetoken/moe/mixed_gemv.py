"""Torch-free direct GEMV primitives for the Qwen3.8 mixed GGML expert banks.

The Q4 artifact uses Q5_K for the layer-2 gate/up bank and Q5_1 or Q8_0 for
down banks in the promoted layers.  This module exposes those packed-row
primitives without coupling them to the CPU executor.  The scalar Python
implementation delegates block decoding to :mod:`ggml_reference`; the
optional C helper adds runtime-dispatched AVX2/FMA implementations while
retaining the same packed layout and a safe scalar fallback.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import platform
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

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

MixedGemvMode = Literal["auto", "scalar", "avx2"]
Q5_1_QUANT_TYPE = 7
Q8_0_QUANT_TYPE = 8
Q5_K_QUANT_TYPE = 13

_Format = tuple[int, int, int]
_FORMATS: dict[str, _Format] = {
    "Q5_1": (Q5_1_QUANT_TYPE, Q5_1_BLOCK_ELEMENTS, Q5_1_BLOCK_BYTES),
    "Q8_0": (Q8_0_QUANT_TYPE, Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES),
    "Q5_K": (Q5_K_QUANT_TYPE, Q5_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES),
}
_FORMAT_BY_TYPE = {value[0]: name for name, value in _FORMATS.items()}
_DECODERS = {
    "Q5_1": decode_q5_1_block,
    "Q8_0": decode_q8_0_block,
    "Q5_K": decode_q5_k_block,
}


def _format_name(quant_name: str | int) -> str:
    if isinstance(quant_name, bool):
        raise ValueError(f"quant_name must be Q5_K, Q5_1 or Q8_0, got {quant_name!r}")
    if isinstance(quant_name, (int, np.integer)):
        try:
            return _FORMAT_BY_TYPE[int(quant_name)]
        except KeyError as error:
            raise ValueError(f"unsupported quant type {quant_name!r}") from error
    if not isinstance(quant_name, str):
        raise ValueError(f"quant_name must be Q5_K, Q5_1 or Q8_0, got {quant_name!r}")
    normalized = quant_name.upper()
    if normalized not in _FORMATS:
        raise ValueError(f"unsupported quant_name {quant_name!r}")
    return normalized


def _uint8_block(block: Any, block_bytes: int, name: str) -> np.ndarray:
    raw = np.asarray(block)
    if raw.dtype != np.dtype(np.uint8):
        raise TypeError(f"{name} block must have dtype uint8, got {raw.dtype}")
    if raw.ndim != 1 or raw.size != block_bytes:
        raise ValueError(f"{name} block must be a contiguous 1-D {block_bytes}-byte view")
    if not raw.flags.c_contiguous:
        raise ValueError(f"{name} block must be C-contiguous")
    return raw


def _decode_output(out: np.ndarray | None, elements: int, name: str) -> np.ndarray:
    if out is None:
        return np.empty(elements, dtype=np.float32)
    if not isinstance(out, np.ndarray) or out.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} output must be a float32 NumPy ndarray")
    if out.shape != (elements,) or not out.flags.c_contiguous:
        raise ValueError(f"{name} output must be a contiguous {elements}-element float32 vector")
    if not out.flags.writeable:
        raise ValueError(f"{name} output must be writable")
    return out


def _scalar_dot_q5_1(block: np.ndarray, vector: np.ndarray) -> np.float32:
    d = np.float32(np.frombuffer(block[:2].tobytes(), dtype="<f2")[0])
    minimum = np.float32(np.frombuffer(block[2:4].tobytes(), dtype="<f2")[0])
    qh = int.from_bytes(block[4:8].tobytes(), "little")
    result = np.float32(0.0)
    for index in range(Q5_1_BLOCK_ELEMENTS):
        lane = index & 15
        packed = int(block[8 + lane])
        code = (packed & 0x0F) if index < 16 else (packed >> 4)
        code |= ((qh >> index) & 1) << 4
        result = np.float32(result + np.float32((np.float32(code) * d + minimum) * vector[index]))
    return result


def _scalar_dot_q8_0(block: np.ndarray, vector: np.ndarray) -> np.float32:
    d = np.float32(np.frombuffer(block[:2].tobytes(), dtype="<f2")[0])
    result = np.float32(0.0)
    for index in range(Q8_0_BLOCK_ELEMENTS):
        code = np.int8(block[2 + index])
        result = np.float32(result + np.float32(np.float32(code) * d * vector[index]))
    return result


def _scale_min(scales: np.ndarray, index: int) -> tuple[int, int]:
    if index < 4:
        return int(scales[index] & 0x3F), int(scales[index + 4] & 0x3F)
    scale = int(scales[index + 4] & 0x0F) | ((int(scales[index - 4]) >> 6) << 4)
    minimum = int(scales[index + 4] >> 4) | ((int(scales[index]) >> 6) << 4)
    return scale, minimum


def _scalar_dot_q5_k(block: np.ndarray, vector: np.ndarray) -> np.float32:
    d = np.float32(np.frombuffer(block[:2].tobytes(), dtype="<f2")[0])
    dmin = np.float32(np.frombuffer(block[2:4].tobytes(), dtype="<f2")[0])
    scales = block[4:16]
    qh = block[16:48]
    ql = block[48:]
    result = np.float32(0.0)
    for subblock in range(8):
        scale, minimum = _scale_min(scales, subblock)
        factor = np.float32(d * scale)
        offset = np.float32(dmin * minimum)
        for lane in range(32):
            packed = int(ql[(subblock // 2) * 32 + lane])
            code = (packed & 0x0F) if subblock % 2 == 0 else (packed >> 4)
            code |= ((int(qh[lane]) >> subblock) & 1) << 4
            value = np.float32(np.float32(code) * factor - offset)
            index = subblock * 32 + lane
            result = np.float32(result + np.float32(value * vector[index]))
    return result


_SCALAR_DOTS = {
    "Q5_1": _scalar_dot_q5_1,
    "Q8_0": _scalar_dot_q8_0,
    "Q5_K": _scalar_dot_q5_k,
}


def _host_supports_avx2() -> bool:
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64", "i386", "i686", "x86"}:
        return False
    try:
        with open("/proc/cpuinfo", encoding="ascii") as source:
            flags = source.read().lower()
    except OSError:
        return False
    return " avx2" in f" {flags}" and " fma" in f" {flags}"


class _NativeMixedGemv:
    def __init__(self, library: ctypes.CDLL, source_path: str | None = None) -> None:
        self.library = library
        self.source_path = source_path or str(library._name)
        float_pointer = ctypes.POINTER(ctypes.c_float)
        byte_pointer = ctypes.POINTER(ctypes.c_uint8)
        for name in _FORMATS:
            stem = name.lower()
            dot = getattr(library, f"freetoken_mixed_{stem}_dot")
            dot.argtypes = [byte_pointer, float_pointer]
            dot.restype = ctypes.c_float
            decode = getattr(library, f"freetoken_mixed_{stem}_decode")
            decode.argtypes = [byte_pointer, float_pointer]
            decode.restype = None
            gemv = getattr(library, f"freetoken_mixed_{stem}_gemv")
            gemv.argtypes = [
                byte_pointer,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                float_pointer,
                float_pointer,
            ]
            gemv.restype = ctypes.c_int

    @staticmethod
    def _byte_pointer(value: np.ndarray) -> Any:
        return value.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    @staticmethod
    def _float_pointer(value: np.ndarray) -> Any:
        return value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def dot(self, name: str, block: np.ndarray, vector: np.ndarray) -> float:
        return float(
            getattr(self.library, f"freetoken_mixed_{name.lower()}_dot")(
                self._byte_pointer(block), self._float_pointer(vector)
            )
        )

    def decode(self, name: str, block: np.ndarray, out: np.ndarray) -> None:
        getattr(self.library, f"freetoken_mixed_{name.lower()}_decode")(
            self._byte_pointer(block), self._float_pointer(out)
        )

    def gemv(
        self,
        name: str,
        rows: np.ndarray,
        row_count: int,
        blocks_per_row: int,
        row_stride_bytes: int,
        vector: np.ndarray,
        out: np.ndarray,
    ) -> None:
        status = getattr(self.library, f"freetoken_mixed_{name.lower()}_gemv")(
            self._byte_pointer(rows),
            row_count,
            blocks_per_row,
            row_stride_bytes,
            self._float_pointer(vector),
            self._float_pointer(out),
        )
        if status != 0:
            raise ValueError(f"native {name} GEMV rejected the validated layout")


def _load_native() -> _NativeMixedGemv | None:
    candidates: list[str] = []
    configured = os.environ.get("FREETOKEN_MIXED_GEMV_NATIVE_LIB")
    if configured:
        candidates.append(configured)
    try:
        spec = importlib.util.find_spec("freetoken.moe._mixed_gemv_native")
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        candidates.append(spec.origin)
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
            available = library.freetoken_mixed_cpu_supports_avx2
            available.argtypes = []
            available.restype = ctypes.c_int
            if available() and _host_supports_avx2():
                return _NativeMixedGemv(library, candidate)
        except (AttributeError, OSError):
            continue
    return None


@dataclass(frozen=True)
class MixedGemvPrimitive:
    """Selected mixed-format GEMV implementation and observable fallback."""

    requested_mode: str
    isa: Literal["scalar", "avx2"]
    fallback_reason: str | None
    native: _NativeMixedGemv | None

    @property
    def backend(self) -> str:
        return f"mixed_gemv_{self.isa}"

    def backend_for(self, quant_name: str | int) -> str:
        return f"{_format_name(quant_name).lower()}_{self.isa}"

    def decode(
        self, block: Any, *, quant_name: str | int, out: np.ndarray | None = None
    ) -> np.ndarray:
        name = _format_name(quant_name)
        _, elements, block_bytes = _FORMATS[name]
        raw = _uint8_block(block, block_bytes, name)
        result = _decode_output(out, elements, name)
        if np.shares_memory(raw, result):
            raise ValueError(f"{name} decode input and output must not overlap")
        if self.native is None:
            _DECODERS[name](raw, out=result)
        else:
            self.native.decode(name, raw, result)
        return result

    def dot(self, block: Any, vector: Any, *, quant_name: str | int) -> float:
        name = _format_name(quant_name)
        _, elements, block_bytes = _FORMATS[name]
        raw = _uint8_block(block, block_bytes, name)
        values = np.asarray(vector)
        if values.dtype != np.dtype(np.float32) or values.shape != (elements,):
            raise ValueError(
                f"{name} dot input must be a contiguous {elements}-element float32 vector"
            )
        if not values.flags.c_contiguous:
            raise ValueError(f"{name} dot input must be a contiguous float32 vector")
        if self.native is None:
            return float(_SCALAR_DOTS[name](raw, values))
        return self.native.dot(name, raw, values)

    def gemv(
        self,
        rows: Any,
        input_dim: int,
        vector: Any,
        *,
        quant_name: str | int,
        out: np.ndarray,
    ) -> np.ndarray:
        name = _format_name(quant_name)
        _, block_elements, block_bytes = _FORMATS[name]
        if isinstance(input_dim, bool) or not isinstance(input_dim, (int, np.integer)):
            raise ValueError(f"{name} GEMV input dimension must be an integer")
        input_dim = int(input_dim)
        if input_dim <= 0 or input_dim % block_elements:
            raise ValueError(
                f"{name} GEMV input dimension must be a positive multiple of {block_elements}, "
                f"got {input_dim}"
            )
        packed = np.asarray(rows)
        blocks_per_row = input_dim // block_elements
        expected_stride = blocks_per_row * block_bytes
        if (
            packed.dtype != np.dtype(np.uint8)
            or packed.ndim != 2
            or packed.shape[0] <= 0
            or packed.shape[1] != expected_stride
            or not packed.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} GEMV rows must be positive contiguous uint8 rows with shape "
                f"(rows, {expected_stride})"
            )
        values = np.asarray(vector)
        if (
            values.dtype != np.dtype(np.float32)
            or values.shape != (input_dim,)
            or not values.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} GEMV input must be a contiguous float32 vector with shape ({input_dim},)"
            )
        if (
            not isinstance(out, np.ndarray)
            or out.dtype != np.dtype(np.float32)
            or out.shape != (packed.shape[0],)
            or not out.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} GEMV output must be a contiguous float32 vector with shape "
                f"({packed.shape[0]},)"
            )
        if not out.flags.writeable:
            raise ValueError(f"{name} GEMV output must be writable")
        if np.shares_memory(values, out):
            raise ValueError(f"{name} GEMV input and output must not overlap")
        if np.shares_memory(packed, out):
            raise ValueError(f"{name} GEMV packed rows and output must not overlap")
        if self.native is not None:
            self.native.gemv(
                name,
                packed,
                packed.shape[0],
                blocks_per_row,
                expected_stride,
                values,
                out,
            )
            return out
        scalar_dot = _SCALAR_DOTS[name]
        for row in range(packed.shape[0]):
            total = np.float32(0.0)
            row_bytes = packed[row]
            for block in range(blocks_per_row):
                block_begin = block * block_bytes
                input_begin = block * block_elements
                total = np.float32(
                    total
                    + scalar_dot(
                        row_bytes[block_begin : block_begin + block_bytes],
                        values[input_begin : input_begin + block_elements],
                    )
                )
            out[row] = total
        return out

    def q5_1_gemv(self, rows: Any, input_dim: int, vector: Any, *, out: np.ndarray) -> np.ndarray:
        return self.gemv(rows, input_dim, vector, quant_name="Q5_1", out=out)

    def q8_0_gemv(self, rows: Any, input_dim: int, vector: Any, *, out: np.ndarray) -> np.ndarray:
        return self.gemv(rows, input_dim, vector, quant_name="Q8_0", out=out)

    def q5_k_gemv(self, rows: Any, input_dim: int, vector: Any, *, out: np.ndarray) -> np.ndarray:
        return self.gemv(rows, input_dim, vector, quant_name="Q5_K", out=out)


def select_mixed_gemv_primitive(mode: MixedGemvMode = "auto") -> MixedGemvPrimitive:
    """Select scalar or runtime-dispatched AVX2 mixed-format GEMV."""

    normalized = str(mode).lower()
    if normalized in {"forced_scalar", "scalar"}:
        return MixedGemvPrimitive(normalized, "scalar", None, None)
    if normalized not in {"auto", "avx2", "forced_avx2"}:
        raise ValueError(f"mixed GEMV mode must be auto, scalar or avx2, got {mode!r}")
    if not _host_supports_avx2():
        return MixedGemvPrimitive(normalized, "scalar", "avx2_unavailable", None)
    native = _load_native()
    if native is None:
        return MixedGemvPrimitive(normalized, "scalar", "native_avx2_unavailable", None)
    return MixedGemvPrimitive(normalized, "avx2", None, native)


def mixed_gemv(
    rows: Any,
    input_dim: int,
    vector: Any,
    *,
    quant_name: str | int,
    out: np.ndarray,
    mode: MixedGemvMode = "auto",
) -> np.ndarray:
    """Run one format-tagged packed-row GEMV with a selected primitive."""

    return select_mixed_gemv_primitive(mode).gemv(
        rows, input_dim, vector, quant_name=quant_name, out=out
    )


def gemv_q5_1(
    rows: Any, input_dim: int, vector: Any, *, out: np.ndarray, mode: MixedGemvMode = "auto"
) -> np.ndarray:
    return mixed_gemv(rows, input_dim, vector, quant_name="Q5_1", out=out, mode=mode)


def gemv_q8_0(
    rows: Any, input_dim: int, vector: Any, *, out: np.ndarray, mode: MixedGemvMode = "auto"
) -> np.ndarray:
    return mixed_gemv(rows, input_dim, vector, quant_name="Q8_0", out=out, mode=mode)


def gemv_q5_k(
    rows: Any, input_dim: int, vector: Any, *, out: np.ndarray, mode: MixedGemvMode = "auto"
) -> np.ndarray:
    return mixed_gemv(rows, input_dim, vector, quant_name="Q5_K", out=out, mode=mode)


__all__ = [
    "Q5_1_QUANT_TYPE",
    "Q5_K_QUANT_TYPE",
    "Q8_0_QUANT_TYPE",
    "MixedGemvPrimitive",
    "gemv_q5_1",
    "gemv_q5_k",
    "gemv_q8_0",
    "mixed_gemv",
    "select_mixed_gemv_primitive",
]
