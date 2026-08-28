"""Torch-free Q4_K oracle and the Issue #15 CPU expert adapter.

Q4_K is the first low-bit format in the host expert path.  The scalar decoder
is deliberately independent of Torch and NumPy's BLAS kernels: it is the
correctness oracle for the optional AVX2 primitive in ``q4_k_native.cpp``.
The native helper is a small C ABI shared library with no Python, Torch or
CUDA link; the adapter uses it only after a runtime CPU-feature and address
check.  Its path may be supplied through ``FREETOKEN_Q4K_NATIVE_LIB`` or the
package extension built by ``setup.py``.

The packed block arithmetic follows the in-tree GGML ``block_q4_K`` layout:
two little-endian FP16 super-scales, twelve packed 6-bit scale/min values, and
128 bytes containing two nibbles for each of the four 64-value groups.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import platform
import threading
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_EXCEPTION, Executor, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from freetoken.moe.cpu_abi import (
    Busy,
    Cancelled,
    CpuAbiError,
    CpuExecutionRequest,
    CpuExecutionResult,
    CpuExecutionTelemetry,
    CpuExpertDescriptor,
    CpuExpertLayout,
    CpuMicrobenchmarkProjection,
    CpuMicrobenchmarkSample,
    ExecutionFailed,
    InvalidRequest,
    ReferenceCpuExpertExecutor,
    UnsupportedQuantType,
    UnsupportedShape,
    WorkspaceNotPrepared,
    _cancelled,
    _checked_int,
    _clear_output,
)
from freetoken.moe.ggml_reference import BUILTIN_REFERENCE_DECODERS
from freetoken.moe.mixed_gemv import _FORMATS as _MIXED_FORMATS
from freetoken.moe.mixed_gemv import MixedGemvPrimitive, select_mixed_gemv_primitive

Q4K_BLOCK_ELEMENTS = 256
Q4K_BLOCK_BYTES = 144
Q4K_SCALE_BYTES = 12
Q4K_DATA_OFFSET = 16
Q4K_MODE = Literal["auto", "scalar", "avx2"]
_Q4K_TYPES = frozenset({12, "Q4_K", "q4_k"})


def partition_q4_k_routes(route_count: int, thread_count: int) -> tuple[tuple[int, int], ...]:
    """Partition route columns into balanced, contiguous worker ranges."""
    route_count = _checked_int(route_count, "route_count")
    thread_count = _checked_int(thread_count, "thread_count")
    if route_count <= 0:
        raise ValueError(f"route_count must be positive, got {route_count}")
    if thread_count <= 0:
        raise ValueError(f"thread_count must be positive, got {thread_count}")
    workers = min(route_count, thread_count)
    base, remainder = divmod(route_count, workers)
    ranges: list[tuple[int, int]] = []
    begin = 0
    for worker in range(workers):
        end = begin + base + int(worker < remainder)
        ranges.append((begin, end))
        begin = end
    return tuple(ranges)


def _is_q4_k_descriptor(descriptor: CpuExpertDescriptor) -> bool:
    return descriptor.quant_type in _Q4K_TYPES and descriptor.quant_name.upper() == "Q4_K"


def _has_q4_k_geometry(descriptor: CpuExpertDescriptor) -> bool:
    return descriptor.input_dim % Q4K_BLOCK_ELEMENTS == 0 and descriptor.row_stride_bytes == (
        descriptor.input_dim // Q4K_BLOCK_ELEMENTS * Q4K_BLOCK_BYTES
    )


def _has_q4_k_marker(descriptor: CpuExpertDescriptor) -> bool:
    return descriptor.quant_type in _Q4K_TYPES or descriptor.quant_name.upper() == "Q4_K"


def _mixed_format(descriptor: CpuExpertDescriptor) -> str | None:
    """Return a canonical companion format when type and name agree."""
    name = descriptor.quant_name.upper()
    if name not in _MIXED_FORMATS:
        return None
    quant_type = _MIXED_FORMATS[name][0]
    if descriptor.quant_type not in {quant_type, name, name.lower()}:
        return None
    return name


def _has_mixed_geometry(descriptor: CpuExpertDescriptor) -> bool:
    name = _mixed_format(descriptor)
    if name is None:
        return False
    _, block_elements, block_bytes = _MIXED_FORMATS[name]
    return descriptor.input_dim % block_elements == 0 and descriptor.row_stride_bytes == (
        descriptor.input_dim // block_elements * block_bytes
    )


def _packed_source(descriptor: CpuExpertDescriptor) -> bool:
    source = descriptor.source
    return callable(getattr(source, "expert_packed", None)) and not isinstance(source, np.ndarray)


def _uint8_block(block: Any) -> np.ndarray:
    raw = np.asarray(block)
    if raw.dtype != np.dtype(np.uint8):
        raise TypeError(f"Q4_K block must have dtype uint8, got {raw.dtype}")
    if raw.ndim != 1 or raw.size != Q4K_BLOCK_BYTES:
        raise ValueError(
            f"Q4_K block must be a contiguous 1-D {Q4K_BLOCK_BYTES}-byte view, got {raw.shape}"
        )
    if not raw.flags.c_contiguous:
        raise ValueError("Q4_K block must be C-contiguous")
    return raw


def _scale_min(scales: np.ndarray, index: int) -> tuple[int, int]:
    if index < 4:
        return int(scales[index] & 0x3F), int(scales[index + 4] & 0x3F)
    scale = int(scales[index + 4] & 0x0F) | ((int(scales[index - 4]) >> 6) << 4)
    minimum = int(scales[index + 4] >> 4) | ((int(scales[index]) >> 6) << 4)
    return scale, minimum


def _half_scalar(raw: np.ndarray, offset: int) -> float:
    # A tiny view instead of an integer reinterpretation keeps IEEE half
    # handling identical on all supported Python architectures.
    return float(np.frombuffer(raw[offset : offset + 2], dtype="<f2", count=1)[0])


def decode_q4_k_block(block: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Decode one native 144-byte Q4_K block to 256 FP32 values.

    ``out`` is optional for the reference API, but native executor adapters pass
    their preallocated workspace so route execution does not allocate a dense
    matrix per expert.
    """

    raw = _uint8_block(block)
    if out is None:
        result = np.empty(Q4K_BLOCK_ELEMENTS, dtype=np.float32)
    else:
        if not isinstance(out, np.ndarray) or out.dtype != np.dtype(np.float32):
            raise TypeError("Q4_K output must be a float32 NumPy ndarray")
        if out.ndim != 1 or out.size != Q4K_BLOCK_ELEMENTS or not out.flags.c_contiguous:
            raise ValueError("Q4_K output must be a contiguous 256-element float32 vector")
        if not out.flags.writeable:
            raise ValueError("Q4_K output must be writable")
        result = out

    d = _half_scalar(raw, 0)
    dmin = _half_scalar(raw, 2)
    scales = raw[4:16]
    qs = raw[Q4K_DATA_OFFSET:]
    for subblock in range(8):
        scale, minimum = _scale_min(scales, subblock)
        factor = np.float32(d * scale)
        offset = np.float32(dmin * minimum)
        group = subblock // 2
        high = bool(subblock & 1)
        packed = qs[group * 32 : group * 32 + 32]
        codes = packed >> 4 if high else packed & 0x0F
        np.subtract(
            np.multiply(codes.astype(np.float32), factor),
            offset,
            out=result[subblock * 32 : (subblock + 1) * 32],
        )
    return result


def decode_q4_k(block: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Compatibility alias for :func:`decode_q4_k_block`."""

    return decode_q4_k_block(block, out=out)


def dequantize_q4_k(packed: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Decode one or more packed blocks, preserving the leading shape."""

    raw = np.asarray(packed)
    if raw.dtype != np.dtype(np.uint8) or raw.ndim < 1 or raw.shape[-1] % Q4K_BLOCK_BYTES:
        raise ValueError(
            f"packed Q4_K rows must end in a multiple of {Q4K_BLOCK_BYTES} uint8 bytes, "
            f"got {raw.shape}"
        )
    blocks = raw.reshape(-1, Q4K_BLOCK_BYTES)
    elements = blocks.shape[0] * Q4K_BLOCK_ELEMENTS
    if out is None:
        result = np.empty(elements, dtype=np.float32)
    else:
        if not isinstance(out, np.ndarray) or out.dtype != np.dtype(np.float32):
            raise TypeError("Q4_K output must be a float32 NumPy ndarray")
        if out.size != elements or not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError("Q4_K output has the wrong size, layout or writeability")
        result = out.reshape(-1)
    for index, block in enumerate(blocks):
        decode_q4_k_block(
            block, out=result[index * Q4K_BLOCK_ELEMENTS : (index + 1) * Q4K_BLOCK_ELEMENTS]
        )
    return result.reshape(*raw.shape[:-1], raw.shape[-1] // Q4K_BLOCK_BYTES * Q4K_BLOCK_ELEMENTS)


def q4_k_dot_scalar(block: Any, x: Any) -> float:
    """Reference FP32 dot for one block, with no Torch/CUDA dependency."""

    values = decode_q4_k_block(block)
    vector = np.asarray(x, dtype=np.float32)
    if vector.ndim != 1 or vector.size != Q4K_BLOCK_ELEMENTS:
        raise ValueError(f"Q4_K dot input must have {Q4K_BLOCK_ELEMENTS} values")
    acc = np.float32(0.0)
    for index in range(Q4K_BLOCK_ELEMENTS):
        acc = np.float32(acc + np.float32(values[index] * vector[index]))
    return float(acc)


def _q4_k_dot_scalar_noalloc(block: np.ndarray, vector: np.ndarray) -> np.float32:
    """Scalar packed dot used by prepared GEMV (no decoded temporary)."""
    raw = _uint8_block(block)
    d = _half_scalar(raw, 0)
    dmin = _half_scalar(raw, 2)
    scales = raw[4:16]
    qs = raw[Q4K_DATA_OFFSET:]
    accumulator = np.float32(0.0)
    for subblock in range(8):
        scale, minimum = _scale_min(scales, subblock)
        factor = np.float32(d * scale)
        offset = np.float32(dmin * minimum)
        group = subblock // 2
        high = bool(subblock & 1)
        for lane in range(32):
            packed = qs[group * 32 + lane]
            code = packed >> 4 if high else packed & 0x0F
            accumulator = np.float32(
                accumulator
                + np.float32((np.float32(code) * factor - offset) * vector[subblock * 32 + lane])
            )
    return accumulator


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


class _NativeQ4K:
    def __init__(self, library: ctypes.CDLL) -> None:
        self.library = library
        self.library.freetoken_q4k_dot_avx2.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.library.freetoken_q4k_dot_avx2.restype = ctypes.c_float
        self.library.freetoken_q4k_decode_avx2.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.library.freetoken_q4k_decode_avx2.restype = None
        self.library.freetoken_q4k_gemv_avx2.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.library.freetoken_q4k_gemv_avx2.restype = None

    def dot(self, block: np.ndarray, x: np.ndarray) -> float:
        return float(
            self.library.freetoken_q4k_dot_avx2(
                block.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        )

    def decode(self, block: np.ndarray, out: np.ndarray) -> None:
        self.library.freetoken_q4k_decode_avx2(
            block.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def gemv(
        self,
        rows: np.ndarray,
        row_count: int,
        blocks_per_row: int,
        row_stride_bytes: int,
        vector: np.ndarray,
        out: np.ndarray,
    ) -> None:
        self.library.freetoken_q4k_gemv_avx2(
            rows.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            row_count,
            blocks_per_row,
            row_stride_bytes,
            vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )


def _load_native() -> _NativeQ4K | None:
    candidates: list[str] = []
    configured = os.environ.get("FREETOKEN_Q4K_NATIVE_LIB")
    if configured:
        candidates.append(configured)
    try:
        spec = importlib.util.find_spec("freetoken.moe._q4_k_native")
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        candidates.append(spec.origin)
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
            available = library.freetoken_q4k_cpu_supports_avx2
            available.argtypes = []
            available.restype = ctypes.c_int
            if available() and _host_supports_avx2():
                return _NativeQ4K(library)
        except (AttributeError, OSError):
            continue
    return None


@dataclass(frozen=True)
class Q4KPrimitive:
    """Runtime-selected Q4_K primitive and its observable fallback decision."""

    requested_mode: str
    isa: Literal["scalar", "avx2"]
    fallback_reason: str | None
    native: _NativeQ4K | None

    @property
    def backend(self) -> str:
        return f"q4_k_{self.isa}"

    def decode(self, block: Any, *, out: np.ndarray | None = None) -> np.ndarray:
        raw = _uint8_block(block)
        if self.native is None:
            return decode_q4_k_block(raw, out=out)
        if out is None:
            result = np.empty(Q4K_BLOCK_ELEMENTS, dtype=np.float32)
        else:
            if not isinstance(out, np.ndarray) or out.dtype != np.dtype(np.float32):
                raise TypeError("Q4_K output must be a float32 NumPy ndarray")
            if (
                out.shape != (Q4K_BLOCK_ELEMENTS,)
                or not out.flags.c_contiguous
                or not out.flags.writeable
            ):
                raise ValueError("Q4_K output must be a contiguous writable 256-element vector")
            result = out
        self.native.decode(raw, result)
        return result

    def dot(self, block: Any, x: Any) -> float:
        raw = _uint8_block(block)
        vector = np.asarray(x, dtype=np.float32)
        if vector.shape != (Q4K_BLOCK_ELEMENTS,) or not vector.flags.c_contiguous:
            raise ValueError("Q4_K dot input must be a contiguous 256-element vector")
        if self.native is None:
            return q4_k_dot_scalar(raw, vector)
        return self.native.dot(raw, vector)

    def gemv(
        self,
        rows: Any,
        input_dim: int,
        x: Any,
        *,
        out: np.ndarray,
        scratch: np.ndarray | None = None,
    ) -> np.ndarray:
        packed = np.asarray(rows)
        vector = np.asarray(x, dtype=np.float32)
        if input_dim <= 0 or input_dim % Q4K_BLOCK_ELEMENTS:
            raise ValueError(
                f"Q4_K GEMV input dimension must be a positive multiple of "
                f"{Q4K_BLOCK_ELEMENTS}, got {input_dim}"
            )
        if (
            packed.dtype != np.dtype(np.uint8)
            or packed.ndim != 2
            or packed.shape[1] != input_dim // Q4K_BLOCK_ELEMENTS * Q4K_BLOCK_BYTES
            or not packed.flags.c_contiguous
        ):
            raise ValueError("Q4_K GEMV rows must be contiguous uint8 rows with valid geometry")
        if vector.shape != (input_dim,) or not vector.flags.c_contiguous:
            raise ValueError("Q4_K GEMV input must be a contiguous float32 vector")
        if (
            out.shape != (packed.shape[0],)
            or out.dtype != np.dtype(np.float32)
            or not out.flags.c_contiguous
            or not out.flags.writeable
        ):
            raise ValueError("Q4_K GEMV output must be a contiguous writable float32 row vector")
        blocks_per_row = input_dim // Q4K_BLOCK_ELEMENTS
        if self.native is not None:
            self.native.gemv(packed, packed.shape[0], blocks_per_row, packed.shape[1], vector, out)
            return out
        for row in range(packed.shape[0]):
            total = np.float32(0.0)
            for block in range(blocks_per_row):
                total = np.float32(
                    total
                    + _q4_k_dot_scalar_noalloc(
                        packed[row, block * Q4K_BLOCK_BYTES : (block + 1) * Q4K_BLOCK_BYTES],
                        vector[block * Q4K_BLOCK_ELEMENTS : (block + 1) * Q4K_BLOCK_ELEMENTS],
                    )
                )
            out[row] = total
        return out


def select_q4_k_primitive(mode: Q4K_MODE = "auto") -> Q4KPrimitive:
    """Select scalar or AVX2 Q4_K with explicit forced-mode fallback metadata."""

    normalized = str(mode).lower()
    if normalized in {"forced_scalar", "scalar"}:
        return Q4KPrimitive(normalized, "scalar", None, None)
    if normalized not in {"auto", "avx2", "forced_avx2"}:
        raise ValueError(f"Q4_K mode must be auto, scalar or avx2, got {mode!r}")
    if not _host_supports_avx2():
        return Q4KPrimitive(normalized, "scalar", "avx2_unavailable", None)
    native = _load_native()
    if native is None:
        return Q4KPrimitive(normalized, "scalar", "native_avx2_unavailable", None)
    return Q4KPrimitive(normalized, "avx2", None, native)


def q4_k_dot(block: Any, x: Any, *, mode: Q4K_MODE = "auto") -> float:
    """Dot one Q4_K block using the selected runtime primitive."""

    return select_q4_k_primitive(mode).dot(block, x)


def _packed_expert_for_gemv(descriptor: CpuExpertDescriptor, expert: int) -> np.ndarray:
    source = descriptor.source
    packed_getter = getattr(source, "expert_packed", None)
    if not callable(packed_getter):
        raise ExecutionFailed(
            f"{descriptor.quant_name} source does not expose expert_packed for direct GEMV"
        )
    packed = np.asarray(packed_getter(expert))
    expected = (descriptor.rows_per_expert, descriptor.row_stride_bytes)
    if packed.shape != expected:
        raise UnsupportedShape(
            f"packed {descriptor.projection} expert shape {packed.shape}, expected {expected}"
        )
    if packed.dtype != np.dtype(np.uint8) or not packed.flags.c_contiguous:
        raise InvalidRequest(
            f"packed {descriptor.projection} expert must be a contiguous uint8 byte view"
        )
    if descriptor.source_address is not None:
        actual_address = int(packed.__array_interface__["data"][0])
        expected_address = descriptor.source_address + expert * descriptor.expert_stride_bytes
        if actual_address != expected_address:
            raise InvalidRequest(
                f"packed {descriptor.projection} expert address {actual_address} "
                f"does not match expected {expected_address}"
            )
    return packed


class _MixedReferenceExecutor(ReferenceCpuExpertExecutor):
    """Issue #15 executor with direct Q4_K and companion-format GEMV seams."""

    def __init__(
        self,
        *args: Any,
        primitive: Q4KPrimitive,
        mixed_primitive: MixedGemvPrimitive,
        q4_descriptors: frozenset[tuple[int, str]],
        mixed_descriptors: frozenset[tuple[int, str]],
        **kwargs: Any,
    ):
        self._q4_primitive = primitive
        self._mixed_primitive = mixed_primitive
        self._q4_descriptors = q4_descriptors
        self._mixed_descriptors = mixed_descriptors
        super().__init__(*args, **kwargs)

    def _matvec(
        self,
        descriptor: CpuExpertDescriptor,
        expert: int,
        vector: np.ndarray,
        out: np.ndarray,
        *,
        decoder_workspace: np.ndarray | None = None,
    ) -> np.ndarray:
        key = (descriptor.layer_id, descriptor.projection)
        if key in self._q4_descriptors:
            packed = _packed_expert_for_gemv(descriptor, expert)
            input_scratch = self._workspace["input"][: descriptor.input_dim]
            np.copyto(input_scratch, vector, casting="unsafe")
            return self._q4_primitive.gemv(
                packed,
                descriptor.input_dim,
                input_scratch,
                out=out,
            )
        if key in self._mixed_descriptors:
            packed = _packed_expert_for_gemv(descriptor, expert)
            input_scratch = self._workspace["input"][: descriptor.input_dim]
            np.copyto(input_scratch, vector, casting="unsafe")
            quant_name = _mixed_format(descriptor)
            assert quant_name is not None
            return self._mixed_primitive.gemv(
                packed,
                descriptor.input_dim,
                input_scratch,
                quant_name=quant_name,
                out=out,
            )
        if _has_q4_k_marker(descriptor):
            if not _is_q4_k_descriptor(descriptor):
                raise UnsupportedQuantType(
                    f"{descriptor.projection} Q4_K type/name contract is inconsistent"
                )
            if not _has_q4_k_geometry(descriptor):
                source = descriptor.source
                if not isinstance(source, np.ndarray) and not hasattr(source, "expert_dense"):
                    raise UnsupportedShape(
                        f"packed {descriptor.projection} Q4_K input dimension "
                        f"{descriptor.input_dim} is not a multiple of {Q4K_BLOCK_ELEMENTS}"
                    )
        if _mixed_format(descriptor) is not None and not _has_mixed_geometry(descriptor):
            if _packed_source(descriptor):
                raise UnsupportedShape(
                    f"packed {descriptor.projection} {descriptor.quant_name} input dimension "
                    f"{descriptor.input_dim} has incompatible block geometry"
                )
        return super()._matvec(
            descriptor,
            expert,
            vector,
            out,
            decoder_workspace=decoder_workspace,
        )


class Q4KExecutor:
    """Issue #15 ABI adapter using direct packed Q4_K/mixed row GEMV."""

    def __init__(
        self,
        layout: CpuExpertLayout,
        *,
        mode: Q4K_MODE = "auto",
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        output_dtype: np.dtype | type = np.float32,
        reference_decoders: Mapping[int | str, Any] | None = None,
        required_alignment: int = 32,
        num_threads: int = 1,
        thread_pool: Executor | None = None,
    ) -> None:
        self.layout = layout
        self.primitive = select_q4_k_primitive(mode)
        self.mixed_primitive = select_mixed_gemv_primitive(mode)
        self.requested_mode = str(mode).lower()
        num_threads = _checked_int(num_threads, "num_threads")
        if num_threads <= 0:
            raise InvalidRequest(f"num_threads must be positive, got {num_threads}")
        self.num_threads = num_threads
        q4_descriptors = frozenset(
            (descriptor.layer_id, descriptor.projection)
            for descriptor in layout.descriptors
            if _is_q4_k_descriptor(descriptor)
            and _has_q4_k_geometry(descriptor)
            and _packed_source(descriptor)
        )
        self._q4_descriptors = q4_descriptors
        primitive_fallback_reason = self.primitive.fallback_reason
        mixed_fallback_reason = self.mixed_primitive.fallback_reason
        capability_fallback_reason: str | None = None
        alignment = int(required_alignment)
        if alignment <= 0:
            raise ValueError("required_alignment must be positive")
        mixed_descriptors = frozenset(
            (descriptor.layer_id, descriptor.projection)
            for descriptor in layout.descriptors
            if _has_mixed_geometry(descriptor)
            and _packed_source(descriptor)
            and self.mixed_primitive.isa == "avx2"
        )
        for descriptor in layout.descriptors:
            if _has_q4_k_marker(descriptor) and not _is_q4_k_descriptor(descriptor):
                capability_fallback_reason = capability_fallback_reason or "unsupported_quant_type"
        for descriptor in layout.descriptors:
            if self.requested_mode in {"scalar", "forced_scalar"}:
                continue
            if _has_q4_k_marker(descriptor) and not _is_q4_k_descriptor(descriptor):
                capability_fallback_reason = capability_fallback_reason or "unsupported_quant_type"
                continue
            if _is_q4_k_descriptor(descriptor) and _has_q4_k_geometry(descriptor):
                if (
                    descriptor.source_address is None
                    or descriptor.source_address % alignment
                    or descriptor.row_stride_bytes % 16
                ):
                    self.primitive = select_q4_k_primitive("scalar")
                    capability_fallback_reason = (
                        capability_fallback_reason or "unsupported_alignment"
                    )
            elif _mixed_format(descriptor) is not None and _has_mixed_geometry(descriptor):
                if descriptor.source_address is None or descriptor.source_address % alignment:
                    mixed_descriptors = frozenset(
                        key
                        for key in mixed_descriptors
                        if key != (descriptor.layer_id, descriptor.projection)
                    )
                    capability_fallback_reason = (
                        capability_fallback_reason or "unsupported_alignment"
                    )

        if not q4_descriptors and not mixed_descriptors:
            for descriptor in layout.descriptors:
                if not _is_q4_k_descriptor(descriptor):
                    capability_fallback_reason = (
                        capability_fallback_reason or "unsupported_quant_type"
                    )
                    break

        # The scalar mixed decoder is deliberately retained as a reference
        # fallback.  Q4_K scalar GEMV predates this adapter and remains direct.
        self._mixed_descriptors = mixed_descriptors
        self._direct_descriptors = q4_descriptors | mixed_descriptors
        self._q4_compatible = bool(q4_descriptors) and all(
            (item.layer_id, item.projection) in q4_descriptors for item in layout.descriptors
        )
        has_q4_formats = any(_has_q4_k_marker(item) for item in layout.descriptors)
        has_mixed_formats = any(_mixed_format(item) is not None for item in layout.descriptors)
        primitive_fallback = primitive_fallback_reason if has_q4_formats else None
        mixed_fallback = mixed_fallback_reason if has_mixed_formats else None
        self._fallback_reason = capability_fallback_reason or primitive_fallback or mixed_fallback
        reference_descriptors = tuple(
            descriptor
            for descriptor in layout.descriptors
            if (descriptor.layer_id, descriptor.projection) not in self._direct_descriptors
        )
        if (
            self._direct_descriptors
            and len(self._direct_descriptors) != len(layout.descriptors)
            and reference_descriptors
            and all(
                not _has_q4_k_marker(descriptor)
                and (
                    descriptor.quant_type in BUILTIN_REFERENCE_DECODERS
                    or descriptor.quant_name in BUILTIN_REFERENCE_DECODERS
                )
                for descriptor in reference_descriptors
            )
        ):
            self._fallback_reason = "mixed_reference_formats"
        decoders = dict(reference_decoders or {})
        for quant_key, decoder in BUILTIN_REFERENCE_DECODERS.items():
            decoders.setdefault(quant_key, decoder)
        for descriptor in layout.descriptors:
            if _is_q4_k_descriptor(descriptor):
                decoders[descriptor.quant_type] = self._decode_rows
                decoders[descriptor.quant_name] = self._decode_rows
        self._reference = _MixedReferenceExecutor(
            layout,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            output_dtype=output_dtype,
            decoders=decoders,
            required_alignment=1,
            primitive=self.primitive,
            mixed_primitive=self.mixed_primitive,
            q4_descriptors=q4_descriptors,
            mixed_descriptors=mixed_descriptors,
        )
        self.hidden_size = self._reference.hidden_size
        self.intermediate_size = self._reference.intermediate_size
        self._threaded_runner: _ThreadedMixedRunner | None = None
        if num_threads > 1 and any(self._layer_direct_eligible(layer) for layer in layout.layers):
            self._threaded_runner = _ThreadedMixedRunner(
                self,
                thread_pool=thread_pool,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                output_dtype=output_dtype,
                required_alignment=required_alignment,
            )

    @property
    def backend(self) -> str:
        return self._backend_for(None)

    @property
    def parallel_enabled(self) -> bool:
        """Whether this executor has an opt-in threaded runner."""
        return self._threaded_runner is not None

    def _layer_direct_eligible(self, layer_id: int) -> bool:
        """Require all three projections of one census geometry to be direct."""
        try:
            descriptors = tuple(
                self.layout.descriptor(layer_id, projection)
                for projection in ("gate", "up", "down")
            )
        except InvalidRequest:
            return False
        keys = {(item.layer_id, item.projection) for item in descriptors}
        if not keys <= self._direct_descriptors:
            return False
        if any(key in self._q4_descriptors for key in keys) and self.primitive.isa != "avx2":
            return False
        if (
            any(key in self._mixed_descriptors for key in keys)
            and self.mixed_primitive.isa != "avx2"
        ):
            return False
        gate, up, down = descriptors
        if gate.num_experts != up.num_experts or gate.num_experts != down.num_experts:
            return False
        if (gate.input_dim, gate.output_dim) != (up.input_dim, up.output_dim):
            return False
        if (down.output_dim, down.input_dim) != (gate.input_dim, gate.output_dim):
            return False
        # This is the actual Q4 census family, without baking layer IDs into
        # the ABI: Q4_K or promoted Q5_K gate/up and Q5_1/Q8_0 down.
        gate_name = _mixed_format(gate) or ("Q4_K" if _is_q4_k_descriptor(gate) else None)
        up_name = _mixed_format(up) or ("Q4_K" if _is_q4_k_descriptor(up) else None)
        down_name = _mixed_format(down) or ("Q4_K" if _is_q4_k_descriptor(down) else None)
        return (
            gate_name in {"Q4_K", "Q5_K"}
            and up_name == gate_name
            and down_name in {"Q4_K", "Q5_1", "Q8_0"}
        )

    def _backend_for(self, layer_id: int | None) -> str:
        descriptors = self.layout.descriptors
        if layer_id is not None:
            descriptors = tuple(item for item in descriptors if item.layer_id == layer_id)
        direct = {
            (descriptor.layer_id, descriptor.projection)
            for descriptor in descriptors
            if (descriptor.layer_id, descriptor.projection) in self._direct_descriptors
        }
        if not direct:
            return "reference"
        if len(direct) == len(descriptors):
            if all(key in self._q4_descriptors for key in direct):
                return self.primitive.backend
            if all(key in self._mixed_descriptors for key in direct):
                return self.mixed_primitive.backend
            return "mixed_avx2"
        return "mixed"

    def _kernel_census(self, layer_id: int | None) -> tuple[str, ...]:
        """Return the selected direct/reference kernels for the executed layer."""
        descriptors = self.layout.descriptors
        if layer_id is not None:
            descriptors = tuple(item for item in descriptors if item.layer_id == layer_id)
        selected: set[str] = set()
        for descriptor in descriptors:
            key = (descriptor.layer_id, descriptor.projection)
            if key in self._q4_descriptors:
                selected.add(self.primitive.backend)
                continue
            if key in self._mixed_descriptors:
                quant_name = _mixed_format(descriptor)
                assert quant_name is not None
                selected.add(self.mixed_primitive.backend_for(quant_name))
                continue
            source = descriptor.source
            if isinstance(source, np.ndarray) or hasattr(source, "expert_dense"):
                selected.add("reference")
            elif (
                descriptor.quant_type in BUILTIN_REFERENCE_DECODERS
                or descriptor.quant_name in BUILTIN_REFERENCE_DECODERS
            ):
                selected.add(f"reference_{descriptor.quant_name.lower()}")
            else:
                selected.add("reference")
        return tuple(sorted(selected or {"reference"}))

    def _fallback_for(self, telemetry: CpuExecutionTelemetry) -> str | None:
        backend = self._backend_for(telemetry.layer_id)
        if backend == "mixed":
            return self._fallback_reason
        if backend == "reference" and not self._direct_descriptors:
            return self._fallback_reason or telemetry.fallback_reason
        return telemetry.fallback_reason if backend == "reference" else self._fallback_reason

    @property
    def last_telemetry(self) -> CpuExecutionTelemetry | None:
        if self._threaded_runner is not None and self._threaded_runner.last_telemetry is not None:
            return self._threaded_runner.last_telemetry
        telemetry = self._reference.last_telemetry
        return None if telemetry is None else self._decorate(telemetry)

    def prepare(self, max_tokens: int, max_routes: int):
        if self._threaded_runner is not None:
            return self._threaded_runner.prepare(max_tokens, max_routes)
        return self._reference.prepare(max_tokens, max_routes)

    def _decode_rows(
        self, packed: np.ndarray, descriptor: CpuExpertDescriptor, *, out: np.ndarray
    ) -> np.ndarray:
        if not _has_q4_k_geometry(descriptor):
            raise UnsupportedShape(
                f"packed {descriptor.projection} Q4_K input dimension "
                f"{descriptor.input_dim} is not a multiple of {Q4K_BLOCK_ELEMENTS}"
            )
        expected = (descriptor.output_dim, descriptor.row_stride_bytes)
        if packed.shape != expected:
            raise UnsupportedShape(
                f"packed {descriptor.projection} expert shape {packed.shape}, expected {expected}"
            )
        blocks_per_row = descriptor.input_dim // Q4K_BLOCK_ELEMENTS
        for row in range(descriptor.output_dim):
            row_bytes = packed[row]
            for block in range(blocks_per_row):
                begin = block * Q4K_BLOCK_BYTES
                values = out[
                    row,
                    block * Q4K_BLOCK_ELEMENTS : (block + 1) * Q4K_BLOCK_ELEMENTS,
                ]
                self.primitive.decode(row_bytes[begin : begin + Q4K_BLOCK_BYTES], out=values)
        return out.reshape(descriptor.output_dim, descriptor.input_dim)

    def _decorate(self, telemetry: CpuExecutionTelemetry) -> CpuExecutionTelemetry:
        return replace(
            telemetry,
            backend=self._backend_for(telemetry.layer_id),
            kernel_census=self._kernel_census(telemetry.layer_id),
            fallback_reason=self._fallback_for(telemetry),
        )

    def execute(self, *args: Any, **kwargs: Any) -> CpuExecutionResult:
        if self._threaded_runner is not None:
            return self._threaded_runner.execute(*args, **kwargs)
        try:
            result = self._reference.execute(*args, **kwargs)
        except CpuAbiError as error:
            if error.telemetry is not None:
                error.telemetry = self._decorate(error.telemetry)
            raise
        return CpuExecutionResult(result.output, self._decorate(result.telemetry))

    def execute_group(
        self, requests: Iterable[CpuExecutionRequest]
    ) -> tuple[CpuExecutionResult, ...]:
        if self._threaded_runner is not None:
            return self._threaded_runner.execute_group(requests)
        return tuple(
            self.execute(
                request.layer_id,
                request.hidden,
                request.expert_ids,
                request.routing_weights,
                num_token_non_padded=request.num_token_non_padded,
                output=request.output,
                accumulate=request.accumulate,
                cancellation=request.cancellation,
            )
            for request in requests
        )

    def microbenchmark(self, *args: Any, **kwargs: Any):
        if self._threaded_runner is not None:
            return self._threaded_runner.microbenchmark(*args, **kwargs)
        if "thread_counts" in kwargs:
            raise InvalidRequest(
                "thread_count sweep requires an AVX2 executor with threaded support"
            )
        samples = self._reference.microbenchmark(*args, **kwargs)
        return tuple(
            replace(sample, telemetry=tuple(self._decorate(item) for item in sample.telemetry))
            for sample in samples
        )

    def close(self) -> None:
        if self._threaded_runner is not None:
            self._threaded_runner.close()

    def __enter__(self) -> Q4KExecutor:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        self.close()


class _WorkerCancellation:
    """Internal cancellation event; user callbacks never cross into workers."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def cancelled(self) -> bool:
        return self._event.is_set()


class _ThreadedMixedRunner:
    """Prepared, request-isolated route partitioning for native mixed GEMV."""

    def __init__(
        self,
        owner: Q4KExecutor,
        *,
        thread_pool: Executor | None,
        activation: str,
        apply_router_weight_on_input: bool,
        output_dtype: np.dtype | type,
        required_alignment: int,
    ) -> None:
        self.owner = owner
        self.max_threads = owner.num_threads
        self._pool = thread_pool or ThreadPoolExecutor(
            max_workers=self.max_threads,
            thread_name_prefix="freetoken-mixed",
        )
        self._owns_pool = thread_pool is None
        self._lock = threading.Lock()
        self._workers = tuple(
            _MixedReferenceExecutor(
                owner.layout,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                output_dtype=np.float32,
                decoders=owner._reference.decoders,
                required_alignment=1,
                primitive=owner.primitive,
                mixed_primitive=owner.mixed_primitive,
                q4_descriptors=owner._q4_descriptors,
                mixed_descriptors=owner._mixed_descriptors,
            )
            for _ in range(self.max_threads)
        )
        self._plan = None
        self._route_ids: np.ndarray | None = None
        self._route_weights: np.ndarray | None = None
        self._outputs: np.ndarray | None = None
        self._merged: np.ndarray | None = None
        self._thread_workspace_bytes = 0
        self._worker_workspace_bytes = 0
        self._last_telemetry: CpuExecutionTelemetry | None = None
        self._output_dtype = np.dtype(output_dtype)
        self._closed = False

    @property
    def last_telemetry(self) -> CpuExecutionTelemetry | None:
        return self._last_telemetry

    def _decorate(self, telemetry: CpuExecutionTelemetry) -> CpuExecutionTelemetry:
        return self.owner._decorate(telemetry)

    @property
    def _workspace_bytes(self) -> int:
        plan_bytes = self._plan.workspace_bytes if self._plan is not None else 0
        return plan_bytes + self._worker_workspace_bytes + self._thread_workspace_bytes

    def _busy(self, started: int) -> Busy:
        error = Busy("executor already has an active request")
        telemetry = self._decorate(
            CpuExecutionTelemetry(
                backend="reference",
                layer_id=None,
                tokens_requested=0,
                tokens_non_padded=0,
                routes_requested=0,
                routes_executed=0,
                unique_experts=0,
                bytes_read_packed=0,
                elapsed_ns=time.perf_counter_ns() - started,
                workspace_bytes=self._workspace_bytes,
                fallback_reason="busy",
                error="Busy",
                error_detail=str(error),
                thread_count=0,
            )
        )
        error.telemetry = telemetry
        self._last_telemetry = telemetry
        return error

    def prepare(self, max_tokens: int, max_routes: int):
        started = time.perf_counter_ns()
        if self._closed:
            raise InvalidRequest("Q4_K executor is closed")
        if not self._lock.acquire(blocking=False):
            raise self._busy(started)
        try:
            plan = self.owner._reference.prepare(max_tokens, max_routes)
            for worker in self._workers:
                worker.prepare(max_tokens, max_routes)
            self._route_ids = np.full(
                (self.max_threads, plan.max_tokens, plan.max_routes), -1, dtype=np.int32
            )
            self._route_weights = np.zeros(
                (self.max_threads, plan.max_tokens, plan.max_routes), dtype=np.float32
            )
            self._outputs = np.empty(
                (self.max_threads, plan.max_tokens, self.owner.hidden_size), dtype=np.float32
            )
            self._merged = np.empty((plan.max_tokens, self.owner.hidden_size), dtype=np.float32)
            self._plan = plan
            self._worker_workspace_bytes = sum(
                worker._plan.workspace_bytes for worker in self._workers if worker._plan is not None
            )
            self._thread_workspace_bytes = (
                self._route_ids.nbytes
                + self._route_weights.nbytes
                + self._outputs.nbytes
                + self._merged.nbytes
            )
            return plan
        finally:
            self._lock.release()

    def _serial(self, *args: Any, **kwargs: Any) -> CpuExecutionResult:
        try:
            result = self.owner._reference.execute(*args, **kwargs)
        except CpuAbiError as error:
            if error.telemetry is not None:
                error.telemetry = self._decorate(replace(error.telemetry, thread_count=1))
                self._last_telemetry = error.telemetry
            raise
        telemetry = self._decorate(replace(result.telemetry, thread_count=1))
        self._last_telemetry = telemetry
        return CpuExecutionResult(result.output, telemetry)

    def _error_telemetry(
        self,
        *,
        layer_id: int | None,
        tokens: int,
        active_tokens: int,
        routes_requested: int,
        unique_experts: set[int],
        started: int,
        error: CpuAbiError,
        thread_count: int,
        observations: Iterable[CpuExecutionTelemetry] = (),
    ) -> CpuExecutionTelemetry:
        observed = tuple(observations)
        telemetry = CpuExecutionTelemetry(
            backend="reference",
            layer_id=layer_id,
            tokens_requested=tokens,
            tokens_non_padded=active_tokens,
            routes_requested=routes_requested,
            routes_executed=sum(item.routes_executed for item in observed),
            unique_experts=len(unique_experts),
            bytes_read_packed=sum(item.bytes_read_packed for item in observed),
            elapsed_ns=time.perf_counter_ns() - started,
            workspace_bytes=self._workspace_bytes,
            fallback_reason=self.owner._fallback_reason,
            cancelled=isinstance(error, Cancelled),
            error=type(error).__name__,
            error_detail=str(error),
            thread_count=thread_count,
        )
        return self._decorate(telemetry)

    @staticmethod
    def _cancel_and_drain(futures: Iterable[Any]) -> None:
        pending = tuple(futures)
        for future in pending:
            try:
                future.cancel()
            except BaseException:
                pass
        for future in pending:
            try:
                future.result()
            except BaseException:
                pass

    def execute(
        self,
        layer_id: int,
        hidden: np.ndarray,
        expert_ids: np.ndarray,
        routing_weights: np.ndarray,
        *,
        num_token_non_padded: int | None = None,
        output: np.ndarray | None = None,
        accumulate: bool = False,
        cancellation: Any = None,
        _thread_count: int | None = None,
    ) -> CpuExecutionResult:
        started = time.perf_counter_ns()
        result_output = output
        parsed_layer: int | None = None
        tokens = 0
        active_tokens = 0
        routes_requested = 0
        unique: set[int] = set()
        submitted: list[Any] = []
        actual_threads = 1
        if self._closed:
            raise InvalidRequest("Q4_K executor is closed")
        if not self._lock.acquire(blocking=False):
            raise self._busy(started)
        try:
            selected_threads = (
                self.max_threads
                if _thread_count is None
                else _checked_int(_thread_count, "thread_count")
            )
            if not 1 <= selected_threads <= self.max_threads:
                raise InvalidRequest(
                    f"thread_count={selected_threads} outside [1, {self.max_threads}]"
                )
            parsed_layer = _checked_int(layer_id, "layer_id")
            hidden, expert_ids, routing_weights, result_output, tokens = (
                self.owner._reference._validate_arrays(
                    parsed_layer, hidden, expert_ids, routing_weights, output
                )
            )
            if num_token_non_padded is None:
                active_tokens = tokens
            else:
                active_tokens = _checked_int(num_token_non_padded, "num_token_non_padded")
                if not 0 <= active_tokens <= tokens:
                    raise InvalidRequest(
                        f"num_token_non_padded={active_tokens} outside [0, {tokens}]"
                    )
            descriptor = self.owner.layout.descriptor(parsed_layer, "gate")
            routes_requested = active_tokens * expert_ids.shape[1]
            _, unique = self.owner._reference._validate_ids(descriptor, expert_ids, active_tokens)
            if _cancelled(cancellation):
                raise Cancelled("CPU expert execution cancelled before compute")
            if (
                selected_threads == 1
                or expert_ids.shape[1] == 0
                or not self.owner._layer_direct_eligible(parsed_layer)
            ):
                return self._serial(
                    parsed_layer,
                    hidden,
                    expert_ids,
                    routing_weights,
                    num_token_non_padded=active_tokens,
                    output=result_output,
                    accumulate=accumulate,
                    cancellation=cancellation,
                )
            route_ids = self._route_ids
            route_weights = self._route_weights
            outputs = self._outputs
            merged = self._merged
            if route_ids is None or route_weights is None or outputs is None or merged is None:
                raise WorkspaceNotPrepared("call prepare before execute")
            ranges = partition_q4_k_routes(expert_ids.shape[1], selected_threads)
            actual_threads = len(ranges)
            worker_cancel = _WorkerCancellation()
            for index, (begin, end) in enumerate(ranges):
                width = end - begin
                ids_view = route_ids[index, :tokens, :width]
                weights_view = route_weights[index, :tokens, :width]
                np.copyto(ids_view, expert_ids[:, begin:end], casting="unsafe")
                np.copyto(weights_view, routing_weights[:, begin:end], casting="unsafe")
                try:
                    future = self._pool.submit(
                        self._workers[index].execute,
                        parsed_layer,
                        hidden,
                        ids_view,
                        weights_view,
                        num_token_non_padded=active_tokens,
                        output=outputs[index, :tokens],
                        accumulate=False,
                        cancellation=worker_cancel,
                    )
                except BaseException as submit_error:
                    worker_cancel.cancel()
                    self._cancel_and_drain(submitted)
                    wrapped = ExecutionFailed(
                        f"threaded mixed GEMV submission failed: {submit_error}"
                    )
                    wrapped.telemetry = self._error_telemetry(
                        layer_id=parsed_layer,
                        tokens=tokens,
                        active_tokens=active_tokens,
                        routes_requested=routes_requested,
                        unique_experts=unique,
                        started=started,
                        error=wrapped,
                        thread_count=len(submitted) or 1,
                    )
                    self._last_telemetry = wrapped.telemetry
                    _clear_output(result_output)
                    raise wrapped from submit_error
                submitted.append(future)

            pending = set(submitted)
            indexed = {future: index for index, future in enumerate(submitted)}
            results: list[CpuExecutionResult | None] = [None] * len(submitted)
            observations: list[CpuExecutionTelemetry] = []
            first_error: CpuAbiError | None = None
            while pending:
                done, pending = wait(pending, timeout=0.01, return_when=FIRST_EXCEPTION)
                for future in sorted(done, key=lambda item: indexed[item]):
                    try:
                        result = future.result()
                    except CpuAbiError as error:
                        first_error = first_error or error
                        if error.telemetry is not None:
                            observations.append(error.telemetry)
                    except BaseException as worker_error:
                        wrapped = ExecutionFailed(
                            f"threaded mixed GEMV worker failed: {worker_error}"
                        )
                        first_error = first_error or wrapped
                    else:
                        results[indexed[future]] = result
                        observations.append(result.telemetry)
                if first_error is not None:
                    worker_cancel.cancel()
                    self._cancel_and_drain(pending)
                    pending.clear()
                    break
                if _cancelled(cancellation):
                    worker_cancel.cancel()
            if first_error is not None:
                telemetry = self._error_telemetry(
                    layer_id=parsed_layer,
                    tokens=tokens,
                    active_tokens=active_tokens,
                    routes_requested=routes_requested,
                    unique_experts=unique,
                    started=started,
                    error=first_error,
                    thread_count=actual_threads,
                    observations=observations,
                )
                first_error.telemetry = telemetry
                self._last_telemetry = telemetry
                _clear_output(result_output)
                raise first_error
            if _cancelled(cancellation):
                cancelled = Cancelled("CPU expert execution cancelled during compute")
                worker_cancel.cancel()
                self._cancel_and_drain(submitted)
                cancelled.telemetry = self._error_telemetry(
                    layer_id=parsed_layer,
                    tokens=tokens,
                    active_tokens=active_tokens,
                    routes_requested=routes_requested,
                    unique_experts=unique,
                    started=started,
                    error=cancelled,
                    thread_count=actual_threads,
                    observations=observations,
                )
                self._last_telemetry = cancelled.telemetry
                _clear_output(result_output)
                raise cancelled
            if any(result is None for result in results):
                raise ExecutionFailed("threaded mixed GEMV completed without all partitions")
            if result_output is None:
                result_output = np.empty((tokens, self.owner.hidden_size), dtype=self._output_dtype)
            if accumulate and output is not None:
                np.copyto(merged[:tokens], result_output, casting="unsafe")
            else:
                merged[:tokens].fill(0.0)
            # The index order is part of the ABI: completion order is deliberately
            # ignored so floating-point reduction is reproducible.
            for result in results:
                assert result is not None
                np.add(merged[:tokens], result.output, out=merged[:tokens])
            np.copyto(result_output, merged[:tokens], casting="unsafe")
            telemetry = self._decorate(
                CpuExecutionTelemetry(
                    backend="reference",
                    layer_id=parsed_layer,
                    tokens_requested=tokens,
                    tokens_non_padded=active_tokens,
                    routes_requested=routes_requested,
                    routes_executed=sum(item.routes_executed for item in observations),
                    unique_experts=len(unique),
                    bytes_read_packed=sum(item.bytes_read_packed for item in observations),
                    elapsed_ns=time.perf_counter_ns() - started,
                    workspace_bytes=self._workspace_bytes,
                    fallback_reason=self.owner._fallback_reason,
                    thread_count=actual_threads,
                )
            )
            self._last_telemetry = telemetry
            return CpuExecutionResult(result_output, telemetry)
        except CpuAbiError as error:
            if error.telemetry is None:
                error.telemetry = self._error_telemetry(
                    layer_id=parsed_layer,
                    tokens=tokens,
                    active_tokens=active_tokens,
                    routes_requested=routes_requested,
                    unique_experts=unique,
                    started=started,
                    error=error,
                    thread_count=actual_threads,
                    observations=(),
                )
                self._last_telemetry = error.telemetry
            _clear_output(result_output)
            raise
        except BaseException as error:
            wrapped = ExecutionFailed(f"threaded mixed GEMV execution failed: {error}")
            wrapped.telemetry = self._error_telemetry(
                layer_id=parsed_layer,
                tokens=tokens,
                active_tokens=active_tokens,
                routes_requested=routes_requested,
                unique_experts=unique,
                started=started,
                error=wrapped,
                thread_count=actual_threads,
            )
            self._last_telemetry = wrapped.telemetry
            _clear_output(result_output)
            raise wrapped from error
        finally:
            self._lock.release()

    def execute_group(
        self, requests: Iterable[CpuExecutionRequest]
    ) -> tuple[CpuExecutionResult, ...]:
        """Execute grouped requests serially at the public request boundary."""
        return tuple(
            self.execute(
                request.layer_id,
                request.hidden,
                request.expert_ids,
                request.routing_weights,
                num_token_non_padded=request.num_token_non_padded,
                output=request.output,
                accumulate=request.accumulate,
                cancellation=request.cancellation,
            )
            for request in requests
        )

    def microbenchmark(
        self,
        layer_id: int,
        hidden: np.ndarray,
        expert_ids: np.ndarray,
        routing_weights: np.ndarray,
        *,
        repeats: int = 1,
        route_counts: Iterable[int] | None = None,
        miss_counts: Iterable[int] | None = None,
        num_token_non_padded: int | None = None,
        thread_counts: Iterable[int] | None = None,
    ) -> tuple[CpuMicrobenchmarkSample, ...]:
        repeats = _checked_int(repeats, "repeats")
        if repeats <= 0:
            raise InvalidRequest("repeats must be positive")
        layer_id = _checked_int(layer_id, "layer_id")
        hidden, expert_ids, routing_weights, _, tokens = self.owner._reference._validate_arrays(
            layer_id, hidden, expert_ids, routing_weights, None
        )
        active_tokens = (
            tokens
            if num_token_non_padded is None
            else _checked_int(num_token_non_padded, "num_token_non_padded")
        )
        if not 0 <= active_tokens <= tokens:
            raise InvalidRequest(f"num_token_non_padded={active_tokens} outside [0, {tokens}]")
        max_width = min(self.owner.layout.top_k, expert_ids.shape[1])
        if max_width <= 0:
            raise InvalidRequest("microbenchmark requires at least one supplied route")
        selected_widths = (
            tuple(width for width in (1, 2, 4, 8, 10) if width <= max_width)
            if route_counts is None
            else tuple(_checked_int(value, "route_count") for value in route_counts)
        )
        if not selected_widths:
            raise InvalidRequest("route_counts must not be empty")
        if any(not 1 <= width <= max_width for width in selected_widths):
            raise InvalidRequest(f"route_count outside [1, {max_width}]")
        expected_misses = (
            (None,) * len(selected_widths)
            if miss_counts is None
            else tuple(_checked_int(value, "miss_count") for value in miss_counts)
        )
        if len(expected_misses) != len(selected_widths):
            raise InvalidRequest("route_counts and miss_counts must have equal lengths")
        selected_threads = (
            tuple(
                dict.fromkeys(
                    count for count in (1, 2, 4, 8, self.max_threads) if count <= self.max_threads
                )
            )
            if thread_counts is None
            else tuple(_checked_int(value, "thread_count") for value in thread_counts)
        )
        if not selected_threads:
            raise InvalidRequest("thread_counts must not be empty")
        if any(not 1 <= count <= self.max_threads for count in selected_threads):
            raise InvalidRequest(f"thread_count outside [1, {self.max_threads}]")
        projections = tuple(
            CpuMicrobenchmarkProjection(
                projection=projection,
                quant_name=self.owner.layout.descriptor(layer_id, projection).quant_name,
                quant_type=self.owner.layout.descriptor(layer_id, projection).quant_type,
                row_stride_bytes=self.owner.layout.descriptor(
                    layer_id, projection
                ).row_stride_bytes,
                expert_stride_bytes=self.owner.layout.descriptor(
                    layer_id, projection
                ).expert_stride_bytes,
            )
            for projection in ("gate", "up", "down")
        )
        if self._plan is None:
            raise WorkspaceNotPrepared("call prepare before microbenchmark")
        samples: list[CpuMicrobenchmarkSample] = []
        descriptor = self.owner.layout.descriptor(layer_id, "gate")
        for width, expected in zip(selected_widths, expected_misses, strict=True):
            ids = expert_ids[:active_tokens, :width]
            actual_misses, _ = self.owner._reference._validate_ids(descriptor, ids, active_tokens)
            if expected is not None and (expected < 0 or expected != actual_misses):
                raise InvalidRequest(
                    f"miss_count={expected} does not match supplied active IDs ({actual_misses})"
                )
            for requested_threads in selected_threads:
                # A warm-up is required by the benchmark contract but is never
                # retained as a raw observation.
                self.execute(
                    layer_id,
                    hidden,
                    expert_ids[:, :width],
                    routing_weights[:, :width],
                    num_token_non_padded=active_tokens,
                    _thread_count=requested_threads,
                )
                elapsed: list[int] = []
                telemetry: list[CpuExecutionTelemetry] = []
                for _ in range(repeats):
                    result = self.execute(
                        layer_id,
                        hidden,
                        expert_ids[:, :width],
                        routing_weights[:, :width],
                        num_token_non_padded=active_tokens,
                        _thread_count=requested_threads,
                    )
                    elapsed.append(result.telemetry.elapsed_ns)
                    telemetry.append(result.telemetry)
                samples.append(
                    CpuMicrobenchmarkSample(
                        layer_id=layer_id,
                        route_count=width,
                        miss_count=actual_misses,
                        repeats=repeats,
                        elapsed_ns=tuple(elapsed),
                        telemetry=tuple(telemetry),
                        tokens_requested=tokens,
                        tokens_non_padded=active_tokens,
                        hidden_size=self.owner.hidden_size,
                        intermediate_size=self.owner.intermediate_size,
                        workspace_bytes=self._workspace_bytes,
                        projections=projections,
                        thread_count=requested_threads,
                    )
                )
        return tuple(samples)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_pool:
            self._pool.shutdown(wait=True)


Q4KCpuExpertExecutor = Q4KExecutor
Q4KExpertExecutor = Q4KExecutor


__all__ = [
    "Q4K_BLOCK_BYTES",
    "Q4K_BLOCK_ELEMENTS",
    "Q4KCpuExpertExecutor",
    "Q4KExecutor",
    "Q4KExpertExecutor",
    "Q4KPrimitive",
    "decode_q4_k",
    "decode_q4_k_block",
    "dequantize_q4_k",
    "partition_q4_k_routes",
    "q4_k_dot",
    "q4_k_dot_scalar",
    "select_q4_k_primitive",
]
