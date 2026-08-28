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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from freetoken.moe.cpu_abi import (
    CpuAbiError,
    CpuExecutionRequest,
    CpuExecutionResult,
    CpuExecutionTelemetry,
    CpuExpertDescriptor,
    CpuExpertLayout,
    ExecutionFailed,
    InvalidRequest,
    ReferenceCpuExpertExecutor,
    UnsupportedQuantType,
    UnsupportedShape,
)
from freetoken.moe.ggml_reference import BUILTIN_REFERENCE_DECODERS

Q4K_BLOCK_ELEMENTS = 256
Q4K_BLOCK_BYTES = 144
Q4K_SCALE_BYTES = 12
Q4K_DATA_OFFSET = 16
Q4K_MODE = Literal["auto", "scalar", "avx2"]
_Q4K_TYPES = frozenset({12, "Q4_K", "q4_k"})


def _is_q4_k_descriptor(descriptor: CpuExpertDescriptor) -> bool:
    return descriptor.quant_type in _Q4K_TYPES and descriptor.quant_name.upper() == "Q4_K"


def _has_q4_k_geometry(descriptor: CpuExpertDescriptor) -> bool:
    return descriptor.input_dim % Q4K_BLOCK_ELEMENTS == 0 and descriptor.row_stride_bytes == (
        descriptor.input_dim // Q4K_BLOCK_ELEMENTS * Q4K_BLOCK_BYTES
    )


def _has_q4_k_marker(descriptor: CpuExpertDescriptor) -> bool:
    return descriptor.quant_type in _Q4K_TYPES or descriptor.quant_name.upper() == "Q4_K"


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


class _Q4KReferenceExecutor(ReferenceCpuExpertExecutor):
    """Issue #15 executor with a packed-row GEMV matvec seam."""

    def __init__(
        self,
        *args: Any,
        primitive: Q4KPrimitive,
        q4_descriptors: frozenset[tuple[int, str]],
        **kwargs: Any,
    ):
        self._q4_primitive = primitive
        self._q4_descriptors = q4_descriptors
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
        return super()._matvec(
            descriptor,
            expert,
            vector,
            out,
            decoder_workspace=decoder_workspace,
        )


class Q4KExecutor:
    """Issue #15 ABI adapter using direct packed Q4_K row GEMV."""

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
    ) -> None:
        self.layout = layout
        self.primitive = select_q4_k_primitive(mode)
        self.requested_mode = str(mode).lower()
        q4_descriptors = frozenset(
            (descriptor.layer_id, descriptor.projection)
            for descriptor in layout.descriptors
            if _is_q4_k_descriptor(descriptor) and _has_q4_k_geometry(descriptor)
        )
        self._q4_descriptors = q4_descriptors
        primitive_fallback_reason = self.primitive.fallback_reason
        capability_fallback_reason: str | None = None
        q4_compatible = True
        alignment = int(required_alignment)
        if alignment <= 0:
            raise ValueError("required_alignment must be positive")
        for descriptor in layout.descriptors:
            if not _is_q4_k_descriptor(descriptor):
                q4_compatible = False
                capability_fallback_reason = "unsupported_quant_type"
                continue
            if not _has_q4_k_geometry(descriptor):
                q4_compatible = False
                capability_fallback_reason = capability_fallback_reason or "unsupported_shape"
                continue
            if self.requested_mode not in {"scalar", "forced_scalar"} and (
                descriptor.source_address is None
                or descriptor.source_address % alignment
                or descriptor.row_stride_bytes % 16
            ):
                # The scalar primitive is safe for any byte address; keep the
                # packed GEMV path and make the downgrade observable.
                self.primitive = select_q4_k_primitive("scalar")
                capability_fallback_reason = capability_fallback_reason or "unsupported_alignment"

        self._q4_compatible = q4_compatible
        self._fallback_reason = capability_fallback_reason or primitive_fallback_reason
        reference_descriptors = tuple(
            descriptor
            for descriptor in layout.descriptors
            if (descriptor.layer_id, descriptor.projection) not in q4_descriptors
        )
        if (
            q4_descriptors
            and not q4_compatible
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
        self._reference = _Q4KReferenceExecutor(
            layout,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            output_dtype=output_dtype,
            decoders=decoders,
            required_alignment=1,
            primitive=self.primitive,
            q4_descriptors=q4_descriptors,
        )
        self.hidden_size = self._reference.hidden_size
        self.intermediate_size = self._reference.intermediate_size

    @property
    def backend(self) -> str:
        return self._backend_for(None)

    def _backend_for(self, layer_id: int | None) -> str:
        descriptors = self.layout.descriptors
        if layer_id is not None:
            descriptors = tuple(item for item in descriptors if item.layer_id == layer_id)
        direct = {
            (descriptor.layer_id, descriptor.projection)
            for descriptor in descriptors
            if (descriptor.layer_id, descriptor.projection) in self._q4_descriptors
        }
        if not direct:
            return "reference"
        if len(direct) == len(descriptors):
            return self.primitive.backend
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
        if backend == "reference" and not self._q4_descriptors:
            return self._fallback_reason or telemetry.fallback_reason
        return telemetry.fallback_reason if backend == "reference" else self._fallback_reason

    @property
    def last_telemetry(self) -> CpuExecutionTelemetry | None:
        telemetry = self._reference.last_telemetry
        return None if telemetry is None else self._decorate(telemetry)

    def prepare(self, max_tokens: int, max_routes: int):
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
        samples = self._reference.microbenchmark(*args, **kwargs)
        return tuple(
            replace(sample, telemetry=tuple(self._decorate(item) for item in sample.telemetry))
            for sample in samples
        )


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
    "q4_k_dot",
    "q4_k_dot_scalar",
    "select_q4_k_primitive",
]
