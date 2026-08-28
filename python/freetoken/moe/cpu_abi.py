"""Torch-free contract for host-side routed expert execution.

This module deliberately stops at the executor boundary.  It owns neither a
quantizer nor a thread pool: production decoders and scheduling policies can be
plugged in without importing a model implementation.  The reference executor
is intentionally small and is used as a correctness oracle for later AVX2
backends.
"""

from __future__ import annotations

import inspect
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class CpuAbiError(RuntimeError):
    """Base class for fail-closed ABI and execution errors."""

    def __init__(self, message: str, *, telemetry: CpuExecutionTelemetry | None = None) -> None:
        super().__init__(message)
        self.telemetry = telemetry


class InvalidRequest(CpuAbiError):
    pass


class InvalidExpertId(CpuAbiError):
    pass


class UnsupportedQuantType(CpuAbiError):
    pass


class UnsupportedShape(CpuAbiError):
    pass


class UnsupportedAlignment(CpuAbiError):
    pass


class WorkspaceTooSmall(CpuAbiError):
    pass


class WorkspaceNotPrepared(CpuAbiError):
    pass


class Busy(CpuAbiError):
    pass


class Cancelled(CpuAbiError):
    pass


class ExecutionFailed(CpuAbiError):
    pass


class ExpertSource(Protocol):
    """Minimum source protocol accepted by a descriptor.

    ``expert_packed`` returns rows with shape ``[output_dim, row_stride_bytes]``.
    A source may instead expose ``expert_dense`` for the reference fixture path.
    """

    def expert_packed(self, expert: int) -> np.ndarray: ...


QuantDecoder = Callable[..., np.ndarray]


class WorkspaceQuantDecoder(Protocol):
    """Optional packed decoder contract for bounded reference execution.

    A decoder implementing the keyword-only ``out`` argument must fill and
    return that supplied ``float32`` matrix.  Legacy two-argument decoders are
    still accepted as a compatibility/reference path and may allocate their
    result; their telemetry is marked accordingly.
    """

    def __call__(
        self,
        packed: np.ndarray,
        descriptor: CpuExpertDescriptor,
        *,
        out: np.ndarray,
    ) -> np.ndarray: ...


def _checked_int(value: Any, name: str) -> int:
    """Accept Python/NumPy integers, but never silently truncate another scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise InvalidRequest(f"{name} must be an integer, got {value!r}")
    return int(value)


class ThreadPoolHook(Protocol):
    """Reserved worker-pool seam; the serial reference executor does not invoke it."""

    def submit(self, callable: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...


class NumaPolicyHook(Protocol):
    """Reserved NUMA-placement seam; production executors may invoke it per bank."""

    def placement(self, layer_id: int, projection: str) -> Any: ...


@dataclass(frozen=True)
class CpuExpertDescriptor:
    """Immutable geometry and source-address contract for one expert bank.

    Mapped sources have their runtime address derived from the mapping.  A
    descriptor with ``source_address=None`` is explicitly offset-only and is
    suitable for metadata/reference adapters only; pointer-based production
    backends must reject it before use.
    """

    layer_id: int
    projection: str
    quant_type: int | str
    quant_name: str
    num_experts: int
    output_dim: int
    input_dim: int
    rows_per_expert: int
    row_stride_bytes: int
    expert_stride_bytes: int
    tensor_bytes: int
    source_offset: int = 0
    source_address: int | None = None
    pool_id: int = -1
    source: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "layer_id",
            "num_experts",
            "output_dim",
            "input_dim",
            "rows_per_expert",
            "row_stride_bytes",
            "expert_stride_bytes",
            "tensor_bytes",
            "source_offset",
            "pool_id",
        ):
            object.__setattr__(self, name, _checked_int(getattr(self, name), name))
        if isinstance(self.quant_type, bool) or not isinstance(
            self.quant_type, (int, str, np.integer)
        ):
            raise InvalidRequest(
                f"quant_type must be an integer or string, got {self.quant_type!r}"
            )
        if isinstance(self.quant_type, np.integer):
            object.__setattr__(self, "quant_type", int(self.quant_type))
        derived_address = _source_address(self.source)
        if self.source_address is not None:
            explicit_address = _checked_int(self.source_address, "source_address")
            if derived_address is not None and explicit_address != derived_address:
                raise InvalidRequest(
                    f"source_address {explicit_address} disagrees with source tensor-start "
                    f"address {derived_address}"
                )
            object.__setattr__(
                self,
                "source_address",
                explicit_address,
            )
        else:
            if derived_address is not None:
                object.__setattr__(
                    self,
                    "source_address",
                    _checked_int(derived_address, "source_address"),
                )
        if self.layer_id < 0:
            raise InvalidRequest(f"layer_id must be non-negative, got {self.layer_id}")
        if not isinstance(self.projection, str) or not self.projection:
            raise InvalidRequest("projection must be a non-empty string")
        if not isinstance(self.quant_name, str) or not self.quant_name:
            raise InvalidRequest("quant_name must be a non-empty string")
        for name in (
            "num_experts",
            "output_dim",
            "input_dim",
            "rows_per_expert",
            "row_stride_bytes",
            "expert_stride_bytes",
            "tensor_bytes",
        ):
            if getattr(self, name) <= 0:
                raise InvalidRequest(f"{name} must be positive")
        if self.rows_per_expert != self.output_dim:
            raise UnsupportedShape(
                f"{self.projection}: rows_per_expert={self.rows_per_expert} "
                f"does not match output_dim={self.output_dim}"
            )
        if self.expert_stride_bytes != self.rows_per_expert * self.row_stride_bytes:
            raise InvalidRequest(
                f"{self.projection}: expert stride {self.expert_stride_bytes} does not "
                f"equal rows {self.rows_per_expert} x row stride {self.row_stride_bytes}"
            )
        if self.tensor_bytes != self.num_experts * self.expert_stride_bytes:
            raise InvalidRequest(
                f"{self.projection}: tensor bytes {self.tensor_bytes} do not equal "
                f"experts {self.num_experts} x stride {self.expert_stride_bytes}"
            )
        if self.source_offset < 0:
            raise InvalidRequest("source_offset must be non-negative")
        if self.source_address is not None and self.source_address < 0:
            raise InvalidRequest("source_address must be non-negative")
        _validate_source_range(self)

    @classmethod
    def from_source_descriptor(cls, descriptor: Any, source: Any) -> CpuExpertDescriptor:
        """Adapt an Issue #13-style descriptor without importing its module."""
        return cls(
            layer_id=_checked_int(descriptor.layer, "source descriptor layer"),
            projection=descriptor.projection,
            quant_type=_checked_int(descriptor.quant_type, "source descriptor quant_type"),
            quant_name=descriptor.quant_name,
            num_experts=_checked_int(descriptor.experts, "source descriptor experts"),
            output_dim=_checked_int(descriptor.output_dim, "source descriptor output_dim"),
            input_dim=_checked_int(descriptor.input_dim, "source descriptor input_dim"),
            rows_per_expert=_checked_int(
                descriptor.output_dim, "source descriptor rows_per_expert"
            ),
            row_stride_bytes=_checked_int(descriptor.row_bytes, "source descriptor row_bytes"),
            expert_stride_bytes=_checked_int(
                descriptor.bytes_per_expert, "source descriptor bytes_per_expert"
            ),
            tensor_bytes=_checked_int(descriptor.tensor_bytes, "source descriptor tensor_bytes"),
            source_offset=_checked_int(descriptor.data_offset, "source descriptor data_offset"),
            pool_id=_checked_int(getattr(descriptor, "pool_id", -1), "source descriptor pool_id"),
            source=source,
        )


def _source_ranges(descriptor: CpuExpertDescriptor) -> tuple[tuple[int, int], ...]:
    """Return every bounded range a source exposes.

    Keeping all independent bounds matters for mapped sources: a descriptor may
    claim a valid tensor span while its actual mapping is shorter.  Validation
    below checks the requested span against every range rather than trusting the
    first piece of metadata encountered.
    """
    source = descriptor.source
    if source is None:
        return ()
    ranges: list[tuple[int, int]] = []
    if isinstance(source, np.ndarray):
        if source.ndim == 3 and source.shape[0] == descriptor.num_experts:
            ranges.append((0, int(source.nbytes)))
        return tuple(ranges)

    for offset_name, size_name in (
        ("range_offset", "range_size"),
        ("mapped_offset", "mapped_size"),
    ):
        offset = getattr(source, offset_name, None)
        size = getattr(source, size_name, None)
        if (offset is None) != (size is None):
            raise InvalidRequest(
                f"source exposes incomplete {offset_name}/{size_name} range metadata"
            )
        if offset is not None and size is not None:
            ranges.append((_checked_int(offset, offset_name), _checked_int(size, size_name)))

    mapping = getattr(source, "mapping", None)
    source_descriptor = getattr(source, "descriptor", None)
    descriptor_offset = getattr(source_descriptor, "data_offset", None)
    if descriptor_offset is None:
        descriptor_offset = getattr(source_descriptor, "source_offset", None)
    descriptor_size = getattr(source_descriptor, "tensor_bytes", None)
    if descriptor_size is None:
        descriptor_size = getattr(source_descriptor, "nbytes", None)
    if (descriptor_offset is None) != (descriptor_size is None):
        raise InvalidRequest("source descriptor exposes incomplete offset/size range metadata")
    if descriptor_offset is not None and descriptor_size is not None:
        ranges.append(
            (
                _checked_int(descriptor_offset, "source descriptor offset"),
                _checked_int(descriptor_size, "source descriptor size"),
            )
        )
    if mapping is not None:
        size = getattr(mapping, "length", None)
        if size is None:
            size = getattr(mapping, "nbytes", None)
        offset = descriptor_offset
        if size is not None and offset is None:
            raise InvalidRequest("source mapping exposes size without a descriptor offset")
        if size is not None and offset is not None:
            ranges.append(
                (
                    _checked_int(offset, "source descriptor data_offset"),
                    _checked_int(size, "source mapping length"),
                )
            )

    size = getattr(source, "size", None)
    if size is not None and not callable(size):
        ranges.append((0, _checked_int(size, "source size")))
    size = getattr(source, "nbytes", None)
    if size is not None:
        size = _checked_int(size, "source nbytes")
        # ``source_offset`` is an offset into the source's exposed range, even
        # when the source is a wrapper rather than an ndarray.  Keep this
        # independent bound instead of assuming that a descriptor's claimed
        # tensor span is sufficient proof of the backing allocation.
        ranges.append((0, size))
    return tuple(ranges)


def _source_address(source: Any) -> int | None:
    """Derive the address of the first source byte when a source exposes one.

    ``MappedFileRange`` keeps its mapping pointer and page prefix private, so
    the adapter handles that concrete shape without importing Issue #13.  A
    missing address is deliberately preserved as an explicit offset-only
    descriptor; pointer-based production backends must reject such a descriptor
    before dereferencing it.
    """
    if source is None:
        return None
    if isinstance(source, np.ndarray):
        return int(source.__array_interface__["data"][0])
    for owner in (source, getattr(source, "mapping", None)):
        if owner is None:
            continue
        for name in ("source_address", "address", "_address"):
            address = getattr(owner, name, None)
            if address is None:
                continue
            address = _checked_int(address, f"source {name}")
            prefix = getattr(owner, "_prefix", 0)
            return address + _checked_int(prefix, "source mapping prefix")
    return None


def _validate_source_range(descriptor: CpuExpertDescriptor) -> None:
    source = descriptor.source
    if source is None:
        return
    if isinstance(source, np.ndarray) and descriptor.source_offset != 0:
        raise InvalidRequest(
            "dense ndarray sources are already tensor-start views and require source_offset=0"
        )
    packed = callable(getattr(source, "expert_packed", None))
    ranges = _source_ranges(descriptor)
    if packed and not ranges:
        raise InvalidRequest(
            f"packed source for {descriptor.projection} does not expose a bounded mapped/range size"
        )
    if not ranges:
        return
    end = descriptor.source_offset + descriptor.tensor_bytes
    for start, size in ranges:
        if start < 0 or size < 0:
            raise InvalidRequest("source range must have non-negative start and size")
        if descriptor.source_offset < start or end > start + size:
            raise InvalidRequest(
                f"{descriptor.projection} source range [{descriptor.source_offset}, {end}) "
                f"falls outside [{start}, {start + size})"
            )


@dataclass(frozen=True)
class CpuExpertLayout:
    """Model-independent collection of heterogeneous expert bank descriptors."""

    descriptors: tuple[CpuExpertDescriptor, ...]
    top_k: int

    def __post_init__(self) -> None:
        try:
            descriptors = tuple(self.descriptors)
        except TypeError as error:
            raise InvalidRequest("descriptors must be a finite iterable") from error
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "top_k", _checked_int(self.top_k, "top_k"))
        if self.top_k <= 0:
            raise InvalidRequest(f"top_k must be positive, got {self.top_k}")
        seen: set[tuple[int, str]] = set()
        for descriptor in self.descriptors:
            if not isinstance(descriptor, CpuExpertDescriptor):
                raise InvalidRequest("descriptors must contain CpuExpertDescriptor values")
            key = (descriptor.layer_id, descriptor.projection)
            if key in seen:
                raise InvalidRequest(f"duplicate expert descriptor for {key}")
            seen.add(key)
        if not self.descriptors:
            raise InvalidRequest("at least one expert descriptor is required")

    def descriptor(self, layer_id: int, projection: str) -> CpuExpertDescriptor:
        for descriptor in self.descriptors:
            if descriptor.layer_id == layer_id and descriptor.projection == projection:
                return descriptor
        raise InvalidRequest(
            f"no expert descriptor for layer={layer_id}, projection={projection!r}"
        )

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted({descriptor.layer_id for descriptor in self.descriptors}))


def cpu_layout_from_source_layout(
    layout: Any,
    source_getter: Callable[[int, str], Any] | Mapping[tuple[int, str], Any],
    *,
    top_k: int,
) -> CpuExpertLayout:
    """Adapt ``GGUFExpertLayout`` structurally, keeping model code out of the ABI."""
    if callable(source_getter):
        get_source = source_getter
    elif hasattr(source_getter, "bank"):
        get_source = source_getter.bank
    else:

        def get_source(layer: int, projection: str) -> Any:
            return source_getter[(layer, projection)]

    descriptors = tuple(
        CpuExpertDescriptor.from_source_descriptor(
            descriptor, get_source(descriptor.layer, descriptor.projection)
        )
        for descriptor in layout.descriptors
    )
    return CpuExpertLayout(descriptors=descriptors, top_k=top_k)


@dataclass(frozen=True)
class WorkspacePlan:
    max_tokens: int
    max_routes: int
    hidden_size: int
    intermediate_size: int
    workspace_bytes: int


@dataclass(frozen=True)
class CpuExecutionTelemetry:
    backend: str
    layer_id: int | None
    tokens_requested: int
    tokens_non_padded: int
    routes_requested: int
    routes_executed: int
    unique_experts: int
    bytes_read_packed: int
    elapsed_ns: int
    workspace_bytes: int
    fallback_reason: str | None = None
    cancelled: bool = False
    error: str | None = None
    error_detail: str | None = None

    @property
    def expert_count(self) -> int:
        """Compatibility alias for telemetry consumers using the issue wording."""
        return self.unique_experts

    def as_dict(self) -> dict[str, int | str | bool | None]:
        return {
            "backend": self.backend,
            "layer_id": self.layer_id,
            "tokens_requested": self.tokens_requested,
            "tokens_non_padded": self.tokens_non_padded,
            "routes_requested": self.routes_requested,
            "routes_executed": self.routes_executed,
            "unique_experts": self.unique_experts,
            "expert_count": self.expert_count,
            "bytes_read_packed": self.bytes_read_packed,
            "elapsed_ns": self.elapsed_ns,
            "workspace_bytes": self.workspace_bytes,
            "fallback_reason": self.fallback_reason,
            "cancelled": self.cancelled,
            "error": self.error,
            "error_detail": self.error_detail,
        }


@dataclass(frozen=True)
class CpuExecutionResult:
    output: np.ndarray
    telemetry: CpuExecutionTelemetry


@dataclass(frozen=True)
class CpuMicrobenchmarkProjection:
    """Serialized descriptor facts bound to one raw microbenchmark sample."""

    projection: str
    quant_name: str
    quant_type: int | str
    row_stride_bytes: int
    expert_stride_bytes: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "projection": self.projection,
            "quant_name": self.quant_name,
            "quant_type": self.quant_type,
            "row_stride_bytes": self.row_stride_bytes,
            "expert_stride_bytes": self.expert_stride_bytes,
        }


@dataclass(frozen=True)
class CpuMicrobenchmarkSample:
    """Raw repeated execution observations for one supplied route width.

    ``route_count`` is the number of leading route columns supplied to the
    executor.  ``miss_count`` is the exact number of non-negative expert IDs in
    those columns across active rows; ``-1`` denotes a hit/padded route.  The
    class intentionally stores no aggregate, threshold, or pass/fail claim.
    """

    layer_id: int
    route_count: int
    miss_count: int
    repeats: int
    elapsed_ns: tuple[int, ...]
    telemetry: tuple[CpuExecutionTelemetry, ...]
    tokens_requested: int
    tokens_non_padded: int
    hidden_size: int
    intermediate_size: int
    workspace_bytes: int
    projections: tuple[CpuMicrobenchmarkProjection, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "route_count": self.route_count,
            "miss_count": self.miss_count,
            "repeats": self.repeats,
            "elapsed_ns": list(self.elapsed_ns),
            "telemetry": [item.as_dict() for item in self.telemetry],
            "tokens_requested": self.tokens_requested,
            "tokens_non_padded": self.tokens_non_padded,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "workspace_bytes": self.workspace_bytes,
            "projections": [item.as_dict() for item in self.projections],
        }


@dataclass(frozen=True)
class CpuExecutionRequest:
    """One item in a grouped/batched executor submission."""

    layer_id: int
    hidden: np.ndarray
    expert_ids: np.ndarray
    routing_weights: np.ndarray
    num_token_non_padded: int | None = None
    output: np.ndarray | None = None
    accumulate: bool = False
    cancellation: Any = None

    def __post_init__(self) -> None:
        # Keep the grouped API subject to the same fail-closed scalar contract as
        # direct ``execute`` calls.  In particular, bool is an ``int`` subclass
        # and accepting it here would silently turn malformed request metadata
        # into a different layer or token count.
        _checked_int(self.layer_id, "request layer_id")
        if self.num_token_non_padded is not None:
            _checked_int(self.num_token_non_padded, "request num_token_non_padded")
        if not isinstance(self.accumulate, bool):
            raise InvalidRequest("request accumulate must be a bool")


class CancellationToken(Protocol):
    def cancelled(self) -> bool: ...


def _cancelled(token: Any) -> bool:
    if token is None:
        return False
    if callable(token):
        return bool(token())
    method = getattr(token, "cancelled", None)
    if callable(method):
        return bool(method())
    value = getattr(token, "is_cancelled", None)
    if callable(value):
        return bool(value())
    return bool(value)


def _decoder_accepts_workspace(decoder: QuantDecoder) -> bool:
    """Inspect a decoder once so execution can use bounded scratch when offered."""
    try:
        parameters = inspect.signature(decoder).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "out" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _clear_output(output: Any) -> None:
    """Best-effort rollback for a writable ndarray without masking the real error."""
    flags = getattr(output, "flags", None)
    if flags is None or not getattr(flags, "writeable", False):
        return
    try:
        output.fill(0)
    except (AttributeError, TypeError, ValueError):
        # Validation errors must remain InvalidRequest/ExecutionFailed even when
        # a foreign ndarray-like object cannot be cleared.
        return


class ReferenceCpuExpertExecutor:
    """Correctness-first executor with reusable bounded executor-owned workspace.

    Packed decoders should implement :class:`WorkspaceQuantDecoder` to write into
    the supplied scratch matrix.  Legacy two-argument decoders remain supported
    for reference compatibility, but may allocate a temporary decoded matrix;
    the reference executor makes no no-churn claim for that legacy path.
    """

    def __init__(
        self,
        layout: CpuExpertLayout,
        *,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        output_dtype: np.dtype | type = np.float32,
        decoders: Mapping[int | str, QuantDecoder] | None = None,
        thread_pool: ThreadPoolHook | None = None,
        numa_policy: NumaPolicyHook | None = None,
        required_alignment: int = 1,
    ) -> None:
        if activation not in {"silu", "gelu", "gelu_tanh"}:
            raise InvalidRequest(f"unsupported activation {activation!r}")
        self.layout = layout
        self.activation = activation
        if not isinstance(apply_router_weight_on_input, bool):
            raise InvalidRequest("apply_router_weight_on_input must be a bool")
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.output_dtype = np.dtype(output_dtype)
        if self.output_dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
            raise InvalidRequest(f"unsupported output dtype {self.output_dtype}")
        self.decoders = dict(decoders or {})
        self._decoder_workspace_capable = {
            id(decoder): _decoder_accepts_workspace(decoder) for decoder in self.decoders.values()
        }
        required_alignment = _checked_int(required_alignment, "required_alignment")
        if required_alignment <= 0:
            raise InvalidRequest("required_alignment must be positive")
        self.required_alignment = required_alignment
        self.thread_pool = thread_pool
        self.numa_policy = numa_policy
        self._lock = threading.Lock()
        self._plan: WorkspacePlan | None = None
        self._workspace: dict[str, np.ndarray] | None = None
        self._last_telemetry: CpuExecutionTelemetry | None = None

        for descriptor in layout.descriptors:
            if descriptor.row_stride_bytes % self.required_alignment:
                raise UnsupportedAlignment(
                    f"{descriptor.projection} row stride {descriptor.row_stride_bytes} is not "
                    f"aligned to {self.required_alignment} bytes"
                )
            if (
                descriptor.source_address is not None
                and descriptor.source_address % self.required_alignment
            ):
                raise UnsupportedAlignment(
                    f"{descriptor.projection} source address {descriptor.source_address} is not "
                    f"aligned to {self.required_alignment} bytes"
                )

        first_geometry: tuple[int, int] | None = None
        for layer_id in layout.layers:
            try:
                gate = layout.descriptor(layer_id, "gate")
                up = layout.descriptor(layer_id, "up")
                down = layout.descriptor(layer_id, "down")
            except InvalidRequest as error:
                raise UnsupportedShape(
                    f"layer {layer_id} must provide gate, up and down expert banks"
                ) from error
            if gate.num_experts != up.num_experts or gate.num_experts != down.num_experts:
                raise UnsupportedShape(f"layer {layer_id} projection expert counts disagree")
            if (gate.input_dim, gate.output_dim) != (up.input_dim, up.output_dim):
                raise UnsupportedShape(f"layer {layer_id} gate/up geometry disagrees")
            if (down.output_dim, down.input_dim) != (gate.input_dim, gate.output_dim):
                raise UnsupportedShape(f"layer {layer_id} down geometry is not transposed")
            geometry = (gate.input_dim, gate.output_dim)
            if first_geometry is None:
                first_geometry = geometry
            elif geometry != first_geometry:
                raise UnsupportedShape(
                    f"layer {layer_id} geometry {geometry} disagrees with the executor "
                    f"workspace geometry {first_geometry}"
                )

        self.hidden_size = self.layout.descriptor(self.layout.layers[0], "gate").input_dim
        self.intermediate_size = self.layout.descriptor(self.layout.layers[0], "gate").output_dim

    def prepare(self, max_tokens: int, max_routes: int) -> WorkspacePlan:
        max_tokens = _checked_int(max_tokens, "max_tokens")
        max_routes = _checked_int(max_routes, "max_routes")
        if max_tokens <= 0 or max_routes < 0:
            raise InvalidRequest("max_tokens must be positive and max_routes non-negative")
        if max_routes > self.layout.top_k:
            raise WorkspaceTooSmall(
                f"requested max_routes={max_routes} exceeds ABI top_k={self.layout.top_k}"
            )
        if not self._lock.acquire(blocking=False):
            raise Busy("cannot reprepare an executor while it is executing")
        try:
            # Four FP32 vectors, one contribution vector, and three decoded-bank
            # scratch matrices are reused by every route.  The decoded matrices
            # remain separate until each projection's matmul completes; otherwise
            # a decoder returning an ``out`` view could overwrite gate/up before
            # the nonlinear product consumes them.  A legacy two-argument packed
            # decoder may still allocate internally; that compatibility path is
            # surfaced in telemetry.
            elements = (
                4 * self.intermediate_size
                + self.hidden_size
                + max(self.hidden_size, self.intermediate_size)
                + max_tokens * self.hidden_size
                + 3 * self.hidden_size * self.intermediate_size
            )
            plan = WorkspacePlan(
                max_tokens=max_tokens,
                max_routes=max_routes,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                workspace_bytes=elements * np.dtype(np.float32).itemsize,
            )
            self._workspace = {
                "gate": np.empty(self.intermediate_size, dtype=np.float32),
                "up": np.empty(self.intermediate_size, dtype=np.float32),
                "activated": np.empty(self.intermediate_size, dtype=np.float32),
                "exp": np.empty(self.intermediate_size, dtype=np.float32),
                "contribution": np.empty(self.hidden_size, dtype=np.float32),
                "input": np.empty(max(self.hidden_size, self.intermediate_size), dtype=np.float32),
                "result": np.empty((max_tokens, self.hidden_size), dtype=np.float32),
                "decoded_gate": np.empty(
                    (self.intermediate_size, self.hidden_size), dtype=np.float32
                ),
                "decoded_up": np.empty(
                    (self.intermediate_size, self.hidden_size), dtype=np.float32
                ),
                "decoded_down": np.empty(
                    (self.hidden_size, self.intermediate_size), dtype=np.float32
                ),
            }
            self._plan = plan
            return plan
        finally:
            self._lock.release()

    @property
    def last_telemetry(self) -> CpuExecutionTelemetry | None:
        return self._last_telemetry

    def _validate_arrays(
        self,
        layer_id: int,
        hidden: np.ndarray,
        expert_ids: np.ndarray,
        routing_weights: np.ndarray,
        output: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int]:
        plan = self._plan
        if plan is None:
            raise WorkspaceNotPrepared("call prepare before execute")
        hidden = np.asarray(hidden)
        expert_ids = np.asarray(expert_ids)
        routing_weights = np.asarray(routing_weights)
        if hidden.ndim != 2:
            raise InvalidRequest(f"hidden must be rank 2, got {hidden.shape}")
        if expert_ids.ndim != 2 or routing_weights.ndim != 2:
            raise InvalidRequest("expert_ids and routing_weights must be rank 2")
        if expert_ids.shape != routing_weights.shape:
            raise InvalidRequest("expert_ids and routing_weights shapes disagree")
        if expert_ids.shape[0] != hidden.shape[0]:
            raise InvalidRequest("hidden and routing arrays have different token counts")
        if hidden.shape[1] != self.hidden_size:
            raise UnsupportedShape(
                f"hidden width {hidden.shape[1]} does not match {self.hidden_size}"
            )
        if hidden.shape[0] > plan.max_tokens or expert_ids.shape[1] > plan.max_routes:
            raise WorkspaceTooSmall(
                f"request shape {hidden.shape} / routes {expert_ids.shape[1]} exceeds "
                f"prepared ({plan.max_tokens}, {plan.max_routes})"
            )
        if not np.issubdtype(hidden.dtype, np.floating):
            raise InvalidRequest(f"hidden must be floating point, got {hidden.dtype}")
        if not np.issubdtype(expert_ids.dtype, np.integer):
            raise InvalidRequest(f"expert_ids must be integer, got {expert_ids.dtype}")
        if not np.issubdtype(routing_weights.dtype, np.floating):
            raise InvalidRequest(
                f"routing_weights must be floating point, got {routing_weights.dtype}"
            )
        if output is not None:
            if not isinstance(output, np.ndarray):
                raise InvalidRequest("output must be a NumPy ndarray")
            output = np.asarray(output)
            if output.shape != (hidden.shape[0], self.hidden_size):
                raise InvalidRequest(f"output shape {output.shape} is incompatible")
            if output.dtype != self.output_dtype:
                raise InvalidRequest(
                    f"output dtype {output.dtype} does not match {self.output_dtype}"
                )
            if not output.flags.writeable:
                raise InvalidRequest("output must be writable")
        return hidden, expert_ids, routing_weights, output, hidden.shape[0]

    def _validate_ids(
        self, descriptor: CpuExpertDescriptor, expert_ids: np.ndarray, num_tokens: int
    ) -> tuple[int, set[int]]:
        routes = 0
        unique: set[int] = set()
        for token in range(num_tokens):
            for route in range(expert_ids.shape[1]):
                expert = int(expert_ids[token, route])
                if expert == -1:
                    continue
                if expert < -1 or expert >= descriptor.num_experts:
                    raise InvalidExpertId(
                        f"layer {descriptor.layer_id}: expert {expert} outside "
                        f"[-1, {descriptor.num_experts}) at token={token}, route={route}"
                    )
                routes += 1
                unique.add(expert)
        return routes, unique

    def _source_mode(self, descriptor: CpuExpertDescriptor) -> str:
        source = descriptor.source
        if isinstance(source, np.ndarray) or hasattr(source, "expert_dense"):
            return "dense"
        packed_getter = getattr(source, "expert_packed", None)
        if not callable(packed_getter):
            return "unknown"
        decoder = self.decoders.get(descriptor.quant_type)
        if decoder is None:
            decoder = self.decoders.get(descriptor.quant_name)
        if decoder is not None and self._decoder_workspace_capable.get(id(decoder), False):
            return "packed_workspace"
        return "packed_legacy"

    @staticmethod
    def _fallback_reason(source_modes: set[str]) -> str:
        if not source_modes:
            return "reference_no_routes"
        if source_modes == {"dense"}:
            return "reference_dense"
        if source_modes == {"packed_workspace"}:
            return "reference_dequant_packed_workspace"
        if source_modes == {"packed_legacy"}:
            return "reference_dequant_packed_legacy"
        if source_modes <= {"packed_workspace", "packed_legacy"}:
            return "reference_dequant_packed_mixed_decoder"
        if "unknown" in source_modes:
            return "reference_unknown_source"
        return "reference_mixed_dense_packed"

    def _dense_expert(
        self,
        descriptor: CpuExpertDescriptor,
        expert: int,
        decoder_workspace: np.ndarray | None = None,
    ) -> np.ndarray:
        source = descriptor.source
        if source is None:
            raise ExecutionFailed(
                f"{descriptor.quant_name} layer {descriptor.layer_id}/{descriptor.projection} "
                "has no source"
            )
        if hasattr(source, "expert_dense"):
            dense = source.expert_dense(expert)
        elif isinstance(source, np.ndarray):
            if source.ndim != 3 or source.shape[0] != descriptor.num_experts:
                raise UnsupportedShape("dense source must have [experts, output, input] shape")
            dense = source[expert]
        else:
            packed_getter = getattr(source, "expert_packed", None)
            if not callable(packed_getter):
                raise ExecutionFailed(
                    "expert source exposes neither expert_dense nor expert_packed"
                )
            packed = np.asarray(packed_getter(expert))
            expected_packed = (descriptor.rows_per_expert, descriptor.row_stride_bytes)
            if packed.shape != expected_packed:
                raise UnsupportedShape(
                    f"packed {descriptor.projection} expert shape {packed.shape}, "
                    f"expected {expected_packed}"
                )
            if packed.dtype != np.dtype(np.uint8) or not packed.flags.c_contiguous:
                raise InvalidRequest(
                    f"packed {descriptor.projection} expert must be a contiguous uint8 byte view"
                )
            if descriptor.source_address is not None:
                actual_address = int(packed.__array_interface__["data"][0])
                expected_address = (
                    descriptor.source_address + expert * descriptor.expert_stride_bytes
                )
                if actual_address != expected_address:
                    raise InvalidRequest(
                        f"packed {descriptor.projection} expert address {actual_address} "
                        f"does not match expected {expected_address}"
                    )
            decoder = self.decoders.get(descriptor.quant_type)
            if decoder is None:
                decoder = self.decoders.get(descriptor.quant_name)
            if decoder is None:
                raise UnsupportedQuantType(
                    f"no reference decoder registered for {descriptor.quant_name}"
                )
            if decoder_workspace is not None and self._decoder_workspace_capable.get(
                id(decoder), False
            ):
                target = decoder_workspace[: descriptor.output_dim, : descriptor.input_dim]
                dense = decoder(packed, descriptor, out=target)
                # A workspace-capable decoder is part of the bounded-allocation
                # contract.  Requiring the exact target (or a view with the
                # same pointer, shape, dtype and strides) prevents an adapter
                # from silently allocating a replacement and leaving the
                # caller's scratch stale.
                try:
                    returned = np.asarray(dense)
                    target_address = int(target.__array_interface__["data"][0])
                    returned_address = int(returned.__array_interface__["data"][0])
                except (AttributeError, TypeError, ValueError, IndexError) as error:
                    raise InvalidRequest(
                        "workspace decoder must return the supplied target"
                    ) from error
                if (
                    returned.shape != target.shape
                    or returned.dtype != target.dtype
                    or returned.strides != target.strides
                    or not returned.flags.c_contiguous
                    or returned_address != target_address
                ):
                    raise InvalidRequest("workspace decoder must return the supplied target")
                dense = returned
            else:
                dense = decoder(packed, descriptor)
        dense = np.asarray(dense, dtype=np.float32)
        expected = (descriptor.output_dim, descriptor.input_dim)
        if dense.shape != expected:
            raise UnsupportedShape(
                f"decoded {descriptor.projection} expert shape {dense.shape}, expected {expected}"
            )
        return dense

    def _activation_inplace(
        self,
        gate: np.ndarray,
        up: np.ndarray,
        activated: np.ndarray,
        exp: np.ndarray,
    ) -> None:
        if self.activation == "silu":
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                np.negative(gate, out=exp)
                np.exp(exp, out=exp)
                np.add(exp, 1.0, out=exp)
                np.divide(gate, exp, out=activated)
        elif self.activation == "gelu":
            for index, value in enumerate(gate):
                activated[index] = 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                np.multiply(
                    0.5 * gate,
                    1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (gate + 0.044715 * gate**3)),
                    out=activated,
                )
        np.multiply(activated, up, out=activated)

    def _matvec(
        self,
        descriptor: CpuExpertDescriptor,
        expert: int,
        vector: np.ndarray,
        out: np.ndarray,
        *,
        decoder_workspace: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute one expert row-matrix vector product into ``out``.

        This protected seam is the Issue #15 reference/optimized boundary:
        the reference implementation decodes to its prepared scratch matrix,
        while format-specific executors may consume packed rows directly.
        """
        dense = self._dense_expert(descriptor, expert, decoder_workspace=decoder_workspace)
        np.matmul(dense, vector, out=out)
        return out

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
    ) -> CpuExecutionResult:
        started = time.perf_counter_ns()
        raw_layer_id = layer_id
        if not self._lock.acquire(blocking=False):
            error = Busy("executor already has an active request")
            error.telemetry = CpuExecutionTelemetry(
                backend="reference",
                layer_id=None,
                tokens_requested=0,
                tokens_non_padded=0,
                routes_requested=0,
                routes_executed=0,
                unique_experts=0,
                bytes_read_packed=0,
                elapsed_ns=time.perf_counter_ns() - started,
                workspace_bytes=self._plan.workspace_bytes if self._plan else 0,
                fallback_reason="busy",
                error="Busy",
                error_detail=str(error),
            )
            self._last_telemetry = error.telemetry
            raise error
        result_output = output
        layer_id: int | None = None
        tokens = 0
        active_tokens = 0
        routes_requested = 0
        routes_executed = 0
        unique: set[int] = set()
        bytes_read = 0
        source_modes: set[str] = set()
        telemetry: CpuExecutionTelemetry | None = None
        try:
            layer_id = _checked_int(raw_layer_id, "layer_id")
            if not isinstance(accumulate, bool):
                raise InvalidRequest("accumulate must be a bool")
            hidden, expert_ids, routing_weights, result_output, tokens = self._validate_arrays(
                layer_id, hidden, expert_ids, routing_weights, output
            )
            if num_token_non_padded is None:
                active_tokens = tokens
            else:
                active_tokens = _checked_int(num_token_non_padded, "num_token_non_padded")
                if not 0 <= active_tokens <= tokens:
                    raise InvalidRequest(
                        f"num_token_non_padded={active_tokens} outside [0, {tokens}]"
                    )
            descriptor = self.layout.descriptor(layer_id, "gate")
            up_descriptor = self.layout.descriptor(layer_id, "up")
            down_descriptor = self.layout.descriptor(layer_id, "down")
            routes_requested = active_tokens * expert_ids.shape[1]
            _, unique = self._validate_ids(descriptor, expert_ids, active_tokens)
            if _cancelled(cancellation):
                raise Cancelled("CPU expert execution cancelled before compute")

            plan = self._plan
            workspace = self._workspace
            assert plan is not None and workspace is not None
            result = workspace["result"][:tokens]
            if accumulate and result_output is not None:
                np.copyto(result, result_output, casting="unsafe")
            else:
                result.fill(0.0)
            result[active_tokens:].fill(0.0)

            for token in range(active_tokens):
                for route in range(expert_ids.shape[1]):
                    expert = int(expert_ids[token, route])
                    if expert == -1:
                        continue
                    if _cancelled(cancellation):
                        raise Cancelled("CPU expert execution cancelled during compute")
                    weight = float(routing_weights[token, route])
                    route_descriptors = (descriptor, up_descriptor, down_descriptor)
                    source_modes.update(self._source_mode(item) for item in route_descriptors)
                    bytes_read += sum(
                        item.expert_stride_bytes
                        for item in route_descriptors
                        if self._source_mode(item).startswith("packed")
                    )
                    # Match the production fused-MoE contract exactly.  In input
                    # mode the route scale is applied to both gate and up outputs
                    # before the nonlinear SwiGLU product.  Because activation is
                    # nonlinear, this intentionally is not equivalent to scaling
                    # the final down output once; the latter is the false-mode
                    # contract used by the current production path.
                    self._matvec(
                        descriptor,
                        expert,
                        hidden[token],
                        workspace["gate"],
                        decoder_workspace=workspace["decoded_gate"],
                    )
                    self._matvec(
                        up_descriptor,
                        expert,
                        hidden[token],
                        workspace["up"],
                        decoder_workspace=workspace["decoded_up"],
                    )
                    if self.apply_router_weight_on_input:
                        np.multiply(workspace["gate"], weight, out=workspace["gate"])
                        np.multiply(workspace["up"], weight, out=workspace["up"])
                    self._activation_inplace(
                        workspace["gate"],
                        workspace["up"],
                        workspace["activated"],
                        workspace["exp"],
                    )
                    self._matvec(
                        down_descriptor,
                        expert,
                        workspace["activated"],
                        workspace["contribution"],
                        decoder_workspace=workspace["decoded_down"],
                    )
                    if not self.apply_router_weight_on_input:
                        np.multiply(
                            workspace["contribution"],
                            weight,
                            out=workspace["contribution"],
                        )
                    np.add(result[token], workspace["contribution"], out=result[token])
                    routes_executed += 1

            if result_output is None:
                result_output = np.empty((tokens, self.hidden_size), dtype=self.output_dtype)
            np.copyto(result_output, result, casting="unsafe")
            telemetry = CpuExecutionTelemetry(
                backend="reference",
                layer_id=layer_id,
                tokens_requested=tokens,
                tokens_non_padded=active_tokens,
                routes_requested=routes_requested,
                routes_executed=routes_executed,
                unique_experts=len(unique),
                bytes_read_packed=bytes_read,
                elapsed_ns=time.perf_counter_ns() - started,
                workspace_bytes=plan.workspace_bytes,
                fallback_reason=self._fallback_reason(source_modes),
            )
            self._last_telemetry = telemetry
            return CpuExecutionResult(output=result_output, telemetry=telemetry)
        except CpuAbiError as error:
            _clear_output(result_output)
            plan = self._plan
            telemetry = CpuExecutionTelemetry(
                backend="reference",
                layer_id=layer_id,
                tokens_requested=tokens,
                tokens_non_padded=active_tokens,
                routes_requested=routes_requested,
                routes_executed=routes_executed,
                unique_experts=len(unique),
                bytes_read_packed=bytes_read,
                elapsed_ns=time.perf_counter_ns() - started,
                workspace_bytes=plan.workspace_bytes if plan else 0,
                fallback_reason=self._fallback_reason(source_modes),
                cancelled=isinstance(error, Cancelled),
                error=type(error).__name__,
                error_detail=str(error),
            )
            error.telemetry = telemetry
            self._last_telemetry = telemetry
            raise
        except Exception as error:
            _clear_output(result_output)
            plan = self._plan
            wrapped = ExecutionFailed(f"reference CPU expert execution failed: {error}")
            wrapped.telemetry = CpuExecutionTelemetry(
                backend="reference",
                layer_id=layer_id,
                tokens_requested=tokens,
                tokens_non_padded=active_tokens,
                routes_requested=routes_requested,
                routes_executed=routes_executed,
                unique_experts=len(unique),
                bytes_read_packed=bytes_read,
                elapsed_ns=time.perf_counter_ns() - started,
                workspace_bytes=plan.workspace_bytes if plan else 0,
                fallback_reason=self._fallback_reason(source_modes),
                error=type(error).__name__,
                error_detail=str(error),
            )
            self._last_telemetry = wrapped.telemetry
            raise wrapped from error
        finally:
            self._lock.release()

    def execute_group(
        self, requests: Iterable[CpuExecutionRequest]
    ) -> tuple[CpuExecutionResult, ...]:
        """Execute a group of independent requests through the same prepared ABI.

        The reference implementation intentionally serializes the group.  A later
        worker-pool backend can use the same request objects to group routes by expert
        without changing the descriptor, cancellation or output contracts.
        Each request commits independently; if a later request fails, earlier
        successful outputs remain committed and no cross-request rollback is attempted.
        """
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
    ) -> tuple[CpuMicrobenchmarkSample, ...]:
        """Collect raw per-repeat observations for the supplied geometry.

        Each route count selects a leading prefix of the caller's route columns;
        the corresponding miss count is the exact number of non-padding expert
        IDs in that prefix.  This measures the real supplied IDs, not a synthetic
        miss pattern.  By default every width from one through the supplied
        ``top_k`` width is returned, with its observed miss count.  No warm-up,
        aggregate, or performance decision is implied by this API.
        """
        repeats = _checked_int(repeats, "repeats")
        if repeats <= 0:
            raise InvalidRequest("repeats must be positive")
        layer_id = _checked_int(layer_id, "layer_id")
        hidden, expert_ids, routing_weights, _, tokens = self._validate_arrays(
            layer_id, hidden, expert_ids, routing_weights, None
        )
        if num_token_non_padded is None:
            active_tokens = tokens
        else:
            active_tokens = _checked_int(num_token_non_padded, "num_token_non_padded")
            if not 0 <= active_tokens <= tokens:
                raise InvalidRequest(f"num_token_non_padded={active_tokens} outside [0, {tokens}]")
        max_width = min(self.layout.top_k, expert_ids.shape[1])
        if max_width <= 0:
            raise InvalidRequest("microbenchmark requires at least one supplied route")

        if route_counts is None:
            selected_widths = tuple(range(1, max_width + 1))
        else:
            try:
                selected_widths = tuple(
                    _checked_int(value, "route_count") for value in route_counts
                )
            except TypeError as error:
                raise InvalidRequest("route_counts must be an iterable of integers") from error
            if not selected_widths:
                raise InvalidRequest("route_counts must not be empty")
        for route_count in selected_widths:
            if not 1 <= route_count <= max_width:
                raise InvalidRequest(f"route_count={route_count} outside [1, {max_width}]")

        if miss_counts is None:
            expected_misses: tuple[int | None, ...] = (None,) * len(selected_widths)
        else:
            try:
                expected_misses = tuple(_checked_int(value, "miss_count") for value in miss_counts)
            except TypeError as error:
                raise InvalidRequest("miss_counts must be an iterable of integers") from error
            if len(expected_misses) != len(selected_widths):
                raise InvalidRequest("route_counts and miss_counts must have equal lengths")

        descriptor = self.layout.descriptor(layer_id, "gate")
        plan = self._plan
        assert plan is not None
        projections = tuple(
            CpuMicrobenchmarkProjection(
                projection=projection,
                quant_name=self.layout.descriptor(layer_id, projection).quant_name,
                quant_type=self.layout.descriptor(layer_id, projection).quant_type,
                row_stride_bytes=self.layout.descriptor(layer_id, projection).row_stride_bytes,
                expert_stride_bytes=self.layout.descriptor(
                    layer_id, projection
                ).expert_stride_bytes,
            )
            for projection in ("gate", "up", "down")
        )
        samples: list[CpuMicrobenchmarkSample] = []
        for route_count, expected in zip(selected_widths, expected_misses, strict=True):
            ids = expert_ids[:active_tokens, :route_count]
            actual_misses, _ = self._validate_ids(descriptor, ids, active_tokens)
            if expected is not None:
                if expected < 0 or expected != actual_misses:
                    raise InvalidRequest(
                        f"miss_count={expected} does not match supplied active IDs "
                        f"({actual_misses}) for route_count={route_count}"
                    )
            output = np.empty((tokens, self.hidden_size), dtype=self.output_dtype)
            observations: list[int] = []
            telemetry: list[CpuExecutionTelemetry] = []
            for _ in range(repeats):
                result = self.execute(
                    layer_id,
                    hidden,
                    expert_ids[:, :route_count],
                    routing_weights[:, :route_count],
                    num_token_non_padded=active_tokens,
                    output=output,
                )
                observations.append(result.telemetry.elapsed_ns)
                telemetry.append(result.telemetry)
            samples.append(
                CpuMicrobenchmarkSample(
                    layer_id=layer_id,
                    route_count=route_count,
                    miss_count=actual_misses,
                    repeats=repeats,
                    elapsed_ns=tuple(observations),
                    telemetry=tuple(telemetry),
                    tokens_requested=tokens,
                    tokens_non_padded=active_tokens,
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    workspace_bytes=plan.workspace_bytes,
                    projections=projections,
                )
            )
        return tuple(samples)


__all__ = [
    "Busy",
    "CancellationToken",
    "Cancelled",
    "CpuAbiError",
    "CpuExecutionRequest",
    "CpuExecutionResult",
    "CpuExecutionTelemetry",
    "CpuExpertDescriptor",
    "CpuExpertLayout",
    "CpuMicrobenchmarkProjection",
    "CpuMicrobenchmarkSample",
    "ExecutionFailed",
    "ExpertSource",
    "InvalidExpertId",
    "InvalidRequest",
    "NumaPolicyHook",
    "ReferenceCpuExpertExecutor",
    "ThreadPoolHook",
    "UnsupportedAlignment",
    "UnsupportedQuantType",
    "UnsupportedShape",
    "WorkspaceNotPrepared",
    "WorkspacePlan",
    "WorkspaceQuantDecoder",
    "WorkspaceTooSmall",
    "cpu_layout_from_source_layout",
]
