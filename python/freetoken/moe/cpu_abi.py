"""Torch-free contract for host-side routed expert execution.

This module deliberately stops at the executor boundary.  It owns neither a
quantizer nor a thread pool: production decoders and scheduling policies can be
plugged in without importing a model implementation.  The reference executor
is intentionally small and is used as a correctness oracle for later AVX2
backends.
"""

from __future__ import annotations

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


QuantDecoder = Callable[[np.ndarray, "CpuExpertDescriptor"], np.ndarray]


class ThreadPoolHook(Protocol):
    """Optional worker-pool seam for optimized executors."""

    def submit(self, callable: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...


class NumaPolicyHook(Protocol):
    """Optional placement seam; the ABI does not prescribe a machine topology."""

    def placement(self, layer_id: int, projection: str) -> Any: ...


@dataclass(frozen=True)
class CpuExpertDescriptor:
    """Immutable geometry and source-address contract for one expert bank."""

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
        if self.layer_id < 0:
            raise InvalidRequest(f"layer_id must be non-negative, got {self.layer_id}")
        if not self.projection:
            raise InvalidRequest("projection must not be empty")
        if not self.quant_name:
            raise InvalidRequest("quant_name must not be empty")
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

    @classmethod
    def from_source_descriptor(cls, descriptor: Any, source: Any) -> CpuExpertDescriptor:
        """Adapt an Issue #13-style descriptor without importing its module."""
        return cls(
            layer_id=int(descriptor.layer),
            projection=str(descriptor.projection),
            quant_type=int(descriptor.quant_type),
            quant_name=str(descriptor.quant_name),
            num_experts=int(descriptor.experts),
            output_dim=int(descriptor.output_dim),
            input_dim=int(descriptor.input_dim),
            rows_per_expert=int(descriptor.output_dim),
            row_stride_bytes=int(descriptor.row_bytes),
            expert_stride_bytes=int(descriptor.bytes_per_expert),
            tensor_bytes=int(descriptor.tensor_bytes),
            source_offset=int(descriptor.data_offset),
            pool_id=int(getattr(descriptor, "pool_id", -1)),
            source=source,
        )


@dataclass(frozen=True)
class CpuExpertLayout:
    """Model-independent collection of heterogeneous expert bank descriptors."""

    descriptors: tuple[CpuExpertDescriptor, ...]
    top_k: int

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise InvalidRequest(f"top_k must be positive, got {self.top_k}")
        seen: set[tuple[int, str]] = set()
        for descriptor in self.descriptors:
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
        }


@dataclass(frozen=True)
class CpuExecutionResult:
    output: np.ndarray
    telemetry: CpuExecutionTelemetry


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


class ReferenceCpuExpertExecutor:
    """Correctness-first dense reference executor with reusable bounded workspace."""

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
        self.apply_router_weight_on_input = bool(apply_router_weight_on_input)
        self.output_dtype = np.dtype(output_dtype)
        if self.output_dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
            raise InvalidRequest(f"unsupported output dtype {self.output_dtype}")
        self.decoders = dict(decoders or {})
        if required_alignment <= 0:
            raise InvalidRequest("required_alignment must be positive")
        self.required_alignment = int(required_alignment)
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

        self.hidden_size = self.layout.descriptor(self.layout.layers[0], "gate").input_dim
        self.intermediate_size = self.layout.descriptor(
            self.layout.layers[0], "gate"
        ).output_dim

    def prepare(self, max_tokens: int, max_routes: int) -> WorkspacePlan:
        if max_tokens <= 0 or max_routes < 0:
            raise InvalidRequest("max_tokens must be positive and max_routes non-negative")
        if max_routes > self.layout.top_k:
            raise WorkspaceTooSmall(
                f"requested max_routes={max_routes} exceeds ABI top_k={self.layout.top_k}"
            )
        if not self._lock.acquire(blocking=False):
            raise Busy("cannot reprepare an executor while it is executing")
        try:
            # Four FP32 vectors and one contribution vector are reused by every route.
            # The result matrix is separate so caller output is only committed after
            # success.  No route-sized allocation is made here or during execute.
            elements = (
                4 * self.intermediate_size
                + self.hidden_size
                + max_tokens * self.hidden_size
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
                "result": np.empty((max_tokens, self.hidden_size), dtype=np.float32),
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

    def _dense_expert(self, descriptor: CpuExpertDescriptor, expert: int) -> np.ndarray:
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
            if packed_getter is None:
                raise ExecutionFailed(
                    "expert source exposes neither expert_dense nor expert_packed"
                )
            packed = np.asarray(packed_getter(expert))
            decoder = self.decoders.get(descriptor.quant_type)
            if decoder is None:
                decoder = self.decoders.get(descriptor.quant_name)
            if decoder is None:
                raise UnsupportedQuantType(
                    f"no reference decoder registered for {descriptor.quant_name}"
                )
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
        if not self._lock.acquire(blocking=False):
            raise Busy("executor already has an active request")
        result_output = output
        tokens = 0
        routes_requested = 0
        routes_executed = 0
        unique: set[int] = set()
        bytes_read = 0
        telemetry: CpuExecutionTelemetry | None = None
        try:
            hidden, expert_ids, routing_weights, result_output, tokens = self._validate_arrays(
                layer_id, hidden, expert_ids, routing_weights, output
            )
            if num_token_non_padded is None:
                active_tokens = tokens
            elif not isinstance(num_token_non_padded, (int, np.integer)):
                raise InvalidRequest("num_token_non_padded must be an integer")
            else:
                active_tokens = int(num_token_non_padded)
                if not 0 <= active_tokens <= tokens:
                    raise InvalidRequest(
                        f"num_token_non_padded={active_tokens} outside [0, {tokens}]"
                    )
            descriptor = self.layout.descriptor(layer_id, "gate")
            up_descriptor = self.layout.descriptor(layer_id, "up")
            down_descriptor = self.layout.descriptor(layer_id, "down")
            routes_requested = active_tokens * expert_ids.shape[1]
            routes_executed, unique = self._validate_ids(descriptor, expert_ids, active_tokens)
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
                    gate = self._dense_expert(descriptor, expert)
                    up = self._dense_expert(up_descriptor, expert)
                    down = self._dense_expert(down_descriptor, expert)
                    bytes_read += (
                        descriptor.expert_stride_bytes
                        + up_descriptor.expert_stride_bytes
                        + down_descriptor.expert_stride_bytes
                    )
                    np.matmul(gate, hidden[token], out=workspace["gate"])
                    np.matmul(up, hidden[token], out=workspace["up"])
                    if self.apply_router_weight_on_input:
                        np.multiply(workspace["gate"], weight, out=workspace["gate"])
                        np.multiply(workspace["up"], weight, out=workspace["up"])
                    self._activation_inplace(
                        workspace["gate"],
                        workspace["up"],
                        workspace["activated"],
                        workspace["exp"],
                    )
                    np.matmul(down, workspace["activated"], out=workspace["contribution"])
                    if not self.apply_router_weight_on_input:
                        np.multiply(
                            workspace["contribution"],
                            weight,
                            out=workspace["contribution"],
                        )
                    np.add(result[token], workspace["contribution"], out=result[token])

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
                fallback_reason="reference_dequant_dense",
            )
            self._last_telemetry = telemetry
            return CpuExecutionResult(output=result_output, telemetry=telemetry)
        except CpuAbiError as error:
            if result_output is not None and result_output.flags.writeable:
                result_output.fill(0)
            plan = self._plan
            telemetry = CpuExecutionTelemetry(
                backend="reference",
                layer_id=layer_id,
                tokens_requested=tokens,
                tokens_non_padded=0,
                routes_requested=routes_requested,
                routes_executed=routes_executed,
                unique_experts=len(unique),
                bytes_read_packed=bytes_read,
                elapsed_ns=time.perf_counter_ns() - started,
                workspace_bytes=plan.workspace_bytes if plan else 0,
                fallback_reason="reference_dequant_dense",
                cancelled=isinstance(error, Cancelled),
                error=type(error).__name__,
            )
            error.telemetry = telemetry
            self._last_telemetry = telemetry
            raise
        except Exception as error:
            if result_output is not None and result_output.flags.writeable:
                result_output.fill(0)
            plan = self._plan
            wrapped = ExecutionFailed(f"reference CPU expert execution failed: {error}")
            wrapped.telemetry = CpuExecutionTelemetry(
                backend="reference",
                layer_id=layer_id,
                tokens_requested=tokens,
                tokens_non_padded=0,
                routes_requested=routes_requested,
                routes_executed=routes_executed,
                unique_experts=len(unique),
                bytes_read_packed=bytes_read,
                elapsed_ns=time.perf_counter_ns() - started,
                workspace_bytes=plan.workspace_bytes if plan else 0,
                fallback_reason="reference_dequant_dense",
                error=type(error).__name__,
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
    "WorkspaceTooSmall",
    "cpu_layout_from_source_layout",
]
