"""Explicit eager device bridge for the CPU GGUF expert adapter.

The Qwen GGUF expert layer is deliberately a CPU-only decode boundary.  This
module provides the smallest explicit bridge around that boundary: device
inputs are copied to CPU, the adapter is invoked exactly once, and its result
is copied back to the input device.  The default transfer implementation is
blocking ``Tensor.to``; streams, pinned buffers, graph capture, overlap, and
cache ownership are intentionally out of scope.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Protocol

from freetoken.moe.gguf_cpu import UnsupportedGGUFCpuConfiguration


class GGUFEagerBridgeError(RuntimeError):
    """Base error for the explicit eager CPU bridge."""


class GGUFEagerBridgeBusy(GGUFEagerBridgeError):
    """The bridge rejected a request because another request owns it."""


class EagerTransferSeam(Protocol):
    """Blocking tensor transfer operations used by :class:`GGUFCpuEagerBridge`.

    Implementations used by tests may translate fake device tensors.  A
    production implementation must return independent tensors and must not
    make asynchronous or stream-ordering claims through this interface.
    """

    def to_cpu(self, tensor: Any, *, name: str) -> Any:
        """Return a blocking CPU copy of ``tensor``."""

    def to_device(self, tensor: Any, *, device: Any, dtype: Any, name: str) -> Any:
        """Return a blocking copy of ``tensor`` on ``device`` and with ``dtype``."""


class BlockingTensorTransfer:
    """Default transfer seam using blocking ``Tensor.to`` calls."""

    def to_cpu(self, tensor: Any, *, name: str) -> Any:
        del name
        return tensor.to(device="cpu")

    def to_device(self, tensor: Any, *, device: Any, dtype: Any, name: str) -> Any:
        del name
        return tensor.to(device=device, dtype=dtype)


def _device_type(device: Any) -> str:
    if device is None:
        return "cpu"
    return str(getattr(device, "type", device)).split(":", 1)[0].lower()


def _shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("GGUF eager bridge inputs must expose a tensor-like shape") from error


def _dtype(value: Any) -> Any:
    try:
        return value.dtype
    except AttributeError as error:
        raise TypeError("GGUF eager bridge inputs must expose a tensor-like dtype") from error


def _device(value: Any) -> Any:
    try:
        return value.device
    except AttributeError as error:
        raise TypeError("GGUF eager bridge inputs must expose a tensor-like device") from error


def _element_bytes(value: Any) -> int:
    try:
        return max(0, int(value.element_size()))
    except (AttributeError, TypeError, ValueError):
        return 0


@dataclass
class _RequestProgress:
    """Mutable, request-local facts retained when an eager call fails."""

    started: int
    generation: int
    phase: str | None
    group_size: int | None
    input_device: str | None = None
    input_dtype: str | None = None
    output_device: str | None = None
    output_dtype: str | None = None
    transfers: list[str] = field(default_factory=list)
    transfer_bytes: int = 0
    transfer_attempted: bool = False
    cpu_execution_count: int = 0
    adapter_started: bool = False
    adapter_completed: bool = False


@dataclass(frozen=True)
class GGUFEagerBridgeTelemetry:
    """Request-scoped facts from one eager bridge attempt."""

    backend: str
    layer_id: int | None
    phase: str | None
    group_size: int | None
    input_device: str | None
    output_device: str | None
    input_dtype: str | None
    output_dtype: str | None
    transfer_path: str
    transfers: tuple[str, ...]
    transfer_count: int
    transfer_bytes: int
    cpu_execution_count: int
    adapter_started: bool
    adapter_completed: bool
    adapter_telemetry: Mapping[str, object] | None = None
    fallback_reason: str | None = None
    error: str | None = None
    error_detail: str | None = None
    elapsed_ns: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "layer_id": self.layer_id,
            "phase": self.phase,
            "group_size": self.group_size,
            "input_device": self.input_device,
            "output_device": self.output_device,
            "input_dtype": self.input_dtype,
            "output_dtype": self.output_dtype,
            "transfer_path": self.transfer_path,
            "transfers": list(self.transfers),
            "transfer_count": self.transfer_count,
            "transfer_bytes": self.transfer_bytes,
            "cpu_execution_count": self.cpu_execution_count,
            "adapter_started": self.adapter_started,
            "adapter_completed": self.adapter_completed,
            "adapter_telemetry": (
                None if self.adapter_telemetry is None else dict(self.adapter_telemetry)
            ),
            "fallback_reason": self.fallback_reason,
            "error": self.error,
            "error_detail": self.error_detail,
            "elapsed_ns": self.elapsed_ns,
        }


class GGUFCpuEagerBridge:
    """Borrow a CPU GGUF layer and expose an explicit eager device boundary.

    The wrapped layer and its bundle remain caller-owned.  The bridge accepts
    one prefill or decode request at a time, is cache-free and TP1-only, and
    never closes the wrapped object.  ``phase`` is intentionally required on
    both request methods so the caller labels the CPU boundary explicitly.
    """

    _BACKEND = "qwen_gguf_eager_cpu_bridge"
    requires_moe_execution_context = True

    def __init__(
        self,
        layer: Any,
        *,
        transfer: EagerTransferSeam | None = None,
        cache_size: int = 0,
        tp_size: int = 1,
    ) -> None:
        if not callable(getattr(layer, "forward", None)) or not callable(
            getattr(layer, "routed_forward", None)
        ):
            raise TypeError("GGUF eager bridge requires a QwenGGUFCpuMoELayer-like object")
        if isinstance(cache_size, bool) or not isinstance(cache_size, Integral):
            raise UnsupportedGGUFCpuConfiguration(
                f"GGUF eager bridge cache_size must be an integer, got {cache_size!r}"
            )
        if int(cache_size) != 0:
            raise UnsupportedGGUFCpuConfiguration(
                "GGUF eager bridge requires cache_size=0; GPU expert caching is unsupported"
            )
        if isinstance(tp_size, bool) or not isinstance(tp_size, Integral) or int(tp_size) != 1:
            raise UnsupportedGGUFCpuConfiguration(
                f"GGUF eager bridge requires TP=1, got tensor-parallel size {tp_size!r}"
            )
        if bool(getattr(layer, "closed", False)):
            raise RuntimeError("GGUF eager bridge cannot wrap a closed CPU expert layer")
        self._layer = layer
        self._transfer = transfer if transfer is not None else BlockingTensorTransfer()
        self._lock = threading.Lock()
        self._telemetry_lock = threading.Lock()
        self._next_generation = 0
        self._committed_generation = 0
        self._closed = False
        self._frozen = False
        self._last_telemetry: GGUFEagerBridgeTelemetry | None = None
        self._last_error_telemetry: GGUFEagerBridgeTelemetry | None = None

    @property
    def layer(self) -> Any:
        """The borrowed adapter; its owner controls its lifetime."""
        return self._layer

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def last_telemetry(self) -> GGUFEagerBridgeTelemetry | None:
        with self._telemetry_lock:
            return self._last_telemetry

    @property
    def last_error_telemetry(self) -> GGUFEagerBridgeTelemetry | None:
        with self._telemetry_lock:
            return self._last_error_telemetry

    @property
    def host_weight_telemetry(self) -> dict[str, object]:
        """Expose adapter telemetry together with the current bridge request."""
        value: dict[str, object] = {}
        adapter_telemetry = getattr(self._layer, "host_weight_telemetry", None)
        if callable(adapter_telemetry):
            value.update(adapter_telemetry())
        elif isinstance(adapter_telemetry, Mapping):
            value.update(adapter_telemetry)
        with self._telemetry_lock:
            telemetry = self._last_telemetry
            error_telemetry = self._last_error_telemetry
        value["eager_bridge_telemetry"] = None if telemetry is None else telemetry.as_dict()
        value["eager_bridge_error_telemetry"] = (
            None if error_telemetry is None else error_telemetry.as_dict()
        )
        return value

    def forward(
        self,
        hidden_states: Any,
        router_logits: Any | None = None,
        debug_observer: Callable[[str, dict[str, object]], None] | None = None,
        *,
        phase: str,
        group_size: int = 1,
        graph_capture: bool = False,
        workspace: Any = None,
        num_token_non_padded: int | None = None,
    ) -> Any:
        """Route through the wrapped adapter once, with an explicit phase."""
        if router_logits is None:
            return self._reject(
                ValueError("router_logits is required; the eager bridge never routes twice"),
                hidden_states=hidden_states,
                phase=phase,
                group_size=group_size,
            )
        return self._run(
            hidden_states,
            (router_logits,),
            phase=phase,
            group_size=group_size,
            graph_capture=graph_capture,
            workspace=workspace,
            num_token_non_padded=num_token_non_padded,
            debug_observer=debug_observer,
            method="forward",
        )

    def routed_forward(
        self,
        hidden_states: Any,
        topk_weights: Any,
        topk_ids: Any,
        debug_observer: Callable[[str, dict[str, object]], None] | None = None,
        *,
        phase: str,
        group_size: int = 1,
        graph_capture: bool = False,
        workspace: Any = None,
        num_token_non_padded: int | None = None,
    ) -> Any:
        """Execute an already prepared route exactly once through the adapter."""
        return self._run(
            hidden_states,
            (topk_weights, topk_ids),
            phase=phase,
            group_size=group_size,
            graph_capture=graph_capture,
            workspace=workspace,
            num_token_non_padded=num_token_non_padded,
            debug_observer=debug_observer,
            method="routed_forward",
        )

    def _run(
        self,
        hidden_states: Any,
        route_inputs: tuple[Any, ...],
        *,
        phase: str,
        group_size: int,
        graph_capture: bool,
        workspace: Any,
        num_token_non_padded: int | None,
        debug_observer: Callable[[str, dict[str, object]], None] | None,
        method: str,
    ) -> Any:
        started = time.perf_counter_ns()
        progress = self._new_request_progress(
            started=started,
            phase=phase,
            group_size=group_size,
        )
        try:
            self._begin_request()
            self._validate_request_mode(
                phase,
                group_size,
                graph_capture=graph_capture,
                workspace=workspace,
            )
            inputs = (hidden_states, *route_inputs)
            for index, value in enumerate(inputs):
                if value is None:
                    raise TypeError(f"eager bridge input {index} cannot be None")
                _device(value)
                _dtype(value)
                _shape(value)
            if not self._lock.acquire(blocking=False):
                raise GGUFEagerBridgeBusy("GGUF eager bridge is busy with another request")
            try:
                # Validation intentionally precedes admission so unsupported calls never
                # perform a transfer. The closed check must remain under this lock: a
                # request validated before close() must not execute after close() wins.
                if self._closed:
                    raise RuntimeError("GGUF eager bridge is closed")
                if self._frozen:
                    raise GGUFEagerBridgeBusy("GGUF eager bridge admission is frozen")
                output, telemetry = self._execute_locked(
                    hidden_states,
                    route_inputs,
                    phase=phase,
                    group_size=group_size,
                    num_token_non_padded=num_token_non_padded,
                    debug_observer=debug_observer,
                    method=method,
                    progress=progress,
                )
                self._commit_success(progress, telemetry)
                return output
            finally:
                self._lock.release()
        except Exception as error:
            telemetry = self._make_error_telemetry(
                error,
                hidden_states=hidden_states,
                progress=progress,
            )
            self._commit_error(progress, telemetry)
            raise

    def _execute_locked(
        self,
        hidden_states: Any,
        route_inputs: tuple[Any, ...],
        *,
        phase: str,
        group_size: int,
        num_token_non_padded: int | None,
        debug_observer: Callable[[str, dict[str, object]], None] | None,
        method: str,
        progress: _RequestProgress,
    ) -> tuple[Any, GGUFEagerBridgeTelemetry]:
        original_device = _device(hidden_states)
        original_dtype = _dtype(hidden_states)
        original_shape = _shape(hidden_states)
        progress.input_device = str(original_device)
        progress.input_dtype = str(original_dtype)
        names = (
            ("hidden_states", "router_logits")
            if method == "forward"
            else (
                "hidden_states",
                "topk_weights",
                "topk_ids",
            )
        )
        inputs = (hidden_states, *route_inputs)
        cpu_inputs: list[Any] = []
        transfers: list[str] = []
        transfer_bytes = 0
        for name, value in zip(names, inputs, strict=True):
            if _device_type(_device(value)) == "cpu":
                cpu_value = value
            else:
                progress.transfer_attempted = True
                cpu_value = self._transfer.to_cpu(value, name=name)
                completed_transfer = f"{name}:to_cpu"
                transfers.append(completed_transfer)
                transfer_bytes += _element_bytes(value) * self._numel(value)
                progress.transfers.append(completed_transfer)
                progress.transfer_bytes += _element_bytes(value) * self._numel(value)
            if _device_type(_device(cpu_value)) != "cpu":
                raise RuntimeError(f"transfer seam returned non-CPU {name}: {_device(cpu_value)}")
            if _shape(cpu_value) != _shape(value):
                raise RuntimeError(
                    "transfer seam changed "
                    f"{name} shape from {_shape(value)} to {_shape(cpu_value)}"
                )
            if _dtype(cpu_value) != _dtype(value):
                raise RuntimeError(
                    "transfer seam changed "
                    f"{name} dtype from {_dtype(value)} to {_dtype(cpu_value)}"
                )
            cpu_inputs.append(cpu_value)

        adapter = getattr(self._layer, method)
        progress.adapter_started = True
        progress.cpu_execution_count = 1
        if method == "forward":
            result = adapter(
                cpu_inputs[0],
                router_logits=cpu_inputs[1],
                debug_observer=debug_observer,
                num_token_non_padded=num_token_non_padded,
                phase=phase,
                group_size=group_size,
            )
        else:
            result = adapter(
                cpu_inputs[0],
                cpu_inputs[1],
                cpu_inputs[2],
                num_token_non_padded=num_token_non_padded,
                debug_observer=debug_observer,
                phase=phase,
                group_size=group_size,
            )
        progress.adapter_completed = True
        progress.output_device = str(_device(result))
        progress.output_dtype = str(_dtype(result))
        if _device_type(_device(result)) != "cpu":
            raise RuntimeError(f"CPU GGUF adapter returned a non-CPU result: {_device(result)}")
        if _shape(result) != original_shape:
            raise RuntimeError(
                f"CPU GGUF adapter changed output shape from {original_shape} to {_shape(result)}"
            )
        if _dtype(result) != original_dtype:
            raise RuntimeError(
                f"CPU GGUF adapter changed output dtype from {original_dtype} to {_dtype(result)}"
            )

        if _device_type(original_device) == "cpu":
            output = result
        else:
            progress.transfer_attempted = True
            output = self._transfer.to_device(
                result,
                device=original_device,
                dtype=original_dtype,
                name="routed_result",
            )
            completed_transfer = "routed_result:to_device"
            transfers.append(completed_transfer)
            transfer_bytes += _element_bytes(result) * self._numel(result)
            progress.transfers.append(completed_transfer)
            progress.transfer_bytes += _element_bytes(result) * self._numel(result)
            progress.output_device = str(_device(output))
            progress.output_dtype = str(_dtype(output))
            if _device_type(_device(output)) != _device_type(original_device):
                raise RuntimeError(
                    "transfer seam returned a result on the wrong device: "
                    f"got {_device(output)}, expected {original_device}"
                )
            if _dtype(output) != original_dtype or _shape(output) != original_shape:
                raise RuntimeError("transfer seam changed routed result shape or dtype")

        telemetry = GGUFEagerBridgeTelemetry(
            backend=self._BACKEND,
            layer_id=getattr(self._layer, "layer_id", None),
            phase=phase,
            group_size=int(group_size),
            input_device=str(original_device),
            output_device=str(_device(output)),
            input_dtype=str(original_dtype),
            output_dtype=str(_dtype(output)),
            transfer_path="cpu_direct" if not transfers else "eager_cpu_round_trip",
            transfers=tuple(transfers),
            transfer_count=len(transfers),
            transfer_bytes=transfer_bytes,
            cpu_execution_count=progress.cpu_execution_count,
            adapter_started=progress.adapter_started,
            adapter_completed=progress.adapter_completed,
            adapter_telemetry=self._adapter_telemetry(progress.adapter_started),
            fallback_reason=(None if not transfers else "explicit_eager_cpu_bridge"),
            elapsed_ns=time.perf_counter_ns() - progress.started,
        )
        return output, telemetry

    @staticmethod
    def _validate_request_mode(
        phase: str,
        group_size: int,
        *,
        graph_capture: bool,
        workspace: Any,
    ) -> None:
        if phase not in ("prefill", "decode"):
            raise UnsupportedGGUFCpuConfiguration(
                f"GGUF eager bridge requires phase='prefill' or 'decode'; got {phase!r}"
            )
        if isinstance(group_size, bool) or not isinstance(group_size, Integral):
            raise UnsupportedGGUFCpuConfiguration(
                f"GGUF eager bridge requires group_size=1, got {group_size!r}"
            )
        if int(group_size) != 1:
            raise UnsupportedGGUFCpuConfiguration(
                f"GGUF eager bridge does not support grouped execution; group_size={group_size}"
            )
        if not isinstance(graph_capture, bool):
            raise UnsupportedGGUFCpuConfiguration(
                f"GGUF eager bridge graph_capture must be bool, got {graph_capture!r}"
            )
        if graph_capture:
            raise UnsupportedGGUFCpuConfiguration(
                "GGUF eager bridge cannot run during CUDA graph capture"
            )
        if workspace is not None:
            raise UnsupportedGGUFCpuConfiguration(
                "GGUF eager bridge does not accept a caller workspace"
            )

    def _begin_request(self) -> None:
        if self._closed:
            raise RuntimeError("GGUF eager bridge is closed")

    def _new_request_progress(
        self,
        *,
        started: int,
        phase: str | None,
        group_size: Any,
    ) -> _RequestProgress:
        with self._telemetry_lock:
            self._next_generation += 1
            generation = self._next_generation
            # A newly started request is the newest telemetry generation, even
            # while it is still validating or waiting for admission.
            self._last_telemetry = None
            self._last_error_telemetry = None
        return _RequestProgress(
            started=started,
            generation=generation,
            phase=phase,
            group_size=(int(group_size) if isinstance(group_size, Integral) else None),
        )

    def _commit_success(
        self,
        progress: _RequestProgress,
        telemetry: GGUFEagerBridgeTelemetry,
    ) -> None:
        with self._telemetry_lock:
            if progress.generation < self._committed_generation:
                return
            self._committed_generation = progress.generation
            self._last_telemetry = telemetry
            self._last_error_telemetry = None

    def _commit_error(
        self,
        progress: _RequestProgress,
        telemetry: GGUFEagerBridgeTelemetry,
    ) -> None:
        with self._telemetry_lock:
            if progress.generation < self._committed_generation:
                return
            self._committed_generation = progress.generation
            self._last_telemetry = None
            self._last_error_telemetry = telemetry

    def _reject(
        self,
        error: Exception,
        *,
        hidden_states: Any,
        phase: str | None,
        group_size: int | None,
    ) -> Any:
        started = time.perf_counter_ns()
        progress = self._new_request_progress(
            started=started,
            phase=phase,
            group_size=group_size,
        )
        try:
            self._begin_request()
        except Exception as request_error:
            error = request_error
        telemetry = self._make_error_telemetry(
            error,
            hidden_states=hidden_states,
            progress=progress,
        )
        self._commit_error(progress, telemetry)
        raise error

    def _make_error_telemetry(
        self,
        error: Exception,
        *,
        hidden_states: Any,
        progress: _RequestProgress,
    ) -> GGUFEagerBridgeTelemetry:
        input_device = progress.input_device
        if input_device is None:
            try:
                input_device = str(_device(hidden_states))
            except TypeError:
                input_device = None
        input_dtype = progress.input_dtype
        if input_dtype is None:
            try:
                input_dtype = str(_dtype(hidden_states))
            except TypeError:
                input_dtype = None
        if progress.transfers:
            transfer_path = "eager_cpu_round_trip"
        elif progress.transfer_attempted:
            transfer_path = "eager_cpu_transfer"
        elif progress.adapter_started:
            transfer_path = "cpu_direct"
        else:
            transfer_path = "rejected"
        return GGUFEagerBridgeTelemetry(
            backend=self._BACKEND,
            layer_id=getattr(self._layer, "layer_id", None),
            phase=progress.phase,
            group_size=progress.group_size,
            input_device=input_device,
            output_device=progress.output_device,
            input_dtype=input_dtype,
            output_dtype=progress.output_dtype,
            transfer_path=transfer_path,
            transfers=tuple(progress.transfers),
            transfer_count=len(progress.transfers),
            transfer_bytes=progress.transfer_bytes,
            cpu_execution_count=progress.cpu_execution_count,
            adapter_started=progress.adapter_started,
            adapter_completed=progress.adapter_completed,
            adapter_telemetry=self._adapter_telemetry(progress.adapter_started),
            fallback_reason=(
                "bridge_validation"
                if not progress.transfer_attempted and not progress.adapter_started
                else "eager_cpu_bridge_error"
            ),
            error=type(error).__name__,
            error_detail=str(error),
            elapsed_ns=time.perf_counter_ns() - progress.started,
        )

    def _adapter_telemetry(self, adapter_called: bool = False) -> Mapping[str, object] | None:
        if not adapter_called:
            return None
        value = getattr(self._layer, "last_error_telemetry", None)
        if value is None:
            value = getattr(self._layer, "last_telemetry", None)
        if value is None:
            return None
        if hasattr(value, "as_dict"):
            return value.as_dict()
        if isinstance(value, Mapping):
            return dict(value)
        return {"value": value}

    @staticmethod
    def _numel(value: Any) -> int:
        try:
            return max(0, int(value.numel()))
        except (AttributeError, TypeError, ValueError):
            try:
                result = 1
                for size in value.shape:
                    result *= int(size)
                return max(0, result)
            except (AttributeError, TypeError, ValueError):
                return 0

    def close(self) -> None:
        """Close admission without closing the borrowed layer or its bundle."""
        if self._closed:
            return
        if not self._lock.acquire(blocking=False):
            raise GGUFEagerBridgeBusy("cannot close GGUF eager bridge while a request is in flight")
        try:
            self._closed = True
            self._frozen = True
        finally:
            self._lock.release()

    def freeze_admission(self) -> None:
        """Freeze new requests without closing the bridge or wrapped layer."""
        if self._closed:
            raise RuntimeError("GGUF eager bridge is closed")
        if not self._lock.acquire(blocking=False):
            raise GGUFEagerBridgeBusy(
                "cannot freeze GGUF eager bridge while a request is in flight"
            )
        try:
            if self._closed:
                raise RuntimeError("GGUF eager bridge is closed")
            self._frozen = True
        finally:
            self._lock.release()

    def unfreeze_admission(self) -> None:
        """Undo a freeze after a multi-bridge lifecycle transaction aborts."""
        with self._lock:
            if not self._closed:
                self._frozen = False

    def rollback_close(self) -> None:
        """Restore an attached bridge after a later close in a transaction fails."""
        with self._lock:
            if self._closed:
                self._closed = False
            self._frozen = True

    def ensure_quiescent(self) -> None:
        """Fail if a request is in flight without changing admission state."""
        if self._closed:
            return
        if not self._lock.acquire(blocking=False):
            raise GGUFEagerBridgeBusy(
                "cannot quiesce GGUF eager bridge while a request is in flight"
            )
        self._lock.release()

    def __enter__(self) -> GGUFCpuEagerBridge:
        if self._closed:
            raise RuntimeError("GGUF eager bridge is closed")
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


# Descriptive aliases keep the public seam discoverable for Qwen callers while the
# implementation remains model-neutral and does not import a model class.
QwenGGUFCpuEagerBridge = GGUFCpuEagerBridge
QwenGGUFCpuDeviceBridge = GGUFCpuEagerBridge
GGUFTransferTelemetry = GGUFEagerBridgeTelemetry


__all__ = [
    "BlockingTensorTransfer",
    "EagerTransferSeam",
    "GGUFCpuEagerBridge",
    "GGUFEagerBridgeBusy",
    "GGUFEagerBridgeError",
    "GGUFEagerBridgeTelemetry",
    "GGUFTransferTelemetry",
    "QwenGGUFCpuDeviceBridge",
    "QwenGGUFCpuEagerBridge",
]
