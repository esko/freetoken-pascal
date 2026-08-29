"""H0 tests for the explicit eager device bridge."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread

import numpy as np
import pytest
from freetoken.moe.gguf_transfer import (
    BlockingTensorTransfer,
    GGUFCpuEagerBridge,
    GGUFEagerBridgeBusy,
)


@dataclass(frozen=True)
class _Device:
    type: str
    index: int | None = None

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


class _Tensor:
    def __init__(self, values, *, device: _Device, dtype: str = "float32") -> None:
        self.values = np.array(values, dtype=np.float32, copy=True)
        self.device = device
        self.dtype = dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def numel(self) -> int:
        return int(self.values.size)

    def element_size(self) -> int:
        return {"float16": 2, "float32": 4, "int32": 4}[self.dtype]

    def copy(self, *, device: _Device | None = None, dtype: str | None = None) -> _Tensor:
        return _Tensor(
            self.values,
            device=self.device if device is None else device,
            dtype=self.dtype if dtype is None else dtype,
        )


class _Transfer:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def to_cpu(self, tensor: _Tensor, *, name: str) -> _Tensor:
        self.events.append(("to_cpu", name))
        return tensor.copy(device=_Device("cpu"))

    def to_device(
        self,
        tensor: _Tensor,
        *,
        device: _Device,
        dtype: str,
        name: str,
    ) -> _Tensor:
        self.events.append(("to_device", name, device, dtype))
        return tensor.copy(device=device, dtype=dtype)


class _FailingTransfer(_Transfer):
    def __init__(self, events: list[object], *, fail_on: int) -> None:
        super().__init__(events)
        self.fail_on = fail_on
        self.calls = 0

    def to_cpu(self, tensor: _Tensor, *, name: str) -> _Tensor:
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError(f"transfer failure {name}")
        return super().to_cpu(tensor, name=name)


class _Layer:
    layer_id = 3
    closed = False

    def __init__(self, events: list[object] | None = None) -> None:
        self.events = events if events is not None else []
        self.forward_calls = 0
        self.routed_calls = 0
        self.block = False
        self.fail = False
        self.entered = Event()
        self.release = Event()
        self.last_telemetry = {"backend": "fake_cpu", "thread_count": 1}
        self.last_error_telemetry = None
        self.host_weight_telemetry = {"kernel_census": ["fake"]}

    def forward(self, hidden, *, router_logits, debug_observer, **kwargs):
        self.events.append(("adapter", "forward", kwargs["phase"]))
        self.forward_calls += 1
        if self.block:
            self.entered.set()
            assert self.release.wait(2)
        if self.fail:
            raise RuntimeError("adapter failure after D2H")
        if debug_observer is not None:
            debug_observer("router", {"ids": router_logits, "weights": router_logits})
        return _Tensor(hidden.values + 1, device=_Device("cpu"), dtype=hidden.dtype)

    def routed_forward(self, hidden, weights, ids, *, debug_observer, **kwargs):
        self.events.append(("adapter", "routed_forward", kwargs["phase"]))
        self.routed_calls += 1
        if self.block:
            self.entered.set()
            assert self.release.wait(2)
        if self.fail:
            raise RuntimeError("adapter failure after D2H")
        if debug_observer is not None:
            debug_observer("router", {"ids": ids, "weights": weights})
        return _Tensor(
            hidden.values + weights.values.sum(),
            device=_Device("cpu"),
            dtype=hidden.dtype,
        )


def _inputs(device: _Device | None = None) -> tuple[_Tensor, _Tensor, _Tensor]:
    if device is None:
        device = _Device("cpu")
    hidden = _Tensor([[1, 2]], device=device)
    routes = _Tensor([[3, 4]], device=device)
    ids = _Tensor([[0, 1]], device=device, dtype="int32")
    return hidden, routes, ids


def test_cpu_direct_path_calls_adapter_once_without_transfer() -> None:
    events: list[object] = []
    layer = _Layer(events)
    bridge = GGUFCpuEagerBridge(layer, transfer=_Transfer(events))
    hidden, weights, ids = _inputs()

    result = bridge.routed_forward(
        hidden,
        weights,
        ids,
        phase="decode",
        debug_observer=lambda *_args: None,
    )

    assert result.device.type == "cpu"
    assert result is not hidden
    assert layer.routed_calls == 1
    assert layer.forward_calls == 0
    assert events == [("adapter", "routed_forward", "decode")]
    assert bridge.last_telemetry is not None
    assert bridge.last_telemetry.transfer_path == "cpu_direct"
    assert bridge.last_telemetry.cpu_execution_count == 1


def test_device_forward_transfers_hidden_and_router_in_order_and_never_routes_twice() -> None:
    events: list[object] = []
    layer = _Layer(events)
    bridge = GGUFCpuEagerBridge(layer, transfer=_Transfer(events))
    hidden, logits, _ids = _inputs(_Device("cuda", 0))
    observed: list[object] = []

    result = bridge.forward(
        hidden,
        router_logits=logits,
        debug_observer=lambda name, payload: observed.append((name, payload)),
        phase="decode",
    )

    assert result.device == hidden.device
    assert result.dtype == hidden.dtype
    assert result.shape == hidden.shape
    assert result is not hidden
    assert layer.forward_calls == 1
    assert layer.routed_calls == 0
    assert events == [
        ("to_cpu", "hidden_states"),
        ("to_cpu", "router_logits"),
        ("adapter", "forward", "decode"),
        ("to_device", "routed_result", hidden.device, hidden.dtype),
    ]
    assert observed and observed[0][0] == "router"
    assert bridge.last_telemetry is not None
    assert bridge.last_telemetry.transfers == (
        "hidden_states:to_cpu",
        "router_logits:to_cpu",
        "routed_result:to_device",
    )
    assert bridge.last_telemetry.transfer_count == 3


def test_device_routed_path_transfers_all_routes_and_preserves_output_independence() -> None:
    events: list[object] = []
    layer = _Layer(events)
    bridge = GGUFCpuEagerBridge(layer, transfer=_Transfer(events))
    hidden, weights, ids = _inputs(_Device("cuda", 1))
    result = bridge.routed_forward(hidden, weights, ids, phase="decode")

    assert [item[1] for item in events if item[0] == "to_cpu"] == [
        "hidden_states",
        "topk_weights",
        "topk_ids",
    ]
    assert layer.routed_calls == 1
    assert layer.forward_calls == 0
    result.values[0, 0] = 999
    assert hidden.values[0, 0] != 999


def test_adapter_failure_after_three_completed_copies_preserves_progress_telemetry() -> None:
    events: list[object] = []
    layer = _Layer(events)
    layer.fail = True
    bridge = GGUFCpuEagerBridge(layer, transfer=_Transfer(events))
    hidden, weights, ids = _inputs(_Device("cuda", 0))

    with pytest.raises(RuntimeError, match="adapter failure"):
        bridge.routed_forward(hidden, weights, ids, phase="decode")

    telemetry = bridge.last_error_telemetry
    assert telemetry is not None
    assert telemetry.transfer_path == "eager_cpu_round_trip"
    assert telemetry.transfers == (
        "hidden_states:to_cpu",
        "topk_weights:to_cpu",
        "topk_ids:to_cpu",
    )
    assert telemetry.transfer_count == 3
    assert telemetry.transfer_bytes == 24
    assert telemetry.cpu_execution_count == 1
    assert telemetry.adapter_started is True
    assert telemetry.adapter_completed is False
    assert layer.routed_calls == 1
    assert bridge.last_telemetry is None


def test_mid_transfer_failure_preserves_only_completed_copy_telemetry() -> None:
    events: list[object] = []
    layer = _Layer(events)
    bridge = GGUFCpuEagerBridge(layer, transfer=_FailingTransfer(events, fail_on=2))
    hidden, weights, ids = _inputs(_Device("cuda", 0))

    with pytest.raises(RuntimeError, match="transfer failure topk_weights"):
        bridge.routed_forward(hidden, weights, ids, phase="decode")

    telemetry = bridge.last_error_telemetry
    assert telemetry is not None
    assert telemetry.transfer_path == "eager_cpu_round_trip"
    assert telemetry.transfers == ("hidden_states:to_cpu",)
    assert telemetry.transfer_count == 1
    assert telemetry.transfer_bytes == 8
    assert telemetry.cpu_execution_count == 0
    assert telemetry.adapter_started is False
    assert telemetry.adapter_completed is False
    assert layer.routed_calls == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"phase": "prefill"}, "decode-only"),
        ({"phase": "decode", "group_size": 2}, "grouped"),
        ({"phase": "decode", "graph_capture": True}, "graph capture"),
        ({"phase": "decode", "workspace": object()}, "workspace"),
    ],
)
def test_unsupported_modes_are_rejected_before_transfer(kwargs, message: str) -> None:
    events: list[object] = []
    bridge = GGUFCpuEagerBridge(_Layer(events), transfer=_Transfer(events))
    hidden, weights, ids = _inputs(_Device("cuda", 0))

    with pytest.raises(ValueError, match=message):
        bridge.routed_forward(hidden, weights, ids, **kwargs)
    assert events == []
    assert bridge.last_telemetry is None
    assert bridge.last_error_telemetry is not None
    assert bridge.last_error_telemetry.transfer_count == 0


def test_phase_is_required_and_missing_router_is_rejected_without_transfer() -> None:
    events: list[object] = []
    bridge = GGUFCpuEagerBridge(_Layer(events), transfer=_Transfer(events))
    hidden, _weights, _ids = _inputs(_Device("cuda", 0))

    with pytest.raises(TypeError, match="phase"):
        bridge.routed_forward(hidden, _weights, _ids)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="router_logits"):
        bridge.forward(hidden, phase="decode")
    assert events == []
    assert bridge.last_error_telemetry is not None


def test_failed_request_clears_success_telemetry_and_does_not_return_stale_output() -> None:
    events: list[object] = []
    layer = _Layer(events)
    bridge = GGUFCpuEagerBridge(layer, transfer=_Transfer(events))
    hidden, weights, ids = _inputs()
    bridge.routed_forward(hidden, weights, ids, phase="decode")
    assert bridge.last_telemetry is not None

    with pytest.raises(ValueError, match="decode-only"):
        bridge.routed_forward(hidden, weights, ids, phase="prefill")
    assert bridge.last_telemetry is None
    assert bridge.last_error_telemetry is not None
    assert bridge.last_error_telemetry.error == "UnsupportedGGUFCpuConfiguration"


def test_close_rejects_in_flight_request_and_never_closes_borrowed_layer() -> None:
    layer = _Layer()
    layer.block = True
    bridge = GGUFCpuEagerBridge(layer)
    hidden, weights, ids = _inputs()
    worker = Thread(
        target=lambda: bridge.routed_forward(hidden, weights, ids, phase="decode"),
        daemon=True,
    )
    worker.start()
    assert layer.entered.wait(2)

    with pytest.raises(GGUFEagerBridgeBusy, match="busy"):
        bridge.routed_forward(hidden, weights, ids, phase="decode")
    with pytest.raises(GGUFEagerBridgeBusy, match="in flight"):
        bridge.close()
    layer.release.set()
    worker.join(2)
    assert not worker.is_alive()
    bridge.close()
    bridge.close()
    assert bridge.closed
    assert layer.closed is False
    with pytest.raises(RuntimeError, match="closed"):
        bridge.routed_forward(hidden, weights, ids, phase="decode")


def test_close_winning_validation_admission_race_cannot_start_request() -> None:
    validated = Event()
    release_validation = Event()

    class _ValidationGateTensor(_Tensor):
        def __init__(self, values, *, device: _Device) -> None:
            super().__init__(values, device=device)
            self._waited = False

        @property
        def shape(self) -> tuple[int, ...]:
            if not self._waited:
                self._waited = True
                validated.set()
                assert release_validation.wait(2)
            return super().shape

    events: list[object] = []
    layer = _Layer(events)
    bridge = GGUFCpuEagerBridge(layer, transfer=_Transfer(events))
    hidden = _ValidationGateTensor([[1, 2]], device=_Device("cuda", 0))
    weights = _Tensor([[3, 4]], device=_Device("cuda", 0))
    ids = _Tensor([[0, 1]], device=_Device("cuda", 0), dtype="int32")
    failures: list[Exception] = []

    def run() -> None:
        try:
            bridge.routed_forward(hidden, weights, ids, phase="decode")
        except Exception as error:
            failures.append(error)

    worker = Thread(target=run, daemon=True)
    worker.start()
    assert validated.wait(2)
    bridge.close()
    release_validation.set()
    worker.join(2)

    assert not worker.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert str(failures[0]) == "GGUF eager bridge is closed"
    assert layer.routed_calls == 0
    assert events == []
    telemetry = bridge.last_error_telemetry
    assert telemetry is not None
    assert telemetry.transfer_path == "rejected"
    assert telemetry.transfer_count == 0
    assert telemetry.cpu_execution_count == 0


def test_default_transfer_uses_blocking_to_without_non_blocking_argument() -> None:
    # The real method is intentionally tiny and dependency-free; this test
    # verifies the seam can be replaced independently of the bridge.
    class _TensorWithTo(_Tensor):
        def to(self, *, device, dtype=None):
            assert device == "cpu"
            assert dtype is None
            return self.copy(device=_Device("cpu"))

    tensor = _TensorWithTo([[1]], device=_Device("cuda", 0))
    result = BlockingTensorTransfer().to_cpu(tensor, name="hidden_states")
    assert result.device.type == "cpu"
