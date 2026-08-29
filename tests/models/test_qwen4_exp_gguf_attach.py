"""H0 tests for explicit borrowed GGUF expert attachment to Qwen4-Exp."""

# The model package is imported only after importorskip below so the H0 test
# remains collectable in the repository's torch-free baseline environment.
# ruff: noqa: E402, I001

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import ClassVar

import pytest

torch = pytest.importorskip("torch")

from freetoken.layers import BaseOP, OPList
from freetoken.moe.cpu_abi import CpuExpertDescriptor, CpuExpertLayout
from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle
from freetoken.models.qwen4_exp.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpModel,
    _MoeExecutionContext,
    _SparseMoE,
)


class _FakeExperts(BaseOP):
    def __init__(self, value: float):
        self.weights = torch.tensor([value])

    def forward(self, *args, **kwargs):
        del args, kwargs
        return torch.zeros(1, 4)


class _FakeMlp(BaseOP):
    def __init__(self, experts):
        self.experts = experts

    def forward(self, hidden, debug_observer=None):
        del debug_observer
        return self.experts.forward(hidden_states=hidden, router_logits=None)


class _FakeLayer(BaseOP):
    def __init__(self, layer_id: int, experts):
        self._layer_id = layer_id
        self.mlp = _FakeMlp(experts)

    def forward(self, hidden, debug_observer=None):
        del debug_observer
        return self.mlp.forward(hidden)


def _config(*, num_layers: int = 3, **changes):
    values = {
        "num_layers": num_layers,
        "num_experts": 3,
        "num_experts_per_tok": 2,
        "hidden_size": 4,
        "moe_intermediate_size": 2,
        "hidden_act": "silu",
        "norm_topk_prob": False,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _descriptor(
    layer_id: int,
    projection: str,
    *,
    experts: int = 3,
    hidden: int = 4,
    intermediate: int = 2,
):
    if projection in {"gate", "up"}:
        input_dim, output_dim = hidden, intermediate
    else:
        input_dim, output_dim = intermediate, hidden
    row_bytes = 32
    expert_stride = output_dim * row_bytes
    return CpuExpertDescriptor(
        layer_id=layer_id,
        projection=projection,
        quant_type=0,
        quant_name="Q4_K",
        num_experts=experts,
        output_dim=output_dim,
        input_dim=input_dim,
        rows_per_expert=output_dim,
        row_stride_bytes=row_bytes,
        expert_stride_bytes=expert_stride,
        tensor_bytes=experts * expert_stride,
    )


def _bundle(
    *,
    layers=(0, 1, 2),
    top_k: int = 2,
    experts: int = 3,
    hidden: int = 4,
    intermediate: int = 2,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
):
    descriptors = tuple(
        _descriptor(
            layer_id,
            projection,
            experts=experts,
            hidden=hidden,
            intermediate=intermediate,
        )
        for layer_id in layers
        for projection in ("gate", "up", "down")
    )
    bundle = object.__new__(QwenGGUFCpuExpertBundle)
    bundle.layout = CpuExpertLayout(descriptors=descriptors, top_k=top_k)
    bundle.executor = SimpleNamespace(
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    bundle._closed = False
    return bundle


def _model(config=None):
    config = config or _config()
    model = object.__new__(Qwen4ExpModel)
    model._config = config
    originals = tuple(_FakeExperts(float(index)) for index in range(config.num_layers))
    model.layers = OPList([_FakeLayer(index, experts) for index, experts in enumerate(originals)])
    model._gguf_attachment_lock = threading.RLock()
    return model, originals


class _FakeAdapter:
    created: ClassVar[list[_FakeAdapter]] = []
    fail_layer: int | None = None

    def __init__(self, bundle, **kwargs):
        layer_id = kwargs["layer_id"]
        if layer_id == self.fail_layer:
            raise RuntimeError(f"injected layer {layer_id} failure")
        self.bundle = bundle
        self.__dict__.update(kwargs)
        self.created.append(self)


class _FakeBridge:
    """Small lifecycle double for the eager attachment transaction."""

    requires_moe_execution_context = True
    created: ClassVar[list[_FakeBridge]] = []
    fail_layer: int | None = None
    busy_layer: int | None = None
    fail_close_layer: int | None = None
    freeze_hook: ClassVar[object | None] = None

    def __init__(self, layer, *, transfer=None, cache_size=0, tp_size=1):
        del cache_size, tp_size
        layer_id = layer.layer_id
        if layer_id == self.fail_layer:
            raise RuntimeError(f"injected bridge {layer_id} failure")
        self.layer = layer
        self.layer_id = layer_id
        self.transfer = transfer
        self.closed = False
        self.frozen = False
        self._request_lock = threading.Lock()
        self.request_started = threading.Event()
        self.request_release = threading.Event()
        self.request_error = None
        self.created.append(self)

    def freeze_admission(self):
        if self.layer_id == self.busy_layer or self.frozen:
            raise RuntimeError("eager bridge is busy")
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("eager bridge is busy")
        self._request_lock.release()
        self.frozen = True
        if self.freeze_hook is not None:
            self.freeze_hook(self)

    def unfreeze_admission(self):
        self.frozen = False

    def close(self):
        if self.layer_id == self.fail_close_layer:
            raise RuntimeError("injected bridge close failure")
        self.closed = True

    def rollback_close(self):
        self.closed = False

    def request(self):
        if self.frozen:
            raise RuntimeError("eager bridge admission is frozen")
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("eager bridge is busy")
        try:
            self.request_started.set()
            if not self.request_release.wait(timeout=2):
                raise RuntimeError("direct request did not drain")
        except BaseException as error:
            self.request_error = error
            raise
        finally:
            self._request_lock.release()

    @property
    def host_weight_telemetry(self):
        return {"layer_id": self.layer_id, "closed": self.closed}


@pytest.fixture
def fake_adapter(monkeypatch):
    from freetoken.models.qwen4_exp import gguf_attach as attach_module

    _FakeAdapter.created = []
    _FakeAdapter.fail_layer = None
    monkeypatch.setattr(attach_module, "QwenGGUFCpuMoELayer", _FakeAdapter)
    return _FakeAdapter


@pytest.fixture
def fake_eager_attachment(monkeypatch):
    from freetoken.models.qwen4_exp import gguf_attach as attach_module

    _FakeAdapter.created = []
    _FakeAdapter.fail_layer = None
    _FakeBridge.created = []
    _FakeBridge.fail_layer = None
    _FakeBridge.busy_layer = None
    _FakeBridge.fail_close_layer = None
    _FakeBridge.freeze_hook = None
    monkeypatch.setattr(attach_module, "QwenGGUFCpuMoELayer", _FakeAdapter)
    monkeypatch.setattr(attach_module, "GGUFCpuEagerBridge", _FakeBridge)
    return _FakeBridge


def test_attach_builds_every_adapter_before_swapping_and_detach_restores_state(
    fake_adapter,
):
    model, originals = _model()
    bundle = _bundle()
    before = model.state_dict()

    model.attach_gguf_cpu_expert_bundle(bundle)

    adapters = tuple(layer.mlp.experts for layer in model.layers.op_list)
    assert adapters == tuple(fake_adapter.created)
    assert [adapter.layer_id for adapter in adapters] == [0, 1, 2]
    with pytest.raises(RuntimeError, match="already attached"):
        model.attach_gguf_cpu_expert_bundle(bundle)

    model.detach_gguf_cpu_expert_bundle()
    assert tuple(layer.mlp.experts for layer in model.layers.op_list) == originals
    assert bundle.closed is False
    after = model.state_dict()
    assert before.keys() == after.keys()
    for name in before:
        assert after[name] is before[name]

    # Detach is deliberately idempotent and does not touch the borrowed bundle.
    model.detach_gguf_cpu_expert_bundle()
    assert bundle.closed is False


def test_middle_adapter_failure_rolls_back_without_partial_mutation(fake_adapter):
    model, originals = _model()
    before = model.state_dict()
    fake_adapter.fail_layer = 1

    with pytest.raises(RuntimeError, match="injected layer 1 failure"):
        model.attach_gguf_cpu_expert_bundle(_bundle())

    assert tuple(layer.mlp.experts for layer in model.layers.op_list) == originals
    assert model.state_dict().keys() == before.keys()


def test_state_dict_serializes_before_concurrent_attachment(fake_adapter, monkeypatch):
    model, originals = _model()
    entered = threading.Event()
    release = threading.Event()
    attach_finished = threading.Event()
    captured = {}
    base_state_dict = BaseOP.state_dict

    def blocked_state_dict(owner, **kwargs):
        if owner is model:
            entered.set()
            assert release.wait(timeout=2)
        return base_state_dict(owner, **kwargs)

    monkeypatch.setattr(BaseOP, "state_dict", blocked_state_dict)
    state_thread = threading.Thread(
        target=lambda: captured.setdefault("result", model.state_dict(prefix="snapshot"))
    )
    state_thread.start()
    assert entered.wait(timeout=2)

    attach_thread = threading.Thread(
        target=lambda: (model.attach_gguf_cpu_expert_bundle(_bundle()), attach_finished.set())
    )
    attach_thread.start()
    assert not attach_finished.wait(timeout=0.1)

    release.set()
    state_thread.join(timeout=2)
    attach_thread.join(timeout=2)
    assert not state_thread.is_alive()
    assert not attach_thread.is_alive()
    state = captured["result"]
    for index, original in enumerate(originals):
        assert state[f"snapshot.layers.{index}.mlp.experts.weights"] is original.weights
    assert tuple(layer.mlp.experts for layer in model.layers.op_list) != originals
    model.detach_gguf_cpu_expert_bundle()


@pytest.mark.parametrize(
    ("bundle_kwargs", "message"),
    [
        ({"layers": (0, 1)}, "layer IDs"),
        ({"top_k": 1}, "top_k"),
        ({"experts": 4}, "num_experts"),
        ({"hidden": 5}, "geometry"),
        ({"activation": "gelu"}, "activation"),
        ({"apply_router_weight_on_input": True}, "router-weight"),
    ],
)
def test_attach_rejects_bundle_metadata_before_mutation(fake_adapter, bundle_kwargs, message):
    model, originals = _model()

    with pytest.raises(ValueError, match=message):
        model.attach_gguf_cpu_expert_bundle(_bundle(**bundle_kwargs))

    assert tuple(layer.mlp.experts for layer in model.layers.op_list) == originals
    assert fake_adapter.created == []


def test_causal_lm_delegates_borrowed_attachment_lifecycle():
    model, _ = _model()
    wrapper = object.__new__(Qwen4ExpForCausalLM)
    wrapper.model = model
    bundle = _bundle()
    calls = []

    model.attach_gguf_cpu_expert_bundle = lambda value: calls.append(("attach", value))
    model.detach_gguf_cpu_expert_bundle = lambda: calls.append(("detach",))

    wrapper.attach_gguf_cpu_expert_bundle(bundle)
    wrapper.detach_gguf_cpu_expert_bundle()
    assert calls == [("attach", bundle), ("detach",)]


def test_attach_rejects_closed_bundle_and_tp2_before_construction(fake_adapter, monkeypatch):
    model, originals = _model()
    bundle = _bundle()
    bundle._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        model.attach_gguf_cpu_expert_bundle(bundle)
    assert tuple(layer.mlp.experts for layer in model.layers.op_list) == originals

    bundle._closed = False
    from freetoken.distributed import info

    monkeypatch.setattr(info, "_TP_INFO", SimpleNamespace(rank=0, size=2))
    with pytest.raises(ValueError, match="TP=1"):
        model.attach_gguf_cpu_expert_bundle(bundle)
    assert tuple(layer.mlp.experts for layer in model.layers.op_list) == originals


def test_eager_attach_builds_bridges_before_swap_and_detach_closes_only_wrappers(
    fake_eager_attachment,
):
    model, originals = _model()
    bundle = _bundle()
    before = model.state_dict()
    transfer = object()

    model.attach_gguf_cpu_eager_bridge(bundle, transfer=transfer)

    bridges = tuple(layer.mlp.experts for layer in model.layers.op_list)
    assert bridges == tuple(fake_eager_attachment.created)
    assert all(bridge.transfer is transfer for bridge in bridges)
    assert model.gguf_cpu_expert_telemetry() == {
        index: {"layer_id": index, "closed": False} for index in range(3)
    }

    model.detach_gguf_cpu_expert_bundle()
    assert tuple(layer.mlp.experts for layer in model.layers.op_list) == originals
    assert all(bridge.closed for bridge in bridges)
    assert bundle.closed is False
    after = model.state_dict()
    assert after.keys() == before.keys()
    for name in before:
        assert after[name] is before[name]


def test_eager_middle_bridge_failure_closes_created_wrappers_without_mutation(
    fake_eager_attachment,
):
    model, originals = _model()
    fake_eager_attachment.fail_layer = 1

    with pytest.raises(RuntimeError, match="injected bridge 1 failure"):
        model.attach_gguf_cpu_eager_bridge(_bundle())

    assert tuple(layer.mlp.experts for layer in model.layers.op_list) == originals
    assert [bridge.closed for bridge in fake_eager_attachment.created] == [True]


def test_eager_detach_busy_preflight_preserves_attachment(fake_eager_attachment):
    model, originals = _model()
    model.attach_gguf_cpu_eager_bridge(_bundle())
    fake_eager_attachment.busy_layer = 1

    with pytest.raises(RuntimeError, match="busy"):
        model.detach_gguf_cpu_expert_bundle()

    assert tuple(layer.mlp.experts for layer in model.layers.op_list) != originals
    assert [bridge.closed for bridge in fake_eager_attachment.created] == [False, False, False]
    assert [bridge.frozen for bridge in fake_eager_attachment.created] == [False, False, False]


def test_eager_detach_freezes_all_bridges_before_direct_request_can_start(
    fake_eager_attachment,
):
    model, originals = _model()
    model.attach_gguf_cpu_eager_bridge(_bundle())
    bridges = fake_eager_attachment.created
    installed = tuple(layer.mlp.experts for layer in model.layers.op_list)

    request_thread = None

    def start_request_between_freezes(bridge):
        nonlocal request_thread
        if bridge.layer_id == 0:
            request_thread = threading.Thread(target=bridges[1].request)
            request_thread.start()
            assert bridges[1].request_started.wait(timeout=2)

    fake_eager_attachment.freeze_hook = staticmethod(start_request_between_freezes)

    try:
        with pytest.raises(RuntimeError, match="busy"):
            model.detach_gguf_cpu_expert_bundle()
    finally:
        bridges[1].request_release.set()
        if request_thread is not None:
            request_thread.join(timeout=2)
            assert not request_thread.is_alive()
            assert bridges[1].request_error is None

    assert tuple(layer.mlp.experts for layer in model.layers.op_list) == installed
    assert tuple(layer.mlp.experts for layer in model.layers.op_list) != originals
    assert [bridge.closed for bridge in bridges] == [False, False, False]
    assert [bridge.frozen for bridge in bridges] == [False, False, False]


def test_eager_detach_close_failure_rolls_back_closed_wrappers(
    fake_eager_attachment,
):
    model, originals = _model()
    model.attach_gguf_cpu_eager_bridge(_bundle())
    fake_eager_attachment.fail_close_layer = 1
    bridges = fake_eager_attachment.created
    installed = tuple(layer.mlp.experts for layer in model.layers.op_list)

    with pytest.raises(RuntimeError, match="injected bridge close failure"):
        model.detach_gguf_cpu_expert_bundle()

    assert tuple(layer.mlp.experts for layer in model.layers.op_list) == installed
    assert tuple(layer.mlp.experts for layer in model.layers.op_list) != originals
    assert [bridge.closed for bridge in bridges] == [False, False, False]
    assert [bridge.frozen for bridge in bridges] == [False, False, False]


def test_forward_checks_eager_mode_inside_attachment_lock(fake_adapter):
    model, _originals = _model()
    model._gguf_cpu_attachment = None
    mode_checked = threading.Event()
    release_mode_check = threading.Event()
    attach_finished = threading.Event()
    seen = []

    original_has_eager = Qwen4ExpModel._has_eager_gguf_attachment

    def delayed_mode_check():
        mode_checked.set()
        assert release_mode_check.wait(timeout=2)
        return original_has_eager(model)

    model._has_eager_gguf_attachment = delayed_mode_check
    model._forward_impl = lambda input_ids, *, eager: seen.append(eager)
    forward_thread = threading.Thread(
        target=Qwen4ExpModel.forward,
        args=(model, torch.zeros(1, 1, dtype=torch.int64)),
    )
    forward_thread.start()
    assert mode_checked.wait(timeout=2)

    attach_thread = threading.Thread(
        target=lambda: (model.attach_gguf_cpu_expert_bundle(_bundle()), attach_finished.set())
    )
    attach_thread.start()
    assert not attach_finished.wait(timeout=0.1)

    release_mode_check.set()
    forward_thread.join(timeout=2)
    attach_thread.join(timeout=2)
    assert not forward_thread.is_alive()
    assert not attach_thread.is_alive()
    assert seen == [False]
    model.detach_gguf_cpu_expert_bundle()


def test_load_state_dict_rejects_any_active_gguf_attachment(fake_adapter, fake_eager_attachment):
    for eager in (False, True):
        model, _originals = _model()
        if eager:
            model.attach_gguf_cpu_eager_bridge(_bundle())
        else:
            model.attach_gguf_cpu_expert_bundle(_bundle())

        with pytest.raises(RuntimeError, match=r"detach.*load_state_dict"):
            model.load_state_dict({})

        model.detach_gguf_cpu_expert_bundle()


def test_eager_execution_context_rejects_before_router_or_shared_work():
    events = []

    class _Gate:
        def forward(self, hidden):
            events.append("router")
            return torch.zeros(hidden.shape[0], 3)

    class _Shared:
        def forward(self, hidden):
            events.append("shared")
            return torch.zeros_like(hidden)

    class _SharedGate:
        def forward(self, hidden):
            events.append("shared_gate")
            return torch.zeros(hidden.shape[0], 1)

    class _EagerRouted:
        requires_moe_execution_context = True

        def forward(self, **kwargs):
            events.append("routed")
            return torch.zeros_like(kwargs["hidden_states"])

    moe = object.__new__(_SparseMoE)
    moe.gate = _Gate()
    moe.shared_expert = _Shared()
    moe.shared_expert_gate = _SharedGate()
    moe.experts = _EagerRouted()
    hidden = torch.zeros(1, 4)

    for context in (
        _MoeExecutionContext("prefill", 1, False),
        _MoeExecutionContext("decode", 2, False),
        _MoeExecutionContext("decode", 1, True),
    ):
        events.clear()
        with pytest.raises(ValueError, match="eager GGUF"):
            moe.forward(hidden, execution_context=context)
        assert events == []


def test_model_eager_context_derives_decode_batch_and_capture_state(monkeypatch):
    batch = SimpleNamespace(
        phase="decode",
        size=1,
        reqs=[SimpleNamespace(extend_len=1)],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False, raising=False)

    context = Qwen4ExpModel._eager_execution_context(batch)

    assert context.phase == "decode"
    assert context.group_size == 1
    assert context.graph_capture is False
    assert context.cache_size == 0
    assert context.workspace is None
    assert context.num_token_non_padded == 1


def test_eager_execution_context_is_explicit_and_preserves_shared_addition_order():
    events = []

    class _Gate:
        def forward(self, hidden):
            events.append("router")
            return torch.zeros(hidden.shape[0], 3)

    class _Shared:
        def forward(self, hidden):
            events.append("shared")
            return torch.full_like(hidden, 2)

    class _SharedGate:
        def forward(self, hidden):
            events.append("shared_gate")
            return torch.full((hidden.shape[0], 1), 100.0)

    class _EagerRouted:
        requires_moe_execution_context = True

        def forward(self, **kwargs):
            events.append("routed")
            assert kwargs["phase"] == "decode"
            assert kwargs["group_size"] == 1
            assert kwargs["graph_capture"] is False
            assert kwargs["workspace"] is None
            return torch.ones_like(kwargs["hidden_states"])

    moe = object.__new__(_SparseMoE)
    moe.gate = _Gate()
    moe.shared_expert = _Shared()
    moe.shared_expert_gate = _SharedGate()
    moe.experts = _EagerRouted()
    hidden = torch.zeros(1, 4)

    actual = moe.forward(hidden, execution_context=_MoeExecutionContext("decode", 1, False))

    assert events == ["router", "shared", "shared_gate", "routed"]
    assert torch.equal(actual, torch.full_like(hidden, 3))


def test_sparse_moe_keeps_adapter_drop_in_forward_contract():
    class _Gate:
        def forward(self, hidden):
            return torch.zeros(hidden.shape[0], 3)

    class _Shared:
        def forward(self, hidden):
            return torch.zeros_like(hidden)

    class _SharedGate:
        def forward(self, hidden):
            return torch.full((hidden.shape[0], 1), -100.0)

    class _Routed:
        def forward(self, *, hidden_states, router_logits):
            assert router_logits.shape == (hidden_states.shape[0], 3)
            return torch.ones_like(hidden_states)

    moe = object.__new__(_SparseMoE)
    moe.gate = _Gate()
    moe.shared_expert = _Shared()
    moe.shared_expert_gate = _SharedGate()
    moe.experts = _Routed()
    hidden = torch.zeros(2, 4)

    assert torch.equal(moe.forward(hidden), torch.ones_like(hidden))
