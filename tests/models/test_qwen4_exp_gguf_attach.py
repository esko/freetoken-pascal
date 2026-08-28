"""H0 tests for explicit borrowed GGUF expert attachment to Qwen4-Exp."""

# The model package is imported only after importorskip below so the H0 test
# remains collectable in the repository's torch-free baseline environment.
# ruff: noqa: E402, I001

from __future__ import annotations

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


@pytest.fixture
def fake_adapter(monkeypatch):
    from freetoken.models.qwen4_exp import gguf_attach as attach_module

    _FakeAdapter.created = []
    _FakeAdapter.fail_layer = None
    monkeypatch.setattr(attach_module, "QwenGGUFCpuMoELayer", _FakeAdapter)
    return _FakeAdapter


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
