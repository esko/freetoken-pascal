"""H0 tests for the isolated Qwen GGUF CPU routed-expert layer adapter."""

from __future__ import annotations

import numpy as np
import pytest
from freetoken.gguf_host import GGUFExpertLayout, QwenGGUFHostWeights, QwenHostLayout

from tests.moe.test_gguf_cpu_bridge import (
    Q4_K,
    Q5_1,
    _Bank,
    _Banks,
    _descriptor,
    _geometry_host,
    _Ple,
)


def _fast_primitives(monkeypatch: pytest.MonkeyPatch) -> None:
    from freetoken.moe import q4_k

    class _FastQ4:
        isa = "avx2"
        backend = "q4_k_test"
        fallback_reason = None

        def gemv(self, rows, input_dim, vector, *, out, scratch=None):
            del input_dim, scratch
            np.multiply(rows[:, 0].astype(np.float32), vector.sum(), out=out)
            return out

    class _FastMixed:
        isa = "avx2"
        backend = "mixed_test"
        fallback_reason = None

        def backend_for(self, quant_name):
            return f"{str(quant_name).lower()}_test"

        def gemv(self, rows, input_dim, vector, *, quant_name, out):
            del input_dim, quant_name
            np.multiply(rows[:, 0].astype(np.float32), vector.sum(), out=out)
            return out

    monkeypatch.setattr(q4_k, "select_q4_k_primitive", lambda mode="auto": _FastQ4())
    monkeypatch.setattr(q4_k, "select_mixed_gemv_primitive", lambda mode="auto": _FastMixed())


def _router_host(experts: int = 10) -> QwenGGUFHostWeights:
    descriptors = []
    banks = {}
    for projection, quant_type, quant_name in (
        ("gate", Q4_K, "Q4_K"),
        ("up", Q4_K, "Q4_K"),
        ("down", Q5_1, "Q5_1"),
    ):
        descriptor, values = _descriptor(
            0,
            projection,
            quant_type,
            quant_name,
            experts=experts,
        )
        descriptors.append(descriptor)
        banks[(0, projection)] = _Bank(descriptor, values)
    expert_layout = GGUFExpertLayout(
        descriptors=tuple(descriptors),
        slot_pools=(),
        num_layers=1,
        num_experts=experts,
    )
    layout = QwenHostLayout(
        experts=expert_layout,
        ple=_Ple(),
        total_tensor_bytes=sum(item.tensor_bytes for item in descriptors),
        shard_paths=("synthetic.gguf",),
    )
    return QwenGGUFHostWeights(layout, _Banks(expert_layout, banks), _Ple())


@pytest.fixture
def bundle(monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip("torch")
    del torch
    _fast_primitives(monkeypatch)
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    value = QwenGGUFCpuExpertBundle.from_host(
        _geometry_host(promoted=False),
        top_k=10,
        mode="avx2",
        max_tokens=3,
        max_routes=10,
        required_alignment=1,
    )
    try:
        yield value
    finally:
        value.close()


def _adapter(bundle):
    from freetoken.moe.gguf_layer import QwenGGUFCpuMoELayer

    return QwenGGUFCpuMoELayer(
        bundle,
        layer_id=0,
        num_experts=2,
        top_k=10,
        hidden_size=2560,
        intermediate_size=640,
        renormalize=True,
        activation="silu",
        apply_router_weight_on_input=False,
    )


@pytest.mark.parametrize("route_width", [1, 2, 4, 8, 10])
def test_routed_forward_matches_direct_bundle_for_route_widths_and_duplicates(
    bundle, route_width: int
) -> None:
    torch = pytest.importorskip("torch")
    layer = _adapter(bundle)
    hidden = torch.ones((1, 2560), dtype=torch.float32)
    ids = torch.tensor([[index % 2 for index in range(route_width)]], dtype=torch.int32)
    weights = torch.arange(1, route_width + 1, dtype=torch.float32).reshape(1, -1)
    expected = bundle.decode(0, hidden, weights, ids)
    actual = layer.routed_forward(hidden, weights, ids)
    torch.testing.assert_close(actual, expected)
    assert ids.tolist() == [[index % 2 for index in range(route_width)]]


def test_routed_forward_matches_direct_bundle_for_mixed_geometry_and_padding(bundle) -> None:
    torch = pytest.importorskip("torch")
    layer = _adapter(bundle)
    hidden = torch.arange(3 * 2560, dtype=torch.float32).reshape(3, 2560) / 1000
    ids = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0], [-1, -1, -1, -1]], dtype=torch.int32)
    weights = torch.ones((3, 4), dtype=torch.float32)
    expected = bundle.decode(0, hidden, weights, ids, num_token_non_padded=2)
    actual = layer.routed_forward(
        hidden,
        weights,
        ids,
        num_token_non_padded=2,
    )
    torch.testing.assert_close(actual, expected)
    assert tuple(actual.shape) == (3, 2560)
    assert bundle.kernel_census == ("q4_k_test", "q5_1_test")


def test_forward_uses_exact_cpu_softmax_topk_and_preserves_semantic_observer(bundle) -> None:
    torch = pytest.importorskip("torch")
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle
    from freetoken.moe.gguf_layer import QwenGGUFCpuMoELayer

    router_bundle = QwenGGUFCpuExpertBundle.from_host(
        _router_host(),
        top_k=10,
        mode="avx2",
        max_tokens=2,
        max_routes=10,
        required_alignment=1,
    )
    layer = QwenGGUFCpuMoELayer(
        router_bundle,
        layer_id=0,
        num_experts=10,
        top_k=10,
        hidden_size=256,
        intermediate_size=256,
        renormalize=True,
        activation="silu",
        apply_router_weight_on_input=False,
    )
    hidden = torch.ones((2, 2560), dtype=torch.float32)
    logits = torch.tensor(
        [
            [0.2, 0.7, -0.1, 0.0, 0.3, -0.4, 0.5, 0.1, -0.2, 0.4],
            [1.0, -0.5, 0.1, 0.3, -0.2, 0.6, -0.1, 0.0, 0.2, -0.3],
        ],
        dtype=torch.float32,
    )
    observer_calls: list[tuple[str, dict[str, object]]] = []
    try:
        actual = layer.forward(
            hidden[:, :256],
            router_logits=logits,
            debug_observer=lambda name, payload: observer_calls.append((name, payload)),
        )
        probs = torch.softmax(logits.float(), dim=-1)
        weights, ids = torch.topk(probs, 10, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        expected = router_bundle.decode(
            0,
            hidden[:, :256],
            weights,
            ids.to(torch.int32),
        )
        torch.testing.assert_close(actual, expected)
        assert len(observer_calls) == 1
        name, payload = observer_calls[0]
        assert name == "router"
        assert payload["layer_id"] == 0
        torch.testing.assert_close(payload["ids"], ids.to(torch.int32))
        torch.testing.assert_close(payload["weights"], weights)
    finally:
        router_bundle.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_size": 640}, "hidden_size"),
        ({"intermediate_size": 2560}, "intermediate_size"),
        ({"num_experts": 3}, "num_experts"),
        ({"top_k": 9}, "top_k"),
        ({"activation": "relu"}, "activation"),
        ({"apply_router_weight_on_input": True}, "apply_router_weight_on_input"),
    ],
)
def test_adapter_rejects_inconsistent_model_geometry(bundle, kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _adapter_with_overrides(bundle, **kwargs)


def _adapter_with_overrides(bundle, **overrides):
    from freetoken.moe.gguf_layer import QwenGGUFCpuMoELayer

    values = {
        "layer_id": 0,
        "num_experts": 2,
        "top_k": 10,
        "hidden_size": 2560,
        "intermediate_size": 640,
        "renormalize": True,
        "activation": "silu",
        "apply_router_weight_on_input": False,
    }
    values.update(overrides)
    return QwenGGUFCpuMoELayer(bundle, **values)


def test_adapter_rejects_invalid_routes_and_runtime_modes(bundle) -> None:
    torch = pytest.importorskip("torch")
    layer = _adapter(bundle)
    hidden = torch.ones((1, 2560), dtype=torch.float32)
    weights = torch.ones((1, 1), dtype=torch.float32)

    with pytest.raises(ValueError, match="expert id"):
        layer.routed_forward(hidden, weights, torch.tensor([[2]], dtype=torch.int32))
    with pytest.raises(ValueError, match="expert id"):
        layer.routed_forward(hidden, weights, torch.tensor([[-2]], dtype=torch.int32))
    with pytest.raises(ValueError, match="route width"):
        layer.routed_forward(
            hidden,
            torch.ones((1, 11), dtype=torch.float32),
            torch.zeros((1, 11), dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="num_token_non_padded"):
        layer.routed_forward(
            hidden,
            weights,
            torch.zeros((1, 1), dtype=torch.int32),
            num_token_non_padded=2,
        )
    with pytest.raises(ValueError, match="CPU"):
        layer.routed_forward(
            hidden,
            weights,
            torch.zeros((1, 1), dtype=torch.int32, device="meta"),
        )


def test_adapter_requires_router_logits_for_compatible_forward(bundle) -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="router_logits"):
        _adapter(bundle).forward(torch.ones((1, 2560), dtype=torch.float32))


def test_adapter_exposes_bundle_telemetry_and_fails_after_bundle_close(bundle) -> None:
    torch = pytest.importorskip("torch")
    layer = _adapter(bundle)
    layer.routed_forward(
        torch.ones((1, 2560), dtype=torch.float32),
        torch.ones((1, 1), dtype=torch.float32),
        torch.zeros((1, 1), dtype=torch.int32),
    )
    assert layer.last_telemetry is bundle.last_telemetry
    assert layer.host_weight_telemetry["execution_telemetry"] is not None
    bundle.close()
    with pytest.raises(RuntimeError, match="closed"):
        layer.routed_forward(
            torch.ones((1, 2560), dtype=torch.float32),
            torch.ones((1, 1), dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.int32),
        )
