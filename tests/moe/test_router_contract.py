import dataclasses
from types import SimpleNamespace

import pytest
import torch
from freetoken.moe.fused import (
    RouterDispatchError,
    _cpu_token_limit,
    _torch_fused_topk,
    fused_topk,
    resolve_router_dispatch,
)
from freetoken.moe.router_contract import RouterDispatchDecision


def test_cuda_token_limit_helper_never_extracts_device_scalar():
    class GuardedCudaScalar:
        device = SimpleNamespace(type="cuda")

        def item(self):
            pytest.fail("CUDA token scalar must not be copied to the host")

    assert _cpu_token_limit(GuardedCudaScalar(), 4) is None


def test_cpu_token_limit_helper_extracts_and_checks_range():
    assert _cpu_token_limit(torch.tensor(3, dtype=torch.int32), 4) == 3
    with pytest.raises(ValueError, match="range"):
        _cpu_token_limit(torch.tensor(5, dtype=torch.int32), 4)


def test_router_dispatch_decision_is_immutable_and_complete():
    decision = resolve_router_dispatch(
        requested_mode="auto",
        topk=10,
        num_experts=512,
        renormalize=True,
        has_token_limit=True,
        triton_candidate_available=True,
        triton_kernels_available=False,
    )

    assert isinstance(decision, RouterDispatchDecision)
    assert dataclasses.is_dataclass(decision)
    assert decision.requested_mode == "auto"
    assert decision.selected_implementation == "torch-reference"
    assert decision.topk == 10
    assert decision.num_experts == 512
    assert decision.renormalize is True
    assert decision.has_token_limit is True
    assert decision.fallback_reason == "candidate-not-qualified"
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.topk = 9


@pytest.mark.parametrize("renormalize", [False, True])
def test_torch_router_qwen_shape_matches_stable_full_softmax_reference(renormalize):
    generator = torch.Generator().manual_seed(38)
    logits = torch.randn((7, 512), generator=generator, dtype=torch.bfloat16)
    hidden_states = torch.zeros((7, 64), dtype=torch.bfloat16)

    weights, ids = fused_topk(
        hidden_states,
        logits,
        topk=10,
        renormalize=renormalize,
        router_mode="torch-reference",
    )

    scores = logits.float()
    probabilities = torch.softmax(scores, dim=-1)
    expected_ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :10]
    expected_weights = probabilities.gather(1, expected_ids)
    if renormalize:
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    assert ids.dtype == torch.int32
    assert torch.equal(ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)


def test_torch_router_ties_choose_lowest_expert_ids():
    logits = torch.full((2, 8), -10.0)
    logits[0, [6, 2, 5]] = 1.0
    logits[1] = 0.0

    _, ids = _torch_fused_topk(logits, 3, True, None)

    assert ids.tolist() == [[2, 5, 6], [0, 1, 2]]


def test_torch_router_nonfinite_inputs_have_deterministic_reference_semantics():
    logits = torch.tensor(
        [
            [float("nan"), float("-inf"), 0.0, float("inf"), float("inf")],
            [float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")],
        ],
        dtype=torch.float32,
    )

    weights, ids = _torch_fused_topk(logits, 3, False, None)

    assert ids.tolist() == [[3, 4, 2], [0, 1, 2]]
    torch.testing.assert_close(
        weights,
        torch.tensor([[0.5, 0.5, 0.0], [0.2, 0.2, 0.2]]),
        rtol=0,
        atol=0,
    )


def test_torch_router_padding_zeroes_ids_and_weights():
    logits = torch.randn((5, 64), generator=torch.Generator().manual_seed(1))
    limit = torch.tensor(2, dtype=torch.int64)

    weights, ids = _torch_fused_topk(logits, 3, True, limit)

    assert torch.equal(ids[2:], torch.full((3, 3), -1, dtype=torch.int32))
    assert torch.equal(weights[2:], torch.zeros((3, 3)))
    assert (ids[:2] >= 0).all()


@pytest.mark.parametrize(
    ("logits", "topk", "renormalize", "limit", "message"),
    [
        (torch.zeros((2, 4, 1)), 1, True, None, "2D"),
        (torch.zeros((2, 4)), 0, True, None, "topk"),
        (torch.zeros((2, 4)), 5, True, None, "topk"),
        (torch.zeros((2, 4)), True, True, None, "topk"),
        (torch.zeros((2, 4)), 1, 1, None, "renormalize"),
        (torch.zeros((2, 4)), 1, True, torch.tensor([1]), "scalar"),
        (torch.zeros((2, 4)), 1, True, torch.tensor(1.0), "integer"),
        (torch.zeros((2, 4)), 1, True, torch.tensor(True), "integer"),
        (torch.zeros((2, 4)), 1, True, torch.tensor(-1), "range"),
        (torch.zeros((2, 4)), 1, True, torch.tensor(3), "range"),
    ],
)
def test_torch_router_rejects_invalid_contract(logits, topk, renormalize, limit, message):
    with pytest.raises((TypeError, ValueError), match=message):
        fused_topk(
            torch.zeros((2, 8)),
            logits,
            topk,
            renormalize,
            limit,
            router_mode="torch-reference",
        )


def test_torch_router_rejects_hidden_token_count_mismatch():
    with pytest.raises(ValueError, match="token count"):
        fused_topk(
            torch.zeros((3, 8)),
            torch.zeros((2, 4)),
            1,
            True,
            router_mode="torch-reference",
        )


def test_auto_qwen_router_stays_reference_with_candidate_reason():
    observed = []

    weights, ids = fused_topk(
        torch.zeros((2, 8)),
        torch.randn((2, 512), generator=torch.Generator().manual_seed(2)),
        10,
        True,
        router_mode="auto",
        triton_candidate_available=True,
        triton_kernels_available=False,
        router_observer=observed.append,
    )

    assert weights.shape == (2, 10)
    assert ids.shape == (2, 10)
    assert len(observed) == 1
    assert observed[0].requested_mode == "auto"
    assert observed[0].selected_implementation == "torch-reference"
    assert observed[0].topk == 10
    assert observed[0].num_experts == 512
    assert observed[0].renormalize is True
    assert observed[0].has_token_limit is False
    assert observed[0].fallback_reason == "candidate-not-qualified"


def test_fallback_warning_suppression_is_keyed_by_full_decision(monkeypatch):
    import freetoken.moe.fused as fused_module

    warnings = []
    previous = set(fused_module._warned_torch_decisions)
    fused_module._warned_torch_decisions.clear()
    monkeypatch.setattr(fused_module.logger, "warning_rank0", warnings.append)
    try:
        for renormalize in (False, True, False):
            fused_topk(
                torch.zeros((1, 8)),
                torch.zeros((1, 512)),
                10,
                renormalize,
                router_mode="auto",
                triton_kernels_available=False,
            )
    finally:
        fused_module._warned_torch_decisions.clear()
        fused_module._warned_torch_decisions.update(previous)

    assert len(warnings) == 2


def test_auto_qualified_power_of_two_preserves_external_triton_choice():
    decision = resolve_router_dispatch(
        requested_mode="auto",
        topk=8,
        num_experts=64,
        renormalize=False,
        has_token_limit=False,
        triton_candidate_available=True,
        triton_kernels_available=True,
    )

    assert decision.selected_implementation == "triton-kernels"
    assert decision.fallback_reason is None


def test_explicit_candidate_requires_availability():
    with pytest.raises(RouterDispatchError, match="candidate-unavailable"):
        resolve_router_dispatch(
            requested_mode="triton-candidate",
            topk=10,
            num_experts=512,
            renormalize=True,
            has_token_limit=False,
            triton_candidate_available=False,
        )


def test_explicit_candidate_dispatches_injected_implementation(monkeypatch):
    import freetoken.moe.fused as fused_module

    called = []

    def fake_candidate(gating_output, topk, renormalize, num_token_non_padded):
        called.append((gating_output, topk, renormalize, num_token_non_padded))
        return torch.ones((2, topk)), torch.zeros((2, topk), dtype=torch.int32)

    monkeypatch.setattr(fused_module, "_run_triton_candidate", fake_candidate)
    observed = []
    weights, ids = fused_module.fused_topk(
        torch.zeros((2, 8)),
        torch.zeros((2, 512)),
        10,
        False,
        router_mode="triton-candidate",
        triton_candidate_available=True,
        router_observer=observed.append,
    )

    assert len(called) == 1
    assert called[0][1:] == (10, False, None)
    assert weights.shape == (2, 10)
    assert ids.dtype == torch.int32
    assert observed[0].selected_implementation == "triton-candidate"
    assert observed[0].fallback_reason is None


def test_explicit_candidate_dispatches_exceptional_logits_without_host_probe(monkeypatch):
    import freetoken.moe.fused as fused_module

    called = []

    def fake_candidate(gating_output, topk, renormalize, num_token_non_padded):
        called.append((gating_output, topk, renormalize, num_token_non_padded))
        return torch.zeros((1, topk)), torch.zeros((1, topk), dtype=torch.int32)

    monkeypatch.setattr(
        fused_module,
        "_run_triton_candidate",
        fake_candidate,
    )
    fused_module.fused_topk(
        torch.zeros((1, 8)),
        torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]),
        2,
        False,
        router_mode="triton-candidate",
        triton_candidate_available=True,
    )
    assert len(called) == 1


def test_router_mode_from_env_defaults_to_auto_and_rejects_invalid_values(monkeypatch):
    from freetoken.moe.router_contract import router_mode_from_env

    monkeypatch.delenv("FREETOKEN_ROUTER_MODE", raising=False)
    assert router_mode_from_env() == "auto"

    for value in ("bogus", " Auto ", ""):
        monkeypatch.setenv("FREETOKEN_ROUTER_MODE", value)
        with pytest.raises(ValueError, match="unsupported router mode"):
            router_mode_from_env()


def test_invalid_router_mode_fails_at_backend_and_layer_construction(monkeypatch):
    monkeypatch.setenv("FREETOKEN_ROUTER_MODE", "bogus")

    from freetoken.moe.fused import FusedMoe

    with pytest.raises(ValueError, match="unsupported router mode"):
        FusedMoe()

    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    from freetoken.layers.moe import MoELayer

    with pytest.raises(ValueError, match="unsupported router mode"):
        MoELayer(
            num_experts=4,
            top_k=2,
            hidden_size=8,
            intermediate_size=16,
            allocate_experts=False,
        )


def test_fused_moe_forced_candidate_emits_dispatch_before_route(monkeypatch):
    import freetoken.moe.fused as fused_module

    monkeypatch.setenv("FREETOKEN_ROUTER_MODE", "triton-candidate")
    monkeypatch.setattr(fused_module, "_probe_triton_candidate", lambda: True)
    candidate_calls = []

    def fake_candidate(gating_output, topk, renormalize, num_token_non_padded):
        candidate_calls.append((gating_output, topk, renormalize, num_token_non_padded))
        return torch.full((2, topk), 0.5), torch.zeros((2, topk), dtype=torch.int32)

    monkeypatch.setattr(fused_module, "_run_triton_candidate", fake_candidate)
    monkeypatch.setattr(
        fused_module,
        "fused_experts_impl",
        lambda hidden_states, *args, **kwargs: hidden_states,
    )
    from freetoken.moe.fused import FusedMoe

    events = []
    output = FusedMoe().forward(
        torch.zeros((2, 8)),
        torch.zeros((4, 16, 8)),
        torch.zeros((4, 8, 8)),
        torch.zeros((2, 4)),
        topk=2,
        renormalize=True,
        debug_observer=lambda name, payload: events.append((name, payload)),
    )

    assert output.shape == (2, 8)
    assert len(candidate_calls) == 1
    assert [name for name, _payload in events] == ["router_dispatch", "router"]
    dispatch = events[0][1]
    assert dispatch == {
        "requested_mode": "triton-candidate",
        "selected_implementation": "triton-candidate",
        "topk": 2,
        "num_experts": 4,
        "renormalize": True,
        "has_token_limit": False,
        "fallback_reason": None,
    }


def test_moe_layer_threads_router_mode_and_dispatch_event(monkeypatch):
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    from freetoken.layers.moe import MoELayer

    layer = MoELayer(
        num_experts=4,
        top_k=2,
        hidden_size=8,
        intermediate_size=16,
        allocate_experts=False,
        router_mode="torch-reference",
    )
    events = []
    weights, ids = layer._route_and_observe(
        torch.zeros((1, 8)),
        torch.zeros((1, 4)),
        lambda name, payload: events.append((name, payload)),
    )

    assert weights.shape == (1, 2)
    assert ids.shape == (1, 2)
    assert [name for name, _payload in events] == ["router_dispatch", "router"]
    assert events[0][1]["requested_mode"] == "torch-reference"
    assert events[0][1]["selected_implementation"] == "torch-reference"
