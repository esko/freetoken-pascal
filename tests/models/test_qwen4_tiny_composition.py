"""H0 composition oracle for the text-only Qwen4-Exp fixture.

This test intentionally drives the registered ``Qwen4ExpForCausalLM`` through its model,
state, PLE, GDN, QSA and MoE components.  It is a CPU/reference test only: it does not stand in
for Engine/scheduler/server integration or any H1/H2 hardware evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from freetoken.core import Batch, Context, Req, SamplingParams, set_global_ctx
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
from freetoken.models.qwen4_exp.moe import Qwen4ExpMoE

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "qwen4-tiny"


class _LastTokenMetadata:
    def __init__(self, last_index: int):
        self.last_index = last_index

    def get_last_indices(self, batch_size: int) -> torch.Tensor:
        assert batch_size == 1
        return torch.tensor([self.last_index], dtype=torch.long)


def _fixture_config():
    raw = json.loads((FIXTURE / "config.json").read_text(encoding="utf-8"))
    raw["text_config"] = SimpleNamespace(**raw["text_config"])
    # The fixture's manifest declares float32-reference tensors.  Keep its published geometry
    # and use an explicit unquantized model config so the tiny CPU oracle does not pretend to
    # exercise block-FP8 kernels whose production dimensions are multiples of 128.
    raw["quantization_config"] = None
    raw["reference_only"] = True
    return parse_config(SimpleNamespace(**raw))


def _new_model(config, seed: int = 17):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    torch.manual_seed(seed)
    model = Qwen4ExpForCausalLM(config)
    for tensor in model.state_dict().values():
        if tensor.is_floating_point():
            tensor.normal_(mean=0.0, std=0.08)
        else:
            tensor.zero_()
    return model


def _new_runtime(config):
    import freetoken.core as core

    core._GLOBAL_CTX = None
    context = Context(page_size=4)
    context.linear_state_pool = LinearStatePool(
        config.linear_attention_group(),
        num_slots=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        slot_states=config.slot_states,
    )
    context.attn_backend = TorchDenseQSAReference(
        config,
        num_slots=4,
        max_len=512,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    set_global_ctx(context)
    return context


def _batch(tokens: list[int], *, phase: str, cached_len: int = 0, slot: int = 1) -> Batch:
    request = Req(
        input_ids=torch.tensor(tokens, dtype=torch.int64),
        table_idx=slot,
        cached_len=cached_len,
        output_len=8,
        uid=slot,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=8),
        cache_handle=None,
    )
    batch = Batch([request], phase=phase)
    batch.padded_reqs = [request]
    batch.input_ids = request.input_ids[cached_len:]
    batch.positions = torch.arange(cached_len, len(tokens), dtype=torch.int64)
    batch.out_loc = None
    batch.attn_metadata = _LastTokenMetadata(len(batch.input_ids) - 1)
    if phase == "decode":
        batch.linear_table_idx = torch.tensor([slot], dtype=torch.int32)
    return batch


def _run(model, context, batch):
    with context.forward_batch(batch):
        return model.forward().detach().clone()


@pytest.fixture
def _tiny_model_runtime():
    config = _fixture_config()
    model = _new_model(config)
    context = _new_runtime(config)
    model.load_host_weights("fixture", dummy=True)
    try:
        yield config, model, context
    finally:
        model.close_host_resources()


@pytest.fixture(autouse=True)
def _reset_context():
    import freetoken.core as core

    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = None


def test_tiny_qwen_text_composes_prefill_decode_and_is_deterministic(_tiny_model_runtime):
    config, model, context = _tiny_model_runtime

    prompt = [3, 4, 5, 6]
    prefill = _batch(prompt, phase="prefill")
    first = _run(model, context, prefill)
    assert first.shape == (1, config.vocab_size)
    assert torch.isfinite(first).all()

    decode = _batch([*prompt, 9], phase="decode", cached_len=len(prompt))
    decoded = _run(model, context, decode)
    assert decoded.shape == (1, config.vocab_size)
    assert torch.isfinite(decoded).all()
    # These are the deterministic greedy IDs for the seeded tiny fixture.  They
    # make an accidental change to the composed text path visible even when the
    # logits remain finite and shapes still agree.
    assert int(first.argmax()) == 61
    assert int(decoded.argmax()) == 56
    # Seeded regression values from this fixture, not an independent HF parity
    # claim.  The complete vector equality above remains the stronger replay
    # check; these values make a changed logit scale visible.
    assert float(first[0, 61]) == pytest.approx(0.8678999, abs=1e-6)
    assert float(decoded[0, 56]) == pytest.approx(0.6091668, abs=1e-6)

    context.linear_state_pool.reset(1)
    context.attn_backend = TorchDenseQSAReference(
        config, num_slots=4, max_len=512, device=torch.device("cpu"), dtype=torch.float32
    )
    replay = _run(model, context, _batch(prompt, phase="prefill"))
    assert torch.equal(first, replay)
    assert int(first.argmax()) == int(replay.argmax())


def test_tiny_qwen_chunked_prefill_matches_full_prefill_and_reset(_tiny_model_runtime):
    config, model, context = _tiny_model_runtime
    prompt = [3, 4, 5, 6]

    full = _run(model, context, _batch(prompt, phase="prefill"))

    context.linear_state_pool.reset(1)
    context.attn_backend = TorchDenseQSAReference(
        config, num_slots=4, max_len=512, device=torch.device("cpu"), dtype=torch.float32
    )
    _run(model, context, _batch(prompt[:2], phase="prefill"))
    chunked = _run(model, context, _batch(prompt, phase="prefill", cached_len=2))
    assert torch.allclose(full, chunked, rtol=2e-5, atol=2e-6)

    context.linear_state_pool.reset(1)
    context.attn_backend = TorchDenseQSAReference(
        config, num_slots=4, max_len=512, device=torch.device("cpu"), dtype=torch.float32
    )
    reset = _run(model, context, _batch(prompt, phase="prefill"))
    assert torch.equal(full, reset)


def test_tiny_qwen_vision_input_fails_closed(_tiny_model_runtime):
    config, model, context = _tiny_model_runtime
    batch = _batch([3], phase="prefill")
    batch.mm_embeds = torch.zeros(1, config.hidden_size)
    with pytest.raises(RuntimeError, match="vision inputs are outside"):
        _run(model, context, batch)


def test_tiny_qwen_debug_hook_reports_reference_boundaries(_tiny_model_runtime):
    config, model, context = _tiny_model_runtime
    records: list[dict[str, object]] = []
    model.set_debug_hook(records.append)

    logits = _run(model, context, _batch([3, 4, 5, 6], phase="prefill"))

    assert len(records) == 1
    assert torch.equal(records[0]["logits"], logits)
    observations = records[0]["observations"]
    assert len(observations["gdn_backend"]) == 1
    assert len(observations["ple"]) == 1
    assert len(observations["router"]) == config.num_layers
    assert observations["gdn_backend"][0]["selected_implementation"] == "torch-reference"
    assert torch.equal(
        observations["ple"][0]["contribution"],
        torch.zeros_like(observations["ple"][0]["contribution"]),
    )
    assert records[0]["ple_state"] == {0: {}}
    route = observations["router"][0]
    assert route["ids"].shape == (4, config.num_experts_per_tok)
    assert route["weights"].shape == route["ids"].shape
    assert route["valid_token_count"] == 4
    torch.testing.assert_close(
        observations["router"][0]["ids"],
        torch.tensor([[0, 3], [2, 0], [2, 1], [1, 3]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        observations["router"][0]["weights"],
        torch.tensor(
            [
                [0.5423497, 0.4576504],
                [0.6755262, 0.3244737],
                [0.5121318, 0.4878682],
                [0.5112431, 0.4887569],
            ]
        ),
        rtol=1e-6,
        atol=1e-6,
    )
    assert all(
        event["requested_mode"] == "torch-reference"
        and event["selected_implementation"] == "torch-reference"
        for event in observations["router_dispatch"]
    )


def test_qwen4_cpu_reference_rejects_offload_expert_layout():
    moe = object.__new__(Qwen4ExpMoE)
    moe.experts = SimpleNamespace(weight_format="bf16")
    with pytest.raises(RuntimeError, match="resident gate_up_proj/down_proj"):
        moe.forward(torch.zeros(1, 4))


def test_qwen4_cpu_reference_rejects_quantized_expert_layout():
    moe = object.__new__(Qwen4ExpMoE)
    moe.experts = SimpleNamespace(
        weight_format="fp8_block",
        gate_up_proj=torch.empty(1),
        down_proj=torch.empty(1),
    )
    with pytest.raises(RuntimeError, match="unquantized resident expert weights"):
        moe.forward(torch.zeros(1, 4))
