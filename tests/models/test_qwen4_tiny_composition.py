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
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference
from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "qwen4-tiny"


class _LastTokenMetadata:
    def get_last_indices(self, batch_size: int) -> torch.Tensor:
        assert batch_size == 1
        return torch.tensor([0], dtype=torch.long)


def _fixture_config():
    raw = json.loads((FIXTURE / "config.json").read_text(encoding="utf-8"))
    raw["text_config"] = SimpleNamespace(**raw["text_config"])
    # The fixture's manifest declares float32-reference tensors.  Keep its published geometry
    # and use an explicit unquantized model config so the tiny CPU oracle does not pretend to
    # exercise block-FP8 kernels whose production dimensions are multiples of 128.
    raw["quantization_config"] = None
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
    batch.attn_metadata = _LastTokenMetadata()
    if phase == "decode":
        batch.linear_table_idx = torch.tensor([slot], dtype=torch.int32)
    return batch


def _run(model, context, batch):
    with context.forward_batch(batch):
        return model.forward().detach().clone()


@pytest.fixture(autouse=True)
def _reset_context():
    import freetoken.core as core

    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = None


def test_tiny_qwen_text_composes_prefill_decode_and_is_deterministic():
    config = _fixture_config()
    model = _new_model(config)
    context = _new_runtime(config)
    model.load_host_weights("fixture", dummy=True)

    prompt = [3, 4, 5, 6]
    prefill = _batch(prompt, phase="prefill")
    first = _run(model, context, prefill)
    assert first.shape == (1, config.vocab_size)
    assert torch.isfinite(first).all()

    decode = _batch(prompt + [9], phase="decode", cached_len=len(prompt))
    decoded = _run(model, context, decode)
    assert decoded.shape == (1, config.vocab_size)
    assert torch.isfinite(decoded).all()

    context.linear_state_pool.reset(1)
    context.attn_backend = TorchDenseQSAReference(
        config, num_slots=4, max_len=512, device=torch.device("cpu"), dtype=torch.float32
    )
    replay = _run(model, context, _batch(prompt, phase="prefill"))
    assert torch.equal(first, replay)
    model.close_host_resources()


def test_tiny_qwen_chunked_prefill_matches_full_prefill_and_reset():
    config = _fixture_config()
    model = _new_model(config)
    context = _new_runtime(config)
    model.load_host_weights("fixture", dummy=True)
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
    model.close_host_resources()


def test_tiny_qwen_vision_input_fails_closed():
    config = _fixture_config()
    model = _new_model(config)
    context = _new_runtime(config)
    model.load_host_weights("fixture", dummy=True)
    batch = _batch([3], phase="prefill")
    batch.mm_embeds = torch.zeros(1, config.hidden_size)
    with pytest.raises(RuntimeError, match="vision inputs are outside"):
        _run(model, context, batch)
    model.close_host_resources()
