from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import freetoken.models.qwen4_exp as qwen4_exp
import pytest
import torch
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.model import (
    Qwen4ExpForCausalLM,
    _ple_request_tokens,
    _PLELayer,
    build_ngram_ids,
)
from freetoken.models.qwen4_exp.reference import (
    gated_delta_chunked_reference,
    gated_delta_reference,
    hyperconnection_reference,
    routed_shared_expert_reference,
)
from freetoken.models.qwen4_exp.weight import _rename, _try_fuse
from freetoken.models.register import get_model_spec

ROOT = Path(__file__).resolve().parents[2]


def _config(quantization_config=None):
    text = SimpleNamespace(
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        head_dim=256,
        rope_parameters={
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        },
        indexer_budget=2048,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        max_position_embeddings=262_144,
        num_key_value_heads=2,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        eos_token_id=248044,
        hc_count=4,
        hc_lowrank=320,
        ple_layer_ids=[2],
        ple_embed_dim=2560,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        split_ngram_parts=128,
        indexer_compress_ratio=4,
        output_gate_type="sigmoid",
        hidden_act="silu",
        num_hidden_layers=4,
        num_attention_heads=24,
        hidden_size=2560,
        vocab_size=248320,
        rms_norm_eps=1e-6,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        norm_topk_prob=None,
        tie_word_embeddings=False,
    )
    return SimpleNamespace(
        text_config=text,
        quantization_config=(
            quantization_config
            if quantization_config is not None
            else {"quant_method": "fp8", "weight_block_size": [128, 128]}
        ),
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        image_token_id=248056,
    )


def test_qwen4_tiny_fixture_parses_as_text_only() -> None:
    raw = json.loads((ROOT / "tests/fixtures/qwen4-tiny/config.json").read_text(encoding="utf-8"))
    raw["text_config"] = SimpleNamespace(**raw["text_config"])
    config = parse_config(SimpleNamespace(**raw))

    assert config.num_layers == 2
    assert config.num_experts == 4
    assert config.attn_type_for_layer(1).value == "qsa"
    assert not config.is_multimodal


def test_qwen4_config_uses_exact_qsa_prefix():
    config = parse_config(_config())
    assert config.rotary_config.max_position == 262_144
    assert config.expert_quant == "fp8_block"
    assert config.attn_quant == "none"
    assert config.qwen4_args.ple_layer_ids == (1,)
    assert config.qwen4_args.output_gate_type == "sigmoid"
    assert config.requires_naive_cache
    assert not config.supports_cuda_graph
    assert not config.is_multimodal
    assert config.is_linear_layer(0)
    assert not config.is_linear_layer(3)


def test_qwen4_config_accepts_transformers_sparse_attention_alias():
    hf_config = _config()
    hf_config.text_config.layer_types[-1] = "qwen_sparse_attention"
    config = parse_config(hf_config)
    assert not config.is_linear_layer(3)
    assert config.attn_type_for_layer(3).value == "qsa"
    spec = config.kv_cache_group_specs()[0]
    assert spec.layer_ids == (3,)
    assert spec.index_head_dim == 128
    assert spec.index_compress_ratio == 4
    assert spec.index_token_budget == 2048


def test_qwen4_config_accepts_missing_norm_topk_prob():
    hf_config = _config()
    del hf_config.text_config.norm_topk_prob
    config = parse_config(hf_config)
    assert not config.norm_topk_prob


def test_qwen4_config_accepts_routed_expert_nvfp4():
    config = parse_config(
        _config(
            {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {"num_bits": 4, "group_size": 16, "type": "float"},
                    }
                },
                "ignore": [
                    "*.self_attn.*",
                    "*.linear_attn.*",
                    "*.mlp.shared_expert.*",
                    "*.ple.*",
                    "model.visual.*",
                    "lm_head",
                ],
            }
        )
    )
    assert config.expert_quant == "nvfp4"
    assert config.weight_block_size is None
    assert config.attn_quant == "none"
    assert config.dense_quant == "none"
    assert config.lm_head_quant == "none"


def test_qwen4_config_rejects_unsupported_expert_quant():
    with pytest.raises(ValueError, match="requires routed experts"):
        parse_config(_config({"quant_method": "gptq"}))


def test_qwen4_exports_nvfp4_loader_hooks():
    assert callable(qwen4_exp.load_nvfp4_expert_sources)
    assert callable(qwen4_exp.load_nvfp4_expert_sources_parallel)


def test_qwen4_registry_entry():
    spec = get_model_spec("Qwen4ExpForConditionalGeneration")
    assert spec.module == "freetoken.models.qwen4_exp"
    assert spec.model_cls == "Qwen4ExpForCausalLM"


def test_qwen4_weight_names():
    assert _rename("model.language_model.layers.1.ple.key_proj.weight") == (
        "model.layers.1.ple.key_proj.weight"
    )
    assert _rename("model.visual.blocks.0.attn.qkv.weight") is None
    assert _rename("model.language_model.layers.3.self_attn.indexer.q_layernorm.weight") == (
        "model.layers.3.self_attn.indexer.q_layernorm.weight"
    )


def test_qwen4_projection_fusion_order():
    buffers = {}
    base = "model.layers.3.self_attn."
    parts = [
        ("q_proj.weight", torch.full((2, 3), 1.0)),
        ("k_proj.weight", torch.full((1, 3), 2.0)),
        ("v_proj.weight", torch.full((1, 3), 3.0)),
    ]
    assert _try_fuse(base + parts[0][0], parts[0][1], buffers) == ()
    assert _try_fuse(base + parts[1][0], parts[1][1], buffers) == ()
    name, fused = _try_fuse(base + parts[2][0], parts[2][1], buffers)
    assert name == base + "qkv_proj.weight"
    assert fused[:, 0].tolist() == [1.0, 1.0, 2.0, 3.0]


def test_ngram_hash_resets_at_eos():
    tokens = torch.tensor([4, 5, 99, 6, 7])
    multipliers = torch.tensor([3, 5, 7])
    sizes = torch.tensor([101, 103])
    offsets = torch.tensor([0, 101])
    ids = build_ngram_ids(
        tokens,
        ngram_size=3,
        heads_per_ngram=1,
        eos_token_id=99,
        multipliers=multipliers,
        vocab_sizes=sizes,
        offsets=offsets,
    )
    assert ids.shape == (5, 2)
    expected_bigram_after_eos = (6 * 3) ^ (99 * 5)
    assert ids[3, 0].item() == expected_bigram_after_eos % 101


def test_ple_request_tokens_uses_complete_prefill_history():
    req = SimpleNamespace(
        input_ids=torch.tensor([11, 12, 13]),
        cached_len=0,
        device_len=3,
        extend_len=3,
    )
    assert _ple_request_tokens(req).tolist() == [11, 12, 13]


def test_ple_request_tokens_joins_overlap_decode_token():
    req = SimpleNamespace(
        input_ids=torch.tensor([11, 12]),
        cached_len=2,
        device_len=3,
        extend_len=1,
    )
    assert _ple_request_tokens(req, torch.tensor([13])).tolist() == [11, 12, 13]


def test_ple_request_tokens_rejects_noncontiguous_host_history():
    req = SimpleNamespace(
        input_ids=torch.tensor([11]),
        cached_len=2,
        device_len=3,
        extend_len=1,
    )
    with pytest.raises(RuntimeError, match="unexpected gap"):
        _ple_request_tokens(req, torch.tensor([13]))


def test_gdn_reference_matches_token_and_chunk_execution() -> None:
    generator = torch.Generator().manual_seed(11)
    query = torch.randn(7, 2, 4, generator=generator)
    key = torch.randn(7, 2, 4, generator=generator)
    value = torch.randn(7, 2, 4, generator=generator)
    log_decay = -torch.rand(7, 2, generator=generator)
    beta = torch.sigmoid(torch.randn(7, 2, generator=generator))

    expected, expected_state = gated_delta_reference(query, key, value, log_decay, beta)
    actual, actual_state = gated_delta_chunked_reference(
        query, key, value, log_decay, beta, (2, 1, 4)
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state, expected_state)


def test_hyperconnection_reference_preserves_stream_shapes() -> None:
    generator = torch.Generator().manual_seed(12)
    streams, hidden = 4, 8
    value = torch.randn(3, streams * hidden, generator=generator)
    mixed, residual, inject = hyperconnection_reference(
        value,
        stream_count=streams,
        hidden_size=hidden,
        norm_weight=torch.randn(streams * hidden, generator=generator),
        down_weight=torch.randn(5, streams * hidden, generator=generator),
        up_weight=torch.randn(streams * hidden, 5, generator=generator),
        inject_weight=torch.randn(streams, streams * hidden, generator=generator),
        eps=1e-6,
    )

    assert mixed.shape == (3, hidden)
    assert residual.shape == (3, streams * hidden)
    assert inject.shape == (3, streams)
    combined = residual + (mixed.unsqueeze(1) * inject.unsqueeze(-1)).flatten(1)
    assert combined.shape == value.shape


def test_router_reference_reports_ids_weights_and_shared_semantics() -> None:
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    logits = torch.tensor([[0.0, 3.0, 1.0], [4.0, 0.0, 2.0]])
    experts = [lambda value, scale=scale: value * scale for scale in (1.0, 2.0, 3.0)]
    shared = hidden * 0.5
    shared_gate = torch.tensor([[0.0], [1.0]])

    output, expert_ids, weights = routed_shared_expert_reference(
        hidden,
        logits,
        experts,
        topk=2,
        shared_output=shared,
        shared_gate=shared_gate,
        renormalize=False,
    )

    assert expert_ids.tolist() == [[1, 2], [0, 2]]
    torch.testing.assert_close(weights, torch.softmax(logits, dim=-1).gather(1, expert_ids.long()))
    assert output.shape == hidden.shape


def _ple_reference_layer(weight: torch.Tensor) -> _PLELayer:
    layer = object.__new__(_PLELayer)
    layer.conv1d = SimpleNamespace(weight=weight)
    layer.dilation = 2
    layer.state_len = (weight.shape[-1] - 1) * layer.dilation
    layer._conv_states = {}
    return layer


def test_ple_state_matches_chunked_execution_and_resets_reused_slot(monkeypatch) -> None:
    weight = torch.tensor([[[0.25, -0.5, 0.75]], [[-0.1, 0.2, 0.3]]])
    values = torch.arange(10, dtype=torch.float32).view(5, 2) / 10

    full = _ple_reference_layer(weight)
    full_batch = SimpleNamespace(
        is_decode=False,
        reqs=[SimpleNamespace(extend_len=5, cached_len=0, table_idx=3)],
    )
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.model.get_global_ctx",
        lambda: SimpleNamespace(batch=full_batch),
    )
    expected = full._short_conv(values)

    chunked = _ple_reference_layer(weight)
    first_batch = SimpleNamespace(
        is_decode=False,
        reqs=[SimpleNamespace(extend_len=2, cached_len=0, table_idx=3)],
    )
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.model.get_global_ctx",
        lambda: SimpleNamespace(batch=first_batch),
    )
    first = chunked._short_conv(values[:2])
    second_batch = SimpleNamespace(
        is_decode=False,
        reqs=[SimpleNamespace(extend_len=3, cached_len=2, table_idx=3)],
    )
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.model.get_global_ctx",
        lambda: SimpleNamespace(batch=second_batch),
    )
    second = chunked._short_conv(values[2:])
    torch.testing.assert_close(torch.cat((first, second)), expected)

    reused = values[:1].neg()
    reset_batch = SimpleNamespace(
        is_decode=False,
        reqs=[SimpleNamespace(extend_len=1, cached_len=0, table_idx=3)],
    )
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.model.get_global_ctx",
        lambda: SimpleNamespace(batch=reset_batch),
    )
    actual_reset = chunked._short_conv(reused)
    fresh = _ple_reference_layer(weight)
    expected_reset = fresh._short_conv(reused)
    torch.testing.assert_close(actual_reset, expected_reset)


def test_qwen4_vision_entrypoint_fails_with_v1_scope_message() -> None:
    model = object.__new__(Qwen4ExpForCausalLM)
    with pytest.raises(RuntimeError, match="outside FreeToken-Pascal v1"):
        model.encode_images(torch.empty(0), torch.empty(0))


def test_qwen4_debug_hook_is_opt_in_and_captures_logits_and_state(monkeypatch) -> None:
    model = object.__new__(Qwen4ExpForCausalLM)
    model.model = SimpleNamespace(
        forward=lambda input_ids: input_ids.float().unsqueeze(-1),
        debug_state=lambda: {1: {7: torch.tensor([3.0])}},
    )
    model.lm_head = SimpleNamespace(forward=lambda hidden: torch.cat((hidden, -hidden), dim=-1))
    model._debug_hook = None
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.model.get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(input_ids=torch.tensor([2, 4]))),
    )

    logits = model.forward()
    captured = []
    model.set_debug_hook(captured.append)
    hooked_logits = model.forward()

    torch.testing.assert_close(hooked_logits, logits)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0]["logits"], logits)
    assert captured[0]["ple_state"][1][7].item() == 3.0
