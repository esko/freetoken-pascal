from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    QSAAttentionGroupConfig,
    RotaryConfig,
    detect_expert_quant,
)

from .args import Qwen4ExpArgs


def parse_config(hf_config: Any) -> ModelConfig:
    text = hf_config.text_config
    layer_types = list(text.layer_types)
    sparse_attention_types = {"full_attention", "qwen_sparse_attention"}
    unsupported = sorted(set(layer_types) - {"linear_attention", *sparse_attention_types})
    if unsupported:
        raise ValueError(f"Unsupported Qwen4-Exp layer types: {unsupported}")

    head_dim = int(text.head_dim)
    rope = text.rope_parameters
    rotary_dim = round(head_dim * float(rope.get("partial_rotary_factor", 1.0)))
    indexer_budget = int(text.indexer_budget)
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=int(text.max_position_embeddings),
        base=float(rope["rope_theta"]),
        scaling=None,
    )

    full_ids = tuple(
        i for i, layer_type in enumerate(layer_types) if layer_type in sparse_attention_types
    )
    linear_ids = tuple(
        i for i, layer_type in enumerate(layer_types) if layer_type == "linear_attention"
    )
    groups = (
        LinearGatedDeltaGroupConfig(
            name="linear",
            layer_ids=linear_ids,
            num_key_heads=int(text.linear_num_key_heads),
            num_value_heads=int(text.linear_num_value_heads),
            key_head_dim=int(text.linear_key_head_dim),
            value_head_dim=int(text.linear_value_head_dim),
            conv_kernel_dim=int(text.linear_conv_kernel_dim),
            output_gate=True,
        ),
        QSAAttentionGroupConfig(
            name="qsa",
            layer_ids=full_ids,
            num_kv_heads=int(text.num_key_value_heads),
            head_dim=head_dim,
            rotary_config=rotary,
            index_num_heads=int(text.indexer_n_heads),
            index_num_kv_heads=int(text.indexer_kv_heads),
            index_head_dim=int(text.indexer_head_dim),
            index_token_budget=indexer_budget,
            index_compress_ratio=int(text.indexer_compress_ratio),
        ),
    )

    eos_token_id = text.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    qwen4_args = Qwen4ExpArgs(
        hc_count=int(text.hc_count),
        hc_lowrank=int(text.hc_lowrank),
        ple_layer_ids=tuple(int(layer_id) - 1 for layer_id in text.ple_layer_ids),
        ple_embed_dim=int(text.ple_embed_dim),
        ple_conv_kernel_size=int(text.ple_conv_kernel_size),
        ngram_size=int(text.ngram_size),
        heads_per_ngram=int(text.heads_per_ngram),
        ngram_vocab_size_base=int(text.ngram_vocab_size_base),
        split_ngram_parts=int(text.split_ngram_parts),
        eos_token_id=int(eos_token_id),
        indexer_n_heads=int(text.indexer_n_heads),
        indexer_kv_heads=int(text.indexer_kv_heads),
        indexer_head_dim=int(text.indexer_head_dim),
        indexer_budget=indexer_budget,
        indexer_compress_ratio=int(text.indexer_compress_ratio),
        output_gate_type=str(text.output_gate_type or text.hidden_act),
    )

    quant = getattr(hf_config, "quantization_config", None)
    get_quant = (
        quant.get
        if isinstance(quant, dict)
        else (lambda key, default=None: getattr(quant, key, default))
    )
    detected_quant = detect_expert_quant(hf_config)
    if detected_quant == "fp8":
        raw_block_size = get_quant("weight_block_size")
        block_size = (
            tuple(int(value) for value in raw_block_size) if raw_block_size is not None else None
        )
        if block_size != (128, 128):
            raise ValueError("Qwen4-Exp block-FP8 checkpoints require a 128x128 weight block size")
        expert_quant = "fp8_block"
    elif detected_quant == "nvfp4":
        # RadixArk's ModelOpt checkpoint quantizes only the routed experts. The
        # attention, GDN, mHC, shared experts, router, embeddings, and lm_head
        # remain BF16; PLE remains its source FP8 format. Vision is not loaded
        # by the downstream text-only runtime.
        block_size = None
        expert_quant = "nvfp4"
    else:
        raise ValueError(
            "Qwen4-Exp requires routed experts in 128x128 block-FP8 or ModelOpt NVFP4; "
            f"detected {detected_quant!r}"
        )

    return ModelConfig(
        num_layers=int(text.num_hidden_layers),
        num_qo_heads=int(text.num_attention_heads),
        num_kv_heads=int(text.num_key_value_heads),
        head_dim=head_dim,
        hidden_size=int(text.hidden_size),
        vocab_size=int(text.vocab_size),
        intermediate_size=int(getattr(text, "intermediate_size", 0) or 0),
        hidden_act=str(text.hidden_act),
        rms_norm_eps=float(text.rms_norm_eps),
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=rotary,
        num_experts=int(text.num_experts),
        num_experts_per_tok=int(text.num_experts_per_tok),
        moe_intermediate_size=int(text.moe_intermediate_size),
        shared_expert_intermediate_size=int(text.shared_expert_intermediate_size),
        # The released Qwen3.8-Flash-Next configs omit this older Qwen MoE
        # field. Omission means that the router weights are not renormalized.
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", False)),
        model_type=str(hf_config.model_type),
        architectures=list(hf_config.architectures),
        moe_enabled=True,
        expert_quant=expert_quant,
        weight_block_size=block_size,
        # Only routed experts and PLE are FP8 in the official checkpoint. All
        # attention, hyper-connection, and shared-expert projections stay BF16.
        attn_quant="none",
        dense_quant="none",
        lm_head_quant="none",
        use_qk_norm=True,
        # The released checkpoint also carries a vision tower. FreeToken-Pascal v1
        # deliberately loads only the language backbone and rejects image inputs.
        vision_config=None,
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        qwen4_args=qwen4_args,
        # PLE keeps per-request dilated-convolution state outside the generic
        # radix cache and performs mmap-backed CPU gathers during every forward.
        requires_naive_cache=True,
        supports_cuda_graph=False,
    )


def parse_gguf_config(shim: Any) -> ModelConfig:
    """Build the text-only Qwen4-Exp config from pinned llama.cpp GGUF metadata."""
    metadata = shim.metadata
    prefix = "qwen4exp."

    def required(key: str):
        full_key = prefix + key
        if full_key not in metadata:
            raise KeyError(f"missing GGUF metadata key {full_key}")
        return metadata[full_key]

    num_layers = int(required("block_count"))
    interval = int(required("full_attention_interval"))
    if interval <= 0:
        raise ValueError(f"qwen4exp.full_attention_interval must be positive, got {interval}")
    full_ids = tuple(layer for layer in range(num_layers) if (layer + 1) % interval == 0)
    linear_ids = tuple(layer for layer in range(num_layers) if layer not in full_ids)
    head_dim = int(required("attention.key_length"))
    rotary_dim = int(required("rope.dimension_count"))
    rope_sections = [int(value) for value in required("rope.dimension_sections")]
    if sum(rope_sections) * 2 != rotary_dim:
        raise ValueError(
            "qwen4exp.rope.dimension_sections do not cover rope.dimension_count: "
            f"{rope_sections} vs {rotary_dim}"
        )
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=int(required("context_length")),
        base=float(required("rope.freq_base")),
        scaling=None,
    )

    compress_ratios = [int(value) for value in required("attention.compress_ratios")]
    if len(compress_ratios) != num_layers:
        raise ValueError("qwen4exp.attention.compress_ratios length != block_count")
    qsa_ratios = {compress_ratios[layer] for layer in full_ids}
    if len(qsa_ratios) != 1:
        raise ValueError(f"QSA layers have heterogeneous compression ratios: {qsa_ratios}")
    index_compress_ratio = qsa_ratios.pop()
    if index_compress_ratio <= 0:
        raise ValueError(f"invalid QSA compression ratio {index_compress_ratio}")

    hidden = int(required("embedding_length"))
    linear_heads = int(required("ssm.group_count"))
    value_heads = int(required("ssm.time_step_rank"))
    state_size = int(required("ssm.state_size"))
    index_heads = int(required("attention.indexer.head_count"))
    index_head_dim = int(required("attention.indexer.key_length"))
    ple_layers = tuple(int(layer) for layer in required("ple.layers"))
    if any(layer < 0 or layer >= num_layers for layer in ple_layers):
        raise ValueError(f"qwen4exp.ple.layers out of range: {ple_layers}")
    heads_per_ngram = int(required("ple.heads_per_ngram"))
    ngram_size = int(required("ple.ngram_size"))
    if ngram_size < 2 or heads_per_ngram < 1:
        raise ValueError(
            "Qwen4-Exp PLE ngram_size must be >= 2 and heads_per_ngram must be positive"
        )
    head_vocab_sizes = [int(value) for value in required("ple.head_vocab_sizes")]
    head_offsets = [int(value) for value in required("ple.head_offsets")]
    layer_multipliers = [int(value) for value in required("ple.layer_multipliers")]
    expected_ngram_heads = (ngram_size - 1) * heads_per_ngram
    if len(head_vocab_sizes) != expected_ngram_heads or len(head_offsets) != expected_ngram_heads:
        raise ValueError("Qwen4-Exp PLE head metadata has the wrong length")
    if len(layer_multipliers) != ngram_size:
        raise ValueError("Qwen4-Exp PLE layer multiplier metadata has the wrong length")
    if any(size <= 0 for size in head_vocab_sizes):
        raise ValueError("Qwen4-Exp PLE head vocabulary sizes must be positive")
    if head_offsets[0] != 0 or any(
        head_offsets[index + 1] != head_offsets[index] + head_vocab_sizes[index]
        for index in range(expected_ngram_heads - 1)
    ):
        raise ValueError("Qwen4-Exp PLE head offsets are not contiguous")
    ple_head_dim = int(required("embedding_length_per_layer_input"))

    qwen4_args = Qwen4ExpArgs(
        hc_count=int(required("hyper_connection.count")),
        hc_lowrank=int(required("hyper_connection.low_rank")),
        ple_layer_ids=ple_layers,
        ple_embed_dim=ple_head_dim * expected_ngram_heads,
        ple_conv_kernel_size=int(required("ple.conv_kernel")),
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        ngram_vocab_size_base=min(head_vocab_sizes),
        # GGUF stores the PLE bank as one contiguous mmap-able tensor. Issue #13
        # installs that source instead of the safetensors shard loader.
        split_ngram_parts=1,
        eos_token_id=int(required("ple.eos_token_id")),
        indexer_n_heads=index_heads,
        indexer_kv_heads=1,
        indexer_head_dim=index_head_dim,
        indexer_budget=int(required("attention.indexer.top_k")),
        indexer_compress_ratio=index_compress_ratio,
        output_gate_type="sigmoid",
        ple_layer_multipliers=tuple(layer_multipliers),
        ple_head_vocab_sizes=tuple(head_vocab_sizes),
        ple_head_offsets=tuple(head_offsets),
    )
    groups = (
        LinearGatedDeltaGroupConfig(
            name="linear",
            layer_ids=linear_ids,
            num_key_heads=linear_heads,
            num_value_heads=value_heads,
            key_head_dim=state_size,
            value_head_dim=state_size,
            conv_kernel_dim=int(required("ssm.conv_kernel")),
            output_gate=True,
        ),
        QSAAttentionGroupConfig(
            name="qsa",
            layer_ids=full_ids,
            num_kv_heads=int(required("attention.head_count_kv")),
            head_dim=head_dim,
            rotary_config=rotary,
            index_num_heads=index_heads,
            index_num_kv_heads=1,
            index_head_dim=index_head_dim,
            index_token_budget=qwen4_args.indexer_budget,
            index_compress_ratio=index_compress_ratio,
        ),
    )
    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=int(required("attention.head_count")),
        num_kv_heads=int(required("attention.head_count_kv")),
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        intermediate_size=0,
        hidden_act="silu",
        rms_norm_eps=float(required("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=rotary,
        num_experts=int(required("expert_count")),
        num_experts_per_tok=int(required("expert_used_count")),
        moe_intermediate_size=int(required("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(required("expert_shared_feed_forward_length")),
        norm_topk_prob=False,
        model_type="qwen4_exp",
        architectures=list(shim.architectures),
        moe_enabled=True,
        expert_quant="gguf",
        attn_quant="gguf",
        dense_quant="gguf",
        lm_head_quant="gguf",
        use_qk_norm=True,
        vision_config=None,
        image_token_id=(
            int(metadata["qwen4exp.ple.image_token_id"])
            if "qwen4exp.ple.image_token_id" in metadata
            else None
        ),
        attention_groups=groups,
        qwen4_args=qwen4_args,
        requires_naive_cache=True,
        supports_cuda_graph=False,
        moe_weight_format="gguf",
        gguf_model_path=shim.model_path,
    )


__all__ = ["parse_config", "parse_gguf_config"]
