from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Tuple

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    SlotStateSpec,
)


@dataclass(frozen=True)
class Qwen4ExpArgs:
    """Qwen3.8-Flash-Next geometry carried in ``ModelConfig.qwen4_args``."""

    hidden_size: int
    hc_count: int
    hc_lowrank: int
    ple_layer_ids: Tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    make_ngram_vocab_size_divisible_by: int
    split_ngram_parts: int
    ngram_boundary_token_id: int
    index_n_heads: int
    index_kv_heads: int
    index_head_dim: int
    index_budget: int
    index_ratio: int

    @property
    def index_topk_blocks(self) -> int:
        return self.index_budget // self.index_ratio

    @property
    def num_ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ngram_head_dim(self) -> int:
        return self.ple_embed_dim // self.num_ngram_heads

    @property
    def ple_conv_dilation(self) -> int:
        return self.ngram_size

    @property
    def ple_conv_state_len(self) -> int:
        return (self.ple_conv_kernel_size - 1) * self.ple_conv_dilation

    @property
    def ple_state_width(self) -> int:
        return self.hc_count * self.hidden_size

    # Compatibility aliases for the pre-#257 downstream Qwen adapter and GGUF converter.
    @property
    def eos_token_id(self) -> int:
        return self.ngram_boundary_token_id

    @property
    def indexer_n_heads(self) -> int:
        return self.index_n_heads

    @property
    def indexer_kv_heads(self) -> int:
        return self.index_kv_heads

    @property
    def indexer_head_dim(self) -> int:
        return self.index_head_dim

    @property
    def indexer_budget(self) -> int:
        return self.index_budget

    @property
    def indexer_compress_ratio(self) -> int:
        return self.index_ratio

    @property
    def output_gate_type(self) -> str:
        return "sigmoid"


PLE_CONV_STATE = "ple_conv"
PLE_NGRAM_STATE = "ple_ngram_ctx"


def ple_slot_states(args: Qwen4ExpArgs) -> Tuple[SlotStateSpec, ...]:
    """Declare the PLE conv history and rolling n-gram context on linear-state slots."""
    if not args.ple_layer_ids:
        return ()
    return (
        SlotStateSpec(
            name=PLE_CONV_STATE,
            shape=(args.ple_state_width, args.ple_conv_state_len),
            layer_ids=args.ple_layer_ids,
        ),
        SlotStateSpec(
            name=PLE_NGRAM_STATE,
            shape=(args.ngram_size - 1,),
            dtype=torch.int32,
            fill_value=float(args.ngram_boundary_token_id),
        ),
    )


def _quant_get(hf_config: Any):
    quant = getattr(hf_config, "quantization_config", None)
    if quant is None:
        return None
    return quant.get if isinstance(quant, dict) else (lambda key, default=None: getattr(quant, key, default))


def _ignored(patterns, module_name: str) -> bool:
    return any(fnmatch(module_name, pattern) for pattern in patterns)


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        return ["full_attention" if t == "qwen_sparse_attention" else t for t in layer_types]
    interval = int(getattr(text, "full_attention_interval", 4))
    return [
        "full_attention" if (index + 1) % interval == 0 else "linear_attention"
        for index in range(int(text.num_hidden_layers))
    ]


def _parse_quantization(hf_config: Any) -> tuple[str, str, str, str]:
    get = _quant_get(hf_config)
    if get is None:
        return "none", "none", "none", "none"
    algo = str(get("quant_algo") or get("quant_method") or "").lower()
    block = get("weight_block_size")
    if algo == "fp8" and block:
        block_size = tuple(int(value) for value in block)
        if block_size != (128, 128):
            raise ValueError(
                "Qwen4-Exp block-FP8 checkpoints require a 128x128 weight block size"
            )
        return "fp8_block", "none", "none", "none"
    if "fp4" not in algo:
        return "none", "none", "none", "none"
    ignore = list(get("ignore") or [])

    def quantized(probe: str) -> str:
        return "nvfp4" if not _ignored(ignore, probe) else "none"

    prefix = "model.language_model.layers.0"
    return (
        quantized(f"{prefix}.mlp.experts.0.gate_proj"),
        quantized(f"{prefix}.self_attn.q_proj"),
        quantized(f"{prefix}.mlp.shared_expert.gate_proj"),
        quantized("lm_head"),
    )


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)
    head_dim = int(getattr(text, "head_dim", 0) or text.hidden_size // text.num_attention_heads)
    num_kv_heads = int(getattr(text, "num_key_value_heads", text.num_attention_heads))
    rope_params = getattr(text, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
    partial = (
        rope_params.get("partial_rotary_factor")
        or getattr(text, "partial_rotary_factor", None)
        or 1.0
    )
    rotary_dim = int(head_dim * float(partial))
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {key: value for key, value in rope_params.items() if not isinstance(value, (list, dict))}
    )
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=int(text.max_position_embeddings),
        base=float(rope_theta),
        scaling=rope_scaling,
    )

    layer_types = _layer_types(text)
    unsupported = sorted(set(layer_types) - {"linear_attention", "full_attention"})
    if unsupported:
        raise ValueError(f"Unsupported Qwen4-Exp layer types: {unsupported}")
    full_ids = tuple(index for index, kind in enumerate(layer_types) if kind == "full_attention")
    linear_ids = tuple(index for index, kind in enumerate(layer_types) if kind == "linear_attention")

    ple_layer_ids = tuple(int(index) - 1 for index in (getattr(text, "ple_layer_ids", None) or ()))
    for layer_id in ple_layer_ids:
        if layer_id < 0 or layer_id >= len(layer_types) or layer_types[layer_id] != "linear_attention":
            raise ValueError(f"PLE must sit on a linear_attention layer, got layer {layer_id}")

    expert_quant, attn_quant, dense_quant, lm_head_quant = _parse_quantization(hf_config)
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=rotary,
        index_head_dim=int(text.indexer_head_dim),
        num_index_layers=len(full_ids),
        index_ratio=int(text.indexer_compress_ratio),
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=int(text.linear_num_key_heads),
        num_value_heads=int(text.linear_num_value_heads),
        key_head_dim=int(text.linear_key_head_dim),
        value_head_dim=int(text.linear_value_head_dim),
        conv_kernel_dim=int(text.linear_conv_kernel_dim),
        output_gate=str(getattr(text, "output_gate_type", None) or text.hidden_act),
    )
    groups = tuple(sorted((full_group, linear_group), key=lambda group: group.layer_ids[0] if group.layer_ids else 1 << 30))

    eos_token_id = getattr(text, "eos_token_id", 0)
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0]
    qwen4_args = Qwen4ExpArgs(
        hidden_size=int(text.hidden_size),
        hc_count=int(text.hc_count),
        hc_lowrank=int(text.hc_lowrank),
        ple_layer_ids=ple_layer_ids,
        ple_embed_dim=int(text.ple_embed_dim),
        ple_conv_kernel_size=int(text.ple_conv_kernel_size),
        ngram_size=int(text.ngram_size),
        heads_per_ngram=int(text.heads_per_ngram),
        ngram_vocab_size_base=int(text.ngram_vocab_size_base),
        make_ngram_vocab_size_divisible_by=int(getattr(text, "make_ngram_vocab_size_divisible_by", 1)),
        split_ngram_parts=int(text.split_ngram_parts),
        ngram_boundary_token_id=int(eos_token_id),
        index_n_heads=int(text.indexer_n_heads),
        index_kv_heads=int(text.indexer_kv_heads),
        index_head_dim=int(text.indexer_head_dim),
        index_budget=int(text.indexer_budget),
        index_ratio=int(text.indexer_compress_ratio),
    )
    return ModelConfig(
        num_layers=int(text.num_hidden_layers),
        num_qo_heads=int(text.num_attention_heads),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=int(text.hidden_size),
        vocab_size=int(text.vocab_size),
        intermediate_size=int(getattr(text, "intermediate_size", 0) or 0),
        hidden_act=str(text.hidden_act),
        rms_norm_eps=float(text.rms_norm_eps),
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=rotary,
        num_experts=int(getattr(text, "num_experts", 0) or 0),
        num_experts_per_tok=int(getattr(text, "num_experts_per_tok", 0) or 0),
        moe_intermediate_size=int(getattr(text, "moe_intermediate_size", 0) or 0),
        shared_expert_intermediate_size=int(getattr(text, "shared_expert_intermediate_size", 0) or 0),
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        moe_enabled=int(getattr(text, "num_experts", 0) or 0) > 0,
        use_qk_norm=True,
        model_type=getattr(hf_config, "model_type", "qwen4_exp"),
        architectures=getattr(hf_config, "architectures", ["Qwen4ExpForConditionalGeneration"]),
        vision_config=None,
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        expert_quant=expert_quant,
        attn_quant=attn_quant,
        dense_quant=dense_quant,
        lm_head_quant=lm_head_quant,
        qwen4_args=qwen4_args,
        slot_states=ple_slot_states(qwen4_args),
    )


def parse_gguf_config(shim: Any) -> ModelConfig:
    """Build a text-only Qwen4-Exp config from pinned llama.cpp GGUF metadata."""
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
    full_ids = tuple(index for index in range(num_layers) if (index + 1) % interval == 0)
    linear_ids = tuple(index for index in range(num_layers) if index not in full_ids)
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
    index_ratio = qsa_ratios.pop()
    if index_ratio <= 0:
        raise ValueError(f"invalid QSA compression ratio {index_ratio}")

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
        raise ValueError("Qwen4-Exp PLE ngram_size must be >= 2 and heads_per_ngram must be positive")
    head_vocab_sizes = tuple(int(value) for value in required("ple.head_vocab_sizes"))
    head_offsets = tuple(int(value) for value in required("ple.head_offsets"))
    layer_multipliers = tuple(int(value) for value in required("ple.layer_multipliers"))
    num_ngram_heads = (ngram_size - 1) * heads_per_ngram
    if len(head_vocab_sizes) != num_ngram_heads or len(head_offsets) != num_ngram_heads:
        raise ValueError("Qwen4-Exp PLE head metadata has the wrong length")
    if len(layer_multipliers) != ngram_size:
        raise ValueError("Qwen4-Exp PLE layer multiplier metadata has the wrong length")
    if any(size <= 0 for size in head_vocab_sizes):
        raise ValueError("Qwen4-Exp PLE head vocabulary sizes must be positive")
    if head_offsets[0] != 0 or any(
        head_offsets[index + 1] != head_offsets[index] + head_vocab_sizes[index]
        for index in range(num_ngram_heads - 1)
    ):
        raise ValueError("Qwen4-Exp PLE head offsets are not contiguous")
    ple_head_dim = int(required("embedding_length_per_layer_input"))
    qwen4_args = Qwen4ExpArgs(
        hidden_size=hidden,
        hc_count=int(required("hyper_connection.count")),
        hc_lowrank=int(required("hyper_connection.low_rank")),
        ple_layer_ids=ple_layers,
        ple_embed_dim=ple_head_dim * num_ngram_heads,
        ple_conv_kernel_size=int(required("ple.conv_kernel")),
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        ngram_vocab_size_base=min(head_vocab_sizes),
        make_ngram_vocab_size_divisible_by=1,
        split_ngram_parts=1,
        ngram_boundary_token_id=int(required("ple.eos_token_id")),
        index_n_heads=index_heads,
        index_kv_heads=1,
        index_head_dim=index_head_dim,
        index_budget=int(required("attention.indexer.top_k")),
        index_ratio=index_ratio,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=int(required("attention.head_count_kv")),
        head_dim=head_dim,
        rotary_config=rotary,
        index_head_dim=index_head_dim,
        num_index_layers=len(full_ids),
        index_ratio=index_ratio,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=linear_heads,
        num_value_heads=value_heads,
        key_head_dim=state_size,
        value_head_dim=state_size,
        conv_kernel_dim=int(required("ssm.conv_kernel")),
        output_gate="sigmoid",
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
        attention_groups=(linear_group, full_group),
        qwen4_args=qwen4_args,
        requires_naive_cache=True,
        supports_cuda_graph=False,
        moe_weight_format="gguf",
        gguf_model_path=shim.model_path,
    )


__all__ = [
    "PLE_CONV_STATE",
    "PLE_NGRAM_STATE",
    "Qwen4ExpArgs",
    "parse_config",
    "parse_gguf_config",
    "ple_slot_states",
]
