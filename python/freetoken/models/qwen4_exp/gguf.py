"""Qwen4-Exp GGUF config-independent weight mapping and converter reversals."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

from freetoken.gguf_shards import gguf_reader, gguf_shard_paths
from freetoken.models.gguf.dequant import dequantize

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def _quant_map(model_path: str) -> dict[tuple[int, str], int]:
    result: dict[tuple[int, str], int] = {}
    for shard in gguf_shard_paths(model_path):
        for tensor in gguf_reader(str(shard)).tensors:
            if tensor.name.startswith("blk."):
                _, layer, suffix = tensor.name.split(".", 2)
                result[(int(layer), suffix)] = int(tensor.tensor_type)
            else:
                result[(-1, tensor.name)] = int(tensor.tensor_type)
    return result


def _ungroup_v(
    tensor: torch.Tensor,
    dim: int,
    num_key_heads: int,
    values_per_key: int,
    head_dim: int,
) -> torch.Tensor:
    """Invert llama.cpp's grouped-to-tiled V-head converter transform."""
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    expected = num_key_heads * values_per_key * head_dim
    if shape[dim] != expected:
        raise ValueError(
            f"axis {dim} has {shape[dim]} elements, expected {expected} for V-head reorder"
        )
    view = [
        *shape[:dim],
        values_per_key,
        num_key_heads,
        head_dim,
        *shape[dim + 1 :],
    ]
    expanded = tensor.reshape(*view)
    permutation = list(range(len(view)))
    permutation[dim], permutation[dim + 1] = permutation[dim + 1], permutation[dim]
    return expanded.permute(*permutation).contiguous().reshape(*shape)


def _grouped_to_tiled_indices(
    num_key_heads: int,
    values_per_key: int,
    head_dim: int,
) -> torch.Tensor:
    grouped = torch.arange(num_key_heads * values_per_key * head_dim)
    return grouped.reshape(num_key_heads, values_per_key, head_dim).permute(1, 0, 2).flatten()


def _to_dense(tensor, dtype: torch.dtype) -> torch.Tensor:
    values = dequantize(tensor.packed().reshape(-1), tensor.ggml_type, dtype)
    return values.reshape(tensor.shape)


def _centered_norm(tensor) -> torch.Tensor:
    # llama.cpp folds Qwen's effective (1 + w) norm scale into GGUF.
    # Subtract in fp32 before narrowing.  Casting an effective scale near one to
    # bf16 first can round it to exactly one and erase the learned centered
    # offset that GemmaPlusOneRMSNorm expects.
    return (_to_dense(tensor, torch.float32) - 1.0).to(torch.bfloat16)


def _require_tp1(what: str) -> None:
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(
            f"Qwen4-Exp GGUF {what} currently supports TP=1 only; "
            "Issue #16 owns deterministic TP=2 GGUF placement"
        )


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield every resident Qwen4-Exp tensor, reversing llama.cpp layout transforms."""
    del device
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    _require_tp1("weight loading")
    if include_moe_experts:
        raise ValueError("Qwen4-Exp GGUF routed experts must use the host expert bank")
    if not include_non_moe:
        return
    shim = cached_load_hf_config(model_path)
    from .config import parse_gguf_config

    config = parse_gguf_config(shim)
    linear_group = config.linear_attention_group()
    if linear_group is None:
        raise ValueError("Qwen4-Exp GGUF has no linear-attention group")
    args = config.qwen4_args
    ple_constants = (
        args.ple_layer_multipliers,
        args.ple_head_vocab_sizes,
        args.ple_head_offsets,
    )
    if args.ple_layer_ids and any(value is None for value in ple_constants):
        raise ValueError("Qwen4-Exp GGUF is missing parsed PLE hash constants")
    if args.ple_layer_ids:
        multipliers, vocab_sizes, offsets = ple_constants
        assert multipliers is not None and vocab_sizes is not None and offsets is not None
        for layer_id in args.ple_layer_ids:
            prefix = f"model.layers.{layer_id}.ple.ple_embedding"
            yield f"{prefix}.layer_multipliers", torch.tensor(multipliers, dtype=torch.int64)
            yield f"{prefix}.ngram_heads_vocab_sizes", torch.tensor(vocab_sizes, dtype=torch.int64)
            yield f"{prefix}.ngram_heads_offsets", torch.tensor(offsets, dtype=torch.int64)
    values_per_key = linear_group.num_value_heads // linear_group.num_key_heads
    qk_rows = 2 * linear_group.num_key_heads * linear_group.key_head_dim
    value_rows = linear_group.num_value_heads * linear_group.value_head_dim
    conv_rows = qk_rows + value_rows
    full_layers = {layer for layer in range(config.num_layers) if not config.is_linear_layer(layer)}
    buffers: dict[str, dict[str, Any]] = {}

    def packed_group(
        name: str,
        slot: str,
        slots: tuple[str, ...],
        target: str,
        tensor,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        group = buffers.setdefault(name, {})
        group[slot] = tensor
        if all(part in group for part in slots):
            types = [group[part].ggml_type for part in slots]
            if len(set(types)) == 1:
                yield (
                    target + ".qweight",
                    torch.cat([group[part].packed() for part in slots], dim=0),
                )
            else:
                for index, part in enumerate(slots):
                    yield f"{target}.qweight_{index}", group[part].packed()
            del buffers[name]

    for tensor in iter_gguf_tensors(model_path):
        name = tensor.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", tensor.packed()
            continue
        if name == "output.weight":
            yield "lm_head.qweight", tensor.packed()
            continue
        if name == "per_layer_token_embd.weight":
            continue
        global_map = {
            "output_hc_down.weight": "model.hyper_connection_mixer.input_mix_weight_down.qweight",
            "output_hc_up.weight": "model.hyper_connection_mixer.input_mix_weight_up.qweight",
        }
        if name in global_map:
            yield global_map[name], tensor.packed()
            continue
        if name == "output_hc_norm.weight":
            yield "model.hyper_connection_mixer.hc_norm.weight", _centered_norm(tensor)
            continue
        if not name.startswith("blk."):
            raise ValueError(f"unmapped Qwen4-Exp GGUF tensor: {name}")

        _, layer_text, suffix = name.split(".", 2)
        layer = int(layer_text)
        if layer >= config.num_layers:
            raise ValueError(f"unexpected Qwen4-Exp GGUF layer {layer}: {name}")
        base = f"model.layers.{layer}"
        if suffix in {
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        }:
            continue

        dense_map = {
            "ffn_gate_inp.weight": f"{base}.mlp.gate.weight",
            "ffn_gate_inp_shexp.weight": f"{base}.mlp.shared_expert_gate.weight",
            "hc_attn_inject.weight": f"{base}.attn_hyper_connection.block_inject_weight.weight",
            "hc_ffn_inject.weight": f"{base}.mlp_hyper_connection.block_inject_weight.weight",
        }
        if suffix in dense_map:
            value = _to_dense(tensor, torch.bfloat16)
            if suffix == "ffn_gate_inp_shexp.weight":
                value = value.reshape(1, -1)
            yield dense_map[suffix], value
            continue

        norm_map = {
            "hc_attn_norm.weight": f"{base}.attn_hyper_connection.hc_norm.weight",
            "hc_ffn_norm.weight": f"{base}.mlp_hyper_connection.hc_norm.weight",
            "ssm_norm.weight": f"{base}.linear_attn.norm.weight",
            "attn_q_norm.weight": f"{base}.self_attn.q_norm.weight",
            "attn_k_norm.weight": f"{base}.self_attn.k_norm.weight",
            "indexer.q_norm.weight": f"{base}.self_attn.indexer.q_layernorm.weight",
            "indexer.k_norm.weight": f"{base}.self_attn.indexer.k_layernorm.weight",
            "ple_norm_key.weight": f"{base}.ple.norm_key.weight",
            "ple_norm_query.weight": f"{base}.ple.norm_query.weight",
            "ple_norm_conv.weight": f"{base}.ple.norm_conv.weight",
        }
        if suffix in norm_map:
            yield norm_map[suffix], _centered_norm(tensor)
            continue

        packed_map = {
            "hc_attn_down.weight": f"{base}.attn_hyper_connection.input_mix_weight_down.qweight",
            "hc_attn_up.weight": f"{base}.attn_hyper_connection.input_mix_weight_up.qweight",
            "hc_ffn_down.weight": f"{base}.mlp_hyper_connection.input_mix_weight_down.qweight",
            "hc_ffn_up.weight": f"{base}.mlp_hyper_connection.input_mix_weight_up.qweight",
            "ffn_down_shexp.weight": f"{base}.mlp.shared_expert.down_proj.qweight",
            "ple_key.weight": f"{base}.ple.key_proj.qweight",
            "ple_value.weight": f"{base}.ple.value_proj.qweight",
        }
        if suffix in packed_map:
            yield packed_map[suffix], tensor.packed()
            continue

        if suffix in {"ffn_gate_shexp.weight", "ffn_up_shexp.weight"}:
            slot = "gate" if suffix.startswith("ffn_gate") else "up"
            yield from packed_group(
                f"shared.{layer}",
                slot,
                ("gate", "up"),
                f"{base}.mlp.shared_expert.gate_up_proj",
                tensor,
            )
            continue

        if suffix == "ple_conv1d.weight":
            value = _to_dense(tensor, torch.bfloat16).reshape(
                config.hidden_size * config.qwen4_args.hc_count,
                1,
                config.qwen4_args.ple_conv_kernel_size,
            )
            yield f"{base}.ple.conv1d.weight", value
            continue

        if layer in full_layers:
            if suffix in {"attn_q.weight", "attn_k.weight", "attn_v.weight"}:
                slot = suffix.split("_")[1].split(".")[0]
                yield from packed_group(
                    f"qkv.{layer}",
                    slot,
                    ("q", "k", "v"),
                    f"{base}.self_attn.qkv_proj",
                    tensor,
                )
                continue
            if suffix == "attn_output.weight":
                yield f"{base}.self_attn.o_proj.qweight", tensor.packed()
                continue
            if suffix in {"indexer.q_proj.weight", "indexer.k_proj.weight"}:
                slot = "q" if ".q_" in suffix else "k"
                yield from packed_group(
                    f"indexer.{layer}",
                    slot,
                    ("q", "k"),
                    f"{base}.self_attn.indexer.index_qk_proj",
                    tensor,
                )
                continue
        else:
            if suffix == "attn_qkv.weight":
                packed = tensor.packed()
                qk, values = packed[:qk_rows], packed[qk_rows:]
                tensor = _replace_packed(
                    tensor,
                    torch.cat(
                        [
                            qk,
                            _ungroup_v(
                                values,
                                0,
                                linear_group.num_key_heads,
                                values_per_key,
                                linear_group.value_head_dim,
                            ),
                        ]
                    ),
                )
                slot = "qkv"
            elif suffix == "attn_gate.weight":
                tensor = _replace_packed(
                    tensor,
                    _ungroup_v(
                        tensor.packed(),
                        0,
                        linear_group.num_key_heads,
                        values_per_key,
                        linear_group.value_head_dim,
                    ),
                )
                slot = "gate"
            elif suffix in {"ssm_beta.weight", "ssm_alpha.weight"}:
                tensor = _replace_packed(
                    tensor,
                    _ungroup_v(
                        tensor.packed(),
                        0,
                        linear_group.num_key_heads,
                        values_per_key,
                        1,
                    ),
                )
                slot = "beta" if suffix.startswith("ssm_beta") else "alpha"
            else:
                slot = ""
            if slot:
                yield from packed_group(
                    f"gdn.{layer}",
                    slot,
                    ("qkv", "gate", "beta", "alpha"),
                    f"{base}.linear_attn.in_proj",
                    tensor,
                )
                continue
            if suffix == "ssm_a":
                value = _to_dense(tensor, torch.float32)
                value = _ungroup_v(value, 0, linear_group.num_key_heads, values_per_key, 1)
                if not bool((value < 0).all()):
                    raise ValueError(f"{name}: expected stored A=-exp(A_log) to be negative")
                yield f"{base}.linear_attn.A_log", torch.log(-value)
                continue
            if suffix == "ssm_dt.bias":
                value = _ungroup_v(
                    _to_dense(tensor, torch.float32),
                    0,
                    linear_group.num_key_heads,
                    values_per_key,
                    1,
                )
                yield f"{base}.linear_attn.dt_bias", value
                continue
            if suffix == "ssm_conv1d.weight":
                value = _to_dense(tensor, torch.bfloat16).reshape(
                    conv_rows, linear_group.conv_kernel_dim
                )
                qk, values = value[:qk_rows], value[qk_rows:]
                value = torch.cat(
                    [
                        qk,
                        _ungroup_v(
                            values,
                            0,
                            linear_group.num_key_heads,
                            values_per_key,
                            linear_group.value_head_dim,
                        ),
                    ]
                ).reshape(conv_rows, 1, linear_group.conv_kernel_dim)
                yield f"{base}.linear_attn.conv1d.weight", value
                continue
            if suffix == "ssm_out.weight":
                yield f"{base}.linear_attn.out_proj.qweight", tensor.packed()
                continue

        raise ValueError(f"unmapped Qwen4-Exp GGUF tensor: {name}")

    if buffers:
        raise ValueError(f"incomplete Qwen4-Exp GGUF projection groups: {sorted(buffers)}")


def _replace_packed(tensor, packed: torch.Tensor):
    from dataclasses import replace

    return replace(tensor, _raw=packed.numpy())


def convert_qwen4_to_gguf(model, config: ModelConfig, *, model_path: str) -> None:
    """Replace resident dense layers with exact native-GGUF packed operators."""
    from freetoken.layers.gguf import (
        GGUFEmbedding,
        GGUFInputPermutedLinear,
        GGUFLinear,
        GGUFLMHead,
        gguf_merged_or_plain,
    )

    _require_tp1("operator construction")
    quant_types = _quant_map(model_path)

    def quant(layer: int, suffix: str) -> int:
        try:
            return quant_types[(layer, suffix)]
        except KeyError as error:
            name = suffix if layer < 0 else f"blk.{layer}.{suffix}"
            raise ValueError(f"GGUF checkpoint lacks required tensor {name}") from error

    def swap(owner, attribute: str, layer: int, suffix: str) -> None:
        linear = getattr(owner, attribute)
        out_features, in_features = linear.weight.shape
        setattr(
            owner,
            attribute,
            GGUFLinear(
                in_features,
                out_features,
                quant(layer, suffix),
                has_bias=linear.bias is not None,
            ),
        )

    inner = model.model
    inner.embed_tokens = GGUFEmbedding(
        config.vocab_size,
        config.hidden_size,
        quant(-1, "token_embd.weight"),
    )
    model.lm_head = GGUFLMHead(
        config.hidden_size,
        config.vocab_size,
        quant(-1, "output.weight"),
    )
    swap(
        inner.hyper_connection_mixer,
        "input_mix_weight_down",
        -1,
        "output_hc_down.weight",
    )
    swap(
        inner.hyper_connection_mixer,
        "input_mix_weight_up",
        -1,
        "output_hc_up.weight",
    )

    linear_group = config.linear_attention_group()
    if linear_group is None:
        raise ValueError("Qwen4-Exp GGUF has no linear-attention group")
    values_per_key = linear_group.num_value_heads // linear_group.num_key_heads
    value_dim = linear_group.num_value_heads * linear_group.value_head_dim
    input_permutation = _grouped_to_tiled_indices(
        linear_group.num_key_heads,
        values_per_key,
        linear_group.value_head_dim,
    )

    for layer_index, layer in enumerate(inner.layers.op_list):
        for hyper, prefix in (
            (layer.attn_hyper_connection, "hc_attn"),
            (layer.mlp_hyper_connection, "hc_ffn"),
        ):
            swap(hyper, "input_mix_weight_down", layer_index, f"{prefix}_down.weight")
            swap(hyper, "input_mix_weight_up", layer_index, f"{prefix}_up.weight")

        width = config.shared_expert_intermediate_size
        layer.mlp.shared_expert.gate_up_proj = gguf_merged_or_plain(
            config.hidden_size,
            [width, width],
            [
                quant(layer_index, "ffn_gate_shexp.weight"),
                quant(layer_index, "ffn_up_shexp.weight"),
            ],
        )
        swap(
            layer.mlp.shared_expert,
            "down_proj",
            layer_index,
            "ffn_down_shexp.weight",
        )

        if config.is_linear_layer(layer_index):
            splits = list(layer.linear_attn._in_proj_split)
            layer.linear_attn.in_proj = gguf_merged_or_plain(
                config.hidden_size,
                splits,
                [
                    quant(layer_index, "attn_qkv.weight"),
                    quant(layer_index, "attn_gate.weight"),
                    quant(layer_index, "ssm_beta.weight"),
                    quant(layer_index, "ssm_alpha.weight"),
                ],
            )
            layer.linear_attn.out_proj = GGUFInputPermutedLinear(
                value_dim,
                config.hidden_size,
                quant(layer_index, "ssm_out.weight"),
                input_permutation,
            )
        else:
            layer.self_attn.qkv_proj = gguf_merged_or_plain(
                config.hidden_size,
                list(layer.self_attn._qkv_split),
                [
                    quant(layer_index, "attn_q.weight"),
                    quant(layer_index, "attn_k.weight"),
                    quant(layer_index, "attn_v.weight"),
                ],
            )
            swap(layer.self_attn, "o_proj", layer_index, "attn_output.weight")
            layer.self_attn.indexer.index_qk_proj = gguf_merged_or_plain(
                config.hidden_size,
                [
                    config.qwen4_args.indexer_n_heads * config.qwen4_args.indexer_head_dim,
                    config.qwen4_args.indexer_kv_heads * config.qwen4_args.indexer_head_dim,
                ],
                [
                    quant(layer_index, "indexer.q_proj.weight"),
                    quant(layer_index, "indexer.k_proj.weight"),
                ],
            )
        if layer.ple is not None:
            swap(layer.ple, "key_proj", layer_index, "ple_key.weight")
            swap(layer.ple, "value_proj", layer_index, "ple_value.weight")


__all__ = [
    "_grouped_to_tiled_indices",
    "_ungroup_v",
    "convert_qwen4_to_gguf",
    "iter_gguf_weights",
]
