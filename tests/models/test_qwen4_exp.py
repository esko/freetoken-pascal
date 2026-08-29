from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import freetoken.models.qwen4_exp as qwen4_exp
import gguf
import numpy as np
import pytest
import safetensors
from safetensors.torch import save_file
import torch
from freetoken.gguf_host import convert_gguf_ple_to_artifact
from freetoken.layers.moe import MoELayer
from freetoken.moe.fused import FusedMoe
from freetoken.models.qwen4_exp.config import parse_config, parse_gguf_config
from freetoken.models.qwen4_exp.gguf import (
    _centered_norm,
    _grouped_to_tiled_indices,
    _ungroup_v,
)
from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM, _MappedPLETable
from freetoken.models.qwen4_exp.ple import NGramEmbedding
from tests.models.qwen4_exp.legacy_downstream import (
    _HostNGramEmbedding,
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


def test_ple_debug_state_is_active_request_order_without_allocator_slots():
    ple = object.__new__(_PLELayer)
    ple.layer_id = 7
    ple.state_len = 2
    ple.conv1d = SimpleNamespace(weight=torch.empty(4, 1, 3))
    ple._conv_states = {
        11: torch.ones(1, 4, 2),
        13: torch.full((1, 4, 2), 2.0),
        99: torch.full((1, 4, 2), 9.0),  # stale allocator state, not active
    }
    active = [
        SimpleNamespace(uid=101, table_idx=13, cached_len=4, device_len=8),
        SimpleNamespace(uid=202, table_idx=11, cached_len=0, device_len=3),
    ]
    padded = [*active, SimpleNamespace(uid=-1, table_idx=99, cached_len=0, device_len=0)]
    state = ple.semantic_debug_state(SimpleNamespace(reqs=active, padded_reqs=padded))

    assert torch.equal(state["request_uids"], torch.tensor([101, 202]))
    assert torch.equal(state["cached_lengths"], torch.tensor([4, 0]))
    assert torch.equal(state["device_lengths"], torch.tensor([8, 3]))
    assert torch.equal(state["state"][:, 0, 0, 0], torch.tensor([2.0, 1.0]))
    assert "state_slots" not in state


def test_qwen4_config_uses_exact_qsa_prefix():
    config = parse_config(_config())
    assert config.rotary_config.max_position == 262_144
    assert config.expert_quant == "fp8_block"
    assert config.attn_quant == "none"
    assert config.qwen4_args.ple_layer_ids == (1,)
    assert config.qwen4_args.output_gate_type == "sigmoid"
    # PLE state is now declared on the snapshot/COW-aware linear-state pool; the pre-#257
    # adapter required naive cache only because it owned an unsnapshotted Python state map.
    assert not config.requires_naive_cache
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
    spec = next(spec for spec in config.kv_cache_group_specs() if spec.attn_type.value == "qsa")
    assert spec.layer_ids == (3,)
    assert spec.index_head_dim == 128
    assert spec.index_ratio == 4
    assert spec.index_token_budget == 2048


def test_qwen4_config_accepts_missing_norm_topk_prob():
    hf_config = _config()
    del hf_config.text_config.norm_topk_prob
    config = parse_config(hf_config)
    # HF Qwen4ExpTextConfig defaults this omitted field to True and the upstream MoE block
    # renormalizes selected routes accordingly.
    assert config.norm_topk_prob


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


def test_qwen4_gguf_expert_setup_never_falls_through_to_nvfp4():
    fixture = ROOT / "tests/fixtures/gguf/qwen-host-layout.gguf"

    with pytest.raises(NotImplementedError, match="refusing to reinterpret them as NVFP4"):
        qwen4_exp.setup_offload_expert_banks(
            str(fixture),
            SimpleNamespace(expert_quant="gguf"),
        )


def test_qwen4_registry_entry():
    spec = get_model_spec("Qwen4ExpForConditionalGeneration")
    assert spec.module == "freetoken.models.qwen4_exp"
    assert spec.model_cls == "Qwen4ExpForCausalLM"


def _gguf_shim():
    head_vocab_sizes = [
        20_000_003,
        20_000_023,
        20_000_033,
        20_000_047,
        20_000_059,
        20_000_063,
        20_000_069,
        20_000_077,
        20_000_081,
        20_000_093,
        20_000_107,
        20_000_147,
        20_000_153,
        20_000_159,
        20_000_161,
        20_000_171,
    ]
    head_offsets = []
    offset = 0
    for size in head_vocab_sizes:
        head_offsets.append(offset)
        offset += size
    metadata = {
        "qwen4exp.block_count": 4,
        "qwen4exp.context_length": 262_144,
        "qwen4exp.embedding_length": 2560,
        "qwen4exp.attention.head_count": 24,
        "qwen4exp.attention.head_count_kv": 2,
        "qwen4exp.rope.dimension_sections": [11, 11, 10, 0],
        "qwen4exp.rope.freq_base": 10_000_000.0,
        "qwen4exp.rope.dimension_count": 64,
        "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen4exp.expert_count": 512,
        "qwen4exp.expert_used_count": 10,
        "qwen4exp.attention.key_length": 256,
        "qwen4exp.expert_feed_forward_length": 640,
        "qwen4exp.expert_shared_feed_forward_length": 640,
        "qwen4exp.ssm.conv_kernel": 4,
        "qwen4exp.ssm.state_size": 128,
        "qwen4exp.ssm.group_count": 16,
        "qwen4exp.ssm.time_step_rank": 48,
        "qwen4exp.full_attention_interval": 4,
        "qwen4exp.hyper_connection.count": 4,
        "qwen4exp.hyper_connection.low_rank": 320,
        "qwen4exp.attention.indexer.head_count": 4,
        "qwen4exp.attention.indexer.key_length": 128,
        "qwen4exp.attention.indexer.top_k": 2048,
        "qwen4exp.attention.compress_ratios": [1, 1, 1, 4],
        "qwen4exp.ple.layers": [1],
        "qwen4exp.ple.ngram_size": 3,
        "qwen4exp.ple.heads_per_ngram": 8,
        "qwen4exp.ple.layer_multipliers": [23703573157769, 20109073645365, 8052911324071],
        "qwen4exp.ple.conv_kernel": 4,
        "qwen4exp.ple.eos_token_id": 248044,
        "qwen4exp.embedding_length_per_layer_input": 160,
        "qwen4exp.ple.head_vocab_sizes": head_vocab_sizes,
        "qwen4exp.ple.head_offsets": head_offsets,
        "qwen4exp.ple.image_token_id": 248056,
    }
    return SimpleNamespace(
        metadata=metadata,
        vocab_size=248320,
        tie_word_embeddings=False,
        architectures=["Qwen4ExpGGUFForCausalLM"],
        model_path="model-00001-of-00004.gguf",
    )


def test_qwen4_gguf_config_uses_exact_artifact_geometry():
    config = parse_gguf_config(_gguf_shim())

    assert config.gguf_model_path == "model-00001-of-00004.gguf"
    assert config.expert_quant == "gguf"
    assert config.qwen4_args.ple_layer_ids == (1,)
    assert config.qwen4_args.ple_embed_dim == 2560
    assert config.qwen4_args.split_ngram_parts == 1
    assert config.qwen4_args.ple_layer_multipliers == (
        23703573157769,
        20109073645365,
        8052911324071,
    )
    assert config.qwen4_args.ple_head_vocab_sizes is not None
    assert config.qwen4_args.ple_head_offsets is not None
    assert config.is_linear_layer(2)
    assert not config.is_linear_layer(3)
    assert config.attention_group_for_layer(3).index_ratio == 4
    assert config.qwen4_args.index_compress_ratio == config.qwen4_args.index_ratio == 4


def test_qwen4_gguf_ple_metadata_seeds_current_embedding_buffers():
    args = parse_gguf_config(_gguf_shim()).qwen4_args
    embedding = NGramEmbedding(args)

    assert torch.equal(
        embedding.layer_multipliers,
        torch.tensor(args.ple_layer_multipliers, dtype=torch.int64),
    )
    assert torch.equal(
        embedding.ngram_heads_vocab_sizes,
        torch.tensor(args.ple_head_vocab_sizes, dtype=torch.int64),
    )
    assert torch.equal(
        embedding.ngram_heads_offsets,
        torch.tensor(args.ple_head_offsets, dtype=torch.int64),
    )


def test_qwen4_mapped_ple_table_flattens_rows_and_checks_geometry():
    class Table:
        descriptor = SimpleNamespace(rows=8, elements_per_row=2)

        @staticmethod
        def lookup(ids):
            return np.arange(ids.size * 2, dtype=np.float32).reshape(*ids.shape, 2)

    table = _MappedPLETable(Table(), head_dim=2)
    row_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    output = table.lookup(row_ids)

    assert output.shape == (2, 4)
    torch.testing.assert_close(output.float(), torch.arange(8).reshape(2, 4).float())
    with pytest.raises(ValueError, match="geometry"):
        _MappedPLETable(Table(), head_dim=3)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("qwen4exp.ple.layer_multipliers", [3, 5], "multiplier metadata"),
        (
            "qwen4exp.ple.head_offsets",
            [0] * 16,
            "offsets are not contiguous",
        ),
        (
            "qwen4exp.ple.head_vocab_sizes",
            [0] + [20_000_003] * 15,
            "vocabulary sizes must be positive",
        ),
    ],
)
def test_qwen4_gguf_rejects_invalid_ple_address_metadata(key, value, message):
    shim = _gguf_shim()
    shim.metadata[key] = value

    with pytest.raises(ValueError, match=message):
        parse_gguf_config(shim)


def test_qwen4_gguf_registry_entry():
    spec = get_model_spec("Qwen4ExpGGUFForCausalLM")
    assert spec.parse_config == "parse_gguf_config"
    assert spec.iter_weights == "iter_gguf_weights"


def test_qwen4_gguf_tokenizer_uses_qwen_bpe_converter():
    from freetoken.models.gguf.tokenizer import _TOKENIZER_ARCH

    assert _TOKENIZER_ARCH["qwen4exp"] == "qwen2"


def test_qwen4_gguf_v_head_reorder_roundtrip():
    grouped = torch.arange(16 * 3 * 4).reshape(16, 3, 4)
    tiled = grouped.permute(1, 0, 2).contiguous().flatten()

    assert torch.equal(_ungroup_v(tiled, 0, 16, 3, 4), grouped.flatten())
    indices = _grouped_to_tiled_indices(16, 3, 4)
    assert torch.equal(grouped.flatten().index_select(0, indices), tiled)


def test_qwen4_gguf_centered_norm_subtracts_before_bf16_narrowing():
    effective = torch.tensor([1.001, 0.999], dtype=torch.float32)
    tensor = SimpleNamespace(
        packed=lambda: effective.view(torch.uint8),
        ggml_type=0,
        shape=(2,),
    )

    centered = _centered_norm(tensor)

    assert centered.dtype == torch.bfloat16
    assert torch.count_nonzero(centered) == 2
    torch.testing.assert_close(
        centered.float(),
        torch.tensor([0.001, -0.001]),
        atol=8e-6,
        rtol=0,
    )


def test_mixed_gguf_projection_allocates_independent_buffers():
    from freetoken.layers.gguf import GGUFLinear, GGUFMergedLinear, gguf_merged_or_plain

    uniform = gguf_merged_or_plain(256, [64, 32], [8, 8])
    mixed = gguf_merged_or_plain(256, [64, 32], [8, 0])

    assert isinstance(uniform, GGUFLinear)
    assert uniform.qweight.shape == (96, 272)
    assert isinstance(mixed, GGUFMergedLinear)
    assert mixed.qweight_0.shape == (64, 272)
    assert mixed.qweight_1.shape == (32, 1024)


def test_real_qwen_q4_and_q3_rows_match_pinned_reference_outputs():
    from freetoken.models.gguf.dequant import dequantize_reference

    fixture = json.loads(
        (ROOT / "tests/fixtures/gguf/qwen38-reference-rows.json").read_text(encoding="utf-8")
    )
    assert fixture["source_revision"] == "c8b5954a88c2775c546b92593eda40ea041d3176"
    assert {row["variant"] for row in fixture["rows"]} == {
        "UD-Q3_K_XL",
        "UD-Q4_K_XL",
    }
    for row in fixture["rows"]:
        packed = torch.frombuffer(
            bytearray(base64.b64decode(row["packed_base64"])),
            dtype=torch.uint8,
        ).reshape(1, row["row_bytes"])
        output = dequantize_reference(packed, row["quant_type"], torch.float32)
        digest = hashlib.sha256(output.numpy().astype("<f4").tobytes()).hexdigest()
        assert output.numel() == row["elements"]
        assert digest == row["reference_f32_sha256"]


def test_metadata_export_accepts_a_zero_tensor_first_shard(tmp_path: Path):
    from freetoken.models.gguf.reader import OUTPUT_WEIGHT_PRESENT_KV, write_metadata_gguf

    writer = gguf.GGUFWriter(
        tmp_path / "tiny.gguf",
        "qwen4exp",
        split_max_tensors=1,
        small_first_shard=True,
    )
    writer.add_tensor("output.weight", np.zeros((32, 32), dtype=np.float32))
    writer.add_tensor("token_embd.weight", np.zeros((32, 32), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    shards = sorted(tmp_path.glob("tiny-*-of-*.gguf"))
    assert not gguf.GGUFReader(shards[0]).tensors

    destination = tmp_path / "source_metadata.gguf"
    write_metadata_gguf(str(shards[1]), str(destination))
    metadata = gguf.GGUFReader(destination)

    assert not metadata.tensors
    assert metadata.fields[OUTPUT_WEIGHT_PRESENT_KV].contents() is True


def test_qwen4_gguf_ple_maps_and_dequantizes_selected_rows(monkeypatch):
    config = parse_gguf_config(_gguf_shim())
    args = replace(
        config.qwen4_args,
        ple_embed_dim=320,
        heads_per_ngram=1,
        ngram_vocab_size_base=8,
        ple_layer_multipliers=(3, 5, 7),
        ple_head_vocab_sizes=(8, 8),
        ple_head_offsets=(0, 8),
    )
    embedding = _HostNGramEmbedding(SimpleNamespace(qwen4_args=args), layer_id=1)
    fixture = ROOT / "tests/fixtures/gguf/qwen-host-layout.gguf"
    embedding.load_host_weights(str(fixture), ple_warm_mode="targeted")
    monkeypatch.setattr(
        embedding,
        "_current_ngram_ids",
        lambda: torch.tensor([[0, 15], [16, 31]], dtype=torch.long),
    )

    output = embedding.forward(torch.device("cpu"), torch.float32)

    assert output.shape == (2, 320)
    assert embedding._gguf_ple.telemetry()["lookup_rows"] == 4
    assert embedding._gguf_ple.telemetry()["mode"] == "targeted"
    assert embedding.telemetry()["source"] == "gguf-mmap"
    assert embedding.telemetry()["packed_bytes_read"] == 4 * 90
    embedding._gguf_ple.close()


def test_qwen4_ple_explicitly_loads_dedicated_artifact(tmp_path: Path) -> None:
    config = parse_gguf_config(_gguf_shim())
    args = replace(
        config.qwen4_args,
        ple_embed_dim=2560,
        ple_head_vocab_sizes=(8, 8),
        ple_head_offsets=(0, 8),
    )
    embedding = _HostNGramEmbedding(SimpleNamespace(qwen4_args=args), layer_id=1)
    fixture = ROOT / "tests/fixtures/gguf/qwen-host-layout.gguf"
    artifact = convert_gguf_ple_to_artifact(fixture, tmp_path / "ple")

    embedding.load_host_weights(
        str(fixture),
        ple_artifact_path=str(artifact),
        ple_warm_mode="cold",
        ple_backend="pread",
    )

    assert embedding.telemetry()["source"] == "dedicated-artifact"
    assert embedding.telemetry()["backend"] == "pread"
    assert embedding.telemetry()["mapped_bytes"] == 0
    assert embedding._gguf_ple.descriptor.shard_path == str(artifact / "ple.bin")
    embedding._gguf_ple.close()


def _write_ple_safetensors_fixture(
    folder: Path,
    *,
    shard_shapes: tuple[tuple[int, ...], ...] = ((2, 2), (2, 2)),
    include_scale: bool = True,
    scale_shape: tuple[int, ...] = (),
    scale_value: float = 1.0,
    shared_file: bool = False,
) -> None:
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    weight_map: dict[str, str] = {}
    tensors: dict[str, torch.Tensor] = {}
    for shard_id, shape in enumerate(shard_shapes):
        key = f"{prefix}.shard_{shard_id}.weight"
        filename = "ple-shared.safetensors" if shared_file else f"ple-{shard_id:05d}.safetensors"
        tensors[key] = torch.zeros(shape, dtype=torch.float8_e4m3fn)
        weight_map[key] = filename
    if include_scale:
        key = f"{prefix}.weight_scale"
        filename = "ple-shared.safetensors" if shared_file else "ple-scale.safetensors"
        tensors[key] = torch.full(scale_shape, scale_value, dtype=torch.bfloat16)
        weight_map[key] = filename
    if shared_file:
        save_file(tensors, str(folder / "ple-shared.safetensors"))
    else:
        for key, tensor in tensors.items():
            save_file({key: tensor}, str(folder / weight_map[key]))
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )


def _tiny_safetensors_ple_embedding() -> _HostNGramEmbedding:
    args = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=1,
        eos_token_id=99,
        ple_embed_dim=4,
        split_ngram_parts=2,
        ple_layer_multipliers=(3, 5, 7),
        ple_head_vocab_sizes=(2, 2),
        ple_head_offsets=(0, 2),
    )
    return _HostNGramEmbedding(SimpleNamespace(qwen4_args=args), layer_id=0)


def test_safetensors_ple_rejects_missing_weight_scale(tmp_path: Path) -> None:
    _write_ple_safetensors_fixture(tmp_path, include_scale=False)

    with pytest.raises(RuntimeError, match="missing .*weight_scale"):
        _tiny_safetensors_ple_embedding().load_host_weights(str(tmp_path))


def test_safetensors_ple_rejects_nonuniform_shard_rows(tmp_path: Path) -> None:
    _write_ple_safetensors_fixture(tmp_path, shard_shapes=((2, 2), (3, 2)))

    with pytest.raises(RuntimeError, match="shard .* shape"):
        _tiny_safetensors_ple_embedding().load_host_weights(str(tmp_path))


def test_safetensors_ple_rejects_non_scalar_weight_scale(tmp_path: Path) -> None:
    _write_ple_safetensors_fixture(tmp_path, scale_shape=(2,))

    with pytest.raises(RuntimeError, match="must be one finite positive"):
        _tiny_safetensors_ple_embedding().load_host_weights(str(tmp_path))


@pytest.mark.parametrize("scale_value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_safetensors_ple_rejects_non_positive_or_non_finite_scale(
    tmp_path: Path, scale_value: float
) -> None:
    _write_ple_safetensors_fixture(tmp_path, scale_value=scale_value)

    with pytest.raises(RuntimeError, match="must be one finite positive"):
        _tiny_safetensors_ple_embedding().load_host_weights(str(tmp_path))


def test_safetensors_ple_undersized_table_closes_open_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_ple_safetensors_fixture(tmp_path)
    embedding = _tiny_safetensors_ple_embedding()
    embedding.ngram_heads_vocab_sizes = torch.tensor([3, 3], dtype=torch.long)
    embedding.ngram_heads_offsets = torch.tensor([0, 3], dtype=torch.long)
    opened: list[object] = []
    exited: list[object] = []
    original_safe_open = safetensors.safe_open

    class TrackedHandle:
        def __init__(self, path: str, **kwargs: object) -> None:
            self._inner = original_safe_open(path, **kwargs)

        def __enter__(self):
            self._inner.__enter__()
            opened.append(self)
            return self

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        def __exit__(self, *args: object) -> None:
            exited.append(self)
            self._inner.__exit__(*args)

    monkeypatch.setattr(
        "tests.models.qwen4_exp.legacy_downstream.safetensors.safe_open",
        lambda path, **kwargs: TrackedHandle(path, **kwargs),
    )
    with pytest.raises(RuntimeError, match="has 4 rows, needs 6"):
        embedding.load_host_weights(str(tmp_path))
    assert opened
    assert set(exited) == set(opened)


def test_safetensors_ple_loads_shared_shard_and_scale_file(tmp_path: Path) -> None:
    _write_ple_safetensors_fixture(tmp_path, shared_file=True)
    embedding = _tiny_safetensors_ple_embedding()
    embedding.load_host_weights(str(tmp_path))
    embedding._current_ngram_ids = lambda: torch.tensor([[0, 3]], dtype=torch.long)

    output = embedding.forward(torch.device("cpu"), torch.float32)

    assert output.shape == (1, 4)
    assert embedding._shard_ends.tolist() == [2, 4]
    assert embedding._scale.item() == 1.0


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
        "tests.models.qwen4_exp.legacy_downstream.get_global_ctx",
        lambda: SimpleNamespace(batch=full_batch),
    )
    expected = full._short_conv(values)

    chunked = _ple_reference_layer(weight)
    first_batch = SimpleNamespace(
        is_decode=False,
        reqs=[SimpleNamespace(extend_len=2, cached_len=0, table_idx=3)],
    )
    monkeypatch.setattr(
        "tests.models.qwen4_exp.legacy_downstream.get_global_ctx",
        lambda: SimpleNamespace(batch=first_batch),
    )
    first = chunked._short_conv(values[:2])
    second_batch = SimpleNamespace(
        is_decode=False,
        reqs=[SimpleNamespace(extend_len=3, cached_len=2, table_idx=3)],
    )
    monkeypatch.setattr(
        "tests.models.qwen4_exp.legacy_downstream.get_global_ctx",
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
        "tests.models.qwen4_exp.legacy_downstream.get_global_ctx",
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
        forward=lambda input_ids, batch=None: input_ids.float().unsqueeze(-1),
        debug_state=lambda: {1: {7: torch.tensor([3.0])}},
        set_debug_observer=lambda observer: None,
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


def test_qwen4_debug_hook_collects_opt_in_semantic_events(monkeypatch) -> None:
    class FakeModel:
        def __init__(self):
            self.observer = None

        def set_debug_observer(self, observer):
            self.observer = observer

        def forward(self, input_ids, batch=None):
            del batch
            if self.observer is not None:
                self.observer(
                    "router",
                    {
                        "layer_id": 3,
                        "ids": torch.tensor([[7, 2]], dtype=torch.int32),
                        "weights": torch.tensor([[0.6, 0.2]], dtype=torch.float32),
                    },
                )
            return input_ids.float().unsqueeze(-1)

        def debug_state(self):
            return {}

    model = object.__new__(Qwen4ExpForCausalLM)
    model.model = FakeModel()
    model.lm_head = SimpleNamespace(forward=lambda hidden: torch.cat((hidden, -hidden), dim=-1))
    model._debug_hook = None
    model._debug_events = {}
    monkeypatch.setattr(
        "freetoken.models.qwen4_exp.model.get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(input_ids=torch.tensor([2]))),
    )

    captured = []
    model.set_debug_hook(captured.append)
    model.forward()

    assert captured[0]["observations"]["router"][0]["layer_id"] == 3
    torch.testing.assert_close(
        captured[0]["observations"]["router"][0]["ids"],
        torch.tensor([[7, 2]], dtype=torch.int32),
    )
    model.set_debug_hook(None)
    assert model.model.observer is None


def test_moe_route_observer_snapshots_semantic_ids_and_weights(monkeypatch) -> None:
    layer = object.__new__(MoELayer)
    layer.top_k = 2
    layer.renormalize = False
    ids = torch.tensor([[4, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.7, 0.2]], dtype=torch.float32)
    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda **kwargs: (weights, ids),
    )

    events = []
    _observed_weights, observed_ids = layer._route_and_observe(
        torch.zeros(1, 3), torch.zeros(1, 5), lambda name, payload: events.append(payload)
    )
    ids.fill_(-1)
    weights.zero_()

    assert observed_ids.tolist() == [[-1, -1]]
    assert events[0]["ids"].tolist() == [[4, 1]]
    torch.testing.assert_close(events[0]["weights"], torch.tensor([[0.7, 0.2]]))


def test_moe_route_observer_excludes_padded_rows(monkeypatch) -> None:
    layer = object.__new__(MoELayer)
    layer.top_k = 2
    layer.renormalize = False
    ids = torch.tensor([[4, 1], [3, 2], [9, 8]], dtype=torch.int32)
    weights = torch.tensor([[0.7, 0.2], [0.6, 0.3], [0.9, 0.1]])
    active = SimpleNamespace(uid=101, extend_len=2, cached_len=4, device_len=6)
    padded = SimpleNamespace(uid=-1, extend_len=1, cached_len=0, device_len=1)
    batch = SimpleNamespace(
        reqs=[active],
        padded_reqs=[active, padded],
        phase="decode",
    )
    monkeypatch.setattr("freetoken.layers.moe.get_global_ctx", lambda: SimpleNamespace(batch=batch))
    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda **kwargs: (weights, ids),
    )

    events = []
    returned_weights, returned_ids = layer._route_and_observe(
        torch.zeros(3, 3), torch.zeros(3, 10), lambda name, payload: events.append(payload)
    )

    assert returned_ids.shape == (3, 2)
    assert returned_weights.shape == (3, 2)
    assert events[0]["ids"].tolist() == [[4, 1], [3, 2]]
    assert events[0]["valid_token_count"] == 2
    assert events[0]["padded_token_count"] == 3
    assert events[0]["request_uids"].tolist() == [101]


def test_fused_backend_observer_reports_the_executed_route(monkeypatch) -> None:
    ids = torch.tensor([[8, 3]], dtype=torch.int32)
    weights = torch.tensor([[0.55, 0.35]], dtype=torch.float32)
    expected_output = torch.tensor([[1.0, 2.0]])
    monkeypatch.setattr(
        "freetoken.moe.fused.fused_topk",
        lambda **kwargs: (weights, ids),
    )

    def fake_experts(hidden_states, w1, w2, actual_weights, actual_ids, *args, **kwargs):
        assert actual_weights is weights
        assert actual_ids is ids
        return expected_output

    monkeypatch.setattr("freetoken.moe.fused.fused_experts_impl", fake_experts)
    events = []

    output = FusedMoe().forward(
        hidden_states=torch.zeros(1, 2),
        w1=torch.empty(0),
        w2=torch.empty(0),
        gating_output=torch.zeros(1, 10),
        topk=2,
        renormalize=False,
        debug_observer=lambda name, payload: events.append((name, payload)),
    )

    assert output is expected_output
    assert events == [("router", {"ids": ids, "weights": weights})]
