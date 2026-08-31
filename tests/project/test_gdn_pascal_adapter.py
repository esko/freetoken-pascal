"""H0/source checks for the explicit Pascal FP32 GDN JIT seam."""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

import pytest
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req, SamplingParams
from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "python/freetoken/kernel/csrc/jit/gdn_pascal.cu"
ADAPTER = ROOT / "python/freetoken/kernel/gdn_pascal.py"

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - hosted source-only environments
    torch = None

_ADAPTER = importlib.import_module("freetoken.kernel.gdn_pascal") if torch is not None else None


def _require_adapter():
    if _ADAPTER is None:
        pytest.skip("Torch is unavailable")
    return _ADAPTER


def test_pascal_source_is_standalone_and_emits_both_head_dimensions() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    assert "template <int kHeadDim>" in source
    assert "template struct PascalGdnKernel<64>;" in source
    assert "template struct PascalGdnKernel<128>;" in source
    assert "pascal_gdn_recurrence_f32" in source
    assert "state_pool" in source and "[K,V]" in source
    assert "pre-sigmoided" in source
    assert "sigmoid(" not in source
    assert "pxa-deltanet-fuse" not in source
    assert "ssm-conv" not in source
    assert "graph" not in source.lower()


def test_adapter_is_a_jit_wrapper_for_the_standalone_source() -> None:
    source = ADAPTER.read_text(encoding="utf-8")

    assert 'cuda_files=["gdn_pascal.cu"]' in source
    assert '"gdn_pascal_f32"' in source
    assert "PascalGdnKernel<{args}>::run" in source
    assert "FREETOKEN_GDN_MODE" not in source


def _inputs(*, duplicate_slots: bool = False, bad_offsets: bool = False, tokens: int = 3):
    if torch is None:
        pytest.skip("Torch is unavailable")
    key_heads, value_heads, dim, slots = 1, 2, 64, 4
    q = torch.randn(tokens, key_heads, dim)
    k = torch.randn_like(q)
    v = torch.randn(tokens, value_heads, dim)
    g = torch.randn(tokens, value_heads)
    beta = torch.sigmoid(torch.randn(tokens, value_heads))
    state = torch.zeros(slots, value_heads, dim, dim)
    request_count = 2 if tokens else 1
    indices = torch.tensor([1, 2 if not duplicate_slots else 1][:request_count], dtype=torch.int32)
    offsets = torch.tensor(
        ([0, 1, tokens] if not bad_offsets else [0, tokens, tokens - 1]) if tokens else [0, 0],
        dtype=torch.int32,
    )
    return q, k, v, g, beta, state, indices, offsets


def _proven_decode_metadata():
    reqs = [
        Req(
            input_ids=torch.tensor([1, 2], dtype=torch.int32),
            table_idx=slot,
            cached_len=1,
            output_len=2,
            uid=slot,
            sampling_params=SamplingParams(),
            cache_handle=None,
        )
        for slot in (1, 2)
    ]
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    batch.linear_table_idx = torch.tensor([1, 2], dtype=torch.int32)
    batch.linear_table_idx_host = (1, 2)
    metadata = build_fla_metadata(batch, torch.device("cpu"))
    operator = object.__new__(Qwen4ExpGatedDeltaNet)
    operator._ensure_pascal_metadata_proof(metadata, torch.device("cpu"))
    return metadata


def test_validator_accepts_ragged_gqa_pool_layout_and_pre_sigmoided_beta() -> None:
    launch = _require_adapter().validate_pascal_gdn_inputs(*_inputs())

    assert launch.head_dim == 64
    assert launch.num_tokens == 3
    assert launch.num_requests == 2
    assert launch.num_k_heads == 1
    assert launch.num_v_heads == 2
    assert launch.num_slots == 4


def test_scheduler_metadata_proof_avoids_duplicate_device_to_host_validation(monkeypatch) -> None:
    adapter = _require_adapter()
    assert not hasattr(adapter, "make_pascal_gdn_metadata_proof")
    inputs = _inputs(tokens=2)
    proof = _proven_decode_metadata().pascal_metadata_proof
    assert proof is not None

    def unexpected_sync(*_args, **_kwargs):
        raise AssertionError("bound metadata must not be copied to the host")

    monkeypatch.setattr(adapter, "_cpu_int_values", unexpected_sync)
    launch = adapter.validate_pascal_gdn_inputs(
        *inputs[:6], proof.slot_indices, proof.cu_seqlens, metadata_proof=proof
    )
    assert launch.num_tokens == 2


def test_metadata_proof_rejects_forgery_tensor_replacement_and_mutation() -> None:
    adapter = _require_adapter()
    inputs = _inputs(tokens=2)
    proof = _proven_decode_metadata().pascal_metadata_proof
    assert proof is not None

    with pytest.raises(TypeError, match="dataclass instances"):
        replace(proof, slot_values=(2, 1))

    replacement = proof.slot_indices.clone()
    with pytest.raises(adapter.PascalGdnContractError, match="stale or unbound"):
        adapter.validate_pascal_gdn_inputs(
            *inputs[:6], replacement, proof.cu_seqlens, metadata_proof=proof
        )

    proof.slot_indices[0] = 0
    with pytest.raises(adapter.PascalGdnContractError, match="stale or unbound"):
        adapter.validate_pascal_gdn_inputs(
            *inputs[:6], proof.slot_indices, proof.cu_seqlens, metadata_proof=proof
        )


def test_metadata_proof_retains_fail_closed_slot_and_offset_checks() -> None:
    proof = _proven_decode_metadata().pascal_metadata_proof
    assert proof is not None
    with pytest.raises(TypeError, match="dataclass instances"):
        replace(proof, slot_values=(4, 2))
    with pytest.raises(TypeError, match="dataclass instances"):
        replace(proof, offset_values=(0, 2, 1))


def test_metadata_builder_distinguishes_cold_and_scheduler_proven_decode_paths(monkeypatch) -> None:
    adapter = _require_adapter()
    batch = Batch(reqs=[], phase="decode")
    batch.padded_reqs = []
    batch.linear_table_idx = torch.empty(0, dtype=torch.int32)

    # An absent host tuple leaves a direct caller on the old synchronous path.
    with monkeypatch.context() as patch:
        patch.setattr(
            adapter,
            "_issue_pascal_gdn_metadata_proof",
            lambda *_args, **_kwargs: pytest.fail("unproven metadata must not issue a proof"),
        )
        cold = build_fla_metadata(batch, torch.device("cpu"))
    assert cold.pascal_metadata_proof is None

    proven = _proven_decode_metadata()
    assert proven.pascal_metadata_proof is not None
    operator = object.__new__(Qwen4ExpGatedDeltaNet)
    assert operator._ensure_pascal_metadata_proof(proven, torch.device("cpu")) is proven.pascal_metadata_proof
    slots, offsets, initial = proven.pascal_metadata_proof.values_for(
        proven.pascal_metadata_proof.slot_indices, proven.pascal_metadata_proof.cu_seqlens
    )
    assert slots == [1, 2]
    assert offsets == [0, 1, 2]
    assert initial is None


def test_proof_is_independent_from_mismatched_generic_tensor_and_prefill_dtype() -> None:
    metadata = _proven_decode_metadata()
    proof = metadata.pascal_metadata_proof
    assert proof is not None
    metadata.cache_indices[0] = 99
    assert metadata.cache_indices.tolist() == [99, 2]
    assert proof.slot_indices.tolist() == [1, 2]

    req = Req(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        table_idx=1,
        cached_len=1,
        output_len=2,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = [req]
    prefill = build_fla_metadata(batch, torch.device("cpu"))
    assert prefill.cu_seqlens.dtype == torch.int64
    assert prefill.pascal_metadata_proof is None
    object.__new__(Qwen4ExpGatedDeltaNet)._ensure_pascal_metadata_proof(
        prefill, torch.device("cpu")
    )
    assert prefill.pascal_metadata_proof is not None
    assert prefill.pascal_metadata_proof.cu_seqlens.dtype == torch.int32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"duplicate_slots": True}, "unique"),
        ({"bad_offsets": True}, "nondecreasing"),
    ],
)
def test_validator_rejects_state_aliasing_or_malformed_ragged_metadata(kwargs, message) -> None:
    with pytest.raises(_require_adapter().PascalGdnContractError, match=message):
        _require_adapter().validate_pascal_gdn_inputs(*_inputs(**kwargs))


def test_explicit_adapter_fails_closed_before_jit_on_cpu() -> None:
    with pytest.raises(_require_adapter().PascalGdnContractError, match="requires a CUDA"):
        _require_adapter().pascal_gdn_recurrence(*_inputs())


def test_validator_rejects_empty_token_batch_before_jit() -> None:
    with pytest.raises(_require_adapter().PascalGdnContractError, match="at least one token"):
        _require_adapter().validate_pascal_gdn_inputs(*_inputs(tokens=0))


@pytest.mark.parametrize(("mutation", "message"), [("dtype", "float32"), ("stride", "contiguous")])
def test_validator_rejects_invalid_output_before_jit(mutation: str, message: str) -> None:
    inputs = _inputs()
    output = torch.empty_like(inputs[2])
    if mutation == "dtype":
        output = output.double()
    else:
        output = torch.empty(output.shape[0], output.shape[2], output.shape[1]).transpose(1, 2)
        assert not output.is_contiguous()

    with pytest.raises(_require_adapter().PascalGdnContractError, match=message):
        _require_adapter().validate_pascal_gdn_inputs(*inputs, output=output)
