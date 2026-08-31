"""H0/source checks for the explicit Pascal FP32 GDN JIT seam."""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

import pytest
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req, SamplingParams

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
    inputs = _inputs()
    proof = adapter.make_pascal_gdn_metadata_proof(
        inputs[6], inputs[7], (1, 2), (0, 1, 3)
    )

    def unexpected_sync(*_args, **_kwargs):
        raise AssertionError("bound metadata must not be copied to the host")

    monkeypatch.setattr(adapter, "_cpu_int_values", unexpected_sync)
    launch = adapter.validate_pascal_gdn_inputs(*inputs, metadata_proof=proof)
    assert launch.num_tokens == 3


def test_metadata_proof_rejects_forgery_tensor_replacement_and_mutation() -> None:
    adapter = _require_adapter()
    inputs = _inputs()
    proof = adapter.make_pascal_gdn_metadata_proof(
        inputs[6], inputs[7], (1, 2), (0, 1, 3)
    )

    with pytest.raises(adapter.PascalGdnContractError, match="values were modified"):
        replace(proof, slot_values=(2, 1))

    replacement = inputs[6].clone()
    with pytest.raises(adapter.PascalGdnContractError, match="stale or unbound"):
        adapter.validate_pascal_gdn_inputs(
            *inputs[:6], replacement, inputs[7], metadata_proof=proof
        )

    inputs[6][0] = 0
    with pytest.raises(adapter.PascalGdnContractError, match="stale or unbound"):
        adapter.validate_pascal_gdn_inputs(*inputs, metadata_proof=proof)


@pytest.mark.parametrize(
    ("slot_values", "offset_values", "message"),
    [((4, 2), (0, 1, 3), "out-of-range"), ((1, 2), (0, 3, 2), "nondecreasing")],
)
def test_metadata_proof_retains_fail_closed_slot_and_offset_checks(
    slot_values, offset_values, message
) -> None:
    adapter = _require_adapter()
    inputs = _inputs()
    proof = adapter.make_pascal_gdn_metadata_proof(
        inputs[6], inputs[7], slot_values, offset_values
    )
    with pytest.raises(adapter.PascalGdnContractError, match=message):
        adapter.validate_pascal_gdn_inputs(*inputs, metadata_proof=proof)


def test_metadata_builder_distinguishes_cold_and_scheduler_proven_decode_paths() -> None:
    req = Req(
        input_ids=torch.tensor([1, 2], dtype=torch.int32),
        table_idx=1,
        cached_len=1,
        output_len=2,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    batch.linear_table_idx = torch.tensor([1], dtype=torch.int32)

    cold = build_fla_metadata(batch, torch.device("cpu"))
    assert cold.pascal_metadata_proof is None

    batch.linear_table_idx_host = (1,)
    batch.linear_metadata_epoch = 7
    proven = build_fla_metadata(batch, torch.device("cpu"))
    assert proven.pascal_metadata_proof is not None
    slots, offsets, initial = proven.pascal_metadata_proof.values_for(
        proven.cache_indices, proven.cu_seqlens
    )
    assert slots == [1]
    assert offsets == [0, 1]
    assert initial is None


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
