"""Bounded H2 parity checks for the standalone Pascal FP32 GDN recurrence."""

from __future__ import annotations

import pytest


def _inputs(torch, *, dim: int = 64):
    generator = torch.Generator(device="cuda").manual_seed(93)
    tokens, key_heads, value_heads, slots = 5, 1, 2, 4
    q = torch.randn(tokens, key_heads, dim, device="cuda", generator=generator)
    k = torch.randn(tokens, key_heads, dim, device="cuda", generator=generator)
    v = torch.randn(tokens, value_heads, dim, device="cuda", generator=generator)
    g = -torch.rand(tokens, value_heads, device="cuda", generator=generator) * 0.2
    beta = torch.sigmoid(torch.randn(tokens, value_heads, device="cuda", generator=generator))
    state = (
        torch.randn(
            slots,
            value_heads,
            dim,
            dim,
            device="cuda",
            generator=generator,
        )
        * 0.01
    )
    slot_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    return q, k, v, g, beta, state, slot_indices, cu_seqlens


def _reference(torch, inputs):
    from freetoken.models.qwen4_exp.gdn_reference import recurrent_gated_delta_rule

    q, k, v, g, beta, state, slots, offsets = inputs
    repeat = v.shape[1] // q.shape[1]
    expected_output = torch.empty_like(v)
    expected_state = state.clone()
    offset_values = offsets.cpu().tolist()
    for request, slot in enumerate(slots.cpu().tolist()):
        start, end = offset_values[request : request + 2]
        output, final_state = recurrent_gated_delta_rule(
            q[start:end].repeat_interleave(repeat, dim=1).unsqueeze(0),
            k[start:end].repeat_interleave(repeat, dim=1).unsqueeze(0),
            v[start:end].unsqueeze(0),
            g[start:end].unsqueeze(0),
            beta[start:end].unsqueeze(0),
            initial_state=state[slot].unsqueeze(0),
        )
        expected_output[start:end].copy_(output[0])
        expected_state[slot].copy_(final_state[0])
    return expected_output, expected_state


@pytest.mark.sm61
@pytest.mark.parametrize("dim", [64, 128])
def test_pascal_gdn_matches_independent_ragged_reference(dim: int) -> None:
    torch = pytest.importorskip("torch")
    from freetoken.kernel.gdn_pascal import pascal_gdn_recurrence

    inputs = _inputs(torch, dim=dim)
    expected_output, expected_state = _reference(torch, inputs)
    actual_state = inputs[5].clone()
    actual = pascal_gdn_recurrence(*inputs[:5], actual_state, *inputs[6:])
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected_output, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(actual_state, expected_state, rtol=3e-5, atol=3e-5)
    assert torch.equal(actual_state[[0, 2]], inputs[5][[0, 2]])


@pytest.mark.sm61
def test_pascal_gdn_chunk_and_tokenwise_decode_preserve_state() -> None:
    torch = pytest.importorskip("torch")
    from freetoken.kernel.gdn_pascal import pascal_gdn_recurrence

    inputs = _inputs(torch)
    chunk_state = inputs[5].clone()
    chunk_output = pascal_gdn_recurrence(*inputs[:5], chunk_state, *inputs[6:])

    decode_state = inputs[5].clone()
    decode_output = torch.empty_like(inputs[2])
    offsets = inputs[7].cpu().tolist()
    slots = inputs[6].cpu().tolist()
    for request, slot in enumerate(slots):
        for token in range(offsets[request], offsets[request + 1]):
            one_slot = torch.tensor([slot], device="cuda", dtype=torch.int32)
            one_offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
            decode_output[token : token + 1].copy_(
                pascal_gdn_recurrence(
                    inputs[0][token : token + 1],
                    inputs[1][token : token + 1],
                    inputs[2][token : token + 1],
                    inputs[3][token : token + 1],
                    inputs[4][token : token + 1],
                    decode_state,
                    one_slot,
                    one_offsets,
                )
            )
    torch.cuda.synchronize()

    torch.testing.assert_close(decode_output, chunk_output, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(decode_state, chunk_state, rtol=3e-5, atol=3e-5)
