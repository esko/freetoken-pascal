"""Small CPU checks for the Qwen4 GDN reference fallback.

These tests exercise the same stateful helpers used by ``Qwen4ExpGatedDeltaNet`` when the
backend contract selects ``torch-reference``.  The oracle is the pure-torch
``gdn_reference`` recurrence, rather than a second copy of the fallback implementation.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

try:
    from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet
    from freetoken.models.qwen4_exp.gdn_reference import recurrent_gated_delta_rule
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"Qwen4 CPU reference dependencies unavailable: {exc}", allow_module_level=True)

F = torch.nn.functional


def _constructed_op(**kwargs):
    """Construct a tiny CPU op for constructor-only contract checks."""
    from freetoken.distributed import set_tp_info, try_get_tp_info

    tp_info = try_get_tp_info()
    if tp_info is None:
        set_tp_info(0, 1)
    elif tp_info.size != 1:
        pytest.skip("constructor contract test requires a single-process TP context")
    defaults = {
        "hidden_size": 4,
        "num_k_heads": 1,
        "num_v_heads": 2,
        "head_k_dim": 2,
        "head_v_dim": 2,
        "conv_kernel_size": 3,
        "rms_norm_eps": 1e-6,
        "layer_id": 0,
        "expert_quant": "none",
        "attn_quant": "none",
    }
    defaults.update(kwargs)
    return Qwen4ExpGatedDeltaNet(**defaults)


class _Pool:
    def __init__(self, slots: int, conv_dim: int, state_len: int, heads: int, dim: int):
        self.conv_states = torch.zeros(1, slots, conv_dim, state_len)
        self.recurrent_states = torch.zeros(1, slots, heads, dim, dim)

    def local_index(self, layer_id: int) -> int:
        assert layer_id == 0
        return 0


def test_gdn_constructor_freezes_env_mode_and_package_probes(monkeypatch):
    gdn_module = importlib.import_module("freetoken.models.qwen4_exp.gdn")
    monkeypatch.setenv("FREETOKEN_GDN_MODE", "reference")
    monkeypatch.delenv("FREETOKEN_GDN_BACKEND", raising=False)
    monkeypatch.setattr(gdn_module, "_probe_fla_available", lambda: True)
    monkeypatch.setattr(gdn_module, "_probe_triton_candidate_available", lambda: True)
    op = _constructed_op()

    monkeypatch.setenv("FREETOKEN_GDN_MODE", "auto")
    monkeypatch.setattr(gdn_module, "_probe_fla_available", lambda: False)
    monkeypatch.setattr(gdn_module, "_probe_triton_candidate_available", lambda: False)
    assert op._gdn_mode == "torch-reference"
    assert op._gdn_fla_available is True
    assert op._gdn_candidate_available is True


def test_gdn_rejects_non_divisible_gqa_head_counts():
    with pytest.raises(ValueError, match="positive multiple"):
        _constructed_op(num_k_heads=2, num_v_heads=3)


def _op(*, num_k_heads: int, num_v_heads: int, head_dim: int, conv_dim: int, kernel: int):
    """Build only the stateful fallback surface; no CUDA/Triton operator is constructed."""
    op = object.__new__(Qwen4ExpGatedDeltaNet)
    op.layer_id = 0
    op.num_k_heads = num_k_heads
    op.num_v_heads = num_v_heads
    op.head_k_dim = head_dim
    op.head_v_dim = head_dim
    op.conv_dim = conv_dim
    op.conv_kernel_size = kernel
    op.conv1d = SimpleNamespace(weight=torch.randn(conv_dim, 1, kernel))
    return op


def _fla(lengths: list[int], slots: list[int], *, has_initial: list[bool] | None = None):
    return SimpleNamespace(
        cu_seqlens=torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32),
        cache_indices=torch.tensor(slots, dtype=torch.int32),
        has_initial_state=(
            None if has_initial is None else torch.tensor(has_initial, dtype=torch.bool)
        ),
        track_boundary_row=None,
        track_dst=None,
    )


def _oracle(q, k, v, g, beta, state):
    repeat = v.shape[1] // q.shape[1]
    if repeat > 1:
        q = q.repeat_interleave(repeat, dim=1)
        k = k.repeat_interleave(repeat, dim=1)
    output, final_state = recurrent_gated_delta_rule(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
        initial_state=state.unsqueeze(0),
        use_qk_l2norm=True,
    )
    return output[0], final_state[0]


def test_reference_conv_prefill_resets_stale_slot_for_fresh_request():
    conv_dim, kernel = 4, 3
    op = _op(num_k_heads=1, num_v_heads=1, head_dim=2, conv_dim=conv_dim, kernel=kernel)
    pool = _Pool(slots=1, conv_dim=conv_dim, state_len=kernel - 1, heads=1, dim=2)
    pool.conv_states[0, 0].copy_(torch.randn_like(pool.conv_states[0, 0]))
    conv_in = torch.randn(3, conv_dim)
    fla = _fla([3], [0], has_initial=[False])

    got = op._reference_conv_prefill(conv_in, pool, fla)
    context = torch.cat((torch.zeros_like(pool.conv_states[0, 0]), conv_in.T), dim=-1)
    expected = F.silu(
        F.conv1d(context.unsqueeze(0), op.conv1d.weight, groups=conv_dim).squeeze(0).transpose(0, 1)
    )

    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(pool.conv_states[0, 0], context[:, -(kernel - 1) :])


def test_reference_conv_decode_carries_prefill_state():
    conv_dim, kernel = 4, 3
    op = _op(num_k_heads=1, num_v_heads=1, head_dim=2, conv_dim=conv_dim, kernel=kernel)
    pool = _Pool(slots=1, conv_dim=conv_dim, state_len=kernel - 1, heads=1, dim=2)
    prefix = torch.randn(3, conv_dim)
    nxt = torch.randn(1, conv_dim)
    op._reference_conv_prefill(prefix, pool, _fla([3], [0], has_initial=[False]))
    got = op._reference_conv_decode(nxt, torch.tensor([0], dtype=torch.int32), pool)

    context = torch.cat((prefix[-(kernel - 1) :], nxt), dim=0)
    expected = F.silu(
        F.conv1d(context.T.reshape(1, conv_dim, -1), op.conv1d.weight, groups=conv_dim).squeeze(-1)
    )[0]
    torch.testing.assert_close(got[0], expected)


def test_reference_recurrent_carries_decode_state_and_applies_q_scale():
    num_k_heads, num_v_heads, dim, value_dim = 1, 2, 3, 3
    op = _op(
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_dim=dim,
        conv_dim=1,
        kernel=2,
    )
    pool = _Pool(slots=1, conv_dim=1, state_len=1, heads=num_v_heads, dim=dim)
    torch.manual_seed(3)
    q = torch.randn(4, num_k_heads, dim)
    k = torch.randn(4, num_k_heads, dim)
    v = torch.randn(4, num_v_heads, value_dim)
    g = torch.randn(4, num_v_heads)
    beta = torch.sigmoid(torch.randn(4, num_v_heads))
    got_prefix = op._reference_recurrent(
        q[:3].unsqueeze(0),
        k[:3].unsqueeze(0),
        v[:3].unsqueeze(0),
        g[:3].unsqueeze(0),
        beta[:3].unsqueeze(0),
        pool.recurrent_states[0],
        _fla([3], [0]),
    )
    got_decode = op._reference_recurrent(
        q[3:].unsqueeze(0),
        k[3:].unsqueeze(0),
        v[3:].unsqueeze(0),
        g[3:].unsqueeze(0),
        beta[3:].unsqueeze(0),
        pool.recurrent_states[0],
        _fla([1], [0]),
    )
    expected, expected_state = _oracle(q, k, v, g, beta, torch.zeros(num_v_heads, dim, value_dim))

    torch.testing.assert_close(got_prefix[0], expected[:3])
    torch.testing.assert_close(got_decode[0], expected[3:])
    torch.testing.assert_close(pool.recurrent_states[0, 0], expected_state)


def test_reference_recurrent_matches_oracle_for_ragged_gqa_requests():
    num_k_heads, num_v_heads, dim = 2, 4, 2
    op = _op(
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_dim=dim,
        conv_dim=1,
        kernel=2,
    )
    pool = _Pool(slots=2, conv_dim=1, state_len=1, heads=num_v_heads, dim=dim)
    torch.manual_seed(7)
    lengths = [2, 3]
    total = sum(lengths)
    q = torch.randn(1, total, num_k_heads, dim)
    k = torch.randn(1, total, num_k_heads, dim)
    v = torch.randn(1, total, num_v_heads, dim)
    g = torch.randn(1, total, num_v_heads)
    beta = torch.sigmoid(torch.randn(1, total, num_v_heads))
    got = op._reference_recurrent(q, k, v, g, beta, pool.recurrent_states[0], _fla(lengths, [0, 1]))

    offset = 0
    for slot, length in enumerate(lengths):
        expected, expected_state = _oracle(
            q[0, offset : offset + length],
            k[0, offset : offset + length],
            v[0, offset : offset + length],
            g[0, offset : offset + length],
            beta[0, offset : offset + length],
            torch.zeros(num_v_heads, dim, dim),
        )
        torch.testing.assert_close(got[0, offset : offset + length], expected)
        torch.testing.assert_close(pool.recurrent_states[0, slot], expected_state)
        offset += length
