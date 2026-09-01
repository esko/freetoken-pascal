"""The QSA backend behind the real Qwen4ExpAttention layer.

(a) dense-oracle equivalence -- while a request sees at most ``index_budget + index_ratio - 1``
    tokens every complete block is selected, so QSA IS dense attention: the selection must be
    exactly the causal prefix and the layer output must match ``TorchDenseQSAReference`` (fp32)
    and a flashinfer dense run over the same pool;
(b) chunked prefill at unaligned cut points equals one-shot prefill (the dual-source compress);
(c) a captured decode replay equals the eager decode step.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .common import Fixture, requires_cuda, parsed_config, selection_spy

QSA_LAYER = 3


def _inputs(fixture: Fixture, lengths, extra: int = 0, seed: int = 11):
    generator = torch.Generator(device=fixture.device).manual_seed(seed)
    return [
        torch.randn(
            n + extra, fixture.config.hidden_size, device=fixture.device,
            dtype=fixture.dtype, generator=generator,
        )
        * 0.5
        for n in lengths
    ]


def _assert_selection_is_causal_prefix(indices: torch.Tensor, positions: torch.Tensor) -> None:
    for row, position in enumerate(positions.tolist()):
        selected = indices[row][indices[row] >= 0]
        assert torch.equal(
            selected.sort().values,
            torch.arange(position + 1, dtype=selected.dtype, device=selected.device),
        ), f"row {row} (position {position}) did not select its whole causal prefix"


def _cpu_reference_fixture(*, budget: int = 16) -> Fixture:
    """Small real QSA pool/backend used for the P4-compatible Torch reference path."""
    return Fixture(
        parsed_config(num_layers=4, budget=budget, ratio=4),
        num_pages=8,
        device="cpu",
        dtype=torch.bfloat16,
    )


def test_cpu_reference_norm_rope_matches_explicit_fp32_formula():
    fixture = _cpu_reference_fixture()
    backend = fixture.backend
    x = torch.arange(backend.index_head_dim, dtype=torch.bfloat16).view(1, -1) / 17
    weight = torch.linspace(-0.1, 0.1, backend.index_head_dim, dtype=torch.bfloat16)
    position = torch.tensor([3], dtype=torch.int32)

    got = backend._torch_index_norm_rope(x, position, weight, 1.0e-6)
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1.0e-6)
    value = value * (1.0 + weight.float())
    cache = backend._index_rope_cache()[3]
    cos, sin = cache[: backend._index_rope_cache().shape[1] // 2], cache[
        backend._index_rope_cache().shape[1] // 2 : backend._index_rope_cache().shape[1]
    ]
    half = cos.numel()
    rotated = value[:, : 2 * half]
    partner = torch.cat((-rotated[:, half:], rotated[:, :half]), dim=-1)
    expected = torch.cat(
        (
            rotated * torch.cat((cos, cos)) + partner * torch.cat((sin, sin)),
            value[:, 2 * half :],
        ),
        dim=-1,
    ).to(torch.bfloat16)
    torch.testing.assert_close(got, expected)


def test_cpu_reference_selection_has_stable_topk_and_causal_tail():
    fixture = _cpu_reference_fixture(budget=8)
    backend = fixture.backend
    req = fixture.req(0, 0, 8)
    batch = fixture.batch([req], "prefill")
    md = batch.attn_metadata
    md.positions = torch.tensor([2, 3, 4], dtype=torch.int32)
    md.token_to_req = torch.zeros(3, dtype=torch.int32)
    md.seq_lens = torch.tensor([8], dtype=torch.int32)
    md.block_table = torch.tensor([[0]], dtype=torch.int32)
    index = SimpleNamespace(
        q=torch.zeros(3, backend.index_heads, backend.index_head_dim, dtype=torch.bfloat16),
        q_norm_weight=torch.zeros(backend.index_head_dim, dtype=torch.bfloat16),
        eps=1.0e-6,
    )
    got = backend._select_torch(index, md, 0)

    assert torch.equal(got[0, :3], torch.tensor([0, 1, 2], dtype=torch.int32))
    assert torch.equal(got[1, :4], torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    assert torch.equal(got[2, :5], torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32))
    assert torch.equal(got[:, 5:], torch.full_like(got[:, 5:], -1))


def test_cpu_reference_selection_tied_blocks_choose_lowest_block_first():
    fixture = _cpu_reference_fixture(budget=8)
    backend = fixture.backend
    req = fixture.req(0, 0, 16)
    fixture.pool.cmp_k_cache(0).zero_()
    md = fixture.batch([req], "prefill").attn_metadata
    md.positions = torch.tensor([15], dtype=torch.int32)
    md.token_to_req = torch.zeros(1, dtype=torch.int32)
    md.seq_lens = torch.tensor([16], dtype=torch.int32)
    md.block_table = torch.tensor([[0]], dtype=torch.int32)
    index = SimpleNamespace(
        q=torch.zeros(1, backend.index_heads, backend.index_head_dim, dtype=torch.bfloat16),
        q_norm_weight=torch.zeros(backend.index_head_dim, dtype=torch.bfloat16),
        eps=1.0e-6,
    )

    got = backend._select(index, md, 0)

    assert torch.equal(got[0, :8], torch.arange(8, dtype=torch.int32))
    assert torch.equal(got[0, 8:], torch.full((3,), -1, dtype=torch.int32))


def test_cpu_reference_attention_isolates_two_requests_by_physical_page():
    fixture = _cpu_reference_fixture()
    backend = fixture.backend
    q = torch.zeros(2, 4, 256, dtype=torch.bfloat16)
    k_cache = torch.zeros(2, 64, 2, 256, dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    v_cache[0].fill_(11)
    v_cache[1].fill_(22)
    indices = torch.zeros(2, 1, dtype=torch.int32)
    block_table = torch.tensor([[1], [0]], dtype=torch.int32)
    token_to_req = torch.tensor([0, 1], dtype=torch.int32)

    got = backend._attend_torch(q, k_cache, v_cache, indices, block_table, token_to_req)

    torch.testing.assert_close(got[0], torch.full_like(got[0], 22))
    torch.testing.assert_close(got[1], torch.full_like(got[1], 11))


@pytest.mark.parametrize("physical_page", [-1, 2])
def test_cpu_reference_attention_rejects_negative_or_out_of_range_physical_page(
    physical_page: int,
):
    fixture = _cpu_reference_fixture()
    backend = fixture.backend
    q = torch.zeros(1, 4, 256, dtype=torch.bfloat16)
    k_cache = torch.zeros(2, 64, 2, 256, dtype=torch.bfloat16)
    indices = torch.zeros(1, 1, dtype=torch.int32)
    block_table = torch.tensor([[physical_page]], dtype=torch.int32)
    token_to_req = torch.zeros(1, dtype=torch.int32)

    with pytest.raises(ValueError, match="selected physical row is out of bounds"):
        backend._attend_torch(q, k_cache, k_cache, indices, block_table, token_to_req)


def test_cpu_reference_select_dispatches_to_torch_reference(monkeypatch):
    fixture = _cpu_reference_fixture()
    backend = fixture.backend
    expected = torch.tensor([[7]], dtype=torch.int32)
    seen = {}

    def fake_select_torch(index, md, slot):
        seen["args"] = (index, md, slot)
        return expected

    monkeypatch.setattr(backend, "_select_torch", fake_select_torch)
    got = backend._select(object(), object(), 3)

    assert got is expected
    assert seen["args"][2] == 3


def test_cpu_reference_capture_graph_is_eager_only():
    fixture = _cpu_reference_fixture()

    with pytest.raises(NotImplementedError, match="eager-only"):
        fixture.backend.init_capture_graph(max_seq_len=128, bs_list=[1])


def test_cpu_reference_paged_gqa_uses_page_table_and_matches_dense_oracle():
    fixture = _cpu_reference_fixture()
    attn = fixture.layer(QSA_LAYER, seed=31)
    req = fixture.req(0, 0, 8)
    # Move the request's page away from page zero after allocation.  Both the K/V store and
    # the reference gather must follow the authoritative page-table mapping.
    fixture.page_table[0, : fixture.page_size] = torch.arange(
        2 * fixture.page_size,
        3 * fixture.page_size,
        dtype=fixture.page_table.dtype,
    )
    x = _inputs(fixture, [8], seed=37)[0]
    qsa = attn.forward(x, fixture.batch([req], "prefill"))

    oracle = _dense_oracle(fixture)
    fixture.ctx.attn_backend = oracle
    oracle_req = fixture.req(1, 0, 8)
    dense = attn.forward(x, fixture.batch([oracle_req], "prefill"))
    torch.testing.assert_close(qsa.float(), dense.float(), rtol=2e-2, atol=2e-2)

    for req, backend in ((req, fixture.backend), (oracle_req, oracle)):
        req.cached_len, req.device_len, req.extend_len = 8, 9, 1
        fixture.page_table[req.table_idx, 8] = 2 * fixture.page_size + 8
        fixture.ctx.attn_backend = backend
        step = attn.forward(
            _inputs(fixture, [1], seed=41)[0],
            fixture.batch([req], "decode"),
        )
        if backend is fixture.backend:
            qsa_decode = step
        else:
            dense_decode = step
    torch.testing.assert_close(qsa_decode.float(), dense_decode.float(), rtol=2e-2, atol=2e-2)


def test_cpu_reference_group_straddle_matches_one_shot_prefill():
    config_kwargs = dict(num_layers=4, budget=16, ratio=4)
    one = Fixture(parsed_config(**config_kwargs), num_pages=8, device="cpu", dtype=torch.bfloat16)
    split = Fixture(parsed_config(**config_kwargs), num_pages=8, device="cpu", dtype=torch.bfloat16)
    one_attn = one.layer(QSA_LAYER, seed=43)
    split_attn = split.layer(QSA_LAYER, seed=43)
    x = _inputs(one, [9], seed=47)[0]
    one_shot = one_attn.forward(x, one.batch([one.req(0, 0, 9)], "prefill"))

    first_req = split.req(0, 0, 3)
    split_attn.forward(x[:3], split.batch([first_req], "prefill"))
    second_req = SimpleNamespace(table_idx=0, cached_len=3, device_len=9, extend_len=6)
    chunked = split_attn.forward(x[3:], split.batch([second_req], "prefill"))
    torch.testing.assert_close(chunked.float(), one_shot[3:].float(), rtol=2e-2, atol=2e-2)


def test_cpu_reference_attention_fails_closed_for_all_invalid_rows():
    fixture = _cpu_reference_fixture()
    backend = fixture.backend
    q = torch.randn(2, 4, 256, dtype=torch.bfloat16)
    cache = torch.randn(2, 64, 2, 256, dtype=torch.bfloat16)
    indices = torch.full((2, 7), -1, dtype=torch.int32)
    block_table = torch.tensor([[0], [1]], dtype=torch.int32)
    requests = torch.tensor([0, 1], dtype=torch.int32)
    got = backend._attend_torch(q, cache, cache.clone(), indices, block_table, requests)
    assert torch.equal(got, torch.zeros_like(got))
    with pytest.raises(ValueError, match="integral GQA"):
        backend._attend_torch(
            torch.randn(2, 3, 256, dtype=torch.bfloat16),
            cache,
            cache,
            indices,
            block_table,
            requests,
        )


@requires_cuda
def test_prefill_is_dense_below_the_budget(monkeypatch):
    """bs=3 ragged prefill, longest request exactly at budget + ratio - 1."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths = [2051, 1000, 137]
    inputs = _inputs(fixture, lengths)
    x = torch.cat([row[:n] for row, n in zip(inputs, lengths)])
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    seen = selection_spy(monkeypatch, fixture.backend)
    batch = fixture.batch(reqs, "prefill")
    got = attn.forward(x, batch)
    _assert_selection_is_causal_prefix(seen["indices"], batch.positions)

    fixture.ctx.attn_backend = _dense_oracle(fixture)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


def _dense_oracle(fixture: Fixture):
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference

    return TorchDenseQSAReference(
        fixture.config,
        num_slots=fixture.num_req_slots,
        max_len=4096,
        device=fixture.device,
        dtype=fixture.dtype,
    )


@requires_cuda
def test_decode_is_dense_below_the_budget(monkeypatch):
    """Prefill then five decode steps, sparse path vs the fp32 dense oracle."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411, 64], 5
    inputs = _inputs(fixture, lengths, extra=steps)
    oracle = _dense_oracle(fixture)

    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    seen = selection_spy(monkeypatch, fixture.backend)

    steps_x = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    steps_x += [
        torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)
    ]
    for step, x in enumerate(steps_x):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        fixture.ctx.attn_backend = fixture.backend
        got = attn.forward(x, batch)
        _assert_selection_is_causal_prefix(seen["indices"], batch.positions)
        fixture.ctx.attn_backend = oracle
        reference = attn.forward(x, batch)
        torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
def test_flashinfer_dense_matches_the_sparse_path():
    """The engine's dense FULL backend over the same pool, as an independent oracle."""
    pytest.importorskip("flashinfer")
    from freetoken.attention.fi import FlashInferBackend

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 500
    x = _inputs(fixture, [length])[0]
    req = fixture.req(0, 0, length)
    got = attn.forward(x, fixture.batch([req], "prefill"))

    dense = FlashInferBackend(config)
    fixture.ctx.attn_backend = SimpleNamespace(
        qsa_forward=lambda q, k, v, index, layer_id, batch: dense.forward(
            q, k, v, layer_id, batch
        )
    )
    batch = fixture.batch([req], "prefill")
    dense.prepare_metadata(batch)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
@pytest.mark.parametrize("cut", [1001, 4096, 4097], ids=["unaligned", "page-boundary", "boundary+1"])
def test_chunked_prefill_matches_one_shot(cut: int):
    """Cut points that are not multiples of index_ratio exercise the dual-source compress."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=512)
    attn = fixture.layer(QSA_LAYER)
    length = 5000
    x = _inputs(fixture, [length])[0]

    one_shot = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))
    head = fixture.req(1, 0, cut)
    attn.forward(x[:cut], fixture.batch([head], "prefill"))
    tail = fixture.req(1, cut, length)
    got = attn.forward(x[cut:], fixture.batch([tail], "prefill"))
    assert torch.equal(got, one_shot[cut:])


@requires_cuda
def test_decode_graph_replay_matches_eager():
    config = parsed_config()
    fixture = Fixture(config, num_pages=256)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411], 4
    bs = len(lengths)
    inputs = _inputs(fixture, lengths, extra=steps)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    attn.forward(
        torch.cat([row[:n] for row, n in zip(inputs, lengths)]),
        fixture.batch(reqs, "prefill"),
    )

    fixture.backend.init_capture_graph(max_seq_len=fixture.page_table.shape[1], bs_list=[bs])
    dummy = SimpleNamespace(
        table_idx=fixture.num_req_slots - 1, cached_len=1, device_len=2, extend_len=1
    )
    static = {
        "x": torch.zeros(bs, config.hidden_size, device=fixture.device, dtype=fixture.dtype),
        "positions": torch.zeros(bs, dtype=torch.int32, device=fixture.device),
        "out_loc": torch.zeros(bs, dtype=torch.int32, device=fixture.device),
    }
    capture_batch = SimpleNamespace(
        padded_reqs=[dummy] * bs, reqs=[dummy] * bs, phase="decode", size=bs, padded_size=bs,
        is_prefill=False, is_decode=True, positions=static["positions"],
        out_loc=static["out_loc"], attn_metadata=None, active_table_idx=None,
    )
    fixture.backend.prepare_for_capture(capture_batch)
    attn.forward(static["x"], capture_batch)  # warmup, same metadata object as the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = attn.forward(static["x"], capture_batch)
    torch.cuda.synchronize()

    for step in range(steps):
        for req in reqs:
            fixture.step(req)
        x = torch.stack([row[n + step] for row, n in zip(inputs, lengths)])
        batch = fixture.batch(reqs, "decode")
        static["x"].copy_(x)
        static["positions"].copy_(batch.positions)
        static["out_loc"].copy_(batch.out_loc)
        fixture.backend.prepare_for_replay(batch)
        # replay must stage into the captured buffers, never reallocate them
        md = batch.attn_metadata
        assert md.block_table.data_ptr() == fixture.backend._graph["block_table"].data_ptr()
        graph.replay()
        replayed = captured_out.clone()
        eager = attn.forward(x, fixture.batch(reqs, "decode"))
        assert torch.equal(replayed, eager), f"graph replay diverged at decode step {step}"


@requires_cuda
def test_row_chunked_scoring_matches_one_chunk(monkeypatch):
    """The scoring workspace bound splits long prefills into row chunks."""
    import freetoken.attention.qsa_sparse as qsa_sparse

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 600
    x = _inputs(fixture, [length])[0]
    whole = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))

    columns = fixture.page_table.shape[1] // config.qwen4_args.index_ratio
    monkeypatch.setattr(qsa_sparse, "_LOGITS_WORKSPACE_BYTES", 64 * columns * 4)
    chunked = attn.forward(x, fixture.batch([fixture.req(1, 0, length)], "prefill"))
    assert torch.equal(chunked, whole)


@requires_cuda
def test_two_qsa_layers_keep_separate_slab_slots(monkeypatch):
    """Both QSA layers of one forward must hit their own slab slot and ring slice."""
    config = parsed_config(num_layers=8)
    assert config.attention_groups[1].layer_ids == (3, 7)
    fixture = Fixture(config, num_pages=64)
    layers = [fixture.layer(layer_id, seed=layer_id) for layer_id in (3, 7)]
    oracle = _dense_oracle(fixture)
    lengths, steps = [200, 71], 3
    inputs = _inputs(fixture, lengths, extra=steps)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    xs = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    xs += [torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)]
    for step, x in enumerate(xs):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        for attn in layers:
            fixture.ctx.attn_backend = fixture.backend
            got = attn.forward(x, batch)
            fixture.ctx.attn_backend = oracle
            reference = attn.forward(x, batch)
            torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)

    slab = fixture.pool.cmp_k_cache
    assert not torch.equal(slab(0), slab(1))
