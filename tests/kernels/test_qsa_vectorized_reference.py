"""H0 parity and dispatch tests for the opt-in vectorized QSA reference."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from freetoken.attention.qsa_sparse import (
    QSARequestSpan,
    QSASparseAttnBackend,
    resolve_qsa_selection_path,
)


def _backend(
    *, token_topk: int = 8, ratio: int = 4, device: torch.device | None = None
) -> QSASparseAttnBackend:
    device = device or torch.device("cpu")
    backend = object.__new__(QSASparseAttnBackend)
    backend.index_heads = 2
    backend.index_head_dim = 4
    backend.token_topk = token_topk
    backend.ratio = ratio
    backend.page_size = 8
    backend.cmp_page_size = backend.page_size // ratio
    backend.block_topk = token_topk // ratio
    backend.select_width = token_topk + ratio - 1
    backend.device = device
    backend.dtype = torch.float32
    backend._graph = {}
    backend._index_rope_cache = lambda: torch.cat(
        (
            torch.ones(4096, 2),
            torch.zeros(4096, 2),
        ),
        dim=1,
    ).to(device)
    return backend


def _fixture(*, seed: int, ragged: bool = False, device: torch.device | None = None):
    torch.manual_seed(seed)
    device = device or torch.device("cpu")
    backend = _backend(device=device)
    if ragged:
        spans = (
            QSARequestSpan(0, 6, 0, 6, 0),
            QSARequestSpan(6, 13, 3, 10, 1),
        )
        positions = torch.tensor([*range(6), *range(3, 10)], dtype=torch.int32)
        req = torch.tensor([0] * 6 + [1] * 7, dtype=torch.int32)
        seq_lens = torch.tensor([6, 10], dtype=torch.int32)
        block_table = torch.tensor([[1, 0], [3, 2]], dtype=torch.int32)
        rows = positions.numel()
        physical_block_count = 8
    else:
        spans = (QSARequestSpan(0, 17, 0, 17, 0),)
        positions = torch.arange(17, dtype=torch.int32)
        req = torch.zeros(17, dtype=torch.int32)
        seq_lens = torch.tensor([17], dtype=torch.int32)
        block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
        rows = positions.numel()
        physical_block_count = 6

    # A shuffled physical-page mapping makes the request-level gather observable.
    physical = torch.randn(physical_block_count, 4)
    cmp_pages = physical.to(device).view(
        -1, backend.cmp_page_size, 1, backend.index_head_dim
    )
    backend._cmp_pages = lambda _slot: cmp_pages
    index = SimpleNamespace(
        q=torch.randn(rows, backend.index_heads, backend.index_head_dim, device=device),
        q_norm_weight=torch.zeros(backend.index_head_dim, device=device),
        eps=1e-5,
    )
    md = SimpleNamespace(
        positions=positions.to(device),
        token_to_req=req.to(device),
        seq_lens=seq_lens.to(device),
        block_table=block_table.to(device),
        request_spans=spans,
    )
    return backend, index, md


def test_vectorized_reference_matches_scalar_oracle_for_random_ragged_requests() -> None:
    backend, index, md = _fixture(seed=11, ragged=True)

    expected = backend._select_torch(index, md, 0)
    actual = backend._select_vectorized_reference(index, md, 0)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_vectorized_reference_matches_scalar_oracle_on_cuda() -> None:
    device = torch.device("cuda", 0)
    backend, index, md = _fixture(seed=16, ragged=True, device=device)

    expected = backend._select_torch(index, md, 0)
    actual = backend._select_vectorized_reference(index, md, 0)
    torch.cuda.synchronize(device)

    assert torch.equal(actual, expected)


def test_vectorized_reference_preserves_stable_tie_order() -> None:
    backend, index, md = _fixture(seed=12)
    index.q.zero_()
    cmp_pages = backend._cmp_pages(0)
    cmp_pages.zero_()

    expected = backend._select_torch(index, md, 0)
    actual = backend._select_vectorized_reference(index, md, 0)

    assert torch.equal(actual, expected)
    assert actual[7, :8].tolist() == list(range(8))
    assert actual[16, :9].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 16]


@pytest.mark.parametrize("length", (2048, 2049))
def test_vectorized_reference_preserves_rows_and_output_at_2048_boundary(length: int) -> None:
    backend = _backend(token_topk=2048, ratio=4)
    positions = torch.tensor([length - 1], dtype=torch.int32)
    req = torch.zeros(1, dtype=torch.int32)
    page_count = (length + backend.page_size - 1) // backend.page_size
    md = SimpleNamespace(
        positions=positions,
        token_to_req=req,
        seq_lens=torch.tensor([length], dtype=torch.int32),
        block_table=torch.arange(page_count, dtype=torch.int32).view(1, -1),
        request_spans=(QSARequestSpan(0, 1, length - 1, length, 0),),
    )
    keys = torch.randn(page_count * backend.cmp_page_size, 4)
    backend._cmp_pages = lambda _slot: keys.view(-1, 2, 1, 4)
    index = SimpleNamespace(
        q=torch.randn(1, 2, 4),
        q_norm_weight=torch.zeros(4),
        eps=1e-5,
    )

    expected = backend._select_torch(index, md, 0)
    actual = backend._select_vectorized_reference(index, md, 0)

    # Batched FP32 GEMM may exchange nearly equal score neighbors, but the exact selected-row
    # multiset and stable exact-tie behavior remain unchanged.
    assert torch.equal(torch.sort(actual, dim=1).values, torch.sort(expected, dim=1).values)

    q = torch.randn(1, 2, 4)
    k_cache = torch.randn(page_count, backend.page_size, 1, 4)
    v_cache = torch.randn_like(k_cache)
    expected_output = backend._attend_torch(q, k_cache, v_cache, expected, md.block_table, req)
    actual_output = backend._attend_torch(q, k_cache, v_cache, actual, md.block_table, req)

    torch.testing.assert_close(actual_output, expected_output, rtol=2e-6, atol=2e-6)


def test_vectorized_reference_requires_host_request_spans() -> None:
    backend, index, md = _fixture(seed=14)
    del md.request_spans

    with pytest.raises(RuntimeError, match="request spans"):
        backend._select_vectorized_reference(index, md, 0)


def test_vectorized_reference_rejects_page_table_that_cannot_cover_request() -> None:
    backend, index, md = _fixture(seed=15)
    md.block_table = torch.empty((1, 0), dtype=torch.int32)

    with pytest.raises(ValueError, match="page table"):
        backend._select_vectorized_reference(index, md, 0)


def test_vectorized_selection_has_no_per_token_scalar_or_selected_token_loop() -> None:
    source = inspect.getsource(QSASparseAttnBackend._select_vectorized_reference)

    assert ".item(" not in source
    assert "for row in" not in source
    assert "tokens.extend" not in source


@pytest.mark.parametrize(
    ("requested", "expected"),
    (
        ("auto", "torch-fp32-reference"),
        ("torch-fp32-reference", "torch-fp32-reference"),
        ("torch-fp32-vectorized-reference", "torch-fp32-vectorized-reference"),
    ),
)
def test_qsa_selection_dispatch_is_typed_and_default_off(requested: str, expected: str) -> None:
    assert (
        resolve_qsa_selection_path(requested, reference_only=True, sm70_supported=False) == expected
    )


def test_qsa_selection_dispatch_rejects_unknown_path() -> None:
    with pytest.raises(ValueError, match="qsa selection path"):
        resolve_qsa_selection_path("not-a-path", reference_only=True, sm70_supported=False)
