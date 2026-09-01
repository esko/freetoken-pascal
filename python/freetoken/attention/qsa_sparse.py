"""Qwen3.8-Flash-Next QSA compressed-block sparse attention backend.

Serves ``AttnType.QSA`` over ``kvcache/qsa_pool.py``: paged GQA K/V for the 12 full-attention
layers, a compressed index-key slab holding one key per ``index_ratio`` tokens, and a
per-request pending ring for the group a forward leaves open. The 36 GDN layers never reach
this backend, and the model has no dense attention layer, so :meth:`forward` is not served --
the only entry point is :meth:`qsa_forward` (``models/qwen4_exp/attention.py``).

One QSA layer's forward, all ragged over ``[T, ...]`` metadata:

1. store K/V at ``batch.out_loc``;
2. pool each row's closing group (members at positions >= ``cached_len`` come from this
   forward's raw index keys, the older ones from the pending ring), zero-centered rmsnorm it
   and rope it at the group's first position, then scatter it into the slab row
   ``out_loc // index_ratio`` (rows whose group does not close land on the request's scratch
   row and are never read);
3. store this forward's last ``ring_capacity`` raw index keys per request into the ring;
4. norm+rope the indexer queries at their own positions;
5. score every COMPLETE visible block (``sum_h relu(<q_h, k_bar_b>) / sqrt(index_head_dim)``,
   clamped to ``kvlen // index_ratio`` -- slab rows are never cleared, so stale rows must stay
   unreachable), take the top ``index_budget // index_ratio`` blocks, expand them to token
   indices plus the causal tail of the open group;
6. attend to exactly those tokens.

Addressing: the engine pins ``page_size == 64`` (this backend's ``page_sizes``), so a group of
``index_ratio`` tokens never straddles a page and ``block_table[req, p] = page_table[req, p *
64] // 64`` names both the K/V page and, viewed as ``page_size // index_ratio`` compressed
rows, the block's slab page. Decode stages that table plus the live lengths and table_idx into
static buffers (``prepare_for_replay``) so the whole path is CUDA-graph capturable.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict

import torch

from freetoken.core import Batch, get_global_ctx
from freetoken.utils import init_logger

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata
from .qsa_workspace import (
    QSA_LOGITS_WORKSPACE_BYTES,
    qsa_score_chunk_rows,
    qsa_vectorized_score_chunk_rows,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

# CPU-only reference fixtures do not have a CUDA pinned allocator.  Keep host metadata pinned
# when CUDA is available (the production path), but make the reference path usable on a plain
# CPU too.
_CPU_PINNED = {
    "device": "cpu",
    "dtype": torch.int32,
    "pin_memory": torch.cuda.is_available(),
}
# Block-score transient budget (vLLM's number): the fp32 [rows, n_blocks] logits tile is
# 256 KB per row at a 1M-token context, so a long prefill must be scored in row chunks.
_LOGITS_WORKSPACE_BYTES = QSA_LOGITS_WORKSPACE_BYTES


TORCH_TOPK_ENV = "FREETOKEN_QSA_TORCH_TOPK"

QSA_SELECTION_PATHS = (
    "auto",
    "torch-fp32-reference",
    "torch-fp32-vectorized-reference",
)
QSASelectionPath = Literal[
    "auto",
    "torch-fp32-reference",
    "torch-fp32-vectorized-reference",
]


@dataclass(frozen=True)
class QSARequestSpan:
    """Host-owned bounds for one ragged request in flattened QSA rows."""

    token_start: int
    token_stop: int
    cached_len: int
    device_len: int
    table_idx: int


def resolve_qsa_selection_path(
    requested: str,
    *,
    reference_only: bool,
    sm70_supported: bool,
) -> str:
    """Resolve QSA selection while keeping the vectorized reference explicit-only."""
    if requested not in QSA_SELECTION_PATHS:
        choices = ", ".join(QSA_SELECTION_PATHS)
        raise ValueError(f"invalid qsa selection path {requested!r}; expected one of {choices}")
    if requested == "auto":
        return "torch-fp32-reference" if reference_only or not sm70_supported else "triton"
    return requested

QSAPhase = Literal[
    "store_kv",
    "index_cache_composite",
    "selection_composite",
    "selected_row_attention",
]
QSAPhaseEvent = Literal["begin", "end"]


class QSAPhaseMetadata(TypedDict):
    """Scalar identity attached to an eager reference phase boundary."""

    phase: QSAPhase
    layer_id: int
    slot: int
    path: str


QSAPhaseObserver = Callable[[QSAPhaseEvent, Mapping[str, object]], None]


def _resolve_block_topk() -> Callable | None:
    """The in-repo Triton block top-k, or None to fall back on torch.topk."""
    if os.getenv(TORCH_TOPK_ENV, "0") == "1":
        logger.info(f"qsa_sparse block top-k: torch.topk ({TORCH_TOPK_ENV}=1)")
        return None
    try:
        from freetoken.kernel.triton.qsa import qsa_block_topk
    except Exception as exc:
        logger.info(f"qsa_sparse block top-k: torch.topk (triton unavailable: {exc})")
        return None
    logger.info("qsa_sparse block top-k: triton qsa_block_topk")
    return qsa_block_topk


@dataclass
class QSASparseMetadata(BaseAttnMetadata):
    # fmt: off
    is_decode:        bool
    last_indices:     torch.Tensor  # gpu
    qo_indptr_cpu:    torch.Tensor  # cpu pinned int32 [bs+1]
    kv_len_cpu:       torch.Tensor  # cpu pinned int32 [bs]
    # Ragged per-token / per-request addressing. Decode defers these to the static graph
    # buffers (prepare_for_replay) or to a lazy eager snapshot at the first QSA layer.
    token_to_req:     torch.Tensor | None = None  # [T] int32
    cu_seqlens:       torch.Tensor | None = None  # [bs+1] int32
    seq_lens:         torch.Tensor | None = None  # [bs] int32, device_len
    ring_slots:       torch.Tensor | None = None  # [bs] int32, Req.table_idx
    block_table:      torch.Tensor | None = None  # [bs, W//page_size] int32, physical page ids
    # Per-forward scatter plans, built once by the first QSA layer and reused by the rest.
    # positions is bound here (not in prepare_metadata) because a capture batch has none yet.
    cmp_rows:         torch.Tensor | None = None  # [T] int32, compressed slab destination
    ring_rows:        torch.Tensor | None = None  # [T] int32, flat ring row or -1
    positions:        torch.Tensor | None = None  # [T] int32, logical query positions
    # Host scheduler spans avoid reading request lengths back from CUDA in the optional
    # vectorized reference selector.
    request_spans:    tuple[QSARequestSpan, ...] = ()
    # fmt: on

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class QSASparseAttnBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from freetoken.kvcache.qsa_pool import QSAKVCache

        args = config.qwen4_args
        assert args is not None, "qsa_sparse backend needs ModelConfig.qwen4_args"
        self.head_dim = config.head_dim
        self.index_heads = args.index_n_heads
        self.token_topk = args.index_budget
        self.kvcache = get_global_ctx().kv_cache
        assert isinstance(self.kvcache, QSAKVCache), (
            f"qsa_sparse backend needs a QSA pool, got {type(self.kvcache).__name__}"
        )
        self.device = self.kvcache.device
        self.dtype = self.kvcache.dtype
        self.reference_only = bool(getattr(config, "reference_only", False))
        self.index_head_dim = self.kvcache.index_head_dim
        self.ratio = self.kvcache.index_ratio
        self.ring_capacity = self.kvcache.ring_capacity
        self.page_size = get_global_ctx().page_size
        assert self.page_size % self.ratio == 0, (
            f"QSA needs page_size ({self.page_size}) divisible by index_ratio ({self.ratio})"
        )
        self.cmp_page_size = self.page_size // self.ratio
        self.block_topk = self.token_topk // self.ratio
        self.select_width = self.token_topk + self.ratio - 1
        assert self.token_topk % self.ratio == 0, "QSA budget must be a whole number of blocks"
        # The sparse attend kernel bakes 1/sqrt(head_dim) into its exp2 scale.
        assert config.attn_sm_scale in (None, self.head_dim**-0.5), (
            "qsa_sparse serves the default 1/sqrt(head_dim) attention scale only"
        )
        # QSA layer -> index slab slot, in sparse-layer order (the pool's own convention).
        group = self._qsa_group(config)
        self._idx_slot = {lid: i for i, lid in enumerate(group.layer_ids)}
        self.rotary_config = group.rotary_config
        self._index_cos_sin: torch.Tensor | None = None

        from freetoken.utils.arch import is_sm70_supported

        self.selected_path = resolve_qsa_selection_path(
            getattr(config, "qsa_selection_path", "auto"),
            reference_only=self.reference_only,
            sm70_supported=is_sm70_supported(),
        )
        self._vectorized_reference = self.selected_path == "torch-fp32-vectorized-reference"
        self._torch_reference = self.selected_path != "triton"
        self._block_topk_kernel = None if self._torch_reference else _resolve_block_topk()
        self._phase_observer: QSAPhaseObserver | None = None
        logger.info_rank0(f"QSA sparse attention path: {self.selected_path}")
        # decode staging (static buffers under CUDA graphs; eager decode snapshots per step)
        self._graph: dict[str, torch.Tensor] = {}
        self.capture_bs: list[int] = []

    def set_phase_observer(self, observer: QSAPhaseObserver | None) -> None:
        """Attach diagnostics to an eager Torch reference path."""
        if observer is not None and not callable(observer):
            raise TypeError("QSA phase observer must be callable or None")
        if observer is not None and not self._torch_reference:
            raise RuntimeError("QSA phase observer is eager Torch-reference only")
        self._phase_observer = observer

    @staticmethod
    def _qsa_group(config: ModelConfig):
        from freetoken.models.config import FullAttentionGroupConfig

        groups = [
            g
            for g in config.attention_groups
            if isinstance(g, FullAttentionGroupConfig) and g.index_ratio > 1
        ]
        assert len(groups) == 1, f"expected one QSA attention group, got {len(groups)}"
        return groups[0]

    # ----- slab views ---------------------------------------------------------------------
    def _cmp_pages(self, slot: int) -> torch.Tensor:
        """The compressed slab as ``[pages, page_size // ratio, 1, dim]``, the score kernel's
        paged layout. The scratch rows past ``cmp_scratch_base`` stay out of the view."""
        rows = self.kvcache.cmp_k_cache(slot)[: self.kvcache.cmp_scratch_base]
        return rows.view(-1, self.cmp_page_size, 1, self.index_head_dim)

    def _index_rope_cache(self) -> torch.Tensor:
        """cos/sin table of the indexer rope: same rotary_dim and frequencies as the main
        attention, ``head_size`` 128 instead of 256, so it is a separate get_rope instance.

        The table itself (not RotaryEmbedding.forward) because the indexer's norm+rope is one
        fused kernel and the compressed keys rope at their group's position, not the query's."""
        if self._index_cos_sin is None:
            from freetoken.layers.rotary import get_rope

            rotary = self.rotary_config
            with torch.device(self.device):
                rope = get_rope(
                    head_dim=self.index_head_dim,
                    rotary_dim=rotary.rotary_dim,
                    max_position=rotary.max_position,
                    base=rotary.base,
                    rope_scaling=tuple(rotary.scaling.items()) if rotary.scaling else None,
                    allow_reference_geometry=self.reference_only,
                )
            self._index_cos_sin = rope._cos_sin_cache.to(self.device)
        return self._index_cos_sin

    # ----- metadata -----------------------------------------------------------------------
    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        seqlens_q = [r.extend_len for r in reqs]
        seqlens_k = [r.device_len for r in reqs]
        is_decode = getattr(batch, "phase", None) == "decode"
        qo_indptr = torch.tensor([0, *seqlens_q], **_CPU_PINNED).cumsum_(0).to(torch.int32)
        kv_len = torch.tensor(seqlens_k, **_CPU_PINNED)
        spans: list[QSARequestSpan] = []
        offset = 0
        for request in reqs:
            extend = int(request.extend_len)
            cached_len = int(request.cached_len)
            device_len = int(request.device_len)
            if extend != device_len - cached_len or extend < 0:
                raise ValueError("QSA request span has inconsistent cached/device lengths")
            spans.append(
                QSARequestSpan(
                    offset,
                    offset + extend,
                    cached_len,
                    device_len,
                    int(request.table_idx),
                )
            )
            offset += extend
        last = (qo_indptr[1:].to(torch.int32) - 1).to(self.device, non_blocking=True)
        md = QSASparseMetadata(
            is_decode=is_decode,
            last_indices=last,
            qo_indptr_cpu=qo_indptr,
            kv_len_cpu=kv_len,
            request_spans=tuple(spans),
        )
        batch.attn_metadata = md
        if not is_decode:
            table_idx = torch.tensor([r.table_idx for r in reqs], **_CPU_PINNED)
            token_to_req = torch.repeat_interleave(
                torch.arange(len(reqs), dtype=torch.int32),
                torch.tensor(seqlens_q, dtype=torch.int32),
            )
            if _CPU_PINNED["pin_memory"]:
                token_to_req = token_to_req.pin_memory()
            md.cu_seqlens = qo_indptr.to(self.device, non_blocking=True)
            md.token_to_req = token_to_req.to(self.device, non_blocking=True)
            md.seq_lens = kv_len.to(self.device, non_blocking=True)
            md.ring_slots = table_idx.to(self.device, non_blocking=True)
            md.block_table = self._block_table(md.ring_slots.to(torch.int64))
        # Decode addressing is DEFERRED: a graph-bound step stages it into the static
        # buffers (prepare_for_replay), an eager step snapshots at the first QSA layer.

    def _block_base_view(self) -> torch.Tensor:
        """Every-``page_size``-th column of the page table: the per-page base slots. A strided
        VIEW, so gathering rows through it materializes only [bs, W/page_size]."""
        return get_global_ctx().page_table[:, :: self.page_size]

    def _block_table(self, table_idx: torch.Tensor) -> torch.Tensor:
        return (self._block_base_view().index_select(0, table_idx) // self.page_size).to(
            torch.int32
        )

    def _stage_decode(self, md: QSASparseMetadata, bs: int, table_idx: torch.Tensor) -> None:
        """Copy this step's addressing into the static graph buffers and point the metadata
        at them (restage-per-replay, m3/dsa precedent)."""
        self._graph["block_table"][:bs].copy_(
            self._block_base_view().index_select(0, table_idx) // self.page_size
        )
        self._graph["kvlen"][:bs].copy_(md.kv_len_cpu.to(self.device, non_blocking=True))
        self._graph["table_idx"][:bs].copy_(table_idx)
        md.block_table = self._graph["block_table"][:bs]
        md.seq_lens = self._graph["kvlen"][:bs]
        md.ring_slots = self._graph["table_idx"][:bs]
        md.token_to_req = self._graph["token_to_req"][:bs]
        md.cu_seqlens = self._graph["cu_seqlens"][: bs + 1]

    def _snapshot_decode(self, md: QSASparseMetadata, batch: Batch) -> None:
        """Eager decode (not graph-staged): this step's rows, once per forward. The live
        page-table row may mutate for the next batch while this one runs, so gather now."""
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        bs = len(reqs)
        table_idx = torch.tensor([r.table_idx for r in reqs], **_CPU_PINNED)
        md.ring_slots = table_idx.to(self.device, non_blocking=True)
        md.block_table = self._block_table(md.ring_slots.to(torch.int64))
        md.seq_lens = md.kv_len_cpu.to(self.device, non_blocking=True)
        md.token_to_req = torch.arange(bs, dtype=torch.int32, device=self.device)
        md.cu_seqlens = torch.arange(bs + 1, dtype=torch.int32, device=self.device)

    # ----- dense layers -------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "qsa_sparse serves QSA layers only (Qwen3.8-Flash-Next has no dense attention "
            "layer); the QSA layer calls qsa_forward"
        )

    # ----- QSA layers ---------------------------------------------------------------------
    def qsa_forward(
        self,
        q: torch.Tensor,  # [T, HQ, D]
        k: torch.Tensor,  # [T, KVH * D]
        v: torch.Tensor,  # [T, KVH * D]
        index,  # models.qwen4_exp.attention.QSAIndexerInputs
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        slot = self._idx_slot[layer_id]
        if self._phase_observer is None:
            return self._qsa_forward_unobserved(q, k, v, index, layer_id, batch, md, slot)
        if self._torch_reference:
            self._observe_phase(
                "store_kv",
                layer_id,
                slot,
                lambda: self._store_kv_torch(k, v, batch.out_loc, layer_id),
            )
        else:
            self._observe_phase(
                "store_kv",
                layer_id,
                slot,
                lambda: self.kvcache.store_kv(k, v, batch.out_loc, layer_id),
            )

        self._observe_phase(
            "index_cache_composite",
            layer_id,
            slot,
            lambda: self._maintain_index_cache(index, md, slot, batch),
        )

        if self._torch_reference:
            indices = self._observe_phase(
                "selection_composite",
                layer_id,
                slot,
                lambda: self._select(index, md, slot),
            )
            return self._observe_phase(
                "selected_row_attention",
                layer_id,
                slot,
                lambda: self._attend_torch(
                    q,
                    self.kvcache.k_cache(layer_id),
                    self.kvcache.v_cache(layer_id),
                    indices,
                    md.block_table,
                    md.token_to_req,
                ),
            )

        from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

        indices = self._observe_phase(
            "selection_composite",
            layer_id,
            slot,
            lambda: self._select(index, md, slot),
        )
        return self._observe_phase(
            "selected_row_attention",
            layer_id,
            slot,
            lambda: qsa_sparse_paged_attention(
                q,
                self.kvcache.k_cache(layer_id),
                self.kvcache.v_cache(layer_id),
                indices,
                md.block_table,
                md.token_to_req,
                torch.empty_like(q),
            ),
        )

    def _qsa_forward_unobserved(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index,
        layer_id: int,
        batch: Batch,
        md: QSASparseMetadata,
        slot: int,
    ) -> torch.Tensor:
        """Original allocation path, retained when diagnostics are disabled."""
        if self._torch_reference:
            self._store_kv_torch(k, v, batch.out_loc, layer_id)
        else:
            self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        if md.block_table is None:
            self._snapshot_decode(md, batch)
        if slot == 0 or md.cmp_rows is None:
            # Rebuilt at the first QSA layer of every forward, not cached on the metadata: a
            # capture batch runs its warmup and its capture through ONE metadata object, and a
            # cached plan would bake the warmup's addresses into the graph.
            self._plan_index_writes(md, batch)

        if self._torch_reference:
            self._update_index_cache_torch(index, md, slot, batch)
            indices = self._select(index, md, slot)
            return self._attend_torch(
                q,
                self.kvcache.k_cache(layer_id),
                self.kvcache.v_cache(layer_id),
                indices,
                md.block_table,
                md.token_to_req,
            )

        from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

        self._update_index_cache(index, md, slot)
        indices = self._select(index, md, slot)
        return qsa_sparse_paged_attention(
            q,
            self.kvcache.k_cache(layer_id),
            self.kvcache.v_cache(layer_id),
            indices,
            md.block_table,
            md.token_to_req,
            torch.empty_like(q),
        )

    def _observe_phase(
        self,
        phase: QSAPhase,
        layer_id: int,
        slot: int,
        fn: Callable[[], torch.Tensor | None],
    ) -> torch.Tensor | None:
        observer = self._phase_observer
        metadata: QSAPhaseMetadata = {
            "phase": phase,
            "layer_id": int(layer_id),
            "slot": int(slot),
            "path": self.selected_path,
        }
        if observer is not None:
            observer("begin", metadata)
        phase_error: BaseException | None = None
        try:
            return fn()
        except BaseException as error:
            phase_error = error
            raise
        finally:
            if observer is not None:
                try:
                    observer("end", metadata)
                except Exception:
                    if phase_error is None:
                        raise
                    logger.exception(
                        "QSA phase observer end callback failed while preserving phase error"
                    )

    def _maintain_index_cache(self, index, md: QSASparseMetadata, slot: int, batch: Batch) -> None:
        if md.block_table is None:
            self._snapshot_decode(md, batch)
        if slot == 0 or md.cmp_rows is None:
            # Rebuilt at the first QSA layer of every forward, not cached on the metadata: a
            # capture batch runs its warmup and its capture through ONE metadata object, and a
            # cached plan would bake the warmup's addresses into the graph.
            self._plan_index_writes(md, batch)
        if self._torch_reference:
            self._update_index_cache_torch(index, md, slot, batch)
        else:
            self._update_index_cache(index, md, slot)

    def _store_kv_torch(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        """Store K/V through the authoritative paged-cache layout without a CUDA kernel."""
        k_cache = self.kvcache.k_cache(layer_id)
        v_cache = self.kvcache.v_cache(layer_id)
        if k.ndim != 2 or v.shape != k.shape:
            raise ValueError("QSA reference K/V must be [tokens, kv_heads * head_dim]")
        expected = (k_cache.shape[-2], k_cache.shape[-1])
        if tuple(k.shape[1:]) != (expected[0] * expected[1],):
            raise ValueError(
                "QSA reference K/V shape "
                f"{tuple(k.shape)} does not match cache heads/dim {expected}"
            )
        if out_loc.ndim != 1 or out_loc.numel() != k.shape[0]:
            raise ValueError("QSA reference K/V locations must match token rows")
        k_cache.view(-1, *expected).index_copy_(0, out_loc.to(torch.long), k.view(-1, *expected))
        v_cache.view(-1, *expected).index_copy_(0, out_loc.to(torch.long), v.view(-1, *expected))

    def _plan_index_writes(self, md: QSASparseMetadata, batch: Batch) -> None:
        """Per-token slab row and ring row for this forward; the other QSA layers reuse it
        (it is layer-invariant). Pure device arithmetic: no host sync, graph-capturable."""
        md.positions = batch.positions
        out_loc = batch.out_loc.to(torch.int64)
        positions = batch.positions.to(torch.int64)
        rows = torch.arange(out_loc.numel(), device=self.device)
        req = md.token_to_req.to(torch.int64)
        slots = md.ring_slots.to(torch.int64).index_select(0, req)
        # out_loc % page_size == position % page_size and index_ratio divides page_size, so a
        # group closes exactly on out_loc % index_ratio == index_ratio - 1.
        closing = out_loc % self.ratio == self.ratio - 1
        scratch = self.kvcache.cmp_scratch_base + slots
        md.cmp_rows = torch.where(closing, out_loc // self.ratio, scratch).to(torch.int32)
        # Only the last ring_capacity rows of a request survive to the next forward; the rest
        # are masked off instead of dumped somewhere (vLLM rule).
        ends = md.cu_seqlens.to(torch.int64).index_select(0, req + 1)
        keep = rows >= ends - self.ring_capacity
        ring_row = slots * self.ring_capacity + positions % self.ring_capacity
        md.ring_rows = torch.where(keep, ring_row, torch.full_like(ring_row, -1)).to(torch.int32)

    def _update_index_cache(self, index, md: QSASparseMetadata, slot: int) -> None:
        """Compress each closing group into the slab, then refresh the pending ring."""
        from freetoken.kernel.triton.qsa import (
            qsa_compress_groups,
            qsa_index_norm_rope,
            qsa_store_rows,
        )

        rows = index.k.shape[0]
        ring = self.kvcache.pending_ring(slot)
        pooled = self._scratch("pooled", rows, self.index_head_dim, dtype=self.dtype)
        first = self._scratch("first_pos", rows, dtype=torch.int32)
        qsa_compress_groups(
            index.k,
            ring,
            md.ring_slots,
            md.token_to_req,
            md.cu_seqlens,
            md.positions,
            self.ratio,
            pooled,
            first,
        )
        qsa_index_norm_rope(
            pooled,
            first,
            self._index_rope_cache(),
            index.k_norm_weight,
            index.eps,
            self.kvcache.cmp_k_cache(slot),
            dest_rows=md.cmp_rows,
        )
        # After the compression read: the ring rows this forward overwrites are exactly the
        # ones a straddling group just consumed.
        qsa_store_rows(ring, md.ring_rows, index.k)

    def _torch_index_norm_rope(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        *,
        heads: int = 1,
    ) -> torch.Tensor:
        """FP32 reference for the indexer's grouped norm and partial NeoX rope."""
        if x.ndim != 2 or x.shape[1] != self.index_head_dim:
            raise ValueError(
                "QSA reference index rows must be "
                f"[rows, {self.index_head_dim}], got {tuple(x.shape)}"
            )
        if positions.ndim != 1 or heads <= 0 or x.shape[0] != positions.numel() * heads:
            raise ValueError("QSA reference index positions must match row/head geometry")
        if tuple(weight.shape) != (self.index_head_dim,):
            raise ValueError("QSA reference index norm weight has invalid geometry")
        cos_sin = self._index_rope_cache()
        rotary_dim = cos_sin.shape[1]
        if rotary_dim % 2 or rotary_dim > self.index_head_dim:
            raise ValueError("QSA reference index rope needs an even rotary_dim <= head_dim")

        value = x.float()
        value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
        value = value * (1.0 + weight.float()).view(1, -1)
        positions = positions.to(device=x.device, dtype=torch.long).repeat_interleave(heads)
        cache = cos_sin.index_select(0, positions)
        cos, sin = cache[:, : rotary_dim // 2], cache[:, rotary_dim // 2 : rotary_dim]
        rotated = value[:, :rotary_dim]
        half = rotary_dim // 2
        partner = torch.cat((-rotated[:, half:], rotated[:, :half]), dim=-1)
        rotated = rotated * torch.cat((cos, cos), dim=-1) + partner * torch.cat((sin, sin), dim=-1)
        return torch.cat((rotated, value[:, rotary_dim:]), dim=-1).to(dtype=x.dtype)

    def _update_index_cache_torch(
        self,
        index,
        md: QSASparseMetadata,
        slot: int,
        batch: Batch,
    ) -> None:
        """Compress pending/current index keys and store them with pure Torch operations."""
        if md.positions is None or md.cmp_rows is None or md.ring_rows is None:
            raise RuntimeError("QSA reference cache update requires planned index rows")
        if index.k.ndim != 2 or index.k.shape[1] != self.index_head_dim:
            raise ValueError("QSA reference index keys must be [tokens, index_head_dim]")
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        ring = self.kvcache.pending_ring(slot)
        compressed = self.kvcache.cmp_k_cache(slot)
        offset = 0
        for request in reqs:
            extend = int(request.extend_len)
            start = int(request.cached_len)
            end = int(request.device_len)
            if extend != end - start or offset + extend > index.k.shape[0]:
                raise ValueError("QSA reference request lengths do not match index rows")
            current = index.k[offset : offset + extend]
            current_positions = md.positions[offset : offset + extend]
            ring_slot = int(request.table_idx)
            first_end = ((start + self.ratio) // self.ratio) * self.ratio - 1
            for group_end in range(first_end, end, self.ratio):
                group_start = group_end - self.ratio + 1
                members = []
                for position in range(group_start, group_end + 1):
                    if position < start:
                        members.append(ring[ring_slot, position % self.ring_capacity])
                    else:
                        members.append(current[position - start])
                pooled = torch.stack(members).float().mean(dim=0).to(dtype=index.k.dtype)
                normalized = self._torch_index_norm_rope(
                    pooled.view(1, -1),
                    torch.tensor([group_start], dtype=torch.int32, device=index.k.device),
                    index.k_norm_weight,
                    index.eps,
                )[0]
                row = md.cmp_rows[offset + group_end - start]
                if int(row) < 0 or int(row) >= compressed.shape[0]:
                    raise ValueError("QSA reference compressed-row destination is out of bounds")
                compressed[row.long()] = normalized

            keep_start = max(start, end - self.ring_capacity)
            for position in range(keep_start, end):
                row = md.ring_rows[offset + position - start]
                if int(row) >= 0:
                    ring[ring_slot, position % self.ring_capacity] = current[position - start]
            # Keep this assertion local to the reference path: a malformed batch must not
            # silently leave stale rows in the next request's compression input.
            if current_positions.numel() != extend:
                raise ValueError("QSA reference index positions do not match request length")
            offset += extend
        if offset != index.k.shape[0]:
            raise ValueError("QSA reference index rows contain an unowned suffix")

    def _select_torch(self, index, md: QSASparseMetadata, slot: int) -> torch.Tensor:
        """Normalize/rope queries, score paged compressed keys, top-k, and expand on device."""
        if md.positions is None or md.block_table is None or md.token_to_req is None:
            raise RuntimeError("QSA reference selection requires complete metadata")
        rows = index.q.shape[0]
        if index.q.ndim != 3 or index.q.shape[1:] != (self.index_heads, self.index_head_dim):
            raise ValueError("QSA reference index queries have invalid geometry")
        q_index = self._torch_index_norm_rope(
            index.q.reshape(-1, self.index_head_dim),
            md.positions,
            index.q_norm_weight,
            index.eps,
            heads=self.index_heads,
        ).view(rows, self.index_heads, self.index_head_dim)
        cmp_pages = self._cmp_pages(slot)
        indices = torch.full(
            (rows, self.select_width), -1, dtype=torch.int32, device=index.q.device
        )
        reqs = md.token_to_req.to(torch.long)
        for row in range(rows):
            request = reqs[row]
            if int(request) < 0 or int(request) >= md.block_table.shape[0]:
                raise ValueError("QSA reference token-to-request index is out of bounds")
            position = md.positions[row]
            # Sequence lengths are device tensors, but this reference path deliberately uses
            # scalar loop bounds: no host copy of keys/values occurs and correctness is clearer.
            sequence_length = md.seq_lens[request]
            visible_blocks = min(
                int((position + 1) // self.ratio), int(sequence_length // self.ratio)
            )
            if visible_blocks:
                block_ids = torch.arange(visible_blocks, dtype=torch.long, device=index.q.device)
                page_ids = (
                    md.block_table[request]
                    .to(torch.long)
                    .index_select(
                        0, torch.div(block_ids, self.cmp_page_size, rounding_mode="floor")
                    )
                )
                keys = cmp_pages[page_ids, block_ids % self.cmp_page_size, 0]
                scores = torch.relu(q_index[row].float() @ keys.float().transpose(0, 1)).sum(0)
                scores.mul_(self.index_head_dim**-0.5)
                take = min(self.block_topk, visible_blocks)
                chosen = torch.argsort(scores, descending=True, stable=True)[:take]
            else:
                chosen = torch.empty(0, dtype=torch.long, device=index.q.device)

            tokens = []
            for block in chosen:
                block_start = int(block) * self.ratio
                tokens.extend(range(block_start, block_start + self.ratio))
            visible_tokens = int(position) + 1
            tail_start = (visible_tokens // self.ratio) * self.ratio
            tokens.extend(range(tail_start, tail_start + visible_tokens - tail_start))
            if tokens:
                values = torch.tensor(
                    tokens[: self.select_width], dtype=torch.int32, device=index.q.device
                )
                indices[row, : values.numel()] = values
        return indices

    def _select_vectorized_reference(self, index, md: QSASparseMetadata, slot: int) -> torch.Tensor:
        """Select QSA rows with request-batched FP32 Torch operations.

        This is an eager, explicit-only reference candidate.  Request bounds come from the
        scheduler's host spans; all score, ordering, expansion, and tail work remains on the
        selected device.  The scalar :meth:`_select_torch` implementation is intentionally kept
        unchanged as the permanent oracle.
        """
        if md.positions is None or md.block_table is None or md.token_to_req is None:
            raise RuntimeError("QSA vectorized selection requires complete metadata")
        spans = getattr(md, "request_spans", ())
        if not spans:
            raise RuntimeError("QSA vectorized selection requires host request spans")
        rows = index.q.shape[0]
        if index.q.ndim != 3 or index.q.shape[1:] != (self.index_heads, self.index_head_dim):
            raise ValueError("QSA vectorized index queries have invalid geometry")
        if md.positions.ndim != 1 or md.positions.numel() != rows:
            raise ValueError("QSA vectorized selection positions have invalid geometry")
        if md.token_to_req.ndim != 1 or md.token_to_req.numel() != rows:
            raise ValueError("QSA vectorized selection request indices have invalid geometry")
        if md.block_table.ndim != 2 or len(spans) != md.block_table.shape[0]:
            raise ValueError("QSA vectorized selection request spans do not match page table")

        expected_start = 0
        sparse_ranges: list[tuple[int, int, int]] = []
        tail_only_ranges: list[tuple[int, int]] = []
        max_blocks = 0
        for request_id, span in enumerate(spans):
            if not isinstance(span, QSARequestSpan):
                raise ValueError("QSA vectorized selection request spans are malformed")
            if span.token_start != expected_start or span.token_stop < span.token_start:
                raise ValueError("QSA vectorized selection request spans are not contiguous")
            if span.device_len - span.cached_len != span.token_stop - span.token_start:
                raise ValueError("QSA vectorized selection request span length is inconsistent")
            if span.cached_len < 0 or span.device_len < 0:
                raise ValueError("QSA vectorized selection request lengths must be non-negative")
            if span.table_idx < 0:
                raise ValueError("QSA vectorized selection request table index is negative")
            expected_start = span.token_stop

            block_count = span.device_len // self.ratio
            if block_count:
                # Even when every visible block fits the budget, preserve the scalar oracle's
                # stable score ordering.  Skipping the sort would change selected-row order and
                # can perturb the downstream reduction.
                sparse_ranges.append((request_id, span.token_start, span.token_stop))
                max_blocks = max(max_blocks, block_count)
            elif span.token_start < span.token_stop:
                tail_only_ranges.append((span.token_start, span.token_stop))
        if expected_start != rows:
            raise ValueError("QSA vectorized selection spans do not cover index rows")

        indices = torch.full(
            (rows, self.select_width), -1, dtype=torch.int32, device=index.q.device
        )
        token_offsets = torch.arange(self.ratio, dtype=torch.long, device=index.q.device)
        for start, stop in tail_only_ranges:
            row_ids = torch.arange(start, stop, dtype=torch.long, device=index.q.device)
            positions = md.positions.index_select(0, row_ids).to(torch.long)
            tail = token_offsets.unsqueeze(0).expand(stop - start, -1)
            tail_valid = token_offsets.unsqueeze(0) < (positions + 1).unsqueeze(1)
            packed = torch.full(
                (stop - start, self.select_width),
                -1,
                dtype=torch.int32,
                device=index.q.device,
            )
            packed[:, : self.ratio] = torch.where(tail_valid, tail, -1).to(torch.int32)
            indices.index_copy_(0, row_ids, packed)

        if not sparse_ranges:
            return indices
        columns = md.block_table.shape[1] * self.cmp_page_size
        if max_blocks <= 0:
            raise ValueError("QSA vectorized selection has no complete page-table blocks")
        if (max_blocks + self.cmp_page_size - 1) // self.cmp_page_size > md.block_table.shape[1]:
            raise ValueError("QSA vectorized selection request exceeds page table")

        q_index = self._torch_index_norm_rope(
            index.q.reshape(-1, self.index_head_dim),
            md.positions,
            index.q_norm_weight,
            index.eps,
            heads=self.index_heads,
        ).view(rows, self.index_heads, self.index_head_dim)
        # Convert once per forward so every score is an FP32 matmul.  Both buffers are bounded
        # by the configured context/page-table shape and are included in the H0 workspace plan.
        q_index_fp32 = self._scratch(
            "vector_q_index_fp32", rows, self.index_heads, self.index_head_dim, dtype=torch.float32
        )
        q_index_fp32.copy_(q_index)
        key_buffer = self._scratch(
            "vector_request_keys", max_blocks, self.index_head_dim, dtype=self.dtype
        )
        key_buffer_fp32 = self._scratch(
            "vector_request_keys_fp32", max_blocks, self.index_head_dim, dtype=torch.float32
        )
        cmp_pages = self._cmp_pages(slot)
        cmp_flat = cmp_pages.reshape(-1, self.index_head_dim)
        rows_per_chunk = qsa_vectorized_score_chunk_rows(rows, columns, self.index_heads)
        head_scores = self._scratch(
            "vector_score_heads",
            rows_per_chunk,
            self.index_heads,
            max_blocks,
            dtype=torch.float32,
        )
        logits = self._scratch("logits", rows_per_chunk, columns, dtype=torch.float32)

        for request_id, start, stop in sparse_ranges:
            span = spans[request_id]
            block_count = span.device_len // self.ratio
            block_ids = torch.arange(block_count, dtype=torch.long, device=index.q.device)
            page_ids = md.block_table[request_id].to(torch.long).index_select(
                0, torch.div(block_ids, self.cmp_page_size, rounding_mode="floor")
            )
            physical = page_ids * self.cmp_page_size + block_ids % self.cmp_page_size
            torch.index_select(cmp_flat, 0, physical, out=key_buffer[:block_count])
            key_buffer_fp32[:block_count].copy_(key_buffer[:block_count])
            keys = key_buffer_fp32[:block_count]
            width = min(self.block_topk, block_count)
            for chunk_start in range(start, stop, rows_per_chunk):
                chunk_stop = min(chunk_start + rows_per_chunk, stop)
                query = q_index_fp32[chunk_start:chunk_stop]
                query_positions = md.positions[chunk_start:chunk_stop].to(torch.long)
                visible = torch.div(query_positions + 1, self.ratio, rounding_mode="floor")
                visible.clamp_max_(block_count)
                chunk_rows = chunk_stop - chunk_start
                head_tile = head_scores[:chunk_rows, :, :block_count]
                scores = logits[:chunk_rows, :block_count]
                torch.matmul(query, keys.transpose(0, 1), out=head_tile)
                torch.relu_(head_tile)
                torch.sum(head_tile, dim=1, out=scores)
                scores.mul_(self.index_head_dim**-0.5)
                scores.masked_fill_(
                    block_ids.unsqueeze(0) >= visible.unsqueeze(1), -float("inf")
                )
                ordered = torch.argsort(scores, dim=-1, descending=True, stable=True)
                chosen = ordered[:, :width]
                take = torch.minimum(
                    visible, torch.full_like(visible, self.block_topk, dtype=torch.long)
                )
                chosen_rank = torch.arange(width, dtype=torch.long, device=index.q.device)
                chosen_valid = chosen_rank.unsqueeze(0) < take.unsqueeze(1)
                expanded = (
                    chosen.clamp_min(0).unsqueeze(-1) * self.ratio
                    + token_offsets.view(1, 1, -1)
                ).reshape(chunk_stop - chunk_start, width * self.ratio)
                expanded_valid = chosen_valid.unsqueeze(-1).expand_as(
                    expanded.view(chunk_stop - chunk_start, width, self.ratio)
                ).reshape_as(expanded)
                expanded = torch.where(expanded_valid, expanded.to(torch.int32), -1)
                packed = torch.full(
                    (chunk_stop - chunk_start, self.select_width),
                    -1,
                    dtype=torch.int32,
                    device=index.q.device,
                )
                packed[:, : width * self.ratio] = expanded

                visible_tokens = query_positions + 1
                tail_start = torch.div(
                    visible_tokens, self.ratio, rounding_mode="floor"
                ) * self.ratio
                tail_length = visible_tokens - tail_start
                # A partial tail contains at most ratio-1 tokens.  Excluding the impossible
                # ratio-th offset avoids clamping it onto the final valid column and
                # overwriting that token through a duplicate scatter index.
                tail_offsets = token_offsets[:-1]
                tail_values = (tail_start.unsqueeze(1) + tail_offsets).to(torch.int32)
                tail_columns = take.unsqueeze(1) * self.ratio + tail_offsets
                tail_valid = tail_offsets.unsqueeze(0) < tail_length.unsqueeze(1)
                old_tail = packed.gather(1, tail_columns)
                packed.scatter_(1, tail_columns, torch.where(tail_valid, tail_values, old_tail))
                row_ids = torch.arange(
                    chunk_start, chunk_stop, dtype=torch.long, device=index.q.device
                )
                indices.index_copy_(0, row_ids, packed)
        return indices

    def _attend_torch(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_req: torch.Tensor,
    ) -> torch.Tensor:
        """Exact selected-row paged GQA attention using FP32 Torch math."""
        if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
            raise ValueError("QSA reference attention received invalid Q/K/V shapes")
        if k_cache.shape[2] <= 0 or q.shape[1] % k_cache.shape[2]:
            raise ValueError("QSA reference attention requires integral GQA groups")
        if q.shape[2] != k_cache.shape[3] or indices.shape[0] != q.shape[0]:
            raise ValueError("QSA reference attention geometry does not match selection")
        if block_table.ndim != 2 or token_to_req.shape != (q.shape[0],):
            raise ValueError("QSA reference attention metadata has invalid shapes")
        if indices.shape[1] <= 0:
            raise ValueError("QSA reference attention requires positive selection width")
        logical = indices.to(torch.long)
        safe_logical = logical.clamp_min(0)
        pages = torch.div(safe_logical, self.page_size, rounding_mode="floor")
        offsets = safe_logical % self.page_size
        request = token_to_req.to(torch.long)
        if bool((request < 0).any()) or bool((request >= block_table.shape[0]).any()):
            raise ValueError("QSA reference attention request indices are out of bounds")
        if bool((pages >= block_table.shape[1]).any()):
            raise ValueError("QSA reference attention selected page exceeds page table")
        physical_pages = block_table[request.unsqueeze(1), pages]
        physical = physical_pages * self.page_size + offsets
        valid = logical >= 0
        total_rows = k_cache.shape[0] * self.page_size
        if bool((valid & ((physical_pages < 0) | (physical < 0) | (physical >= total_rows))).any()):
            raise ValueError("QSA reference attention selected physical row is out of bounds")
        selected_rows = torch.where(valid, physical, torch.zeros_like(physical)).to(torch.int32)
        counts = valid.sum(dim=1, dtype=torch.int32)
        from freetoken.kernel.triton.qsa_legacy import qsa_sparse_gqa

        return qsa_sparse_gqa(
            q,
            k_cache.view(-1, k_cache.shape[-2], k_cache.shape[-1]),
            v_cache.view(-1, v_cache.shape[-2], v_cache.shape[-1]),
            selected_rows,
            counts,
            q.shape[-1] ** -0.5,
        )

    def _select(self, index, md: QSASparseMetadata, slot: int) -> torch.Tensor:
        """Score complete visible blocks, take the top-k, expand them to token indices."""
        if self._vectorized_reference:
            return self._select_vectorized_reference(index, md, slot)
        if self._torch_reference:
            return self._select_torch(index, md, slot)
        from freetoken.kernel.triton.qsa import (
            expand_qsa_block_indices,
            qsa_index_norm_rope,
            qsa_mqa_paged,
        )

        rows = index.q.shape[0]
        positions = md.positions
        q_index = self._scratch(
            "q_index", rows, self.index_heads, self.index_head_dim, dtype=self.dtype
        )
        qsa_index_norm_rope(
            index.q.view(-1, self.index_head_dim),
            positions,
            self._index_rope_cache(),
            index.q_norm_weight,
            index.eps,
            q_index.view(-1, self.index_head_dim),
            heads=self.index_heads,
        )
        cmp_pages = self._cmp_pages(slot)
        columns = md.block_table.shape[1] * self.cmp_page_size
        indices = self._scratch("indices", rows, self.select_width, dtype=torch.int32)
        rows_per_chunk = qsa_score_chunk_rows(rows, columns)
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            chunk = slice(start, end)
            logits = self._scratch("logits", end - start, columns, dtype=torch.float32)
            visible = self._scratch("visible", end - start, dtype=torch.int32)
            qsa_mqa_paged(
                q_index[chunk],
                cmp_pages,
                md.block_table,
                md.token_to_req[chunk],
                positions[chunk],
                md.seq_lens,
                self.ratio,
                logits,
                visible,
            )
            blocks = self._scratch("blocks", end - start, self.block_topk, dtype=torch.int32)
            self._top_blocks(logits, visible, blocks)
            expand_qsa_block_indices(
                blocks,
                positions[chunk],
                md.seq_lens,
                md.token_to_req[chunk],
                self.ratio,
                self.token_topk,
                indices[chunk],
            )
        return indices

    def _top_blocks(
        self,
        logits: torch.Tensor,
        visible: torch.Tensor,
        blocks: torch.Tensor,
    ) -> None:
        """Top ``block_topk`` complete blocks per row, row-relative, -1 padded."""
        assert blocks.shape == (logits.shape[0], self.block_topk), (
            f"qsa block top-k output must be [rows, {self.block_topk}], got {tuple(blocks.shape)}"
        )
        if self._block_topk_kernel is not None:
            scratch_width = self._topk_scratch_width(logits.shape[1])
            scratch = (
                self._scratch("topk_scratch", logits.shape[0], scratch_width, dtype=torch.int32)
                if scratch_width
                else None
            )
            self._block_topk_kernel(logits, visible, blocks, scratch)
            return
        # The score kernel only writes columns below visible_blocks; mask the rest so a
        # stale row cannot win a slot. Real block scores are relu sums, never -inf.
        columns = logits.shape[1]
        column = torch.arange(columns, dtype=torch.int32, device=logits.device)
        logits.masked_fill_(column.unsqueeze(0) >= visible.unsqueeze(1), -float("inf"))
        width = min(self.block_topk, columns)
        values, chosen = torch.topk(logits, width, dim=-1)
        blocks[:, :width] = torch.where(values > -float("inf"), chosen.to(torch.int32), -1)
        if width < self.block_topk:
            blocks[:, width:] = -1

    def _topk_scratch_width(self, columns: int) -> int:
        """int32 columns per row the block top-k wants as scratch, 0 when it wants none."""
        if self._block_topk_kernel is None:
            return 0
        from freetoken.kernel.triton.qsa import qsa_block_topk_scratch_width

        return qsa_block_topk_scratch_width(columns, self.block_topk)

    # ----- scratch ------------------------------------------------------------------------
    def _scratch(self, name: str, rows: int, *shape: int, dtype: torch.dtype) -> torch.Tensor:
        """A per-forward transient: the static decode buffer when it is wide enough (so a
        captured graph keeps one address), otherwise a fresh allocation."""
        buffer = self._graph.get(name)
        if buffer is not None and rows <= buffer.shape[0] and buffer.shape[1:] == shape:
            return buffer[:rows]
        return torch.empty((rows, *shape), dtype=dtype, device=self.device)

    # ----- CUDA graph (decode) --------------------------------------------------------------
    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        if self._torch_reference:
            raise NotImplementedError("QSA torch-fp32-reference is eager-only")
        self.capture_bs = sorted(bs_list)
        max_bs = max(bs_list)
        width = get_global_ctx().page_table.shape[1]
        pages = -(-width // self.page_size)
        columns = pages * self.cmp_page_size
        chunk = qsa_score_chunk_rows(max_bs, columns)
        topk_scratch = self._topk_scratch_width(columns)

        def empty(*shape: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device=self.device)

        self._graph = {
            "block_table": torch.zeros((max_bs, pages), dtype=torch.int32, device=self.device),
            "kvlen": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "table_idx": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "token_to_req": torch.arange(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens": torch.arange(max_bs + 1, dtype=torch.int32, device=self.device),
            "logits": empty(chunk, columns, dtype=torch.float32),
            "visible": empty(max_bs, dtype=torch.int32),
            "blocks": empty(max_bs, self.block_topk, dtype=torch.int32),
            "indices": empty(max_bs, self.select_width, dtype=torch.int32),
            "pooled": empty(max_bs, self.index_head_dim, dtype=self.dtype),
            "first_pos": empty(max_bs, dtype=torch.int32),
            "q_index": empty(max_bs, self.index_heads, self.index_head_dim, dtype=self.dtype),
        }
        if topk_scratch:
            self._graph["topk_scratch"] = empty(chunk, topk_scratch, dtype=torch.int32)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.prepare_metadata(batch)
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        bs = batch.size
        dummy = torch.full(
            (bs,), batch.padded_reqs[0].table_idx, dtype=torch.int64, device=self.device
        )
        self._stage_decode(md, bs, dummy)

    def prepare_for_replay(self, batch: Batch) -> None:
        md = batch.attn_metadata
        assert isinstance(md, QSASparseMetadata)
        assert batch.active_table_idx is not None, "decode batch is missing its page-table rows"
        self._stage_decode(md, batch.padded_size, batch.active_table_idx.to(torch.int64))

    def reset_capture(self) -> None:
        super().reset_capture()
        self._graph = {}


__all__ = [
    "QSA_SELECTION_PATHS",
    "QSAPhase",
    "QSAPhaseEvent",
    "QSAPhaseMetadata",
    "QSAPhaseObserver",
    "QSARequestSpan",
    "QSASelectionPath",
    "QSASparseAttnBackend",
    "QSASparseMetadata",
    "resolve_qsa_selection_path",
]
