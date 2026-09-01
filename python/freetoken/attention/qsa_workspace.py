"""Pure H0 shape and byte accounting for the merged QSA backend.

This module performs no Torch/CUDA allocation and mirrors concrete QSA buffer shapes.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

MAX_QSA_WORKSPACE_BYTES = (1 << 63) - 1
_I32 = 4
_I64 = 8
_BOOL = 1
_FP32 = 4
_QSA_DTYPE = 2
# Keep this in lockstep with qsa_sparse._LOGITS_WORKSPACE_BYTES.  The score kernel's
# row tile is a bounded transient; it is not the complete [token_rows, columns] matrix.
QSA_LOGITS_WORKSPACE_BYTES = 128 << 20
QSA_WORKSPACE_CATEGORIES = ("score", "top_k", "expand_gather", "attention", "state")
QSA_SELECTION_PATHS = (
    "auto",
    "torch-fp32-reference",
    "torch-fp32-vectorized-reference",
)


class QSAWorkspaceError(ValueError):
    """Base class for QSA accounting errors."""


class QSAWorkspaceInputError(QSAWorkspaceError):
    """Malformed shape, phase, backend, dtype, or overflowing arithmetic input."""


class QSAWorkspaceCapacityError(QSAWorkspaceError):
    """Supplied capacity is below the planned high-water mark."""

    def __init__(self, required_bytes: int, capacity_bytes: int, telemetry: QSAWorkspaceTelemetry):
        self.required_bytes = required_bytes
        self.capacity_bytes = capacity_bytes
        self.deficit_bytes = required_bytes - capacity_bytes
        self.telemetry = telemetry
        super().__init__(
            f"QSA workspace capacity is insufficient: required={required_bytes} bytes, "
            f"capacity={capacity_bytes} bytes, deficit={self.deficit_bytes} bytes"
        )


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise QSAWorkspaceInputError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise QSAWorkspaceInputError(f"{name} must be an integer") from exc
    if result < 0:
        raise QSAWorkspaceInputError(f"{name} must be non-negative, got {result}")
    if result > MAX_QSA_WORKSPACE_BYTES:
        raise QSAWorkspaceInputError(f"{name} exceeds the supported integer range")
    return result


def _positive(value: Any, name: str) -> int:
    result = _int(value, name)
    if result == 0:
        raise QSAWorkspaceInputError(f"{name} must be positive")
    return result


def _mul(*values: int, label: str) -> int:
    result = 1
    for value in values:
        if value < 0 or (value and result > MAX_QSA_WORKSPACE_BYTES // value):
            raise QSAWorkspaceInputError(f"{label} integer overflow")
        result *= value
    return result


def _add(*values: int, label: str) -> int:
    result = 0
    for value in values:
        if value < 0 or result > MAX_QSA_WORKSPACE_BYTES - value:
            raise QSAWorkspaceInputError(f"{label} integer overflow")
        result += value
    return result


def _ceil(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _pow2(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def qsa_score_chunk_rows(rows: int, columns: int) -> int:
    """Return the row chunk used by the runtime's bounded score tile."""
    rows = _positive(rows, "rows")
    columns = _positive(columns, "columns")
    return min(
        rows,
        max(1, QSA_LOGITS_WORKSPACE_BYTES // _mul(columns, _FP32, label="score tile")),
    )


def qsa_vectorized_score_chunk_rows(rows: int, columns: int, index_heads: int = 1) -> int:
    """Return a row chunk bounded for per-head FP32 scores plus stable-sort indices."""
    rows = _positive(rows, "rows")
    columns = _positive(columns, "columns")
    index_heads = _positive(index_heads, "index_heads")
    # The device tile retains per-head scores until their reduction, the reduced score logits,
    # and stable-sort indices.  Keep all three within the existing bounded score budget.
    per_row = _mul(
        columns,
        index_heads * _FP32 + _FP32 + _I64,
        label="vectorized score tile",
    )
    return min(rows, max(1, QSA_LOGITS_WORKSPACE_BYTES // per_row))


def qsa_topk_scratch_width(columns: int, top_k: int, backend: str = "triton") -> int:
    """Return the runtime Triton top-k candidate width, or zero for one-program selection."""
    columns = _positive(columns, "columns")
    top_k = _positive(top_k, "top_k")
    if backend not in {"triton", "torch"}:
        raise QSAWorkspaceInputError("topk_backend must be 'triton' or 'torch'")
    if backend == "torch" or columns <= 4096:
        return 0
    max_splits = 8192 // _pow2(top_k)
    if max_splits < 2:
        return 0
    chunk = max(4096, _pow2(_ceil(columns, max_splits)))
    if chunk > 8192:
        return 0
    splits = _ceil(columns, chunk)
    if splits < 2 or _mul(splits, top_k, label="top-k candidates") >= columns:
        return 0
    return _mul(2, splits, top_k, label="top-k scratch")


def qsa_attention_split_plan(
    rows: int, kv_heads: int, query_heads: int, width: int
) -> tuple[int, int, int]:
    """Return the runtime ``(block_n, target_splits, num_splits)`` attention policy."""
    rows = _positive(rows, "rows")
    kv_heads = _positive(kv_heads, "kv_heads")
    query_heads = _positive(query_heads, "query_heads")
    width = _positive(width, "width")
    if query_heads % kv_heads:
        raise QSAWorkspaceInputError("query_heads must be divisible by kv_heads")
    block_m = _pow2(query_heads // kv_heads)
    base = _mul(rows, kv_heads, label="attention programs")
    if base <= (8 if block_m <= 8 else 4):
        block_n, target = 16, 64
    elif base < 32:
        block_n, target = 16, 32
    elif base <= 256:
        block_n, target = 64, 8
    elif base <= 512:
        block_n, target = 64, 4
    else:
        block_n, target = 64, 1
    tiles = _ceil(width, block_n)
    return block_n, target, min(1 << (tiles.bit_length() - 1), target)


def qsa_attention_split_count(rows: int, kv_heads: int, query_heads: int, width: int) -> int:
    """Return the number of split-attention partial buffers the runtime will allocate."""
    return qsa_attention_split_plan(rows, kv_heads, query_heads, width)[2]


@dataclass(frozen=True)
class QSAWorkspaceInputs:
    context_tokens: int
    token_rows: int
    page_table_width: int
    page_size: int
    index_heads: int
    query_heads: int
    kv_heads: int
    head_dim: int
    index_head_dim: int
    top_k: int
    compression_ratio: int
    num_index_layers: int
    num_req_slots: int
    ring_capacity: int
    num_pages: int
    max_position: int
    rotary_dim: int
    # ``token_rows`` is the ragged query count, while ``batch_size`` is the request count
    # used by the page-table and metadata buffers.  They are intentionally independent.
    batch_size: int = 1
    # CUDA graph buffers are allocated for the largest capture batch, not the active batch.
    # Eager plans ignore this value; capture plans default it to the active batch.
    capture_max_batch_size: int | None = None
    phase: str = "eager"
    topk_backend: str = "triton"
    qsa_selection_path: str = "auto"
    dtype_bytes: int = _QSA_DTYPE

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name not in {
                "phase",
                "topk_backend",
                "qsa_selection_path",
                "capture_max_batch_size",
            }:
                object.__setattr__(self, name, _positive(getattr(self, name), name))
        capture_max = (
            self.batch_size
            if self.capture_max_batch_size is None
            else _positive(self.capture_max_batch_size, "capture_max_batch_size")
        )
        if capture_max < self.batch_size:
            raise QSAWorkspaceInputError("capture_max_batch_size must cover the active batch_size")
        object.__setattr__(self, "capture_max_batch_size", capture_max)
        if self.dtype_bytes != _QSA_DTYPE:
            raise QSAWorkspaceInputError("QSA currently requires a two-byte dtype")
        if self.phase not in {"eager", "capture"}:
            raise QSAWorkspaceInputError("phase must be 'eager' or 'capture'")
        if self.topk_backend not in {"triton", "torch"}:
            raise QSAWorkspaceInputError("topk_backend must be 'triton' or 'torch'")
        if self.qsa_selection_path not in QSA_SELECTION_PATHS:
            raise QSAWorkspaceInputError(
                "qsa_selection_path must be 'auto', 'torch-fp32-reference', or "
                "'torch-fp32-vectorized-reference'"
            )
        if self.phase == "capture" and self.topk_backend == "torch":
            raise QSAWorkspaceInputError(
                "capture phase requires Triton top-k; the torch.topk fallback is eager-only"
            )
        if self.phase == "capture" and self.qsa_selection_path == "torch-fp32-vectorized-reference":
            raise QSAWorkspaceInputError(
                "torch-fp32-vectorized-reference is eager-only and cannot be captured"
            )
        if self.query_heads % self.kv_heads:
            raise QSAWorkspaceInputError("query_heads must be divisible by kv_heads")
        if self.top_k % self.compression_ratio:
            raise QSAWorkspaceInputError("top_k must be divisible by compression_ratio")
        if self.page_size % self.compression_ratio:
            raise QSAWorkspaceInputError("page_size must be divisible by compression_ratio")
        if self.rotary_dim > self.index_head_dim or self.rotary_dim % 2:
            raise QSAWorkspaceInputError("rotary_dim must be even and fit index_head_dim")
        if self.batch_size > self.num_req_slots:
            raise QSAWorkspaceInputError("batch_size exceeds num_req_slots")
        if self.capture_max_batch_size > self.num_req_slots:
            raise QSAWorkspaceInputError("capture_max_batch_size exceeds num_req_slots")
        if self.context_tokens // self.compression_ratio > self.score_columns:
            raise QSAWorkspaceInputError("page-table shape is smaller than the complete context")

    @property
    def page_count(self) -> int:
        """Number of page-table rows after converting raw token-slot width to pages."""
        return _ceil(self.page_table_width, self.page_size)

    @property
    def score_columns(self) -> int:
        return _mul(
            self.page_count,
            self.page_size // self.compression_ratio,
            label="score columns",
        )

    @property
    def chunk_rows(self) -> int:
        if self.qsa_selection_path == "torch-fp32-vectorized-reference":
            return qsa_vectorized_score_chunk_rows(
                self.token_rows, self.score_columns, self.index_heads
            )
        return qsa_score_chunk_rows(self.token_rows, self.score_columns)

    @property
    def capture_chunk_rows(self) -> int:
        """Rows reserved by ``init_capture_graph`` for its bounded logits tile."""
        return qsa_score_chunk_rows(self.capture_max_batch_size, self.score_columns)

    @property
    def request_key_rows(self) -> int:
        """Maximum complete compressed keys in one vectorized request gather."""
        if self.qsa_selection_path != "torch-fp32-vectorized-reference":
            return 0
        return min(self.context_tokens // self.compression_ratio, self.score_columns)

    @property
    def selection_width(self) -> int:
        return self.top_k + self.compression_ratio - 1

    @property
    def attention_splits(self) -> int:
        return qsa_attention_split_count(
            self.token_rows, self.kv_heads, self.query_heads, self.selection_width
        )


@dataclass(frozen=True)
class QSAWorkspaceCategory:
    name: str
    bytes: int
    components: Mapping[str, int]
    shapes: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if self.name not in QSA_WORKSPACE_CATEGORIES:
            raise QSAWorkspaceInputError(f"unknown QSA workspace category {self.name!r}")
        if set(self.components) != set(self.shapes):
            raise QSAWorkspaceInputError(f"{self.name} components and shapes disagree")
        normalized_components = {
            name: _int(value, f"{self.name}.{name} bytes")
            for name, value in self.components.items()
        }
        if _int(self.bytes, f"{self.name} bytes") != _add(
            *normalized_components.values(), label=f"{self.name} category"
        ):
            raise QSAWorkspaceInputError(f"{self.name} category total does not match components")
        normalized_shapes: dict[str, tuple[int, ...]] = {}
        for component, shape in self.shapes.items():
            if not isinstance(shape, tuple):
                raise QSAWorkspaceInputError(f"{self.name}.{component} shape must be a tuple")
            normalized_shapes[component] = tuple(
                _int(dim, f"{self.name}.{component} shape") for dim in shape
            )
        object.__setattr__(self, "bytes", _int(self.bytes, f"{self.name} bytes"))
        object.__setattr__(self, "components", MappingProxyType(normalized_components))
        object.__setattr__(self, "shapes", MappingProxyType(normalized_shapes))


@dataclass(frozen=True)
class QSAWorkspaceInventory:
    categories: Mapping[str, QSAWorkspaceCategory]

    def __post_init__(self) -> None:
        keys = set(self.categories)
        missing = set(QSA_WORKSPACE_CATEGORIES) - keys
        unknown = keys - set(QSA_WORKSPACE_CATEGORIES)
        if missing:
            raise QSAWorkspaceInputError(f"missing QSA workspace categories: {sorted(missing)}")
        if unknown:
            raise QSAWorkspaceInputError(f"unknown QSA workspace categories: {sorted(unknown)}")
        for name, category in self.categories.items():
            if not isinstance(category, QSAWorkspaceCategory) or category.name != name:
                raise QSAWorkspaceInputError(f"malformed QSA workspace category {name!r}")
        object.__setattr__(self, "categories", MappingProxyType(dict(self.categories)))

    def __getitem__(self, name: str) -> QSAWorkspaceCategory:
        return self.categories[name]

    def __iter__(self):
        return iter(QSA_WORKSPACE_CATEGORIES)

    def values(self):
        return tuple(self.categories[name] for name in QSA_WORKSPACE_CATEGORIES)


@dataclass(frozen=True)
class QSAWorkspaceTelemetry:
    status: str
    required_bytes: int
    capacity_bytes: int | None
    headroom_bytes: int | None
    persistent_bytes: int
    capture_resident_bytes: int
    eager_transient_peak_bytes: int
    categories: Mapping[str, int]
    shapes: Mapping[str, Mapping[str, tuple[int, ...]]]
    request: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "categories", MappingProxyType(dict(self.categories)))
        object.__setattr__(
            self,
            "shapes",
            MappingProxyType(
                {name: MappingProxyType(dict(value)) for name, value in self.shapes.items()}
            ),
        )
        object.__setattr__(self, "request", MappingProxyType(dict(self.request)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "required_bytes": self.required_bytes,
            "capacity_bytes": self.capacity_bytes,
            "headroom_bytes": self.headroom_bytes,
            "persistent_bytes": self.persistent_bytes,
            "capture_resident_bytes": self.capture_resident_bytes,
            "eager_transient_peak_bytes": self.eager_transient_peak_bytes,
            "categories": dict(self.categories),
            "shapes": {name: dict(value) for name, value in self.shapes.items()},
            "request": dict(self.request),
        }

    to_dict = as_dict


@dataclass(frozen=True)
class QSAWorkspacePlan:
    request: QSAWorkspaceInputs
    inventory: QSAWorkspaceInventory
    required_bytes: int
    persistent_bytes: int
    capture_resident_bytes: int
    eager_transient_peak_bytes: int
    telemetry: QSAWorkspaceTelemetry

    def validate_capacity(self, capacity_bytes: int) -> QSAWorkspaceTelemetry:
        capacity = _int(capacity_bytes, "capacity_bytes")
        status = "ready" if capacity >= self.required_bytes else "insufficient-capacity"
        telemetry = _telemetry(self, status, capacity)
        if status != "ready":
            raise QSAWorkspaceCapacityError(self.required_bytes, capacity, telemetry)
        return telemetry

    def as_dict(self) -> dict[str, Any]:
        return self.telemetry.as_dict()

    to_dict = as_dict


def _telemetry(plan: QSAWorkspacePlan, status: str, capacity: int | None) -> QSAWorkspaceTelemetry:
    return QSAWorkspaceTelemetry(
        status,
        plan.required_bytes,
        capacity,
        None if capacity is None else capacity - plan.required_bytes,
        plan.persistent_bytes,
        plan.capture_resident_bytes,
        plan.eager_transient_peak_bytes,
        {name: plan.inventory[name].bytes for name in QSA_WORKSPACE_CATEGORIES},
        {name: plan.inventory[name].shapes for name in QSA_WORKSPACE_CATEGORIES},
        {name: getattr(plan.request, name) for name in plan.request.__dataclass_fields__},
    )


def _category(
    name: str, components: Mapping[str, int], shapes: Mapping[str, tuple[int, ...]]
) -> QSAWorkspaceCategory:
    return QSAWorkspaceCategory(
        name, _add(*components.values(), label=f"{name} category"), components, shapes
    )


def calculate_qsa_workspace(request: QSAWorkspaceInputs) -> QSAWorkspacePlan:
    """Calculate concrete QSA category shapes and the lifetime-aware high-water mark."""
    if not isinstance(request, QSAWorkspaceInputs):
        raise QSAWorkspaceInputError("request must be QSAWorkspaceInputs")
    rows, columns, chunk, ratio = (
        request.token_rows,
        request.score_columns,
        request.chunk_rows,
        request.compression_ratio,
    )
    block_top_k = request.top_k // ratio
    page_count = request.page_count

    # These are device-resident metadata tensors in QSASparseMetadata.  The pinned CPU
    # copies (qo_indptr_cpu, kv_len_cpu, and eager table_idx) are deliberately reported only
    # in documentation: this planner's capacity is a device-workspace capacity, while their
    # device mirrors are accounted here with their actual active batch shape.
    metadata_components = {
        "last_indices": _mul(request.batch_size, _I32, label="last indices"),
        "token_to_req": _mul(rows, _I32, label="token-to-request"),
        "cu_seqlens": _mul(request.batch_size + 1, _I32, label="cu seqlens"),
        "seq_lens": _mul(request.batch_size, _I32, label="sequence lengths"),
        "ring_slots": _mul(request.batch_size, _I32, label="ring slots"),
        "block_table": _mul(request.batch_size, page_count, _I32, label="block table"),
    }
    metadata_shapes = {
        "last_indices": (request.batch_size,),
        "token_to_req": (rows,),
        "cu_seqlens": (request.batch_size + 1,),
        "seq_lens": (request.batch_size,),
        "ring_slots": (request.batch_size,),
        "block_table": (request.batch_size, page_count),
    }
    score = _category(
        "score",
        {
            "q_index": _mul(
                rows, request.index_heads, request.index_head_dim, _QSA_DTYPE, label="q_index"
            ),
            "logits": _mul(chunk, columns, _FP32, label="score logits"),
            "visible": _mul(chunk, _I32, label="visible blocks"),
            "q_index_fp32": (
                _mul(rows, request.index_heads, request.index_head_dim, _FP32, label="FP32 q_index")
                if request.qsa_selection_path == "torch-fp32-vectorized-reference"
                else 0
            ),
            "request_keys": (
                _mul(
                    request.request_key_rows,
                    request.index_head_dim,
                    _QSA_DTYPE,
                    label="request key gather",
                )
                if request.qsa_selection_path == "torch-fp32-vectorized-reference"
                else 0
            ),
            "request_keys_fp32": (
                _mul(
                    request.request_key_rows,
                    request.index_head_dim,
                    _FP32,
                    label="FP32 request key gather",
                )
                if request.qsa_selection_path == "torch-fp32-vectorized-reference"
                else 0
            ),
            "vector_score_heads": (
                _mul(
                    chunk,
                    request.index_heads,
                    request.request_key_rows,
                    _FP32,
                    label="vectorized per-head score tile",
                )
                if request.qsa_selection_path == "torch-fp32-vectorized-reference"
                else 0
            ),
            **metadata_components,
        },
        {
            "q_index": (rows, request.index_heads, request.index_head_dim),
            "logits": (chunk, columns),
            "visible": (chunk,),
            "q_index_fp32": (
                (rows, request.index_heads, request.index_head_dim)
                if request.qsa_selection_path == "torch-fp32-vectorized-reference"
                else (0, 0, 0)
            ),
            "request_keys": (
                (request.request_key_rows, request.index_head_dim)
                if request.qsa_selection_path == "torch-fp32-vectorized-reference"
                else (0, 0)
            ),
            "request_keys_fp32": (
                (request.request_key_rows, request.index_head_dim)
                if request.qsa_selection_path == "torch-fp32-vectorized-reference"
                else (0, 0)
            ),
            "vector_score_heads": (
                (chunk, request.index_heads, request.request_key_rows)
                if request.qsa_selection_path == "torch-fp32-vectorized-reference"
                else (0, 0, 0)
            ),
            **metadata_shapes,
        },
    )
    scratch = qsa_topk_scratch_width(columns, block_top_k, request.topk_backend)
    # The Torch fallback has several Python-visible temporaries that the Triton path avoids.
    # Keep them in the inventory so an eager plan is conservative.  PyTorch's allocator
    # fragmentation and opaque kernel-internal topk workspace are not observable from this
    # Torch-free planner and therefore cannot be represented as exact components.
    torch_width = min(block_top_k, columns)
    torch_fallback = request.topk_backend == "torch"
    top_k = _category(
        "top_k",
        {
            "blocks": _mul(chunk, block_top_k, _I32, label="top-k blocks"),
            "candidate_scratch": _mul(chunk, scratch, _I32, label="top-k scratch"),
            "torch_columns": (
                _mul(columns, _I32, label="torch top-k columns") if torch_fallback else 0
            ),
            "torch_visibility_mask": (
                _mul(chunk, columns, _BOOL, label="torch top-k visibility mask")
                if torch_fallback
                else 0
            ),
            "torch_values": (
                _mul(chunk, torch_width, _FP32, label="torch top-k values") if torch_fallback else 0
            ),
            "torch_chosen": (
                _mul(chunk, torch_width, _I64, label="torch top-k chosen") if torch_fallback else 0
            ),
            "torch_valid": (
                _mul(chunk, torch_width, _BOOL, label="torch top-k valid mask")
                if torch_fallback
                else 0
            ),
            "torch_chosen_i32": (
                _mul(chunk, torch_width, _I32, label="torch top-k chosen cast")
                if torch_fallback
                else 0
            ),
            "torch_where": (
                _mul(chunk, torch_width, _I32, label="torch top-k where output")
                if torch_fallback
                else 0
            ),
            "vector_sort_indices": (
                _mul(chunk, columns, _I64, label="vectorized stable sort indices")
                if request.qsa_selection_path == "torch-fp32-vectorized-reference"
                else 0
            ),
        },
        {
            "blocks": (chunk, block_top_k),
            "candidate_scratch": (chunk, scratch),
            "torch_columns": (columns,) if torch_fallback else (0,),
            "torch_visibility_mask": (chunk, columns) if torch_fallback else (0, 0),
            "torch_values": (chunk, torch_width) if torch_fallback else (0, 0),
            "torch_chosen": (chunk, torch_width) if torch_fallback else (0, 0),
            "torch_valid": (chunk, torch_width) if torch_fallback else (0, 0),
            "torch_chosen_i32": (chunk, torch_width) if torch_fallback else (0, 0),
            "torch_where": (chunk, torch_width) if torch_fallback else (0, 0),
            "vector_sort_indices": (
                (chunk, columns)
                if request.qsa_selection_path == "torch-fp32-vectorized-reference"
                else (0, 0)
            ),
        },
    )
    expand = _category(
        "expand_gather",
        {"indices": _mul(rows, request.selection_width, _I32, label="expanded indices")},
        {"indices": (rows, request.selection_width)},
    )
    splits = request.attention_splits
    attention = _category(
        "attention",
        {
            "output": _mul(
                rows, request.query_heads, request.head_dim, _QSA_DTYPE, label="attention output"
            ),
            "partial_output": _mul(
                splits, rows, request.query_heads, request.head_dim, _FP32, label="partial output"
            )
            if splits > 1
            else 0,
            "partial_lse": _mul(splits, rows, request.query_heads, _FP32, label="partial lse")
            if splits > 1
            else 0,
        },
        {
            "output": (rows, request.query_heads, request.head_dim),
            "partial_output": (splits, rows, request.query_heads, request.head_dim),
            "partial_lse": (splits, rows, request.query_heads),
        },
    )
    state_rows = _add(
        _mul(request.num_pages, request.page_size, label="state rows") // ratio,
        request.num_req_slots,
        label="state rows",
    )
    state = _category(
        "state",
        {
            "compressed_slab": _mul(
                request.num_index_layers,
                state_rows,
                request.index_head_dim,
                _QSA_DTYPE,
                label="compressed slab",
            ),
            "pending_ring": _mul(
                request.num_req_slots,
                request.num_index_layers,
                request.ring_capacity,
                request.index_head_dim,
                _QSA_DTYPE,
                label="pending ring",
            ),
            "index_rope": _mul(request.max_position, request.rotary_dim, _FP32, label="index RoPE"),
            "pooled": _mul(rows, request.index_head_dim, _QSA_DTYPE, label="pooled rows"),
            "first_positions": _mul(rows, _I32, label="first positions"),
            "cmp_rows": _mul(rows, _I32, label="compressed row destinations"),
            "ring_rows": _mul(rows, _I32, label="pending ring rows"),
        },
        {
            "compressed_slab": (request.num_index_layers, state_rows, request.index_head_dim),
            "pending_ring": (
                request.num_req_slots,
                request.num_index_layers,
                request.ring_capacity,
                request.index_head_dim,
            ),
            "index_rope": (request.max_position, request.rotary_dim),
            "pooled": (rows, request.index_head_dim),
            "first_positions": (rows,),
            "cmp_rows": (rows,),
            "ring_rows": (rows,),
        },
    )
    inventory = QSAWorkspaceInventory(
        {
            "score": score,
            "top_k": top_k,
            "expand_gather": expand,
            "attention": attention,
            "state": state,
        }
    )
    persistent = _add(
        state.components["compressed_slab"],
        state.components["pending_ring"],
        state.components["index_rope"],
        label="persistent state",
    )
    scatter_rows = _add(
        state.components["cmp_rows"],
        state.components["ring_rows"],
        label="per-forward scatter rows",
    )

    # Eager metadata remains live while index compression, selection, and sparse attention
    # execute.  Selection retains q_index and the complete indices output while replacing only
    # the per-chunk score/top-k buffers.  Attention retains indices but not q_index or score
    # buffers; compression's pooled rows are released before selection starts.
    metadata_bytes = _add(*metadata_components.values(), label="eager metadata")
    eager_state = _add(
        metadata_bytes,
        state.components["pooled"],
        state.components["first_positions"],
        scatter_rows,
        label="eager index-update phase",
    )
    eager_select = _add(
        metadata_bytes,
        scatter_rows,
        score.components["q_index"],
        score.components["q_index_fp32"],
        score.components["request_keys"],
        score.components["request_keys_fp32"],
        score.components["vector_score_heads"],
        expand.components["indices"],
        score.components["logits"],
        score.components["visible"],
        top_k.bytes,
        label="eager selection phase",
    )
    eager_attention = _add(
        metadata_bytes,
        scatter_rows,
        expand.components["indices"],
        attention.bytes,
        label="eager attention phase",
    )
    eager_peak = max(eager_state, eager_select, eager_attention)

    # CUDA graph setup keeps every entry of QSASparseAttnBackend._graph alive at once.  The
    # graph uses max_bs for all row buffers and its own bounded score chunk, even if the captured
    # batch is ragged.  qsa_sparse_paged_attention's output/partial tensors and last_indices are
    # graph-capture allocations outside _graph, so include their active-batch peak as well.
    capture_bs = request.capture_max_batch_size
    capture_chunk = request.capture_chunk_rows
    capture_scratch = qsa_topk_scratch_width(columns, block_top_k, request.topk_backend)
    graph_metadata = _add(
        _mul(capture_bs, page_count, _I32, label="capture block table"),
        _mul(capture_bs, _I32, label="capture kvlen"),
        _mul(capture_bs, _I32, label="capture table index"),
        _mul(capture_bs, _I32, label="capture token-to-request"),
        _mul(capture_bs + 1, _I32, label="capture cu seqlens"),
        label="capture graph metadata",
    )
    graph_buffers = _add(
        graph_metadata,
        _mul(capture_chunk, columns, _FP32, label="capture logits"),
        _mul(capture_bs, _I32, label="capture visible"),
        _mul(capture_bs, block_top_k, _I32, label="capture blocks"),
        _mul(capture_bs, request.selection_width, _I32, label="capture indices"),
        _mul(capture_bs, request.index_head_dim, _QSA_DTYPE, label="capture pooled"),
        _mul(capture_bs, _I32, label="capture first positions"),
        _mul(
            capture_bs,
            request.index_heads,
            request.index_head_dim,
            _QSA_DTYPE,
            label="capture q_index",
        ),
        _mul(capture_chunk, capture_scratch, _I32, label="capture top-k scratch"),
        label="capture graph buffers",
    )
    capture_dynamic = _add(
        _mul(request.batch_size, _I32, label="capture last indices"),
        scatter_rows,
        attention.bytes,
        label="capture dynamic attention",
    )
    capture_all = _add(graph_buffers, capture_dynamic, label="capture resident")
    capture = capture_all if request.phase == "capture" else 0
    peak = eager_peak
    required = _add(
        persistent,
        capture if request.phase == "capture" else peak,
        label="QSA high-water",
    )
    plan = QSAWorkspacePlan(request, inventory, required, persistent, capture, peak, None)  # type: ignore[arg-type]
    object.__setattr__(plan, "telemetry", _telemetry(plan, "unvalidated", None))
    return plan


def validate_qsa_workspace_capacity(
    request: QSAWorkspaceInputs, capacity_bytes: int
) -> QSAWorkspaceTelemetry:
    """Calculate and validate a plan before an allocation/launch."""
    return calculate_qsa_workspace(request).validate_capacity(capacity_bytes)


__all__ = [
    "MAX_QSA_WORKSPACE_BYTES",
    "QSA_LOGITS_WORKSPACE_BYTES",
    "QSA_SELECTION_PATHS",
    "QSA_WORKSPACE_CATEGORIES",
    "QSAWorkspaceCapacityError",
    "QSAWorkspaceCategory",
    "QSAWorkspaceError",
    "QSAWorkspaceInputError",
    "QSAWorkspaceInputs",
    "QSAWorkspaceInventory",
    "QSAWorkspacePlan",
    "QSAWorkspaceTelemetry",
    "calculate_qsa_workspace",
    "qsa_attention_split_count",
    "qsa_attention_split_plan",
    "qsa_score_chunk_rows",
    "qsa_topk_scratch_width",
    "qsa_vectorized_score_chunk_rows",
    "validate_qsa_workspace_capacity",
]
