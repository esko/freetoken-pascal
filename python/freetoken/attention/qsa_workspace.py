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
_FP32 = 4
_QSA_DTYPE = 2
_SCORE_BUDGET = 64 << 20
QSA_WORKSPACE_CATEGORIES = ("score", "top_k", "expand_gather", "attention", "state")


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


def _topk_scratch(columns: int, top_k: int, backend: str) -> int:
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


def _attention_splits(rows: int, kv_heads: int, query_heads: int, width: int) -> int:
    block_m = _pow2(query_heads // kv_heads)
    base = rows * kv_heads
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
    return min(1 << (tiles.bit_length() - 1), target)


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
    phase: str = "eager"
    topk_backend: str = "triton"
    dtype_bytes: int = _QSA_DTYPE

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name not in {"phase", "topk_backend"}:
                object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.dtype_bytes != _QSA_DTYPE:
            raise QSAWorkspaceInputError("QSA currently requires a two-byte dtype")
        if self.phase not in {"eager", "capture"}:
            raise QSAWorkspaceInputError("phase must be 'eager' or 'capture'")
        if self.topk_backend not in {"triton", "torch"}:
            raise QSAWorkspaceInputError("topk_backend must be 'triton' or 'torch'")
        if self.query_heads % self.kv_heads:
            raise QSAWorkspaceInputError("query_heads must be divisible by kv_heads")
        if self.top_k % self.compression_ratio:
            raise QSAWorkspaceInputError("top_k must be divisible by compression_ratio")
        if self.page_size % self.compression_ratio:
            raise QSAWorkspaceInputError("page_size must be divisible by compression_ratio")
        if self.rotary_dim > self.index_head_dim or self.rotary_dim % 2:
            raise QSAWorkspaceInputError("rotary_dim must be even and fit index_head_dim")
        if self.context_tokens // self.compression_ratio > self.score_columns:
            raise QSAWorkspaceInputError("page-table shape is smaller than the complete context")

    @property
    def score_columns(self) -> int:
        return _mul(
            self.page_table_width, self.page_size // self.compression_ratio, label="score columns"
        )

    @property
    def chunk_rows(self) -> int:
        return min(
            self.token_rows,
            max(1, _SCORE_BUDGET // _mul(self.score_columns, _FP32, label="score tile")),
        )

    @property
    def selection_width(self) -> int:
        return self.top_k + self.compression_ratio - 1

    @property
    def attention_splits(self) -> int:
        return _attention_splits(
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
        if _add(*self.components.values(), label=f"{self.name} category") != self.bytes:
            raise QSAWorkspaceInputError(f"{self.name} category total does not match components")
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        object.__setattr__(self, "shapes", MappingProxyType(dict(self.shapes)))


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
    score = _category(
        "score",
        {
            "q_index": _mul(
                rows, request.index_heads, request.index_head_dim, _QSA_DTYPE, label="q_index"
            ),
            "logits": _mul(chunk, columns, _FP32, label="score logits"),
            "visible": _mul(chunk, _I32, label="visible blocks"),
            "metadata": _add(
                _mul(rows, _I32, label="last_indices"),
                _mul(rows + 1, _I32, label="cu_seqlens"),
                _mul(rows, _I32, label="token_to_req"),
                _mul(request.num_req_slots, _I32, label="seq_lens"),
                _mul(request.num_req_slots, _I32, label="ring_slots"),
                _mul(request.num_req_slots, request.page_table_width, _I32, label="block_table"),
                label="metadata",
            ),
            "index_rope": _mul(request.max_position, request.rotary_dim, _FP32, label="index RoPE"),
        },
        {
            "q_index": (rows, request.index_heads, request.index_head_dim),
            "logits": (chunk, columns),
            "visible": (chunk,),
            "metadata": (rows,),
            "index_rope": (request.max_position, request.rotary_dim),
        },
    )
    scratch = _topk_scratch(columns, block_top_k, request.topk_backend)
    top_k = _category(
        "top_k",
        {
            "blocks": _mul(chunk, block_top_k, _I32, label="top-k blocks"),
            "candidate_scratch": _mul(chunk, scratch, _I32, label="top-k scratch"),
        },
        {"blocks": (chunk, block_top_k), "candidate_scratch": (chunk, scratch)},
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
            "pooled": _mul(rows, request.index_head_dim, _QSA_DTYPE, label="pooled rows"),
            "first_positions": _mul(rows, _I32, label="first positions"),
        },
        {
            "compressed_slab": (request.num_index_layers, state_rows, request.index_head_dim),
            "pending_ring": (
                request.num_req_slots,
                request.num_index_layers,
                request.ring_capacity,
                request.index_head_dim,
            ),
            "pooled": (rows, request.index_head_dim),
            "first_positions": (rows,),
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
        label="persistent state",
    )
    capture = (
        _add(
            _mul(
                request.num_req_slots, request.page_table_width, _I32, label="capture block table"
            ),
            _mul(request.num_req_slots * 2 + 1, _I32, label="capture metadata"),
            label="capture resident metadata",
        )
        if request.phase == "capture"
        else 0
    )
    stages = (
        score.components["q_index"] + score.components["logits"] + score.components["visible"],
        top_k.bytes,
        expand.bytes,
        attention.bytes,
        state.components["pooled"] + state.components["first_positions"],
    )
    peak = max(stages)
    required = _add(persistent, capture, peak, label="QSA high-water")
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
    "validate_qsa_workspace_capacity",
]
