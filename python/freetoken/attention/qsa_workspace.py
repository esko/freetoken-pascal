"""Pure shape and byte accounting for QSA's bounded workspaces.

The planner mirrors the allocation shapes in the merged QSA backend, but never imports
Torch or allocates device memory.  A later placement planner can use the immutable plan
to reject an unsafe launch before entering a kernel.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

# The runtime's device allocations use signed 64-bit sizes on supported hosts.
MAX_QSA_WORKSPACE_BYTES = (1 << 63) - 1
_I32_BYTES = 4
_FP32_BYTES = 4

QSA_WORKSPACE_CATEGORIES = (
    "score",
    "top_k",
    "expand_gather",
    "attention",
    "state",
)


class QSAWorkspaceError(ValueError):
    """Base class for deterministic QSA workspace planning failures."""


class QSAWorkspaceInputError(QSAWorkspaceError):
    """Raised when shape, dtype or arithmetic inputs cannot describe a QSA launch."""


class QSAWorkspaceCapacityError(QSAWorkspaceError):
    """Raised when the supplied capacity cannot hold the complete QSA plan."""

    def __init__(self, required_bytes: int, capacity_bytes: int, telemetry: QSAWorkspaceTelemetry):
        self.required_bytes = required_bytes
        self.capacity_bytes = capacity_bytes
        self.deficit_bytes = required_bytes - capacity_bytes
        self.telemetry = telemetry
        super().__init__(
            "QSA workspace capacity is insufficient: "
            f"required={required_bytes} bytes, capacity={capacity_bytes} bytes, "
            f"deficit={self.deficit_bytes} bytes"
        )


def _int(value: Any, name: str) -> int:
    """Coerce an integer-like value while rejecting booleans and fractional values."""
    if isinstance(value, bool):
        raise QSAWorkspaceInputError(f"{name} must be an integer, got bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise QSAWorkspaceInputError(f"{name} must be an integer") from exc
    if value < 0:
        raise QSAWorkspaceInputError(f"{name} must be non-negative, got {value}")
    if value > MAX_QSA_WORKSPACE_BYTES:
        raise QSAWorkspaceInputError(f"{name} exceeds the supported integer range")
    return value


def _positive(value: Any, name: str) -> int:
    value = _int(value, name)
    if value == 0:
        raise QSAWorkspaceInputError(f"{name} must be positive")
    return value


def _mul(*values: int, label: str) -> int:
    result = 1
    for value in values:
        if value < 0:
            raise QSAWorkspaceInputError(f"{label} contains a negative factor")
        if value and result > MAX_QSA_WORKSPACE_BYTES // value:
            raise QSAWorkspaceInputError(f"{label} integer overflow")
        result *= value
    return result


def _add(*values: int, label: str) -> int:
    result = 0
    for value in values:
        if value < 0:
            raise QSAWorkspaceInputError(f"{label} contains a negative value")
        if result > MAX_QSA_WORKSPACE_BYTES - value:
            raise QSAWorkspaceInputError(f"{label} integer overflow")
        result += value
    return result


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def qsa_top_k_scratch_columns(block_count: int, block_top_k: int) -> int:
    """Return the per-row int32 scratch width used by upstream block top-k.

    The split path is kept here as arithmetic so planning does not import Triton.  A zero
    result is the upstream one-program path, which has no top-k candidate scratch.
    """
    columns = _positive(block_count, "block_count")
    top_k = _positive(block_top_k, "block_top_k")
    max_resident = 8192
    min_chunk = 4096
    if columns <= min_chunk:
        return 0
    max_splits = max_resident // _next_power_of_two(top_k)
    if max_splits < 2:
        return 0
    chunk = max(min_chunk, _next_power_of_two(_ceil_div(columns, max_splits)))
    if chunk > max_resident:
        return 0
    splits = _ceil_div(columns, chunk)
    if splits < 2 or _mul(splits, top_k, label="top-k split candidates") >= columns:
        return 0
    return _mul(2, splits, top_k, label="top-k scratch columns")


@dataclass(frozen=True)
class QSAWorkspaceInputs:
    """Concrete shapes and byte widths needed to account one QSA launch.

    ``token_rows`` is the ragged query-row count for this launch, while ``context_tokens``
    describes the longest logical context represented by ``block_count`` compressed score
    columns.  The page and state fields mirror ``QSAKVCache``'s actual persistent shapes.
    """

    context_tokens: int
    token_rows: int
    block_count: int
    index_heads: int
    query_heads: int
    kv_heads: int
    head_dim: int
    index_head_dim: int
    top_k: int
    dtype_bytes: int
    compression_ratio: int = 1
    attention_splits: int = 1
    num_index_layers: int = 1
    num_req_slots: int = 1
    ring_capacity: int = 1
    num_pages: int = 1
    page_size: int = 1

    def __post_init__(self) -> None:
        positive_fields = (
            "context_tokens",
            "token_rows",
            "block_count",
            "index_heads",
            "query_heads",
            "kv_heads",
            "head_dim",
            "index_head_dim",
            "top_k",
            "dtype_bytes",
            "compression_ratio",
            "attention_splits",
            "num_index_layers",
            "num_req_slots",
            "ring_capacity",
            "num_pages",
            "page_size",
        )
        for name in positive_fields:
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.query_heads % self.kv_heads:
            raise QSAWorkspaceInputError(
                "query_heads must be divisible by kv_heads for grouped-query attention"
            )
        if self.top_k % self.compression_ratio:
            raise QSAWorkspaceInputError(
                "top_k must be divisible by compression_ratio for QSA block selection"
            )
        if self.page_size % self.compression_ratio:
            raise QSAWorkspaceInputError(
                "page_size must be divisible by compression_ratio for QSA state rows"
            )
        complete_context_blocks = self.context_tokens // self.compression_ratio
        if self.block_count < complete_context_blocks:
            raise QSAWorkspaceInputError(
                "block_count is smaller than the complete compressed context"
            )


@dataclass(frozen=True)
class QSAWorkspaceCategory:
    """One named workspace category with byte totals and concrete allocation shapes."""

    name: str
    bytes: int
    components: Mapping[str, int]
    shapes: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if self.name not in QSA_WORKSPACE_CATEGORIES:
            raise QSAWorkspaceInputError(f"unknown QSA workspace category {self.name!r}")
        total = _add(*self.components.values(), label=f"{self.name} category")
        if total != self.bytes:
            raise QSAWorkspaceInputError(
                f"{self.name} category bytes do not match its components: {self.bytes} != {total}"
            )
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        object.__setattr__(self, "shapes", MappingProxyType(dict(self.shapes)))


@dataclass(frozen=True)
class QSAWorkspaceInventory:
    """Complete, immutable inventory of the five QSA allocation categories."""

    categories: Mapping[str, QSAWorkspaceCategory]

    def __post_init__(self) -> None:
        keys = set(self.categories)
        expected = set(QSA_WORKSPACE_CATEGORIES)
        missing = expected - keys
        unknown = keys - expected
        if missing:
            raise QSAWorkspaceInputError(
                f"QSA workspace inventory is missing categories: {', '.join(sorted(missing))}"
            )
        if unknown:
            raise QSAWorkspaceInputError(
                f"QSA workspace inventory has unknown categories: {', '.join(sorted(unknown))}"
            )
        for name, category in self.categories.items():
            if not isinstance(category, QSAWorkspaceCategory) or category.name != name:
                raise QSAWorkspaceInputError(f"QSA inventory category {name!r} is malformed")
        object.__setattr__(self, "categories", MappingProxyType(dict(self.categories)))

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, int | QSAWorkspaceCategory]
    ) -> QSAWorkspaceInventory:
        """Build an inventory from category totals for #73 integration tests."""
        categories: dict[str, QSAWorkspaceCategory] = {}
        for name, value in values.items():
            if isinstance(value, QSAWorkspaceCategory):
                categories[name] = value
                continue
            total = _int(value, f"{name} bytes")
            categories[name] = QSAWorkspaceCategory(
                name=name,
                bytes=total,
                components=MappingProxyType({"total": total}),
                shapes=MappingProxyType({"total": (total,)}),
            )
        return cls(categories)

    def __iter__(self):
        return iter(QSA_WORKSPACE_CATEGORIES)

    def __len__(self) -> int:
        return len(QSA_WORKSPACE_CATEGORIES)

    def __getitem__(self, name: str) -> QSAWorkspaceCategory:
        aliases = {"topk": "top_k", "top-k": "top_k", "expand": "expand_gather"}
        try:
            return self.categories[aliases.get(name, name)]
        except KeyError as exc:
            raise KeyError(name) from exc

    def __getattr__(self, name: str) -> QSAWorkspaceCategory:
        if name in QSA_WORKSPACE_CATEGORIES or name in {"topk", "top-k", "expand"}:
            return self[name]
        raise AttributeError(name)

    def values(self):
        return tuple(self.categories[name] for name in QSA_WORKSPACE_CATEGORIES)


@dataclass(frozen=True)
class QSAWorkspaceTelemetry:
    """Machine-readable plan status for a later placement/readiness consumer."""

    status: str
    required_bytes: int
    capacity_bytes: int | None
    headroom_bytes: int | None
    categories: Mapping[str, int]
    shapes: Mapping[str, Mapping[str, tuple[int, ...]]]
    request: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "required_bytes": self.required_bytes,
            "capacity_bytes": self.capacity_bytes,
            "headroom_bytes": self.headroom_bytes,
            "categories": dict(self.categories),
            "shapes": {name: dict(shapes) for name, shapes in self.shapes.items()},
            "request": dict(self.request),
        }

    to_dict = as_dict


@dataclass(frozen=True)
class QSAWorkspacePlan:
    """Immutable QSA workspace plan that is safe to pass to #73 without CUDA allocation."""

    request: QSAWorkspaceInputs
    inventory: QSAWorkspaceInventory
    required_bytes: int
    capacity_bytes: int | None = None
    telemetry: QSAWorkspaceTelemetry | None = None

    def __post_init__(self) -> None:
        expected = _add(
            *(category.bytes for category in self.inventory.values()), label="QSA workspace total"
        )
        if expected != self.required_bytes:
            raise QSAWorkspaceInputError(
                f"QSA workspace total does not match inventory: {self.required_bytes} != {expected}"
            )
        if self.capacity_bytes is not None:
            capacity = _int(self.capacity_bytes, "capacity_bytes")
            object.__setattr__(self, "capacity_bytes", capacity)
        if self.telemetry is None:
            object.__setattr__(self, "telemetry", self._telemetry("unvalidated", None))

    def _telemetry(self, status: str, capacity_bytes: int | None) -> QSAWorkspaceTelemetry:
        headroom = None if capacity_bytes is None else capacity_bytes - self.required_bytes
        return QSAWorkspaceTelemetry(
            status=status,
            required_bytes=self.required_bytes,
            capacity_bytes=capacity_bytes,
            headroom_bytes=headroom,
            categories={name: self.inventory[name].bytes for name in QSA_WORKSPACE_CATEGORIES},
            shapes={name: self.inventory[name].shapes for name in QSA_WORKSPACE_CATEGORIES},
            request={
                name: getattr(self.request, name) for name in self.request.__dataclass_fields__
            },
        )

    def validate_capacity(self, capacity_bytes: int | None = None) -> QSAWorkspaceTelemetry:
        """Validate capacity before a QSA launch and return structured ready telemetry."""
        if capacity_bytes is None:
            capacity_bytes = self.capacity_bytes
        if capacity_bytes is None:
            raise QSAWorkspaceInputError("capacity_bytes is required before launching QSA")
        capacity_bytes = _int(capacity_bytes, "capacity_bytes")
        if capacity_bytes < self.required_bytes:
            telemetry = self._telemetry("insufficient-capacity", capacity_bytes)
            raise QSAWorkspaceCapacityError(self.required_bytes, capacity_bytes, telemetry)
        telemetry = self._telemetry("ready", capacity_bytes)
        object.__setattr__(self, "capacity_bytes", capacity_bytes)
        object.__setattr__(self, "telemetry", telemetry)
        return telemetry

    def as_dict(self) -> dict[str, Any]:
        assert self.telemetry is not None
        return {
            "request": self.telemetry.as_dict()["request"],
            "required_bytes": self.required_bytes,
            "capacity_bytes": self.capacity_bytes,
            "categories": self.telemetry.as_dict()["categories"],
            "shapes": self.telemetry.as_dict()["shapes"],
            "status": self.telemetry.status,
        }

    to_dict = as_dict


def _category(
    name: str,
    components: Mapping[str, int],
    shapes: Mapping[str, tuple[int, ...]],
) -> QSAWorkspaceCategory:
    return QSAWorkspaceCategory(
        name=name,
        bytes=_add(*components.values(), label=f"{name} category"),
        components=components,
        shapes=shapes,
    )


def calculate_qsa_workspace(request: QSAWorkspaceInputs) -> QSAWorkspacePlan:
    """Calculate all QSA workspace categories from concrete launch and cache shapes."""
    if not isinstance(request, QSAWorkspaceInputs):
        raise QSAWorkspaceInputError("request must be a QSAWorkspaceInputs instance")

    rows = request.token_rows
    ratio = request.compression_ratio
    block_top_k = request.top_k // ratio
    state_rows = _add(
        _mul(request.num_pages, request.page_size, label="compressed state rows") // ratio,
        request.num_req_slots,
        label="compressed state rows",
    )

    score_components = {
        "q_index": _mul(
            rows,
            request.index_heads,
            request.index_head_dim,
            request.dtype_bytes,
            label="score q_index",
        ),
        "logits": _mul(rows, request.block_count, _FP32_BYTES, label="score logits"),
        "visible": _mul(rows, _I32_BYTES, label="score visible blocks"),
    }
    score_shapes = {
        "q_index": (rows, request.index_heads, request.index_head_dim),
        "logits": (rows, request.block_count),
        "visible": (rows,),
    }

    topk_scratch = qsa_top_k_scratch_columns(request.block_count, block_top_k)
    top_k_components = {
        "blocks": _mul(rows, block_top_k, _I32_BYTES, label="top-k blocks"),
        "candidate_scratch": _mul(rows, topk_scratch, _I32_BYTES, label="top-k candidate scratch"),
    }
    top_k_shapes = {
        "blocks": (rows, block_top_k),
        "candidate_scratch": (rows, topk_scratch),
    }

    expand_components = {
        "indices": _mul(rows, request.top_k + ratio - 1, _I32_BYTES, label="expand-gather indices")
    }
    expand_shapes = {"indices": (rows, request.top_k + ratio - 1)}

    attention_components = {
        # qsa_sparse_paged_attention receives an output tensor shaped like q.
        "output": _mul(
            rows,
            request.query_heads,
            request.head_dim,
            request.dtype_bytes,
            label="attention output",
        ),
        "partial_output": (
            _mul(
                request.attention_splits,
                rows,
                request.query_heads,
                request.head_dim,
                _FP32_BYTES,
                label="attention partial output",
            )
            if request.attention_splits > 1
            else 0
        ),
        "partial_lse": (
            _mul(
                request.attention_splits,
                rows,
                request.query_heads,
                _FP32_BYTES,
                label="attention partial lse",
            )
            if request.attention_splits > 1
            else 0
        ),
    }
    attention_shapes = {
        "output": (rows, request.query_heads, request.head_dim),
        "partial_output": (request.attention_splits, rows, request.query_heads, request.head_dim),
        "partial_lse": (request.attention_splits, rows, request.query_heads),
    }

    state_components = {
        "compressed_slab": _mul(
            request.num_index_layers,
            state_rows,
            request.index_head_dim,
            request.dtype_bytes,
            label="compressed state slab",
        ),
        "pending_ring": _mul(
            request.num_req_slots,
            request.num_index_layers,
            request.ring_capacity,
            request.index_head_dim,
            request.dtype_bytes,
            label="pending state ring",
        ),
        "pooled": _mul(
            rows, request.index_head_dim, request.dtype_bytes, label="state pooled rows"
        ),
        "first_positions": _mul(rows, _I32_BYTES, label="state first positions"),
    }
    state_shapes = {
        "compressed_slab": (
            request.num_index_layers,
            state_rows,
            request.index_head_dim,
        ),
        "pending_ring": (
            request.num_req_slots,
            request.num_index_layers,
            request.ring_capacity,
            request.index_head_dim,
        ),
        "pooled": (rows, request.index_head_dim),
        "first_positions": (rows,),
    }

    inventory = QSAWorkspaceInventory(
        {
            "score": _category("score", score_components, score_shapes),
            "top_k": _category("top_k", top_k_components, top_k_shapes),
            "expand_gather": _category("expand_gather", expand_components, expand_shapes),
            "attention": _category("attention", attention_components, attention_shapes),
            "state": _category("state", state_components, state_shapes),
        }
    )
    required = _add(
        *(category.bytes for category in inventory.values()), label="QSA workspace total"
    )
    return QSAWorkspacePlan(request=request, inventory=inventory, required_bytes=required)


def validate_qsa_workspace_capacity(
    request: QSAWorkspaceInputs, capacity_bytes: int
) -> QSAWorkspaceTelemetry:
    """Calculate and validate a plan in one pre-launch operation."""
    return calculate_qsa_workspace(request).validate_capacity(capacity_bytes)


# Naming aliases make the shape contract easy to discover for the placement planner.
QSAWorkspaceRequest = QSAWorkspaceInputs
build_qsa_workspace_plan = calculate_qsa_workspace
plan_qsa_workspace = calculate_qsa_workspace


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
    "QSAWorkspaceRequest",
    "QSAWorkspaceTelemetry",
    "build_qsa_workspace_plan",
    "calculate_qsa_workspace",
    "plan_qsa_workspace",
    "qsa_top_k_scratch_columns",
    "validate_qsa_workspace_capacity",
]
