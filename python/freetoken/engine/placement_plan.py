"""Torch-free per-GPU VRAM placement accounting and startup-canary contract.

The planner is deliberately an H0 boundary.  It performs checked integer accounting and
returns immutable data; it never imports Torch, calls CUDA, allocates memory, or changes a
runtime placement.  A later serving owner can feed allocator observations to
:func:`evaluate_canary` without changing this contract.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Any

MAX_PLACEMENT_BYTES = (1 << 63) - 1
PLACEMENT_SCHEMA_VERSION = 1
_PLACEHOLDER_GPU_UUIDS = frozenset({"", "na", "n/a", "none", "null", "placeholder", "unknown"})

# Keep every release-visible bucket explicit.  In particular, the GDN/KV recurrent state is
# distinct from QSA state, and QSA phase/workspace names are not collapsed into generic_workspaces:
# post-prefill growth in one phase must be attributable to the planned phase.
PLACEMENT_CATEGORIES = (
    "dense_resident_weights",
    "shared_experts",
    "gdn_kv_recurrent_state",
    "qsa_persistent_score",
    "qsa_persistent_top_k",
    "qsa_persistent_expand_gather",
    "qsa_persistent_attention",
    "qsa_persistent_state",
    "qsa_transient_score",
    "qsa_transient_top_k",
    "qsa_transient_expand_gather",
    "qsa_transient_attention",
    "qsa_transient_state",
    "cuda_context",
    "generic_workspaces",
    "transfer_buffers",
    "static_expert_cache_slots",
    "dynamic_expert_cache_slots",
    "safety_reserve",
)
QSA_PERSISTENT_CATEGORIES = tuple(
    name for name in PLACEMENT_CATEGORIES if name.startswith("qsa_persistent_")
)
QSA_TRANSIENT_CATEGORIES = tuple(
    name for name in PLACEMENT_CATEGORIES if name.startswith("qsa_transient_")
)
_QSA_CATEGORIES = frozenset(QSA_PERSISTENT_CATEGORIES + QSA_TRANSIENT_CATEGORIES)

_CATEGORY_SET = frozenset(PLACEMENT_CATEGORIES)
_CHECKPOINTS = frozenset(
    {
        "post-load",
        "post-canary",
        "post-first-small-prefill",
        "post-first-large-prefill",
        "steady-decode",
        "cancellation",
        "checkpoint-restore",
    }
)
_READINESS_CHECKPOINTS = frozenset({"post-load", "post-first-large-prefill"})
PLACEMENT_CHECKPOINTS = tuple(
    (
        "post-load",
        "post-canary",
        "post-first-small-prefill",
        "post-first-large-prefill",
        "steady-decode",
        "cancellation",
        "checkpoint-restore",
    )
)


class PlacementPlannerError(ValueError):
    """Base class for malformed placement inputs and fail-closed evaluations."""


class PlacementInputError(PlacementPlannerError):
    """A placement input violates the immutable accounting schema."""


class PlacementCapacityError(PlacementPlannerError):
    """A placement plan or canary observation cannot satisfy its safety envelope."""


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise PlacementInputError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise PlacementInputError(f"{name} must be an integer") from exc
    if result < 0:
        raise PlacementInputError(f"{name} must be non-negative, got {result}")
    if result > MAX_PLACEMENT_BYTES:
        raise PlacementInputError(f"{name} exceeds the supported integer range")
    return result


def _positive(value: Any, name: str) -> int:
    result = _int(value, name)
    if result == 0:
        raise PlacementInputError(f"{name} must be positive")
    return result


def _add(*values: int, label: str) -> int:
    result = 0
    for value in values:
        if value < 0 or result > MAX_PLACEMENT_BYTES - value:
            raise PlacementInputError(f"{label} integer overflow")
        result += value
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PlacementInputError(f"{name} must be a boolean")
    return value


def _gpu_uuid(value: Any, name: str = "gpu_uuid") -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlacementInputError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if normalized.casefold() in _PLACEHOLDER_GPU_UUIDS:
        raise PlacementInputError(f"{name} must be an explicit non-placeholder GPU UUID")
    return normalized


def _signed_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise PlacementInputError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise PlacementInputError(f"{name} must be an integer") from exc
    if result < -MAX_PLACEMENT_BYTES or result > MAX_PLACEMENT_BYTES:
        raise PlacementInputError(f"{name} exceeds the supported integer range")
    return result


def _categories(value: Mapping[str, int], *, name: str = "categories") -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise PlacementInputError(f"{name} must be a mapping")
    try:
        keys = set(value)
    except TypeError as exc:
        raise PlacementInputError(f"{name} keys must be hashable category names") from exc
    missing = _CATEGORY_SET - keys
    unknown = keys - _CATEGORY_SET
    if missing:
        raise PlacementInputError(f"{name} missing categories: {sorted(missing)}")
    if unknown:
        raise PlacementInputError(
            f"{name} contains unknown categories: {sorted(unknown, key=repr)}"
        )
    normalized = {
        category: _int(value[category], f"{name}.{category}") for category in PLACEMENT_CATEGORIES
    }
    return MappingProxyType(normalized)


def _required_totals(
    categories: Mapping[str, int], *, label: str
) -> tuple[int, int, int, int, int]:
    """Return ``(non-reserve, persistent-QSA, transient-QSA, QSA total, total)`` bytes.

    QSA categories describe two allocator lifetimes.  The persistent set is retained while the
    transient set is a phase high-water, so the aggregate is built from those lifetime totals,
    not from a generic workspace allowance or an accidental second inventory sum.
    """
    persistent = _add(
        *(categories[name] for name in QSA_PERSISTENT_CATEGORIES),
        label=f"{label} QSA persistent bytes",
    )
    transient = _add(
        *(categories[name] for name in QSA_TRANSIENT_CATEGORIES),
        label=f"{label} QSA transient high-water bytes",
    )
    non_qsa = _add(
        *(
            categories[name]
            for name in PLACEMENT_CATEGORIES
            if name not in _QSA_CATEGORIES and name != "safety_reserve"
        ),
        label=f"{label} non-QSA bytes",
    )
    qsa_required = _add(persistent, transient, label=f"{label} QSA required high-water")
    total = _add(
        non_qsa,
        qsa_required,
        categories["safety_reserve"],
        label=f"{label} required high-water",
    )
    return non_qsa, persistent, transient, qsa_required, total


@dataclass(frozen=True, slots=True)
class PlacementPlanInput:
    """One rank's capacity and exact planned bytes before reserve normalization."""

    capacity_bytes: int
    categories: Mapping[str, int]
    gpu_uuid: str
    available_bytes: int | None = None

    def __post_init__(self) -> None:
        capacity = _positive(self.capacity_bytes, "capacity_bytes")
        categories = _categories(self.categories)
        gpu_uuid = _gpu_uuid(self.gpu_uuid)
        available = (
            capacity
            if self.available_bytes is None
            else _int(self.available_bytes, "available_bytes")
        )
        if available > capacity:
            raise PlacementInputError(
                f"available_bytes ({available}) exceeds capacity_bytes ({capacity})"
            )
        _add(*categories.values(), label="placement categories")
        object.__setattr__(self, "capacity_bytes", capacity)
        object.__setattr__(self, "available_bytes", available)
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "gpu_uuid", gpu_uuid)


# The shorter spelling is useful to callers that use the GPU term explicitly.
GpuPlacementInput = PlacementPlanInput
GPUPlacementInput = PlacementPlanInput


@dataclass(frozen=True, slots=True)
class GPUPlacementPlan:
    """Immutable normalized plan and capacity telemetry for one GPU rank."""

    rank: int
    gpu_uuid: str
    capacity_bytes: int
    available_bytes: int
    categories: Mapping[str, int]
    required_bytes: int
    headroom_bytes: int
    deficit_bytes: int
    status: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise PlacementInputError("rank must be a non-negative integer")
        gpu_uuid = _gpu_uuid(self.gpu_uuid)
        capacity = _positive(self.capacity_bytes, "capacity_bytes")
        available = _int(self.available_bytes, "available_bytes")
        required_input = _int(self.required_bytes, "required_bytes")
        if available > capacity:
            raise PlacementInputError("available_bytes exceeds capacity_bytes")
        categories = _categories(self.categories)
        required = _required_totals(categories, label="placement")[4]
        if required != required_input:
            raise PlacementInputError("required_bytes does not match placement categories")
        headroom = available - required
        deficit = max(0, -headroom)
        if self.headroom_bytes != headroom or self.deficit_bytes != deficit:
            raise PlacementInputError("placement headroom/deficit does not match capacity")
        if self.status not in {"ready", "insufficient-capacity"}:
            raise PlacementInputError(f"unknown placement status {self.status!r}")
        expected_status = "ready" if deficit == 0 else "insufficient-capacity"
        if self.status != expected_status:
            raise PlacementInputError("placement status does not match capacity")
        object.__setattr__(self, "capacity_bytes", capacity)
        object.__setattr__(self, "available_bytes", available)
        object.__setattr__(self, "required_bytes", required_input)
        object.__setattr__(self, "headroom_bytes", headroom)
        object.__setattr__(self, "deficit_bytes", deficit)
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "gpu_uuid", gpu_uuid)
        try:
            reasons = tuple(self.reasons)
        except TypeError as exc:
            raise PlacementInputError("placement reasons must be an iterable") from exc
        if any(not isinstance(reason, str) for reason in reasons):
            raise PlacementInputError("placement reasons must be strings")
        object.__setattr__(self, "reasons", reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLACEMENT_SCHEMA_VERSION,
            "rank": self.rank,
            "gpu_uuid": self.gpu_uuid,
            "key": self.key,
            "status": self.status,
            "required_bytes": self.required_bytes,
            "non_qsa_required_bytes": self.non_qsa_required_bytes,
            "qsa_persistent_bytes": self.qsa_persistent_bytes,
            "qsa_transient_high_water_bytes": self.qsa_transient_high_water_bytes,
            "qsa_required_bytes": self.qsa_required_bytes,
            "live_required_bytes": self.live_required_bytes,
            "peak_required_bytes": self.peak_required_bytes,
            "available_bytes": self.available_bytes,
            "headroom_bytes": self.headroom_bytes,
            "deficit_bytes": self.deficit_bytes,
            "categories": dict(self.categories),
            "reasons": list(self.reasons),
        }

    @property
    def key(self) -> str:
        """Stable serialized identity, even when a topology has asymmetric ranks."""
        return f"{self.gpu_uuid}:{self.rank}"

    @property
    def nonreserve_required_bytes(self) -> int:
        return self.required_bytes - self.categories["safety_reserve"]

    @property
    def live_required_bytes(self) -> int:
        """Resident high-water floor excluding transient QSA workspaces and reserve."""
        return _add(
            self.non_qsa_required_bytes,
            self.qsa_persistent_bytes,
            label="live required bytes",
        )

    @property
    def peak_required_bytes(self) -> int:
        """Resident plus transient QSA high-water demand, excluding safety reserve."""
        return self.nonreserve_required_bytes

    @property
    def safety_reserve_bytes(self) -> int:
        return self.categories["safety_reserve"]

    @property
    def non_qsa_required_bytes(self) -> int:
        return _required_totals(self.categories, label="placement")[0]

    @property
    def qsa_persistent_bytes(self) -> int:
        return _add(
            *(self.categories[name] for name in QSA_PERSISTENT_CATEGORIES),
            label="QSA persistent bytes",
        )

    @property
    def qsa_transient_high_water_bytes(self) -> int:
        return _add(
            *(self.categories[name] for name in QSA_TRANSIENT_CATEGORIES),
            label="QSA transient high-water bytes",
        )

    @property
    def qsa_required_bytes(self) -> int:
        return _add(
            self.qsa_persistent_bytes,
            self.qsa_transient_high_water_bytes,
            label="QSA required high-water bytes",
        )

    to_dict = as_dict


@dataclass(frozen=True, slots=True)
class PlacementPlan:
    """Immutable per-GPU placement plan, including all release-visible categories."""

    gpus: tuple[GPUPlacementPlan, ...]
    safety_reserve_bytes: int

    def __post_init__(self) -> None:
        try:
            gpus = tuple(self.gpus)
        except TypeError as exc:
            raise PlacementInputError("placement plan GPUs must be iterable") from exc
        if len(gpus) not in {1, 2}:
            raise PlacementInputError("placement plan requires 1 or 2 GPUs")
        if any(not isinstance(item, GPUPlacementPlan) for item in gpus):
            raise PlacementInputError("placement plan entries must be GPUPlacementPlan values")
        if tuple(item.rank for item in gpus) != tuple(range(len(gpus))):
            raise PlacementInputError("GPU ranks must be contiguous and start at zero")
        if len({item.gpu_uuid for item in gpus}) != len(gpus):
            raise PlacementInputError("duplicate GPU UUIDs are not allowed")
        reserve = _int(self.safety_reserve_bytes, "safety_reserve_bytes")
        for item in gpus:
            if item.categories["safety_reserve"] != reserve:
                raise PlacementInputError(
                    f"GPU {item.rank} safety reserve disagrees with plan reserve"
                )
        object.__setattr__(self, "gpus", gpus)
        object.__setattr__(self, "safety_reserve_bytes", reserve)

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def per_gpu(self) -> tuple[GPUPlacementPlan, ...]:
        return self.gpus

    @property
    def by_key(self) -> Mapping[str, GPUPlacementPlan]:
        return MappingProxyType({item.key: item for item in self.gpus})

    @property
    def ready(self) -> bool:
        return all(item.status == "ready" for item in self.gpus)

    @property
    def telemetry(self) -> tuple[GPUPlacementPlan, ...]:
        return self.gpus

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLACEMENT_SCHEMA_VERSION,
            "gpu_count": self.gpu_count,
            "safety_reserve_bytes": self.safety_reserve_bytes,
            "gpus": [item.as_dict() for item in self.gpus],
            "gpus_by_key": {key: item.as_dict() for key, item in self.by_key.items()},
        }

    to_dict = as_dict

    def backoff(self, profiles: Sequence[BackoffProfile]) -> BackoffStateMachine:
        return BackoffStateMachine(profiles)


def _normalize_input(value: PlacementPlanInput | Mapping[str, Any]) -> PlacementPlanInput:
    if isinstance(value, PlacementPlanInput):
        return value
    if isinstance(value, Mapping):
        try:
            return PlacementPlanInput(**value)
        except TypeError as exc:
            raise PlacementInputError("GPU placement input has an invalid field set") from exc
    raise PlacementInputError("each GPU placement input must be PlacementPlanInput or a mapping")


def plan_placement(
    gpus: Iterable[PlacementPlanInput | Mapping[str, Any]],
    *,
    safety_reserve_bytes: int | None = None,
) -> PlacementPlan:
    """Build a deterministic one- or two-GPU plan with checked category totals.

    ``available_bytes`` is the preflight free/usable budget when supplied, otherwise total
    capacity.  An explicit reserve replaces each input's reserve category, which keeps reserve
    policy in one call-site while retaining the exact category in the serialized plan.
    """
    if isinstance(gpus, (str, bytes)):
        raise PlacementInputError("gpus must be an iterable of one or two GPU inputs")
    try:
        inputs = tuple(_normalize_input(item) for item in gpus)
    except TypeError as exc:
        raise PlacementInputError("gpus must be an iterable of one or two GPU inputs") from exc
    if len(inputs) not in {1, 2}:
        raise PlacementInputError("placement planner supports 1 or 2 GPUs")
    reserve = (
        None if safety_reserve_bytes is None else _int(safety_reserve_bytes, "safety_reserve_bytes")
    )
    result: list[GPUPlacementPlan] = []
    for rank, item in enumerate(inputs):
        values = dict(item.categories)
        if reserve is not None:
            values["safety_reserve"] = reserve
        values = dict(_categories(values))
        required = _required_totals(values, label=f"GPU {rank} placement")[4]
        available = item.available_bytes
        headroom = available - required
        deficit = max(0, -headroom)
        status = "ready" if deficit == 0 else "insufficient-capacity"
        reasons = () if status == "ready" else (f"insufficient-capacity deficit={deficit}",)
        result.append(
            GPUPlacementPlan(
                rank,
                item.gpu_uuid,
                item.capacity_bytes,
                available,
                values,
                required,
                headroom,
                deficit,
                status,
                reasons,
            )
        )
    effective_reserve = result[0].categories["safety_reserve"]
    if any(item.categories["safety_reserve"] != effective_reserve for item in result):
        raise PlacementInputError("all GPU safety reserves must match")
    return PlacementPlan(tuple(result), effective_reserve)


calculate_placement_plan = plan_placement


@dataclass(frozen=True, slots=True)
class PlacementObservation:
    """Synthetic or runtime-observed allocator state for one GPU checkpoint."""

    rank: int
    gpu_uuid: str
    driver_total_bytes: int
    driver_free_bytes: int
    allocator_allocated_bytes: int
    allocator_reserved_bytes: int
    allocator_high_water_bytes: int
    categories: Mapping[str, int]
    managed_memory: bool = False
    host_spill: bool = False
    fallback: bool = False
    allocation_retries: int = 0
    allocation_failures: int = 0
    cache_overcommit: bool = False
    unplanned_placement: bool = False
    retained_workspace_growth_bytes: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise PlacementInputError("rank must be a non-negative integer")
        gpu_uuid = _gpu_uuid(self.gpu_uuid)
        driver_total = _positive(self.driver_total_bytes, "driver_total_bytes")
        driver_free = _int(self.driver_free_bytes, "driver_free_bytes")
        allocated = _int(self.allocator_allocated_bytes, "allocator_allocated_bytes")
        reserved = _int(self.allocator_reserved_bytes, "allocator_reserved_bytes")
        high_water = _int(self.allocator_high_water_bytes, "allocator_high_water_bytes")
        categories = _categories(self.categories, name="observed categories")
        if driver_free > driver_total:
            raise PlacementInputError("driver_free_bytes exceeds driver_total_bytes")
        if allocated > reserved:
            raise PlacementInputError("allocator_allocated_bytes exceeds allocator_reserved_bytes")
        if reserved > driver_total:
            raise PlacementInputError("allocator_reserved_bytes exceeds driver_total_bytes")
        if high_water < allocated:
            raise PlacementInputError("allocator_high_water_bytes is below allocated bytes")
        if high_water > driver_total:
            raise PlacementInputError("allocator_high_water_bytes exceeds driver_total_bytes")
        if reserved + driver_free > driver_total:
            raise PlacementInputError("allocation reserved/free counters are inconsistent")
        for name in (
            "managed_memory",
            "host_spill",
            "fallback",
            "cache_overcommit",
            "unplanned_placement",
        ):
            _strict_bool(getattr(self, name), name)
        retries = _int(self.allocation_retries, "allocation_retries")
        failures = _int(self.allocation_failures, "allocation_failures")
        growth = _int(self.retained_workspace_growth_bytes, "retained_workspace_growth_bytes")
        object.__setattr__(self, "gpu_uuid", gpu_uuid)
        object.__setattr__(self, "driver_total_bytes", driver_total)
        object.__setattr__(self, "driver_free_bytes", driver_free)
        object.__setattr__(self, "allocator_allocated_bytes", allocated)
        object.__setattr__(self, "allocator_reserved_bytes", reserved)
        object.__setattr__(self, "allocator_high_water_bytes", high_water)
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "allocation_retries", retries)
        object.__setattr__(self, "allocation_failures", failures)
        object.__setattr__(self, "retained_workspace_growth_bytes", growth)

    @classmethod
    def from_plan(cls, plan: GPUPlacementPlan) -> PlacementObservation:
        """Create a deterministic passing observation for synthetic H0 tests."""
        allocated = plan.live_required_bytes
        peak = plan.peak_required_bytes
        return cls(
            rank=plan.rank,
            gpu_uuid=plan.gpu_uuid,
            driver_total_bytes=plan.capacity_bytes,
            driver_free_bytes=plan.capacity_bytes - peak,
            allocator_allocated_bytes=allocated,
            allocator_reserved_bytes=peak,
            allocator_high_water_bytes=peak,
            categories=plan.categories,
        )

    @property
    def key(self) -> str:
        return f"{self.gpu_uuid}:{self.rank}"

    # Compatibility/readability aliases for callers that use the concise allocator terms.
    @property
    def allocated_bytes(self) -> int:
        return self.allocator_allocated_bytes

    @property
    def reserved_bytes(self) -> int:
        return self.allocator_reserved_bytes

    @property
    def free_bytes(self) -> int:
        return self.driver_free_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLACEMENT_SCHEMA_VERSION,
            "rank": self.rank,
            "gpu_uuid": self.gpu_uuid,
            "key": self.key,
            "driver_total_bytes": self.driver_total_bytes,
            "driver_free_bytes": self.driver_free_bytes,
            "allocator_allocated_bytes": self.allocator_allocated_bytes,
            "allocator_reserved_bytes": self.allocator_reserved_bytes,
            "allocator_high_water_bytes": self.allocator_high_water_bytes,
            "categories": dict(self.categories),
            "managed_memory": self.managed_memory,
            "host_spill": self.host_spill,
            "fallback": self.fallback,
            "allocation_retries": self.allocation_retries,
            "allocation_failures": self.allocation_failures,
            "cache_overcommit": self.cache_overcommit,
            "unplanned_placement": self.unplanned_placement,
            "retained_workspace_growth_bytes": self.retained_workspace_growth_bytes,
        }

    to_dict = as_dict


@dataclass(frozen=True, slots=True)
class GPUCanaryTelemetry:
    """Structured canary decision and raw allocator counters for one rank."""

    rank: int
    gpu_uuid: str
    status: str
    required_bytes: int
    non_qsa_required_bytes: int
    qsa_persistent_bytes: int
    qsa_transient_high_water_bytes: int
    qsa_required_bytes: int
    available_bytes: int
    headroom_bytes: int
    deficit_bytes: int
    driver_total_bytes: int
    driver_free_bytes: int
    allocator_allocated_bytes: int
    allocator_reserved_bytes: int
    allocator_high_water_bytes: int
    planned_categories: Mapping[str, int]
    observed_categories: Mapping[str, int]
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise PlacementInputError("rank must be a non-negative integer")
        if not isinstance(self.status, str) or self.status not in {"pass", "fail"}:
            raise PlacementInputError(f"unknown canary status {self.status!r}")
        gpu_uuid = _gpu_uuid(self.gpu_uuid)
        planned = _categories(self.planned_categories, name="planned categories")
        observed = _categories(self.observed_categories, name="observed categories")
        non_qsa, persistent, transient, qsa_required, required = _required_totals(
            planned, label="planned"
        )
        counters = {
            "required_bytes": self.required_bytes,
            "non_qsa_required_bytes": self.non_qsa_required_bytes,
            "qsa_persistent_bytes": self.qsa_persistent_bytes,
            "qsa_transient_high_water_bytes": self.qsa_transient_high_water_bytes,
            "qsa_required_bytes": self.qsa_required_bytes,
            "available_bytes": self.available_bytes,
            "deficit_bytes": self.deficit_bytes,
            "driver_total_bytes": self.driver_total_bytes,
            "driver_free_bytes": self.driver_free_bytes,
            "allocator_allocated_bytes": self.allocator_allocated_bytes,
            "allocator_reserved_bytes": self.allocator_reserved_bytes,
            "allocator_high_water_bytes": self.allocator_high_water_bytes,
        }
        normalized = {name: _int(value, name) for name, value in counters.items()}
        headroom = _signed_int(self.headroom_bytes, "headroom_bytes")
        try:
            reasons = tuple(self.reasons)
        except TypeError as exc:
            raise PlacementInputError("canary reasons must be an iterable") from exc
        if any(not isinstance(reason, str) for reason in reasons):
            raise PlacementInputError("canary reasons must be strings")

        if normalized["required_bytes"] != required:
            raise PlacementInputError("canary required_bytes does not match planned categories")
        expected = {
            "non_qsa_required_bytes": non_qsa,
            "qsa_persistent_bytes": persistent,
            "qsa_transient_high_water_bytes": transient,
            "qsa_required_bytes": qsa_required,
        }
        for name, value in expected.items():
            if normalized[name] != value:
                raise PlacementInputError(f"canary {name} does not match planned categories")
        reserve = planned["safety_reserve"]
        if normalized["available_bytes"] != normalized["driver_free_bytes"]:
            raise PlacementInputError("canary available_bytes must equal driver_free_bytes")
        if headroom != normalized["driver_free_bytes"] - reserve:
            raise PlacementInputError("canary headroom_bytes does not match driver free/reserve")
        if normalized["deficit_bytes"] != max(0, -headroom):
            raise PlacementInputError("canary deficit_bytes does not match headroom")
        if normalized["driver_total_bytes"] == 0:
            raise PlacementInputError("driver_total_bytes must be positive")
        if normalized["driver_free_bytes"] > normalized["driver_total_bytes"]:
            raise PlacementInputError("driver_free_bytes exceeds driver_total_bytes")
        if normalized["allocator_allocated_bytes"] > normalized["allocator_reserved_bytes"]:
            raise PlacementInputError("allocator_allocated_bytes exceeds allocator_reserved_bytes")
        if normalized["allocator_reserved_bytes"] > normalized["driver_total_bytes"]:
            raise PlacementInputError("allocator_reserved_bytes exceeds driver_total_bytes")
        if normalized["allocator_high_water_bytes"] < normalized["allocator_allocated_bytes"]:
            raise PlacementInputError("allocator_high_water_bytes is below allocated bytes")
        if normalized["allocator_high_water_bytes"] > normalized["driver_total_bytes"]:
            raise PlacementInputError("allocator_high_water_bytes exceeds driver_total_bytes")
        if (
            normalized["allocator_reserved_bytes"] + normalized["driver_free_bytes"]
            > normalized["driver_total_bytes"]
        ):
            raise PlacementInputError("allocation reserved/free counters are inconsistent")
        if self.status == "pass" and (
            normalized["deficit_bytes"] != 0
            or headroom < 0
            or normalized["driver_free_bytes"] < reserve
            or normalized["required_bytes"] > normalized["driver_total_bytes"]
        ):
            raise PlacementInputError("passing canary telemetry is not capacity-safe")
        if (self.status == "pass") != (not reasons):
            raise PlacementInputError("canary status must agree with reasons")

        object.__setattr__(self, "rank", self.rank)
        object.__setattr__(self, "gpu_uuid", gpu_uuid)
        for name, value in normalized.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "headroom_bytes", headroom)
        object.__setattr__(self, "planned_categories", planned)
        object.__setattr__(self, "observed_categories", observed)
        object.__setattr__(self, "reasons", reasons)

    @property
    def key(self) -> str:
        return f"{self.gpu_uuid}:{self.rank}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLACEMENT_SCHEMA_VERSION,
            "rank": self.rank,
            "gpu_uuid": self.gpu_uuid,
            "key": f"{self.gpu_uuid}:{self.rank}",
            "status": self.status,
            "required_bytes": self.required_bytes,
            "non_qsa_required_bytes": self.non_qsa_required_bytes,
            "qsa_persistent_bytes": self.qsa_persistent_bytes,
            "qsa_transient_high_water_bytes": self.qsa_transient_high_water_bytes,
            "qsa_required_bytes": self.qsa_required_bytes,
            "live_required_bytes": self.live_required_bytes,
            "peak_required_bytes": self.peak_required_bytes,
            "available_bytes": self.available_bytes,
            "headroom_bytes": self.headroom_bytes,
            "deficit_bytes": self.deficit_bytes,
            "driver_total_bytes": self.driver_total_bytes,
            "driver_free_bytes": self.driver_free_bytes,
            "allocator_allocated_bytes": self.allocator_allocated_bytes,
            "allocator_reserved_bytes": self.allocator_reserved_bytes,
            "allocator_high_water_bytes": self.allocator_high_water_bytes,
            "planned_categories": dict(self.planned_categories),
            "observed_categories": dict(self.observed_categories),
            "reasons": list(self.reasons),
        }

    @property
    def live_required_bytes(self) -> int:
        return _add(
            self.non_qsa_required_bytes,
            self.qsa_persistent_bytes,
            label="canary live required bytes",
        )

    @property
    def peak_required_bytes(self) -> int:
        return _add(
            self.live_required_bytes,
            self.qsa_transient_high_water_bytes,
            label="canary peak required bytes",
        )

    to_dict = as_dict


@dataclass(frozen=True, slots=True)
class CanaryResult:
    """Deterministic aggregate result for one named startup/high-water checkpoint."""

    checkpoint: str
    status: str
    gpus: tuple[GPUCanaryTelemetry, ...]
    reasons: tuple[str, ...] = ()
    tolerance_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, str) or self.checkpoint not in _CHECKPOINTS:
            raise PlacementInputError(
                f"unknown checkpoint {self.checkpoint!r}; expected one of {PLACEMENT_CHECKPOINTS}"
            )
        if not isinstance(self.status, str) or self.status not in {"pass", "fail"}:
            raise PlacementInputError(f"unknown canary status {self.status!r}")
        tolerance = _int(self.tolerance_bytes, "tolerance_bytes")
        try:
            gpus = tuple(self.gpus)
        except TypeError as exc:
            raise PlacementInputError("canary result telemetry must be iterable") from exc
        if len(gpus) not in {1, 2}:
            raise PlacementInputError("canary result requires telemetry for exactly 1 or 2 GPUs")
        if any(not isinstance(item, GPUCanaryTelemetry) for item in gpus):
            raise PlacementInputError("canary result telemetry must be GPUCanaryTelemetry values")
        if tuple(item.rank for item in gpus) != tuple(range(len(gpus))):
            raise PlacementInputError("canary telemetry ranks must be contiguous and start at zero")
        if len({item.key for item in gpus}) != len(gpus):
            raise PlacementInputError("canary telemetry GPU identities must be unique")
        expected_status = "pass" if all(item.status == "pass" for item in gpus) else "fail"
        if self.status != expected_status:
            raise PlacementInputError("canary aggregate status does not match child statuses")
        try:
            reasons = tuple(self.reasons)
        except TypeError as exc:
            raise PlacementInputError("canary result reasons must be iterable") from exc
        if any(not isinstance(reason, str) for reason in reasons):
            raise PlacementInputError("canary result reasons must be strings")
        if (self.status == "pass") != (not reasons):
            raise PlacementInputError("canary result status must agree with reasons")
        object.__setattr__(self, "gpus", gpus)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "tolerance_bytes", tolerance)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLACEMENT_SCHEMA_VERSION,
            "checkpoint": self.checkpoint,
            "status": self.status,
            "tolerance_bytes": self.tolerance_bytes,
            "gpus": [item.as_dict() for item in self.gpus],
            "reasons": list(self.reasons),
        }

    to_dict = as_dict


def evaluate_canary(
    plan: PlacementPlan,
    *,
    checkpoint: str,
    observations: Iterable[PlacementObservation],
    tolerance_bytes: int = 0,
) -> CanaryResult:
    """Compare planned and observed state and fail closed on any unsafe signal.

    The comparison is deterministic and allocation-free.  Allocator ``allocated`` is compared
    with the live resident demand (non-QSA plus persistent QSA), while allocator high-water is
    compared with live demand plus the transient QSA peak.  Both counters require bounded
    absolute agreement: under-materialized live state is unsafe just as over-materialized state
    is, while high-water includes the transient QSA peak.
    Reserved bytes remain a separate allocator-consistency signal and are not treated as live
    demand.  ``tolerance_bytes`` applies per category and to those two comparisons; reserve,
    allocator consistency, and explicit fallback/spill signals are always enforced regardless
    of category tolerance.
    """
    if not isinstance(plan, PlacementPlan):
        raise PlacementInputError("plan must be a PlacementPlan")
    if checkpoint not in _CHECKPOINTS:
        raise PlacementInputError(
            f"unknown checkpoint {checkpoint!r}; expected one of {PLACEMENT_CHECKPOINTS}"
        )
    tolerance = _int(tolerance_bytes, "tolerance_bytes")
    if tolerance > plan.safety_reserve_bytes:
        raise PlacementInputError("tolerance_bytes must not exceed plan safety reserve")
    values = tuple(observations)
    by_rank: dict[int, PlacementObservation] = {}
    for observation in values:
        if not isinstance(observation, PlacementObservation):
            raise PlacementInputError("observations must contain PlacementObservation values")
        if observation.rank in by_rank:
            raise PlacementInputError(f"duplicate observation rank {observation.rank}")
        by_rank[observation.rank] = observation
    expected_ranks = set(range(plan.gpu_count))
    if set(by_rank) != expected_ranks:
        missing = sorted(expected_ranks - set(by_rank))
        unknown = sorted(set(by_rank) - expected_ranks)
        details = []
        if missing:
            details.append(f"missing ranks {missing}")
        if unknown:
            details.append(f"unknown ranks {unknown}")
        raise PlacementInputError("canary observations are incomplete: " + ", ".join(details))

    telemetry: list[GPUCanaryTelemetry] = []
    all_reasons: list[str] = []
    for expected in plan.gpus:
        observed = by_rank[expected.rank]
        reasons: list[str] = list(expected.reasons)
        if observed.gpu_uuid != expected.gpu_uuid:
            reasons.append(
                f"gpu-identity-mismatch planned={expected.gpu_uuid!r} "
                f"observed={observed.gpu_uuid!r}"
            )
        if observed.driver_total_bytes != expected.capacity_bytes:
            reasons.append(
                f"driver-total-mismatch planned={expected.capacity_bytes} "
                f"observed={observed.driver_total_bytes}"
            )
        if observed.allocator_allocated_bytes > observed.allocator_reserved_bytes:
            reasons.append("allocator-allocated-exceeds-reserved")
        if observed.allocator_reserved_bytes > observed.driver_total_bytes:
            reasons.append("reserved-bytes-exceed-capacity")
        if observed.driver_free_bytes > observed.driver_total_bytes:
            reasons.append("free-bytes-exceed-capacity")
        if observed.allocator_high_water_bytes < observed.allocator_allocated_bytes:
            reasons.append("allocator-high-water-below-allocated")
        if observed.allocator_high_water_bytes > observed.driver_total_bytes:
            reasons.append("allocator-high-water-exceeds-capacity")
        if (
            observed.allocator_reserved_bytes + observed.driver_free_bytes
            > observed.driver_total_bytes
        ):
            reasons.append("allocation-reserved-free-inconsistency")
        expected_live = expected.live_required_bytes
        expected_peak = expected.peak_required_bytes
        if abs(observed.allocator_allocated_bytes - expected_live) > tolerance:
            relation = "exceeds" if observed.allocator_allocated_bytes > expected_live else "below"
            reasons.append(
                f"allocator-live-{relation}-planned "
                f"planned={expected_live} observed={observed.allocator_allocated_bytes} "
                f"tolerance={tolerance}"
            )
        if abs(observed.allocator_high_water_bytes - expected_peak) > tolerance:
            relation = "exceeds" if observed.allocator_high_water_bytes > expected_peak else "below"
            reasons.append(
                f"allocator-high-water-{relation}-planned "
                f"planned={expected_peak} observed={observed.allocator_high_water_bytes} "
                f"tolerance={tolerance}"
            )
        if observed.driver_free_bytes < plan.safety_reserve_bytes:
            reasons.append(
                f"insufficient-reserve free={observed.driver_free_bytes} "
                f"reserve={plan.safety_reserve_bytes}"
            )
        for category in PLACEMENT_CATEGORIES:
            delta = abs(observed.categories[category] - expected.categories[category])
            if delta > tolerance:
                reasons.append(
                    f"planned-observed-mismatch:{category} delta={delta} tolerance={tolerance}"
                )
        if observed.managed_memory:
            reasons.append("managed-memory")
        if observed.host_spill:
            reasons.append("host-spill")
        if observed.fallback:
            reasons.append("fallback")
        if observed.allocation_retries:
            reasons.append(f"allocation-retries={observed.allocation_retries}")
        if observed.allocation_failures:
            reasons.append(f"allocation-failures={observed.allocation_failures}")
        if observed.cache_overcommit:
            reasons.append("cache-overcommit")
        if observed.unplanned_placement:
            reasons.append("unplanned-placement")
        if observed.retained_workspace_growth_bytes > tolerance:
            reasons.append(
                "retained-workspace-growth="
                f"{observed.retained_workspace_growth_bytes} tolerance={tolerance}"
            )
        status = "pass" if not reasons else "fail"
        headroom = observed.driver_free_bytes - plan.safety_reserve_bytes
        deficit = max(0, -headroom)
        item = GPUCanaryTelemetry(
            expected.rank,
            expected.gpu_uuid,
            status,
            expected.required_bytes,
            expected.non_qsa_required_bytes,
            expected.qsa_persistent_bytes,
            expected.qsa_transient_high_water_bytes,
            expected.qsa_required_bytes,
            observed.driver_free_bytes,
            headroom,
            deficit,
            observed.driver_total_bytes,
            observed.driver_free_bytes,
            observed.allocator_allocated_bytes,
            observed.allocator_reserved_bytes,
            observed.allocator_high_water_bytes,
            expected.categories,
            observed.categories,
            tuple(reasons),
        )
        telemetry.append(item)
        all_reasons.extend(f"gpu{expected.rank}:{reason}" for reason in reasons)
    status = "pass" if not all_reasons else "fail"
    return CanaryResult(checkpoint, status, tuple(telemetry), tuple(all_reasons), tolerance)


run_startup_canary = evaluate_canary
evaluate_placement_canary = evaluate_canary
CanaryObservation = PlacementObservation


@dataclass(frozen=True, slots=True)
class BackoffProfile:
    """One ordered, observable startup profile for the bounded backoff state machine."""

    name: str
    cache_slots: int
    context_tokens: int
    batch_size: int
    gpu_placement: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise PlacementInputError("backoff profile name must be non-empty")
        object.__setattr__(self, "cache_slots", _int(self.cache_slots, "cache_slots"))
        object.__setattr__(self, "context_tokens", _positive(self.context_tokens, "context_tokens"))
        object.__setattr__(self, "batch_size", _positive(self.batch_size, "batch_size"))
        _strict_bool(self.gpu_placement, "gpu_placement")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLACEMENT_SCHEMA_VERSION,
            "name": self.name,
            "cache_slots": self.cache_slots,
            "context_tokens": self.context_tokens,
            "batch_size": self.batch_size,
            "gpu_placement": self.gpu_placement,
        }


@dataclass(frozen=True, slots=True)
class BackoffDecision:
    """One state transition, suitable for machine-readable startup telemetry."""

    status: str
    profile: BackoffProfile
    index: int
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or self.status not in {
            "pending",
            "safe",
            "backoff",
            "fail-readiness",
        }:
            raise PlacementInputError(f"unknown backoff decision status {self.status!r}")
        if not isinstance(self.profile, BackoffProfile):
            raise PlacementInputError("backoff decision profile must be a BackoffProfile")
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise PlacementInputError("backoff decision index must be a non-negative integer")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise PlacementInputError("backoff decision reason must be non-empty text")
        object.__setattr__(self, "reason", self.reason.strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLACEMENT_SCHEMA_VERSION,
            "status": self.status,
            "profile": self.profile.as_dict(),
            "index": self.index,
            "reason": self.reason,
        }


class BackoffStateMachine:
    """Bounded ordered backoff with pending, safe, and fail-readiness states."""

    __slots__ = ("_index", "_passed_checkpoints", "_profiles", "_status")

    def __init__(self, profiles: Sequence[BackoffProfile]) -> None:
        values = tuple(profiles)
        if not values:
            raise PlacementInputError("backoff requires at least one profile")
        if any(not isinstance(profile, BackoffProfile) for profile in values):
            raise PlacementInputError("backoff profiles must be BackoffProfile values")
        names = [profile.name for profile in values]
        if len(names) != len(set(names)):
            raise PlacementInputError("backoff profile names must be unique")
        for previous, current in pairwise(values):
            if (
                current.cache_slots > previous.cache_slots
                or current.context_tokens > previous.context_tokens
                or current.batch_size > previous.batch_size
                or (not previous.gpu_placement and current.gpu_placement)
            ):
                raise PlacementInputError(
                    "backoff profiles must be ordered monotonically toward less GPU demand"
                )
            if (
                current.cache_slots == previous.cache_slots
                and current.context_tokens == previous.context_tokens
                and current.batch_size == previous.batch_size
                and current.gpu_placement == previous.gpu_placement
            ):
                raise PlacementInputError(
                    "backoff profiles require strict progress between candidates"
                )
        self._profiles = values
        self._index = 0
        self._status = "pending"
        self._passed_checkpoints = frozenset()

    @property
    def profiles(self) -> tuple[BackoffProfile, ...]:
        return self._profiles

    @property
    def index(self) -> int:
        return self._index

    @property
    def profile(self) -> BackoffProfile:
        return self._profiles[self._index]

    @property
    def ready(self) -> bool:
        return self._status == "safe"

    @property
    def status(self) -> str:
        return self._status

    @property
    def passed_checkpoints(self) -> tuple[str, ...]:
        return tuple(
            checkpoint
            for checkpoint in PLACEMENT_CHECKPOINTS
            if checkpoint in self._passed_checkpoints
        )

    def observe(self, result: CanaryResult) -> BackoffDecision:
        """Advance on failure and mark a profile safe after both readiness checkpoints pass.

        A pass is tracked only for the current profile.  Safe is checkpoint-local rather than
        terminal: a later failed checkpoint returns to pending and advances, which is required
        when post-load passes but large-prefill fails.
        """
        if not isinstance(result, CanaryResult):
            raise PlacementInputError("result must be a CanaryResult")
        if result.status == "pass" and self._status != "fail-readiness":
            self._passed_checkpoints = frozenset((*self._passed_checkpoints, result.checkpoint))
            if _READINESS_CHECKPOINTS.issubset(self._passed_checkpoints):
                self._status = "safe"
                return BackoffDecision("safe", self.profile, self._index, "canary-pass")
            self._status = "pending"
            return BackoffDecision(
                "pending",
                self.profile,
                self._index,
                f"checkpoint-pass:{result.checkpoint}",
            )
        if result.status == "fail" and self._index + 1 < len(self._profiles):
            self._index += 1
            self._status = "pending"
            self._passed_checkpoints = frozenset()
            return BackoffDecision("backoff", self.profile, self._index, "canary-fail")
        self._status = "fail-readiness"
        self._passed_checkpoints = frozenset()
        return BackoffDecision("fail-readiness", self.profile, self._index, "no-safe-profile")


__all__ = [
    "MAX_PLACEMENT_BYTES",
    "PLACEMENT_CATEGORIES",
    "PLACEMENT_CHECKPOINTS",
    "PLACEMENT_SCHEMA_VERSION",
    "QSA_PERSISTENT_CATEGORIES",
    "QSA_TRANSIENT_CATEGORIES",
    "BackoffDecision",
    "BackoffProfile",
    "BackoffStateMachine",
    "CanaryObservation",
    "CanaryResult",
    "GPUCanaryTelemetry",
    "GPUPlacementInput",
    "GPUPlacementPlan",
    "GpuPlacementInput",
    "PlacementCapacityError",
    "PlacementInputError",
    "PlacementObservation",
    "PlacementPlan",
    "PlacementPlanInput",
    "PlacementPlannerError",
    "calculate_placement_plan",
    "evaluate_canary",
    "evaluate_placement_canary",
    "plan_placement",
    "run_startup_canary",
]
