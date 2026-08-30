"""Host-bank allocation and residency primitives shared by expert-load paths.

Source mappings are pageable by default.  Pinning is an explicit, preflighted
policy choice so a complete model cannot consume the process pin quota by
accident.

Two ideas the parallel read of the original checkpoint and FTW (read a repacked
contiguous cache) paths both rely on:

* **pin-after-fill** -- allocate the bank as a *lazy* anonymous ``mmap`` (no pages
  resident, instant), fill it with real data, and only THEN ``cudaHostRegister`` it.
  Registering already-resident pages just page-locks them; registering a lazy mmap first
  faults+zero-fills every page (~137 GiB -> ~47 s for DSV4) and that zero-fill is then
  immediately overwritten by the read. So pin-after-fill removes a whole redundant pass.
* **chunked multi-threaded O_DIRECT** -- DMA straight from disk into the (page-aligned)
  bank, bypassing the page cache, with many concurrent ``preadv`` on one fd (scales to the
  device's queue-depth ceiling even for a single file).

The mmaps are held for the process lifetime (the banks live as long as the offload cache).
"""

from __future__ import annotations

import contextlib
import ctypes
import math
import mmap
import os
import queue
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

import torch

from freetoken.moe.host_memory import SwapProbe, probe_swap
from freetoken.moe.numa_memory import (
    NumaPlacementController,
    NumaSyscallBackend,
    resolve_numa_placement,
)
from freetoken.utils import init_logger

logger = init_logger(__name__)

_BLK = 4096  # O_DIRECT alignment (page size)


class HostResidency(str, Enum):
    """Residency class of a host bank layer.

    Only PINNED (cudaHostRegister'd) memory can feed the GPU movement paths; LOCKED (mlock'd, no device address) and PAGEABLE layers must decode on the CPU executor.
    The non-pinned classes exist for hosts that cap CUDA pin quota (WSL/WDDM: ~half of RAM).
    """

    PINNED = "pinned"
    LOCKED = "locked"
    PAGEABLE = "pageable"


class HostBankStrategy(str, Enum):
    """How a host expert bank is exposed to the execution paths.

    ``PAGEABLE`` keeps the source mapping in ordinary host memory.  ``PINNED``
    selectively registers complete source layers after they have been filled.
    ``BOUNDED_STAGING`` keeps the source pageable and uses a separately bounded
    pinned ring for transfers.  The default is deliberately pageable.
    """

    PAGEABLE = "pageable"
    PAGEABLE_DIRECT = "pageable"
    PINNED = "pinned"
    SELECTIVE_PINNED = "pinned"
    BOUNDED_STAGING = "bounded-staging"


class NumaPolicy(str, Enum):
    """Placement mode; it is telemetry-only unless enforcement is explicitly enabled."""

    PREFERRED = "preferred"
    BIND = "bind"
    INTERLEAVE = "interleave"


def _normalize_strategy(strategy: HostBankStrategy | str) -> HostBankStrategy:
    try:
        if isinstance(strategy, HostBankStrategy):
            return strategy
        aliases = {
            "pageable-direct": HostBankStrategy.PAGEABLE,
            "selective-pinned": HostBankStrategy.PINNED,
            "bounded_staging": HostBankStrategy.BOUNDED_STAGING,
        }
        if strategy in aliases:
            return aliases[strategy]
        return HostBankStrategy(strategy)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown host-bank strategy {strategy!r}") from exc


def _aligned_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    """Return the allocation size used by :class:`HostBank`, including page rounding."""
    if any(int(size) < 0 for size in shape):
        raise ValueError(f"bank shape must be non-negative, got {shape}")
    raw = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    return max(_BLK, ((raw + _BLK - 1) // _BLK) * _BLK)


@dataclass(frozen=True)
class HostBankPlan:
    """Validated pre-load decision and byte accounting for a host-bank set."""

    strategy: HostBankStrategy
    source_bytes: int
    layer_bytes: tuple[int, ...]
    layer_residency: tuple[str, ...]
    pinned_bytes: int
    staging_bytes: int
    staging_slots: int
    numa_policy: NumaPolicy
    numa_node: int | None
    numa_target_nodes: tuple[int, ...] = ()
    numa_status: str = "not-requested"

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy.value,
            "source_bytes": self.source_bytes,
            "layer_bytes": list(self.layer_bytes),
            "layer_residency": list(self.layer_residency),
            "pinned_bytes": self.pinned_bytes,
            "staging_bytes": self.staging_bytes,
            "staging_slots": self.staging_slots,
            "numa_policy": self.numa_policy.value,
            "numa_node": self.numa_node,
            "numa_target_nodes": list(self.numa_target_nodes),
            "numa_status": self.numa_status,
            "numa_requested": self.numa_status != "not-requested",
            "numa_applied": self.numa_status == "applied",
            "numa_fallback": self.numa_status == "fallback",
        }


@dataclass
class HostBankAccounting:
    """Observable host-bank selection and budget accounting."""

    strategy: HostBankStrategy
    source_bytes: int = 0
    # Requested values are the pre-load reservation; applied values are updated
    # only after a source bank has actually settled.
    pinned_bytes: int = 0
    staging_bytes: int = 0
    applied_pinned_bytes: int = 0
    applied_staging_bytes: int = 0
    applied_layers: tuple[str, ...] = ()
    layer_bytes: tuple[int, ...] = ()
    layers: tuple[str, ...] = ()
    numa_policy: NumaPolicy = NumaPolicy.PREFERRED
    numa_node: int | None = None
    require_no_swap: bool = False
    swap_status: str = "not-requested"
    swap_probe_source: str | None = None
    swap_probe_errors: tuple[str, ...] = ()
    vm_swap_bytes: int | None = None
    swap_total_bytes: int | None = None
    swap_free_bytes: int | None = None
    active_swap_devices: tuple[str, ...] = ()
    numa_enforced: bool = False
    numa_status: str = "not-requested"
    numa_target_nodes: tuple[int, ...] = ()
    numa_allowed_nodes: tuple[int, ...] = ()
    numa_applied_mappings: int = 0
    numa_fallback_reason: str | None = None
    numa_errors: tuple[str, ...] = ()
    numa_sample_status: str = "not-requested"
    numa_sample_counts: tuple[tuple[int, int], ...] = ()
    numa_sample_unknown: int = 0
    numa_sampled_pages: int = 0
    numa_sample_target_pages: int = 0
    numa_sample_off_target_pages: int = 0
    numa_sample_off_target_counts: tuple[tuple[int, int], ...] = ()
    numa_sample_target_fraction: float | None = None
    numa_sample_placement_match: bool | None = None
    numa_sample_errors: tuple[str, ...] = ()
    numa_sample_banks: tuple[dict[str, object], ...] = ()
    numa_sample_error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy.value,
            "source_bytes": self.source_bytes,
            "pinned_bytes": self.pinned_bytes,
            "staging_bytes": self.staging_bytes,
            "applied_pinned_bytes": self.applied_pinned_bytes,
            "applied_staging_bytes": self.applied_staging_bytes,
            "layer_bytes": list(self.layer_bytes),
            "layers": list(self.layers),
            "applied_layers": list(self.applied_layers),
            "numa_policy": self.numa_policy.value,
            "numa_node": self.numa_node,
            "require_no_swap": self.require_no_swap,
            "swap_status": self.swap_status,
            "swap_probe_source": self.swap_probe_source,
            "swap_probe_errors": list(self.swap_probe_errors),
            "vm_swap_bytes": self.vm_swap_bytes,
            "process_vm_swap_bytes": self.vm_swap_bytes,
            "swap_total_bytes": self.swap_total_bytes,
            "swap_free_bytes": self.swap_free_bytes,
            "active_swap_devices": list(self.active_swap_devices),
            "no_swap_observed": (
                True
                if self.swap_status == "clear"
                else False
                if self.swap_status in ("swap-active", "process-swapped")
                else None
            ),
            "numa_enforced": self.numa_enforced,
            "numa_requested": self.numa_enforced,
            "numa_status": self.numa_status,
            "numa_applied": self.numa_status == "applied",
            "numa_fallback": self.numa_status == "fallback",
            "numa_target_nodes": list(self.numa_target_nodes),
            "numa_allowed_nodes": list(self.numa_allowed_nodes),
            "numa_applied_mappings": self.numa_applied_mappings,
            "numa_fallback_reason": self.numa_fallback_reason,
            "numa_errors": list(self.numa_errors),
            "numa_sample_status": self.numa_sample_status,
            "numa_sample_counts": {str(node): count for node, count in self.numa_sample_counts},
            "numa_sample_unknown": self.numa_sample_unknown,
            "numa_sampled_pages": self.numa_sampled_pages,
            "numa_sampled_total": self.numa_sampled_pages,
            "numa_sample_target_pages": self.numa_sample_target_pages,
            "numa_sample_off_target_pages": self.numa_sample_off_target_pages,
            "numa_sample_off_target_counts": {
                str(node): count for node, count in self.numa_sample_off_target_counts
            },
            "numa_sample_target_fraction": self.numa_sample_target_fraction,
            "numa_sample_placement_match": self.numa_sample_placement_match,
            "numa_sample_errors": list(self.numa_sample_errors),
            "numa_sample_banks": list(self.numa_sample_banks),
            "numa_sample_error": self.numa_sample_error,
        }


@dataclass
class HostBankPolicy:
    """Fail-closed host-bank residency and budget policy.

    ``prepare`` must run before allocation or checkpoint reads.  A pageable
    policy is the safe default.  A pinned policy needs an explicit finite byte
    limit.  ``selected_layers`` limits registration to those layer IDs and
    leaves the other source layers pageable.
    """

    strategy: HostBankStrategy | str = HostBankStrategy.PAGEABLE
    max_pinned_bytes: int | None = 0
    max_staging_bytes: int | None = 0
    staging_bytes: int = 0
    staging_slots: int = 2
    selected_layers: tuple[int, ...] | None = None
    numa_policy: NumaPolicy | str = NumaPolicy.PREFERRED
    numa_node: int | None = None
    require_no_swap: bool = False
    swap_probe_reader: Callable[[str], str] | None = field(default=None, repr=False, compare=False)
    enforce_numa_placement: bool = False
    numa_backend: object | None = field(default=None, repr=False, compare=False)
    numa_reader: Callable[[str], str] | None = field(default=None, repr=False, compare=False)
    sample_numa_residency: bool = False
    numa_sample_stride: int = 4096
    numa_sample_max_pages: int = 64
    accounting: HostBankAccounting = field(init=False)
    _plan: HostBankPlan | None = field(default=None, init=False, repr=False)
    _swap_probe: SwapProbe | None = field(default=None, init=False, repr=False)
    _numa_controller: NumaPlacementController | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_configuration()
        self.accounting = HostBankAccounting(
            strategy=self.strategy,
            numa_policy=self.numa_policy,
            numa_node=self.numa_node,
            require_no_swap=self.require_no_swap,
        )

    def _validate_configuration(self) -> None:
        """Normalize and validate mutable configuration before every preflight."""
        self.strategy = _normalize_strategy(self.strategy)
        if not isinstance(self.require_no_swap, bool):
            raise ValueError("require_no_swap must be a boolean")
        if not isinstance(self.enforce_numa_placement, bool):
            raise ValueError("enforce_numa_placement must be a boolean")
        if not isinstance(self.sample_numa_residency, bool):
            raise ValueError("sample_numa_residency must be a boolean")
        if self.sample_numa_residency and not self.enforce_numa_placement:
            raise ValueError("sample_numa_residency requires enforce_numa_placement=True")
        for name, value in (
            ("numa_sample_stride", self.numa_sample_stride),
            ("numa_sample_max_pages", self.numa_sample_max_pages),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        try:
            self.numa_policy = (
                self.numa_policy
                if isinstance(self.numa_policy, NumaPolicy)
                else NumaPolicy(self.numa_policy)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown NUMA policy {self.numa_policy!r}") from exc
        for name, value in (
            ("max_pinned_bytes", self.max_pinned_bytes),
            ("max_staging_bytes", self.max_staging_bytes),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if (
            isinstance(self.staging_bytes, bool)
            or not isinstance(self.staging_bytes, int)
            or self.staging_bytes < 0
        ):
            raise ValueError("staging_bytes must be a non-negative integer")
        if (
            isinstance(self.staging_slots, bool)
            or not isinstance(self.staging_slots, int)
            or self.staging_slots <= 0
        ):
            raise ValueError("staging_slots must be a positive integer")
        if self.numa_node is not None and (
            isinstance(self.numa_node, bool)
            or not isinstance(self.numa_node, int)
            or self.numa_node < 0
        ):
            raise ValueError("numa_node must be a non-negative integer or None")
        if self.selected_layers is not None:
            self.selected_layers = tuple(sorted(set(self.selected_layers)))
            if any(
                isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
                for layer in self.selected_layers
            ):
                raise ValueError("selected_layers must contain non-negative integer IDs")

    def _record_swap_probe(self, probe: SwapProbe) -> None:
        self._swap_probe = probe
        self.accounting.require_no_swap = self.require_no_swap
        self.accounting.swap_status = probe.status
        self.accounting.swap_probe_source = probe.source
        self.accounting.swap_probe_errors = probe.errors
        self.accounting.vm_swap_bytes = probe.vm_swap_bytes
        self.accounting.swap_total_bytes = probe.swap_total_bytes
        self.accounting.swap_free_bytes = probe.swap_free_bytes
        self.accounting.active_swap_devices = probe.active_swap_devices

    def _reset_preflight_state(self) -> None:
        """Discard a prior reservation before any new validation or probe."""
        self._plan = None
        self._swap_probe = None
        self._numa_controller = None
        self.accounting = HostBankAccounting(
            strategy=HostBankStrategy.PAGEABLE,
            require_no_swap=self.require_no_swap,
        )

    def _prepare_numa(self) -> None:
        if not self.enforce_numa_placement:
            self._numa_controller = None
            return
        plan = resolve_numa_placement(
            self.numa_policy,
            self.numa_node,
            enforce=True,
            read_text=self.numa_reader,
        )
        self._numa_controller = NumaPlacementController(
            plan, self.numa_backend if self.numa_backend is not None else NumaSyscallBackend()
        )

    def _sync_numa_accounting(self) -> None:
        controller = self._numa_controller
        if controller is None:
            self.accounting.numa_enforced = False
            self.accounting.numa_status = "not-requested"
            self.accounting.numa_target_nodes = ()
            self.accounting.numa_allowed_nodes = ()
            self.accounting.numa_applied_mappings = 0
            self.accounting.numa_fallback_reason = None
            self.accounting.numa_errors = ()
            self.accounting.numa_sample_status = "not-requested"
            self.accounting.numa_sample_counts = ()
            self.accounting.numa_sample_unknown = 0
            self.accounting.numa_sampled_pages = 0
            self.accounting.numa_sample_target_pages = 0
            self.accounting.numa_sample_off_target_pages = 0
            self.accounting.numa_sample_off_target_counts = ()
            self.accounting.numa_sample_target_fraction = None
            self.accounting.numa_sample_placement_match = None
            self.accounting.numa_sample_errors = ()
            self.accounting.numa_sample_banks = ()
            self.accounting.numa_sample_error = None
            return
        telemetry = controller.telemetry()
        self.accounting.numa_enforced = bool(telemetry["requested"])
        self.accounting.numa_status = str(telemetry["status"])
        self.accounting.numa_target_nodes = tuple(telemetry["target_nodes"])
        self.accounting.numa_allowed_nodes = tuple(telemetry["allowed_nodes"])
        self.accounting.numa_applied_mappings = int(telemetry["applied_mappings"])
        self.accounting.numa_fallback_reason = telemetry["fallback_reason"]
        self.accounting.numa_errors = tuple(telemetry["errors"])
        sample = telemetry["sample"]
        self.accounting.numa_sample_status = str(sample["status"])
        self.accounting.numa_sample_counts = tuple(
            (int(node), int(count)) for node, count in sample["counts"].items()
        )
        self.accounting.numa_sample_unknown = int(sample["unknown"])
        self.accounting.numa_sampled_pages = int(sample["sampled_pages"])
        self.accounting.numa_sample_target_pages = int(sample["target_pages"])
        self.accounting.numa_sample_off_target_pages = int(sample["off_target_pages"])
        self.accounting.numa_sample_off_target_counts = tuple(
            (int(node), int(count)) for node, count in sample["off_target_counts"].items()
        )
        fraction = sample["target_fraction"]
        self.accounting.numa_sample_target_fraction = None if fraction is None else float(fraction)
        match = sample["placement_match"]
        self.accounting.numa_sample_placement_match = None if match is None else bool(match)
        self.accounting.numa_sample_errors = tuple(sample["errors"])
        self.accounting.numa_sample_banks = tuple(telemetry["sample_banks"])
        self.accounting.numa_sample_error = sample["error"]

    @property
    def numa_placement(self) -> NumaPlacementController | None:
        """The prepared placement owner, or ``None`` for the default no-op path."""
        return self._numa_controller

    def refresh_numa_accounting(self) -> None:
        self._sync_numa_accounting()

    def sample_numa_bank(self, bank: HostBank) -> None:
        if self.sample_numa_residency and self._numa_controller is not None:
            self._numa_controller.sample(
                bank.addr,
                bank.allocated_nbytes,
                stride=self.numa_sample_stride,
                max_pages=self.numa_sample_max_pages,
            )
            self._sync_numa_accounting()

    def preflight_swap(self, *, probe: SwapProbe | None = None) -> SwapProbe:
        """Observe swap state and enforce ``require_no_swap`` before allocation.

        The observation is procfs-only and point-in-time. It deliberately does
        not attempt to alter swap, lock pages, or infer swap state from an
        allocation attempt.
        """
        self._reset_preflight_state()
        self.validate_for_config()
        if probe is None:
            probe = probe_swap(read_text=self.swap_probe_reader)
        if not isinstance(probe, SwapProbe):
            raise ValueError("swap preflight requires a SwapProbe result")
        self._record_swap_probe(probe)
        if self.require_no_swap and not probe.no_swap:
            detail = "; ".join(probe.errors) or probe.status
            raise ValueError(f"require_no_swap preflight failed: {probe.status}: {detail}")
        return probe

    @property
    def swap_probe(self) -> SwapProbe | None:
        """The latest read-only swap observation, if preparation reached it."""
        return self._swap_probe

    def validate_for_config(self) -> None:
        """Validate strategy-level requirements before a model is loaded."""
        self._validate_configuration()
        if self.strategy is HostBankStrategy.PINNED and os.environ.get(
            "FREETOKEN_SKIP_BANK_PIN", ""
        ).strip().lower() in ("1", "true", "yes", "on"):
            raise ValueError(
                "pinned host-bank policy cannot run with FREETOKEN_SKIP_BANK_PIN enabled"
            )
        if self.strategy is HostBankStrategy.PINNED and self.max_pinned_bytes is None:
            raise ValueError("pinned strategy requires finite max_pinned_bytes")
        if self.strategy is HostBankStrategy.BOUNDED_STAGING:
            if self.staging_bytes <= 0:
                raise ValueError("bounded-staging requires staging_bytes > 0")
            if self.max_staging_bytes is None:
                raise ValueError("bounded-staging requires finite max_staging_bytes")
            if self.staging_bytes % self.staging_slots:
                raise ValueError(
                    "staging_bytes must divide evenly across staging_slots "
                    f"({self.staging_bytes} / {self.staging_slots})"
                )

    def prepare(
        self,
        specs: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
        num_layers: int,
    ) -> HostBankPlan:
        """Validate all limits and produce the per-layer decision before allocation."""
        self._reset_preflight_state()
        # Policy fields remain public for compatibility, so callers may mutate
        # them after construction.  Revalidate before deriving any allocation
        # decision to keep that mutation fail-closed.
        self.validate_for_config()
        swap_probe = self.preflight_swap() if self.require_no_swap else None
        self.accounting = HostBankAccounting(
            strategy=self.strategy,
            numa_policy=self.numa_policy,
            numa_node=self.numa_node,
            require_no_swap=self.require_no_swap,
        )
        if swap_probe is not None:
            self._record_swap_probe(swap_probe)
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers}")
        if not specs:
            raise ValueError("host-bank policy requires at least one bank spec")
        if self.selected_layers is not None and any(
            layer >= num_layers for layer in self.selected_layers
        ):
            raise ValueError(
                f"selected_layers {self.selected_layers} contain an ID outside [0, {num_layers})"
            )
        per_layer = [0] * num_layers
        for name, spec in specs.items():
            try:
                shape, dtype = spec
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid bank spec for {name!r}: {spec!r}") from exc
            bytes_per_bank = _aligned_nbytes(tuple(shape), dtype)
            for layer in range(num_layers):
                per_layer[layer] += bytes_per_bank
        return self._prepare_layer_bytes(per_layer, swap_probe=swap_probe)

    def prepare_layer_bytes(
        self,
        layer_bytes: Sequence[int],
        *,
        source_bytes: int | None = None,
        swap_probe: SwapProbe | None = None,
    ) -> HostBankPlan:
        """Prepare an exact per-layer allocation plan supplied by a checkpoint index."""
        self._reset_preflight_state()
        self.validate_for_config()
        if self.require_no_swap:
            swap_probe = self.preflight_swap(probe=swap_probe)
        elif swap_probe is not None:
            self._record_swap_probe(swap_probe)
        per_layer = list(layer_bytes)
        if not per_layer or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in per_layer
        ):
            raise ValueError("layer_bytes must contain positive integers")
        if source_bytes is None:
            source_bytes = sum(per_layer)
        if isinstance(source_bytes, bool) or not isinstance(source_bytes, int) or source_bytes < 0:
            raise ValueError("source_bytes must be a non-negative integer")
        return self._prepare_layer_bytes(
            per_layer, source_bytes=source_bytes, swap_probe=swap_probe
        )

    def _prepare_layer_bytes(
        self,
        per_layer: Sequence[int],
        *,
        source_bytes: int | None = None,
        swap_probe: SwapProbe | None,
    ) -> HostBankPlan:
        num_layers = len(per_layer)
        if source_bytes is None:
            source_bytes = sum(per_layer)
        if self.selected_layers is not None and any(
            layer >= num_layers for layer in self.selected_layers
        ):
            raise ValueError(
                f"selected_layers {self.selected_layers} contain an ID outside [0, {num_layers})"
            )
        selected = (
            set(range(num_layers)) if self.selected_layers is None else set(self.selected_layers)
        )
        labels = tuple(
            HostResidency.PINNED.value
            if self.strategy is HostBankStrategy.PINNED and layer in selected
            else HostResidency.PAGEABLE.value
            for layer in range(num_layers)
        )
        pinned_bytes = (
            sum(per_layer[layer] for layer in selected)
            if self.strategy is HostBankStrategy.PINNED
            else 0
        )
        staging_bytes = (
            self.staging_bytes if self.strategy is HostBankStrategy.BOUNDED_STAGING else 0
        )
        if self.strategy is HostBankStrategy.PINNED:
            if self.max_pinned_bytes is None:
                raise ValueError("pinned strategy requires finite max_pinned_bytes")
            if pinned_bytes > self.max_pinned_bytes:
                raise ValueError(
                    f"pinned host-bank budget {self.max_pinned_bytes} bytes is smaller "
                    f"than the selected {pinned_bytes} bytes"
                )
        if self.strategy is HostBankStrategy.BOUNDED_STAGING:
            if staging_bytes <= 0:
                raise ValueError("bounded-staging requires staging_bytes > 0")
            if staging_bytes % self.staging_slots:
                raise ValueError(
                    "staging_bytes must divide evenly across staging_slots "
                    f"({staging_bytes} / {self.staging_slots})"
                )
            if self.max_staging_bytes is None:
                raise ValueError("bounded-staging requires finite max_staging_bytes")
            if staging_bytes > self.max_staging_bytes:
                raise ValueError(
                    f"staging host-bank budget {self.max_staging_bytes} bytes is smaller "
                    f"than the configured {staging_bytes} bytes"
                )
        elif self.staging_bytes:
            raise ValueError("staging_bytes is only valid with the bounded-staging strategy")
        self._prepare_numa()
        plan = HostBankPlan(
            strategy=self.strategy,
            source_bytes=source_bytes,
            layer_bytes=tuple(per_layer),
            layer_residency=labels,
            pinned_bytes=pinned_bytes,
            staging_bytes=staging_bytes,
            staging_slots=self.staging_slots,
            numa_policy=self.numa_policy,
            numa_node=self.numa_node,
            numa_target_nodes=(
                self._numa_controller.plan.target_nodes if self._numa_controller is not None else ()
            ),
            numa_status=(
                self._numa_controller.status
                if self._numa_controller is not None
                else "not-requested"
            ),
        )
        self._plan = plan
        self.accounting = HostBankAccounting(
            strategy=self.strategy,
            source_bytes=source_bytes,
            pinned_bytes=pinned_bytes,
            staging_bytes=staging_bytes,
            layer_bytes=tuple(per_layer),
            layers=labels,
            numa_policy=self.numa_policy,
            numa_node=self.numa_node,
            require_no_swap=self.require_no_swap,
        )
        if swap_probe is not None:
            self._record_swap_probe(swap_probe)
        self._sync_numa_accounting()
        return plan

    @property
    def plan(self) -> HostBankPlan:
        if self._plan is None:
            raise RuntimeError("host-bank policy must be prepared before use")
        return self._plan

    def staging_ring(
        self,
        *,
        allocator: Callable[[int], object] | None = None,
    ) -> HostStagingRing:
        """Allocate the explicitly bounded staging ring after successful preflight."""
        plan = self.plan
        if plan.strategy is not HostBankStrategy.BOUNDED_STAGING:
            raise ValueError("staging_ring requires the bounded-staging strategy")
        ring = HostStagingRing(
            plan.staging_bytes,
            slots=plan.staging_slots,
            allocator=allocator,
        )
        self.accounting.applied_staging_bytes = plan.staging_bytes
        return ring

    def settle(self, banks: dict[str, HostBank | list[HostBank]]) -> None:
        """Apply the prepared source policy after banks have been filled."""
        pin_banks(banks, policy=self)
        self._sync_numa_accounting()


class HostStagingRing:
    """Fixed-size staging slots; no growth or per-transfer allocation is allowed."""

    def __init__(
        self,
        nbytes: int,
        *,
        slots: int = 2,
        allocator: Callable[[int], object] | None = None,
    ) -> None:
        if nbytes <= 0:
            raise ValueError("staging ring size must be positive")
        if slots <= 0 or nbytes % slots:
            raise ValueError("staging ring bytes must divide evenly across positive slots")
        self.nbytes = nbytes
        self.slots_count = slots
        self.slot_bytes = nbytes // slots
        self._closed = False
        if allocator is None:
            allocator = self._default_allocator
        allocated: list[object] = []
        current: object | None = None
        try:
            for _ in range(slots):
                current = allocator(self.slot_bytes)
                actual = getattr(current, "nbytes", None)
                if actual is None:
                    try:
                        actual = len(current)
                    except TypeError:
                        actual = self.slot_bytes
                if actual != self.slot_bytes:
                    raise ValueError(
                        f"staging allocator returned {actual} bytes, require exactly "
                        f"{self.slot_bytes}"
                    )
                allocated.append(current)
                current = None
        except BaseException:
            slots_to_close = ([] if current is None else [current]) + allocated
            for slot in slots_to_close:
                close = getattr(slot, "close", None)
                if close is not None:
                    try:
                        close()
                    except BaseException:
                        pass
            raise
        self.slots = tuple(allocated)

    @staticmethod
    def _default_allocator(size: int) -> object:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "bounded-staging requires CUDA pinned allocation; provide an allocator "
                "for host-only tests"
            )
        from freetoken.kernel.pinned import alloc_pinned_tensor

        return alloc_pinned_tensor(size, dtype=torch.uint8)

    def acquire(self, slot: int) -> object:
        if self._closed:
            raise RuntimeError("host staging ring is closed")
        if not 0 <= slot < self.slots_count:
            raise IndexError(f"staging slot {slot} outside [0, {self.slots_count})")
        return self.slots[slot]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        slots, self.slots = self.slots, ()
        first_error: BaseException | None = None
        for slot in slots:
            close = getattr(slot, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error

    def __enter__(self) -> HostStagingRing:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


_DEFAULT_CHUNK = 8 << 20

# Keep mmap objects alive until their owning bank is explicitly closed.
_LIVE_BUFFERS: list[mmap.mmap] = []


def _env_born_pinned() -> bool | None:
    """``FREETOKEN_BANK_CUDA_ALLOC`` tri-state: unset -> ``None`` (default applies), else the parsed boolean."""
    v = os.environ.get("FREETOKEN_BANK_CUDA_ALLOC", "").strip().lower()
    if not v:
        return None
    return v in ("1", "true", "yes", "on")


def born_pinned_default() -> bool:
    """Whether PINNED serving banks use cudaHostAlloc instead of mmap + register-after-fill.

    Off by default: registered mmaps already read at the PCIe roofline and lazy mmaps commit pages only on fill. ``FREETOKEN_BANK_CUDA_ALLOC`` overrides."""
    env = _env_born_pinned()
    if env is not None:
        return env
    return False


class HostBank:
    """A page-aligned host buffer + its torch view, page-locked on demand: allocate -> fill -> ``pin()``/``lock()``.

    * ``"mmap"`` (default) -- lazy anonymous mmap; pages materialize on fill, then ``pin()`` registers or ``lock()`` OS-locks it.
    * ``"cuda"`` -- cudaHostAlloc, born pinned+mapped; ``pin()``/``lock()``/``release()`` are no-ops and it never takes LOCKED. See :func:`born_pinned_default`.

    The buffer is rounded up to the O_DIRECT block; ``tensor`` views exactly ``nbytes``. ``backing=None`` follows ``FREETOKEN_BANK_CUDA_ALLOC``."""

    __slots__ = (
        "_backing",
        "_buf",
        "_closed",
        "_locked",
        "_pinned",
        "_registered",
        "addr",
        "nbytes",
        "tensor",
    )

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        backing: str | None = None,
        numa_placement: NumaPlacementController | None = None,
    ):
        if backing is None:
            plan = _requested_residency
            # a plan with non-pinned labels vetoes born-pinned: cudaHostAlloc spends the pin quota the plan exists to save
            born = _env_born_pinned() and (plan is None or not plan.has_unpinned)
            backing = "cuda" if born else "mmap"
        assert backing in ("mmap", "cuda"), backing
        self._backing = backing
        self._closed = False
        elsize = torch.empty((), dtype=dtype).element_size()
        self.nbytes = math.prod(shape) * elsize
        asize = max(_BLK, ((self.nbytes + _BLK - 1) // _BLK) * _BLK)
        if backing == "cuda":
            from freetoken.kernel.pinned import alloc_pinned_tensor

            # direct-IO readers need page alignment, but cudaHostAlloc only guarantees ~512 in practice
            # over-allocate one block and carve the aligned window; the numpy slice keeps the pinned storage alive via .base
            raw = alloc_pinned_tensor(asize + _BLK, dtype=torch.uint8)  # cudaMallocHost
            raw.zero_()  # keep the anonymous-mmap guarantee: unwritten regions stay zero
            off = (-raw.data_ptr()) % _BLK
            self._buf = raw.numpy()[off : off + asize]
            self.addr = raw.data_ptr() + off
            assert self.addr % _BLK == 0
            self._pinned = True  # born pinned+mapped; pin() is a no-op
            self._registered = False
        else:
            # Explicit flags keep the placement contract auditable: this is a
            # private anonymous writable mapping, not a file/shared mapping.
            self._buf = mmap.mmap(
                -1,
                asize,
                flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )  # lazy: address space only, no resident pages yet
            self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
            if numa_placement is not None:
                try:
                    numa_placement.apply(
                        self.addr,
                        asize,
                        private_anonymous=True,
                        before_touch=True,
                    )
                except BaseException:
                    self._buf.close()
                    raise
            _LIVE_BUFFERS.append(self._buf)
            self._pinned = False
            self._registered = False
        try:
            self.tensor = torch.frombuffer(
                self._buf, dtype=dtype, count=self.nbytes // elsize
            ).view(*shape)
        except BaseException:
            if backing == "mmap":
                try:
                    _LIVE_BUFFERS.remove(self._buf)
                except ValueError:
                    pass
                self._buf.close()
            raise
        self._locked = False

    @property
    def residency(self) -> HostResidency:
        if self._pinned:
            return HostResidency.PINNED
        if self._locked:
            return HostResidency.LOCKED
        return HostResidency.PAGEABLE

    @property
    def allocated_nbytes(self) -> int:
        """Page-rounded allocation size used by residency accounting."""
        return len(self._buf)

    def memoryview(self) -> memoryview:
        if self._closed:
            raise RuntimeError("host bank is closed")
        return memoryview(self._buf)

    def pin(self) -> None:
        """cudaHostRegister the (now-filled) buffer -- pin-after-fill.

        ``FREETOKEN_SKIP_BANK_PIN=1`` makes this a no-op for CPU-only tooling (the FTW converter); never set it when serving, the GPU paths need registered banks."""
        if self._closed:
            raise RuntimeError("host bank is closed")
        if self._pinned:
            return
        if os.environ.get("FREETOKEN_SKIP_BANK_PIN", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return
        from freetoken.kernel.pinned import host_register

        try:
            host_register(self.addr, len(self._buf))
        except RuntimeError as exc:
            raise RuntimeError(
                f"cudaHostRegister failed for {len(self._buf) / 2**30:.1f} GiB"
            ) from exc
        self._pinned = True
        self._registered = True

    def release(self) -> None:
        """Drop the resident pages; the address space stays valid, the contents become undefined.

        For buffers that are done being read (the converter). No-op for born-pinned banks: registered pages cannot be dropped."""
        if self._closed or self._pinned or self._locked:
            return
        self._buf.madvise(mmap.MADV_DONTNEED)

    def lock(self) -> None:
        """mlock the (now-filled) buffer: resident without CUDA pin quota, but no device address -- only the CPU executor can serve a locked layer.

        Lock after fill, or the lazy mmap faults+zero-fills every page. A failed lock (RLIMIT_MEMLOCK) warns once and leaves the bank PAGEABLE, which every consumer treats the same."""
        if self._closed:
            raise RuntimeError("host bank is closed")
        if self._locked or self._pinned:  # cudaHostRegister already page-locks
            return
        global _os_lock_failed
        if _os_lock_failed:
            return  # the quota is exhausted for good; skip the syscall spam
        try:
            _os_lock(self.addr, len(self._buf))
        except (OSError, ImportError) as exc:
            _os_lock_failed = True
            logger.warning(f"bank lock failed; leaving this and later banks pageable: {exc}")
            return
        self._locked = True

    def close(self) -> None:
        """Release a bank's registration/lock and close its anonymous mapping."""
        if self._closed:
            return
        if self._registered:
            try:
                from freetoken.kernel.pinned import host_unregister

                host_unregister(self.addr, len(self._buf))
            except (ImportError, AttributeError):
                # Older installed extensions do not expose unregister.  Keep
                # the mapping alive rather than leaving CUDA with a dangling
                # registered range.
                logger.warning("host-bank unregister is unavailable; retaining mapping")
                return
            self._registered = False
            self._pinned = False
        if self._locked:
            _os_unlock(self.addr, len(self._buf))
            self._locked = False
        buf = self._buf
        tensor = self.tensor
        if self._backing == "mmap":
            self.tensor = None
            try:
                buf.close()
            except BufferError:
                self.tensor = tensor
                raise
            try:
                _LIVE_BUFFERS.remove(buf)
            except ValueError:
                pass
        else:
            self.tensor = None
        self._buf = None
        self._closed = True

    def __enter__(self) -> HostBank:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


_os_locked_total = 0  # bytes locked so far; the OS lock ceiling is a per-process quota
_os_lock_failed = False  # sticky: once over quota, later (bigger-total) locks fail too


def _os_lock(addr: int, nbytes: int) -> None:
    global _os_locked_total
    import resource

    # grow the soft RLIMIT_MEMLOCK (defaults to a few MiB); the hard limit needs privilege, past it mlock fails below
    want = _os_locked_total + nbytes + (256 << 20)
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != resource.RLIM_INFINITY and soft < want:
        new_soft = want if hard == resource.RLIM_INFINITY else min(want, hard)
        if new_soft > soft:
            try:
                resource.setrlimit(resource.RLIMIT_MEMLOCK, (new_soft, hard))
            except (OSError, ValueError):
                pass  # keep the old limit; mlock below reports the real ceiling
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes)):
        err = ctypes.get_errno()
        raise OSError(
            err,
            f"mlock({nbytes / 2**30:.1f} GiB): {os.strerror(err)} "
            f"(RLIMIT_MEMLOCK / `ulimit -l` caps OS-locked bytes; raise it or "
            f"shrink --moe-cpu-layers)",
        )
    _os_locked_total += nbytes


def _os_unlock(addr: int, nbytes: int) -> None:
    """Undo :func:`_os_lock` and keep process accounting bounded."""
    global _os_locked_total
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes)):
        err = ctypes.get_errno()
        raise OSError(err, f"munlock failed: {os.strerror(err)}")
    _os_locked_total = max(0, _os_locked_total - nbytes)


def alloc_banks(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
    *,
    policy: HostBankPolicy | None = None,
) -> dict[str, HostBank]:
    """Allocate host banks, optionally after an explicit policy preflight."""
    if policy is not None:
        policy.prepare(specs, 1)
    backing = "mmap" if policy is not None else None
    numa_placement = policy.numa_placement if policy is not None else None
    allocated: dict[str, HostBank] = {}
    try:
        for name, (shape, dtype) in specs.items():
            allocated[name] = HostBank(shape, dtype, backing=backing, numa_placement=numa_placement)
        if policy is not None:
            policy.refresh_numa_accounting()
        return allocated
    except BaseException:
        # The policy path is the owner of allocations created by this call.
        # HostBank.close is idempotent and safe for the already-created prefix.
        for bank in allocated.values():
            bank.close()
        raise


def alloc_layer_banks(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
    num_layers: int,
    *,
    policy: HostBankPolicy | None = None,
) -> dict[str, list[HostBank]]:
    """Allocate per-layer host banks: ``{name: ([num_experts, ...] row shape, dtype)}``
    -> one independently allocated (page-aligned, independently pin/lock-able)
    ``HostBank`` per layer per name."""
    if policy is not None:
        # This is deliberately before the first HostBank constructor.  In
        # particular, FREETOKEN_BANK_CUDA_ALLOC cannot bypass a finite policy.
        policy.prepare(specs, num_layers)
    backing = "mmap" if policy is not None else None
    numa_placement = policy.numa_placement if policy is not None else None
    allocated: dict[str, list[HostBank]] = {}
    try:
        for name, (shape, dtype) in specs.items():
            per_layer: list[HostBank] = []
            allocated[name] = per_layer
            for _ in range(num_layers):
                per_layer.append(
                    HostBank(
                        shape,
                        dtype,
                        backing=backing,
                        numa_placement=numa_placement,
                    )
                )
        if policy is not None:
            policy.refresh_numa_accounting()
    except BaseException:
        for per_layer in allocated.values():
            for bank in per_layer:
                bank.close()
        raise
    return allocated


class _ResidencyPlan:
    """Per-layer ``HostResidency`` labels, ambiently visible to the bank settle points.

    Installed by ``load_expert_banks`` around the provider dispatch so every loader honors --moe-cpu-layers without a new parameter in each signature. ``applied`` flips once a settle point consults the plan."""

    __slots__ = ("actual", "applied", "has_unpinned", "labels")

    def __init__(self, labels: list[str]):
        self.labels = list(labels)
        self.applied = False
        self.has_unpinned = any(r != HostResidency.PINNED.value for r in labels)
        self.actual: dict[int, str] = {}

    def residency_for(self, layer_id: int) -> str:
        self.applied = True
        return self.labels[layer_id]

    def record(self, layer_id: int, achieved: str) -> None:
        """One pageable bank downgrades the whole layer (a failed lock settles PAGEABLE)."""
        if self.actual.get(layer_id) != HostResidency.PAGEABLE.value:
            self.actual[layer_id] = achieved


_requested_residency: _ResidencyPlan | None = None


@contextlib.contextmanager
def requested_residency(labels: list[str] | None):
    """Install the ambient per-layer residency plan for the enclosed bank load (``None`` = no plan, everything pins)."""
    global _requested_residency
    if labels is None:
        yield None
        return
    plan = _ResidencyPlan(labels)
    prev, _requested_residency = _requested_residency, plan
    try:
        yield plan
    finally:
        _requested_residency = prev


def _settle(bank: HostBank, residency: str) -> None:
    """Route a filled bank to its residency class (PAGEABLE = leave the plain mmap)."""
    if residency == HostResidency.PINNED.value:
        bank.pin()
    elif residency == HostResidency.LOCKED.value:
        bank.lock()


def pin_banks(
    banks: dict[str, HostBank | list[HostBank]],
    *,
    policy: HostBankPolicy | None = None,
) -> None:
    """Settle every bank after it has been filled -- pin-after-fill by default.
    List-valued entries are per-layer and honor the ambient :func:`requested_residency` plan; scalar banks always pin.
    An explicit ``policy`` uses its already-prepared plan and never consults
    the ambient policy state."""
    if policy is not None:
        plan = None
        residency_for = policy.plan.layer_residency
        expected_layer_bytes = policy.plan.layer_bytes
    else:
        plan = _requested_residency
        residency_for = None
        expected_layer_bytes = None
    by_layer: dict[int, list[HostBank]] = {}
    if expected_layer_bytes is not None:
        observed_layer_bytes = [0] * len(expected_layer_bytes)
        for bank in banks.values():
            per_layer = bank if isinstance(bank, list) else [bank]
            if len(per_layer) != len(expected_layer_bytes):
                raise ValueError("policy settlement requires a complete bank set for every layer")
            for layer_id, layer_bank in enumerate(per_layer):
                observed_layer_bytes[layer_id] += layer_bank.allocated_nbytes
        if tuple(observed_layer_bytes) != expected_layer_bytes:
            raise ValueError(
                "policy settlement bank bytes do not match the prepared complete bank set"
            )
    for bank in banks.values():
        if isinstance(bank, list):
            for layer_id, layer_bank in enumerate(bank):
                residency = (
                    HostResidency.PINNED.value
                    if plan is None and residency_for is None
                    else (
                        residency_for[layer_id]
                        if residency_for is not None
                        else plan.residency_for(layer_id)
                    )
                )
                _settle(layer_bank, residency)
                if plan is not None and residency == HostResidency.LOCKED.value:
                    plan.record(layer_id, layer_bank.residency.value)
                by_layer.setdefault(layer_id, []).append(layer_bank)
        else:
            # Scalar banks predate per-layer residency plans and were always
            # pinned.  Preserve that legacy contract unless an explicit policy
            # supplies the new strategy decision.
            residency = (
                residency_for[0] if residency_for is not None else HostResidency.PINNED.value
            )
            _settle(bank, residency)
            by_layer.setdefault(0, []).append(bank)
    if policy is not None:
        applied = list(policy.accounting.applied_layers or ("",) * len(residency_for))
        applied_pinned = 0
        for layer_id, layer_banks in by_layer.items():
            if layer_id >= len(residency_for):
                continue
            actual = (
                HostResidency.PINNED.value
                if all(bank.residency is HostResidency.PINNED for bank in layer_banks)
                else HostResidency.PAGEABLE.value
            )
            applied[layer_id] = actual
            if actual == HostResidency.PINNED.value:
                applied_pinned += sum(bank.allocated_nbytes for bank in layer_banks)
        policy.accounting.applied_layers = tuple(applied)
        policy.accounting.applied_pinned_bytes = applied_pinned
        policy.refresh_numa_accounting()


class PinPipeline:
    """Settle (pin or lock) filled banks while other banks are still being read.

    cudaHostRegister is driver-serialized, so one background thread drains a queue and submitters never block: load time ~= max(read, settle).
    LOCKED banks mlock on the same thread (the quota bookkeeping in ``_os_lock`` is not thread-safe).
    A clean context-manager exit drains the queue and re-raises the first settle failure.
    """

    def __init__(self) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._exc: BaseException | None = None
        # the current device is thread-local: a fresh thread sits on device 0 and cudaHostRegister would build its context there -- carry the creator's (bound) device into the worker
        self._device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if self._device is not None:
            torch.cuda.set_device(self._device)
        while True:
            item = self._q.get()
            if item is None:
                return
            if self._exc is not None:
                continue  # drain without settling after a failure
            bank, residency, plan, layer_id = item
            try:
                _settle(bank, residency)
                if plan is not None and residency == HostResidency.LOCKED.value:
                    plan.record(layer_id, bank.residency.value)
            except BaseException as exc:  # surfaced by wait()/__exit__
                self._exc = exc

    def submit(
        self,
        bank: HostBank,
        residency: str = HostResidency.PINNED.value,
        plan=None,
        layer_id: int | None = None,
    ) -> None:
        self._q.put((bank, residency, plan, layer_id))

    def __call__(self, layer_id: int, banks: dict[str, HostBank]) -> None:
        """Layer-completion sink: queue every bank of the completed layer at its ambient :func:`requested_residency` label."""
        plan = _requested_residency
        residency = HostResidency.PINNED.value if plan is None else plan.residency_for(layer_id)
        for bank in banks.values():
            self.submit(bank, residency, plan, layer_id)

    def _join(self) -> None:
        self._q.put(None)
        self._thread.join()

    def wait(self) -> None:
        self._join()
        if self._exc is not None:
            raise self._exc

    def __enter__(self) -> PinPipeline:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._join()  # no thread leak; the in-flight exception wins
            return
        self.wait()


class LayerCompletionTracker:
    """Fire a sink once per layer, when all of that layer's writes have landed.

    ``note(layer_id)`` is called after each write; at ``expected_per_layer``
    notes the layer's banks are handed to ``on_layer(layer_id, {name: bank})``
    exactly once. Thread-safe (shard-driven loaders write layers from many
    threads in arbitrary order).
    """

    def __init__(
        self,
        expected_per_layer: int,
        banks: dict[str, list],
        on_layer,
    ) -> None:
        assert expected_per_layer > 0
        self._expected = expected_per_layer
        self._banks = banks
        self._on_layer = on_layer
        self._counts: dict[int, int] = {}
        self._lock = threading.Lock()

    def note(self, layer_id: int) -> None:
        with self._lock:
            n = self._counts.get(layer_id, 0) + 1
            self._counts[layer_id] = n
            fire = n == self._expected
        if fire:
            self._on_layer(layer_id, {name: per[layer_id] for name, per in self._banks.items()})


def read_file_into(
    buf: memoryview | mmap.mmap,
    path: str,
    *,
    workers: int = 8,
    chunk: int = _DEFAULT_CHUNK,
    drop_cache: bool = True,
) -> int:
    """Chunked multi-threaded O_DIRECT read of the whole file ``path`` into ``buf``
    (page-aligned). Returns the file size. The buffer must be >= the rounded-up file size."""
    size = os.path.getsize(path)
    if drop_cache:
        try:
            fd0 = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd0, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd0)
        except OSError:
            pass
    mv = buf if isinstance(buf, memoryview) else memoryview(buf)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    offs = list(range(0, size, chunk))

    def rd(o):
        want = min(chunk, len(mv) - o)
        want = min(want, ((size - o + _BLK - 1) // _BLK) * _BLK)
        os.preadv(fd, [mv[o : o + want]], o)

    try:
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        os.close(fd)
    return size


def _preadv_all(fd: int, dst: memoryview, offset: int, need: int) -> None:
    """preadv into ``dst`` until ``need`` bytes have landed; O_DIRECT may return a short count."""
    done = 0
    while done < need:
        if done % _BLK:  # a continuation read has to stay block-aligned on both sides
            raise OSError(f"unaligned short O_DIRECT read: {done} of {need} bytes at {offset}")
        got = os.preadv(fd, [dst[done:]], offset + done)
        if got <= 0:
            raise OSError(f"short O_DIRECT read: {done} of {need} bytes at {offset}")
        done += got


def read_range_into(buf: memoryview | mmap.mmap, path: str, *, file_offset: int, nbytes: int,
                    dest_offset: int = 0, workers: int = 8, chunk: int = _DEFAULT_CHUNK,
                    drop_cache: bool = True) -> int:
    """Chunked multi-threaded O_DIRECT read of ``path[file_offset : file_offset + nbytes]`` into ``buf`` at ``dest_offset``. Returns ``nbytes``.

    Byte-range counterpart of :func:`read_file_into`, for one tensor inside a shard. O_DIRECT needs the file offset AND the destination address block-aligned at the same time, which only holds when the two share their offset mod 4096 -- a safetensors data offset practically never lines up with the tensor's slot in the bank. Chunks that do line up DMA straight into ``buf``; the rest DMA into a page-aligned bounce (source window rounded out to whole blocks) and are copied into place, which also covers the unaligned head and tail.
    """
    mv = (buf if isinstance(buf, memoryview) else memoryview(buf)).cast("B")
    if dest_offset + nbytes > len(mv):
        raise ValueError(f"destination holds {len(mv)} bytes, need {dest_offset + nbytes}")
    base = ctypes.addressof(ctypes.c_char.from_buffer(mv))
    if drop_cache:
        try:
            fd0 = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd0, file_offset, nbytes, os.POSIX_FADV_DONTNEED)
            os.close(fd0)
        except OSError:
            pass
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    scratch = threading.local()

    def rd(i: int) -> None:
        n = min(chunk, nbytes - i)
        src, dst = file_offset + i, dest_offset + i
        if src % _BLK == 0 and (base + dst) % _BLK == 0 and n % _BLK == 0:
            _preadv_all(fd, mv[dst:dst + n], src, n)
            return
        head = src % _BLK
        span = ((head + n + _BLK - 1) // _BLK) * _BLK
        bounce = getattr(scratch, "buf", None)
        if bounce is None or len(bounce) < span:
            bounce = scratch.buf = mmap.mmap(-1, span)  # anonymous mmaps are page-aligned
        bmv = memoryview(bounce)
        _preadv_all(fd, bmv[:span], src - head, head + n)
        mv[dst:dst + n] = bmv[head:head + n]

    try:
        offs = list(range(0, nbytes, chunk))
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        os.close(fd)
    return nbytes


__all__ = [
    "HostBank",
    "HostBankAccounting",
    "HostBankPlan",
    "HostBankPolicy",
    "HostBankStrategy",
    "HostResidency",
    "HostStagingRing",
    "LayerCompletionTracker",
    "NumaPlacementController",
    "NumaSyscallBackend",
    "NumaPolicy",
    "PinPipeline",
    "alloc_banks",
    "alloc_layer_banks",
    "born_pinned_default",
    "pin_banks",
    "read_file_into",
    "read_range_into",
    "requested_residency",
]
