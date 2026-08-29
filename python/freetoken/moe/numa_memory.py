"""Small, explicit NUMA placement and residency probes.

This module deliberately has no torch dependency.  Placement is opt-in: callers
must construct a :class:`NumaPlacementController` with ``enforce=True`` before
an anonymous mapping is touched.  The Linux implementation only issues the
``mbind`` and (optional) read-only ``move_pages`` syscalls for the current
process; it never changes system policy or migrates pages.
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

NumaPlacementStatus = Literal[
    "not-requested", "planned", "applied", "fallback", "failed", "unavailable"
]
NumaSampleStatus = Literal["not-requested", "verified", "partial", "unavailable"]

_MPOL_PREFERRED = 1
_MPOL_BIND = 2
_MPOL_INTERLEAVE = 3
_MAX_NODE = 4096


class NumaPlacementError(ValueError):
    """A requested placement cannot be applied safely."""


def _parse_node_list(raw: str, *, field_name: str) -> tuple[int, ...]:
    values: set[int] = set()
    text = raw.strip()
    if not text:
        raise ValueError(f"{field_name} is empty")
    for component in text.split(","):
        component = component.strip()
        if not component:
            raise ValueError(f"{field_name} contains an empty component")
        parts = component.split("-")
        if len(parts) > 2 or any(not part.isdigit() for part in parts):
            raise ValueError(f"{field_name} contains an invalid component {component!r}")
        start = int(parts[0])
        end = int(parts[-1])
        if end < start:
            raise ValueError(f"{field_name} contains descending range {component!r}")
        if end > _MAX_NODE:
            raise ValueError(f"{field_name} contains an out-of-range node")
        values.update(range(start, end + 1))
    return tuple(sorted(values))


def _normalize_nodes(nodes: Sequence[int]) -> tuple[int, ...]:
    values = []
    for node in nodes:
        if isinstance(node, bool) or not isinstance(node, int) or node < 0 or node > _MAX_NODE:
            raise NumaPlacementError(f"NUMA node IDs must be integers in [0, {_MAX_NODE}]")
        values.append(node)
    return tuple(sorted(set(values)))


def _read_default(path: str) -> str:
    with open(path, encoding="ascii") as stream:
        return stream.read()


def _status_field(text: str, name: str) -> str:
    matches = []
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == name:
            matches.append(value.strip())
    if len(matches) != 1:
        raise ValueError(f"NUMA status is missing or duplicates {name}")
    return matches[0]


@dataclass(frozen=True, slots=True)
class AllowedNumaNodes:
    """Online NUMA nodes allowed to this process."""

    nodes: tuple[int, ...]
    status: Literal["available", "unavailable"]
    source: str
    errors: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.status == "available" and bool(self.nodes)


def resolve_allowed_numa_nodes(
    *,
    read_text: Callable[[str], str] | None = None,
    status_path: str = "/proc/self/status",
    online_path: str = "/sys/devices/system/node/online",
) -> AllowedNumaNodes:
    """Return ``online ∩ Mems_allowed_list`` without changing process policy."""
    reader = _read_default if read_text is None else read_text
    try:
        status_text = reader(status_path)
        allowed = _parse_node_list(
            _status_field(status_text, "Mems_allowed_list"),
            field_name="Mems_allowed_list",
        )
        online = _parse_node_list(reader(online_path).strip(), field_name="online NUMA nodes")
    except Exception as error:
        return AllowedNumaNodes((), "unavailable", f"{status_path},{online_path}", (str(error),))
    nodes = tuple(node for node in online if node in set(allowed))
    if not nodes:
        return AllowedNumaNodes(
            (),
            "unavailable",
            f"{status_path},{online_path}",
            ("online nodes do not intersect Mems_allowed_list",),
        )
    return AllowedNumaNodes(nodes, "available", f"{status_path},{online_path}")


def _policy_value(policy: object) -> str:
    value = getattr(policy, "value", policy)
    return str(value).strip().lower()


@dataclass(frozen=True, slots=True)
class NumaPlacementPlan:
    """Validated target mask for one explicit placement request."""

    policy: str
    requested_node: int | None
    allowed_nodes: tuple[int, ...]
    target_nodes: tuple[int, ...]
    enforce: bool
    status: NumaPlacementStatus
    source: str
    errors: tuple[str, ...] = ()
    fallback_reason: str | None = None


def resolve_numa_placement(
    policy: object,
    node: int | None,
    *,
    enforce: bool = False,
    allowed: AllowedNumaNodes | Sequence[int] | None = None,
    read_text: Callable[[str], str] | None = None,
) -> NumaPlacementPlan:
    """Validate policy semantics and, when enabled, resolve a target node mask.

    A preferred/interleave request can fall back when procfs topology or the
    syscall is unavailable.  A bind request always raises in those cases because
    silently running on another node would violate its contract.
    """
    name = _policy_value(policy)
    if name not in {"preferred", "bind", "interleave"}:
        raise NumaPlacementError(f"unknown NUMA policy {policy!r}")
    if node is not None and (isinstance(node, bool) or not isinstance(node, int) or node < 0):
        raise NumaPlacementError("numa_node must be a non-negative integer or None")
    if not isinstance(enforce, bool):
        raise NumaPlacementError("enforce_numa_placement must be a boolean")
    if not enforce:
        return NumaPlacementPlan(name, node, (), (), False, "not-requested", "disabled")

    topology = (
        allowed
        if isinstance(allowed, AllowedNumaNodes)
        else AllowedNumaNodes(_normalize_nodes(allowed), "available", "injected")
        if allowed is not None
        else resolve_allowed_numa_nodes(read_text=read_text)
    )
    if not topology.available:
        detail = "; ".join(topology.errors) or "allowed NUMA topology is unavailable"
        if name == "bind":
            raise NumaPlacementError(f"NUMA bind placement failed closed: {detail}")
        return NumaPlacementPlan(
            name,
            node,
            topology.nodes,
            (),
            True,
            "fallback",
            topology.source,
            topology.errors,
            detail,
        )
    if node is not None and node not in topology.nodes:
        raise NumaPlacementError(
            f"requested NUMA node {node} is not online and allowed (nodes={topology.nodes})"
        )
    if name == "bind" and node is None:
        raise NumaPlacementError("NUMA bind placement requires numa_node")
    if name == "preferred" and node is None:
        return NumaPlacementPlan(
            name,
            None,
            topology.nodes,
            (),
            True,
            "fallback",
            topology.source,
            (),
            "preferred placement requires numa_node when enforcement is enabled",
        )
    target = (node,) if node is not None else topology.nodes
    return NumaPlacementPlan(name, node, topology.nodes, target, True, "planned", topology.source)


def linux_syscall_numbers(
    *, system: str | None = None, machine: str | None = None
) -> dict[str, int] | None:
    """Return direct syscall numbers for the supported Linux x86_64 ABI."""
    system = platform.system() if system is None else system
    machine = platform.machine() if machine is None else machine
    if system.lower() != "linux" or machine.lower() not in {"x86_64", "amd64"}:
        return None
    return {"mbind": 237, "move_pages": 279}


class NumaSyscallBackend:
    """Linux x86_64 direct syscall backend; unsupported hosts report ENOSYS."""

    def __init__(self, *, system: str | None = None, machine: str | None = None, libc=None):
        self.numbers = linux_syscall_numbers(system=system, machine=machine)
        self._libc = libc if libc is not None else ctypes.CDLL(None, use_errno=True)
        if hasattr(self._libc, "syscall"):
            try:
                self._libc.syscall.restype = ctypes.c_long
            except (AttributeError, TypeError):
                pass

    def _call(self, name: str, *args):
        number = None if self.numbers is None else self.numbers.get(name)
        if number is None:
            raise OSError(errno.ENOSYS, f"{name} is unavailable on this platform")
        result = self._libc.syscall(ctypes.c_long(number), *args)
        if result == -1:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return result

    def mbind(self, addr: int, nbytes: int, mode: int, nodes: Sequence[int]) -> None:
        if not nodes:
            raise OSError(errno.EINVAL, "mbind requires a non-empty node mask")
        highest = max(nodes)
        words = (highest + 64) // 64
        mask = (ctypes.c_ulong * words)()
        for node in nodes:
            mask[node // 64] |= 1 << (node % 64)
        self._call(
            "mbind",
            ctypes.c_void_p(addr),
            ctypes.c_ulong(nbytes),
            ctypes.c_int(mode),
            ctypes.cast(mask, ctypes.POINTER(ctypes.c_ulong)),
            ctypes.c_ulong(highest + 1),
            ctypes.c_uint(0),
        )

    def move_pages(self, addresses: Sequence[int]) -> tuple[int, ...]:
        if not addresses:
            return ()
        count = len(addresses)
        pages = (ctypes.c_void_p * count)(*(ctypes.c_void_p(address) for address in addresses))
        statuses = (ctypes.c_int * count)()
        self._call(
            "move_pages",
            ctypes.c_int(0),
            ctypes.c_ulong(count),
            pages,
            None,
            ctypes.cast(statuses, ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(0),
        )
        return tuple(int(status) for status in statuses)


@dataclass(frozen=True, slots=True)
class NumaSample:
    status: NumaSampleStatus
    counts: tuple[tuple[int, int], ...] = ()
    unknown: int = 0
    error: str | None = None
    sampled_pages: int = 0
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "counts": {str(node): count for node, count in self.counts},
            "unknown": self.unknown,
            "error": self.error,
            "sampled_pages": self.sampled_pages,
            "sampled_total": self.sampled_pages,
            "errors": list(self.errors),
        }


@dataclass
class NumaPlacementController:
    """Apply one prepared plan and retain truthful placement/sample telemetry."""

    plan: NumaPlacementPlan
    backend: object = field(default_factory=NumaSyscallBackend, repr=False)
    status: NumaPlacementStatus = field(init=False)
    errors: list[str] = field(default_factory=list, init=False)
    applied_mappings: int = field(default=0, init=False)
    sample_result: NumaSample = field(
        default_factory=lambda: NumaSample("not-requested"), init=False
    )
    sample_history: list[NumaSample] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.status = self.plan.status
        self.errors.extend(self.plan.errors)

    @property
    def fallback_reason(self) -> str | None:
        return self.plan.fallback_reason or (self.errors[0] if self.errors else None)

    def apply(
        self,
        addr: int,
        nbytes: int,
        *,
        private_anonymous: bool,
        before_touch: bool,
    ) -> None:
        """Apply ``mbind`` before a mapping is touched; no-op for disabled/fallback."""
        if not private_anonymous or not before_touch:
            raise NumaPlacementError(
                "NUMA placement is restricted to self private-anonymous mappings before first touch"
            )
        if not self.plan.enforce or self.status in {"not-requested", "fallback", "unavailable"}:
            return
        if self.status == "failed":
            raise NumaPlacementError(self.fallback_reason or "NUMA placement already failed")
        mode = {"preferred": _MPOL_PREFERRED, "bind": _MPOL_BIND, "interleave": _MPOL_INTERLEAVE}[
            self.plan.policy
        ]
        try:
            method = getattr(self.backend, "mbind", None)
            if method is None:
                method = self.backend.apply_mbind
            result = method(addr, nbytes, mode, self.plan.target_nodes)
            if result is False or (result is not None and result != 0 and result is not True):
                raise OSError(errno.EIO, f"mbind returned {result!r}")
        except Exception as error:
            detail = f"mbind failed: {error}"
            self.errors.append(detail)
            if self.plan.policy == "bind":
                self.status = "failed"
                raise NumaPlacementError(detail) from error
            self.status = "fallback"
            return
        self.applied_mappings += 1
        self.status = "applied"

    def sample(
        self,
        addr: int,
        nbytes: int,
        *,
        stride: int = 4096,
        max_pages: int = 64,
    ) -> NumaSample:
        """Read a bounded sample of page locations for this process only."""
        if stride <= 0 or max_pages <= 0:
            raise ValueError("NUMA sample stride and max_pages must be positive")
        if not self.plan.enforce or self.status in {"fallback", "unavailable", "failed"}:
            return self._record_sample(
                NumaSample(
                    "unavailable",
                    error=self.fallback_reason,
                    errors=(self.fallback_reason,) if self.fallback_reason else (),
                )
            )
        count = min(max_pages, max(0, (nbytes + stride - 1) // stride))
        addresses = tuple(addr + i * stride for i in range(count))
        try:
            statuses = tuple(self.backend.move_pages(addresses))
        except Exception as error:
            detail = str(error)
            return self._record_sample(NumaSample("unavailable", error=detail, errors=(detail,)))
        if len(statuses) != len(addresses):
            detail = "move_pages returned wrong sample length"
            return self._record_sample(NumaSample("unavailable", error=detail, errors=(detail,)))
        counts: dict[int, int] = {}
        unknown = 0
        for status in statuses:
            if isinstance(status, int) and status >= 0:
                counts[status] = counts.get(status, 0) + 1
            else:
                unknown += 1
        status: NumaSampleStatus = "verified" if unknown == 0 else "partial"
        return self._record_sample(
            NumaSample(status, tuple(sorted(counts.items())), unknown, sampled_pages=len(statuses))
        )

    def _record_sample(self, result: NumaSample) -> NumaSample:
        self.sample_history.append(result)
        counts: dict[int, int] = {}
        unknown = 0
        sampled_pages = 0
        errors: list[str] = []
        for sample in self.sample_history:
            for node, count in sample.counts:
                counts[node] = counts.get(node, 0) + count
            unknown += sample.unknown
            sampled_pages += sample.sampled_pages
            errors.extend(sample.errors or ((sample.error,) if sample.error else ()))
        usable = [sample for sample in self.sample_history if sample.sampled_pages > 0]
        if not usable:
            aggregate_status: NumaSampleStatus = "unavailable"
        elif (
            any(
                sample.status != "verified" or sample.error is not None
                for sample in self.sample_history
            )
            or unknown
        ):
            aggregate_status = "partial"
        else:
            aggregate_status = "verified"
        aggregate_error = "; ".join(errors) if errors else None
        self.sample_result = NumaSample(
            aggregate_status,
            tuple(sorted(counts.items())),
            unknown,
            aggregate_error,
            sampled_pages,
            tuple(errors),
        )
        return self.sample_result

    def telemetry(self) -> dict[str, object]:
        return {
            "status": self.status,
            "requested": self.plan.enforce,
            "applied": self.status == "applied",
            "fallback": self.status == "fallback",
            "requested_policy": self.plan.policy,
            "requested_node": self.plan.requested_node,
            "allowed_nodes": self.plan.allowed_nodes,
            "target_nodes": self.plan.target_nodes,
            "source": self.plan.source,
            "applied_mappings": self.applied_mappings,
            "errors": tuple(self.errors),
            "fallback_reason": self.fallback_reason,
            "sample": self.sample_result.as_dict(),
            "sample_banks": tuple(sample.as_dict() for sample in self.sample_history),
        }


# Compatibility aliases for call sites that describe this as a placement result.
NumaPlacement = NumaPlacementController

__all__ = [
    "AllowedNumaNodes",
    "NumaPlacement",
    "NumaPlacementController",
    "NumaPlacementError",
    "NumaPlacementPlan",
    "NumaSample",
    "NumaSyscallBackend",
    "linux_syscall_numbers",
    "resolve_allowed_numa_nodes",
    "resolve_numa_placement",
]
