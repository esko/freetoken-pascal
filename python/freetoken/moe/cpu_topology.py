"""CPU topology discovery and immutable worker-placement plans.

This module is deliberately independent of Torch, CUDA and model code.  It turns the
process CPU affinity mask and Linux CPU topology files into a checked, serializable
description that later executors may consume.  Planning is not enforcement: no worker
or process affinity is changed here, and every plan reports ``planned-unverified``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Literal, TypeVar

TopologyConfidence = Literal["full", "partial", "logical-only"]
AffinityStatus = Literal["planned-unverified"]

_PART_RE = re.compile(r"^(\d+)(?:-(\d+))?$")
_CONFIDENCES = frozenset({"full", "partial", "logical-only"})
_PARTITIONS = frozenset({"mask", "contiguous", "numa"})
_T = TypeVar("_T")


def _cpu_tuple(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    result: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must contain integer CPU IDs, got {value!r}")
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must contain non-negative CPU IDs, got {value}")
        result.add(value)
    return tuple(sorted(result))


def _parse_cpu_list(value: str) -> tuple[int, ...] | None:
    """Parse Linux's comma-separated CPU-list syntax, returning ``None`` when invalid."""
    value = value.strip()
    if not value:
        return None
    result: set[int] = set()
    for part in value.split(","):
        match = _PART_RE.fullmatch(part.strip())
        if match is None:
            return None
        begin = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        if end < begin:
            return None
        result.update(range(begin, end + 1))
    return tuple(sorted(result))


def _parse_nonnegative_int(value: str) -> int | None:
    value = value.strip()
    if not value or not value.isdigit():
        return None
    return int(value)


def _default_read_text(path: Path) -> str:
    return path.read_text(encoding="ascii")


def _read_optional(path: Path, reader: Callable[[Path], str]) -> str | None:
    try:
        return reader(path)
    except (OSError, UnicodeError):
        return None


def _read_numa_node(cpu_root: Path, reader: Callable[[Path], str]) -> int | None:
    """Read a CPU's NUMA node from the Linux ``nodeN`` link or ``numa_node`` file."""
    nodes: list[int] = []
    try:
        for entry in cpu_root.glob("node[0-9]*"):
            suffix = entry.name[4:]
            if suffix.isdigit():
                nodes.append(int(suffix))
    except OSError:
        pass
    if nodes:
        return min(nodes)
    raw = _read_optional(cpu_root / "numa_node", reader)
    return _parse_nonnegative_int(raw) if raw is not None else None


@dataclass
class _CpuFacts:
    cpu: int
    sibling_raw: str | None
    siblings: tuple[int, ...] | None
    package: int | None
    core_id: int | None
    numa_node: int | None


def _sibling_components(
    facts: tuple[_CpuFacts, ...], visible: tuple[int, ...]
) -> tuple[dict[int, tuple[int, ...]], set[int], set[int]]:
    """Canonicalize valid sibling lists and identify inferred/conflicting CPUs."""
    parent = {cpu: cpu for cpu in visible}

    def find(cpu: int) -> int:
        while parent[cpu] != cpu:
            parent[cpu] = parent[parent[cpu]]
            cpu = parent[cpu]
        return cpu

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    local_valid = {
        fact.cpu for fact in facts if fact.siblings is not None and fact.cpu in fact.siblings
    }
    for fact in facts:
        if fact.cpu not in local_valid:
            continue
        members = [cpu for cpu in fact.siblings or () if cpu in parent]
        for member in members[1:]:
            union(members[0], member)

    # Package/core is a useful fallback, but when it overlaps any sibling component
    # it must reconcile that component rather than create a second SMT core.  A later
    # consistency pass marks contradictory sources partial.
    package_groups: dict[tuple[int, int], list[int]] = {}
    for fact in facts:
        if fact.package is not None and fact.core_id is not None:
            package_groups.setdefault((fact.package, fact.core_id), []).append(fact.cpu)
    initial_members_by_root: dict[int, set[int]] = {}
    for cpu in visible:
        initial_members_by_root.setdefault(find(cpu), set()).add(cpu)
    initial_component_by_cpu: dict[int, tuple[int, ...]] = {}
    for _root, members in initial_members_by_root.items():
        if any(cpu in local_valid for cpu in members):
            component = tuple(sorted(members))
            for cpu in component:
                initial_component_by_cpu[cpu] = component
    for members in package_groups.values():
        sibling_members = [cpu for cpu in members if cpu in initial_component_by_cpu]
        if sibling_members:
            for member in members[1:]:
                union(sibling_members[0], member)

    members_by_root: dict[int, set[int]] = {}
    for cpu in visible:
        members_by_root.setdefault(find(cpu), set()).add(cpu)
    component_by_cpu: dict[int, tuple[int, ...]] = {}
    valid_roots = {find(fact.cpu) for fact in facts if fact.cpu in local_valid}
    for root, members in members_by_root.items():
        if root not in valid_roots:
            continue
        component = tuple(sorted(members))
        for cpu in component:
            component_by_cpu[cpu] = component

    inferred = set(component_by_cpu) - local_valid
    conflicting: set[int] = set()
    for fact in facts:
        if fact.cpu not in local_valid:
            continue
        observed = {cpu for cpu in fact.siblings or () if cpu in parent}
        component = set(component_by_cpu[fact.cpu])
        if observed != component:
            conflicting.update(component)
    package_keys_by_component: dict[tuple[int, ...], set[tuple[int, int]]] = {}
    for fact in facts:
        if fact.cpu not in component_by_cpu or fact.package is None or fact.core_id is None:
            continue
        component = component_by_cpu[fact.cpu]
        package_keys_by_component.setdefault(component, set()).add((fact.package, fact.core_id))
    for component, package_keys in package_keys_by_component.items():
        if len(package_keys) > 1:
            conflicting.update(component)
    return component_by_cpu, inferred, conflicting


@dataclass(frozen=True, slots=True)
class PhysicalCore:
    """One physical core and the logical CPUs visible in the process mask."""

    key: str
    representative: int
    logical_cpus: tuple[int, ...]
    siblings: tuple[int, ...]
    numa_node: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("physical core key must be a non-empty string")
        logical = _cpu_tuple(self.logical_cpus, name="logical_cpus")
        siblings = _cpu_tuple(self.siblings, name="siblings")
        if not logical:
            raise ValueError("physical core must contain an allowed logical CPU")
        if not siblings or not set(logical).issubset(siblings):
            raise ValueError("logical_cpus must be a non-empty subset of siblings")
        if isinstance(self.representative, bool) or not isinstance(self.representative, Integral):
            raise ValueError("physical core representative must be an integer CPU ID")
        representative = int(self.representative)
        if representative not in logical:
            raise ValueError("physical core representative must be an allowed logical CPU")
        if self.numa_node is not None:
            if isinstance(self.numa_node, bool) or not isinstance(self.numa_node, Integral):
                raise ValueError("numa_node must be a non-negative integer or None")
            if int(self.numa_node) < 0:
                raise ValueError("numa_node must be a non-negative integer or None")
            object.__setattr__(self, "numa_node", int(self.numa_node))
        object.__setattr__(self, "representative", representative)
        object.__setattr__(self, "logical_cpus", logical)
        object.__setattr__(self, "siblings", siblings)


@dataclass(frozen=True, slots=True)
class CpuTopology:
    """Immutable topology restricted to the process's allowed CPU set."""

    allowed_cpus: tuple[int, ...]
    cores: tuple[PhysicalCore, ...]
    confidence: TopologyConfidence
    source: str
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        allowed = _cpu_tuple(self.allowed_cpus, name="allowed_cpus")
        cores = tuple(self.cores)
        if self.confidence not in _CONFIDENCES:
            raise ValueError(f"unknown topology confidence {self.confidence!r}")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("topology source must be a non-empty string")
        keys = [core.key for core in cores]
        if len(keys) != len(set(keys)):
            raise ValueError("topology contains duplicate physical core keys")
        representatives = [core.representative for core in cores]
        if len(representatives) != len(set(representatives)):
            raise ValueError("topology contains duplicate physical core representatives")
        owners: dict[int, str] = {}
        for core in cores:
            for cpu in core.logical_cpus:
                previous = owners.get(cpu)
                if previous is not None:
                    raise ValueError(
                        f"topology contains overlapping logical CPU {cpu} in cores "
                        f"{previous!r} and {core.key!r}"
                    )
                owners[cpu] = core.key
        visible = {cpu for core in cores for cpu in core.logical_cpus}
        if visible != set(allowed):
            raise ValueError("physical cores must account for exactly the allowed CPU set")
        cores = tuple(sorted(cores, key=lambda core: (min(core.logical_cpus), core.key)))
        object.__setattr__(self, "allowed_cpus", allowed)
        object.__setattr__(self, "cores", cores)

    @property
    def physical_core_cpus(self) -> tuple[int, ...]:
        """One allowed representative per discovered physical core."""
        return tuple(core.representative for core in self.cores)

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed_cpus": list(self.allowed_cpus),
            "physical_core_cpus": list(self.physical_core_cpus),
            "confidence": self.confidence,
            "source": self.source,
            "fallback_reason": self.fallback_reason,
            "cores": [
                {
                    "key": core.key,
                    "representative": core.representative,
                    "logical_cpus": list(core.logical_cpus),
                    "siblings": list(core.siblings),
                    "numa_node": core.numa_node,
                }
                for core in self.cores
            ],
        }


def discover_cpu_topology(
    *,
    allowed_cpus: Iterable[int] | None = None,
    affinity_getter: Callable[[], Iterable[int]] | None = None,
    cpu_count: int | None = None,
    sysfs_root: str | Path = "/sys/devices/system/cpu",
    read_text: Callable[[Path], str] | None = None,
) -> CpuTopology:
    """Discover topology from an affinity mask and injectable Linux sysfs reads.

    ``allowed_cpus`` and ``affinity_getter`` are test seams and deployment hooks.  If
    the process affinity API is unavailable, the fallback uses ``cpu_count`` (or the
    host count) but records that it could not establish a cpuset restriction.
    Missing or malformed topology never fabricates SMT groups: those CPUs become
    logical-only groups and the returned confidence/fallback fields make that visible.
    """
    reasons: list[str] = []
    if cpu_count is not None:
        if isinstance(cpu_count, bool) or not isinstance(cpu_count, Integral):
            raise ValueError("cpu_count must be a non-negative integer or None")
        cpu_count = int(cpu_count)
        if cpu_count < 0:
            raise ValueError("cpu_count must be a non-negative integer or None")
    if allowed_cpus is not None:
        visible = _cpu_tuple(allowed_cpus, name="allowed_cpus")
        source = "provided-affinity"
    else:
        getter = affinity_getter or (lambda: os.sched_getaffinity(0))
        try:
            visible = _cpu_tuple(getter(), name="process affinity")
            source = "sched_getaffinity"
        except (AttributeError, OSError) as error:
            count = os.cpu_count() if cpu_count is None else cpu_count
            if count is None:
                count = 0
            visible = tuple(range(count))
            source = "cpu-count"
            reasons.append(f"process affinity unavailable: {error}")

    root = Path(sysfs_root)
    reader = read_text or _default_read_text
    if not visible:
        reasons.append("no allowed CPUs were discovered")
        return CpuTopology(
            allowed_cpus=(),
            cores=(),
            confidence="logical-only",
            source=f"{source}+sysfs",
            fallback_reason="; ".join(reasons),
        )

    facts: list[_CpuFacts] = []
    for cpu in visible:
        cpu_root = root / f"cpu{cpu}"
        topology_root = cpu_root / "topology"

        sibling_raw = _read_optional(topology_root / "thread_siblings_list", reader)
        siblings = _parse_cpu_list(sibling_raw) if sibling_raw is not None else None
        if sibling_raw is not None and siblings is None:
            reasons.append(f"malformed thread_siblings_list for cpu {cpu}")
        elif siblings is not None and cpu not in siblings:
            reasons.append(f"thread_siblings_list omits cpu {cpu}")

        package_raw = _read_optional(topology_root / "physical_package_id", reader)
        core_raw = _read_optional(topology_root / "core_id", reader)
        package = _parse_nonnegative_int(package_raw) if package_raw is not None else None
        core_id = _parse_nonnegative_int(core_raw) if core_raw is not None else None
        if package_raw is not None and package is None:
            reasons.append(f"malformed physical_package_id for cpu {cpu}")
        if core_raw is not None and core_id is None:
            reasons.append(f"malformed core_id for cpu {cpu}")

        facts.append(
            _CpuFacts(
                cpu=cpu,
                sibling_raw=sibling_raw,
                siblings=siblings,
                package=package,
                core_id=core_id,
                numa_node=_read_numa_node(cpu_root, reader),
            )
        )

    facts_tuple = tuple(facts)
    component_by_cpu, inferred, conflicting = _sibling_components(facts_tuple, visible)
    for cpu in sorted(inferred):
        reasons.append(f"sibling topology inferred for cpu {cpu}")
    for cpu in sorted(conflicting):
        reasons.append(f"conflicting sibling topology for cpu {cpu}")

    groups: dict[str, dict[str, object]] = {}
    for fact in facts_tuple:
        cpu = fact.cpu
        sibling_component = component_by_cpu.get(cpu)
        if sibling_component is not None:
            key = "siblings:" + ",".join(str(item) for item in sibling_component)
            has_physical_identity = cpu not in inferred and cpu not in conflicting
        elif fact.package is not None and fact.core_id is not None:
            key = f"package-core:{fact.package}:{fact.core_id}"
            has_physical_identity = fact.sibling_raw is None
        else:
            key = f"logical:{cpu}"
            has_physical_identity = False
            reasons.append(f"physical identity unavailable for cpu {cpu}")

        group = groups.setdefault(
            key,
            {
                "logical_cpus": set(),
                "siblings": set(),
                "nodes": set(),
                "quality": True,
            },
        )
        group["logical_cpus"].add(cpu)  # type: ignore[union-attr]
        group["siblings"].update((fact.siblings or ()) + (cpu,))  # type: ignore[union-attr]
        if fact.numa_node is not None:
            group["nodes"].add(fact.numa_node)  # type: ignore[union-attr]
        group["quality"] = bool(group["quality"]) and has_physical_identity

    cores: list[PhysicalCore] = []
    for key, group in sorted(
        groups.items(),
        key=lambda item: min(item[1]["logical_cpus"]),  # type: ignore[arg-type]
    ):
        logical = tuple(sorted(group["logical_cpus"]))  # type: ignore[arg-type]
        siblings = tuple(sorted(group["siblings"]))  # type: ignore[arg-type]
        nodes = tuple(sorted(group["nodes"]))  # type: ignore[arg-type]
        if len(nodes) > 1:
            reasons.append(f"conflicting NUMA nodes for physical core {key}")
        cores.append(
            PhysicalCore(
                key=key,
                representative=logical[0],
                logical_cpus=logical,
                siblings=siblings,
                numa_node=nodes[0] if len(nodes) == 1 else None,
            )
        )

    qualities = [bool(group["quality"]) for group in groups.values()]
    physical_groups = [not key.startswith("logical:") for key in groups]
    if any(physical_groups) and all(qualities):
        confidence: TopologyConfidence = "full"
    elif any(physical_groups):
        confidence = "partial"
    else:
        confidence = "logical-only"
    return CpuTopology(
        allowed_cpus=visible,
        cores=tuple(cores),
        confidence=confidence,
        source=f"{source}+sysfs",
        fallback_reason="; ".join(dict.fromkeys(reasons)) or None,
    )


def _split_contiguous(  # noqa: UP047 - the package still supports Python 3.10
    items: tuple[_T, ...], rank: int, world_size: int
) -> tuple[_T, ...]:
    base, remainder = divmod(len(items), world_size)
    begin = rank * base + min(rank, remainder)
    end = begin + base + int(rank < remainder)
    return items[begin:end]


@dataclass(frozen=True, slots=True)
class WorkerAffinityPolicy:
    """Requested worker sizing and explicit rank partition policy.

    ``partition='mask'`` is intentionally the default: the discovered process mask is
    authoritative and is not divided by ``world_size``.  ``numa`` and ``contiguous``
    are explicit planning modes only.  This object never changes affinity.
    """

    requested_threads: int | None = 0
    rank: int = 0
    world_size: int = 1
    partition: Literal["mask", "contiguous", "numa"] = "mask"
    reserve_coordinator: bool = False

    def __post_init__(self) -> None:
        if self.requested_threads is not None:
            if isinstance(self.requested_threads, bool) or not isinstance(
                self.requested_threads, Integral
            ):
                raise ValueError("requested_threads must be a non-negative integer or None")
            requested = int(self.requested_threads)
            if requested < 0:
                raise ValueError("requested_threads must be a non-negative integer or None")
            object.__setattr__(self, "requested_threads", requested)
        for name in ("rank", "world_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be a positive integer")
            value = int(value)
            if (name == "world_size" and value < 1) or (name == "rank" and value < 0):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, value)
        if self.partition not in _PARTITIONS:
            raise ValueError(f"unknown worker partition {self.partition!r}")
        if not isinstance(self.reserve_coordinator, bool):
            raise ValueError("reserve_coordinator must be a bool")
        if self.rank >= self.world_size:
            raise ValueError("rank must be less than world_size")

    def plan(self, topology: CpuTopology) -> WorkerPlan:
        if not topology.allowed_cpus:
            raise ValueError("cannot plan workers: no allowed CPUs")
        selected_cores, partition_applied, numa_node, partition_reason = self._partition(topology)
        if not selected_cores:
            raise ValueError(
                f"worker partition {partition_applied!r} has no allowed physical cores "
                f"for rank {self.rank}/{self.world_size}"
            )
        selected_representatives = tuple(core.representative for core in selected_cores)
        fallback_reasons = [
            reason for reason in (topology.fallback_reason, partition_reason) if reason
        ]

        auto = self.requested_threads in (None, 0)
        if auto:
            if topology.confidence == "full":
                usable = selected_representatives
            else:
                usable = selected_representatives[:1]
                fallback_reasons.append(
                    "physical topology is not fully known; auto mode uses one logical worker"
                )
            requested = self.requested_threads
            worker_count = len(usable)
        else:
            requested = self.requested_threads
            if requested > len(selected_representatives):
                raise ValueError(
                    f"requested_threads={requested} exceeds physical-core capacity of "
                    f"{len(selected_representatives)} for the selected partition"
                )
            usable = selected_representatives
            worker_count = requested

        coordinator: int | None = None
        if self.reserve_coordinator:
            if auto:
                if len(usable) < 2:
                    raise ValueError(
                        "coordinator reservation requires one additional allowed physical core"
                    )
                worker_count = len(usable) - 1
            if len(usable) <= worker_count:
                raise ValueError(
                    "coordinator reservation requires one additional allowed physical core"
                )
            coordinator = usable[worker_count]
            workers = usable[:worker_count]
        else:
            workers = usable[:worker_count]
        if not workers:
            raise ValueError("worker plan must retain at least one worker")

        return WorkerPlan(
            allowed_cpus=topology.allowed_cpus,
            physical_core_cpus=topology.physical_core_cpus,
            partition_cpus=selected_representatives,
            worker_cpus=workers,
            requested_threads=requested,
            effective_threads=len(workers),
            coordinator_cpu=coordinator,
            rank=self.rank,
            world_size=self.world_size,
            partition_requested=self.partition,
            partition_applied=partition_applied,
            numa_node=numa_node,
            topology_confidence=topology.confidence,
            topology_source=topology.source,
            affinity_status="planned-unverified",
            fallback_reason="; ".join(dict.fromkeys(fallback_reasons)) or None,
        )

    def _partition(
        self, topology: CpuTopology
    ) -> tuple[tuple[PhysicalCore, ...], str, int | None, str | None]:
        cores = topology.cores
        if self.partition == "mask":
            return cores, self.partition, _single_numa_node(cores), None
        if self.partition == "contiguous":
            selected = _split_contiguous(cores, self.rank, self.world_size)
            return selected, "contiguous", _single_numa_node(selected), None

        nodes = sorted({core.numa_node for core in cores if core.numa_node is not None})
        metadata_complete = all(core.numa_node is not None for core in cores)
        if metadata_complete and len(nodes) == self.world_size:
            node = nodes[self.rank]
            selected = tuple(core for core in cores if core.numa_node == node)
            if selected:
                return selected, "numa", node, None
        selected = _split_contiguous(cores, self.rank, self.world_size)
        if not metadata_complete:
            reason = "NUMA metadata is incomplete; used contiguous partition"
        else:
            reason = (
                f"NUMA node count {len(nodes)} does not match world_size {self.world_size}; "
                "used contiguous partition"
            )
        return (
            selected,
            "contiguous",
            _single_numa_node(selected),
            reason,
        )


def _single_numa_node(cores: Iterable[PhysicalCore]) -> int | None:
    nodes = {core.numa_node for core in cores}
    return next(iter(nodes)) if len(nodes) == 1 else None


@dataclass(frozen=True, slots=True)
class WorkerPlan:
    """Immutable, observable worker selection; affinity remains unverified."""

    allowed_cpus: tuple[int, ...]
    physical_core_cpus: tuple[int, ...]
    partition_cpus: tuple[int, ...]
    worker_cpus: tuple[int, ...]
    requested_threads: int | None
    effective_threads: int
    coordinator_cpu: int | None
    rank: int
    world_size: int
    partition_requested: str
    partition_applied: str
    numa_node: int | None
    topology_confidence: TopologyConfidence
    topology_source: str
    affinity_status: AffinityStatus
    fallback_reason: str | None

    def __post_init__(self) -> None:
        for name in ("allowed_cpus", "physical_core_cpus", "partition_cpus", "worker_cpus"):
            object.__setattr__(self, name, _cpu_tuple(getattr(self, name), name=name))
        if self.effective_threads != len(self.worker_cpus) or self.effective_threads < 1:
            raise ValueError("effective_threads must equal a non-empty worker_cpus set")
        if len(self.worker_cpus) != len(set(self.worker_cpus)):
            raise ValueError("worker_cpus must not contain duplicates")
        if not set(self.worker_cpus).issubset(self.allowed_cpus):
            raise ValueError("worker_cpus must be contained in allowed_cpus")
        if self.coordinator_cpu is not None:
            if self.coordinator_cpu in self.worker_cpus:
                raise ValueError("coordinator_cpu must not overlap worker_cpus")
            if self.coordinator_cpu not in self.allowed_cpus:
                raise ValueError("coordinator_cpu must be contained in allowed_cpus")

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed_cpus": list(self.allowed_cpus),
            "physical_core_cpus": list(self.physical_core_cpus),
            "partition_cpus": list(self.partition_cpus),
            "worker_cpus": list(self.worker_cpus),
            "requested_threads": self.requested_threads,
            "effective_threads": self.effective_threads,
            "coordinator_cpu": self.coordinator_cpu,
            "rank": self.rank,
            "world_size": self.world_size,
            "partition_requested": self.partition_requested,
            "partition_applied": self.partition_applied,
            "numa_node": self.numa_node,
            "topology_confidence": self.topology_confidence,
            "topology_source": self.topology_source,
            "affinity_status": self.affinity_status,
            "fallback_reason": self.fallback_reason,
        }


def plan_workers(topology: CpuTopology, policy: WorkerAffinityPolicy | None = None) -> WorkerPlan:
    """Convenience wrapper for callers that do not need to retain the policy object."""
    return (policy or WorkerAffinityPolicy()).plan(topology)


__all__ = [
    "AffinityStatus",
    "CpuTopology",
    "PhysicalCore",
    "TopologyConfidence",
    "WorkerAffinityPolicy",
    "WorkerPlan",
    "discover_cpu_topology",
    "plan_workers",
]
