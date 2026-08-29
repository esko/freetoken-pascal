from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import freetoken.moe.cpu_topology as cpu_topology
import pytest
from freetoken.moe.cpu_topology import (
    CpuTopology,
    PhysicalCore,
    WorkerAffinityPolicy,
    discover_cpu_topology,
)


def _gorilla_sysfs(root: Path, *, cpus: tuple[int, ...] = tuple(range(48))) -> None:
    for cpu in cpus:
        cpu_root = root / f"cpu{cpu}"
        topology = cpu_root / "topology"
        topology.mkdir(parents=True)
        physical_core = cpu if cpu < 24 else cpu - 24
        node = 0 if physical_core < 12 else 1
        (topology / "thread_siblings_list").write_text(
            f"{physical_core},{physical_core + 24}", encoding="ascii"
        )
        (topology / "physical_package_id").write_text(str(node), encoding="ascii")
        (topology / "core_id").write_text(str(physical_core), encoding="ascii")
        (cpu_root / f"node{node}").mkdir()


def _topology(*cores: tuple[int, tuple[int, ...], int | None]) -> CpuTopology:
    allowed = tuple(cpu for representative, siblings, _ in cores for cpu in siblings)
    return CpuTopology(
        allowed_cpus=allowed,
        cores=tuple(
            PhysicalCore(
                key=f"fixture:{representative}",
                representative=representative,
                logical_cpus=siblings,
                siblings=siblings,
                numa_node=node,
            )
            for representative, siblings, node in cores
        ),
        confidence="full",
        source="fixture",
    )


def test_discovery_deduplicates_smt_and_preserves_allowed_noncontiguous_ids(tmp_path: Path) -> None:
    _gorilla_sysfs(tmp_path)

    topology = discover_cpu_topology(
        allowed_cpus=(0, 2, 24, 26, 35, 47),
        sysfs_root=tmp_path,
    )

    assert topology.allowed_cpus == (0, 2, 24, 26, 35, 47)
    assert topology.physical_core_cpus == (0, 2, 35, 47)
    assert topology.cores[0].logical_cpus == (0, 24)
    assert topology.cores[-2].logical_cpus == (35,)
    assert topology.cores[-1].logical_cpus == (47,)
    assert topology.confidence == "full"
    assert topology.source == "provided-affinity+sysfs"


def test_discovery_single_allowed_sibling_is_one_core(tmp_path: Path) -> None:
    _gorilla_sysfs(tmp_path, cpus=(24, 25, 26))

    topology = discover_cpu_topology(allowed_cpus=(24, 25), sysfs_root=tmp_path)

    assert topology.physical_core_cpus == (24, 25)
    assert all(core.logical_cpus == (core.representative,) for core in topology.cores)


def test_missing_or_malformed_topology_is_explicitly_logical_only(tmp_path: Path) -> None:
    cpu_root = tmp_path / "cpu3" / "topology"
    cpu_root.mkdir(parents=True)
    (cpu_root / "thread_siblings_list").write_text("not-a-cpu-list", encoding="ascii")

    topology = discover_cpu_topology(allowed_cpus=(3, 7), sysfs_root=tmp_path)

    assert topology.physical_core_cpus == (3, 7)
    assert topology.confidence == "logical-only"
    assert topology.fallback_reason is not None
    assert "cpu 3" in topology.fallback_reason


def test_mixed_package_core_and_sibling_metadata_keeps_one_core_and_is_partial(
    tmp_path: Path,
) -> None:
    cpu_root = tmp_path / "cpu24" / "topology"
    cpu_root.mkdir(parents=True)
    (cpu_root / "thread_siblings_list").write_text("0,24", encoding="ascii")
    (cpu_root / "physical_package_id").write_text("0", encoding="ascii")
    (cpu_root / "core_id").write_text("0", encoding="ascii")

    cpu_root = tmp_path / "cpu0" / "topology"
    cpu_root.mkdir(parents=True)
    (cpu_root / "physical_package_id").write_text("0", encoding="ascii")
    (cpu_root / "core_id").write_text("0", encoding="ascii")

    topology = discover_cpu_topology(allowed_cpus=(0, 24), sysfs_root=tmp_path)

    assert topology.physical_core_cpus == (0,)
    assert topology.cores[0].logical_cpus == (0, 24)
    assert topology.confidence == "partial"
    assert topology.fallback_reason is not None
    assert "inferred" in topology.fallback_reason


def test_sibling_list_omitting_current_cpu_degrades_without_splitting(tmp_path: Path) -> None:
    for cpu, siblings in ((0, "24"), (24, "0,24")):
        topology_root = tmp_path / f"cpu{cpu}" / "topology"
        topology_root.mkdir(parents=True)
        (topology_root / "thread_siblings_list").write_text(siblings, encoding="ascii")

    topology = discover_cpu_topology(allowed_cpus=(0, 24), sysfs_root=tmp_path)

    assert topology.physical_core_cpus == (0,)
    assert topology.cores[0].logical_cpus == (0, 24)
    assert topology.confidence == "partial"
    assert topology.fallback_reason is not None
    assert "omits cpu 0" in topology.fallback_reason


def test_contradictory_singleton_siblings_merge_by_package_core_and_are_partial(
    tmp_path: Path,
) -> None:
    for cpu in (0, 24):
        topology_root = tmp_path / f"cpu{cpu}" / "topology"
        topology_root.mkdir(parents=True)
        (topology_root / "thread_siblings_list").write_text(str(cpu), encoding="ascii")
        (topology_root / "physical_package_id").write_text("0", encoding="ascii")
        (topology_root / "core_id").write_text("0", encoding="ascii")

    topology = discover_cpu_topology(allowed_cpus=(0, 24), sysfs_root=tmp_path)

    assert topology.physical_core_cpus == (0,)
    assert topology.cores[0].logical_cpus == (0, 24)
    assert topology.confidence == "partial"
    assert topology.fallback_reason is not None
    assert "conflicting sibling topology" in topology.fallback_reason
    with pytest.raises(ValueError, match="physical-core capacity of 1"):
        WorkerAffinityPolicy(requested_threads=2).plan(topology)


def test_one_sibling_component_with_conflicting_package_core_ids_is_partial(
    tmp_path: Path,
) -> None:
    for cpu, core_id in ((0, 0), (24, 1)):
        topology_root = tmp_path / f"cpu{cpu}" / "topology"
        topology_root.mkdir(parents=True)
        (topology_root / "thread_siblings_list").write_text("0,24", encoding="ascii")
        (topology_root / "physical_package_id").write_text("0", encoding="ascii")
        (topology_root / "core_id").write_text(str(core_id), encoding="ascii")

    topology = discover_cpu_topology(allowed_cpus=(0, 24), sysfs_root=tmp_path)

    assert topology.physical_core_cpus == (0,)
    assert topology.cores[0].logical_cpus == (0, 24)
    assert topology.confidence == "partial"
    assert topology.fallback_reason is not None
    assert "conflicting sibling topology" in topology.fallback_reason


def test_affinity_oserror_uses_explicit_cpu_count_fallback(tmp_path: Path) -> None:
    def fail_affinity() -> set[int]:
        raise OSError("affinity unavailable")

    topology = discover_cpu_topology(
        affinity_getter=fail_affinity,
        cpu_count=3,
        sysfs_root=tmp_path,
    )

    assert topology.allowed_cpus == (0, 1, 2)
    assert topology.source == "cpu-count+sysfs"
    assert topology.fallback_reason is not None
    assert "affinity unavailable" in topology.fallback_reason


@pytest.mark.parametrize("invalid_count", [-1, True, 1.5, "3"])
def test_explicit_invalid_cpu_count_is_rejected(tmp_path: Path, invalid_count: object) -> None:
    def fail_affinity() -> set[int]:
        raise OSError("affinity unavailable")

    with pytest.raises(ValueError, match="cpu_count"):
        discover_cpu_topology(
            affinity_getter=fail_affinity,
            cpu_count=invalid_count,  # type: ignore[arg-type]
            sysfs_root=tmp_path,
        )


def test_explicit_invalid_cpu_count_is_rejected_before_affinity_lookup() -> None:
    with pytest.raises(ValueError, match="cpu_count"):
        discover_cpu_topology(allowed_cpus=(0,), cpu_count=-1)


def test_os_cpu_count_none_is_an_empty_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_affinity() -> set[int]:
        raise OSError("affinity unavailable")

    monkeypatch.setattr(cpu_topology.os, "cpu_count", lambda: None)

    topology = discover_cpu_topology(affinity_getter=fail_affinity)

    assert topology.allowed_cpus == ()
    assert topology.cores == ()


def test_empty_allowed_set_is_represented_and_cannot_be_planned(tmp_path: Path) -> None:
    topology = discover_cpu_topology(allowed_cpus=(), sysfs_root=tmp_path)

    assert topology.allowed_cpus == ()
    assert topology.cores == ()
    with pytest.raises(ValueError, match="no allowed CPUs"):
        WorkerAffinityPolicy().plan(topology)


def test_gorilla_fixture_has_24_physical_cores_and_two_numa_domains(tmp_path: Path) -> None:
    _gorilla_sysfs(tmp_path)

    topology = discover_cpu_topology(allowed_cpus=tuple(range(48)), sysfs_root=tmp_path)

    assert topology.physical_core_cpus == tuple(range(24))
    assert tuple(core.numa_node for core in topology.cores[:12]) == (0,) * 12
    assert tuple(core.numa_node for core in topology.cores[12:]) == (1,) * 12


def test_default_plan_uses_entire_process_mask_without_rank_heuristic() -> None:
    topology = _topology((2, (2,), 0), (8, (8,), 1), (13, (13,), 1))

    plan = WorkerAffinityPolicy(rank=1, world_size=2).plan(topology)

    assert plan.partition_requested == "mask"
    assert plan.partition_applied == "mask"
    assert plan.worker_cpus == (2, 8, 13)
    assert plan.effective_threads == 3
    assert plan.affinity_status == "planned-unverified"


def test_explicit_contiguous_partition_is_exact_and_noncontiguous() -> None:
    topology = _topology(
        (2, (2,), 0),
        (8, (8,), 0),
        (13, (13,), 1),
        (21, (21,), 1),
        (34, (34,), 1),
    )

    first = WorkerAffinityPolicy(rank=0, world_size=2, partition="contiguous").plan(topology)
    second = WorkerAffinityPolicy(rank=1, world_size=2, partition="contiguous").plan(topology)

    assert first.worker_cpus == (2, 8, 13)
    assert second.worker_cpus == (21, 34)
    assert set(first.worker_cpus).isdisjoint(second.worker_cpus)


def test_explicit_numa_partition_is_local_and_labeled_planned() -> None:
    topology = _topology(
        (2, (2,), 0),
        (8, (8,), 0),
        (13, (13,), 1),
        (21, (21,), 1),
    )

    first = WorkerAffinityPolicy(rank=0, world_size=2, partition="numa").plan(topology)
    second = WorkerAffinityPolicy(rank=1, world_size=2, partition="numa").plan(topology)

    assert first.worker_cpus == (2, 8)
    assert second.worker_cpus == (13, 21)
    assert first.numa_node == 0
    assert second.numa_node == 1
    assert first.affinity_status == "planned-unverified"


def test_numa_partition_falls_back_when_metadata_is_unavailable() -> None:
    topology = _topology((2, (2,), None), (8, (8,), None), (13, (13,), None))

    plan = WorkerAffinityPolicy(rank=1, world_size=2, partition="numa").plan(topology)

    assert plan.partition_applied == "contiguous"
    assert plan.fallback_reason is not None
    assert "NUMA" in plan.fallback_reason


@pytest.mark.parametrize(
    "nodes",
    [
        (0, 1, None),
        (0, 1, 2),
    ],
)
def test_numa_partition_keeps_unknown_or_extra_nodes_in_contiguous_fallback(
    nodes: tuple[int | None, ...],
) -> None:
    topology = _topology(
        *((cpu, (cpu,), node) for cpu, node in zip((2, 8, 13), nodes, strict=True))
    )

    first = WorkerAffinityPolicy(rank=0, world_size=2, partition="numa").plan(topology)
    second = WorkerAffinityPolicy(rank=1, world_size=2, partition="numa").plan(topology)

    assert first.partition_applied == "contiguous"
    assert second.partition_applied == "contiguous"
    assert set(first.partition_cpus + second.partition_cpus) == {2, 8, 13}
    assert first.fallback_reason is not None
    assert "contiguous" in first.fallback_reason


@pytest.mark.parametrize("requested", [True, -1])
def test_invalid_thread_count_is_rejected(requested: int | bool) -> None:
    with pytest.raises(ValueError, match="requested_threads"):
        WorkerAffinityPolicy(requested_threads=requested)


def test_explicit_thread_count_is_exact_and_over_capacity_fails() -> None:
    topology = _topology((2, (2,), 0), (8, (8,), 0), (13, (13,), 1))

    plan = WorkerAffinityPolicy(requested_threads=2).plan(topology)
    assert plan.worker_cpus == (2, 8)
    assert plan.effective_threads == 2

    with pytest.raises(ValueError, match="physical-core capacity of 3"):
        WorkerAffinityPolicy(requested_threads=4).plan(topology)


def test_auto_and_coordinator_reservation_are_modeled_separately() -> None:
    topology = _topology((2, (2,), 0), (8, (8,), 0), (13, (13,), 1))

    plan = WorkerAffinityPolicy(reserve_coordinator=True).plan(topology)

    assert plan.requested_threads == 0
    assert plan.worker_cpus == (2, 8)
    assert plan.effective_threads == 2
    assert plan.coordinator_cpu == 13
    assert plan.coordinator_cpu not in plan.worker_cpus

    with pytest.raises(ValueError, match="coordinator"):
        WorkerAffinityPolicy(requested_threads=3, reserve_coordinator=True).plan(topology)


@pytest.mark.parametrize(
    ("rank", "world_size"),
    [(-1, 2), (2, 2), (0, 0)],
)
def test_invalid_rank_partition_is_rejected(rank: int, world_size: int) -> None:
    topology = _topology((2, (2,), 0), (8, (8,), 0))

    with pytest.raises(ValueError, match=r"rank|world_size"):
        WorkerAffinityPolicy(rank=rank, world_size=world_size).plan(topology)


def test_plan_as_dict_exposes_requested_effective_and_unverified_state() -> None:
    topology = _topology((2, (2,), 0), (8, (8,), 0))

    values = WorkerAffinityPolicy(requested_threads=1).plan(topology).as_dict()

    assert values["requested_threads"] == 1
    assert values["effective_threads"] == 1
    assert values["worker_cpus"] == [2]
    assert values["affinity_status"] == "planned-unverified"


def test_topology_and_policy_records_are_immutable() -> None:
    topology = _topology((2, (2,), 0))
    policy = WorkerAffinityPolicy(requested_threads=1)

    with pytest.raises(FrozenInstanceError):
        topology.allowed_cpus = (8,)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        policy.requested_threads = 2  # type: ignore[misc]


def test_topology_rejects_overlapping_logical_core_membership() -> None:
    with pytest.raises(ValueError, match="overlapping logical CPU"):
        CpuTopology(
            allowed_cpus=(2, 8),
            cores=(
                PhysicalCore(
                    key="fixture:first",
                    representative=2,
                    logical_cpus=(2, 8),
                    siblings=(2, 8),
                ),
                PhysicalCore(
                    key="fixture:second",
                    representative=8,
                    logical_cpus=(8,),
                    siblings=(8,),
                ),
            ),
            confidence="full",
            source="fixture",
        )


def test_topology_and_plan_are_canonical_across_core_input_order() -> None:
    first = _topology((13, (13,), 1), (2, (2,), 0), (8, (8,), 0))
    second = _topology((8, (8,), 0), (13, (13,), 1), (2, (2,), 0))

    assert first == second
    assert tuple(core.representative for core in first.cores) == (2, 8, 13)
    assert WorkerAffinityPolicy(requested_threads=2).plan(first) == WorkerAffinityPolicy(
        requested_threads=2
    ).plan(second)
