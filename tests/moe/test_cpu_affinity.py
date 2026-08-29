from __future__ import annotations

import pytest
from freetoken.moe.cpu_affinity import (
    affinity_telemetry,
    resolve_cpu_moe_affinity,
)
from freetoken.moe.cpu_topology import CpuTopology, PhysicalCore


def _topology(*cores: tuple[int, int | None]) -> CpuTopology:
    return CpuTopology(
        allowed_cpus=tuple(cpu for cpu, _ in cores),
        cores=tuple(
            PhysicalCore(
                key=f"fixture:{cpu}",
                representative=cpu,
                logical_cpus=(cpu,),
                siblings=(cpu,),
                numa_node=node,
            )
            for cpu, node in cores
        ),
        confidence="full",
        source="fixture",
    )


def test_auto_cpu_moe_plan_is_exact_and_has_no_coordinator_without_flag_sync() -> None:
    selection = resolve_cpu_moe_affinity(
        0,
        flag_sync=False,
        topology=_topology((2, 0), (8, 0), (13, 1)),
    )

    assert selection.plan.worker_cpus == (2, 8, 13)
    assert selection.plan.effective_threads == 3
    assert selection.plan.coordinator_cpu is None
    assert selection.flag_sync is False
    assert selection.fallback_reason is None


def test_explicit_cpu_moe_over_capacity_fails_before_native_construction() -> None:
    with pytest.raises(ValueError, match="physical-core capacity of 2"):
        resolve_cpu_moe_affinity(3, flag_sync=False, topology=_topology((2, 0), (8, 0)))


def test_flag_sync_reserves_a_coordinator_only_when_capacity_allows() -> None:
    selection = resolve_cpu_moe_affinity(
        0,
        flag_sync=True,
        topology=_topology((2, 0), (8, 0), (13, 1)),
    )

    assert selection.flag_sync is True
    assert selection.plan.worker_cpus == (2, 8)
    assert selection.plan.coordinator_cpu == 13


def test_flag_sync_falls_back_to_host_func_when_no_spare_core_exists() -> None:
    selection = resolve_cpu_moe_affinity(
        0,
        flag_sync=True,
        topology=_topology((2, 0)),
    )

    assert selection.flag_sync is False
    assert selection.plan.worker_cpus == (2,)
    assert selection.plan.coordinator_cpu is None
    assert selection.fallback_reason is not None
    assert "coordinator" in selection.fallback_reason


def test_affinity_telemetry_distinguishes_plan_from_native_verification() -> None:
    selection = resolve_cpu_moe_affinity(
        1,
        flag_sync=False,
        topology=_topology((8, 0)),
    )

    planned = affinity_telemetry(selection)
    assert planned["affinity_status"] == "planned-unverified"
    assert planned["native_affinity_status"] == "unavailable"

    verified = affinity_telemetry(
        selection,
        {
            "status": "verified",
            "worker_observed_affinity_cpus": [8],
            "worker_affinity_errors": [0],
        },
    )
    assert verified["affinity_status"] == "verified"
    assert verified["native_affinity_status"] == "verified"
    assert verified["worker_observed_affinity_cpus"] == [8]

    failed = affinity_telemetry(selection, {"status": "failed", "reason": "EINVAL"})
    assert failed["affinity_status"] == "fallback"
    assert failed["fallback_reason"] == "EINVAL"


def test_affinity_fallback_reason_is_derived_from_native_error_fields() -> None:
    selection = resolve_cpu_moe_affinity(
        1,
        flag_sync=False,
        topology=_topology((8, 0)),
    )

    failed = affinity_telemetry(
        selection,
        {
            "status": "failed",
            "worker_affinity_errors": [22],
            "reason": "worker affinity error 22 (EINVAL)",
        },
    )
    assert failed["affinity_status"] == "fallback"
    assert failed["fallback_reason"] == "worker affinity error 22 (EINVAL)"
