from __future__ import annotations

import pytest
from freetoken.moe.cpu_affinity import (
    affinity_telemetry,
    native_flag_sync_supported,
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


def test_stale_native_extension_cannot_enable_flag_sync_without_shutdown_hook() -> None:
    class StaleExtension:
        def affinity_report(self):
            return {}

    class CurrentExtension:
        def affinity_report(self):
            return {}

        def stop_flag_coordinator(self):
            return None

    assert native_flag_sync_supported(StaleExtension) is False
    assert native_flag_sync_supported(CurrentExtension) is True
    assert native_flag_sync_supported(None) is False


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
    assert verified["flag_sync_requested"] is False
    assert verified["flag_sync"] is False
    assert verified["flag_sync_applied"] is False

    requested = resolve_cpu_moe_affinity(
        0,
        flag_sync=True,
        topology=_topology((8, 0), (13, 1)),
    )
    workers_only = affinity_telemetry(
        requested,
        {
            "status": "verified",
            "worker_observed_affinity_cpus": [8],
            "worker_affinity_errors": [0],
        },
    )
    assert workers_only["flag_sync_requested"] is True
    assert workers_only["flag_sync"] is False
    applied = affinity_telemetry(
        requested,
        {
            "status": "verified",
            "worker_observed_affinity_cpus": [8],
            "worker_affinity_errors": [0],
            "coordinator_requested_cpu": 13,
            "coordinator_ready": True,
        },
    )
    assert applied["flag_sync"] is True
    assert applied["flag_sync_applied"] is True

    failed = affinity_telemetry(selection, {"status": "failed", "reason": "EINVAL"})
    assert failed["affinity_status"] == "fallback"
    assert failed["fallback_reason"] == "EINVAL"

    coordinator_failed = affinity_telemetry(
        resolve_cpu_moe_affinity(0, flag_sync=True, topology=_topology((8, 0), (13, 1))),
        {
            "status": "failed",
            "coordinator_affinity_startup_timed_out": True,
            "reason": "coordinator affinity startup timed out",
        },
    )
    assert coordinator_failed["affinity_status"] == "fallback"
    assert coordinator_failed["flag_sync_requested"] is True
    assert coordinator_failed["flag_sync"] is False
    assert coordinator_failed["flag_sync_applied"] is False
    assert coordinator_failed["fallback_reason"] == "coordinator affinity startup timed out"

    timed_out = affinity_telemetry(
        selection,
        {"status": "timed-out", "workers_ready": False, "reason": "worker timeout"},
    )
    assert timed_out["affinity_status"] == "fallback"
    assert timed_out["flag_sync"] is False
    assert timed_out["fallback_reason"] == "worker timeout"


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
