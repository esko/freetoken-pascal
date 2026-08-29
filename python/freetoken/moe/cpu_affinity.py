"""CPU affinity selection for the compiled CPU MoE executor.

The topology module describes a plan but never changes process affinity.  This
module is the small policy boundary used by the compiled executor: it validates
the requested worker count against visible physical-core capacity and reserves
one additional core for the flag-sync coordinator only when that capacity is
available.  Native enforcement and verification are reported separately.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from freetoken.moe.cpu_topology import (
    CpuTopology,
    WorkerAffinityPolicy,
    WorkerPlan,
    discover_cpu_topology,
)


@dataclass(frozen=True, slots=True)
class CpuMoeAffinitySelection:
    """Validated worker plan plus the selected synchronization mode."""

    topology: CpuTopology
    plan: WorkerPlan
    flag_sync: bool
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.flag_sync, bool):
            raise ValueError("flag_sync must be a bool")

    def as_dict(self) -> dict[str, object]:
        """Return the requested/applied planning fields for startup telemetry."""
        return {
            "requested_threads": self.plan.requested_threads,
            "effective_threads": self.plan.effective_threads,
            "worker_cpus": list(self.plan.worker_cpus),
            "coordinator_cpu": self.plan.coordinator_cpu,
            "flag_sync": self.flag_sync,
            "topology_confidence": self.plan.topology_confidence,
            "topology_source": self.plan.topology_source,
            "partition_requested": self.plan.partition_requested,
            "partition_applied": self.plan.partition_applied,
            "fallback_reason": self.fallback_reason or self.plan.fallback_reason,
            "affinity_status": "planned-unverified",
        }


def resolve_cpu_moe_affinity(
    requested_threads: int,
    *,
    flag_sync: bool,
    topology: CpuTopology | None = None,
) -> CpuMoeAffinitySelection:
    """Resolve an exact compiled-executor worker plan.

    The process affinity mask discovered by :func:`discover_cpu_topology` is
    authoritative.  A positive request that exceeds visible physical-core
    capacity raises before native executor construction.  Flag synchronization
    is opportunistic only with respect to coordinator capacity: workers retain
    their validated plan and the host-function path is selected when no spare
    physical core exists.
    """
    if not isinstance(flag_sync, bool):
        raise ValueError("flag_sync must be a bool")
    topology = topology or discover_cpu_topology()
    worker_plan = WorkerAffinityPolicy(requested_threads=requested_threads).plan(topology)
    if not flag_sync:
        return CpuMoeAffinitySelection(topology, worker_plan, False)

    try:
        coordinator_plan = WorkerAffinityPolicy(
            requested_threads=requested_threads,
            reserve_coordinator=True,
        ).plan(topology)
    except ValueError as error:
        # A coordinator is an optional optimization.  Preserve all sizing errors
        # for the worker request itself; only the explicit extra-core failure falls
        # back to the functional host-function path.
        if "coordinator reservation" not in str(error):
            raise
        return CpuMoeAffinitySelection(
            topology,
            worker_plan,
            False,
            fallback_reason=str(error),
        )
    return CpuMoeAffinitySelection(topology, coordinator_plan, True)


def affinity_telemetry(
    selection: CpuMoeAffinitySelection,
    native_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Combine the planned selection with optional native startup verification.

    No native report is treated as unverified.  A report can upgrade the status
    only when the native helper verified the exact masks; failed enforcement is
    explicitly surfaced as fallback and never relabeled as successful pinning.
    """
    report = dict(native_report) if native_report is not None else {}
    native_status = str(report.get("status", "unavailable"))
    if native_status == "verified":
        status = "verified"
    elif native_status in {"failed", "unsupported"}:
        status = "fallback"
    else:
        status = "planned-unverified"

    telemetry = selection.as_dict()
    telemetry.update(
        {
            "affinity_status": status,
            "planned_affinity_status": "planned-unverified",
            "native_affinity_status": native_status,
        }
    )
    for key in (
        "worker_requested_cpus",
        "worker_observed_affinity_cpus",
        "worker_affinity_errors",
        "worker_affinity_verified",
        "workers_ready",
        "coordinator_requested_cpu",
        "coordinator_observed_affinity_cpu",
        "coordinator_affinity_error",
        "coordinator_affinity_verified",
        "coordinator_ready",
    ):
        if key in report:
            telemetry[key] = report[key]
    if status == "fallback" and report.get("reason"):
        telemetry["fallback_reason"] = report["reason"]
    if report.get("reason"):
        telemetry["native_affinity_reason"] = report["reason"]
    return telemetry


__all__ = [
    "CpuMoeAffinitySelection",
    "affinity_telemetry",
    "resolve_cpu_moe_affinity",
]
