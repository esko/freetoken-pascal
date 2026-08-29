"""No-GPU smoke coverage for the compiled executor's affinity report."""

from __future__ import annotations

import ctypes
import os

import pytest


def _new_executor(ext, core_ids: list[int]):
    return ext.CpuMoeExecutor(
        num_threads=len(core_ids),
        num_layers=1,
        num_experts=1,
        top_k=1,
        hidden_size=32,
        inter_size=32,
        max_tokens=1,
        activation_id=0,
        apply_router_weight_on_input=0,
        weight_format=0,
        gate_up_ptr=0,
        down_ptr=0,
        gate_up_scale_ptr=0,
        gate_up_global_ptr=0,
        down_scale_ptr=0,
        down_global_ptr=0,
        gate_up_bias_ptr=0,
        down_bias_ptr=0,
        swiglu_alpha=1.702,
        swiglu_limit=float("inf"),
        core_ids=core_ids,
    )


def _noop_task(executor):
    """Build a task whose invalid route avoids dereferencing the null bank tables."""
    x = (ctypes.c_uint16 * 32)()
    ids = (ctypes.c_int32 * 1)(-1)
    weights = (ctypes.c_float * 1)(0.0)
    y = (ctypes.c_uint16 * 32)()
    task = executor.create_task(
        0,
        1,
        ctypes.addressof(x),
        ctypes.addressof(ids),
        ctypes.addressof(weights),
        ctypes.addressof(y),
    )
    return task, (x, ids, weights, y)


def _load_extension():
    torch = pytest.importorskip("torch")
    ext = pytest.importorskip("freetoken.kernel._cpu_moe")
    if not hasattr(ext.CpuMoeExecutor, "affinity_report"):
        pytest.skip("compiled extension predates the affinity report API")
    if not hasattr(ext.CpuMoeExecutor, "stop_flag_coordinator"):
        pytest.skip("compiled extension predates coordinator shutdown API")
    return torch, ext


def test_native_affinity_report_verifies_workers_and_optional_coordinator() -> None:
    torch, ext = _load_extension()
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        pytest.skip("the test process has no allowed CPUs")
    worker_cpus = allowed[:2]

    before = set(os.sched_getaffinity(0))
    executor = _new_executor(ext, worker_cpus)
    report = executor.affinity_report()
    assert report["status"] == "verified"
    assert report["worker_requested_cpus"] == worker_cpus
    assert report["worker_observed_affinity_cpus"] == worker_cpus
    assert report["worker_affinity_errors"] == [0] * len(worker_cpus)
    if len(allowed) >= 3:
        coordinator_cpu = allowed[2]
        flags = (ctypes.c_int64 * 1)()
        executor.start_flag_coordinator(
            ctypes.addressof(flags),
            ctypes.addressof(flags),
            1,
            coordinator_cpu,
        )
        report = executor.affinity_report()
        assert report["coordinator_requested_cpu"] == coordinator_cpu
        assert report["coordinator_observed_affinity_cpu"] == coordinator_cpu
        assert report["coordinator_affinity_error"] == 0
        assert report["coordinator_affinity_verified"] is True
        assert report["status"] == "verified"
        executor.stop_flag_coordinator()
        executor.stop_flag_coordinator()
        assert executor.affinity_report() == report
    del executor
    assert set(os.sched_getaffinity(0)) == before
    del torch


def test_native_invalid_cpu_is_reported_without_changing_parent_affinity() -> None:
    torch, ext = _load_extension()
    before = set(os.sched_getaffinity(0))
    executor = _new_executor(ext, [1024])
    report = executor.affinity_report()
    assert report["status"] == "failed"
    assert report["worker_affinity_errors"][0] != 0
    assert report["worker_observed_affinity_cpus"] == [-1]
    assert report["worker_pool_usable"] is True
    assert "worker" in report["reason"]
    executor.ensure_worker_pool_usable()
    task, keepalive = _noop_task(executor)
    executor.run_task(task)
    del keepalive
    del executor
    assert set(os.sched_getaffinity(0)) == before
    del torch


def test_native_coordinator_shutdown_is_idempotent() -> None:
    torch, ext = _load_extension()
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("coordinator shutdown test requires two allowed CPUs")
    executor = _new_executor(ext, [allowed[0]])
    flags = (ctypes.c_int64 * 1)()
    executor.start_flag_coordinator(ctypes.addressof(flags), ctypes.addressof(flags), 1, allowed[1])
    before = executor.affinity_report()
    executor.stop_flag_coordinator()
    executor.stop_flag_coordinator()
    assert executor.affinity_report() == before
    del executor
    del torch


def test_native_failed_coordinator_is_stopped_before_fallback() -> None:
    torch, ext = _load_extension()
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        pytest.skip("coordinator failure test requires an allowed CPU")
    executor = _new_executor(ext, [allowed[0]])
    flags = (ctypes.c_int64 * 1)()
    executor.start_flag_coordinator(ctypes.addressof(flags), ctypes.addressof(flags), 1, 1024)
    report = executor.affinity_report()
    assert report["status"] == "failed"
    assert report["coordinator_affinity_error"] != 0
    executor.stop_flag_coordinator()
    executor.stop_flag_coordinator()
    del executor
    del torch


def test_native_coordinator_reentry_is_rejected() -> None:
    torch, ext = _load_extension()
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("coordinator re-entry test requires two allowed CPUs")
    executor = _new_executor(ext, [allowed[0]])
    flags = (ctypes.c_int64 * 1)()
    executor.start_flag_coordinator(ctypes.addressof(flags), ctypes.addressof(flags), 1, allowed[1])
    first = executor.affinity_report()
    with pytest.raises(RuntimeError, match="already started"):
        executor.start_flag_coordinator(
            ctypes.addressof(flags), ctypes.addressof(flags), 1, allowed[1]
        )
    assert executor.affinity_report() == first
    del executor
    del torch
