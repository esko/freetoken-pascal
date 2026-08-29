"""No-GPU smoke coverage for the compiled executor's affinity report."""

from __future__ import annotations

import ctypes
import os

import pytest


def test_native_affinity_report_verifies_a_single_worker() -> None:
    torch = pytest.importorskip("torch")
    ext = pytest.importorskip("freetoken.kernel._cpu_moe")
    if not hasattr(ext.CpuMoeExecutor, "affinity_report"):
        pytest.skip("compiled extension predates the affinity report API")
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        pytest.skip("the test process has no allowed CPUs")

    executor = ext.CpuMoeExecutor(
        num_threads=1,
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
        core_ids=[allowed[0]],
    )
    report = executor.affinity_report()
    assert report["status"] == "verified"
    assert report["worker_requested_cpus"] == [allowed[0]]
    assert report["worker_actual_cpus"] == [allowed[0]]
    assert report["worker_affinity_errors"] == [0]
    if len(allowed) >= 2:
        flags = (ctypes.c_int64 * 1)()
        executor.start_flag_coordinator(
            ctypes.addressof(flags),
            ctypes.addressof(flags),
            1,
            allowed[1],
        )
        report = executor.affinity_report()
        assert report["coordinator_requested_cpu"] == allowed[1]
        assert report["coordinator_actual_cpu"] == allowed[1]
        assert report["coordinator_affinity_error"] == 0
        assert report["coordinator_affinity_verified"] is True
        assert report["status"] == "verified"
    del executor
    del torch
