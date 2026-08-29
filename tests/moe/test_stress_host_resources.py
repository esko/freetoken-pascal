"""H0 tests for the bounded host-resource stress harness."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from freetoken.moe import cpu_abi
from freetoken.moe.cpu_abi import ExecutionFailed

_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "stress_host_resources.py"
_SPEC = importlib.util.spec_from_file_location("stress_host_resources", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
stress_host_resources = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stress_host_resources
_SPEC.loader.exec_module(stress_host_resources)


def _deterministic(result: dict[str, object]) -> dict[str, object]:
    """Remove host observations whose values are intentionally environmental."""
    return {
        key: result[key]
        for key in (
            "schema_version",
            "claim_status",
            "seed",
            "iterations",
            "threads",
            "backend",
            "kernel_census",
            "telemetry",
            "busy_telemetry",
            "accepted",
            "busy",
            "cancelled",
            "recovered",
            "scenario_counts",
            "policy_accounting",
        )
    }


def test_stress_result_is_schema_complete_and_cleans_up() -> None:
    from freetoken.moe import q4_k

    original_q4 = q4_k.select_q4_k_primitive
    original_mixed = q4_k.select_mixed_gemv_primitive
    result = stress_host_resources.run_stress(
        seed=17,
        iterations=3,
        threads=2,
        include_resource_probes=False,
        allow_fallback=True,
    )

    assert result["schema_version"] == 2
    assert result["claim_status"] == "observation_only"
    assert result["seed"] == 17
    assert result["iterations"] == 3
    assert result["threads"] == 2
    # Five admissions per iteration: parity, Busy owner, cancellation,
    # recovery, and close-race owner.  Cancelled requests count as admitted.
    assert result["accepted"] == 15
    assert result["busy"] == 3
    assert result["cancelled"] == 3
    assert result["recovered"] == 3
    assert result["scenario_counts"] == {
        "serial_thread_parity": 3,
        "busy": 3,
        "cancelled": 3,
        "recovered": 3,
        "close_race": 3,
    }
    assert result["live_buffers"]["restored"] is True  # type: ignore[index]
    assert result["thread_counts"]["restored"] is True  # type: ignore[index]
    assert result["rss_bytes"]["sampled_max_after_iteration"] == max(  # type: ignore[index]
        result["rss_bytes"]["after_iteration_samples"]  # type: ignore[index]
    )
    assert q4_k.select_q4_k_primitive is original_q4
    assert q4_k.select_mixed_gemv_primitive is original_mixed


def test_seeded_core_observations_are_deterministic() -> None:
    first = stress_host_resources.run_stress(
        seed=91, iterations=2, threads=3, include_resource_probes=False, allow_fallback=True
    )
    second = stress_host_resources.run_stress(
        seed=91, iterations=2, threads=3, include_resource_probes=False, allow_fallback=True
    )
    assert _deterministic(first) == _deterministic(second)


def test_script_emits_json_without_test_imports() -> None:
    root = Path(__file__).parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "stress_host_resources.py"),
            "--seed",
            "5",
            "--iterations",
            "1",
            "--threads",
            "2",
            "--allow-fallback",
            "--no-resource-probes",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(completed.stdout)
    assert result["claim_status"] == "observation_only"
    assert result["seed"] == 5
    assert result["iterations"] == 1
    assert result["backend"] == "Q4KExecutor+mixed_avx2"
    assert result["kernel_census"]["backend"] == "mixed_avx2"  # type: ignore[index]
    assert result["kernel_census"]["fallback_reason"] is None  # type: ignore[index]
    assert result["telemetry"]["backend"] == "mixed_avx2"  # type: ignore[index]
    assert result["telemetry"]["error"] is None  # type: ignore[index]
    assert result["busy_telemetry"]["error"] == "Busy"  # type: ignore[index]


def test_invalid_stress_arguments_fail_closed() -> None:
    for kwargs in (
        {"iterations": 0},
        {"threads": 0},
        {"threads": stress_host_resources._ROUTES + 1},
        {"seed": True},
    ):
        try:
            stress_host_resources.run_stress(
                seed=kwargs.get("seed", 1),
                iterations=kwargs.get("iterations", 1),
                threads=kwargs.get("threads", 2),
                include_resource_probes=False,
                allow_fallback=True,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid stress arguments: {kwargs}")


def test_threaded_failure_drains_and_restores_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    before_live = stress_host_resources._live_buffer_ids()
    before_workers = tuple(stress_host_resources._thread_snapshot()["workers"])
    before_fds = stress_host_resources._fd_count()
    serial_calls = 0
    worker_entered = threading.Event()
    original = stress_host_resources._StressQ4Primitive.gemv

    def fail(self: object, *args: object, **kwargs: object) -> object:
        nonlocal serial_calls
        if threading.current_thread().name.startswith("freetoken-mixed"):
            worker_entered.set()
            raise RuntimeError("injected fake worker failure")
        serial_calls += 1
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(stress_host_resources._StressQ4Primitive, "gemv", fail)
    with pytest.raises(ExecutionFailed, match="injected fake worker failure"):
        stress_host_resources.run_stress(
            seed=1,
            iterations=1,
            threads=2,
            include_resource_probes=False,
            allow_fallback=True,
        )
    assert serial_calls > 0, "serial parity did not run before injected failure"
    assert worker_entered.is_set(), "injected failure did not reach a production worker"
    assert stress_host_resources._live_buffer_ids() == before_live
    assert tuple(stress_host_resources._thread_snapshot()["workers"]) == before_workers
    after_fds = stress_host_resources._fd_count()
    assert before_fds is None or after_fds == before_fds


def test_thread_cap_is_rejected_by_cli() -> None:
    root = Path(__file__).parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "stress_host_resources.py"),
            "--threads",
            str(stress_host_resources._ROUTES + 1),
            "--allow-fallback",
            "--no-resource-probes",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0
    assert "threads must not exceed 4 routes" in completed.stderr


def test_tiny_timeout_drains_before_fixture_close() -> None:
    before_live = stress_host_resources._live_buffer_ids()
    before_workers = tuple(stress_host_resources._thread_snapshot()["workers"])
    before_fds = stress_host_resources._fd_count()
    with pytest.raises((TimeoutError, ExecutionFailed)):
        stress_host_resources.run_stress(
            seed=1,
            iterations=1,
            threads=2,
            include_resource_probes=False,
            allow_fallback=True,
            per_iteration_timeout=0.0001,
        )
    assert stress_host_resources._live_buffer_ids() == before_live
    assert tuple(stress_host_resources._thread_snapshot()["workers"]) == before_workers
    after_fds = stress_host_resources._fd_count()
    assert before_fds is None or after_fds == before_fds


def test_partial_fallback_constructor_failure_releases_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_live = stress_host_resources._live_buffer_ids()
    original = cpu_abi.CpuExpertDescriptor
    calls = 0

    def fail_second(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second descriptor failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(cpu_abi, "CpuExpertDescriptor", fail_second)
    with pytest.raises(RuntimeError, match="injected second descriptor failure"):
        stress_host_resources._HostFixture._fallback(fill_seed=1)
    assert calls == 2
    assert stress_host_resources._live_buffer_ids() == before_live


def test_partial_torch_constructor_failure_releases_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("Torch is unavailable")
    before_live = stress_host_resources._live_buffer_ids()
    original = cpu_abi.CpuExpertDescriptor
    calls = 0

    def fail_second(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second descriptor failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(cpu_abi, "CpuExpertDescriptor", fail_second)
    with pytest.raises(RuntimeError, match="injected second descriptor failure"):
        stress_host_resources._HostFixture.open(fill_seed=1, allow_fallback=False)
    assert calls == 2
    assert stress_host_resources._live_buffer_ids() == before_live


def test_fallback_layout_failure_releases_banks(monkeypatch: pytest.MonkeyPatch) -> None:
    before_live = stress_host_resources._live_buffer_ids()

    def fail_layout(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("injected layout construction failure")

    monkeypatch.setattr(cpu_abi, "CpuExpertLayout", fail_layout)
    with pytest.raises(RuntimeError, match="injected layout construction failure"):
        stress_host_resources._HostFixture._fallback(fill_seed=1)
    assert stress_host_resources._live_buffer_ids() == before_live


def test_torch_layout_failure_releases_banks(monkeypatch: pytest.MonkeyPatch) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("Torch is unavailable")
    before_live = stress_host_resources._live_buffer_ids()

    def fail_layout(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("injected layout construction failure")

    monkeypatch.setattr(cpu_abi, "CpuExpertLayout", fail_layout)
    with pytest.raises(RuntimeError, match="injected layout construction failure"):
        stress_host_resources._HostFixture.open(fill_seed=1, allow_fallback=False)
    assert stress_host_resources._live_buffer_ids() == before_live


def test_default_requires_torch_hostbank() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("Torch is installed; the production path is exercised by the other tests")
    with pytest.raises(RuntimeError, match=r"Torch.*HostBank"):
        stress_host_resources.run_stress(
            seed=1,
            iterations=1,
            threads=2,
            include_resource_probes=False,
        )
