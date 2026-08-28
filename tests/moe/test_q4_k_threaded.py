"""Thread-partitioned Q4_K execution and raw CPU benchmark coverage."""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import numpy as np
import pytest
from freetoken.moe import q4_k
from freetoken.moe.cpu_abi import Cancelled, CpuExpertLayout, InvalidRequest
from freetoken.moe.q4_k import (
    Q4KExecutor,
    partition_q4_k_routes,
)

from .test_q4_k import _q4_layout


def _benchmark_module():
    path = Path(__file__).resolve().parents[2] / "scripts/bench_q4_k_threaded.py"
    spec = importlib.util.spec_from_file_location("bench_q4_k_threaded", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partition_q4_k_routes_balances_contiguous_route_columns() -> None:
    assert partition_q4_k_routes(10, 1) == ((0, 10),)
    assert partition_q4_k_routes(10, 2) == ((0, 5), (5, 10))
    assert partition_q4_k_routes(10, 3) == ((0, 4), (4, 7), (7, 10))
    assert partition_q4_k_routes(3, 10) == ((0, 1), (1, 2), (2, 3))
    with pytest.raises(ValueError, match="route_count"):
        partition_q4_k_routes(0, 1)
    with pytest.raises(ValueError, match="thread_count"):
        partition_q4_k_routes(10, 0)


def _threaded_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = np.linspace(-0.5, 0.5, 3 * 256, dtype=np.float32).reshape(3, 256)
    expert_ids = np.array(
        [
            [0, 1, 1, 2, 0, 2, 1, 0, 2, 1],
            [2, 0, 1, 2, 1, 0, 2, 1, 0, 2],
            [1, 2, -1, 0, 2, 1, 0, -1, 1, 2],
        ],
        dtype=np.int32,
    )
    weights = np.linspace(-0.4, 0.7, expert_ids.size, dtype=np.float32).reshape(expert_ids.shape)
    return hidden, expert_ids, weights


def _threaded_layout() -> CpuExpertLayout:
    layout, _ = _q4_layout(experts=3)
    return CpuExpertLayout(layout.descriptors, top_k=10)


class _FastFakeAvx2:
    def __init__(self, isa: str = "avx2") -> None:
        self.isa = isa
        self.backend = f"q4_k_{isa}"

    fallback_reason = None

    def decode(self, block: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        from freetoken.moe.q4_k import decode_q4_k_block

        return decode_q4_k_block(block, out=out)

    def gemv(
        self,
        rows: np.ndarray,
        input_dim: int,
        vector: np.ndarray,
        *,
        out: np.ndarray,
        scratch: np.ndarray | None = None,
    ) -> np.ndarray:
        del scratch
        del input_dim
        # This is deliberately a cheap reentrant stand-in for the native
        # primitive.  The test verifies route partitioning and merge order;
        # Q4_K arithmetic remains covered by the existing differential tests.
        np.multiply(rows[:, 0].astype(np.float32), np.float32(vector.sum()), out=out)
        return out


class _FastFakeMixed:
    isa = "avx2"
    backend = "mixed_gemv_avx2"
    fallback_reason = None

    def backend_for(self, quant_name: str) -> str:
        return f"{quant_name.lower()}_avx2"

    def gemv(
        self,
        rows: np.ndarray,
        input_dim: int,
        vector: np.ndarray,
        *,
        quant_name: str,
        out: np.ndarray,
    ) -> np.ndarray:
        del input_dim, quant_name
        # Keep synthetic mixed benchmarks bounded without decoding a full matrix.
        np.multiply(rows[:, 0].astype(np.float32), np.float32(vector.sum()), out=out)
        return out


def _fake_avx2(monkeypatch: pytest.MonkeyPatch) -> None:
    primitive = _FastFakeAvx2()
    mixed = _FastFakeMixed()
    monkeypatch.setattr(
        q4_k,
        "select_q4_k_primitive",
        lambda mode: primitive,
    )
    monkeypatch.setattr(
        q4_k,
        "select_mixed_gemv_primitive",
        lambda mode: mixed,
    )


def test_threaded_q4_k_matches_serial_and_preserves_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_avx2(monkeypatch)
    layout = _threaded_layout()
    serial = Q4KExecutor(layout, mode="avx2", num_threads=1, required_alignment=1)
    threaded = Q4KExecutor(layout, mode="avx2", num_threads=4, required_alignment=1)
    serial.prepare(max_tokens=3, max_routes=10)
    threaded.prepare(max_tokens=3, max_routes=10)
    hidden, expert_ids, weights = _threaded_inputs()

    expected = serial.execute(0, hidden, expert_ids, weights)
    actual = threaded.execute(0, hidden, expert_ids, weights)
    np.testing.assert_allclose(actual.output, expected.output, rtol=3e-5, atol=3e-5)
    assert actual.telemetry.backend == "q4_k_avx2"
    assert actual.telemetry.kernel_census == ("q4_k_avx2",)
    assert actual.telemetry.routes_executed == 28
    assert actual.telemetry.unique_experts == 3
    assert actual.telemetry.thread_count == 4

    destination = np.full_like(hidden, 2.0)
    accumulated = threaded.execute(
        0,
        hidden,
        expert_ids,
        weights,
        output=destination,
        accumulate=True,
    )
    np.testing.assert_allclose(accumulated.output, expected.output + 2.0, rtol=3e-5, atol=3e-5)
    assert accumulated.output is destination


def test_one_thread_q4_k_is_the_serial_reference_fallback() -> None:
    layout = _threaded_layout()
    executor = Q4KExecutor(layout, mode="scalar", num_threads=1)
    executor.prepare(max_tokens=1, max_routes=10)
    hidden, expert_ids, weights = _threaded_inputs()
    result = executor.execute(0, hidden[:1], expert_ids[:1, :1], weights[:1, :1])
    assert result.telemetry.thread_count == 1
    assert result.telemetry.routes_executed == 1


def test_threaded_q4_k_cancellation_rolls_back_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_avx2(monkeypatch)
    layout = _threaded_layout()
    executor = Q4KExecutor(layout, mode="avx2", num_threads=4, required_alignment=1)
    executor.prepare(max_tokens=1, max_routes=10)
    hidden, expert_ids, weights = _threaded_inputs()
    destination = np.full((1, 256), 9.0, dtype=np.float32)

    with pytest.raises(Cancelled) as raised:
        executor.execute(
            0,
            hidden[:1],
            expert_ids[:1],
            weights[:1],
            output=destination,
            cancellation=lambda: True,
        )
    assert np.all(destination == 0)
    assert raised.value.telemetry is not None
    assert raised.value.telemetry.cancelled

    recovered = executor.execute(0, hidden[:1], expert_ids[:1], weights[:1])
    assert np.isfinite(recovered.output).all()


def test_threaded_q4_k_evaluates_user_cancellation_only_on_owner_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_avx2(monkeypatch)
    layout = _threaded_layout()
    executor = Q4KExecutor(layout, mode="avx2", num_threads=4, required_alignment=1)
    executor.prepare(max_tokens=1, max_routes=10)
    hidden, expert_ids, weights = _threaded_inputs()
    callback_threads: list[int] = []

    def cancellation() -> bool:
        callback_threads.append(threading.get_ident())
        return False

    executor.execute(
        0,
        hidden[:1],
        expert_ids[:1],
        weights[:1],
        cancellation=cancellation,
    )
    assert callback_threads
    assert set(callback_threads) == {threading.get_ident()}


def test_threaded_q4_k_uses_prepared_result_buffer_without_execute_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_avx2(monkeypatch)
    layout = _threaded_layout()
    executor = Q4KExecutor(layout, mode="avx2", num_threads=4, required_alignment=1)
    executor.prepare(max_tokens=1, max_routes=10)
    hidden, expert_ids, weights = _threaded_inputs()
    monkeypatch.setattr(
        np,
        "empty",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("threaded execute allocated a result")
        ),
    )
    result = executor.execute(0, hidden[:1], expert_ids[:1], weights[:1])
    assert result.output.shape == (1, 256)


def test_threaded_q4_k_owned_pool_lifecycle_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_avx2(monkeypatch)
    layout = _threaded_layout()
    executor = Q4KExecutor(layout, mode="avx2", num_threads=2, required_alignment=1)
    executor.prepare(max_tokens=1, max_routes=1)
    executor.close()
    executor.close()
    hidden, expert_ids, weights = _threaded_inputs()
    with pytest.raises(InvalidRequest, match="closed"):
        executor.execute(0, hidden[:1], expert_ids[:, :1], weights[:, :1])


def test_threaded_q4_k_microbenchmark_records_raw_route_and_thread_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_avx2(monkeypatch)
    layout = _threaded_layout()
    executor = Q4KExecutor(layout, mode="avx2", num_threads=4, required_alignment=1)
    executor.prepare(max_tokens=3, max_routes=10)
    hidden, expert_ids, weights = _threaded_inputs()

    samples = executor.microbenchmark(
        0,
        hidden,
        expert_ids,
        weights,
        repeats=2,
        route_counts=(1, 2, 4, 8, 10),
        thread_counts=(1, 2, 4),
    )

    assert [(sample.route_count, sample.thread_count) for sample in samples] == [
        (route, threads) for route in (1, 2, 4, 8, 10) for threads in (1, 2, 4)
    ]
    for sample in samples:
        assert sample.miss_count == int(np.count_nonzero(expert_ids[:, : sample.route_count] >= 0))
        assert sample.repeats == 2
        assert len(sample.elapsed_ns) == 2
        assert all(isinstance(value, int) and value > 0 for value in sample.elapsed_ns)
        assert all(
            item.thread_count == min(sample.thread_count, sample.route_count)
            for item in sample.telemetry
        )
        encoded = sample.as_dict()
        assert encoded["route_count"] == sample.route_count
        assert encoded["thread_count"] == sample.thread_count
        assert len(encoded["elapsed_ns"]) == 2


def test_threaded_q4_k_microbenchmark_rejects_threads_above_prepared_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_avx2(monkeypatch)
    layout = _threaded_layout()
    executor = Q4KExecutor(layout, mode="avx2", num_threads=2, required_alignment=1)
    executor.prepare(max_tokens=1, max_routes=10)
    hidden, expert_ids, weights = _threaded_inputs()
    with pytest.raises(InvalidRequest, match="thread_count"):
        executor.microbenchmark(
            0,
            hidden[:1],
            expert_ids[:1],
            weights[:1],
            thread_counts=(1, 4),
        )


def test_scalar_q4_k_keeps_thread_count_opt_in_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    primitive = _FastFakeAvx2("scalar")
    monkeypatch.setattr(
        q4_k,
        "select_q4_k_primitive",
        lambda mode: primitive,
    )
    layout = _threaded_layout()
    executor = Q4KExecutor(layout, mode="scalar", num_threads=4)
    executor.prepare(max_tokens=1, max_routes=10)
    hidden, expert_ids, weights = _threaded_inputs()
    result = executor.execute(0, hidden[:1], expert_ids[:1], weights[:1])
    assert result.telemetry.thread_count == 1
    with pytest.raises(InvalidRequest, match="thread_count"):
        executor.microbenchmark(
            0,
            hidden[:1],
            expert_ids[:1],
            weights[:1],
            thread_counts=(1, 2),
        )


def test_q4_k_benchmark_harness_records_serial_fallback_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primitive = _FastFakeAvx2("scalar")
    monkeypatch.setattr(q4_k, "select_q4_k_primitive", lambda mode: primitive)
    monkeypatch.setattr(q4_k, "select_mixed_gemv_primitive", lambda mode: _FastFakeMixed())
    benchmark = _benchmark_module()
    args = benchmark._parser().parse_args(
        [
            "--mode",
            "scalar",
            "--experts",
            "2",
            "--tokens",
            "1",
            "--repeats",
            "1",
            "--route-counts",
            "1,2",
            "--thread-counts",
            "1,2",
        ]
    )
    document = benchmark.collect(args)
    assert document["evidence_status"] == "synthetic"
    assert document["selected_behavior"]["parallel_enabled"] is False
    assert document["workload"]["thread_counts_requested"] == [1, 2]
    assert document["workload"]["thread_counts_executed"] == [1]
    assert {(item["route_count"], item["thread_count"]) for item in document["raw_samples"]} == {
        (1, 1),
        (2, 1),
    }


def test_q4_k_benchmark_harness_records_parallel_thread_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primitive = _FastFakeAvx2("avx2")
    monkeypatch.setattr(q4_k, "select_q4_k_primitive", lambda mode: primitive)
    monkeypatch.setattr(q4_k, "select_mixed_gemv_primitive", lambda mode: _FastFakeMixed())
    benchmark = _benchmark_module()
    args = benchmark._parser().parse_args(
        [
            "--mode",
            "avx2",
            "--experts",
            "2",
            "--tokens",
            "1",
            "--repeats",
            "1",
            "--route-counts",
            "1,2",
            "--thread-counts",
            "1,2",
        ]
    )
    document = benchmark.collect(args)
    assert document["selected_behavior"]["parallel_enabled"] is True
    assert document["workload"]["thread_counts_executed"] == [1, 2]
    assert {(item["route_count"], item["thread_count"]) for item in document["raw_samples"]} == {
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    }
    assert all(item["repeats"] == 1 for item in document["raw_samples"])
