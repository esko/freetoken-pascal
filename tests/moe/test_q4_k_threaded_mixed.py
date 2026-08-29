"""Threaded mixed-format route execution contract tests."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
from freetoken.moe import q4_k
from freetoken.moe.cpu_abi import (
    Cancelled,
    CpuExecutionRequest,
    CpuExpertDescriptor,
    CpuExpertLayout,
    ExecutionFailed,
    InvalidRequest,
)
from freetoken.moe.cpu_topology import CpuTopology, PhysicalCore, WorkerAffinityPolicy
from freetoken.moe.ggml_reference import (
    Q5_1_BLOCK_BYTES,
    Q5_K_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES,
)
from freetoken.moe.q4_k import Q4K_BLOCK_BYTES, Q4KExecutor, partition_q4_k_routes

_BLOCKS = {
    "Q4_K": (256, Q4K_BLOCK_BYTES, 12),
    "Q5_K": (256, Q5_K_BLOCK_BYTES, 13),
    "Q5_1": (32, Q5_1_BLOCK_BYTES, 7),
    "Q8_0": (32, Q8_0_BLOCK_BYTES, 8),
}


class _PackedSource:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.range_offset = 0
        self.range_size = int(values.nbytes)
        self.source_address = int(values.__array_interface__["data"][0])

    def expert_packed(self, expert: int) -> np.ndarray:
        return self.values[expert]


class _MetadataSource:
    """Bounded metadata-only source for exact census eligibility tests."""

    range_offset = 0
    range_size = 10**12
    source_address = 0

    def expert_packed(self, expert: int) -> np.ndarray:
        raise AssertionError(f"metadata-only source was unexpectedly executed: {expert}")


def _source(experts: int, rows: int, input_dim: int, quant_name: str) -> _PackedSource:
    block_elements, block_bytes, _ = _BLOCKS[quant_name]
    row_bytes = input_dim // block_elements * block_bytes
    values = np.arange(experts * rows * row_bytes, dtype=np.uint8).reshape(experts, rows, row_bytes)
    return _PackedSource(np.ascontiguousarray(values))


def _descriptor(
    layer: int,
    projection: str,
    quant_name: str,
    source: _PackedSource,
    *,
    experts: int = 3,
    output_dim: int = 256,
    input_dim: int = 256,
) -> CpuExpertDescriptor:
    block_elements, block_bytes, quant_type = _BLOCKS[quant_name]
    row_bytes = input_dim // block_elements * block_bytes
    return CpuExpertDescriptor(
        layer_id=layer,
        projection=projection,
        quant_type=quant_type,
        quant_name=quant_name,
        num_experts=experts,
        output_dim=output_dim,
        input_dim=input_dim,
        rows_per_expert=output_dim,
        row_stride_bytes=row_bytes,
        expert_stride_bytes=output_dim * row_bytes,
        tensor_bytes=experts * output_dim * row_bytes,
        source=source,
    )


def _layout(*, promoted: bool = False, experts: int = 3) -> CpuExpertLayout:
    gate_up = "Q5_K" if promoted else "Q4_K"
    down = "Q8_0" if promoted else "Q5_1"
    descriptors = []
    for projection, quant_name in (("gate", gate_up), ("up", gate_up), ("down", down)):
        descriptors.append(
            _descriptor(
                2 if promoted else 0,
                projection,
                quant_name,
                _source(experts, 256, 256, quant_name),
                experts=experts,
            )
        )
    return CpuExpertLayout(tuple(descriptors), top_k=10)


class _FakeQ4:
    isa = "avx2"
    backend = "q4_k_avx2"
    fallback_reason = None

    def gemv(self, rows, input_dim, vector, *, out, scratch=None):
        del input_dim, scratch
        np.multiply(rows[:, 0].astype(np.float32), vector.sum(), out=out)
        return out


class _FakeMixed:
    isa = "avx2"
    backend = "mixed_gemv_avx2"
    fallback_reason = None

    def backend_for(self, quant_name):
        return f"{str(quant_name).lower()}_avx2"

    def gemv(self, rows, input_dim, vector, *, quant_name, out):
        del input_dim, quant_name
        np.multiply(rows[:, 0].astype(np.float32), vector.sum(), out=out)
        return out


def _patch_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(q4_k, "select_q4_k_primitive", lambda mode="auto": _FakeQ4())
    monkeypatch.setattr(q4_k, "select_mixed_gemv_primitive", lambda mode="auto": _FakeMixed())


def _inputs(tokens: int = 2, routes: int = 10):
    hidden = np.linspace(-0.5, 0.5, tokens * 256, dtype=np.float32).reshape(tokens, 256)
    ids = np.array(
        [[0, 1, 1, 2, 0, 2, 1, 0, 2, 1], [2, 0, 1, 2, 1, 0, 2, 1, 0, 2]],
        dtype=np.int32,
    )[:tokens, :routes]
    weights = np.linspace(-0.4, 0.7, ids.size, dtype=np.float32).reshape(ids.shape)
    return hidden, ids, weights


def _worker_plan(thread_count: int = 2):
    topology = CpuTopology(
        allowed_cpus=tuple(range(100, 100 + thread_count)),
        cores=tuple(
            PhysicalCore(
                key=f"core-{cpu}",
                representative=cpu,
                logical_cpus=(cpu,),
                siblings=(cpu,),
            )
            for cpu in range(100, 100 + thread_count)
        ),
        confidence="full",
        source="test",
    )
    return WorkerAffinityPolicy(requested_threads=thread_count).plan(topology)


def test_q4_worker_plan_verifies_distinct_worker_masks_without_changing_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_native(monkeypatch)
    plan = _worker_plan()
    owner_mask = frozenset({7, 9})
    masks: dict[int, frozenset[int]] = {}

    def set_affinity(_pid: int, cpus: set[int]) -> None:
        masks[threading.get_ident()] = frozenset(cpus)

    def get_affinity(_pid: int) -> frozenset[int]:
        return masks.get(threading.get_ident(), owner_mask)

    monkeypatch.setattr(q4_k.os, "sched_setaffinity", set_affinity)
    monkeypatch.setattr(q4_k.os, "sched_getaffinity", get_affinity)
    executor = Q4KExecutor(
        _layout(), mode="avx2", num_threads=2, worker_plan=plan, required_alignment=1
    )
    executor.prepare(1, 2)
    hidden, ids, weights = _inputs(tokens=1, routes=2)
    result = executor.execute(0, hidden, ids, weights)
    report = executor.affinity_telemetry
    assert result.telemetry.thread_count == 2
    assert report["affinity_status"] == "verified"
    assert report["verification_status"] == "verified"
    assert report["requested_worker_cpus"] == [100, 101]
    assert report["worker_requested_cpus"] == [100, 101]
    assert report["observed_worker_cpus"] == [100, 101]
    assert report["worker_observed_affinity_cpus"] == [100, 101]
    assert report["affinity_errors"] == []
    assert frozenset(get_affinity(0)) == owner_mask
    executor.close()


def test_q4_worker_affinity_failure_drains_and_falls_back_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_native(monkeypatch)
    plan = _worker_plan()

    def fail_affinity(_pid: int, _cpus: set[int]) -> None:
        raise OSError("affinity denied")

    monkeypatch.setattr(q4_k.os, "sched_setaffinity", fail_affinity)
    executor = Q4KExecutor(
        _layout(), mode="avx2", num_threads=2, worker_plan=plan, required_alignment=1
    )
    executor.prepare(1, 2)
    hidden, ids, weights = _inputs(tokens=1, routes=2)
    result = executor.execute(0, hidden, ids, weights)
    assert result.telemetry.thread_count == 1
    report = executor.affinity_telemetry
    assert report["affinity_status"] == "fallback"
    assert report["verification_status"] == "fallback"
    assert "affinity denied" in report["fallback_reason"]
    assert report["affinity_errors"]
    executor.close()


def test_q4_explicit_worker_plan_rejects_external_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_native(monkeypatch)
    with pytest.raises(InvalidRequest, match="internally owned"):
        Q4KExecutor(
            _layout(),
            mode="avx2",
            num_threads=2,
            worker_plan=_worker_plan(),
            thread_pool=_ImmediatePool(),
            required_alignment=1,
        )


def test_partition_is_balanced_and_index_ordered() -> None:
    assert partition_q4_k_routes(10, 1) == ((0, 10),)
    assert partition_q4_k_routes(10, 3) == ((0, 4), (4, 7), (7, 10))
    assert partition_q4_k_routes(3, 10) == ((0, 1), (1, 2), (2, 3))


def test_threaded_runner_closes_owned_pool_if_worker_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TrackedPool:
        def __init__(self, **_kwargs) -> None:
            self.shutdown_calls = 0

        def shutdown(self, *, wait: bool) -> None:
            assert wait
            self.shutdown_calls += 1

    pool = _TrackedPool()
    monkeypatch.setattr(q4_k, "ThreadPoolExecutor", lambda **_kwargs: pool)

    def fail_worker(*_args, **_kwargs):
        raise RuntimeError("worker construction failed")

    monkeypatch.setattr(q4_k, "_MixedReferenceExecutor", fail_worker)
    owner = SimpleNamespace(
        num_threads=2,
        layout=object(),
        _reference=SimpleNamespace(decoders={}),
        primitive=None,
        mixed_primitive=None,
        _q4_descriptors=frozenset(),
        _mixed_descriptors=frozenset(),
    )
    with pytest.raises(RuntimeError, match="worker construction failed"):
        q4_k._ThreadedMixedRunner(
            owner,
            thread_pool=None,
            activation="silu",
            apply_router_weight_on_input=False,
            output_dtype=np.float32,
            required_alignment=1,
        )
    assert pool.shutdown_calls == 1


def test_exact_qwen_census_geometry_gates_every_normal_and_promoted_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_native(monkeypatch)
    promoted = {2, 4, 30, 46, 47}
    descriptors = []
    for layer in (0, 2, 4, 30, 46, 47):
        gate_up = "Q5_K" if layer == 2 else "Q4_K"
        down = "Q8_0" if layer in promoted else "Q5_1"
        for projection, quant_name, output_dim, input_dim in (
            ("gate", gate_up, 640, 2560),
            ("up", gate_up, 640, 2560),
            ("down", down, 2560, 640),
        ):
            descriptors.append(
                _descriptor(
                    layer,
                    projection,
                    quant_name,
                    _MetadataSource(),
                    experts=512,
                    output_dim=output_dim,
                    input_dim=input_dim,
                )
            )
    layout = CpuExpertLayout(tuple(descriptors), top_k=10)
    executor = Q4KExecutor(layout, mode="avx2", num_threads=2, required_alignment=1)
    assert executor.parallel_enabled
    assert all(executor._layer_direct_eligible(layer) for layer in layout.layers)
    assert executor._kernel_census(0) == ("q4_k_avx2", "q5_1_avx2")
    assert executor._kernel_census(2) == ("q5_k_avx2", "q8_0_avx2")
    executor.close()


@pytest.mark.parametrize("promoted", [False, True])
def test_threaded_mixed_routes_match_serial_for_normal_and_promoted_layers(
    promoted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_native(monkeypatch)
    layout = _layout(promoted=promoted)
    serial = Q4KExecutor(layout, mode="avx2", num_threads=1, required_alignment=1)
    threaded = Q4KExecutor(layout, mode="avx2", num_threads=3, required_alignment=1)
    serial.prepare(2, 10)
    threaded.prepare(2, 10)
    hidden, ids, weights = _inputs()
    expected = serial.execute(layout.layers[0], hidden, ids, weights)
    actual = threaded.execute(layout.layers[0], hidden, ids, weights)
    np.testing.assert_allclose(actual.output, expected.output, rtol=3e-5, atol=3e-5)
    assert actual.telemetry.thread_count == 3
    assert actual.telemetry.routes_executed == 20
    assert actual.telemetry.kernel_census == (
        ("q4_k_avx2", "q5_1_avx2") if not promoted else ("q5_k_avx2", "q8_0_avx2")
    )


def test_reduction_uses_partition_index_not_completion_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_native(monkeypatch)
    layout = _layout()
    executor = Q4KExecutor(layout, mode="avx2", num_threads=3, required_alignment=1)
    executor.prepare(1, 3)
    hidden, ids, weights = _inputs(tokens=1, routes=3)
    expected = Q4KExecutor(layout, mode="avx2", num_threads=1, required_alignment=1)
    expected.prepare(1, 3)
    reference = expected.execute(0, hidden, ids, weights).output.copy()

    runner = executor._threaded_runner
    assert runner is not None
    original = runner._workers[0].execute

    def delayed(*args, **kwargs):
        time.sleep(0.03)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner._workers[0], "execute", delayed)
    actual = executor.execute(0, hidden, ids, weights).output
    np.testing.assert_allclose(actual, reference)


def test_output_without_caller_buffer_is_fresh_and_group_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_native(monkeypatch)
    executor = Q4KExecutor(_layout(), mode="avx2", num_threads=2, required_alignment=1)
    executor.prepare(1, 2)
    hidden, ids, weights = _inputs(tokens=1, routes=2)
    first = executor.execute(0, hidden, ids, weights)
    first_copy = first.output.copy()
    second = executor.execute(0, hidden, ids, weights)
    assert first.output is not second.output
    np.testing.assert_array_equal(first.output, first_copy)
    grouped = executor.execute_group(
        (
            CpuExecutionRequest(0, hidden, ids, weights),
            CpuExecutionRequest(0, hidden, ids, weights),
        )
    )
    assert grouped[0].output is not grouped[1].output


def test_group_outputs_remain_stable_beyond_prepared_result_ring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_native(monkeypatch)
    executor = Q4KExecutor(_layout(), mode="avx2", num_threads=2, required_alignment=1)
    executor.prepare(1, 2)
    hidden, ids, weights = _inputs(tokens=1, routes=2)
    requests = tuple(
        CpuExecutionRequest(
            0,
            hidden,
            ids,
            weights + np.float32(index),
        )
        for index in range(4)
    )

    grouped = executor.execute_group(requests)
    snapshots = tuple(result.output.copy() for result in grouped)
    pointers = tuple(result.output.__array_interface__["data"][0] for result in grouped)
    assert len(set(pointers)) == len(grouped)
    for result, snapshot in zip(grouped, snapshots, strict=True):
        np.testing.assert_array_equal(result.output, snapshot)


def test_threaded_accumulate_zeros_padded_rows_like_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_native(monkeypatch)
    layout = _layout()
    serial = Q4KExecutor(layout, mode="avx2", num_threads=1, required_alignment=1)
    threaded = Q4KExecutor(layout, mode="avx2", num_threads=2, required_alignment=1)
    serial.prepare(2, 2)
    threaded.prepare(2, 2)
    hidden, ids, weights = _inputs(tokens=2, routes=2)
    serial_output = np.full((2, 256), 3.0, dtype=np.float32)
    threaded_output = serial_output.copy()
    expected = serial.execute(
        0,
        hidden,
        ids,
        weights,
        num_token_non_padded=1,
        output=serial_output,
        accumulate=True,
    )
    actual = threaded.execute(
        0,
        hidden,
        ids,
        weights,
        num_token_non_padded=1,
        output=threaded_output,
        accumulate=True,
    )
    np.testing.assert_array_equal(actual.output, expected.output)
    assert np.all(actual.output[1] == 0)


class _SubmitFailurePool:
    def __init__(self) -> None:
        self.submitted: list[Future] = []

    def submit(self, function, *args, **kwargs):
        if self.submitted:
            raise RuntimeError("submit failed")
        future = Future()
        self.submitted.append(future)
        future.set_running_or_notify_cancel()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future


class _ImmediatePool:
    def submit(self, function, *args, **kwargs):
        future = Future()
        future.set_running_or_notify_cancel()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future


def test_submission_failure_drains_and_wraps(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_native(monkeypatch)
    pool = _SubmitFailurePool()
    executor = Q4KExecutor(
        _layout(), mode="avx2", num_threads=2, thread_pool=pool, required_alignment=1
    )
    executor.prepare(1, 2)
    hidden, ids, weights = _inputs(tokens=1, routes=2)
    output = np.full((1, 256), 9.0, dtype=np.float32)
    with pytest.raises(ExecutionFailed, match="submit") as raised:
        executor.execute(0, hidden, ids, weights, output=output)
    assert pool.submitted[0].done()
    assert raised.value.telemetry is not None
    assert raised.value.telemetry.thread_count == 1
    assert np.all(output == 0)


def test_worker_failure_cancels_and_drains_siblings(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_native(monkeypatch)
    executor = Q4KExecutor(_layout(), mode="avx2", num_threads=2, required_alignment=1)
    executor.prepare(1, 2)
    hidden, ids, weights = _inputs(tokens=1, routes=2)
    runner = executor._threaded_runner
    assert runner is not None
    original = runner._workers[1].execute

    def fail(*args, **kwargs):
        time.sleep(0.02)
        raise RuntimeError("worker exploded")

    monkeypatch.setattr(runner._workers[0], "execute", fail)
    sibling_finished = threading.Event()

    def sibling(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            sibling_finished.set()

    monkeypatch.setattr(runner._workers[1], "execute", sibling)
    output = np.full((1, 256), 7.0, dtype=np.float32)
    with pytest.raises(ExecutionFailed, match="worker exploded"):
        executor.execute(0, hidden, ids, weights, output=output)
    assert sibling_finished.is_set()
    assert np.all(output == 0)
    assert executor.last_telemetry is not None
    assert executor.last_telemetry.thread_count == 2


def test_validation_and_precompute_cancellation_report_one_used_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_native(monkeypatch)
    executor = Q4KExecutor(_layout(), mode="avx2", num_threads=4, required_alignment=1)
    executor.prepare(1, 2)
    hidden, ids, weights = _inputs(tokens=1, routes=2)
    with pytest.raises(InvalidRequest) as invalid:
        executor.execute(0, hidden, ids, weights, _thread_count=0)
    assert invalid.value.telemetry is not None
    assert invalid.value.telemetry.thread_count == 1
    with pytest.raises(Cancelled) as cancelled:
        executor.execute(0, hidden, ids, weights, cancellation=lambda: True)
    assert cancelled.value.telemetry is not None
    assert cancelled.value.telemetry.thread_count == 1


def test_threaded_rejects_non_boolean_accumulate(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_native(monkeypatch)
    executor = Q4KExecutor(_layout(), mode="avx2", num_threads=2, required_alignment=1)
    executor.prepare(1, 2)
    hidden, ids, weights = _inputs(tokens=1, routes=2)
    with pytest.raises(InvalidRequest, match="accumulate must be a bool"):
        executor.execute(0, hidden, ids, weights, accumulate=1)


def test_cancellation_callback_runs_only_on_owner_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_native(monkeypatch)
    executor = Q4KExecutor(_layout(), mode="avx2", num_threads=2, required_alignment=1)
    executor.prepare(1, 2)
    hidden, ids, weights = _inputs(tokens=1, routes=2)
    owner = threading.get_ident()
    callback_threads: list[int] = []

    def cancellation() -> bool:
        callback_threads.append(threading.get_ident())
        return False

    executor.execute(0, hidden, ids, weights, cancellation=cancellation)
    assert callback_threads and set(callback_threads) == {owner}


def test_transient_owner_cancellation_is_not_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_native(monkeypatch)
    executor = Q4KExecutor(
        _layout(),
        mode="avx2",
        num_threads=2,
        thread_pool=_ImmediatePool(),
        required_alignment=1,
    )
    executor.prepare(1, 2)
    hidden, ids, weights = _inputs(tokens=1, routes=2)
    answers = iter((False, True, False, False))

    with pytest.raises(Cancelled):
        executor.execute(0, hidden, ids, weights, cancellation=lambda: next(answers, False))


def test_cancellation_callback_error_drains_workers_before_unlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_native(monkeypatch)
    executor = Q4KExecutor(_layout(), mode="avx2", num_threads=2, required_alignment=1)
    executor.prepare(1, 2)
    hidden, ids, weights = _inputs(tokens=1, routes=2)
    finished = threading.Event()
    original = executor._threaded_runner._workers[1].execute

    def slow_worker(*args, **kwargs):
        try:
            time.sleep(0.03)
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(executor._threaded_runner._workers[1], "execute", slow_worker)
    calls = 0

    def broken_cancellation() -> bool:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("callback failed")
        return False

    with pytest.raises(ExecutionFailed, match="callback failed"):
        executor.execute(0, hidden, ids, weights, cancellation=broken_cancellation)
    assert finished.is_set()
    recovered = executor.execute(0, hidden, ids, weights)
    assert recovered.telemetry.routes_executed == 2


def test_scalar_or_unsupported_mixed_layout_stays_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ScalarQ4(_FakeQ4):
        isa = "scalar"

    class _ScalarMixed(_FakeMixed):
        isa = "scalar"

    monkeypatch.setattr(q4_k, "select_q4_k_primitive", lambda mode="auto": _ScalarQ4())
    monkeypatch.setattr(
        q4_k,
        "select_mixed_gemv_primitive",
        lambda mode="auto": _ScalarMixed(),
    )
    executor = Q4KExecutor(_layout(), mode="scalar", num_threads=4, required_alignment=1)
    executor.prepare(1, 2)
    hidden, ids, weights = _inputs(tokens=1, routes=2)
    result = executor.execute(0, hidden, ids, weights)
    assert result.telemetry.thread_count == 1
    assert not executor.parallel_enabled
