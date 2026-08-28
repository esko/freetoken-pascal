from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np
import pytest
from freetoken.moe.cpu_abi import (
    Busy,
    Cancelled,
    CpuExecutionRequest,
    CpuExpertDescriptor,
    CpuExpertLayout,
    InvalidExpertId,
    InvalidRequest,
    ReferenceCpuExpertExecutor,
    UnsupportedAlignment,
    UnsupportedQuantType,
    UnsupportedShape,
    WorkspaceTooSmall,
    cpu_layout_from_source_layout,
)


def _descriptor(layer: int, projection: str, source: np.ndarray, *, pool_id: int = 0):
    experts, output_dim, input_dim = source.shape
    row_stride = input_dim * 4
    return CpuExpertDescriptor(
        layer_id=layer,
        projection=projection,
        quant_type="F32",
        quant_name="F32",
        num_experts=experts,
        output_dim=output_dim,
        input_dim=input_dim,
        rows_per_expert=output_dim,
        row_stride_bytes=row_stride,
        expert_stride_bytes=output_dim * row_stride,
        tensor_bytes=experts * output_dim * row_stride,
        source_offset=0,
        source_address=layer * 4096,
        pool_id=pool_id,
        source=source,
    )


def _executor(
    *, apply_router_weight_on_input: bool = False, activation: str = "silu"
) -> ReferenceCpuExpertExecutor:
    rng = np.random.default_rng(38)
    hidden, intermediate, experts = 4, 6, 5
    gate = rng.normal(size=(experts, intermediate, hidden)).astype(np.float32)
    up = rng.normal(size=(experts, intermediate, hidden)).astype(np.float32)
    down = rng.normal(size=(experts, hidden, intermediate)).astype(np.float32)
    layout = CpuExpertLayout(
        descriptors=(
            _descriptor(0, "gate", gate, pool_id=1),
            _descriptor(0, "up", up, pool_id=1),
            _descriptor(0, "down", down, pool_id=2),
        ),
        top_k=10,
    )
    executor = ReferenceCpuExpertExecutor(
        layout,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    executor.prepare(max_tokens=128, max_routes=10)
    return executor


def _independent_reference(executor, hidden, ids, weights, *, apply_input=False):
    output = np.zeros((hidden.shape[0], hidden.shape[1]), dtype=np.float32)
    gate = executor.layout.descriptor(0, "gate").source
    up = executor.layout.descriptor(0, "up").source
    down = executor.layout.descriptor(0, "down").source
    for token in range(hidden.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            if expert < 0:
                continue
            gate_up_gate = gate[expert] @ hidden[token]
            gate_up_up = up[expert] @ hidden[token]
            if apply_input:
                # This deliberately mirrors production: both gate and up values
                # are scaled before the nonlinear activation.
                gate_up_gate *= weights[token, route]
                gate_up_up *= weights[token, route]
            if executor.activation == "silu":
                activated = 1.0 / (1.0 + np.exp(-gate_up_gate)) * gate_up_gate
            elif executor.activation == "gelu":
                activated = (
                    0.5
                    * gate_up_gate
                    * (1.0 + np.vectorize(math.erf)(gate_up_gate / math.sqrt(2.0)))
                )
            else:
                activated = (
                    0.5
                    * gate_up_gate
                    * (
                        1.0
                        + np.tanh(
                            math.sqrt(2.0 / math.pi) * (gate_up_gate + 0.044715 * gate_up_gate**3)
                        )
                    )
                )
            activated *= gate_up_up
            contribution = down[expert] @ activated
            if not apply_input:
                contribution *= weights[token, route]
            output[token] += contribution
    return output


def test_reference_executor_matches_independent_dense_contract_and_accumulates() -> None:
    executor = _executor()
    hidden = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
    ids = np.array([[3, 1, 3], [0, -1, 4], [2, 2, -1]], dtype=np.int32)
    weights = np.array([[0.2, -0.4, 0.1], [1.0, 0.0, -0.5], [0.3, 0.7, 0.0]], dtype=np.float32)

    result = executor.execute(0, hidden, ids, weights)
    expected = _independent_reference(executor, hidden, ids, weights)
    np.testing.assert_allclose(result.output, expected, rtol=2e-6, atol=2e-6)
    assert result.telemetry.routes_executed == 7
    assert result.telemetry.unique_experts == 5
    assert result.telemetry.bytes_read_packed == 0
    assert result.telemetry.fallback_reason == "reference_dense"
    assert result.telemetry.as_dict()["expert_count"] == 5

    partial = np.full((3, 4), 2, dtype=np.float32)
    accumulated = executor.execute(0, hidden, ids, weights, output=partial, accumulate=True)
    np.testing.assert_allclose(accumulated.output, expected + 2, rtol=2e-6, atol=2e-6)
    assert accumulated.output is partial


@pytest.mark.parametrize("apply_input", [False, True])
def test_reference_executor_matches_randomized_pure_torch_reference(
    apply_input: bool,
) -> None:
    torch = pytest.importorskip("torch")
    executor = _executor(apply_router_weight_on_input=apply_input)
    rng = np.random.default_rng(3815)
    hidden = rng.normal(size=(8, 4)).astype(np.float32)
    ids = rng.integers(0, 5, size=(8, 10), dtype=np.int32)
    weights = rng.normal(size=(8, 10)).astype(np.float32)

    actual = executor.execute(0, hidden, ids, weights).output
    hidden_t = torch.from_numpy(hidden)
    gate = torch.from_numpy(executor.layout.descriptor(0, "gate").source)
    up = torch.from_numpy(executor.layout.descriptor(0, "up").source)
    down = torch.from_numpy(executor.layout.descriptor(0, "down").source)
    weights_t = torch.from_numpy(weights)
    expected = torch.zeros_like(hidden_t)
    for token in range(hidden.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            gate_out = gate[expert] @ hidden_t[token]
            up_out = up[expert] @ hidden_t[token]
            if apply_input:
                gate_out = gate_out * weights_t[token, route]
                up_out = up_out * weights_t[token, route]
            contribution = down[expert] @ (torch.nn.functional.silu(gate_out) * up_out)
            if not apply_input:
                contribution = contribution * weights_t[token, route]
            expected[token] += contribution

    np.testing.assert_allclose(actual, expected.numpy(), rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("apply_input", [False, True])
def test_apply_router_weight_contract_is_explicit(apply_input: bool) -> None:
    executor = _executor(apply_router_weight_on_input=apply_input)
    hidden = np.ones((2, 4), dtype=np.float32)
    ids = np.array([[1, 2], [2, 1]], dtype=np.int32)
    weights = np.array([[0.25, 0.75], [0.5, 0.25]], dtype=np.float32)

    actual = executor.execute(0, hidden, ids, weights).output
    expected = _independent_reference(executor, hidden, ids, weights, apply_input=apply_input)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    if apply_input:
        gate = executor.layout.descriptor(0, "gate").source
        up = executor.layout.descriptor(0, "up").source
        down = executor.layout.descriptor(0, "down").source
        squared = np.zeros_like(expected)
        for token in range(hidden.shape[0]):
            for route in range(ids.shape[1]):
                weight = weights[token, route]
                expert = int(ids[token, route])
                gate_out = gate[expert] @ hidden[token]
                up_out = up[expert] @ hidden[token]
                activated = 1.0 / (1.0 + np.exp(-gate_out * weight))
                activated *= gate_out * weight * up_out * weight
                squared[token] += down[expert] @ activated
        np.testing.assert_allclose(actual, squared, rtol=2e-6, atol=2e-6)

        # The nonlinear input mode must not be replaced by the otherwise-linear
        # one-scale down-output form.  This catches a tempting but incorrect
        # optimization that silently changes production semantics.
        one_scale = _independent_reference(executor, hidden, ids, weights, apply_input=False)
        assert not np.allclose(actual, one_scale, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh"])
def test_reference_executor_matches_independent_non_silu_activations(activation: str) -> None:
    executor = _executor(activation=activation)
    hidden = np.array([[0.1, -0.2, 0.3, -0.4]], dtype=np.float32)
    ids = np.array([[0, 3]], dtype=np.int32)
    weights = np.array([[0.25, 0.75]], dtype=np.float32)

    actual = executor.execute(0, hidden, ids, weights).output
    expected = _independent_reference(executor, hidden, ids, weights)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_padded_rows_ignore_invalid_ids_and_are_zero() -> None:
    executor = _executor()
    hidden = np.ones((4, 4), dtype=np.float32)
    ids = np.array([[0], [1], [99], [-2]], dtype=np.int32)
    weights = np.ones((4, 1), dtype=np.float32)
    actual = executor.execute(0, hidden, ids, weights, num_token_non_padded=2).output
    assert np.all(actual[2:] == 0)

    with pytest.raises(InvalidExpertId):
        executor.execute(0, hidden, ids, weights, num_token_non_padded=3)


@pytest.mark.parametrize("ids", [np.array([[-2]], dtype=np.int32), np.array([[5]], dtype=np.int32)])
def test_invalid_active_ids_clear_caller_output(ids: np.ndarray) -> None:
    executor = _executor()
    output = np.full((1, 4), 7, dtype=np.float32)
    with pytest.raises(InvalidExpertId) as raised:
        executor.execute(
            0,
            np.ones((1, 4), dtype=np.float32),
            ids,
            np.ones((1, 1), dtype=np.float32),
            output=output,
        )
    assert np.all(output == 0)
    assert raised.value.telemetry is not None
    assert raised.value.telemetry.error == "InvalidExpertId"
    assert "outside" in raised.value.telemetry.error_detail


def test_cancellation_is_transactional_and_workspace_is_reusable() -> None:
    executor = _executor()
    hidden = np.ones((2, 4), dtype=np.float32)
    ids = np.array([[0, 1], [2, 3]], dtype=np.int32)
    weights = np.ones((2, 2), dtype=np.float32)
    output = np.full((2, 4), 9, dtype=np.float32)

    def token():
        return True

    with pytest.raises(Cancelled) as raised:
        executor.execute(0, hidden, ids, weights, output=output, cancellation=token)
    assert np.all(output == 0)
    assert raised.value.telemetry is not None
    assert raised.value.telemetry.cancelled

    recovered = executor.execute(0, hidden, ids, weights)
    assert np.isfinite(recovered.output).all()


def test_mid_route_cancellation_rolls_back_and_releases_executor() -> None:
    executor = _executor()
    hidden = np.ones((1, 4), dtype=np.float32)
    ids = np.array([[0, 1, 2]], dtype=np.int32)
    weights = np.ones((1, 3), dtype=np.float32)
    output = np.full((1, 4), 5, dtype=np.float32)
    checks = 0

    def cancel_after_one_route() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3  # pre-compute check, then first route, then cancellation

    with pytest.raises(Cancelled):
        executor.execute(
            0,
            hidden,
            ids,
            weights,
            output=output,
            cancellation=cancel_after_one_route,
        )
    assert checks >= 3
    assert np.all(output == 0)

    recovered = executor.execute(0, hidden, ids, weights)
    assert np.isfinite(recovered.output).all()


def test_group_api_reuses_the_same_prepared_contract() -> None:
    executor = _executor()
    request = CpuExecutionRequest(
        layer_id=0,
        hidden=np.ones((1, 4), dtype=np.float32),
        expert_ids=np.array([[0]], dtype=np.int32),
        routing_weights=np.ones((1, 1), dtype=np.float32),
    )
    results = executor.execute_group((request, request))
    assert len(results) == 2
    np.testing.assert_array_equal(results[0].output, results[1].output)


def test_group_api_commits_prior_requests_before_a_later_failure() -> None:
    executor = _executor()
    first_output = np.full((1, 4), 9, dtype=np.float32)
    second_output = np.full((1, 4), 9, dtype=np.float32)
    first = CpuExecutionRequest(
        layer_id=0,
        hidden=np.ones((1, 4), dtype=np.float32),
        expert_ids=np.zeros((1, 1), dtype=np.int32),
        routing_weights=np.ones((1, 1), dtype=np.float32),
        output=first_output,
    )
    second = CpuExecutionRequest(
        layer_id=0,
        hidden=np.ones((1, 4), dtype=np.float32),
        expert_ids=np.full((1, 1), 99, dtype=np.int32),
        routing_weights=np.ones((1, 1), dtype=np.float32),
        output=second_output,
    )
    with pytest.raises(InvalidExpertId):
        executor.execute_group((first, second))
    assert not np.all(first_output == 9)
    assert np.all(second_output == 0)


def test_microbenchmark_returns_raw_samples_for_supplied_route_widths() -> None:
    executor = _executor()
    hidden = np.ones((3, 4), dtype=np.float32)
    ids = np.array([[0, -1, 1, 2], [2, 3, -1, 4], [1, 1, 0, -1]], dtype=np.int32)
    weights = np.ones(ids.shape, dtype=np.float32)

    samples = executor.microbenchmark(0, hidden, ids, weights, repeats=2)

    assert [sample.route_count for sample in samples] == [1, 2, 3, 4]
    for sample in samples:
        expected_misses = int(np.count_nonzero(ids[:, : sample.route_count] >= 0))
        assert sample.miss_count == expected_misses
        assert sample.repeats == 2
        assert len(sample.elapsed_ns) == 2
        assert all(isinstance(value, int) and value > 0 for value in sample.elapsed_ns)
        assert len(sample.telemetry) == 2
        assert all(item.routes_executed == expected_misses for item in sample.telemetry)
        document = sample.as_dict()
        assert set(document) == {
            "layer_id",
            "route_count",
            "miss_count",
            "repeats",
            "elapsed_ns",
            "telemetry",
        }
        assert document["elapsed_ns"] == list(sample.elapsed_ns)
        assert len(document["telemetry"]) == 2


def test_microbenchmark_validates_counts_and_preserves_padded_rows() -> None:
    executor = _executor()
    hidden = np.ones((4, 4), dtype=np.float32)
    ids = np.array([[0, -1], [1, 2], [-2, 99], [-2, 99]], dtype=np.int32)
    weights = np.ones(ids.shape, dtype=np.float32)

    samples = executor.microbenchmark(
        0,
        hidden,
        ids,
        weights,
        repeats=1,
        route_counts=(1, 2),
        miss_counts=(2, 3),
        num_token_non_padded=2,
    )
    assert [sample.miss_count for sample in samples] == [2, 3]
    assert all(item.telemetry[0].tokens_non_padded == 2 for item in samples)

    with pytest.raises(InvalidRequest, match="does not match"):
        executor.microbenchmark(
            0,
            hidden,
            ids,
            weights,
            route_counts=(1,),
            miss_counts=(1,),
            num_token_non_padded=2,
        )
    with pytest.raises(InvalidRequest, match="route_count"):
        executor.microbenchmark(0, hidden, ids, weights, route_counts=(0,))
    with pytest.raises(InvalidRequest, match="repeats"):
        executor.microbenchmark(0, hidden, ids, weights, repeats=0)


def test_microbenchmark_rejects_empty_supplied_selection() -> None:
    executor = _executor()
    with pytest.raises(InvalidRequest, match="at least one supplied route"):
        executor.microbenchmark(
            0,
            np.ones((1, 4), dtype=np.float32),
            np.empty((1, 0), dtype=np.int32),
            np.empty((1, 0), dtype=np.float32),
        )


def test_executor_handles_empty_selection_and_full_prepared_route_width() -> None:
    executor = _executor()
    hidden = np.ones((2, 4), dtype=np.float32)
    empty = executor.execute(
        0,
        hidden,
        np.empty((2, 0), dtype=np.int32),
        np.empty((2, 0), dtype=np.float32),
    )
    assert empty.output.shape == hidden.shape
    assert np.all(empty.output == 0)
    assert empty.telemetry.routes_requested == 0
    assert empty.telemetry.fallback_reason == "reference_no_routes"

    ids = np.tile(np.arange(10, dtype=np.int32), (2, 1)) % 5
    weights = np.ones((2, 10), dtype=np.float32)
    full = executor.execute(0, hidden, ids, weights)
    assert full.telemetry.routes_requested == 20
    assert full.telemetry.routes_executed == 20


def test_prepared_workspace_and_caller_output_are_reused() -> None:
    executor = _executor()
    assert executor._workspace is not None
    workspace_ids = {name: id(value) for name, value in executor._workspace.items()}
    hidden = np.ones((2, 4), dtype=np.float32)
    ids = np.array([[0, 1], [2, 3]], dtype=np.int32)
    weights = np.ones((2, 2), dtype=np.float32)
    output = np.empty((2, 4), dtype=np.float32)

    first = executor.execute(0, hidden, ids, weights, output=output)
    second = executor.execute(0, hidden, ids, weights, output=output)

    assert first.output is output
    assert second.output is output
    assert executor._workspace is not None
    assert {name: id(value) for name, value in executor._workspace.items()} == workspace_ids


def test_prepared_dense_execution_makes_no_executor_array_allocations(monkeypatch) -> None:
    executor = _executor()
    hidden = np.ones((2, 4), dtype=np.float32)
    ids = np.array([[0, 1], [2, 3]], dtype=np.int32)
    weights = np.ones((2, 2), dtype=np.float32)
    output = np.empty((2, 4), dtype=np.float32)

    def reject_empty(*_args, **_kwargs):
        raise AssertionError("prepared execution must reuse its NumPy workspace")

    monkeypatch.setattr("freetoken.moe.cpu_abi.np.empty", reject_empty)
    result = executor.execute(0, hidden, ids, weights, output=output)
    assert result.output is output


def test_malformed_output_preserves_invalid_request_type() -> None:
    executor = _executor()
    with pytest.raises(InvalidRequest, match="NumPy ndarray"):
        executor.execute(
            0,
            np.ones((1, 4), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int32),
            np.ones((1, 1), dtype=np.float32),
            output=[[7.0, 7.0, 7.0, 7.0]],
        )


def test_prepare_rejects_oversized_request_without_dynamic_resize() -> None:
    executor = _executor()
    plan = executor.last_telemetry
    assert plan is None
    with pytest.raises(WorkspaceTooSmall):
        executor.execute(
            0,
            np.ones((129, 4), dtype=np.float32),
            np.zeros((129, 1), dtype=np.int32),
            np.ones((129, 1), dtype=np.float32),
        )


def test_single_flight_executor_reports_busy() -> None:
    executor = _executor()
    acquired = threading.Event()
    release = threading.Event()

    original = executor._dense_expert

    def blocked(descriptor, expert, *args, **kwargs):
        acquired.set()
        release.wait(timeout=2)
        return original(descriptor, expert, *args, **kwargs)

    executor._dense_expert = blocked
    errors: list[Exception] = []

    def run() -> None:
        try:
            executor.execute(
                0,
                np.ones((1, 4), dtype=np.float32),
                np.zeros((1, 1), dtype=np.int32),
                np.ones((1, 1), dtype=np.float32),
            )
        except Exception as error:  # pragma: no cover - assertion below identifies errors
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    assert acquired.wait(timeout=2)
    with pytest.raises(Busy):
        executor.execute(
            0,
            np.ones((1, 4), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int32),
            np.ones((1, 1), dtype=np.float32),
        )
    release.set()
    thread.join(timeout=2)
    assert not errors


def test_independent_executors_can_progress_concurrently() -> None:
    left = _executor()
    right = _executor()
    barrier = threading.Barrier(2)

    def pause_once(executor):
        original = executor._dense_expert
        entered = False

        def wrapped(*args, **kwargs):
            nonlocal entered
            if not entered:
                entered = True
                barrier.wait(timeout=2)
            return original(*args, **kwargs)

        executor._dense_expert = wrapped

    pause_once(left)
    pause_once(right)
    errors: list[Exception] = []

    def run(executor) -> None:
        try:
            executor.execute(
                0,
                np.ones((1, 4), dtype=np.float32),
                np.zeros((1, 1), dtype=np.int32),
                np.ones((1, 1), dtype=np.float32),
            )
        except Exception as error:  # pragma: no cover - assertion below identifies errors
            errors.append(error)

    threads = [threading.Thread(target=run, args=(executor,)) for executor in (left, right)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert not errors
    assert all(not thread.is_alive() for thread in threads)


def test_thread_pool_and_numa_hooks_are_reserved_by_reference_executor() -> None:
    base = _executor()
    calls: list[object] = []

    class Pool:
        def submit(self, *_args, **_kwargs):
            calls.append("pool")
            raise AssertionError("reference executor must not submit worker tasks")

    class Numa:
        def placement(self, *_args, **_kwargs):
            calls.append("numa")
            raise AssertionError("reference executor must not select NUMA placement")

    executor = ReferenceCpuExpertExecutor(
        base.layout,
        thread_pool=Pool(),
        numa_policy=Numa(),
    )
    executor.prepare(1, 1)
    assert calls == []


def test_structural_adapter_preserves_heterogeneous_pools_without_qwen_imports() -> None:
    class SourceDescriptor:
        def __init__(self, layer, projection, quant_type, quant_name, pool_id):
            self.layer = layer
            self.projection = projection
            self.quant_type = quant_type
            self.quant_name = quant_name
            self.experts = 3
            self.output_dim = 4 if projection != "down" else 2
            self.input_dim = 2 if projection != "down" else 4
            self.row_bytes = self.input_dim * 4
            self.bytes_per_expert = self.output_dim * self.row_bytes
            self.tensor_bytes = self.experts * self.bytes_per_expert
            self.data_offset = 4096 + layer * 100
            self.pool_id = pool_id

    class SourceLayout:
        descriptors = tuple(
            SourceDescriptor(0, projection, quant, name, pool)
            for projection, quant, name, pool in (
                ("gate", 12, "Q4_K", 1),
                ("up", 13, "Q5_K", 2),
                ("down", 18, "IQ3_XXS", 3),
            )
        )

    sources = {(0, descriptor.projection): object() for descriptor in SourceLayout.descriptors}
    adapted = cpu_layout_from_source_layout(SourceLayout(), sources, top_k=10)
    assert [item.quant_name for item in adapted.descriptors] == [
        "Q4_K",
        "Q5_K",
        "IQ3_XXS",
    ]
    assert [item.pool_id for item in adapted.descriptors] == [1, 2, 3]
    assert adapted.descriptor(0, "gate").source is sources[(0, "gate")]


@pytest.mark.parametrize(
    ("filename", "required_quant_names"),
    [
        ("qwen38-q4-census.metadata.json", {"Q4_K", "Q5_1", "Q8_0"}),
        ("qwen38-q3-census.metadata.json", {"IQ3_XXS", "IQ4_NL", "Q8_0"}),
    ],
)
def test_qwen38_census_descriptors_cover_q4_and_three_bit_artifacts(
    filename: str, required_quant_names: set[str]
) -> None:
    root = Path(__file__).resolve().parents[2]
    census = json.loads((root / "tests/fixtures/results" / filename).read_text(encoding="utf-8"))
    # Import the host-layout parser only for the real pinned census shape; no
    # model-specific type is imported by the ABI adapter itself.
    from freetoken.gguf_host import expert_layout_from_census

    source_layout = expert_layout_from_census(census)
    adapted = cpu_layout_from_source_layout(
        source_layout,
        lambda _layer, _projection: None,
        top_k=10,
    )
    assert len(adapted.descriptors) == 48 * 3
    assert {item.num_experts for item in adapted.descriptors} == {512}
    assert required_quant_names <= {item.quant_name for item in adapted.descriptors}
    assert {item.projection for item in adapted.descriptors} == {"gate", "up", "down"}


def test_reference_executor_uses_registered_packed_decoder_and_reports_packed_bytes() -> None:
    rng = np.random.default_rng(7)
    experts, hidden_size, intermediate_size = 2, 3, 4
    dense = {
        projection: rng.normal(size=(experts, output, input_size)).astype(np.float32)
        for projection, output, input_size in (
            ("gate", intermediate_size, hidden_size),
            ("up", intermediate_size, hidden_size),
            ("down", hidden_size, intermediate_size),
        )
    }

    class PackedSource:
        def __init__(self, values):
            self.values = values
            self.range_offset = 0
            self.range_size = int(values.nbytes)

        def expert_packed(self, expert):
            return self.values[expert].view(np.uint8)

    sources = {projection: PackedSource(values) for projection, values in dense.items()}
    descriptors = []
    for pool_id, (projection, values) in enumerate(dense.items()):
        output, input_size = values.shape[1:]
        row_bytes = input_size * 4
        descriptors.append(
            CpuExpertDescriptor(
                layer_id=0,
                projection=projection,
                quant_type=12,
                quant_name="Q4_K",
                num_experts=experts,
                output_dim=output,
                input_dim=input_size,
                rows_per_expert=output,
                row_stride_bytes=row_bytes,
                expert_stride_bytes=output * row_bytes,
                tensor_bytes=experts * output * row_bytes,
                pool_id=pool_id,
                source=sources[projection],
            )
        )

    def decode(packed, descriptor):
        assert packed.shape == (descriptor.output_dim, descriptor.row_stride_bytes)
        return packed.view(np.float32).reshape(descriptor.output_dim, descriptor.input_dim)

    executor = ReferenceCpuExpertExecutor(
        CpuExpertLayout(tuple(descriptors), top_k=2),
        decoders={12: decode},
    )
    executor.prepare(1, 2)
    result = executor.execute(
        0,
        np.ones((1, hidden_size), dtype=np.float32),
        np.array([[0, 1]], dtype=np.int32),
        np.array([[0.25, 0.75]], dtype=np.float32),
    )
    assert np.isfinite(result.output).all()
    expected_bytes = sum(item.expert_stride_bytes for item in descriptors) * 2
    assert result.telemetry.bytes_read_packed == expected_bytes
    assert result.telemetry.fallback_reason == "reference_dequant_packed_legacy"


def test_workspace_aware_decoder_reuses_bounded_scratch() -> None:
    experts, hidden_size, intermediate_size = 2, 3, 4
    dense = {
        projection: np.arange(experts * output * input_size, dtype=np.float32).reshape(
            experts, output, input_size
        )
        for projection, output, input_size in (
            ("gate", intermediate_size, hidden_size),
            ("up", intermediate_size, hidden_size),
            ("down", hidden_size, intermediate_size),
        )
    }

    class PackedSource:
        def __init__(self, values):
            self.values = values
            self.range_offset = 0
            self.range_size = int(values.nbytes)

        def expert_packed(self, expert):
            return self.values[expert].view(np.uint8)

    sources = {projection: PackedSource(values) for projection, values in dense.items()}
    descriptors = []
    for pool_id, (projection, values) in enumerate(dense.items()):
        output, input_size = values.shape[1:]
        row_stride = input_size * 4
        descriptors.append(
            CpuExpertDescriptor(
                layer_id=0,
                projection=projection,
                quant_type=12,
                quant_name="Q4_K",
                num_experts=experts,
                output_dim=output,
                input_dim=input_size,
                rows_per_expert=output,
                row_stride_bytes=row_stride,
                expert_stride_bytes=output * row_stride,
                tensor_bytes=experts * output * row_stride,
                pool_id=pool_id,
                source=sources[projection],
            )
        )
    scratch_addresses: dict[str, list[int]] = {name: [] for name in dense}

    def decode(packed, descriptor, *, out):
        scratch_addresses[descriptor.projection].append(int(out.__array_interface__["data"][0]))
        values = packed.view(np.float32).reshape(descriptor.output_dim, descriptor.input_dim)
        np.copyto(out, values)
        return out

    executor = ReferenceCpuExpertExecutor(
        CpuExpertLayout(descriptors, top_k=2),
        decoders={12: decode},
    )
    executor.prepare(1, 1)
    result = executor.execute(
        0,
        np.ones((1, 3), dtype=np.float32),
        np.array([[0]], dtype=np.int32),
        np.ones((1, 1), dtype=np.float32),
    )
    gate_out = dense["gate"][0] @ np.ones(3, dtype=np.float32)
    up_out = dense["up"][0] @ np.ones(3, dtype=np.float32)
    activated = 1.0 / (1.0 + np.exp(-gate_out)) * gate_out * up_out
    expected = (dense["down"][0] @ activated).reshape(1, 3)
    np.testing.assert_allclose(result.output, expected)
    assert all(len(addresses) == 1 for addresses in scratch_addresses.values())
    assert len({addresses[0] for addresses in scratch_addresses.values()}) == 3
    assert result.telemetry.fallback_reason == "reference_dequant_packed_workspace"


@pytest.mark.parametrize("bad_shape", [(1, 16), (2, 7), (1, 2, 8)])
def test_packed_source_shape_must_match_descriptor_byte_rows(bad_shape: tuple[int, ...]) -> None:
    class BadPackedSource:
        range_offset = 0
        range_size = 32

        def expert_packed(self, _expert):
            return np.zeros(bad_shape, dtype=np.uint8)

    descriptor = CpuExpertDescriptor(
        layer_id=0,
        projection="gate",
        quant_type=12,
        quant_name="Q4_K",
        num_experts=2,
        output_dim=2,
        input_dim=2,
        rows_per_expert=2,
        row_stride_bytes=8,
        expert_stride_bytes=16,
        tensor_bytes=32,
        source=BadPackedSource(),
    )
    with pytest.raises(UnsupportedShape, match=r"packed.*expected"):
        executor = ReferenceCpuExpertExecutor(
            CpuExpertLayout(
                (
                    descriptor,
                    _descriptor(0, "up", np.ones((2, 2, 2), dtype=np.float32)),
                    _descriptor(0, "down", np.ones((2, 2, 2), dtype=np.float32)),
                ),
                top_k=1,
            ),
            decoders={12: lambda packed, item: packed},
        )
        executor.prepare(1, 1)
        executor.execute(
            0,
            np.ones((1, 2), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int32),
            np.ones((1, 1), dtype=np.float32),
        )


def test_packed_source_must_be_a_contiguous_uint8_view() -> None:
    class NonContiguousPackedSource:
        range_offset = 0
        range_size = 32

        def expert_packed(self, _expert):
            return np.zeros((2, 16), dtype=np.uint8)[:, ::2]

    descriptor = CpuExpertDescriptor(
        layer_id=0,
        projection="gate",
        quant_type=12,
        quant_name="Q4_K",
        num_experts=2,
        output_dim=2,
        input_dim=2,
        rows_per_expert=2,
        row_stride_bytes=8,
        expert_stride_bytes=16,
        tensor_bytes=32,
        source=NonContiguousPackedSource(),
    )
    executor = ReferenceCpuExpertExecutor(
        CpuExpertLayout(
            (
                descriptor,
                _descriptor(0, "up", np.ones((2, 2, 2), dtype=np.float32)),
                _descriptor(0, "down", np.ones((2, 2, 2), dtype=np.float32)),
            ),
            top_k=1,
        ),
        decoders={12: lambda packed, item: packed},
    )
    executor.prepare(1, 1)
    with pytest.raises(InvalidRequest, match="contiguous uint8"):
        executor.execute(
            0,
            np.ones((1, 2), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int32),
            np.ones((1, 1), dtype=np.float32),
        )


def test_unsupported_packed_decoder_fails_explicitly() -> None:
    executor = _executor()

    class PackedSource:
        range_offset = 0
        range_size = 480

        def expert_packed(self, _expert):
            return np.zeros((6, 16), dtype=np.uint8)

    descriptor = executor.layout.descriptor(0, "gate")
    # Replace only the gate source with an unsupported packed source; the descriptor
    # itself still records the source geometry and quant contract unchanged.
    replacement = CpuExpertDescriptor(
        **{
            name: getattr(descriptor, name)
            for name in (
                "layer_id",
                "projection",
                "quant_type",
                "quant_name",
                "num_experts",
                "output_dim",
                "input_dim",
                "rows_per_expert",
                "row_stride_bytes",
                "expert_stride_bytes",
                "tensor_bytes",
                "source_offset",
                "source_address",
                "pool_id",
            )
        },
        source=PackedSource(),
    )
    descriptors = tuple(
        replacement if item.projection == "gate" else item for item in executor.layout.descriptors
    )
    failing = ReferenceCpuExpertExecutor(CpuExpertLayout(descriptors, top_k=10))
    failing.prepare(1, 1)
    with pytest.raises(UnsupportedQuantType, match="decoder"):
        failing.execute(
            0,
            np.ones((1, 4), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int32),
            np.ones((1, 1), dtype=np.float32),
        )
    assert failing.last_telemetry is not None
    assert failing.last_telemetry.fallback_reason == "reference_mixed_dense_packed"
    assert failing.last_telemetry.error_detail is not None


def test_packed_source_without_a_provable_range_fails_closed() -> None:
    class UnboundedPackedSource:
        def expert_packed(self, _expert):
            return np.zeros((2, 8), dtype=np.uint8)

    with pytest.raises(InvalidRequest, match="bounded"):
        CpuExpertDescriptor(
            layer_id=0,
            projection="gate",
            quant_type=12,
            quant_name="Q4_K",
            num_experts=2,
            output_dim=2,
            input_dim=2,
            rows_per_expert=2,
            row_stride_bytes=8,
            expert_stride_bytes=16,
            tensor_bytes=32,
            source=UnboundedPackedSource(),
        )


def test_source_range_bounds_include_absolute_offset() -> None:
    source = np.zeros((2, 2, 2), dtype=np.float32)
    with pytest.raises(InvalidRequest, match="outside"):
        CpuExpertDescriptor(
            layer_id=0,
            projection="gate",
            quant_type="F32",
            quant_name="F32",
            num_experts=2,
            output_dim=2,
            input_dim=2,
            rows_per_expert=2,
            row_stride_bytes=8,
            expert_stride_bytes=16,
            tensor_bytes=32,
            source_offset=8,
            source=source,
        )


def test_mapped_source_address_is_derived_at_the_tensor_start() -> None:
    class MappedSource:
        range_offset = 4096
        range_size = 32

        class Descriptor:
            data_offset = 4096
            tensor_bytes = 32

        class Mapping:
            _address = 0x1003
            _prefix = 5
            length = 32

        descriptor = Descriptor()
        mapping = Mapping()

        def expert_packed(self, _expert):
            return np.zeros((2, 16), dtype=np.uint8)

    descriptor = CpuExpertDescriptor(
        layer_id=0,
        projection="gate",
        quant_type=12,
        quant_name="Q4_K",
        num_experts=2,
        output_dim=2,
        input_dim=2,
        rows_per_expert=2,
        row_stride_bytes=8,
        expert_stride_bytes=16,
        tensor_bytes=32,
        source_offset=4096,
        source=MappedSource(),
    )
    assert descriptor.source_address == 0x1008


def test_packed_source_must_agree_with_every_exposed_range() -> None:
    class ConflictingMappedSource:
        range_offset = 0
        range_size = 32

        class Descriptor:
            data_offset = 0
            tensor_bytes = 32

        class Mapping:
            length = 16

        descriptor = Descriptor()
        mapping = Mapping()

        def expert_packed(self, _expert):
            return np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(InvalidRequest, match="outside"):
        CpuExpertDescriptor(
            layer_id=0,
            projection="gate",
            quant_type=12,
            quant_name="Q4_K",
            num_experts=2,
            output_dim=2,
            input_dim=2,
            rows_per_expert=2,
            row_stride_bytes=8,
            expert_stride_bytes=16,
            tensor_bytes=32,
            source=ConflictingMappedSource(),
        )


@pytest.mark.parametrize("field", ["layer_id", "num_experts", "output_dim", "input_dim"])
@pytest.mark.parametrize("value", [True, 1.5])
def test_descriptor_rejects_bool_and_non_integral_scalars(field: str, value) -> None:
    source = np.zeros((2, 2, 2), dtype=np.float32)
    values = dict(
        layer_id=0,
        projection="gate",
        quant_type="F32",
        quant_name="F32",
        num_experts=2,
        output_dim=2,
        input_dim=2,
        rows_per_expert=2,
        row_stride_bytes=8,
        expert_stride_bytes=16,
        tensor_bytes=32,
        source=source,
    )
    values[field] = value
    with pytest.raises(InvalidRequest, match="integer"):
        CpuExpertDescriptor(**values)


def test_prepare_and_request_scalars_are_strict() -> None:
    executor = _executor()
    with pytest.raises(InvalidRequest, match="integer"):
        executor.prepare(True, 1)
    with pytest.raises(InvalidRequest, match="integer"):
        executor.prepare(1, 1.5)
    with pytest.raises(InvalidRequest, match="integer"):
        executor.execute(
            True,
            np.ones((1, 4), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int32),
            np.ones((1, 1), dtype=np.float32),
        )
    with pytest.raises(InvalidRequest, match="integer"):
        executor.execute(
            0,
            np.ones((1, 4), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int32),
            np.ones((1, 1), dtype=np.float32),
            num_token_non_padded=1.5,
        )
    with pytest.raises(InvalidRequest, match=r"accumulate.*bool"):
        executor.execute(
            0,
            np.ones((1, 4), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int32),
            np.ones((1, 1), dtype=np.float32),
            accumulate=1,
        )
    with pytest.raises(InvalidRequest, match=r"request layer_id.*integer"):
        CpuExecutionRequest(
            layer_id=True,
            hidden=np.ones((1, 4), dtype=np.float32),
            expert_ids=np.zeros((1, 1), dtype=np.int32),
            routing_weights=np.ones((1, 1), dtype=np.float32),
        )
    with pytest.raises(InvalidRequest, match=r"request num_token_non_padded.*integer"):
        CpuExecutionRequest(
            layer_id=0,
            hidden=np.ones((1, 4), dtype=np.float32),
            expert_ids=np.zeros((1, 1), dtype=np.int32),
            routing_weights=np.ones((1, 1), dtype=np.float32),
            num_token_non_padded=1.5,
        )
    with pytest.raises(InvalidRequest, match=r"request accumulate.*bool"):
        CpuExecutionRequest(
            layer_id=0,
            hidden=np.ones((1, 4), dtype=np.float32),
            expert_ids=np.zeros((1, 1), dtype=np.int32),
            routing_weights=np.ones((1, 1), dtype=np.float32),
            accumulate=1,
        )
    with pytest.raises(InvalidRequest, match="bool"):
        ReferenceCpuExpertExecutor(
            executor.layout,
            apply_router_weight_on_input=1,
        )


def test_all_layers_must_share_the_prepared_workspace_geometry() -> None:
    rng = np.random.default_rng(12)
    layer0 = (
        rng.normal(size=(2, 4, 3)).astype(np.float32),
        rng.normal(size=(2, 4, 3)).astype(np.float32),
        rng.normal(size=(2, 3, 4)).astype(np.float32),
    )
    layer1 = (
        rng.normal(size=(2, 5, 3)).astype(np.float32),
        rng.normal(size=(2, 5, 3)).astype(np.float32),
        rng.normal(size=(2, 3, 5)).astype(np.float32),
    )
    descriptors = tuple(
        _descriptor(layer, projection, source)
        for layer, banks in ((0, layer0), (1, layer1))
        for projection, source in zip(("gate", "up", "down"), banks, strict=True)
    )
    with pytest.raises(UnsupportedShape, match="workspace geometry"):
        ReferenceCpuExpertExecutor(CpuExpertLayout(descriptors, top_k=2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layer_id": 0, "projection": "gate", "rows_per_expert": 3, "output_dim": 2},
        {"layer_id": 0, "projection": "gate", "expert_stride_bytes": 7},
    ],
)
def test_descriptor_rejects_inconsistent_geometry(kwargs) -> None:
    values = dict(
        layer_id=0,
        projection="gate",
        quant_type="F32",
        quant_name="F32",
        num_experts=2,
        output_dim=2,
        input_dim=2,
        rows_per_expert=2,
        row_stride_bytes=8,
        expert_stride_bytes=16,
        tensor_bytes=32,
    )
    values.update(kwargs)
    with pytest.raises((InvalidRequest, UnsupportedShape)):
        CpuExpertDescriptor(**values)


def test_executor_rejects_a_source_alignment_it_cannot_guarantee() -> None:
    executor = _executor()
    descriptor = executor.layout.descriptor(0, "gate")
    values = {
        name: getattr(descriptor, name)
        for name in (
            "layer_id",
            "projection",
            "quant_type",
            "quant_name",
            "num_experts",
            "output_dim",
            "input_dim",
            "rows_per_expert",
            "row_stride_bytes",
            "expert_stride_bytes",
            "tensor_bytes",
            "source_offset",
            "pool_id",
            "source",
        )
    }
    values["source_address"] = 3
    replacement = CpuExpertDescriptor(**values)
    layout = CpuExpertLayout(
        tuple(
            replacement if item.projection == "gate" else item
            for item in executor.layout.descriptors
        ),
        top_k=10,
    )
    with pytest.raises(UnsupportedAlignment, match="aligned"):
        ReferenceCpuExpertExecutor(layout, required_alignment=16)
