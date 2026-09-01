"""CPU-only production bridge for heterogeneous Qwen GGUF expert banks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from freetoken.gguf_host import (
    ExpertBankDescriptor,
    GGUFExpertLayout,
    QwenGGUFHostWeights,
    QwenHostLayout,
)
from freetoken.moe.cpu_topology import CpuTopology, PhysicalCore

Q4_K = 12
Q5_K = 13
Q5_1 = 7
Q8_0 = 8
IQ3_XXS = 18
IQ4_NL = 20
IQ4_XS = 23


@dataclass
class _Mapping:
    length: int
    source_address: int


class _Bank:
    def __init__(self, descriptor: ExpertBankDescriptor, values: np.ndarray) -> None:
        self.descriptor = descriptor
        self.values = values
        self.mapping = _Mapping(
            length=int(values.nbytes),
            source_address=int(values.__array_interface__["data"][0]),
        )

    def expert_packed(self, expert: int) -> np.ndarray:
        result = self.values[expert]
        result.flags.writeable = False
        return result

    def close(self) -> None:
        self.values = np.empty((0, 0), dtype=np.uint8)


class _Banks:
    def __init__(self, layout: GGUFExpertLayout, banks: dict[tuple[int, str], _Bank]) -> None:
        self.layout = layout
        self._banks = banks

    def bank(self, layer: int, projection: str) -> _Bank:
        return self._banks[(layer, projection)]

    def close(self) -> None:
        for bank in self._banks.values():
            bank.close()


class _Ple:
    tensor_bytes = 0

    def close(self) -> None:
        return None


def _descriptor(
    layer: int,
    projection: str,
    quant_type: int,
    quant_name: str,
    *,
    input_dim: int = 256,
    output_dim: int = 256,
    experts: int = 2,
) -> tuple[ExpertBankDescriptor, np.ndarray]:
    block_bytes = {
        "Q4_K": 144,
        "Q5_K": 176,
        "Q5_1": 24,
        "Q8_0": 34,
        "IQ3_XXS": 98,
        "IQ4_NL": 18,
        "IQ4_XS": 136,
        "MYSTERY": 34,
    }[quant_name]
    block_elements = 256 if quant_name in {"Q4_K", "Q5_K", "IQ3_XXS", "IQ4_XS"} else 32
    row_bytes = input_dim // block_elements * block_bytes
    values = np.ascontiguousarray(
        np.arange(experts * output_dim * row_bytes, dtype=np.uint8).reshape(
            experts, output_dim, row_bytes
        )
    )
    descriptor = ExpertBankDescriptor(
        layer=layer,
        projection=projection,
        tensor_name=f"blk.{layer}.ffn_{projection}_exps.weight",
        quant_type=quant_type,
        quant_name=quant_name,
        experts=experts,
        output_dim=output_dim,
        input_dim=input_dim,
        row_bytes=row_bytes,
        bytes_per_expert=output_dim * row_bytes,
        tensor_bytes=int(values.nbytes),
        shard_index=0,
        shard_path="synthetic.gguf",
        data_offset=0,
    )
    return descriptor, values


def _host() -> QwenGGUFHostWeights:
    descriptors = []
    banks = {}
    for projection, quant_type, quant_name in (
        ("gate", Q4_K, "Q4_K"),
        ("up", Q4_K, "Q4_K"),
        ("down", Q5_1, "Q5_1"),
    ):
        descriptor, values = _descriptor(0, projection, quant_type, quant_name)
        descriptors.append(descriptor)
        banks[(0, projection)] = _Bank(descriptor, values)
    expert_layout = GGUFExpertLayout(
        descriptors=tuple(descriptors),
        slot_pools=(),
        num_layers=1,
        num_experts=2,
    )
    layout = QwenHostLayout(
        experts=expert_layout,
        ple=_Ple(),  # type: ignore[arg-type]
        total_tensor_bytes=sum(item.tensor_bytes for item in descriptors),
        shard_paths=("synthetic.gguf",),
    )
    return QwenGGUFHostWeights(layout, _Banks(expert_layout, banks), _Ple())  # type: ignore[arg-type]


def _geometry_host(*, promoted: bool) -> QwenGGUFHostWeights:
    descriptors = []
    banks = {}
    gate_up = (Q5_K, "Q5_K") if promoted else (Q4_K, "Q4_K")
    down = (Q8_0, "Q8_0") if promoted else (Q5_1, "Q5_1")
    for projection, (quant_type, quant_name) in (
        ("gate", gate_up),
        ("up", gate_up),
        ("down", down),
    ):
        descriptor, values = _descriptor(
            0,
            projection,
            quant_type,
            quant_name,
            input_dim=2560 if projection != "down" else 640,
            output_dim=640 if projection != "down" else 2560,
        )
        descriptors.append(descriptor)
        banks[(0, projection)] = _Bank(descriptor, values)
    expert_layout = GGUFExpertLayout(
        descriptors=tuple(descriptors),
        slot_pools=(),
        num_layers=1,
        num_experts=2,
    )
    layout = QwenHostLayout(
        experts=expert_layout,
        ple=_Ple(),  # type: ignore[arg-type]
        total_tensor_bytes=sum(item.tensor_bytes for item in descriptors),
        shard_paths=("synthetic.gguf",),
    )
    return QwenGGUFHostWeights(layout, _Banks(expert_layout, banks), _Ple())  # type: ignore[arg-type]


def _qwen_census_geometry_host(
    *, layer: int, gate_up: tuple[int, str], down: tuple[int, str]
) -> QwenGGUFHostWeights:
    """Build one expert with the Qwen census geometry without a 512-expert bank."""
    descriptors = []
    banks = {}
    for projection, (quant_type, quant_name) in (
        ("gate", gate_up),
        ("up", gate_up),
        ("down", down),
    ):
        descriptor, values = _descriptor(
            layer,
            projection,
            quant_type,
            quant_name,
            input_dim=2560 if projection != "down" else 640,
            output_dim=640 if projection != "down" else 2560,
            experts=1,
        )
        descriptors.append(descriptor)
        banks[(layer, projection)] = _Bank(descriptor, values)
    expert_layout = GGUFExpertLayout(
        descriptors=tuple(descriptors),
        slot_pools=(),
        num_layers=layer + 1,
        num_experts=1,
    )
    layout = QwenHostLayout(
        experts=expert_layout,
        ple=_Ple(),  # type: ignore[arg-type]
        total_tensor_bytes=sum(item.tensor_bytes for item in descriptors),
        shard_paths=("synthetic.gguf",),
    )
    return QwenGGUFHostWeights(layout, _Banks(expert_layout, banks), _Ple())  # type: ignore[arg-type]


def test_bundle_keeps_host_alive_and_closes_owned_mapping_once() -> None:
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    host = _host()
    bundle = QwenGGUFCpuExpertBundle.from_host(host, top_k=1, mode="scalar")
    del host
    assert bundle.host.layout.experts.num_layers == 1
    bundle.close()
    bundle.close()
    assert bundle.closed
    with pytest.raises(RuntimeError, match="closed"):
        bundle.decode(0, None, None, None)  # type: ignore[arg-type]
    # A retained host reference or an enclosing host context may close again
    # after the owner bundle has completed cleanup; that call is idempotent.
    bundle.host.close()


def test_bundle_thread_policy_defaults_to_safe_serial_and_reports_no_execution() -> None:
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    bundle = QwenGGUFCpuExpertBundle.from_host(_host(), top_k=1, mode="scalar")
    assert bundle.requested_num_threads is None
    assert bundle.effective_num_threads == 1
    assert bundle.actual_thread_count is None
    telemetry = bundle.host_weight_telemetry()
    assert telemetry["requested_num_threads"] is None
    assert telemetry["effective_num_threads"] == 1
    assert telemetry["actual_thread_count"] is None
    assert telemetry["threading_fallback_reason"] is None
    bundle.close()


def test_direct_bundle_constructor_keeps_legacy_serial_defaults() -> None:
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    class _Host:
        closed = False

    bundle = QwenGGUFCpuExpertBundle(_Host(), object(), object(), output_dtype=np.float32)
    assert bundle.requested_num_threads is None
    assert bundle.effective_num_threads == 1


def test_bundle_zero_thread_request_is_safe_serial() -> None:
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    bundle = QwenGGUFCpuExpertBundle.from_host(_host(), top_k=1, mode="scalar", num_threads=0)
    assert bundle.requested_num_threads == 0
    assert bundle.effective_num_threads == 1
    bundle.close()


def test_bundle_rejects_thread_request_above_visible_physical_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import freetoken.moe.gguf_cpu as bridge

    topology = CpuTopology(
        allowed_cpus=(40, 42),
        cores=tuple(
            PhysicalCore(
                key=f"core-{cpu}",
                representative=cpu,
                logical_cpus=(cpu,),
                siblings=(cpu,),
            )
            for cpu in (40, 42)
        ),
        confidence="full",
        source="synthetic",
    )
    monkeypatch.setattr(bridge, "discover_cpu_topology", lambda: topology)
    with pytest.raises(ValueError, match="affinity-visible physical-core capacity of 2"):
        bridge.QwenGGUFCpuExpertBundle.from_host(_host(), top_k=1, mode="scalar", num_threads=3)


def test_positive_bridge_threads_resolve_shared_worker_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import freetoken.moe.gguf_cpu as bridge

    topology = CpuTopology(
        allowed_cpus=(40, 42),
        cores=tuple(
            PhysicalCore(
                key=f"core-{cpu}",
                representative=cpu,
                logical_cpus=(cpu,),
                siblings=(cpu,),
            )
            for cpu in (40, 42)
        ),
        confidence="full",
        source="synthetic",
    )
    monkeypatch.setattr(bridge, "discover_cpu_topology", lambda: topology)
    requested, effective, plan = bridge._resolve_bridge_worker_plan(2)
    assert requested == 2
    assert effective == 2
    assert plan is not None
    assert plan.worker_cpus == (40, 42)
    assert plan.affinity_status == "planned-unverified"


def test_open_rejects_thread_plan_before_host_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    import freetoken.moe.gguf_cpu as bridge

    topology = CpuTopology(
        allowed_cpus=(40, 42),
        cores=tuple(
            PhysicalCore(
                key=f"core-{cpu}",
                representative=cpu,
                logical_cpus=(cpu,),
                siblings=(cpu,),
            )
            for cpu in (40, 42)
        ),
        confidence="full",
        source="synthetic",
    )
    monkeypatch.setattr(bridge, "discover_cpu_topology", lambda: topology)
    monkeypatch.setattr(
        bridge,
        "open_qwen_host_weights",
        lambda *_args, **_kwargs: pytest.fail("host mapping must not start"),
    )
    with pytest.raises(ValueError, match="physical-core capacity of 2"):
        bridge.open_qwen_gguf_cpu_expert_bundle("/does/not/exist.gguf", top_k=1, num_threads=3)


def test_open_passes_one_resolved_plan_to_bundle_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import freetoken.moe.gguf_cpu as bridge

    topology = CpuTopology(
        allowed_cpus=(40, 42),
        cores=tuple(
            PhysicalCore(
                key=f"core-{cpu}",
                representative=cpu,
                logical_cpus=(cpu,),
                siblings=(cpu,),
            )
            for cpu in (40, 42)
        ),
        confidence="full",
        source="synthetic",
    )
    discoveries = 0

    def discover() -> CpuTopology:
        nonlocal discoveries
        discoveries += 1
        return topology

    captured: dict[str, object] = {}
    sentinel = object()

    def from_host(host: object, **kwargs: object) -> object:
        assert host is not None
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(bridge, "discover_cpu_topology", discover)
    monkeypatch.setattr(bridge, "open_qwen_host_weights", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        bridge.QwenGGUFCpuExpertBundle,
        "from_host",
        staticmethod(from_host),
    )
    result = bridge.open_qwen_gguf_cpu_expert_bundle("/synthetic.gguf", top_k=1, num_threads=2)
    assert result is sentinel
    assert discoveries == 1
    resolved = captured["_resolved_thread_policy"]
    assert isinstance(resolved, tuple)
    assert resolved[2].worker_cpus == (40, 42)


@pytest.mark.parametrize("num_threads", [-1, True, 1.5, "2"])
def test_bundle_rejects_malformed_thread_request(num_threads: object) -> None:
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    with pytest.raises(ValueError, match="num_threads"):
        QwenGGUFCpuExpertBundle.from_host(
            _host(),
            top_k=1,
            mode="scalar",
            num_threads=num_threads,  # type: ignore[arg-type]
        )


def test_bundle_persists_threaded_decode_telemetry_and_actual_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    from freetoken.moe import q4_k
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    class _FastQ4:
        isa = "avx2"
        backend = "q4_k_test"
        fallback_reason = None

        def gemv(self, rows, input_dim, vector, *, out, scratch=None):
            del input_dim, scratch
            np.multiply(rows[:, 0].astype(np.float32), vector.sum(), out=out)
            return out

    class _FastMixed:
        isa = "avx2"
        backend = "mixed_test"
        fallback_reason = None

        def backend_for(self, quant_name):
            return f"{str(quant_name).lower()}_test"

        def gemv(self, rows, input_dim, vector, *, quant_name, out):
            del input_dim, quant_name
            np.multiply(rows[:, 0].astype(np.float32), vector.sum(), out=out)
            return out

    monkeypatch.setattr(q4_k, "select_q4_k_primitive", lambda mode="auto": _FastQ4())
    monkeypatch.setattr(q4_k, "select_mixed_gemv_primitive", lambda mode="auto": _FastMixed())
    bundle = QwenGGUFCpuExpertBundle.from_host(
        _geometry_host(promoted=False),
        top_k=10,
        mode="avx2",
        num_threads=2,
        required_alignment=1,
    )
    assert bundle.effective_num_threads == 2
    output = bundle.decode(
        0,
        torch.ones((1, 2560), dtype=torch.float32),
        torch.ones((1, 2), dtype=torch.float32),
        torch.tensor([[0, 1]], dtype=torch.int32),
    )
    assert tuple(output.shape) == (1, 2560)
    assert bundle.last_telemetry is not None
    assert bundle.last_telemetry.thread_count == 2
    assert bundle.actual_thread_count == 2
    telemetry = bundle.host_weight_telemetry()
    assert telemetry["requested_num_threads"] == 2
    assert telemetry["effective_num_threads"] == 2
    assert telemetry["actual_thread_count"] == 2
    assert telemetry["kernel_census"] == ("q4_k_test", "q5_1_test")
    assert telemetry["execution_telemetry"]["thread_count"] == 2
    assert telemetry["execution_telemetry"]["fallback_reason"] is None
    bundle.close()


def test_bundle_reports_serial_fallback_when_threading_is_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    topology = CpuTopology(
        allowed_cpus=(40, 42, 44, 46),
        cores=tuple(
            PhysicalCore(
                key=f"core-{cpu}",
                representative=cpu,
                logical_cpus=(cpu,),
                siblings=(cpu,),
            )
            for cpu in (40, 42, 44, 46)
        ),
        confidence="full",
        source="synthetic",
    )
    monkeypatch.setattr("freetoken.moe.gguf_cpu.discover_cpu_topology", lambda: topology)
    bundle = QwenGGUFCpuExpertBundle.from_host(
        _host(), top_k=1, mode="scalar", num_threads=4, required_alignment=1
    )
    bundle.decode(
        0,
        torch.ones((1, 256), dtype=torch.float32),
        torch.ones((1, 1), dtype=torch.float32),
        torch.tensor([[0]], dtype=torch.int32),
    )
    assert bundle.last_telemetry is not None
    assert bundle.last_telemetry.thread_count == 1
    assert bundle.actual_thread_count == 1
    telemetry = bundle.host_weight_telemetry()
    assert telemetry["effective_num_threads"] == 4
    assert telemetry["actual_thread_count"] == 1
    assert telemetry["threading_fallback_reason"] == "native_avx2_or_layout_ineligible"
    bundle.close()


def test_bundle_preserves_mixed_layout_and_kernel_census() -> None:
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    bundle = QwenGGUFCpuExpertBundle.from_host(_host(), top_k=1, mode="scalar")
    assert [item.quant_name for item in bundle.layout.descriptors] == ["Q4_K", "Q4_K", "Q5_1"]
    assert bundle.kernel_census == ("q4_k_scalar", "reference_q5_1")
    bundle.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"device": "cuda"}, "CPU"),
        ({"backend": "hybrid"}, "CPU-only"),
        ({"backend": "offload"}, "CPU-only"),
        ({"cache_size": 1}, "cache_size=0"),
        ({"grouped": True}, "group"),
    ],
)
def test_bundle_rejects_unsupported_runtime_modes(kwargs, message: str) -> None:
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    with pytest.raises((ValueError, RuntimeError), match=message):
        QwenGGUFCpuExpertBundle.from_host(_host(), top_k=1, mode="scalar", **kwargs)


def test_bundle_rejects_closed_host_before_building_executor() -> None:
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    host = _host()
    host.close()
    with pytest.raises(RuntimeError, match="closed"):
        QwenGGUFCpuExpertBundle.from_host(host, top_k=1, mode="scalar")


def test_bundle_accepts_single_request_prefill_mode() -> None:
    torch = pytest.importorskip("torch")
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    bundle = QwenGGUFCpuExpertBundle.from_host(
        _host(), top_k=1, mode="scalar", prefill=True, max_tokens=2
    )
    output = bundle.prefill(
        0,
        torch.ones((2, 256), dtype=torch.float32),
        torch.ones((2, 1), dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.int32),
    )
    assert output.shape == (2, 256)
    bundle.close()


def test_bundle_reuses_configured_workspace_and_rejects_over_bound_before_growth() -> None:
    torch = pytest.importorskip("torch")
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle, UnsupportedGGUFCpuConfiguration

    bundle = QwenGGUFCpuExpertBundle.from_host(
        _host(), top_k=1, mode="scalar", max_tokens=2, max_routes=1
    )
    try:
        plan = bundle.workspace_plan
        output = bundle.prefill(
            0,
            torch.ones((2, 256), dtype=torch.float32),
            torch.ones((2, 1), dtype=torch.float32),
            torch.zeros((2, 1), dtype=torch.int32),
        )
        assert output.shape == (2, 256)
        assert bundle.workspace_plan is plan
        with pytest.raises(UnsupportedGGUFCpuConfiguration, match="workspace bound 2"):
            bundle.prefill(
                0,
                torch.ones((3, 256), dtype=torch.float32),
                torch.ones((3, 1), dtype=torch.float32),
                torch.zeros((3, 1), dtype=torch.int32),
            )
        assert bundle.workspace_plan is plan
        with pytest.raises(UnsupportedGGUFCpuConfiguration, match="workspace bound 1"):
            bundle.prepare(max_tokens=2, max_routes=2)
        assert bundle.workspace_plan is plan
    finally:
        bundle.close()


def test_bundle_does_not_replace_explicit_zero_route_workspace() -> None:
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    bundle = QwenGGUFCpuExpertBundle.from_host(_host(), top_k=1, mode="scalar", max_routes=0)
    assert bundle.workspace_plan.max_routes == 0
    bundle.close()


@pytest.mark.parametrize("promoted", [False, True])
def test_bridge_preserves_nonsquare_qwen_geometry_and_real_packed_strides(
    promoted: bool,
) -> None:
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    bundle = QwenGGUFCpuExpertBundle.from_host(
        _geometry_host(promoted=promoted),
        top_k=10,
        mode="scalar",
        required_alignment=1,
    )
    expected_quants = ("Q5_K", "Q5_K", "Q8_0") if promoted else ("Q4_K", "Q4_K", "Q5_1")
    expected_strides = (1760, 1760, 680) if promoted else (1440, 1440, 480)
    assert bundle.layout.top_k == 10
    assert [item.quant_name for item in bundle.layout.descriptors] == list(expected_quants)
    assert [item.input_dim for item in bundle.layout.descriptors] == [2560, 2560, 640]
    assert [item.output_dim for item in bundle.layout.descriptors] == [640, 640, 2560]
    assert [item.row_stride_bytes for item in bundle.layout.descriptors] == list(expected_strides)
    bundle.close()


@pytest.mark.parametrize(
    ("layer", "gate_up", "down", "expected_strides", "expected_census"),
    [
        (
            0,
            (IQ3_XXS, "IQ3_XXS"),
            (IQ4_NL, "IQ4_NL"),
            (980, 980, 360),
            ("reference_iq3_xxs", "reference_iq4_nl"),
        ),
        (
            2,
            (IQ4_XS, "IQ4_XS"),
            (Q8_0, "Q8_0"),
            (1360, 1360, 680),
            ("reference_iq4_xs", "reference_q8_0"),
        ),
        (
            4,
            (IQ3_XXS, "IQ3_XXS"),
            (Q8_0, "Q8_0"),
            (980, 980, 680),
            ("reference_iq3_xxs", "reference_q8_0"),
        ),
        (
            30,
            (IQ3_XXS, "IQ3_XXS"),
            (Q8_0, "Q8_0"),
            (980, 980, 680),
            ("reference_iq3_xxs", "reference_q8_0"),
        ),
        (
            46,
            (IQ3_XXS, "IQ3_XXS"),
            (Q8_0, "Q8_0"),
            (980, 980, 680),
            ("reference_iq3_xxs", "reference_q8_0"),
        ),
        (
            47,
            (IQ3_XXS, "IQ3_XXS"),
            (Q8_0, "Q8_0"),
            (980, 980, 680),
            ("reference_iq3_xxs", "reference_q8_0"),
        ),
    ],
)
def test_bridge_accepts_one_expert_qwen38_q3_census_geometries(
    layer: int,
    gate_up: tuple[int, str],
    down: tuple[int, str],
    expected_strides: tuple[int, int, int],
    expected_census: tuple[str, str],
) -> None:
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    host = _qwen_census_geometry_host(layer=layer, gate_up=gate_up, down=down)
    bundle = QwenGGUFCpuExpertBundle.from_host(
        host,
        top_k=1,
        mode="scalar",
        required_alignment=1,
    )
    try:
        assert [item.num_experts for item in bundle.layout.descriptors] == [1, 1, 1]
        assert [item.input_dim for item in bundle.layout.descriptors] == [2560, 2560, 640]
        assert [item.output_dim for item in bundle.layout.descriptors] == [640, 640, 2560]
        assert [item.row_stride_bytes for item in bundle.layout.descriptors] == list(
            expected_strides
        )
        assert bundle.kernel_census_for_layer(layer) == expected_census
    finally:
        bundle.close()


def test_bridge_rejects_unknown_quant_name_instead_of_claiming_reference_support() -> None:
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle, UnsupportedGGUFCpuConfiguration

    descriptors = []
    banks = {}
    for projection in ("gate", "up", "down"):
        descriptor, values = _descriptor(
            0,
            projection,
            999,
            "MYSTERY",
            input_dim=256,
            output_dim=256,
            experts=1,
        )
        descriptors.append(descriptor)
        banks[(0, projection)] = _Bank(descriptor, values)
    expert_layout = GGUFExpertLayout(tuple(descriptors), (), 1, 1)
    host_layout = QwenHostLayout(
        expert_layout, _Ple(), sum(item.tensor_bytes for item in descriptors), ("synthetic.gguf",)
    )
    host = QwenGGUFHostWeights(host_layout, _Banks(expert_layout, banks), _Ple())  # type: ignore[arg-type]
    with pytest.raises(UnsupportedGGUFCpuConfiguration, match="MYSTERY"):
        QwenGGUFCpuExpertBundle.from_host(host, top_k=1, mode="scalar")
    host.close()


@pytest.mark.parametrize(
    ("quant_type", "quant_name"),
    [
        (999, "Q8_0"),
        (8, "MYSTERY"),
        (7, "Q8_0"),
        (13, "Q5_1"),
        (7, "Q5_K"),
        (13, "Q8_0"),
    ],
)
def test_bridge_rejects_supported_name_with_noncanonical_quant_type(
    quant_type: int, quant_name: str
) -> None:
    from freetoken.moe.gguf_cpu import (
        QwenGGUFCpuExpertBundle,
        UnsupportedGGUFCpuConfiguration,
    )

    descriptors = []
    banks = {}
    for projection in ("gate", "up", "down"):
        descriptor, values = _descriptor(
            0,
            projection,
            quant_type,
            quant_name,
            input_dim=256,
            output_dim=256,
            experts=1,
        )
        descriptors.append(descriptor)
        banks[(0, projection)] = _Bank(descriptor, values)
    expert_layout = GGUFExpertLayout(tuple(descriptors), (), 1, 1)
    host_layout = QwenHostLayout(
        expert_layout,
        _Ple(),
        sum(item.tensor_bytes for item in descriptors),
        ("synthetic.gguf",),
    )
    host = QwenGGUFHostWeights(host_layout, _Banks(expert_layout, banks), _Ple())  # type: ignore[arg-type]
    with pytest.raises(UnsupportedGGUFCpuConfiguration, match=quant_name):
        QwenGGUFCpuExpertBundle.from_host(host, top_k=1, mode="scalar")
    host.close()


def test_failed_bundle_constructor_closes_executor_without_claiming_or_closing_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import freetoken.moe.gguf_cpu as bridge

    host = _host()
    created = []

    class _FailingExecutor:
        def __init__(self, *_args, **_kwargs) -> None:
            self.closed = False
            created.append(self)

        def prepare(self, **_kwargs):
            raise RuntimeError("prepare failed")

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(bridge, "Q4KExecutor", _FailingExecutor)
    with pytest.raises(RuntimeError, match="prepare failed"):
        bridge.QwenGGUFCpuExpertBundle.from_host(host, top_k=1, mode="scalar")
    assert created and created[0].closed
    assert not host.closed


def test_host_can_be_claimed_by_only_one_bundle_and_claim_is_permanent() -> None:
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    host = _host()
    bundle = QwenGGUFCpuExpertBundle.from_host(host, top_k=1, mode="scalar")
    with pytest.raises(RuntimeError, match="already claimed"):
        QwenGGUFCpuExpertBundle.from_host(host, top_k=1, mode="scalar")
    bundle.close()
    with pytest.raises(RuntimeError, match="already claimed"):
        QwenGGUFCpuExpertBundle.from_host(host, top_k=1, mode="scalar")


def test_retained_host_cannot_close_a_live_bundle() -> None:
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    host = _host()
    bundle = QwenGGUFCpuExpertBundle.from_host(host, top_k=1, mode="scalar")
    with pytest.raises(RuntimeError, match="owned"):
        host.close()
    assert not host.closed
    bundle.close()
    assert host.closed


def test_host_close_attempts_all_resources_and_retries_after_failure() -> None:
    host = _host()
    calls: list[str] = []

    class _FailOnce:
        def __init__(self, name: str) -> None:
            self.name = name
            self.failed = False

        def close(self) -> None:
            calls.append(self.name)
            if not self.failed:
                self.failed = True
                raise RuntimeError(f"{self.name} close failed")

    host.ple = _FailOnce("ple")  # type: ignore[assignment]
    host.experts = _FailOnce("experts")  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="ple close failed"):
        host.close()
    assert calls == ["ple", "experts"]
    assert not host.closed
    host.close()
    assert calls == ["ple", "experts", "ple", "experts"]
    assert host.closed
    host.close()
    assert calls == ["ple", "experts", "ple", "experts"]


def test_mapped_expert_banks_close_attempts_all_banks_and_retries() -> None:
    from freetoken.gguf_host import MappedExpertBanks

    calls: list[str] = []

    class _FailOnce:
        def __init__(self, name: str) -> None:
            self.name = name
            self.failed = False

        def close(self) -> None:
            calls.append(self.name)
            if not self.failed:
                self.failed = True
                raise RuntimeError(f"{self.name} close failed")

    banks = object.__new__(MappedExpertBanks)
    banks._closed = False
    banks._banks = {("first", "gate"): _FailOnce("first"), ("second", "up"): _FailOnce("second")}
    with pytest.raises(RuntimeError, match="first close failed"):
        banks.close()
    assert calls == ["first", "second"]
    assert not banks._closed
    banks.close()
    assert calls == ["first", "second", "first", "second"]
    assert banks._closed


@pytest.mark.parametrize("promoted", [False, True])
def test_bridge_executes_one_nonsquare_packed_route_with_selected_census(
    promoted: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from freetoken.moe import q4_k
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    class _FastQ4:
        isa = "avx2"
        backend = "q4_k_test"
        fallback_reason = None

        def gemv(self, rows, input_dim, vector, *, out, scratch=None):
            del input_dim, scratch
            np.multiply(rows[:, 0].astype(np.float32), vector.sum(), out=out)
            return out

    class _FastMixed:
        isa = "avx2"
        backend = "mixed_test"
        fallback_reason = None

        def backend_for(self, quant_name):
            return f"{str(quant_name).lower()}_test"

        def gemv(self, rows, input_dim, vector, *, quant_name, out):
            del input_dim, quant_name
            np.multiply(rows[:, 0].astype(np.float32), vector.sum(), out=out)
            return out

    monkeypatch.setattr(q4_k, "select_q4_k_primitive", lambda mode="auto": _FastQ4())
    monkeypatch.setattr(q4_k, "select_mixed_gemv_primitive", lambda mode="auto": _FastMixed())
    bundle = QwenGGUFCpuExpertBundle.from_host(
        _geometry_host(promoted=promoted),
        top_k=10,
        mode="avx2",
        required_alignment=1,
    )
    hidden = np.ones((1, 2560), dtype=np.float32)
    expert_ids = np.zeros((1, 1), dtype=np.int32)
    routing_weights = np.ones((1, 1), dtype=np.float32)
    result = bundle.executor.execute(0, hidden, expert_ids, routing_weights)
    assert result.output.shape == (1, 2560)
    expected_census = ("q5_k_test", "q8_0_test") if promoted else ("q4_k_test", "q5_1_test")
    assert bundle.kernel_census_for_layer(0) == expected_census
    assert result.telemetry.kernel_census == expected_census
    bundle.close()


def test_config_registration_guard_is_fail_closed() -> None:
    from freetoken.moe.gguf_cpu import qwen_gguf_cpu_bridge_supported

    class Config:
        moe_backend = "cpu"
        moe_cache_size = 0
        device = "cpu"

    assert qwen_gguf_cpu_bridge_supported(Config())
    assert qwen_gguf_cpu_bridge_supported(Config(), num_threads=0)
    Config.moe_backend = "hybrid"
    assert not qwen_gguf_cpu_bridge_supported(Config())
    Config.moe_backend = "cpu"
    Config.moe_cache_size = 1
    assert not qwen_gguf_cpu_bridge_supported(Config())

    del Config.device
    Config.moe_cache_size = 0
    assert not qwen_gguf_cpu_bridge_supported(Config())

    Config.device = None
    assert not qwen_gguf_cpu_bridge_supported(Config())
    assert not qwen_gguf_cpu_bridge_supported(device=None)
    assert not qwen_gguf_cpu_bridge_supported(backend=None)


def test_config_registration_forwards_bridge_thread_policy() -> None:
    from freetoken.moe.gguf_cpu import register_qwen_gguf_cpu_expert_bundle

    class Config:
        moe_backend = "cpu"
        moe_cache_size = 0
        device = "cpu"

    bundle = register_qwen_gguf_cpu_expert_bundle(
        Config(), host=_host(), top_k=1, mode="scalar", num_threads=0
    )
    assert bundle.requested_num_threads == 0
    assert bundle.effective_num_threads == 1
    bundle.close()


def test_moe_cpu_threads_cli_semantics_remain_zero_auto_and_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from freetoken.server.args import parse_args

    class _Config:
        architectures: ClassVar[list[str]] = ["LlamaForCausalLM"]

        def to_dict(self) -> dict[str, str | list[str]]:
            return {"architectures": self.architectures, "torch_dtype": "float32"}

    monkeypatch.setattr("freetoken.utils.cached_load_hf_config", lambda _path: _Config())
    default, _ = parse_args(["--model", "/models/anon"])
    explicit, _ = parse_args(["--model", "/models/anon", "--moe-cpu-threads", "3"])
    assert default.moe_cpu_threads == 0
    assert explicit.moe_cpu_threads == 3


def test_cpu_tensor_decode_adapter_returns_cpu_tensor() -> None:
    torch = pytest.importorskip("torch")
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    bundle = QwenGGUFCpuExpertBundle.from_host(_host(), top_k=1, mode="scalar")
    hidden = torch.full((1, 256), 0.01, dtype=torch.float32)
    ids = torch.tensor([[1]], dtype=torch.int32)
    weights = torch.tensor([[1.0]], dtype=torch.float32)
    output = bundle.decode(0, hidden, weights, ids)
    assert output.device.type == "cpu"
    assert output.dtype == hidden.dtype
    assert tuple(output.shape) == (1, 256)
    bundle.close()


@pytest.mark.parametrize("ple_backend", ["mmap", "pread"])
def test_bundle_owns_dedicated_ple_artifact_for_each_io_backend(
    tmp_path: Path, ple_backend: str
) -> None:
    from freetoken.gguf_host import convert_gguf_ple_to_artifact
    from freetoken.moe.gguf_cpu import open_qwen_gguf_cpu_expert_bundle

    source = Path(__file__).resolve().parents[1] / "fixtures" / "gguf" / "qwen4-tiny-experts.gguf"
    artifact = tmp_path / "ple-artifact"
    convert_gguf_ple_to_artifact(source, artifact)
    bundle = open_qwen_gguf_cpu_expert_bundle(
        source,
        top_k=2,
        mode="scalar",
        max_tokens=1,
        ple_artifact_path=artifact,
        ple_backend=ple_backend,
    )
    try:
        assert bundle.host.ple.source_kind == "dedicated-artifact"
        assert bundle.host.ple.backend == ple_backend
        assert bundle.host.ple.descriptor.rows == bundle.host.layout.ple.rows
        with pytest.raises(RuntimeError, match="owned by a CPU expert bundle"):
            bundle.host.close()
    finally:
        bundle.close()
    assert bundle.closed
    assert bundle.host.closed


def test_tensor_decode_rejects_non_cpu_inputs() -> None:
    torch = pytest.importorskip("torch")
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

    bundle = QwenGGUFCpuExpertBundle.from_host(_host(), top_k=1, mode="scalar")
    if not torch.cuda.is_available():
        bundle.close()
        pytest.skip("CUDA unavailable for non-CPU tensor failure path")
    hidden = torch.zeros((1, 256), device="cuda")
    ids = torch.zeros((1, 1), dtype=torch.int32)
    weights = torch.ones((1, 1))
    with pytest.raises(ValueError, match="CPU"):
        bundle.decode(0, hidden, weights, ids)
    bundle.close()


def test_engine_guard_rejects_gguf_before_homogeneous_cache_setup() -> None:
    pytest.importorskip("torch")
    from freetoken.engine.engine import _guard_qwen_gguf_engine_setup

    class ModelConfig:
        model_type = "qwen4_exp"
        expert_quant = "gguf"
        moe_weight_format = "gguf"

    class Config:
        model_config = ModelConfig()
        moe_backend = "offload"

    with pytest.raises(NotImplementedError, match="homogeneous OffloadMoeCache"):
        _guard_qwen_gguf_engine_setup(Config())
