"""CPU-only production bridge for heterogeneous Qwen GGUF expert banks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest
from freetoken.gguf_host import (
    ExpertBankDescriptor,
    GGUFExpertLayout,
    QwenGGUFHostWeights,
    QwenHostLayout,
)

Q4_K = 12
Q5_K = 13
Q5_1 = 7
Q8_0 = 8


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
    block_bytes = {"Q4_K": 144, "Q5_K": 176, "Q5_1": 24, "Q8_0": 34}[quant_name]
    block_elements = 256 if quant_name in {"Q4_K", "Q5_K"} else 32
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

    monkeypatch.setattr(bridge, "_affinity_visible_physical_core_count", lambda: 2)
    with pytest.raises(ValueError, match="affinity-visible physical-core capacity of 2"):
        bridge.QwenGGUFCpuExpertBundle.from_host(_host(), top_k=1, mode="scalar", num_threads=3)


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

    monkeypatch.setattr("freetoken.moe.gguf_cpu._affinity_visible_physical_core_count", lambda: 4)
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
        ({"prefill": True}, "prefill"),
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
