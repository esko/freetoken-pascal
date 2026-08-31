"""H0 tests for the explicit Qwen GGUF CPU Engine registration seam."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _config(**overrides):
    model_config = SimpleNamespace(
        model_type="qwen4_exp",
        expert_quant="gguf",
        num_experts_per_tok=10,
    )
    values = {
        "model_path": "/models/qwen38.gguf",
        "model_config": model_config,
        "moe_backend": "cpu",
        "moe_cache_size": 0,
        "moe_cache_auto": False,
        "moe_cache_rate": None,
        "moe_cpu_threads": 0,
        "ple_warm_mode": "cold",
        "moe_cpu_layers": None,
        "max_running_req": 1,
        "tp_info": SimpleNamespace(size=1),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Bundle:
    effective_num_threads = 1

    def __init__(self):
        self.close_calls = 0

    def host_weight_telemetry(self):
        return {"source": "synthetic"}

    def close(self):
        self.close_calls += 1


class _Model:
    def __init__(self):
        self.attached = []
        self.detached = 0

    def attach_gguf_cpu_eager_bridge(self, bundle, *, transfer=None):
        del transfer
        self.attached.append(bundle)

    def detach_gguf_cpu_expert_bundle(self):
        self.detached += 1

    def gguf_cpu_expert_telemetry(self):
        return {0: {"source": "synthetic"}}


def test_engine_attaches_borrowed_bundle_without_homogeneous_cache():
    from freetoken.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.model = _Model()
    engine.moe_offload_cache = None
    bundle = _Bundle()
    config = _config()
    engine.config = config
    assert engine.moe_offload_cache is None
    engine.attach_qwen_gguf_cpu_expert_bundle(bundle)
    assert engine._gguf_cpu_expert_bundle is bundle
    assert engine.model.attached == [bundle]
    with pytest.raises(RuntimeError, match="already attached"):
        engine.attach_qwen_gguf_cpu_expert_bundle(_Bundle())
    assert engine.model.attached == [bundle]


def test_engine_detach_is_explicit_and_never_closes_borrowed_bundle():
    from freetoken.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.model = _Model()
    engine.moe_offload_cache = None
    engine.config = _config()
    bundle = _Bundle()

    engine.attach_qwen_gguf_cpu_expert_bundle(bundle)
    engine.detach_qwen_gguf_cpu_expert_bundle()

    assert engine.model.detached == 1
    assert engine._gguf_cpu_expert_bundle is None
    assert bundle.close_calls == 0

    # The public lifecycle seam is idempotent after the Engine-owned wrappers are gone.
    engine.detach_qwen_gguf_cpu_expert_bundle()
    assert engine.model.detached == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"moe_cache_size": 1}, "cache_size=0"),
        ({"moe_cache_auto": True}, "cache_size=0"),
        ({"moe_cache_rate": 0.5}, "cache_size=0"),
        ({"tp_info": SimpleNamespace(size=2)}, "TP=1"),
        ({"max_running_req": 2}, "one decode request"),
        ({"model_config": SimpleNamespace(model_type="llama", expert_quant="gguf")}, "Qwen4"),
    ],
)
def test_engine_rejects_unsupported_gguf_cpu_configuration(changes, message):
    from freetoken.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.model = _Model()
    engine.config = _config(**changes)
    with pytest.raises(ValueError, match=message):
        engine.attach_qwen_gguf_cpu_expert_bundle(_Bundle())


def test_engine_cleanup_detaches_before_closing_owned_bundle():
    from freetoken.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.model = _Model()
    bundle = _Bundle()
    engine._gguf_cpu_expert_bundle = bundle
    engine.config = _config()
    engine.moe_offload_cache = None
    engine.cpu_moe_executor = None
    engine._expert_banks = None

    engine._cleanup_host_bank_resources()

    assert engine.model.detached == 1
    assert engine._gguf_cpu_expert_bundle is None
    assert bundle.close_calls == 0


def test_engine_cleanup_keeps_tracking_when_detach_fails():
    from freetoken.engine.engine import Engine

    class FailingModel(_Model):
        def detach_gguf_cpu_expert_bundle(self):
            raise RuntimeError("bridge is busy")

    engine = Engine.__new__(Engine)
    engine.model = FailingModel()
    bundle = _Bundle()
    engine._gguf_cpu_expert_bundle = bundle
    engine.moe_offload_cache = None
    engine.cpu_moe_executor = None
    engine._expert_banks = None
    engine._model_host_resources_closed = True

    engine._cleanup_host_bank_resources()

    assert engine._gguf_cpu_expert_bundle is bundle
    assert bundle.close_calls == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"prefill": True},
        {"grouped": True},
        {"cuda_graph_bs": [1]},
        {"cuda_graph_max_bs": 1},
    ],
)
def test_engine_rejects_non_decode_or_graph_configuration(changes):
    from freetoken.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.model = _Model()
    engine.config = _config(**changes)
    with pytest.raises(ValueError, match=r"decode-only|grouped|graph"):
        engine.attach_qwen_gguf_cpu_expert_bundle(_Bundle())


def test_engine_telemetry_exposes_attached_gguf_cpu_layers():
    from freetoken.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.model = _Model()
    assert engine.gguf_cpu_expert_telemetry() == {0: {"source": "synthetic"}}
