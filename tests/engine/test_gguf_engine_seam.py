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
        "ple_artifact_path": "/ple-artifact",
        "ple_backend": "mmap",
        "ple_planner_mode": "vectorized",
        "ple_planner_direct_threshold": 8,
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


def test_engine_cleanup_closes_startup_owned_bundle_after_detach():
    from freetoken.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.model = _Model()
    bundle = _Bundle()
    engine._gguf_cpu_expert_bundle = bundle
    engine._gguf_cpu_expert_bundle_owned = True
    engine.config = _config()
    engine.moe_offload_cache = None
    engine.cpu_moe_executor = None
    engine._expert_banks = None

    engine._cleanup_host_bank_resources()

    assert engine.model.detached == 1
    assert engine._gguf_cpu_expert_bundle is None
    assert bundle.close_calls == 1


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


def test_qwen_cache_zero_startup_composes_one_bundle_and_ple_owner(monkeypatch):
    from freetoken.engine.engine import _initialize_qwen_gguf_cpu_composition

    events = []
    bundle = _Bundle()

    def open_bundle(path, **kwargs):
        events.append(("open", path, kwargs))
        return bundle

    monkeypatch.setattr(
        "freetoken.moe.gguf_cpu.open_qwen_gguf_cpu_expert_bundle", open_bundle
    )

    class StartupModel(_Model):
        def attach_gguf_cpu_host_resources(self, value):
            events.append(("ple", value))
            return 37

        def attach_gguf_cpu_eager_bridge(self, value, *, transfer=None):
            del transfer
            events.append(("experts", value))
            self.attached.append(value)

    config = _config(max_forward_len=8)
    model = StartupModel()
    composed, host_bytes = _initialize_qwen_gguf_cpu_composition(model, config)

    assert composed is bundle
    assert host_bytes == 37
    assert [event[0] for event in events] == ["open", "ple", "experts"]
    assert events[0][2]["cache_size"] == 0
    assert events[0][2]["max_tokens"] == 1
    assert model.attached == [bundle]
    assert bundle.close_calls == 0


def test_qwen_cache_zero_startup_rolls_back_bundle_when_attachment_fails(monkeypatch):
    from freetoken.engine.engine import _initialize_qwen_gguf_cpu_composition

    bundle = _Bundle()

    monkeypatch.setattr(
        "freetoken.moe.gguf_cpu.open_qwen_gguf_cpu_expert_bundle",
        lambda *_args, **_kwargs: bundle,
    )

    class FailingModel(_Model):
        def attach_gguf_cpu_host_resources(self, value):
            assert value is bundle
            return 0

        def attach_gguf_cpu_eager_bridge(self, value, *, transfer=None):
            del value, transfer
            raise RuntimeError("adapter construction failed")

        def close_host_resources(self):
            return None

    with pytest.raises(RuntimeError, match="adapter construction"):
        _initialize_qwen_gguf_cpu_composition(FailingModel(), _config())
    assert bundle.close_calls == 1


def test_qwen_cache_zero_startup_requires_dedicated_ple_artifact():
    from freetoken.engine.engine import _initialize_qwen_gguf_cpu_composition

    with pytest.raises(ValueError, match="dedicated PLE"):
        _initialize_qwen_gguf_cpu_composition(
            _Model(), _config(ple_artifact_path=None)
        )


def test_engine_attachment_accepts_single_request_prefill():
    from freetoken.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.model = _Model()
    engine.moe_offload_cache = None
    engine.config = _config(prefill=True)
    bundle = _Bundle()

    engine.attach_qwen_gguf_cpu_expert_bundle(bundle)

    assert engine._gguf_cpu_expert_bundle is bundle
