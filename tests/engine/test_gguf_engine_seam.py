"""H0 tests for the explicit Qwen GGUF CPU Engine registration seam."""

from __future__ import annotations

import json
from pathlib import Path
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
        "use_dummy_weight": False,
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


def test_engine_cleanup_retries_model_close_after_fail_once():
    from freetoken.engine.engine import Engine

    calls = []

    class Model(_Model):
        def close_host_resources(self):
            calls.append("close")
            if len(calls) == 1:
                raise RuntimeError("transient close failure")

    engine = Engine.__new__(Engine)
    engine.model = Model()
    engine.moe_offload_cache = None
    engine.cpu_moe_executor = None
    engine._expert_banks = None
    engine._model_host_resources_closed = False

    engine._cleanup_host_bank_resources()
    assert calls == ["close"]
    assert engine._model_host_resources_closed is False

    engine._cleanup_host_bank_resources()
    assert calls == ["close", "close"]
    assert engine._model_host_resources_closed is True


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

    monkeypatch.setattr("freetoken.moe.gguf_cpu.open_qwen_gguf_cpu_expert_bundle", open_bundle)

    class StartupModel(_Model):
        def attach_gguf_cpu_host_resources(self, value):
            events.append(("ple", value))
            return 37

        def attach_gguf_cpu_eager_bridge(self, value, *, transfer=None):
            del transfer
            events.append(("experts", value))
            self.attached.append(value)

    config = _config(max_forward_len=8, max_extend_tokens=3)
    model = StartupModel()
    composed, host_bytes = _initialize_qwen_gguf_cpu_composition(model, config)

    assert composed is bundle
    assert host_bytes == 37
    assert [event[0] for event in events] == ["open", "ple", "experts"]
    assert events[0][2]["cache_size"] == 0
    assert events[0][2]["max_tokens"] == 3
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


def test_qwen_cache_zero_composition_exposes_bundle_when_cleanup_close_fails(monkeypatch):
    from freetoken.engine.engine import _initialize_qwen_gguf_cpu_composition

    class CloseFailBundle(_Bundle):
        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("bundle close is temporarily unavailable")

    bundle = CloseFailBundle()
    monkeypatch.setattr(
        "freetoken.moe.gguf_cpu.open_qwen_gguf_cpu_expert_bundle",
        lambda *_args, **_kwargs: bundle,
    )

    class FailingModel(_Model):
        def attach_gguf_cpu_eager_bridge(self, value, *, transfer=None):
            del value, transfer
            raise RuntimeError("adapter construction failed")

        def close_host_resources(self):
            return None

    with pytest.raises(RuntimeError, match="adapter construction") as raised:
        _initialize_qwen_gguf_cpu_composition(FailingModel(), _config())

    assert raised.value.bundle is bundle
    assert bundle.close_calls == 1
    bundle.close()
    assert bundle.close_calls == 2


def test_engine_startup_adopts_bundle_when_composition_cleanup_close_fails(monkeypatch):
    from freetoken.engine.engine import Engine, _initialize_qwen_gguf_cpu_composition

    class CloseFailBundle(_Bundle):
        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("bundle close is temporarily unavailable")

    bundle = CloseFailBundle()
    monkeypatch.setattr(
        "freetoken.moe.gguf_cpu.open_qwen_gguf_cpu_expert_bundle",
        lambda *_args, **_kwargs: bundle,
    )
    captured = {}

    def initialize(self, config):
        self.config = config
        self.model = _Model()
        self.moe_offload_cache = None
        self.cpu_moe_executor = None
        self._expert_banks = None
        captured["engine"] = self
        _initialize_qwen_gguf_cpu_composition(self.model, config)

    monkeypatch.setattr(Engine, "_initialize", initialize)
    monkeypatch.setattr("freetoken.engine.engine._preflight_qwen_gguf_ple_artifact", lambda _: None)
    with pytest.raises(RuntimeError, match=r"adapter|Qwen GGUF"):
        Engine(_config())

    engine = captured["engine"]
    assert bundle.close_calls == 2
    assert engine._gguf_cpu_expert_bundle is None
    assert engine._gguf_cpu_expert_bundle_owned is False


def test_qwen_cache_zero_startup_requires_dedicated_ple_artifact():
    from freetoken.engine.engine import _initialize_qwen_gguf_cpu_composition

    with pytest.raises(ValueError, match="dedicated PLE"):
        _initialize_qwen_gguf_cpu_composition(_Model(), _config(ple_artifact_path=None))


@pytest.mark.parametrize("mode", ["full-model-warm", "unknown"])
def test_qwen_cache_zero_preflight_rejects_non_artifact_warm_mode(mode):
    from freetoken.engine.engine import _preflight_qwen_gguf_cpu_engine_config

    with pytest.raises(ValueError, match="dedicated PLE warm mode"):
        _preflight_qwen_gguf_cpu_engine_config(
            _config(ple_warm_mode=mode), require_ple_artifact=True
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tp_info": SimpleNamespace(size=2)}, "TP=1"),
        ({"max_running_req": 2}, "one decode request"),
        ({"moe_cache_size": 1}, "cache_size=0"),
        ({"moe_cache_auto": True}, "cache_size=0"),
        ({"cuda_graph_max_bs": 1}, "CUDA graphs"),
        ({"ple_artifact_path": ""}, "dedicated PLE"),
    ],
)
def test_engine_qwen_preflight_rejects_before_initialize(monkeypatch, changes, message):
    from freetoken.engine.engine import Engine

    calls = []

    def unexpected_initialize(self, _config):
        calls.append("initialize")
        raise AssertionError("invalid Qwen configuration reached Engine initialization")

    monkeypatch.setattr(Engine, "_initialize", unexpected_initialize)
    with pytest.raises(ValueError, match=message):
        Engine(_config(**changes))
    assert calls == []


def test_engine_rejects_invalid_ple_artifact_before_initialize(tmp_path, monkeypatch):
    from freetoken.engine.engine import Engine
    from freetoken.gguf_host import convert_gguf_ple_to_artifact

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "gguf" / "qwen-host-layout.gguf"
    artifact = convert_gguf_ple_to_artifact(fixture, tmp_path / "ple")
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = _config(model_path=str(fixture), ple_artifact_path=str(artifact))
    calls = []

    def unexpected_initialize(self, _config):
        calls.append("initialize")
        raise AssertionError("invalid PLE artifact reached Engine initialization")

    monkeypatch.setattr(Engine, "_initialize", unexpected_initialize)
    with pytest.raises(ValueError, match=r"dedicated PLE artifact preflight.*sha256"):
        Engine(config)
    assert calls == []


def test_engine_load_weight_state_dict_excludes_routed_experts_and_keeps_strict_dense_load(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from freetoken.engine.engine import Engine

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "gguf" / "qwen4-tiny-experts.gguf"
    model = torch.nn.Linear(2, 2, bias=True)
    engine = Engine.__new__(Engine)
    engine.model = model
    engine.device = torch.device("cpu")
    config = _config(model_path=str(fixture))
    calls = []

    def fake_load(path, device, *, include_moe_experts):
        calls.append((path, device, include_moe_experts))
        assert include_moe_experts is False
        return [("weight", torch.ones_like(model.weight))]

    monkeypatch.setattr("freetoken.engine.engine.load_weight", fake_load)
    loaded = engine._load_weight_state_dict(config)
    assert set(loaded) == {"weight"}
    assert not any(".experts." in key for key in loaded)
    with pytest.raises(RuntimeError, match=r"Missing key\(s\).*bias"):
        model.load_state_dict(loaded)
    assert calls == [(str(fixture), torch.device("cpu"), False)]

    def fake_load_with_unexpected(path, device, *, include_moe_experts):
        assert include_moe_experts is False
        return [
            ("weight", torch.ones_like(model.weight)),
            ("bias", torch.ones_like(model.bias)),
            ("unexpected", torch.ones(1)),
        ]

    monkeypatch.setattr("freetoken.engine.engine.load_weight", fake_load_with_unexpected)
    with pytest.raises(RuntimeError, match=r"Unexpected key\(s\).*unexpected"):
        model.load_state_dict(engine._load_weight_state_dict(config))


def test_engine_startup_rollback_retries_owned_bundle_after_late_failure(monkeypatch):
    from freetoken.engine.engine import Engine

    bundle = _Bundle()
    close_attempts = []

    def close_once_then_succeed():
        close_attempts.append("close")
        if len(close_attempts) == 1:
            raise RuntimeError("transient bundle close failure")

    bundle.close = close_once_then_succeed
    captured = {}

    class StartupModel(_Model):
        def close_host_resources(self):
            captured.setdefault("model_close", 0)
            captured["model_close"] += 1

    def initialize(self, config):
        self.config = config
        self.model = StartupModel()
        self.moe_offload_cache = None
        self.cpu_moe_executor = None
        self._expert_banks = None
        self.attach_qwen_gguf_cpu_expert_bundle(bundle)
        self._gguf_cpu_expert_bundle_owned = True
        captured["engine"] = self
        raise RuntimeError("late startup failure")

    monkeypatch.setattr(Engine, "_initialize", initialize)
    # This test uses a synthetic config and exercises late rollback, not artifact parsing.
    monkeypatch.setattr("freetoken.engine.engine._preflight_qwen_gguf_ple_artifact", lambda _: None)
    with pytest.raises(RuntimeError, match="late startup failure"):
        Engine(_config())

    engine = captured["engine"]
    assert close_attempts == ["close"]
    assert engine._gguf_cpu_expert_bundle is bundle
    assert engine._gguf_cpu_expert_bundle_owned is True
    assert engine._model_host_resources_closed is True

    engine._cleanup_host_bank_resources()
    assert close_attempts == ["close", "close"]
    assert engine._gguf_cpu_expert_bundle is None
    assert engine._gguf_cpu_expert_bundle_owned is False


def test_engine_qwen_cache_zero_never_enters_homogeneous_cache_constructor():
    from freetoken.engine.engine import _should_initialize_offload_moe_cache

    assert _should_initialize_offload_moe_cache(_config(moe_backend="offload")) is False
    dense_config = _config(
        moe_backend="offload",
        model_config=SimpleNamespace(model_type="llama", expert_quant="none"),
    )
    assert _should_initialize_offload_moe_cache(dense_config) is True


def test_engine_attachment_accepts_single_request_prefill():
    from freetoken.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.model = _Model()
    engine.moe_offload_cache = None
    engine.config = _config(prefill=True)
    bundle = _Bundle()

    engine.attach_qwen_gguf_cpu_expert_bundle(bundle)

    assert engine._gguf_cpu_expert_bundle is bundle
