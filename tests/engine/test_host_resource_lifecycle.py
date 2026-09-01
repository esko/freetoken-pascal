from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _config(**overrides):
    values = {
        "model_path": "/tmp/freetoken-host-resource-lifecycle",
        "use_dummy_weight": False,
        "ple_warm_mode": "full",
        "ple_artifact_path": None,
        "ple_backend": "mmap",
        "ple_planner_mode": "vectorized",
        "ple_planner_direct_threshold": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_engine_prefers_explicit_host_tables_loader_and_counts_once():
    from freetoken.engine.engine import _load_model_host_resources

    calls = []

    class Model:
        def load_host_tables(self, config):
            calls.append(("tables", config.ple_warm_mode))
            return 123

        def load_host_weights(self, *args, **kwargs):
            calls.append(("weights", args, kwargs))
            raise AssertionError("legacy loader must not run when explicit loader exists")

    config = _config()

    host_tables_bytes = _load_model_host_resources(Model(), config)
    assert host_tables_bytes == 123
    assert calls == [("tables", "full")]


def test_engine_uses_legacy_host_weights_loader_when_tables_hook_is_absent():
    from freetoken.engine.engine import _load_model_host_resources

    calls = []

    class Model:
        def load_host_weights(self, model_path, **kwargs):
            calls.append((model_path, kwargs))
            return 456

    config = _config()
    assert _load_model_host_resources(Model(), config) == 456
    assert calls == [
        (
            config.model_path,
            {
                "dummy": False,
                "ple_warm_mode": "full",
                "ple_artifact_path": None,
                "ple_backend": "mmap",
                "ple_planner_mode": "vectorized",
                "ple_planner_direct_threshold": 8,
            },
        )
    ]


def test_engine_loader_accepts_models_without_host_resources():
    from freetoken.engine.engine import _load_model_host_resources

    assert _load_model_host_resources(object(), _config()) == 0


def test_startup_failure_closes_acquired_model_host_resources(monkeypatch):
    from freetoken.engine.engine import Engine

    calls = []

    class Model:
        def load_host_tables(self, _config):
            calls.append("load")
            return 789

        def close_host_resources(self):
            calls.append("close")

    model = Model()

    def initialize(self, config):
        self.model = model
        from freetoken.engine.engine import _load_model_host_resources

        self._host_tables_bytes = _load_model_host_resources(self.model, config)
        raise RuntimeError("failure after PLE acquisition")

    with patch.object(Engine, "_initialize", initialize):
        # This test uses a synthetic config and exercises late rollback, not artifact parsing.
        monkeypatch.setattr(
            "freetoken.engine.engine._preflight_qwen_gguf_ple_artifact", lambda _: None
        )
        with pytest.raises(RuntimeError, match="failure after PLE acquisition"):
            Engine(_config())

    assert calls == ["load", "close"]


def test_shutdown_closes_model_host_resources_once_and_is_repeatable():
    from freetoken.engine.engine import Engine

    calls = []

    class Model:
        def close_host_resources(self):
            calls.append("close")

    engine = Engine.__new__(Engine)
    engine.model = Model()
    engine.graph_runner = SimpleNamespace(destroy_cuda_graphs=lambda: calls.append("graphs"))
    engine.moe_offload_cache = None
    engine.cpu_moe_executor = None
    engine._expert_banks = None

    with (
        patch("freetoken.engine.engine.destroy_distributed"),
        patch("torch.distributed.destroy_process_group"),
    ):
        engine.shutdown()
        engine.shutdown()

    assert calls == ["graphs", "close", "graphs"]


def test_qwen_model_host_resource_close_detaches_tables_and_is_idempotent():
    from freetoken.models.qwen4_exp.model import Qwen4ExpModel

    calls = []

    class Table:
        def close(self):
            calls.append("close")

    table = Table()

    class Embedding:
        def __init__(self, table):
            self._table = table

        def attach_table(self, table):
            self._table = table

    embedding = Embedding(table)
    layer = SimpleNamespace(ple_embedding=embedding)
    model = object.__new__(Qwen4ExpModel)
    model._ple = (layer,)
    model._ple_tables = [table]
    model._gguf_attachment_lock = threading.RLock()

    model.close_host_resources()
    model.close_host_resources()

    assert calls == ["close"]
    assert embedding._table is None


def test_qwen_wrapper_delegates_host_resource_close():
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    calls = []
    wrapper = object.__new__(Qwen4ExpForCausalLM)
    wrapper.model = SimpleNamespace(close_host_resources=lambda: calls.append("close"))

    wrapper.close_host_resources()

    assert calls == ["close"]
