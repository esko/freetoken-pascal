from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class _Owner:
    def __init__(self, name: str, *, fail_once: bool = False):
        self.name = name
        self.fail_once = fail_once
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_once and self.close_calls == 1:
            raise RuntimeError(f"{self.name} close failed")


class _Embedding:
    def __init__(self, *, fail: bool = False):
        self.table = None
        self.fail = fail

    def attach_table(self, table) -> None:
        if self.fail:
            raise RuntimeError("attachment failed")
        self.table = table


def _model(*layers):
    from freetoken.models.qwen4_exp.model import Qwen4ExpModel

    model = object.__new__(Qwen4ExpModel)
    model._ple = tuple(SimpleNamespace(ple_embedding=layer) for layer in layers)
    model._ple_tables = []
    model._gguf_attachment_lock = threading.RLock()
    model._host_resources_closed = False
    return model


def test_attachment_failure_retains_new_owner_for_model_cleanup():
    model = _model(_Embedding(fail=True))
    owner = _Owner("mapped")

    with pytest.raises(RuntimeError, match="attachment failed"):
        model._attach_ple_table(owner)

    assert model._ple_tables == [owner]
    model._ple[0].ple_embedding.fail = False
    model.close_host_resources()
    assert owner.close_calls == 1


def test_mapped_adapter_construction_failure_closes_open_mapping():
    from freetoken.models.qwen4_exp.model import Qwen4ExpModel

    mapped = _Owner("mapped")
    mapped.descriptor = SimpleNamespace(tensor_bytes=17)
    model = _model(_Embedding())
    model._config = SimpleNamespace(qwen4_args=SimpleNamespace(ngram_head_dim=128))

    with (
        patch("freetoken.gguf_host.MappedPLETable.open_from_artifact", return_value=mapped),
        patch(
            "freetoken.models.qwen4_exp.model._MappedPLETable",
            side_effect=RuntimeError("adapter construction failed"),
        ),
    ):
        with pytest.raises(RuntimeError, match="adapter construction failed"):
            Qwen4ExpModel.load_host_weights(
                model, "/unused", ple_artifact_path="/fake/ple-artifact"
            )

    assert mapped.close_calls == 1
    assert model._ple_tables == []


def test_pinned_adapter_construction_failure_closes_ple_owner():
    from freetoken.models.qwen4_exp.model import Qwen4ExpModel

    owner = _Owner("safetensors")
    owner.bank = SimpleNamespace(tensor=object(), nbytes=23)
    owner.weight_scale = 1.0
    model = _model(_Embedding())
    model._config = SimpleNamespace(qwen4_args=SimpleNamespace())

    with (
        patch("freetoken.models.qwen4_exp.weight.load_ple_table", return_value=owner),
        patch(
            "freetoken.models.qwen4_exp.ple.PinnedUVATable",
            side_effect=RuntimeError("adapter construction failed"),
        ),
    ):
        with pytest.raises(RuntimeError, match="adapter construction failed"):
            Qwen4ExpModel.load_host_weights(model, "/unused")

    assert owner.close_calls == 1
    assert model._ple_tables == []


def test_close_attempts_all_owners_retains_failure_and_retries():
    model = _model(_Embedding())
    first = _Owner("first", fail_once=True)
    second = _Owner("second")
    model._ple_tables = [first, second]

    with pytest.raises(RuntimeError, match="first close failed"):
        model.close_host_resources()

    assert first.close_calls == 1
    assert second.close_calls == 1
    assert model._ple_tables == [first]
    assert not model._host_resources_closed
    assert model._ple[0].ple_embedding.table is None

    model.close_host_resources()
    model.close_host_resources()
    assert first.close_calls == 2
    assert second.close_calls == 1
    assert model._ple_tables == []
    assert model._host_resources_closed
