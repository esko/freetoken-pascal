import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from freetoken.utils.torch_utils import nvtx_annotate


def test_nvtx_annotation_is_optional_on_cpu():
    class Operation:
        @nvtx_annotate("cpu-reference")
        def run(self, value: int) -> int:
            return value + 1

    assert Operation().run(41) == 42


def test_nvtx_annotation_does_not_mask_operation_failure(monkeypatch):
    @contextmanager
    def fake_range(_name):
        yield

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "torch.cuda.nvtx", SimpleNamespace(range=fake_range))

    class Operation:
        @nvtx_annotate("failure-preservation")
        def run(self):
            raise RuntimeError("sentinel operation failure")

    with pytest.raises(RuntimeError, match="sentinel operation failure"):
        Operation().run()
