from __future__ import annotations

import functools
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@contextmanager
def _nvtx_range(name: str):
    """Annotate CUDA work when the runtime provides NVTX, otherwise do nothing.

    CPU/reference execution is a supported validation mode, including with a
    CPU-only Torch build where importing ``torch.cuda.nvtx`` succeeds but
    entering a range raises.  Instrumentation must not turn that mode into a
    model failure.
    """
    import torch

    if torch.cuda.is_available():
        annotation = None
        try:
            import torch.cuda.nvtx as nvtx

            annotation = nvtx.range(name)
            annotation.__enter__()
        except (ImportError, RuntimeError):
            # NVTX is optional instrumentation; the operation itself remains
            # valid when a wheel omits the NVTX runtime library.  Entering the
            # context is kept outside the body ``try`` so model exceptions are
            # never mistaken for missing instrumentation.
            annotation = None
        if annotation is not None:
            try:
                yield
            except BaseException:
                error = sys.exc_info()
                if not annotation.__exit__(*error):
                    raise
            else:
                annotation.__exit__(None, None, None)
            return
    yield


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with _nvtx_range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
