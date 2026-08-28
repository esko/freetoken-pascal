from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .base import BaseMoeBackend


class _Registry:
    """Small dependency-free registry used during H0 package imports.

    The MoE package is also the import parent for ``cpu_abi``.  Keeping this
    registry local prevents a Torch/Hugging Face import merely to load the
    Torch-free ABI; backend construction still imports the full runtime lazily.
    """

    def __init__(self, type_name: str):
        self._registry = {}
        self._info = {}
        self._type = type_name

    def register(self, name: str, info: object | None = None):
        if name in self._registry:
            raise KeyError(f"{self._type} '{name}' is already registered.")

        def decorator(item):
            self._registry[name] = item
            self._info[name] = info
            return item

        return decorator

    def info(self, name: str) -> object | None:
        if name not in self._registry:
            raise KeyError(f"Unsupported {self._type}: {name}")
        return self._info.get(name)

    def __getitem__(self, name: str):
        if name not in self._registry:
            raise KeyError(f"Unsupported {self._type}: {name}")
        return self._registry[name]

    def supported_names(self) -> list[str]:
        return list(self._registry)

    def assert_supported(self, names: str | Iterable[str]) -> None:
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name not in self._registry:
                from argparse import ArgumentTypeError

                raise ArgumentTypeError(
                    f"Unsupported {self._type}: {name}. Supported items: {self.supported_names()}"
                )


class MoeBackendCreator(Protocol):
    def __call__(self) -> BaseMoeBackend: ...


SUPPORTED_MOE_BACKENDS = _Registry("MoE Backend")

# Backends that serve experts from CPU (pinned) host banks through an
# ``OffloadMoeCache`` -- the GPU only holds the two-layer prefill double buffer.
# They differ only in how *decode* gets the experts: ``offload`` streams the
# missing experts over PCIe into a GPU slot cache and runs the GEMM on the GPU;
# ``cpu`` ships the activations to the CPU, computes the experts there (high RAM
# bandwidth), and ships the results back; ``hybrid`` keeps a full GPU slot cache
# AND a CPU executor -- it fetches at most K missing experts/layer over PCIe
# (computed on the GPU with the cache hits) and computes the remaining misses on
# the CPU, overlapped, then merges (capped PCIe + CPU overflow). All build their
# cache the same way, so model layer construction and the engine wiring key off
# this set rather than a bare ``== "offload"`` check.
OFFLOAD_MOE_BACKENDS = frozenset({"offload", "cpu", "hybrid"})


def is_offload_moe_backend(backend: str) -> bool:
    return backend in OFFLOAD_MOE_BACKENDS


@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend():
    from .fused import FusedMoe

    return FusedMoe()


@SUPPORTED_MOE_BACKENDS.register("offload")
def create_offload_moe_backend():
    from .offload import OffloadMoeBackend

    return OffloadMoeBackend()


@SUPPORTED_MOE_BACKENDS.register("cpu")
def create_cpu_moe_backend():
    from .cpu_offload import CpuOffloadMoeBackend

    return CpuOffloadMoeBackend()


@SUPPORTED_MOE_BACKENDS.register("hybrid")
def create_hybrid_moe_backend():
    from .cpu_offload import HybridMoeBackend

    return HybridMoeBackend()


def create_moe_backend(backend: str) -> BaseMoeBackend:
    return SUPPORTED_MOE_BACKENDS[backend]()


def __getattr__(name: str):
    if name == "QwenGGUFCpuMoELayer":
        from .gguf_layer import QwenGGUFCpuMoELayer

        globals()[name] = QwenGGUFCpuMoELayer
        return QwenGGUFCpuMoELayer
    if name == "BaseMoeBackend":
        from .base import BaseMoeBackend

        globals()[name] = BaseMoeBackend
        return BaseMoeBackend
    if name == "Registry":
        from freetoken.utils import Registry

        globals()[name] = Registry
        return Registry
    if name == "logger":
        from freetoken.utils import init_logger

        value = init_logger(__name__)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "OFFLOAD_MOE_BACKENDS",
    "SUPPORTED_MOE_BACKENDS",
    "BaseMoeBackend",
    "QwenGGUFCpuMoELayer",
    "create_moe_backend",
    "is_offload_moe_backend",
]
