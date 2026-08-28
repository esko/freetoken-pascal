"""CPU-only bridge from heterogeneous GGUF expert mappings to the CPU ABI.

The regular engine's :class:`~freetoken.moe.offload_cache.OffloadMoeCache` owns a
homogeneous GPU slot layout.  Qwen GGUF expert banks are intentionally heterogeneous,
so this module keeps the mapped source as the source of truth and stops at a decode-only
CPU boundary until the model layer can consume that boundary directly.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from freetoken.gguf_host import QwenGGUFHostWeights, open_qwen_host_weights
from freetoken.moe.cpu_abi import CpuAbiError, CpuExpertLayout, cpu_layout_from_source_layout
from freetoken.moe.q4_k import Q4KExecutor

if TYPE_CHECKING:
    import torch


class GGUFCpuBridgeError(RuntimeError):
    """Base error raised when the CPU-only GGUF bridge cannot be used safely."""


class UnsupportedGGUFCpuConfiguration(ValueError):
    """A requested runtime mode is outside the decode-only bridge contract."""


_SUPPORTED_QUANTS = frozenset({"Q4_K", "Q5_K", "Q5_1", "Q8_0"})
_UNSET = object()


def _device_type(device: Any) -> str:
    if device is None:
        return "cpu"
    value = getattr(device, "type", device)
    return str(value).split(":", 1)[0].lower()


def _validate_cpu_bridge_config(
    config: Any = None,
    *,
    backend: str | object | None = _UNSET,
    device: Any = _UNSET,
    cache_size: int | None = None,
    prefill: bool = False,
    grouped: bool = False,
) -> None:
    """Validate the small runtime surface that can be safely adapted today."""
    if config is not None:
        if backend is _UNSET:
            backend = getattr(
                config,
                "moe_backend",
                getattr(config, "backend", "cpu"),
            )
        if device is _UNSET:
            # EngineConfig deliberately has no device field because the CUDA engine binds
            # one after distributed setup.  Treat that absence as unknown rather than
            # silently declaring a production config CPU-safe.
            if not hasattr(config, "device") or config.device is None:
                raise UnsupportedGGUFCpuConfiguration(
                    "Qwen GGUF CPU bridge registration requires config.device='cpu'"
                )
            device = config.device
        if cache_size is None:
            cache_size = getattr(config, "moe_cache_size", 0)
        prefill = bool(prefill or getattr(config, "prefill", False))
        grouped = bool(grouped or getattr(config, "grouped", False))
        decode_target = getattr(config, "decode_target", "cpu")
        if decode_target != "cpu":
            raise UnsupportedGGUFCpuConfiguration(
                f"Qwen GGUF CPU bridge only supports decode_target='cpu', got {decode_target!r}"
            )
        model_format = getattr(config, "model_format", None)
        if model_format is not None and str(model_format).lower() not in {"gguf", "qwen_gguf"}:
            raise UnsupportedGGUFCpuConfiguration(
                f"Qwen GGUF CPU bridge requires GGUF model_format, got {model_format!r}"
            )

    if backend is _UNSET:
        backend = "cpu"
    if backend is None:
        raise UnsupportedGGUFCpuConfiguration(
            "Qwen GGUF CPU bridge requires an explicit backend; got None"
        )
    if backend != "cpu":
        raise UnsupportedGGUFCpuConfiguration(
            "Qwen GGUF CPU bridge is CPU-only; GPU, hybrid, and offload backends are unsupported"
        )
    if device is _UNSET:
        device = "cpu"
    if device is None:
        raise UnsupportedGGUFCpuConfiguration(
            "Qwen GGUF CPU bridge requires an explicit device; got None"
        )
    if _device_type(device) != "cpu":
        raise UnsupportedGGUFCpuConfiguration(
            f"Qwen GGUF CPU bridge requires a CPU device, got {device!r}"
        )
    if cache_size is None:
        cache_size = 0
    if isinstance(cache_size, bool) or not isinstance(cache_size, (int, np.integer)):
        raise UnsupportedGGUFCpuConfiguration(
            f"Qwen GGUF CPU bridge cache_size must be an integer, got {cache_size!r}"
        )
    if int(cache_size) != 0:
        raise UnsupportedGGUFCpuConfiguration(
            "Qwen GGUF CPU bridge requires cache_size=0; GPU slot allocation is unsupported"
        )
    if prefill:
        raise UnsupportedGGUFCpuConfiguration(
            "Qwen GGUF CPU bridge currently supports decode only; prefill is unsupported"
        )
    if grouped:
        raise UnsupportedGGUFCpuConfiguration(
            "Qwen GGUF CPU bridge currently supports one decode request; "
            "grouped execution is unsupported"
        )


def qwen_gguf_cpu_bridge_supported(config: Any = None, **kwargs: Any) -> bool:
    """Return whether a config can use the standalone CPU-only bridge."""
    try:
        _validate_cpu_bridge_config(config, **kwargs)
    except (TypeError, ValueError):
        return False
    return True


class QwenGGUFCpuExpertBundle:
    """Owned Qwen GGUF host mappings, heterogeneous CPU layout, and Q4 executor.

    The bundle owns ``host`` for its entire lifetime.  Its public runtime operation is
    one decode request at a time; prefill and grouped execution remain explicit errors.
    """

    def __init__(
        self,
        host: QwenGGUFHostWeights,
        layout: CpuExpertLayout,
        executor: Q4KExecutor,
        *,
        output_dtype: Any,
    ) -> None:
        self.host = host
        self.layout = layout
        self.executor = executor
        self.output_dtype = output_dtype
        self._closed = False
        self._host_owner_token: object | None = None

    @classmethod
    def from_host(
        cls,
        host: QwenGGUFHostWeights,
        *,
        top_k: int,
        mode: str = "auto",
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        num_threads: int = 1,
        max_tokens: int = 1,
        max_routes: int | None = None,
        required_alignment: int = 32,
        backend: str = "cpu",
        device: Any = "cpu",
        cache_size: int = 0,
        prefill: bool = False,
        grouped: bool = False,
    ) -> QwenGGUFCpuExpertBundle:
        _validate_cpu_bridge_config(
            backend=backend,
            device=device,
            cache_size=cache_size,
            prefill=prefill,
            grouped=grouped,
        )
        if host is None or not hasattr(host, "layout") or not hasattr(host, "experts"):
            raise TypeError("Qwen GGUF CPU bridge requires a QwenGGUFHostWeights-like host")
        if not callable(getattr(host, "claim_cpu_bridge", None)):
            raise TypeError("Qwen GGUF CPU bridge requires an ownership-capable host")
        if bool(getattr(host, "cpu_bridge_claimed", False)):
            raise RuntimeError("Qwen GGUF host is already claimed by a CPU expert bundle")
        if bool(getattr(host, "closed", getattr(host, "_closed", False))):
            raise RuntimeError("Qwen GGUF host mappings are closed")

        # Keep this call explicit: it is the ownership boundary between the GGUF mapper and
        # the model-independent CPU ABI.  In particular, never normalize these banks into
        # OffloadMoeCache's homogeneous gate_up/down tensors.
        layout = cpu_layout_from_source_layout(
            host.layout.experts,
            host.experts,
            top_k=top_k,
        )
        unsupported = sorted(
            {
                descriptor.quant_name
                for descriptor in layout.descriptors
                if descriptor.quant_name not in _SUPPORTED_QUANTS
            }
        )
        if unsupported:
            message = "Qwen GGUF CPU bridge does not support expert quant types "
            message += ", ".join(unsupported)
            raise UnsupportedGGUFCpuConfiguration(message)
        executor: Q4KExecutor | None = None
        try:
            executor = Q4KExecutor(
                layout,
                mode=mode,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                # Torch conversion is explicit at decode and the ABI always computes in f32.
                output_dtype=np.float32,
                required_alignment=required_alignment,
                num_threads=num_threads,
            )
            executor.prepare(
                max_tokens=max_tokens,
                max_routes=max_routes if max_routes is not None else top_k,
            )
        except BaseException:
            if executor is not None:
                try:
                    executor.close()
                except BaseException:
                    pass
            # The caller still owns a host passed to from_host; do not claim or close it on a
            # failed constructor.  The path factory below transfers ownership only after success.
            raise
        try:
            bundle = cls(host, layout, executor, output_dtype=np.float32)
            bundle._host_owner_token = host.claim_cpu_bridge()
            return bundle
        except BaseException:
            try:
                executor.close()
            except BaseException:
                pass
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def backend(self) -> str:
        self._require_open()
        return self.executor.backend

    @property
    def kernel_census(self) -> tuple[str, ...]:
        """The union of selected per-projection kernels across all mapped layers."""
        self._require_open()
        selected: set[str] = set()
        for layer_id in self.layout.layers:
            selected.update(self.executor._kernel_census(layer_id))
        return tuple(sorted(selected))

    def kernel_census_for_layer(self, layer_id: int) -> tuple[str, ...]:
        self._require_open()
        return self.executor._kernel_census(layer_id)

    def prepare(self, max_tokens: int, max_routes: int) -> Any:
        self._require_open()
        return self.executor.prepare(max_tokens=max_tokens, max_routes=max_routes)

    @property
    def workspace_plan(self) -> Any:
        """The executor's prepared bounded-workspace plan."""
        self._require_open()
        runner = getattr(self.executor, "_threaded_runner", None)
        plan = runner._plan if runner is not None else self.executor._reference._plan
        if plan is None:
            raise RuntimeError("Qwen GGUF CPU bridge workspace is not prepared")
        return plan

    def decode(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        num_token_non_padded: int | None = None,
    ) -> torch.Tensor:
        """Run one CPU decode request through an explicit Torch/NumPy conversion.

        The adapter copies all inputs into contiguous float32/int32 NumPy arrays and copies
        the result back to a CPU tensor in the hidden-state dtype.  CUDA tensors are rejected
        before conversion so a caller cannot accidentally create a hidden synchronization or
        a GPU allocation behind this CPU-only boundary.
        """
        self._require_open()
        torch = _torch()
        tensors = {
            "hidden_states": hidden_states,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
        }
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.device.type != "cpu":
                raise ValueError(f"{name} must be a CPU tensor, got {value.device}")

        allowed_hidden = {torch.float16, torch.float32}
        if hasattr(torch, "bfloat16"):
            allowed_hidden.add(torch.bfloat16)
        if hidden_states.dtype not in allowed_hidden:
            raise ValueError(
                "hidden_states must use float16, bfloat16, or float32 on the CPU bridge; "
                f"got {hidden_states.dtype}"
            )
        if not hidden_states.dtype.is_floating_point:
            raise ValueError(f"hidden_states must be floating point, got {hidden_states.dtype}")
        if not topk_weights.dtype.is_floating_point:
            raise ValueError(f"topk_weights must be floating point, got {topk_weights.dtype}")
        if topk_ids.dtype not in {torch.int32, torch.int64}:
            raise ValueError(f"topk_ids must be int32 or int64, got {topk_ids.dtype}")

        hidden_np = hidden_states.detach().contiguous().to(dtype=torch.float32).numpy()
        weights_np = topk_weights.detach().contiguous().to(dtype=torch.float32).numpy()
        ids_np = topk_ids.detach().contiguous().to(dtype=torch.int32).numpy()
        try:
            result = self.executor.execute(
                layer_id,
                hidden_np,
                ids_np,
                weights_np,
                num_token_non_padded=num_token_non_padded,
            )
        except CpuAbiError:
            raise
        return torch.from_numpy(np.array(result.output, dtype=np.float32, copy=True)).to(
            device="cpu",
            dtype=hidden_states.dtype,
        )

    def prefill(self, *_args: Any, **_kwargs: Any) -> None:
        self._require_open()
        raise UnsupportedGGUFCpuConfiguration(
            "Qwen GGUF CPU bridge currently supports decode only; prefill is unsupported"
        )

    def execute_group(self, *_args: Any, **_kwargs: Any) -> None:
        self._require_open()
        raise UnsupportedGGUFCpuConfiguration(
            "Qwen GGUF CPU bridge currently supports one decode request; "
            "grouped execution is unsupported"
        )

    def memory_report(self) -> dict[str, int]:
        self._require_open()
        return self.host.memory_report()

    def host_weight_telemetry(self) -> dict[str, object]:
        self._require_open()
        return {
            "source": "gguf-mmap",
            "memory": self.host.memory_report(),
            "kernel_census": self.kernel_census,
        }

    def _require_open(self) -> None:
        if self._closed or bool(getattr(self.host, "closed", getattr(self.host, "_closed", False))):
            raise RuntimeError("Qwen GGUF CPU bridge is closed")

    def close(self) -> None:
        if self._closed:
            return
        if self._host_owner_token is None:
            raise RuntimeError("Qwen GGUF CPU bridge has no host ownership token")
        try:
            self.executor.close()
            self.host.close_cpu_bridge(self._host_owner_token)
        except BaseException:
            raise
        self._closed = True

    def __enter__(self) -> QwenGGUFCpuExpertBundle:
        self._require_open()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def open_qwen_gguf_cpu_expert_bundle(
    path: str | Path,
    *,
    top_k: int,
    mode: str = "auto",
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    num_threads: int = 1,
    max_tokens: int = 1,
    max_routes: int | None = None,
    required_alignment: int = 32,
    backend: str = "cpu",
    device: Any = "cpu",
    cache_size: int = 0,
    prefill: bool = False,
    grouped: bool = False,
    supported_expert_types: Collection[int] | None = None,
    ple_warm_mode: str = "cold",
) -> QwenGGUFCpuExpertBundle:
    """Open GGUF mappings and transfer ownership to a CPU expert bundle."""
    _validate_cpu_bridge_config(
        backend=backend,
        device=device,
        cache_size=cache_size,
        prefill=prefill,
        grouped=grouped,
    )
    host = open_qwen_host_weights(
        path,
        supported_expert_types=supported_expert_types,
        ple_warm_mode=ple_warm_mode,
    )
    try:
        return QwenGGUFCpuExpertBundle.from_host(
            host,
            top_k=top_k,
            mode=mode,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            num_threads=num_threads,
            max_tokens=max_tokens,
            max_routes=max_routes,
            required_alignment=required_alignment,
            backend=backend,
            device=device,
            cache_size=cache_size,
            prefill=prefill,
            grouped=grouped,
        )
    except BaseException:
        host.close()
        raise


def register_qwen_gguf_cpu_expert_bundle(
    config: Any,
    *,
    host: QwenGGUFHostWeights | None = None,
    **kwargs: Any,
) -> QwenGGUFCpuExpertBundle:
    """Narrow registration seam for callers that can provide a CPU-only config.

    The regular CUDA engine must not call this until its model MoE layer accepts the bundle;
    registering it today is deliberately explicit and never creates a homogeneous cache.
    """
    _validate_cpu_bridge_config(config, **kwargs)
    path = getattr(config, "model_path", None)
    if host is None:
        if not path:
            raise ValueError("GGUF CPU bridge registration requires config.model_path")
        return open_qwen_gguf_cpu_expert_bundle(path, **kwargs)
    return QwenGGUFCpuExpertBundle.from_host(host, **kwargs)


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - package dependency in production
        raise RuntimeError("Torch is required for the GGUF CPU tensor adapter") from error
    return torch


# Descriptive aliases for callers that prefer CPU before GGUF in the type name.
GGUFCpuExpertBundle = QwenGGUFCpuExpertBundle
QwenGGUFCPUExpertBundle = QwenGGUFCpuExpertBundle
build_qwen_gguf_cpu_expert_bundle = open_qwen_gguf_cpu_expert_bundle
open_qwen_gguf_cpu_bundle = open_qwen_gguf_cpu_expert_bundle
register_qwen_gguf_cpu_bundle = register_qwen_gguf_cpu_expert_bundle


__all__ = [
    "GGUFCpuBridgeError",
    "GGUFCpuExpertBundle",
    "QwenGGUFCPUExpertBundle",
    "QwenGGUFCpuExpertBundle",
    "UnsupportedGGUFCpuConfiguration",
    "build_qwen_gguf_cpu_expert_bundle",
    "open_qwen_gguf_cpu_bundle",
    "open_qwen_gguf_cpu_expert_bundle",
    "qwen_gguf_cpu_bridge_supported",
    "register_qwen_gguf_cpu_bundle",
    "register_qwen_gguf_cpu_expert_bundle",
]
