"""CPU-only Qwen GGUF routed-expert layer adapter.

This module is deliberately narrower than the CUDA ``OffloadMoELayer``.  It
adapts an already-owned :class:`QwenGGUFCpuExpertBundle` to the routed-expert
layer shape, without creating a homogeneous cache or moving tensors between
devices.  The explicit CPU boundary is useful for H0 correctness probes while
the production Engine registration seam remains fail-closed.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np

from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle

if TYPE_CHECKING:
    import torch

    DebugObserver = Callable[[str, dict[str, object]], None]


class QwenGGUFCpuMoELayer:
    """Adapt one Qwen GGUF layer in a shared CPU expert bundle.

    The bundle owns all mapped projections and the executor.  Each layer
    adapter only supplies its immutable ``layer_id`` and validates the Qwen
    geometry before forwarding a routed request.  ``forward`` uses the same
    full-softmax Torch router fallback as the generic MoE layer; callers that
    already own routing can use ``routed_forward`` directly.
    """

    _PROJECTIONS = ("gate", "up", "down")

    def __init__(
        self,
        bundle: QwenGGUFCpuExpertBundle,
        *,
        layer_id: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ) -> None:
        if not isinstance(bundle, QwenGGUFCpuExpertBundle):
            raise TypeError("Qwen GGUF CPU MoE layer requires a QwenGGUFCpuExpertBundle")
        self._bundle = bundle
        self.layer_id = self._checked_positive_or_zero(layer_id, "layer_id")
        self.num_experts = self._checked_positive(num_experts, "num_experts")
        self.top_k = self._checked_positive(top_k, "top_k")
        self.hidden_size = self._checked_positive(hidden_size, "hidden_size")
        self.intermediate_size = self._checked_positive(intermediate_size, "intermediate_size")
        if not isinstance(renormalize, bool):
            raise ValueError(f"renormalize must be a bool, got {renormalize!r}")
        if not isinstance(activation, str) or not activation:
            raise ValueError(f"activation must be a non-empty string, got {activation!r}")
        if not isinstance(apply_router_weight_on_input, bool):
            raise ValueError(
                f"apply_router_weight_on_input must be a bool, got {apply_router_weight_on_input!r}"
            )
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input

        info = self._tp_info()
        if info is not None and info.size != 1:
            raise ValueError(
                f"Qwen GGUF CPU MoE layer requires TP=1, got tensor-parallel size {info.size}"
            )
        if bundle.closed:
            raise RuntimeError("Qwen GGUF CPU expert bundle is closed")

        try:
            descriptors = [
                bundle.layout.descriptor(self.layer_id, projection)
                for projection in self._PROJECTIONS
            ]
        except Exception as error:
            raise ValueError(
                "Qwen GGUF CPU MoE layer requires gate, up, and down descriptors for "
                f"layer {self.layer_id}"
            ) from error

        for descriptor in descriptors:
            if descriptor.num_experts != self.num_experts:
                raise ValueError(
                    f"num_experts={self.num_experts} disagrees with layer {self.layer_id} "
                    f"{descriptor.projection} descriptor ({descriptor.num_experts})"
                )
        gate, up, down = descriptors
        if (gate.input_dim, gate.output_dim) != (
            self.hidden_size,
            self.intermediate_size,
        ):
            raise ValueError(
                f"hidden_size/intermediate_size ({self.hidden_size}, {self.intermediate_size}) "
                f"disagree with layer {self.layer_id} gate geometry "
                f"({gate.input_dim}, {gate.output_dim})"
            )
        if (up.input_dim, up.output_dim) != (self.hidden_size, self.intermediate_size):
            raise ValueError(
                f"layer {self.layer_id} up geometry ({up.input_dim}, {up.output_dim}) "
                "does not match gate geometry"
            )
        if (down.input_dim, down.output_dim) != (
            self.intermediate_size,
            self.hidden_size,
        ):
            raise ValueError(
                f"layer {self.layer_id} down geometry ({down.input_dim}, {down.output_dim}) "
                "is not the transposed gate geometry"
            )
        if bundle.layout.top_k != self.top_k:
            raise ValueError(
                f"top_k={self.top_k} disagrees with bundle ABI top_k={bundle.layout.top_k}"
            )

        executor = bundle.executor
        executor_reference = getattr(executor, "_reference", None)
        executor_activation = getattr(
            executor,
            "activation",
            getattr(executor_reference, "activation", None),
        )
        if executor_activation != self.activation:
            raise ValueError(
                f"activation={self.activation!r} disagrees with bundle executor "
                f"activation={executor_activation!r}"
            )
        executor_router_weight = getattr(
            executor,
            "apply_router_weight_on_input",
            getattr(executor_reference, "apply_router_weight_on_input", None),
        )
        if executor_router_weight != self.apply_router_weight_on_input:
            raise ValueError("apply_router_weight_on_input disagrees with the bundle executor")

    @staticmethod
    def _checked_positive(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    @staticmethod
    def _checked_positive_or_zero(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
        return value

    @property
    def bundle(self) -> QwenGGUFCpuExpertBundle:
        """The shared bundle; its caller owns its lifetime."""
        return self._bundle

    @property
    def closed(self) -> bool:
        return self._bundle.closed

    @property
    def last_telemetry(self):
        return self._bundle.last_telemetry

    @property
    def host_weight_telemetry(self) -> dict[str, object]:
        return self._bundle.host_weight_telemetry()

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        debug_observer: DebugObserver | None = None,
        *,
        num_token_non_padded: int | np.integer | None = None,
    ) -> torch.Tensor:
        """Route CPU logits with full-softmax Torch semantics and execute experts.

        The GGUF packed router is not part of this adapter.  ``router_logits``
        must therefore be supplied by a CPU-compatible caller; omitting it is
        rejected instead of inventing a different routing policy.
        """
        torch = self._torch()
        self._require_tp1()
        self._validate_hidden(hidden_states, torch)
        valid_tokens = self._validate_num_token_non_padded(
            num_token_non_padded,
            int(hidden_states.shape[0]),
        )
        if router_logits is None:
            raise ValueError(
                "router_logits is required for the CPU-only Qwen GGUF layer; "
                "the packed GGUF router is not a CPU adapter"
            )
        self._validate_cpu_tensor(router_logits, "router_logits", torch)
        if router_logits.ndim != 2:
            raise ValueError(f"router_logits must be rank 2, got {tuple(router_logits.shape)}")
        if router_logits.shape != (hidden_states.shape[0], self.num_experts):
            raise ValueError(
                "router_logits shape "
                f"{tuple(router_logits.shape)} does not match "
                f"({hidden_states.shape[0]}, {self.num_experts})"
            )
        if not router_logits.dtype.is_floating_point:
            raise ValueError(f"router_logits must be floating point, got {router_logits.dtype}")

        topk_weights, topk_ids = self._cpu_topk(
            router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            num_token_non_padded=valid_tokens,
        )
        return self._execute(
            hidden_states,
            topk_weights,
            topk_ids,
            debug_observer=debug_observer,
            num_token_non_padded=valid_tokens,
            torch=torch,
        )

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        num_token_non_padded: int | np.integer | None = None,
        debug_observer: DebugObserver | None = None,
    ) -> torch.Tensor:
        """Execute an externally prepared CPU routing decision.

        Route widths from one through ``top_k`` are accepted.  A ``-1`` expert
        ID is reserved for padded rows and is passed through to the ABI; every
        other ID must identify an expert in this layer.
        """
        torch = self._torch()
        self._require_tp1()
        self._validate_hidden(hidden_states, torch)
        valid_tokens = self._validate_num_token_non_padded(
            num_token_non_padded,
            int(hidden_states.shape[0]),
        )
        self._validate_cpu_tensor(topk_weights, "topk_weights", torch)
        self._validate_cpu_tensor(topk_ids, "topk_ids", torch)
        if topk_weights.ndim != 2 or topk_ids.ndim != 2:
            raise ValueError("topk_weights and topk_ids must be rank 2")
        if topk_weights.shape != topk_ids.shape:
            raise ValueError(
                "topk_weights and topk_ids shapes disagree: "
                f"{tuple(topk_weights.shape)} vs {tuple(topk_ids.shape)}"
            )
        if topk_weights.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "hidden_states and routing arrays have different token counts: "
                f"{hidden_states.shape[0]} vs {topk_weights.shape[0]}"
            )
        route_width = int(topk_ids.shape[1])
        if route_width <= 0 or route_width > self.top_k:
            raise ValueError(f"route width must be in [1, {self.top_k}], got {route_width}")
        if not topk_weights.dtype.is_floating_point:
            raise ValueError(f"topk_weights must be floating point, got {topk_weights.dtype}")
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"topk_ids must be int32 or int64, got {topk_ids.dtype}")
        invalid = (topk_ids < -1) | (topk_ids >= self.num_experts)
        if bool(invalid.any().item()):
            raise ValueError(
                f"expert id must be -1 or in [0, {self.num_experts}), got an out-of-range route"
            )
        return self._execute(
            hidden_states,
            topk_weights,
            topk_ids,
            debug_observer=debug_observer,
            num_token_non_padded=valid_tokens,
            torch=torch,
        )

    def _execute(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        debug_observer: DebugObserver | None,
        num_token_non_padded: int | None,
        torch: Any,
    ) -> torch.Tensor:
        if debug_observer is not None and not callable(debug_observer):
            raise TypeError("debug_observer must be callable")
        self._observe_route(
            debug_observer,
            topk_weights,
            topk_ids,
            num_token_non_padded,
        )
        result = self._bundle.decode(
            self.layer_id,
            hidden_states,
            topk_weights,
            topk_ids,
            num_token_non_padded=num_token_non_padded,
        )
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("Qwen GGUF CPU bundle returned a non-Torch result")
        if result.device.type != "cpu" or result.shape != hidden_states.shape:
            raise RuntimeError(
                "Qwen GGUF CPU bundle returned an invalid result: "
                f"device={result.device}, shape={tuple(result.shape)}, "
                f"expected CPU/{tuple(hidden_states.shape)}"
            )
        if result.dtype != hidden_states.dtype:
            raise RuntimeError(
                "Qwen GGUF CPU bundle changed the result dtype: "
                f"got {result.dtype}, expected {hidden_states.dtype}"
            )
        return result

    @staticmethod
    def _torch():
        try:
            import torch
        except ImportError as error:  # pragma: no cover - package dependency in production
            raise RuntimeError("Torch is required for the Qwen GGUF CPU layer adapter") from error
        return torch

    @staticmethod
    def _cpu_topk(
        router_logits: torch.Tensor,
        *,
        topk: int,
        renormalize: bool,
        num_token_non_padded: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match the pure-Torch branch of ``fused_topk`` without CUDA imports."""
        torch = QwenGGUFCpuMoELayer._torch()
        probs = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(probs, topk, dim=-1)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_ids = topk_ids.to(torch.int32)
        if num_token_non_padded is not None:
            indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
            topk_ids[indices >= num_token_non_padded, :] = -1
        return topk_weights.contiguous(), topk_ids.contiguous()

    @staticmethod
    def _validate_cpu_tensor(value: Any, name: str, torch: Any) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor, got {value.device}")

    def _validate_hidden(self, hidden_states: Any, torch: Any) -> None:
        self._validate_cpu_tensor(hidden_states, "hidden_states", torch)
        if hidden_states.ndim != 2:
            raise ValueError(f"hidden_states must be rank 2, got {tuple(hidden_states.shape)}")
        if hidden_states.shape[0] <= 0:
            raise ValueError("hidden_states must contain at least one token")
        if hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"hidden_states width {hidden_states.shape[1]} does not match "
                f"hidden_size={self.hidden_size}"
            )
        allowed = {torch.float16, torch.float32}
        if hasattr(torch, "bfloat16"):
            allowed.add(torch.bfloat16)
        if hidden_states.dtype not in allowed:
            raise ValueError(
                "hidden_states must use float16, bfloat16, or float32 on the CPU adapter; "
                f"got {hidden_states.dtype}"
            )

    @staticmethod
    def _validate_num_token_non_padded(
        value: int | np.integer | None,
        token_count: int,
    ) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"num_token_non_padded must be a non-negative integer, got {value!r}")
        value = int(value)
        if value < 0 or value > token_count:
            raise ValueError(f"num_token_non_padded must be within [0, {token_count}], got {value}")
        return value

    def _observe_route(
        self,
        observer: DebugObserver | None,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_token_non_padded: int | None,
    ) -> None:
        if observer is None:
            return
        valid_tokens = topk_ids.shape[0] if num_token_non_padded is None else num_token_non_padded
        observer(
            "router",
            {
                "layer_id": self.layer_id,
                "ids": topk_ids[:valid_tokens].detach().clone(),
                "weights": topk_weights[:valid_tokens].detach().clone(),
                "token_count": int(topk_ids.shape[0]),
                "valid_token_count": int(valid_tokens),
            },
        )

    @staticmethod
    def _require_tp1() -> None:
        info = QwenGGUFCpuMoELayer._tp_info()
        if info is not None and info.size != 1:
            raise ValueError(
                f"Qwen GGUF CPU MoE layer requires TP=1, got tensor-parallel size {info.size}"
            )

    @staticmethod
    def _tp_info():
        try:
            from freetoken.distributed.info import try_get_tp_info
        except ImportError:  # pragma: no cover - Torch is a runtime dependency of execution
            return None
        return try_get_tp_info()


__all__ = ["QwenGGUFCpuMoELayer"]
