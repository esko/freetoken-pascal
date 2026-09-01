from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.layers import LinearReplicated, silu_and_mul
from freetoken.layers.moe import MoELayer, make_moe_layer
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE, _SharedExpert

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None) -> None:
        if (
            getattr(config, "expert_quant", "none") == "gguf"
            or getattr(config, "moe_weight_format", None) == "gguf"
        ):
            # GGUF routed projections are heterogeneous host mappings.  Do not let the
            # generic CPU backend construct its homogeneous OffloadMoELayer (or allocate
            # dense resident expert tensors); the Engine attaches the explicit CPU adapter
            # after loading the non-expert state.
            self.experts = MoELayer(
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                renormalize=config.norm_topk_prob,
                activation="silu",
                allocate_experts=False,
                weight_format="gguf",
            )
            self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
            self.shared_expert = _SharedExpert(
                config, config.hidden_size, config.shared_expert_intermediate_size
            )
            self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)
            self.experts.layer_id = layer_id
            return
        if getattr(config, "expert_quant", "none") != "fp8_block":
            super().__init__(config, layer_id=layer_id)
            return
        # Qwen3.8's block-fp8 checkpoint quantizes only the routed experts; the shared
        # expert stays bf16, so hide expert_quant from _SharedExpert's fp8 branch and
        # rebuild the routed experts with the fp8_block bank layout.
        super().__init__(replace(config, expert_quant="none"), layer_id=layer_id)
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            weight_format="fp8_block",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        debug_observer: Callable[[str, dict[str, object]], None] | None = None,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if not hidden_states.is_cuda:
            if not all(hasattr(self.experts, name) for name in ("gate_up_proj", "down_proj")):
                raise RuntimeError(
                    "Qwen4 CPU reference MoE requires resident gate_up_proj/down_proj; "
                    "offload and adapter layouts are not supported by this path"
                )
            if getattr(self.experts, "weight_format", "bf16") != "bf16":
                raise RuntimeError(
                    "Qwen4 CPU reference MoE requires unquantized resident expert weights"
                )
        router_logits = self.gate.forward(hidden_states)
        if not hidden_states.is_cuda:
            # The model's CUDA MoE backends are intentionally not available on a
            # CPU-only host.  Reuse the model-neutral reference dispatcher for
            # the tiny text oracle, retaining exactly the same router, shared
            # gate, and expert tensors as the production layer.
            from .reference import routed_shared_expert_reference

            shared = self.shared_expert.forward(hidden_states)
            shared_gate = self.shared_expert_gate.forward(hidden_states)
            topk_weights, topk_ids = self.experts._route_and_observe(
                hidden_states, router_logits, debug_observer
            )
            gate_up = self.experts.gate_up_proj
            down = self.experts.down_proj

            def expert_forward(x: torch.Tensor, expert_id: int) -> torch.Tensor:
                intermediate = silu_and_mul(F.linear(x, gate_up[expert_id]))
                return F.linear(intermediate, down[expert_id])

            experts = [
                lambda x, expert_id=expert_id: expert_forward(x, expert_id)
                for expert_id in range(self.experts.num_experts)
            ]
            output, _, _ = routed_shared_expert_reference(
                hidden_states,
                router_logits,
                experts,
                topk=self.experts.top_k,
                shared_output=shared,
                shared_gate=shared_gate,
                renormalize=self.experts.renormalize,
                routing=(topk_weights, topk_ids),
            )
            return output.view(num_tokens, hidden_dim)
        shared = self.shared_expert.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)


__all__ = ["Qwen4ExpMoE"]
