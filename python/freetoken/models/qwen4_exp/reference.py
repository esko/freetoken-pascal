"""Small, device-neutral reference operations for Qwen3.8 correctness tests.

These functions intentionally favor readable equations over fused execution. They
are the A/B oracle for the runtime's GDN recurrence, hyperconnections, routing,
and PLE state; production forwards do not call them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F


def gated_delta_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the normalized gated-delta recurrence one token at a time.

    Inputs are ``[tokens, heads, dim]`` and the state is ``[heads, dim, dim]``.
    ``log_decay`` is the already-negative per-token log gate used by the fused
    kernels; ``beta`` is the sigmoid interpolation factor.
    """
    if not (query.shape == key.shape == value.shape):
        raise ValueError("GDN query, key, and value shapes must match")
    if log_decay.shape != query.shape[:2] or beta.shape != query.shape[:2]:
        raise ValueError("GDN gate shapes must be [tokens, heads]")
    state = (
        query.new_zeros(query.shape[1], query.shape[2], value.shape[2])
        if initial_state is None
        else initial_state.clone()
    )
    outputs = []
    scale = query.shape[-1] ** -0.5
    for token in range(query.shape[0]):
        q = F.normalize(query[token].float(), dim=-1)
        k = F.normalize(key[token].float(), dim=-1)
        v = value[token].float()
        state = state * log_decay[token].float().exp()[:, None, None]
        predicted = torch.einsum("hkv,hk->hv", state, k)
        delta = (v - predicted) * beta[token].float()[:, None]
        state = state + torch.einsum("hk,hv->hkv", k, delta)
        outputs.append(torch.einsum("hk,hkv->hv", q, state) * scale)
    output = torch.stack(outputs).to(value.dtype) if outputs else value.clone()
    return output, state


def gated_delta_chunked_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    chunks: Iterable[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the same recurrence over explicit chunk boundaries."""
    state = None
    outputs = []
    offset = 0
    for size in chunks:
        stop = offset + int(size)
        output, state = gated_delta_reference(
            query[offset:stop],
            key[offset:stop],
            value[offset:stop],
            log_decay[offset:stop],
            beta[offset:stop],
            state,
        )
        outputs.append(output)
        offset = stop
    if offset != query.shape[0]:
        raise ValueError(f"GDN chunks cover {offset} tokens, expected {query.shape[0]}")
    return torch.cat(outputs, dim=0), state


def hyperconnection_reference(
    hyper_input: torch.Tensor,
    *,
    stream_count: int,
    hidden_size: int,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inject_weight: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
    """Reference the four-stream mHC mixing/injection equations."""
    view = hyper_input.view(-1, stream_count, hidden_size).float()
    variance = view.square().mean(dim=-1, keepdim=True)
    normalized = view * torch.rsqrt(variance + eps)
    normalized = normalized * (1 + norm_weight.view(stream_count, hidden_size).float())
    flat = normalized.flatten(1)
    mix = F.silu(F.linear(flat, down_weight.float()) / stream_count)
    mix = torch.sigmoid(F.linear(mix, up_weight.float())).view_as(normalized)
    mixed = (mix * normalized).mean(dim=1).to(hyper_input.dtype)
    if inject_weight is None:
        return mixed
    inject = 2 * torch.sigmoid(F.linear(flat, inject_weight.float()) / stream_count)
    return mixed, hyper_input, inject.to(hyper_input.dtype)


def routed_shared_expert_reference(
    hidden: torch.Tensor,
    router_logits: torch.Tensor,
    experts: Iterable[Callable[[torch.Tensor], torch.Tensor]],
    *,
    topk: int,
    shared_output: torch.Tensor,
    shared_gate: torch.Tensor,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference Qwen routed experts plus its independently gated shared expert."""
    probabilities = torch.softmax(router_logits.float(), dim=-1)
    weights, expert_ids = torch.topk(probabilities, topk, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    expert_list = list(experts)
    routed = torch.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for slot in range(topk):
            expert_id = int(expert_ids[token, slot])
            routed[token] += expert_list[expert_id](hidden[token : token + 1])[0] * weights[
                token, slot
            ].to(hidden.dtype)
    output = routed + shared_output * torch.sigmoid(shared_gate).to(shared_output.dtype)
    return output, expert_ids.to(torch.int32), weights


__all__ = [
    "gated_delta_chunked_reference",
    "gated_delta_reference",
    "hyperconnection_reference",
    "routed_shared_expert_reference",
]
