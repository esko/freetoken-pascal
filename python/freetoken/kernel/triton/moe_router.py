# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# Adapted from SGLang (kernels/ops/moe/moe_fused_gate.py)
"""Fused softmax top-k MoE router (bias-free, ungrouped experts).

The ``num_token_non_padded`` row mask reads the device tensor, so it survives CUDA-graph capture.
"""

from __future__ import annotations

import functools
from typing import Tuple

import torch
import triton
import triton.language as tl
from freetoken.utils.arch import is_sm90_supported


@functools.cache
def _pdl_supported() -> bool:
    return is_sm90_supported()


@triton.jit
def _router_triton_kernel(
    scores_ptr,
    out_weights_ptr,
    out_indices_ptr,
    num_token_non_padded_ptr,
    M,
    stride_sm,
    stride_sn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    HAS_TOKEN_LIMIT: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    # Row-tiled: each program handles BLOCK_M rows; all reductions run along the
    # expert (N) axis. Tiling rows keeps CTAs large enough to stay occupancy-bound
    # rather than launch-bound at small N (many tiny 1-warp CTAs otherwise).
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # offs_m * stride can overflow int32 for large token counts.
    row_ptr = scores_ptr + offs_m[:, None].to(tl.int64) * stride_sm + offs_n[None, :] * stride_sn
    mask2d = mask_m[:, None] & mask_n[None, :]
    logits = tl.load(row_ptr, mask=mask2d, other=0.0).to(tl.float32)

    # Keep exceptional-value behavior identical to the Torch reference without a
    # device-to-host probe. NaNs and negative infinities are ranked last; positive
    # infinities share the probability mass; an all-invalid row is uniform.
    is_finite = mask2d & (logits == logits) & (logits > -float("inf")) & (logits < float("inf"))
    is_positive_inf = mask2d & (logits == float("inf"))
    finite_count = tl.sum(is_finite.to(tl.int32), axis=1)
    positive_inf_count = tl.sum(is_positive_inf.to(tl.int32), axis=1)
    has_finite = finite_count > 0
    has_positive_inf = positive_inf_count > 0

    finite_logits = tl.where(is_finite, logits, -float("inf"))
    row_max = tl.max(finite_logits, axis=1)[:, None]
    exp_row = tl.where(is_finite, tl.exp(logits - row_max), 0.0)
    normal_sum = tl.sum(exp_row, axis=1)[:, None]
    normal_activated = exp_row / tl.where(normal_sum > 0.0, normal_sum, 1.0)
    uniform_activated = tl.where(mask_n[None, :], 1.0 / N, 0.0)
    inf_activated = is_positive_inf.to(tl.float32) / tl.maximum(positive_inf_count[:, None], 1)
    activated = tl.where(
        has_positive_inf[:, None],
        inf_activated,
        tl.where(has_finite[:, None], normal_activated, uniform_activated),
    )

    # Map NaN and masked experts to the same lowest-ranking sentinel.
    ranked = tl.where(mask2d & (logits == logits), logits, -float("inf"))

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    selected_vals = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    selected_idx = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)

    cur = ranked
    available = mask_n[None, :]
    for k in tl.static_range(K):
        max_val = tl.max(tl.where(available, cur, -float("inf")), axis=1)[:, None]
        lane_id = tl.where(
            available & (cur == max_val), offs_n[None, :], N + 1
        )  # lowest expert id wins ties
        win_lane = tl.min(lane_id, axis=1)[:, None].to(tl.int32)
        win_activated = tl.sum(tl.where(offs_n[None, :] == win_lane, activated, 0.0), axis=1)[
            :, None
        ]
        slot = offs_k[None, :] == k
        selected_vals = tl.where(slot, win_activated, selected_vals)
        selected_idx = tl.where(slot, win_lane, selected_idx)
        winner = offs_n[None, :] == win_lane
        available = tl.where(winner, False, available)
        cur = tl.where(winner, -float("inf"), cur)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    if RENORMALIZE:
        routed_sum = tl.sum(tl.where(mask_k[None, :], selected_vals, 0.0), axis=1)[:, None]
        selected_vals = selected_vals / tl.where(routed_sum > 0.0, routed_sum, 1.0)

    if HAS_TOKEN_LIMIT:
        limit = tl.load(num_token_non_padded_ptr)
        selected_idx = tl.where(offs_m[:, None] < limit, selected_idx, -1)
        selected_vals = tl.where(offs_m[:, None] < limit, selected_vals, 0.0)

    out_w_ptr = (
        out_weights_ptr + offs_m[:, None].to(tl.int64) * stride_wm + offs_k[None, :] * stride_wk
    )
    out_i_ptr = (
        out_indices_ptr + offs_m[:, None].to(tl.int64) * stride_im + offs_k[None, :] * stride_ik
    )
    store_mask = mask_m[:, None] & mask_k[None, :]
    tl.store(out_w_ptr, selected_vals, mask=store_mask)
    tl.store(out_i_ptr, selected_idx, mask=store_mask)


def fused_topk_softmax(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax over all experts, top-k, then renormalize; ties keep the lowest expert id.

    ``num_token_non_padded`` is a device scalar; rows at or past it get expert id -1
    and zero weight. The candidate implements the same deterministic NaN, infinity and
    all-invalid row semantics as the Torch reference directly on device.
    """
    if gating_output.ndim != 2:
        raise ValueError("gating_output must be 2D")
    if type(topk) is not int or isinstance(topk, bool):
        raise TypeError("topk must be an integer")
    if type(renormalize) is not bool:
        raise TypeError("renormalize must be a bool")
    M, N = gating_output.shape
    if topk < 1 or topk > N:
        raise ValueError("topk must be between 1 and the number of experts")
    if num_token_non_padded is not None:
        integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        if not isinstance(num_token_non_padded, torch.Tensor):
            raise TypeError("num_token_non_padded must be an integer scalar tensor")
        if num_token_non_padded.ndim != 0:
            raise ValueError("num_token_non_padded must be a scalar tensor")
        if num_token_non_padded.dtype not in integer_dtypes:
            raise TypeError("num_token_non_padded must have an integer dtype")
        if num_token_non_padded.device != gating_output.device:
            raise ValueError("num_token_non_padded must be on the logits device")
        if num_token_non_padded.device.type == "cpu":
            token_limit = int(num_token_non_padded.item())
            if token_limit < 0 or token_limit > M:
                raise ValueError("num_token_non_padded must be within the token range")
    weights = torch.empty((M, topk), dtype=torch.float32, device=gating_output.device)
    indices = torch.empty((M, topk), dtype=torch.int32, device=gating_output.device)

    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_K = triton.next_power_of_2(topk)
    # Single warp per program keeps the per-row top-k reductions on cheap warp
    # shuffles; pack a few rows per program only when N is small so tiny launches
    # stay occupancy-bound. Swept on H100/B200; larger tiles / more warps regress
    # (register pressure).
    BLOCK_M = max(1, min(4, 256 // BLOCK_N))
    # For wide rows the K sequential argmax passes dominate and benefit from more
    # warps despite the cross-warp reduction cost.
    num_warps = 1 if BLOCK_N <= 512 else 4

    _router_triton_kernel[(triton.cdiv(M, BLOCK_M),)](
        gating_output,
        weights,
        indices,
        num_token_non_padded,
        M,
        gating_output.stride(0),
        gating_output.stride(1),
        weights.stride(0),
        weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        N=N,
        K=topk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        RENORMALIZE=renormalize,
        HAS_TOKEN_LIMIT=num_token_non_padded is not None,
        launch_pdl=_pdl_supported(),
        num_warps=num_warps,
    )
    return weights, indices


__all__ = ["fused_topk_softmax"]
