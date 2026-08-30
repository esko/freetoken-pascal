from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import torch
import torch.nn.functional as F


def _mul_activation_reference(
    x: torch.Tensor,
    activation: str,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    if activation == "silu":
        value = F.silu(gate) * up
    elif activation == "gelu":
        value = F.gelu(gate) * up
    elif activation == "gelu_tanh":
        value = F.gelu(gate, approximate="tanh") * up
    else:
        raise ValueError(f"unsupported activation: {activation}")
    if out is not None:
        out.copy_(value)
        return out
    return value


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if not x.is_cuda:
        return _mul_activation_reference(x, "silu", out)
    from freetoken.kernel.backend import is_flashinfer_usable

    if is_flashinfer_usable():
        from flashinfer import silu_and_mul
    else:
        from freetoken.kernel.triton.activation import silu_and_mul

    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if not x.is_cuda:
        return _mul_activation_reference(x, "gelu", out)
    from freetoken.kernel.backend import is_flashinfer_usable

    if is_flashinfer_usable():
        from flashinfer import gelu_and_mul
    else:
        from freetoken.kernel.triton.activation import gelu_and_mul

    return gelu_and_mul(x, out=out)


def gelu_tanh_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    """tanh-approximate GELU gate (`gelu_pytorch_tanh`) followed by elementwise mul."""
    if not x.is_cuda:
        return _mul_activation_reference(x, "gelu_tanh", out)
    from freetoken.kernel.backend import is_flashinfer_usable

    if is_flashinfer_usable():
        from flashinfer import gelu_tanh_and_mul
    else:
        from freetoken.kernel.triton.activation import gelu_tanh_and_mul

    return gelu_tanh_and_mul(x, out=out)


def swigluoai_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
):
    """SwiGLU-OAI (gpt-oss / MiniMax-M3 ``swigluoai``) over UNINTERLEAVED halves
    (gate ``x[..., :d]``, up ``x[..., d:]``): ``clamp(gate, max=limit) *
    sigmoid(alpha * gate) * (clamp(up, +-limit) + 1)``. Always the in-repo Triton
    kernel (flashinfer ships no clamped-swiglu *_and_mul)."""
    if not x.is_cuda:
        gate, up = x.chunk(2, dim=-1)
        value = torch.clamp(gate, max=limit) * torch.sigmoid(alpha * gate)
        value = value * (torch.clamp(up, -limit, limit) + 1)
        if out is not None:
            out.copy_(value)
            return out
        return value
    from freetoken.kernel.triton.activation import swigluoai_and_mul

    return swigluoai_and_mul(x, out=out, alpha=alpha, limit=limit)


__all__ = ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul", "swigluoai_and_mul"]
