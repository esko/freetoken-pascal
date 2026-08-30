"""Torch-free routing dispatch contract.

The router implementation is intentionally kept separate from this module.  A
decision is an immutable value that can be passed to an observer without global
state, making the selected path visible even when several model workers route at
the same time.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

RouterMode = Literal["auto", "torch-reference", "triton-candidate"]
RouterImplementation = Literal["torch-reference", "triton-candidate", "triton-kernels"]

_MODES = frozenset({"auto", "torch-reference", "triton-candidate"})
_IMPLEMENTATIONS = frozenset({"torch-reference", "triton-candidate", "triton-kernels"})


class RouterDispatchError(RuntimeError):
    """Raised when a requested router implementation cannot be used safely."""


@dataclass(frozen=True, slots=True)
class RouterDispatchDecision:
    """Immutable, structured description of one router dispatch decision."""

    requested_mode: RouterMode
    selected_implementation: RouterImplementation
    topk: int
    num_experts: int
    renormalize: bool
    has_token_limit: bool
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.requested_mode) is not str or self.requested_mode not in _MODES:
            raise ValueError(f"unsupported router mode: {self.requested_mode!r}")
        if (
            type(self.selected_implementation) is not str
            or self.selected_implementation not in _IMPLEMENTATIONS
        ):
            raise ValueError(f"unsupported router implementation: {self.selected_implementation!r}")
        if type(self.topk) is not int or self.topk < 1:
            raise ValueError("router topk must be a positive integer")
        if type(self.num_experts) is not int or self.num_experts < 1:
            raise ValueError("router num_experts must be a positive integer")
        if self.topk > self.num_experts:
            raise ValueError("router topk cannot exceed num_experts")
        if type(self.renormalize) is not bool:
            raise TypeError("router renormalize must be a bool")
        if type(self.has_token_limit) is not bool:
            raise TypeError("router has_token_limit must be a bool")
        if self.fallback_reason is not None:
            if type(self.fallback_reason) is not str or not self.fallback_reason:
                raise ValueError("router fallback_reason must be a non-empty string")

    @property
    def token_limit(self) -> bool:
        """Compatibility alias for observers that call the field ``token_limit``."""

        return self.has_token_limit

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot suitable for a telemetry observer."""

        return asdict(self)


def _validate_availability(name: str, value: bool | None) -> bool | None:
    if value is not None and type(value) is not bool:
        raise TypeError(f"{name} must be a bool or None")
    return value


def _validate_mode(mode: str) -> RouterMode:
    if type(mode) is not str or mode not in _MODES:
        raise ValueError(f"unsupported router mode: {mode!r}")
    return mode  # type: ignore[return-value]


def parse_router_mode(mode: str) -> RouterMode:
    """Validate an explicit router mode without normalizing or falling back."""
    return _validate_mode(mode)


def router_mode_from_env() -> RouterMode:
    """Read the process router mode once at a configuration seam.

    The variable is intentionally exact and lowercase. An invalid value raises during
    backend/layer construction rather than silently changing the selected kernel.
    """
    value = os.environ.get("FREETOKEN_ROUTER_MODE")
    if value is None:
        return "auto"
    return parse_router_mode(value)


def resolve_router_dispatch(
    *,
    requested_mode: str = "auto",
    topk: int,
    num_experts: int,
    renormalize: bool,
    has_token_limit: bool,
    triton_candidate_available: bool | None = None,
    triton_kernels_available: bool | None = None,
    candidate_inputs_supported: bool = True,
) -> RouterDispatchDecision:
    """Resolve a router path without importing Torch, CUDA, or Triton.

    ``auto`` preserves the qualified external ``triton_kernels`` route for
    power-of-two K.  The in-tree arbitrary-K router is deliberately not enabled
    by auto; it is a candidate until its Pascal H1/H2 gates pass.  Availability
    values are supplied by the caller so this function remains deterministic and
    side-effect free.
    """

    mode = _validate_mode(requested_mode)
    if type(topk) is not int or isinstance(topk, bool) or topk < 1:
        raise ValueError("router topk must be a positive integer")
    if type(num_experts) is not int or isinstance(num_experts, bool) or num_experts < 1:
        raise ValueError("router num_experts must be a positive integer")
    if topk > num_experts:
        raise ValueError("router topk cannot exceed num_experts")
    if type(renormalize) is not bool:
        raise TypeError("router renormalize must be a bool")
    if type(has_token_limit) is not bool:
        raise TypeError("router has_token_limit must be a bool")
    candidate_available = _validate_availability(
        "triton_candidate_available", triton_candidate_available
    )
    external_available = _validate_availability(
        "triton_kernels_available", triton_kernels_available
    )
    if type(candidate_inputs_supported) is not bool:
        raise TypeError("candidate_inputs_supported must be a bool")

    common = {
        "requested_mode": mode,
        "topk": topk,
        "num_experts": num_experts,
        "renormalize": renormalize,
        "has_token_limit": has_token_limit,
    }
    if mode == "torch-reference":
        return RouterDispatchDecision(selected_implementation="torch-reference", **common)

    if mode == "triton-candidate":
        if candidate_available is not True:
            raise RouterDispatchError("triton-candidate unavailable: candidate-unavailable")
        if not candidate_inputs_supported:
            raise RouterDispatchError(
                "triton-candidate rejected input: candidate-unsupported-exceptional-input"
            )
        return RouterDispatchDecision(selected_implementation="triton-candidate", **common)

    # ``auto`` intentionally never selects the in-tree arbitrary-K candidate.
    if topk & (topk - 1):
        return RouterDispatchDecision(
            selected_implementation="torch-reference",
            fallback_reason="candidate-not-qualified",
            **common,
        )
    if external_available is True:
        return RouterDispatchDecision(selected_implementation="triton-kernels", **common)
    return RouterDispatchDecision(
        selected_implementation="torch-reference",
        fallback_reason="external-triton-unavailable",
        **common,
    )


__all__ = [
    "RouterDispatchDecision",
    "RouterDispatchError",
    "RouterImplementation",
    "RouterMode",
    "parse_router_mode",
    "resolve_router_dispatch",
    "router_mode_from_env",
]
