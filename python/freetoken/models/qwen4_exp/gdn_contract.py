"""Torch-free dispatch contract for the Qwen4 GatedDeltaNet backend.

The contract deliberately knows nothing about Torch, CUDA, Triton, or the FLA implementation.
Callers supply the observed device capability, activation dtype, and package probes, then pass the
immutable decision to telemetry before importing a kernel.  Keeping this boundary device-neutral
also makes Pascal fallback tests runnable in the hosted CPU environment.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

GdnMode = Literal["auto", "torch-reference", "triton-candidate", "pascal-fp32"]
GdnImplementation = Literal["torch-reference", "fla", "triton-candidate", "pascal-fp32"]

GDN_MIN_FLA_CAPABILITY = (7, 0)
GDN_FLA_DTYPES = frozenset({"bfloat16", "float16"})
GDN_PASCAL_CAPABILITY = (6, 1)
GDN_PASCAL_DTYPE = "float32"
GDN_PASCAL_MODEL_DTYPES = frozenset({"bfloat16", "float16", "float32"})
_MODES = frozenset({"auto", "torch-reference", "triton-candidate", "pascal-fp32"})
_MODE_ALIASES = {"reference": "torch-reference"}
_IMPLEMENTATIONS = frozenset({"torch-reference", "fla", "triton-candidate", "pascal-fp32"})
_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "half": "float16",
    "fp16": "float16",
    "fp32": "float32",
    "fp64": "float64",
}


class GdnDispatchError(RuntimeError):
    """Raised when a requested GDN implementation cannot be used safely."""


@dataclass(frozen=True, slots=True)
class GdnDispatchDecision:
    """Immutable, serializable description of one GDN backend decision."""

    requested_mode: GdnMode
    selected_implementation: GdnImplementation
    capability: tuple[int, int]
    dtype: str
    fla_available: bool
    triton_candidate_available: bool
    fallback_reason: str | None = None
    pascal_fp32_available: bool = False

    def __post_init__(self) -> None:
        if type(self.requested_mode) is not str or self.requested_mode not in _MODES:
            raise ValueError(f"unsupported GDN mode: {self.requested_mode!r}")
        if (
            type(self.selected_implementation) is not str
            or self.selected_implementation not in _IMPLEMENTATIONS
        ):
            raise ValueError(f"unsupported GDN implementation: {self.selected_implementation!r}")
        _validate_capability(self.capability)
        if type(self.dtype) is not str or not self.dtype:
            raise ValueError("GDN dtype must be a non-empty string")
        if type(self.fla_available) is not bool:
            raise TypeError("fla_available must be a bool")
        if type(self.triton_candidate_available) is not bool:
            raise TypeError("triton_candidate_available must be a bool")
        if type(self.pascal_fp32_available) is not bool:
            raise TypeError("pascal_fp32_available must be a bool")
        if self.fallback_reason is not None:
            if type(self.fallback_reason) is not str or not self.fallback_reason:
                raise ValueError("GDN fallback_reason must be a non-empty string")

    @property
    def selected_backend(self) -> GdnImplementation:
        """Compatibility spelling for consumers that call implementations backends."""

        return self.selected_implementation

    @property
    def selected_mode(self) -> GdnImplementation:
        """Compatibility spelling used by startup telemetry readers."""

        return self.selected_implementation

    def as_dict(self) -> dict[str, Any]:
        """Return a telemetry-safe snapshot without exposing mutable decision state."""

        return asdict(self)


def _validate_capability(capability: tuple[int, int]) -> tuple[int, int]:
    if (
        type(capability) is not tuple
        or len(capability) != 2
        or any(type(part) is not int or isinstance(part, bool) or part < 0 for part in capability)
    ):
        raise TypeError("GDN capability must be a (major, minor) integer tuple")
    return capability


def _normalize_dtype(dtype: str) -> str:
    if type(dtype) is not str or not dtype.strip():
        raise TypeError("GDN dtype must be a non-empty string")
    normalized = dtype.strip().lower()
    if normalized.startswith("torch."):
        normalized = normalized[6:]
    return _DTYPE_ALIASES.get(normalized, normalized)


def _validate_availability(name: str, value: bool | None) -> bool | None:
    if value is not None and type(value) is not bool:
        raise TypeError(f"{name} must be a bool or None")
    return value


def _package_value(packages: Mapping[str, bool], names: tuple[str, ...]) -> bool | None:
    for name in names:
        if name in packages:
            value = packages[name]
            if type(value) is not bool:
                raise TypeError(f"package availability for {name!r} must be a bool")
            return value
    return None


def parse_gdn_mode(mode: str) -> GdnMode:
    """Validate and canonicalize an explicit GDN mode without fallback."""

    if type(mode) is not str:
        raise TypeError("unsupported GDN mode: mode must be a string")
    canonical = _MODE_ALIASES.get(mode, mode)
    if canonical not in _MODES:
        raise ValueError(
            "unsupported GDN mode: "
            f"{mode!r}; expected auto, reference, triton-candidate, or pascal-fp32"
        )
    return canonical  # type: ignore[return-value]


def gdn_mode_from_env() -> GdnMode:
    """Read the exact process-wide GDN mode at the model construction seam."""

    value = os.environ.get("FREETOKEN_GDN_MODE")
    if value is None:
        # Keep the backend spelling as a compatibility alias for launch scripts that group
        # model-family backend flags under *_BACKEND.  An explicitly set empty value is invalid.
        value = os.environ.get("FREETOKEN_GDN_BACKEND", "auto")
    return parse_gdn_mode(value)


def resolve_gdn_dispatch(
    *,
    requested_mode: str = "auto",
    capability: tuple[int, int],
    dtype: str,
    fla_available: bool | None = None,
    triton_candidate_available: bool | None = None,
    pascal_fp32_available: bool | None = None,
    package_availability: Mapping[str, bool] | None = None,
    candidate_inputs_supported: bool = True,
    observer: Callable[[GdnDispatchDecision], None] | None = None,
) -> GdnDispatchDecision:
    """Resolve a GDN implementation without importing Torch, CUDA, or Triton.

    ``auto`` preserves the current FLA path only for the qualified modern-GPU capability and
    supported model dtype.  Pascal and every unavailable/unsupported case select the pure-Torch
    reference implementation with a reason.  The candidate paths are never selected by ``auto``;
    they must be explicitly requested and have an affirmative availability probe.  The
    ``pascal-fp32`` path is restricted to ``sm_61`` and supported model activation dtypes;
    its recurrence is staged to FP32 by the model adapter. It remains unavailable until the
    caller supplies a positive qualification gate (H2 evidence is not inferred here).
    """

    mode = parse_gdn_mode(requested_mode)
    _validate_capability(capability)
    dtype_name = _normalize_dtype(dtype)
    fla = _validate_availability("fla_available", fla_available)
    candidate = _validate_availability("triton_candidate_available", triton_candidate_available)
    pascal = _validate_availability("pascal_fp32_available", pascal_fp32_available)
    if package_availability is not None:
        if not isinstance(package_availability, Mapping):
            raise TypeError("package_availability must be a mapping of package names to bools")
        package_fla = _package_value(package_availability, ("fla", "freetoken_fla", "triton"))
        package_candidate = _package_value(
            package_availability, ("triton-candidate", "triton_candidate", "candidate")
        )
        package_pascal = _package_value(
            package_availability, ("pascal-fp32", "pascal_fp32", "pascal")
        )
        if fla is not None and package_fla is not None and fla != package_fla:
            raise ValueError("fla_available conflicts with package_availability")
        if (
            candidate is not None
            and package_candidate is not None
            and candidate != package_candidate
        ):
            raise ValueError("triton_candidate_available conflicts with package_availability")
        if pascal is not None and package_pascal is not None and pascal != package_pascal:
            raise ValueError("pascal_fp32_available conflicts with package_availability")
        fla = package_fla if fla is None else fla
        candidate = package_candidate if candidate is None else candidate
        pascal = package_pascal if pascal is None else pascal
    if type(candidate_inputs_supported) is not bool:
        raise TypeError("candidate_inputs_supported must be a bool")
    if observer is not None and not callable(observer):
        raise TypeError("GDN observer must be callable")

    common = {
        "requested_mode": mode,
        "capability": capability,
        "dtype": dtype_name,
        "fla_available": fla is True,
        "triton_candidate_available": candidate is True,
        "pascal_fp32_available": pascal is True,
    }

    if mode == "torch-reference":
        decision = GdnDispatchDecision(selected_implementation="torch-reference", **common)
    elif mode == "triton-candidate":
        if candidate is not True:
            raise GdnDispatchError("triton-candidate unavailable: candidate-unavailable")
        if not candidate_inputs_supported:
            raise GdnDispatchError("triton-candidate rejected input: candidate-unsupported-input")
        decision = GdnDispatchDecision(selected_implementation="triton-candidate", **common)
    elif mode == "pascal-fp32":
        if capability != GDN_PASCAL_CAPABILITY:
            raise GdnDispatchError("pascal-fp32 requires sm_61 capability")
        if dtype_name not in GDN_PASCAL_MODEL_DTYPES:
            raise GdnDispatchError(
                "pascal-fp32 requires bfloat16, float16, or float32 model inputs"
            )
        if pascal is not True:
            raise GdnDispatchError("pascal-fp32 unavailable: pascal-fp32-unqualified")
        decision = GdnDispatchDecision(selected_implementation="pascal-fp32", **common)
    else:
        reason: str | None = None
        if capability < GDN_MIN_FLA_CAPABILITY:
            reason = "unsupported-capability"
        elif dtype_name not in GDN_FLA_DTYPES:
            reason = "unsupported-dtype"
        elif fla is not True:
            reason = "fla-unavailable"
        if reason is None:
            decision = GdnDispatchDecision(selected_implementation="fla", **common)
        else:
            decision = GdnDispatchDecision(
                selected_implementation="torch-reference",
                fallback_reason=reason,
                **common,
            )

    if observer is not None:
        observer(decision)
    return decision


# Upper-case aliases make the contract discoverable to callers that use acronym-style names,
# while the canonical names follow the repository's RouterDispatch* convention.
GDNDispatchDecision = GdnDispatchDecision
GDNDispatchError = GdnDispatchError

__all__ = [
    "GDN_FLA_DTYPES",
    "GDN_MIN_FLA_CAPABILITY",
    "GDN_PASCAL_CAPABILITY",
    "GDN_PASCAL_DTYPE",
    "GDN_PASCAL_MODEL_DTYPES",
    "GDNDispatchDecision",
    "GDNDispatchError",
    "GdnDispatchDecision",
    "GdnDispatchError",
    "GdnImplementation",
    "GdnMode",
    "gdn_mode_from_env",
    "parse_gdn_mode",
    "resolve_gdn_dispatch",
]
