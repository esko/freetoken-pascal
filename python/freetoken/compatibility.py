"""Architecture-neutral GPU compatibility policy and startup reporting."""

from __future__ import annotations

from typing import Any

MINIMUM_CUDA_CAPABILITY = (6, 0)
PASCAL_MAX_CAPABILITY = (6, 2)
OPTIONAL_PACKAGE_MINIMUM_CC = {
    "flashinfer": (7, 5),
    "sgl_kernel": (7, 5),
    "triton_kernels": (7, 0),
}


def _sm(capability: tuple[int, int]) -> str:
    return f"sm_{capability[0]}{capability[1]}"


def validate_runtime_compatibility(capability: tuple[int, int], cuda_version: str | None) -> None:
    if capability < MINIMUM_CUDA_CAPABILITY:
        raise RuntimeError(
            f"FreeToken-Pascal requires compute capability 6.0 or newer; found {_sm(capability)}"
        )
    try:
        cuda_major = int((cuda_version or "0").split(".", 1)[0])
    except ValueError:
        cuda_major = 0
    if capability < (7, 5) and cuda_major >= 13:
        raise RuntimeError(
            f"CUDA {cuda_version} does not support {_sm(capability)}; use the pinned CUDA 12.6 "
            "environment"
        )


def build_compatibility_profile(
    capability: tuple[int, int],
    cuda_version: str | None,
    *,
    device_name: str = "unknown",
    installed_packages: dict[str, bool] | None = None,
) -> dict[str, Any]:
    validate_runtime_compatibility(capability, cuda_version)
    packages = installed_packages or {}
    pre_volta = capability < (7, 0)
    pre_turing = capability < (7, 5)
    package_modes = {}
    for name, minimum in OPTIONAL_PACKAGE_MINIMUM_CC.items():
        installed = bool(packages.get(name, False))
        package_modes[name] = {
            "installed": installed,
            "minimum_capability": _sm(minimum),
            "selected": installed and capability >= minimum,
        }
    return {
        "profile": "pascal" if capability <= PASCAL_MAX_CAPABILITY else "modern",
        "device_name": device_name,
        "capability": _sm(capability),
        "cuda": cuda_version or "unknown",
        "features": {
            "jit_parameter_storage": "plain" if pre_volta else "grid-constant",
            "streaming_global_load": "plain" if pre_volta else "l1-no-allocate",
            "gelu_tanh": "libdevice" if pre_turing else "tanh-approx",
            "moe_align": "staged-atomic-free" if pre_volta else "fused-atomic",
            "sampling": "torch-sort" if pre_volta else "triton-atomic",
            "attention_tiles": "shared-memory-budgeted",
        },
        "optional_packages": package_modes,
    }


def format_compatibility_profile(profile: dict[str, Any]) -> str:
    features = ", ".join(f"{key}={value}" for key, value in profile["features"].items())
    packages = ", ".join(
        f"{name}={'enabled' if state['selected'] else 'disabled'}"
        for name, state in profile["optional_packages"].items()
    )
    return (
        f"GPU compatibility profile: profile={profile['profile']}, "
        f"device={profile['device_name']!r}, capability={profile['capability']}, "
        f"cuda={profile['cuda']}, {features}, packages=[{packages}]"
    )


def runtime_compatibility_profile(device: int | None = None) -> dict[str, Any]:
    import torch

    from freetoken.kernel.backend import (
        is_flashinfer_installed,
        is_sgl_kernel_installed,
        is_triton_kernels_installed,
    )

    index = torch.cuda.current_device() if device is None else device
    return build_compatibility_profile(
        torch.cuda.get_device_capability(index),
        torch.version.cuda,
        device_name=torch.cuda.get_device_name(index),
        installed_packages={
            "flashinfer": is_flashinfer_installed(),
            "sgl_kernel": is_sgl_kernel_installed(),
            "triton_kernels": is_triton_kernels_installed(),
        },
    )


__all__ = [
    "MINIMUM_CUDA_CAPABILITY",
    "OPTIONAL_PACKAGE_MINIMUM_CC",
    "build_compatibility_profile",
    "format_compatibility_profile",
    "runtime_compatibility_profile",
    "validate_runtime_compatibility",
]
