from __future__ import annotations

import json
from pathlib import Path

import pytest
from freetoken.compatibility import (
    OPTIONAL_PACKAGE_MINIMUM_CC,
    build_compatibility_profile,
    format_compatibility_profile,
    validate_runtime_compatibility,
)

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "manifests" / "gpu-compatibility.json"


def test_pascal_profile_selects_every_required_fallback() -> None:
    profile = build_compatibility_profile(
        (6, 1),
        "12.6",
        device_name="Tesla P4",
        installed_packages={name: True for name in OPTIONAL_PACKAGE_MINIMUM_CC},
    )

    assert profile["profile"] == "pascal"
    assert profile["features"] == {
        "jit_parameter_storage": "plain",
        "streaming_global_load": "plain",
        "gelu_tanh": "libdevice",
        "moe_align": "staged-atomic-free",
        "sampling": "torch-sort",
        "attention_tiles": "shared-memory-budgeted",
    }
    assert not any(state["selected"] for state in profile["optional_packages"].values())
    report = format_compatibility_profile(profile)
    assert "profile=pascal" in report
    assert "capability=sm_61" in report
    assert "sampling=torch-sort" in report


def test_modern_profile_preserves_fast_paths() -> None:
    profile = build_compatibility_profile(
        (9, 0),
        "12.6",
        installed_packages={name: True for name in OPTIONAL_PACKAGE_MINIMUM_CC},
    )

    assert profile["profile"] == "modern"
    assert profile["features"]["jit_parameter_storage"] == "grid-constant"
    assert profile["features"]["gelu_tanh"] == "tanh-approx"
    assert all(state["selected"] for state in profile["optional_packages"].values())


def test_cuda_13_is_rejected_for_pascal() -> None:
    with pytest.raises(RuntimeError, match=r"pinned CUDA 12\.6"):
        validate_runtime_compatibility((6, 1), "13.0")


def test_pre_pascal_device_fails_clearly() -> None:
    with pytest.raises(RuntimeError, match=r"requires compute capability 6\.0"):
        validate_runtime_compatibility((5, 2), "12.6")


def test_compatibility_inventory_is_complete_and_points_to_sources() -> None:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    assert inventory["target"] == "sm_61"
    assert {entry["id"] for entry in inventory["features"]} == {
        "grid-constant-parameters",
        "streaming-global-load-hint",
        "approximate-tanh",
        "moe-align-atomics",
        "sampling-atomics",
        "attention-shared-memory",
    }
    for entry in inventory["features"]:
        assert entry["fallback"]
        assert entry["validation"]
        for source in entry["sources"]:
            assert (ROOT / source).is_file(), source

    package_entries = {entry["name"]: entry for entry in inventory["optional_packages"]}
    assert {
        name: package_entries[name]["minimum_capability"] for name in OPTIONAL_PACKAGE_MINIMUM_CC
    } == {
        name: f"{minimum[0]}.{minimum[1]}" for name, minimum in OPTIONAL_PACKAGE_MINIMUM_CC.items()
    }


def test_optional_package_callsites_use_capability_aware_probes() -> None:
    backend = (ROOT / "python/freetoken/kernel/backend.py").read_text(encoding="utf-8")
    assert "installed and is_arch_supported(*minimum)" in backend

    callsites = {
        "flashinfer": [
            "python/freetoken/engine/engine.py",
            "python/freetoken/engine/sample.py",
            "python/freetoken/layers/activation.py",
            "python/freetoken/layers/norm.py",
            "python/freetoken/layers/rotary.py",
        ],
        "sgl_kernel": [
            "python/freetoken/engine/engine.py",
            "python/freetoken/kernel/causal_conv1d.py",
            "python/freetoken/layers/norm.py",
            "python/freetoken/moe/fused.py",
        ],
        "triton_kernels": ["python/freetoken/moe/fused.py"],
    }
    for package, paths in callsites.items():
        for path in paths:
            source = (ROOT / path).read_text(encoding="utf-8")
            assert f"is_{package}_usable" in source, path
