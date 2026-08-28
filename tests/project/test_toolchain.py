from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "check_toolchain.py"
SPEC = importlib.util.spec_from_file_location("check_toolchain", SCRIPT_PATH)
assert SPEC and SPEC.loader
CHECK_TOOLCHAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK_TOOLCHAIN)


def manifest() -> dict:
    return json.loads((ROOT / "manifests" / "toolchain.json").read_text(encoding="utf-8"))


def test_repository_toolchain_contract_is_valid() -> None:
    assert CHECK_TOOLCHAIN.validate_toolchain(manifest(), root=ROOT) == []


def test_cuda_13_is_rejected() -> None:
    invalid = copy.deepcopy(manifest())
    invalid["cuda"] = "13.0.0"
    invalid["torch"] = "2.11.0+cu130"

    errors = CHECK_TOOLCHAIN.validate_toolchain(invalid, root=ROOT)

    assert "cuda must be pinned to the 12.6 release line" in errors
    assert "torch must be an explicit +cu126 build" in errors


def test_missing_sm61_target_is_rejected() -> None:
    invalid = copy.deepcopy(manifest())
    invalid["target_architectures"] = []

    errors = CHECK_TOOLCHAIN.validate_toolchain(invalid, root=ROOT)

    assert "target_architectures must contain exactly sm_61" in errors


def test_unpinned_container_digest_is_rejected() -> None:
    invalid = copy.deepcopy(manifest())
    invalid["images"]["cuda"]["digest"] = "latest"

    errors = CHECK_TOOLCHAIN.validate_toolchain(invalid, root=ROOT)

    assert "images.cuda.digest must be a sha256 digest" in errors


def test_missing_pascal_wheel_architecture_is_rejected() -> None:
    invalid = copy.deepcopy(manifest())
    invalid["torch_pascal_architectures"] = ["sm_80"]

    errors = CHECK_TOOLCHAIN.validate_toolchain(invalid, root=ROOT)

    assert "torch_pascal_architectures must include sm_60 or sm_61" in errors


def test_cpu_lock_excludes_gpu_runtime() -> None:
    locked = (ROOT / "requirements" / "cpu.lock").read_text(encoding="utf-8")

    assert "\ntorch==" not in locked
    assert "\nnvidia-" not in locked
    assert "\ncuda-toolkit==" not in locked


def test_cuda_container_also_installs_hosted_test_lock() -> None:
    dockerfile = (ROOT / "containers" / "cuda126.Dockerfile").read_text(encoding="utf-8")

    assert "--require-hashes -r requirements/cpu.lock" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "-r requirements/cuda126.lock" in dockerfile
