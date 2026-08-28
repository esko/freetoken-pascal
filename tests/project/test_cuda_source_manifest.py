from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "compile_cuda_sources.py"
SPEC = importlib.util.spec_from_file_location("compile_cuda_sources", SCRIPT)
assert SPEC and SPEC.loader
COMPILE_CUDA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPILE_CUDA)


def manifest() -> dict:
    return json.loads((ROOT / "manifests" / "cuda-sources.json").read_text(encoding="utf-8"))


def test_every_shipping_cuda_translation_unit_is_declared() -> None:
    assert COMPILE_CUDA.validate_manifest(manifest()) == []


def test_missing_sm61_target_is_rejected() -> None:
    invalid = copy.deepcopy(manifest())
    invalid["target"] = "sm_80"

    assert COMPILE_CUDA.validate_manifest(invalid) == ["CUDA source manifest target must be sm_61"]


def test_missing_translation_unit_is_rejected() -> None:
    invalid = copy.deepcopy(manifest())
    removed = invalid["translation_units"].pop()

    errors = COMPILE_CUDA.validate_manifest(invalid)

    assert f"shipping CUDA source is absent from manifest: {removed['path']}" in errors


def test_grid_constant_uses_the_pascal_compatibility_guard() -> None:
    source_root = ROOT / "python" / "freetoken" / "kernel" / "csrc"
    raw_uses = []
    for path in source_root.rglob("*.cu*"):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "__grid_constant__" in line:
                raw_uses.append(f"{path.relative_to(ROOT)}:{line_number}")

    assert len(raw_uses) == 2
    assert all(
        path.startswith("python/freetoken/kernel/csrc/include/freetoken/utils.cuh:")
        for path in raw_uses
    )
