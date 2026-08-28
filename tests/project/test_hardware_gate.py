from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_hardware_inventory.py"
SPEC = importlib.util.spec_from_file_location("check_hardware_inventory", SCRIPT)
assert SPEC and SPEC.loader
CHECK_HARDWARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK_HARDWARE)


def inventory(capabilities: list[str]) -> dict:
    return {
        "evidence_status": "measured",
        "gpus": [
            {"index": index, "compute_capability": capability}
            for index, capability in enumerate(capabilities)
        ],
    }


def test_single_pascal_inventory_passes() -> None:
    assert CHECK_HARDWARE.validate_pascal_inventory(inventory(["6.1"])) == []


def test_non_pascal_inventory_fails_instead_of_skipping() -> None:
    errors = CHECK_HARDWARE.validate_pascal_inventory(inventory(["8.0"]))

    assert errors == ["gpus[0] must have compute capability 6.1, found '8.0'"]


def test_dual_gate_requires_two_verified_pascal_devices() -> None:
    errors = CHECK_HARDWARE.validate_pascal_inventory(inventory(["6.1"]), minimum_gpus=2)

    assert errors == ["expected at least 2 GPUs, found 1"]


def test_synthetic_schema_example_cannot_unlock_hardware_gate() -> None:
    example = json.loads(
        (ROOT / "tests" / "fixtures" / "results" / "hardware.json").read_text(encoding="utf-8")
    )

    assert CHECK_HARDWARE.validate_pascal_inventory(example) == [
        "hardware gate requires evidence_status 'measured'"
    ]


def test_unclassified_inventory_cannot_unlock_hardware_gate() -> None:
    unclassified = inventory(["6.1"])
    del unclassified["evidence_status"]

    assert CHECK_HARDWARE.validate_pascal_inventory(unclassified) == [
        "hardware gate requires evidence_status 'measured'"
    ]
