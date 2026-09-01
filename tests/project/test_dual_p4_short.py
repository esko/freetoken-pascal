from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_dual_p4_short.py"
SPEC = importlib.util.spec_from_file_location("run_dual_p4_short", SCRIPT)
assert SPEC and SPEC.loader
DUAL_SHORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DUAL_SHORT)


def _devices() -> list[dict]:
    return json.loads(
        (ROOT / "tests/fixtures/results/qwen38-dual-p4-device.json").read_text(encoding="utf-8")
    )["devices"]


def test_nvidia_smi_capture_is_instantaneous_and_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_check_output(command: list[str], *, text: bool) -> str:
        assert text is True
        calls.append(command)
        return (
            "0, Tesla P4, GPU-00000000-0000-0000-0000-000000000000, 6.1, 8192, "
            "0000:02:00.0, 885, 2999, 42, 25.0, 75.0, Disabled\n"
        )

    sample = DUAL_SHORT.capture_nvidia_smi(0, check_output=fake_check_output)

    assert sample["uuid"].startswith("GPU-")
    assert sample["memory_mib"] == 8192
    assert sample["ecc_mode"] == "disabled"
    assert calls and calls[0][0:2] == ["nvidia-smi", "--id=0"]


def test_builder_emits_schema_valid_non_serving_evidence(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps({"profile_id": "ecc-off"}), encoding="utf-8")
    inventory = {"profile_id": "ecc-off"}

    evidence = DUAL_SHORT.build_evidence(
        inventory,
        inventory_path=str(inventory_path),
        devices=_devices(),
        operation_seconds=0.2,
        total_seconds=0.4,
    )

    assert DUAL_SHORT._load_module
    DUAL_SHORT._validate_evidence(evidence)
    assert evidence["serving"]["classification"] == "non-serving"
    assert evidence["hardware_inventory"]["sha256"] == DUAL_SHORT._sha256_file(inventory_path)


def test_builder_rejects_allocation_over_hard_bound(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text("{}", encoding="utf-8")
    devices = _devices()
    devices[0]["allocation_bytes"] = DUAL_SHORT.MAX_ALLOCATION_BYTES + 1

    with pytest.raises(ValueError, match="allocation exceeds"):
        DUAL_SHORT.build_evidence(
            {"profile_id": "ecc-off"},
            inventory_path=str(inventory_path),
            devices=devices,
            operation_seconds=0.2,
            total_seconds=0.4,
        )


def test_builder_rejects_more_than_two_devices(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text("{}", encoding="utf-8")
    devices = _devices()

    with pytest.raises(ValueError, match="exactly two"):
        DUAL_SHORT.build_evidence(
            {"profile_id": "ecc-off"},
            inventory_path=str(inventory_path),
            devices=[*devices, copy.deepcopy(devices[0])],
            operation_seconds=0.2,
            total_seconds=0.4,
        )
