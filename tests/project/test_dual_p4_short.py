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


def _inventory() -> dict:
    devices = _devices()
    return {
        "profile_id": "ecc-off",
        "gpus": [
            {
                "index": device["index"],
                "uuid": device["uuid"],
                "pci_bus_id": device["pci_bus_id"],
                "ecc_mode": "disabled",
                "ecc_pending_mode": "disabled",
                "topology": {
                    "pci_root": device["pci_root"],
                    "numa_node": device["numa_node"],
                },
            }
            for device in devices
        ],
    }


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
    inventory = _inventory()
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    evidence = DUAL_SHORT.build_evidence(
        inventory,
        inventory_path=str(inventory_path),
        devices=_devices(),
        operation_seconds=0.2,
        total_seconds=0.4,
        repository_commit="1" * 40,
    )

    assert DUAL_SHORT._load_module
    DUAL_SHORT._validate_evidence(evidence)
    assert evidence["serving"]["classification"] == "non-serving"
    assert evidence["hardware_inventory"]["sha256"] == DUAL_SHORT._sha256_file(inventory_path)


def test_builder_rejects_allocation_over_hard_bound(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.json"
    inventory = _inventory()
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    devices = _devices()
    devices[0]["allocation_bytes"] = DUAL_SHORT.MAX_ALLOCATION_BYTES + 1

    with pytest.raises(ValueError, match="allocation exceeds"):
        DUAL_SHORT.build_evidence(
            inventory,
            inventory_path=str(inventory_path),
            devices=devices,
            operation_seconds=0.2,
            total_seconds=0.4,
            repository_commit="1" * 40,
        )


def test_builder_rejects_more_than_two_devices(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.json"
    inventory = _inventory()
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    devices = _devices()

    with pytest.raises(ValueError, match="exactly two"):
        DUAL_SHORT.build_evidence(
            inventory,
            inventory_path=str(inventory_path),
            devices=[*devices, copy.deepcopy(devices[0])],
            operation_seconds=0.2,
            total_seconds=0.4,
            repository_commit="1" * 40,
        )


@pytest.mark.parametrize("field", ["uuid", "pci_bus_id"])
def test_builder_rejects_duplicate_device_identity(tmp_path: Path, field: str) -> None:
    inventory = _inventory()
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    devices = _devices()
    devices[1][field] = devices[0][field]

    with pytest.raises(ValueError, match="must be unique"):
        DUAL_SHORT.build_evidence(
            inventory,
            inventory_path=str(inventory_path),
            devices=devices,
            operation_seconds=0.2,
            total_seconds=0.4,
            repository_commit="1" * 40,
        )


def test_builder_rejects_device_profile_or_inventory_mismatch(tmp_path: Path) -> None:
    inventory = _inventory()
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    devices = _devices()
    devices[0]["ecc_profile"] = "ecc-on"

    with pytest.raises(ValueError, match="ECC profile disagrees"):
        DUAL_SHORT.build_evidence(
            inventory,
            inventory_path=str(inventory_path),
            devices=devices,
            operation_seconds=0.2,
            total_seconds=0.4,
            repository_commit="1" * 40,
        )


def test_builder_rejects_duplicate_inventory_identity(tmp_path: Path) -> None:
    inventory = _inventory()
    inventory["gpus"][1]["uuid"] = inventory["gpus"][0]["uuid"]
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ValueError, match="inventory GPU UUIDs must be unique"):
        DUAL_SHORT.build_evidence(
            inventory,
            inventory_path=str(inventory_path),
            devices=_devices(),
            operation_seconds=0.2,
            total_seconds=0.4,
            repository_commit="1" * 40,
        )

    inventory = _inventory()
    inventory["gpus"][1]["pci_bus_id"] = "0000:82:00.0"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with inventory"):
        DUAL_SHORT.build_evidence(
            inventory,
            inventory_path=str(inventory_path),
            devices=_devices(),
            operation_seconds=0.2,
            total_seconds=0.4,
            repository_commit="1" * 40,
        )
