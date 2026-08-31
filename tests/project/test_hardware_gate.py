from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_hardware_inventory.py"
SPEC = importlib.util.spec_from_file_location("check_hardware_inventory", SCRIPT)
assert SPEC and SPEC.loader
CHECK_HARDWARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK_HARDWARE)


def inventory(capabilities: list[str]) -> dict:
    return {
        "schema_name": "hardware-inventory.schema.json",
        "schema_version": 1,
        "evidence_status": "measured",
        "captured_at": "2026-08-30T00:00:00Z",
        "commit": "1" * 40,
        "host": {
            "hostname": "gorilla",
            "os": "Ubuntu 26.04",
            "cpu": "Intel Xeon E5-2673 v3",
            "numa_nodes": 2,
            "memory_bytes": 128 * 1024**3,
        },
        "gpus": [
            {
                "index": index,
                "name": "Tesla P4",
                "compute_capability": capability,
                "memory_mib": 7680,
                "pci_bus_id": f"0000:{index + 2:02x}:00.0",
                "driver": "580.173.02",
                "ecc_mode": "enabled",
                "uuid": f"GPU-00000000-0000-0000-0000-{index:012d}",
                "clocks": {"graphics_mhz": 1, "memory_mhz": 1},
                "topology": {
                    "numa_node": index,
                    "pci_root": f"pci0000:{index:02x}",
                    "sysfs_path": f"/sys/devices/pci0000:{index:02x}/gpu",
                },
                "pci_link": {
                    "maximum": {"generation": 3, "width": 16},
                    "current": {"generation": 1, "width": 16},
                    "current_is_idle": True,
                    "load_qualified": False,
                    "qualification": "idle-only; load link not qualified",
                },
            }
            for index, capability in enumerate(capabilities)
        ],
        "software": {
            "cuda_driver": "580.173.02",
            "cuda_runtime": "12.6",
            "cuda_toolkit": "12.6",
            "nvidia_container_toolkit": "1.20.0",
            "torch": "2.11.0+cu126",
            "torch_cuda_device_count": len(capabilities),
            "triton": "3.6.0",
        },
        "storage": {
            "nvme": [
                {
                    "name": "nvme0",
                    "model": "Lenovo PS8",
                    "serial": "test-serial",
                    "pci_bus_id": "0000:03:00.0",
                    "topology": {
                        "numa_node": 0,
                        "pci_root": "pci0000:00",
                        "sysfs_path": "/sys/devices/pci0000:00/0000:03:00.0",
                    },
                    "pci_link": {
                        "maximum": {"generation": 3, "width": 4},
                        "current": {"generation": 3, "width": 4},
                    },
                    "mounts": ["/srv/nvme"],
                }
            ]
        },
        "thermal_qualification": {
            "status": "unqualified",
            "load_test": "not-run",
            "throttling": "intentionally-throttled",
            "reason": "airflow optimization incomplete; no sustained load run",
        },
        "capture": {
            "deterministic": True,
            "commands": ["nvidia-smi", "lspci", "numactl", "lsblk"],
        },
    }


def test_single_pascal_inventory_passes() -> None:
    assert CHECK_HARDWARE.validate_pascal_inventory(inventory(["6.1"])) == []


def test_measured_inventory_contract_is_schema_valid() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "hardware-inventory.schema.json").read_text(encoding="utf-8")
    )

    Draft202012Validator(schema).validate(inventory(["6.1", "6.1"]))


def test_measured_schema_requires_capture_and_valid_timestamp() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "hardware-inventory.schema.json").read_text(encoding="utf-8")
    )
    measured = inventory(["6.1"])
    del measured["capture"]
    measured["captured_at"] = "not-a-date"

    errors = list(
        Draft202012Validator(schema, format_checker=CHECK_HARDWARE.FORMAT_CHECKER).iter_errors(
            measured
        )
    )

    assert any(error.validator == "required" and "capture" in error.message for error in errors)
    assert any(
        error.validator == "format" and list(error.path) == ["captured_at"] for error in errors
    )


def test_ecc_off_inventory_accepts_full_physical_memory() -> None:
    measured = inventory(["6.1"])
    measured["gpus"][0]["ecc_mode"] = "disabled"
    measured["gpus"][0]["memory_mib"] = 8192

    assert CHECK_HARDWARE.validate_pascal_inventory(measured) == []


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


def test_inventory_rejects_duplicate_gpu_identity() -> None:
    measured = inventory(["6.1", "6.1"])
    measured["gpus"][1]["uuid"] = measured["gpus"][0]["uuid"]

    assert CHECK_HARDWARE.validate_pascal_inventory(measured) == [
        "gpus UUIDs must be unique",
    ]


def test_inventory_rejects_idle_pcie_link_as_load_qualified() -> None:
    measured = inventory(["6.1"])
    measured["gpus"][0]["pci_link"]["load_qualified"] = True

    assert CHECK_HARDWARE.validate_pascal_inventory(measured) == [
        "gpus[0] idle PCIe link cannot be load qualified",
    ]


def test_inventory_requires_thermal_state_to_remain_explicitly_unqualified() -> None:
    measured = inventory(["6.1"])
    del measured["thermal_qualification"]["load_test"]

    assert CHECK_HARDWARE.validate_pascal_inventory(measured) == [
        "thermal_qualification.load_test must be 'not-run'",
    ]
