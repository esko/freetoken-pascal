from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "capture_hardware_inventory.py"
SPEC = importlib.util.spec_from_file_location("capture_hardware_inventory", SCRIPT)
assert SPEC and SPEC.loader
CAPTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPTURE)


@pytest.mark.parametrize(
    ("reported", "generation"),
    [("2.5 GT/s PCIe", 1), ("5.0 GT/s PCIe", 2), ("8.0 GT/s PCIe", 3)],
)
def test_pci_generation_accepts_linux_sysfs_format(reported: str, generation: int) -> None:
    assert CAPTURE._pci_generation(reported) == generation


def test_pci_generation_rejects_unknown_speed() -> None:
    with pytest.raises(RuntimeError, match="unsupported PCIe speed"):
        CAPTURE._pci_generation("64.0 GT/s PCIe")


def test_pci_generation_rejects_malformed_speed() -> None:
    with pytest.raises(RuntimeError, match="unable to parse PCIe speed"):
        CAPTURE._pci_generation("unknown")


def test_gpu_inventory_captures_current_and_pending_ecc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        CAPTURE,
        "_run",
        lambda *command: (
            "0, Tesla P4, GPU-00000000-0000-0000-0000-000000000000, 6.1, 7680, "
            "0000:02:00.0, 580.173.02, 810, 500, Enabled, Disabled"
        ),
    )
    monkeypatch.setattr(
        CAPTURE,
        "_pci_link",
        lambda _bus: ({"generation": 3, "width": 16}, {"generation": 1, "width": 16}),
    )
    monkeypatch.setattr(
        CAPTURE,
        "_pci_topology",
        lambda _bus: {"numa_node": 0, "pci_root": "pci0000:00", "sysfs_path": "/sys/gpu"},
    )

    gpu = CAPTURE._gpu_inventory()[0]

    assert gpu["ecc_mode"] == "enabled"
    assert gpu["ecc_pending_mode"] == "disabled"


def test_capture_profile_rejects_inconsistent_gpu_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        CAPTURE,
        "_gpu_inventory",
        lambda: [{"index": 0, "ecc_mode": "enabled", "ecc_pending_mode": "disabled"}],
    )
    monkeypatch.setattr(CAPTURE, "_run", lambda *command: "1" * 40)

    with pytest.raises(RuntimeError, match="ECC profile 'ecc-on'"):
        CAPTURE.capture(
            cuda_runtime="12.6",
            torch_version="2.11.0+cu126",
            torch_device_count=1,
            triton_version="3.6.0",
            profile_id="ecc-on",
        )


def test_capture_omits_optional_profile_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CAPTURE, "_gpu_inventory", lambda: [])
    monkeypatch.setattr(CAPTURE, "_nvme_inventory", lambda: [])
    monkeypatch.setattr(CAPTURE, "_run", lambda *command: "1" * 40)
    monkeypatch.setattr(CAPTURE, "_cpu_model", lambda: "test-cpu")
    monkeypatch.setattr(CAPTURE.os, "sysconf", lambda _name: 1)
    monkeypatch.setattr(CAPTURE.Path, "glob", lambda _self, _pattern: [])

    document = CAPTURE.capture(
        cuda_runtime="12.6",
        torch_version="2.11.0+cu126",
        torch_device_count=0,
        triton_version="3.6.0",
    )

    assert "profile_id" not in document
