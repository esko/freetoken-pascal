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
