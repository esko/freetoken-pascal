from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _run(*command: str) -> str:
    return subprocess.check_output(command, text=True).strip()


def _pci_generation(speed: str) -> int:
    match = re.search(r"[0-9.]+", speed)
    if match is None:
        raise RuntimeError(f"unable to parse PCIe speed {speed!r}")
    value = float(match.group())
    mapping = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5}
    try:
        return mapping[value]
    except KeyError as error:
        raise RuntimeError(f"unsupported PCIe speed {speed!r}") from error


def _pci_link(bus: str) -> tuple[dict[str, int], dict[str, int]]:
    device = Path("/sys/bus/pci/devices") / bus
    return (
        {
            "generation": _pci_generation(
                (device / "max_link_speed").read_text(encoding="utf-8").strip()
            ),
            "width": int((device / "max_link_width").read_text(encoding="utf-8").strip()),
        },
        {
            "generation": _pci_generation(
                (device / "current_link_speed").read_text(encoding="utf-8").strip()
            ),
            "width": int((device / "current_link_width").read_text(encoding="utf-8").strip()),
        },
    )


def _pci_topology(bus: str) -> dict[str, Any]:
    device = Path("/sys/bus/pci/devices") / bus
    resolved = device.resolve()
    root = next((part for part in resolved.parts if part.startswith("pci0000:")), None)
    if root is None:
        raise RuntimeError(f"unable to identify PCI root for {bus}")
    return {
        "numa_node": int((device / "numa_node").read_text(encoding="utf-8").strip()),
        "pci_root": root,
        "sysfs_path": str(resolved),
    }


def _gpu_inventory() -> list[dict[str, Any]]:
    fields = (
        "index,name,uuid,compute_cap,memory.total,pci.bus_id,driver_version,"
        "clocks.current.graphics,clocks.current.memory,ecc.mode.current"
    )
    rows = _run(
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ).splitlines()
    gpus: list[dict[str, Any]] = []
    for row in rows:
        index, name, uuid, capability, memory, bus, driver, graphics, memory_clock, ecc = (
            part.strip() for part in row.split(",")
        )
        canonical_bus = bus.lower().replace("00000000:", "0000:")
        maximum, current = _pci_link(canonical_bus)
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "uuid": uuid,
                "compute_capability": capability,
                "memory_mib": int(memory),
                "pci_bus_id": canonical_bus,
                "driver": driver,
                "ecc_mode": ecc.lower(),
                "clocks": {
                    "graphics_mhz": int(graphics),
                    "memory_mhz": int(memory_clock),
                },
                "topology": _pci_topology(canonical_bus),
                "pci_link": {
                    "maximum": maximum,
                    "current": current,
                    "current_is_idle": True,
                    "load_qualified": False,
                    "qualification": "idle snapshot only; active-load link is not qualified",
                },
            }
        )
    return gpus


def _mounts_for(device_name: str) -> list[str]:
    raw = json.loads(_run("lsblk", "-J", "-o", "NAME,PKNAME,MOUNTPOINTS"))
    mounts: list[str] = []

    def visit(node: dict[str, Any], parent: str | None = None) -> None:
        name = str(node.get("name", ""))
        owner = str(node.get("pkname") or parent or "")
        if name == device_name or owner == device_name:
            mounts.extend(item for item in (node.get("mountpoints") or []) if item)
        for child in node.get("children") or []:
            visit(child, name)

    for block in raw.get("blockdevices", []):
        visit(block)
    return sorted(set(mounts))


def _nvme_inventory() -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for controller in sorted(Path("/sys/class/nvme").glob("nvme*")):
        if not controller.name[4:].isdigit():
            continue
        pci_device = (controller / "device").resolve()
        bus = pci_device.name.lower()
        maximum, current = _pci_link(bus)
        devices.append(
            {
                "name": controller.name,
                "model": (controller / "model").read_text(encoding="utf-8").strip(),
                "serial": (controller / "serial").read_text(encoding="utf-8").strip(),
                "pci_bus_id": bus,
                "topology": _pci_topology(bus),
                "pci_link": {"maximum": maximum, "current": current},
                "mounts": _mounts_for(f"{controller.name}n1"),
            }
        )
    return devices


def capture(
    *,
    cuda_runtime: str,
    torch_version: str,
    torch_device_count: int,
    triton_version: str,
) -> dict[str, Any]:
    commit = _run("git", "rev-parse", "HEAD")
    toolkit = _run("dpkg-query", "-W", "-f=${Version}", "nvidia-container-toolkit")
    return {
        "schema_name": "hardware-inventory.schema.json",
        "schema_version": 1,
        "evidence_status": "measured",
        "captured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "commit": commit,
        "host": {
            "hostname": socket.gethostname(),
            "os": platform.platform(),
            "cpu": platform.processor() or _run("lscpu", "-J"),
            "numa_nodes": len(list(Path("/sys/devices/system/node").glob("node[0-9]*"))),
            "memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        },
        "gpus": _gpu_inventory(),
        "software": {
            "cuda_driver": _run(
                "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
            ).splitlines()[0],
            "cuda_runtime": cuda_runtime,
            "cuda_toolkit": "12.6",
            "nvidia_container_toolkit": toolkit,
            "torch": torch_version,
            "torch_cuda_device_count": torch_device_count,
            "triton": triton_version,
        },
        "storage": {"nvme": _nvme_inventory()},
        "thermal_qualification": {
            "status": "unqualified",
            "load_test": "not-run",
            "throttling": "intentionally-throttled",
            "reason": "airflow optimization incomplete; sustained load was not run",
        },
        "capture": {
            "deterministic": True,
            "commands": ["nvidia-smi", "sysfs", "lsblk", "dpkg-query"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture fail-closed Pascal hardware inventory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--cuda-runtime", required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--torch-device-count", required=True, type=int)
    parser.add_argument("--triton-version", required=True)
    args = parser.parse_args()
    document = capture(
        cuda_runtime=args.cuda_runtime,
        torch_version=args.torch_version,
        torch_device_count=args.torch_device_count,
        triton_version=args.triton_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
