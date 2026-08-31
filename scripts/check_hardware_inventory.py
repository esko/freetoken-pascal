from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
FORMAT_CHECKER = FormatChecker()


@FORMAT_CHECKER.checks("date-time", raises=ValueError)
def _is_rfc3339_datetime(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if (
        re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
            r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})",
            value,
        )
        is None
    ):
        return False
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.tzinfo is not None


def validate_pascal_inventory(data: Any, *, minimum_gpus: int = 1) -> list[str]:
    if not isinstance(data, dict):
        return ["inventory root must be an object"]
    gpus = data.get("gpus")
    if not isinstance(gpus, list):
        return ["gpus must be an array"]
    errors: list[str] = []
    if data.get("evidence_status") != "measured":
        return ["hardware gate requires evidence_status 'measured'"]
    if len(gpus) < minimum_gpus:
        errors.append(f"expected at least {minimum_gpus} GPUs, found {len(gpus)}")
    uuids: list[str] = []
    buses: list[str] = []
    for index, gpu in enumerate(gpus):
        if not isinstance(gpu, dict):
            errors.append(f"gpus[{index}] must be an object")
            continue
        if gpu.get("name") != "Tesla P4":
            errors.append(f"gpus[{index}] must be Tesla P4, found {gpu.get('name')!r}")
        if gpu.get("compute_capability") != "6.1":
            errors.append(
                f"gpus[{index}] must have compute capability 6.1, "
                f"found {gpu.get('compute_capability')!r}"
            )
        ecc_mode = gpu.get("ecc_mode")
        memory_mib = gpu.get("memory_mib")
        valid_memory = memory_mib == 7680 if ecc_mode == "enabled" else memory_mib in (7680, 8192)
        if ecc_mode not in ("enabled", "disabled") or not valid_memory:
            errors.append(
                f"gpus[{index}] has invalid ECC/memory profile {ecc_mode!r}/{memory_mib!r} MiB"
            )
        uuid = gpu.get("uuid")
        if not isinstance(uuid, str) or re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid) is None:
            errors.append(f"gpus[{index}].uuid must be a GPU UUID")
        else:
            uuids.append(uuid)
        bus = gpu.get("pci_bus_id")
        canonical_bus = (
            isinstance(bus, str)
            and re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", bus) is not None
        )
        if not canonical_bus:
            errors.append(f"gpus[{index}].pci_bus_id must be a canonical PCI address")
        else:
            buses.append(bus)
        topology = gpu.get("topology")
        numa_node = topology.get("numa_node") if isinstance(topology, dict) else None
        if not isinstance(numa_node, int) or isinstance(numa_node, bool):
            errors.append(f"gpus[{index}].topology must identify a NUMA node")
        elif (
            numa_node < 0
            or not isinstance(topology.get("pci_root"), str)
            or not topology["pci_root"]
        ):
            errors.append(f"gpus[{index}].topology must identify a measured PCI root")
        link = gpu.get("pci_link")
        if not isinstance(link, dict):
            errors.append(f"gpus[{index}].pci_link must be an object")
        else:
            maximum = link.get("maximum")
            current = link.get("current")
            if maximum != {"generation": 3, "width": 16}:
                errors.append(f"gpus[{index}] must report a Gen3 x16 maximum PCIe link")
            valid_current = (
                isinstance(current, dict)
                and current.get("generation") in (1, 2, 3)
                and current.get("width") == 16
            )
            if not valid_current:
                errors.append(f"gpus[{index}] current PCIe link must report generation 1..3 at x16")
            if link.get("current_is_idle") is True and link.get("load_qualified") is True:
                errors.append(f"gpus[{index}] idle PCIe link cannot be load qualified")
            if not isinstance(link.get("qualification"), str):
                errors.append(f"gpus[{index}].pci_link must describe its qualification")
    if len(uuids) != len(set(uuids)):
        errors.append("gpus UUIDs must be unique")
    if len(buses) != len(set(buses)):
        errors.append("gpus PCI bus IDs must be unique")

    thermal = data.get("thermal_qualification")
    if not isinstance(thermal, dict):
        errors.append("thermal_qualification must be an object")
    else:
        if thermal.get("status") != "unqualified":
            errors.append("thermal_qualification.status must be 'unqualified'")
        if thermal.get("load_test") != "not-run":
            errors.append("thermal_qualification.load_test must be 'not-run'")
        if not thermal.get("reason"):
            errors.append("thermal_qualification.reason must explain the deferred load gate")

    software = data.get("software")
    if not isinstance(software, dict):
        errors.append("software must be an object")
    else:
        if software.get("cuda_runtime") != "12.6":
            errors.append("software.cuda_runtime must be '12.6'")
        if software.get("torch_cuda_device_count") != len(gpus):
            errors.append("software.torch_cuda_device_count must match measured GPUs")
        if not software.get("nvidia_container_toolkit"):
            errors.append("software.nvidia_container_toolkit must be recorded")

    storage = data.get("storage")
    nvme = storage.get("nvme") if isinstance(storage, dict) else None
    if not isinstance(nvme, list) or not nvme:
        errors.append("storage.nvme must contain at least one measured device")
    else:
        for index, device in enumerate(nvme):
            topology = device.get("topology") if isinstance(device, dict) else None
            numa_node = topology.get("numa_node") if isinstance(topology, dict) else None
            if not isinstance(numa_node, int) or isinstance(numa_node, bool):
                errors.append(f"storage.nvme[{index}] must identify NUMA ownership")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reject non-Pascal hardware evidence")
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--minimum-gpus", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.inventory.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"ERROR: unable to read hardware inventory: {error}", file=sys.stderr)
        return 1
    schema = json.loads((ROOT / "schemas" / "hardware-inventory.schema.json").read_text())
    validator = Draft202012Validator(schema, format_checker=FORMAT_CHECKER)
    schema_errors = sorted(validator.iter_errors(data), key=lambda error: list(error.path))
    errors = [
        f"schema {'.'.join(str(part) for part in error.path) or '$'}: {error.message}"
        for error in schema_errors
    ]
    errors.extend(validate_pascal_inventory(data, minimum_gpus=args.minimum_gpus))
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"validated {len(data['gpus'])} sm_61 GPU(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
