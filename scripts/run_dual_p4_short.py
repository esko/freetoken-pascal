#!/usr/bin/env python3
"""Produce a bounded, non-serving two-P4 device evidence record.

This probe deliberately does not construct FreeToken, load a model, or exercise
peer-to-peer transfers.  It allocates one small tensor per visible device and
performs one local arithmetic operation, then captures instantaneous
``nvidia-smi`` telemetry for the evidence record.  The inventory remains the
authority for PCI/NUMA topology and ECC profile identity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_NAME = "qwen38-dual-p4-device-evidence.schema.json"
MAX_ITERATIONS = 3
MAX_ALLOCATION_BYTES = 4 * 1024 * 1024
_NVIDIA_QUERY = (
    "index,name,uuid,compute_cap,memory.total,pci.bus_id,"
    "clocks.current.graphics,clocks.current.memory,temperature.gpu,"
    "power.draw,power.limit,ecc.mode.current"
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inventory(path: Path, *, expected_profile: str | None) -> dict[str, Any]:
    try:
        inventory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unable to read hardware inventory {path}: {error}") from error
    checker = _load_module(
        "check_hardware_inventory_for_dual_short",
        ROOT / "scripts/check_hardware_inventory.py",
    )
    schema_path = ROOT / "schemas/hardware-inventory.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = [
        error.message
        for error in checker.Draft202012Validator(
            schema, format_checker=checker.FORMAT_CHECKER
        ).iter_errors(inventory)
    ]
    errors.extend(
        checker.validate_pascal_inventory(
            inventory,
            minimum_gpus=2,
            expected_profile=expected_profile,
        )
    )
    if errors:
        raise RuntimeError("hardware inventory is not accepted: " + "; ".join(errors))
    if not isinstance(inventory.get("profile_id"), str):
        raise RuntimeError("dual-p4-short requires a profiled inventory")
    if expected_profile is not None and inventory["profile_id"] != expected_profile:
        raise RuntimeError(
            f"inventory profile_id must be {expected_profile!r}, found {inventory['profile_id']!r}"
        )
    return inventory


def _parse_number(value: str, *, field: str) -> float:
    value = value.strip()
    if value.upper() in {"N/A", "NA", "NOT SUPPORTED"}:
        raise RuntimeError(f"nvidia-smi returned no instantaneous {field} telemetry")
    try:
        result = float(value)
    except ValueError as error:
        raise RuntimeError(f"invalid nvidia-smi {field} telemetry {value!r}") from error
    if result != result or result in {float("inf"), float("-inf")}:
        raise RuntimeError(f"non-finite nvidia-smi {field} telemetry")
    return result


def capture_nvidia_smi(
    index: int,
    *,
    check_output: Callable[..., str] = subprocess.check_output,
) -> dict[str, Any]:
    """Capture one real instantaneous GPU telemetry row.

    ``check_output`` is injectable for H0 tests, but the production default is
    intentionally the host/container ``nvidia-smi`` command rather than a
    synthetic or cached value.
    """
    output = check_output(
        [
            "nvidia-smi",
            f"--id={index}",
            f"--query-gpu={_NVIDIA_QUERY}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = [row for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"nvidia-smi returned {len(rows)} rows for GPU {index}")
    fields = [part.strip() for part in rows[0].split(",")]
    if len(fields) != 12:
        raise RuntimeError(f"nvidia-smi returned {len(fields)} fields, expected 12")
    (
        reported_index,
        name,
        uuid,
        compute_capability,
        memory_mib,
        pci_bus_id,
        graphics_mhz,
        memory_clock_mhz,
        temperature_celsius,
        power_watts,
        power_limit_watts,
        ecc_mode,
    ) = fields
    if int(reported_index) != index:
        raise RuntimeError(
            f"nvidia-smi index mismatch: requested {index}, reported {reported_index!r}"
        )
    current_ecc = ecc_mode.lower()
    if current_ecc not in {"enabled", "disabled"}:
        raise RuntimeError(f"unsupported instantaneous ECC mode {ecc_mode!r}")
    return {
        "index": index,
        "name": name,
        "uuid": uuid,
        "compute_capability": compute_capability,
        "memory_mib": int(memory_mib),
        "pci_bus_id": pci_bus_id.lower().replace("00000000:", "0000:"),
        "clocks": {
            "graphics_mhz": int(_parse_number(graphics_mhz, field="graphics clock")),
            "memory_mhz": int(_parse_number(memory_clock_mhz, field="memory clock")),
        },
        "temperature_celsius": _parse_number(temperature_celsius, field="temperature"),
        "power_watts": _parse_number(power_watts, field="power"),
        "power_limit_watts": _parse_number(power_limit_watts, field="power limit"),
        "ecc_mode": current_ecc,
    }


def _device_record(
    inventory_gpu: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    output_index: int,
    iterations: int,
    allocation_bytes: int,
) -> dict[str, Any]:
    if inventory_gpu.get("index") != output_index:
        raise RuntimeError("dual-p4-short requires inventory GPU indices 0 and 1")
    for identity in ("uuid", "pci_bus_id"):
        if inventory_gpu.get(identity) != telemetry.get(identity):
            raise RuntimeError(f"GPU {output_index} {identity} changed between inventory and probe")
    if telemetry.get("compute_capability") != "6.1" or telemetry.get("name") != "Tesla P4":
        raise RuntimeError("dual-p4-short requires instantaneous Tesla P4 sm_61 telemetry")
    ecc_profile = {"enabled": "ecc-on", "disabled": "ecc-off"}.get(str(telemetry.get("ecc_mode")))
    if ecc_profile is None:
        raise RuntimeError("instantaneous ECC mode cannot be mapped to an evidence profile")
    if inventory_gpu.get("ecc_mode") != telemetry.get("ecc_mode"):
        raise RuntimeError("instantaneous ECC mode disagrees with measured inventory")
    topology = inventory_gpu.get("topology")
    if not isinstance(topology, Mapping):
        raise RuntimeError(f"GPU {output_index} inventory has no measured topology")
    return {
        "index": output_index,
        "name": str(telemetry["name"]),
        "compute_capability": str(telemetry["compute_capability"]),
        "memory_mib": int(telemetry["memory_mib"]),
        "uuid": str(telemetry["uuid"]),
        "pci_bus_id": str(telemetry["pci_bus_id"]),
        "pci_root": str(topology["pci_root"]),
        "numa_node": int(topology["numa_node"]),
        "ecc_profile": ecc_profile,
        "clocks": dict(telemetry["clocks"]),
        "temperature_celsius": float(telemetry["temperature_celsius"]),
        "power_watts": float(telemetry["power_watts"]),
        "power_limit_watts": float(telemetry["power_limit_watts"]),
        "throttle_status": "not-assessed",
        "operation": "direct-device-addition",
        "iterations": iterations,
        "allocation_bytes": allocation_bytes,
        "result": "passed",
    }


def _inventory_gpu_identities(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact identity projection bound into dual-device evidence."""
    profile = inventory.get("profile_id")
    if profile not in {"ecc-on", "ecc-off"}:
        raise ValueError("dual-p4-short requires an ecc-on or ecc-off inventory profile")
    expected_ecc = "enabled" if profile == "ecc-on" else "disabled"
    gpus = inventory.get("gpus")
    if not isinstance(gpus, list) or len(gpus) != 2:
        raise ValueError("dual-p4-short requires exactly two inventory GPUs")

    identities: list[dict[str, Any]] = []
    seen_uuids: set[str] = set()
    seen_buses: set[str] = set()
    for gpu in gpus:
        if not isinstance(gpu, Mapping):
            raise ValueError("dual-p4-short inventory GPU records must be objects")
        index = gpu.get("index")
        uuid = gpu.get("uuid")
        pci_bus_id = gpu.get("pci_bus_id")
        topology = gpu.get("topology")
        if not isinstance(index, int) or isinstance(index, bool) or index not in (0, 1):
            raise ValueError("dual-p4-short inventory GPU indices must be 0 and 1")
        if not isinstance(uuid, str) or not uuid:
            raise ValueError(f"inventory GPU {index} has no UUID")
        if uuid in seen_uuids:
            raise ValueError("dual-p4-short inventory GPU UUIDs must be unique")
        seen_uuids.add(uuid)
        if not isinstance(pci_bus_id, str) or not pci_bus_id:
            raise ValueError(f"inventory GPU {index} has no PCI bus ID")
        if pci_bus_id in seen_buses:
            raise ValueError("dual-p4-short inventory PCI bus IDs must be unique")
        seen_buses.add(pci_bus_id)
        if gpu.get("ecc_mode") != expected_ecc or gpu.get("ecc_pending_mode") != expected_ecc:
            raise ValueError(f"inventory GPU {index} does not match profile {profile!r}")
        if not isinstance(topology, Mapping):
            raise ValueError(f"inventory GPU {index} has no measured topology")
        pci_root = topology.get("pci_root")
        numa_node = topology.get("numa_node")
        if not isinstance(pci_root, str) or not pci_root:
            raise ValueError(f"inventory GPU {index} has no PCI root")
        if not isinstance(numa_node, int) or isinstance(numa_node, bool) or numa_node < 0:
            raise ValueError(f"inventory GPU {index} has no valid NUMA node")
        identities.append(
            {
                "index": index,
                "uuid": uuid,
                "pci_bus_id": pci_bus_id,
                "pci_root": pci_root,
                "numa_node": numa_node,
                "ecc_profile": profile,
            }
        )
    if {identity["index"] for identity in identities} != {0, 1}:
        raise ValueError("dual-p4-short inventory GPU indices must be exactly 0 and 1")
    identities.sort(key=lambda identity: identity["index"])
    return identities


def _validate_device_identity_consistency(
    inventory: Mapping[str, Any], devices: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Ensure device observations are bound to the profiled inventory identities."""
    identities = _inventory_gpu_identities(inventory)
    expected_by_index = {identity["index"]: identity for identity in identities}
    seen_uuids: set[str] = set()
    seen_buses: set[str] = set()
    for device in devices:
        index = device.get("index")
        if index not in expected_by_index:
            raise ValueError(f"dual-p4-short device index {index!r} is not in the inventory")
        expected = expected_by_index[index]
        uuid = device.get("uuid")
        pci_bus_id = device.get("pci_bus_id")
        if not isinstance(uuid, str) or not uuid:
            raise ValueError(f"dual-p4-short device {index} has no UUID")
        if not isinstance(pci_bus_id, str) or not pci_bus_id:
            raise ValueError(f"dual-p4-short device {index} has no PCI bus ID")
        if uuid in seen_uuids:
            raise ValueError("dual-p4-short device UUIDs must be unique")
        if pci_bus_id in seen_buses:
            raise ValueError("dual-p4-short device PCI bus IDs must be unique")
        seen_uuids.add(uuid)
        seen_buses.add(pci_bus_id)
        for field in ("uuid", "pci_bus_id", "pci_root", "numa_node"):
            if device.get(field) != expected[field]:
                raise ValueError(f"dual-p4-short device {index} {field} disagrees with inventory")
        if device.get("ecc_profile") != expected["ecc_profile"]:
            raise ValueError(f"dual-p4-short device {index} ECC profile disagrees with inventory")
    return identities


def build_evidence(
    inventory: Mapping[str, Any],
    *,
    inventory_path: str,
    devices: list[Mapping[str, Any]],
    operation_seconds: float,
    total_seconds: float,
    repository_commit: str,
    observed_at: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Build the schema-shaped record from measured, injectable observations."""
    del observed_at  # retained in the seam so callers can provide a clock in future probes
    if len(devices) != 2:
        raise ValueError("dual-p4-short requires exactly two device records")
    if not 1 <= len(devices) <= MAX_ITERATIONS:
        raise ValueError("device iteration count exceeds the hard bound")
    if operation_seconds <= 0 or operation_seconds > 30:
        raise ValueError("operation_seconds must be in (0, 30]")
    if total_seconds <= 0 or total_seconds > 300:
        raise ValueError("total_seconds must be in (0, 300]")
    gpu_identities = _validate_device_identity_consistency(inventory, devices)
    for device in devices:
        if int(device["iterations"]) > MAX_ITERATIONS:
            raise ValueError("device iteration count exceeds the hard bound")
        if int(device["allocation_bytes"]) > MAX_ALLOCATION_BYTES:
            raise ValueError("device allocation exceeds the hard bound")
    temperatures = [float(device["temperature_celsius"]) for device in devices]
    powers = [float(device["power_watts"]) for device in devices]
    limits = [float(device["power_limit_watts"]) for device in devices]
    inventory_sha256 = _sha256_file(Path(inventory_path))
    if len(repository_commit) != 40 or any(
        character not in "0123456789abcdef" for character in repository_commit
    ):
        raise ValueError("repository_commit must be a lowercase 40-character commit")
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": 1,
        "evidence_status": "measured",
        "evidence_kind": "dual-p4-direct-device-smoke",
        "claim_status": "non-serving-device-only",
        "repository_commit": repository_commit,
        "hardware_inventory": {
            "path": str(inventory_path),
            "sha256": inventory_sha256,
            "profile_id": inventory["profile_id"],
            "gpu_identities": gpu_identities,
        },
        "request": {
            "kind": "direct-device-smoke",
            "request_count": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "timeout_seconds": 30,
        },
        "warmup": {"iterations": 0, "elapsed_seconds": 0.0, "max_duration_seconds": 15},
        "timing": {
            "scope": "short-device-smoke",
            "operation_seconds": operation_seconds,
            "total_seconds": total_seconds,
            "measured_iterations": 1,
        },
        "thermal": {
            "qualification": "unqualified",
            "peak_celsius": max(temperatures),
            "observation_seconds": total_seconds,
        },
        "power": {
            "qualification": "observed-only",
            "peak_watts": max(powers),
            "power_limit_watts": min(limits),
        },
        "throttling": {
            "qualification": "not-qualified",
            "assessment": "not-assessed",
            "throttle_events": 0,
            "claim": False,
        },
        "performance": {
            "status": "not-claimed",
            "claim": False,
            "steady_state": False,
            "tokens_per_second": None,
        },
        "serving": {
            "classification": "non-serving",
            "model_loaded": False,
            "model_forward": False,
            "tps_claimed": False,
            "thermal_qualified": False,
            "dual_gpu_policy": "not-selected",
        },
        "devices": list(devices),
        "claims": {
            "model_execution": False,
            "steady_state_tps": False,
            "thermal_qualification": False,
            "dual_p4_serving": False,
        },
    }


def _validate_evidence(document: Mapping[str, Any]) -> None:
    validator = _load_module(
        "validate_evidence_for_dual_short", ROOT / "scripts/validate_evidence.py"
    )
    errors = validator.validate_document(document, schema_dir=ROOT / "schemas")
    if errors:
        raise RuntimeError("generated evidence is invalid: " + "; ".join(errors))


def run_probe(
    *,
    inventory_path: Path,
    output_path: Path,
    expected_profile: str,
    repository_commit: str,
) -> dict[str, Any]:
    """Run one isolated arithmetic operation on each of the two visible P4s."""
    inventory = _load_inventory(inventory_path, expected_profile=expected_profile)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("dual-p4-short requires exactly two visible CUDA devices")
    allocation_bytes = 1024 * 1024
    measured_devices: list[dict[str, Any]] = []
    started = time.monotonic()
    operation_started = started
    for index in range(2):
        properties = torch.cuda.get_device_properties(index)
        if properties.name != "Tesla P4" or (properties.major, properties.minor) != (6, 1):
            raise RuntimeError(f"visible GPU {index} is not a Tesla P4 sm_61")
        elements = allocation_bytes // torch.tensor([], dtype=torch.float32).element_size()
        with torch.cuda.device(index):
            values = torch.ones(elements, dtype=torch.float32, device=f"cuda:{index}")
            result = values + index
            torch.cuda.synchronize(index)
            if float(result[0].item()) != float(index + 1):
                raise RuntimeError(f"GPU {index} arithmetic result mismatch")
            del result, values
    operation_seconds = time.monotonic() - operation_started
    telemetry = [capture_nvidia_smi(index) for index in range(2)]
    for index, (inventory_gpu, sample) in enumerate(zip(inventory["gpus"], telemetry, strict=True)):
        measured_devices.append(
            _device_record(
                inventory_gpu,
                sample,
                output_index=index,
                iterations=1,
                allocation_bytes=allocation_bytes,
            )
        )
    total_seconds = time.monotonic() - started
    document = build_evidence(
        inventory,
        inventory_path=str(inventory_path),
        devices=measured_devices,
        operation_seconds=operation_seconds,
        total_seconds=total_seconds,
        repository_commit=repository_commit,
    )
    _validate_evidence(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded non-serving dual-P4 evidence")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-profile", choices=("ecc-on", "ecc-off"), required=True)
    parser.add_argument("--repository-commit", required=True)
    args = parser.parse_args(argv)
    run_probe(
        inventory_path=args.inventory,
        output_path=args.output,
        expected_profile=args.expected_profile,
        repository_commit=args.repository_commit,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
