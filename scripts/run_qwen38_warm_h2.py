#!/usr/bin/env python3
"""Run one bounded warm-cache Qwen request without rehashing model shards.

The canonical full-H2 document remains authoritative for model-shard content.
This probe verifies its schema and SHA-256, checks the current shard names and
sizes, then lets normal Engine startup perform the dedicated PLE integrity
hash.  One request warms the selected working set and one identical request is
measured.  The result deliberately makes no steady-state TPS or thermal claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_NAME = "qwen38-gguf-cache-zero-warm-h2-evidence.schema.json"
PROMPT = "Write one short greeting."
MAX_NEW_TOKENS = 2


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unable to read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def load_inputs(
    *,
    full_h2_path: Path,
    inventory_path: Path,
    expected_profile: str,
    model_path: Path,
    ple_artifact_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validator = _load_module("validate_evidence_for_warm_h2", ROOT / "scripts/validate_evidence.py")
    full_h2 = _read_json(full_h2_path, label="full-H2 evidence")
    errors = validator.validate_document(full_h2, schema_dir=ROOT / "schemas")
    if errors:
        raise RuntimeError("full-H2 evidence is invalid: " + "; ".join(errors))
    if full_h2.get("schema_name") != "qwen38-gguf-cache-zero-h2-evidence.schema.json":
        raise RuntimeError("warm H2 requires canonical Qwen cache-zero full-H2 evidence")

    checker = _load_module(
        "check_hardware_inventory_for_warm_h2", ROOT / "scripts/check_hardware_inventory.py"
    )
    inventory = _read_json(inventory_path, label="hardware inventory")
    schema = _read_json(ROOT / "schemas/hardware-inventory.schema.json", label="inventory schema")
    inventory_errors = [
        error.message
        for error in checker.Draft202012Validator(
            schema, format_checker=checker.FORMAT_CHECKER
        ).iter_errors(inventory)
    ]
    inventory_errors.extend(
        checker.validate_pascal_inventory(
            inventory, minimum_gpus=1, expected_profile=expected_profile
        )
    )
    if inventory_errors:
        raise RuntimeError("hardware inventory is not accepted: " + "; ".join(inventory_errors))
    if inventory.get("profile_id") != expected_profile:
        raise RuntimeError("warm H2 requires an explicitly matching ECC profile")

    from freetoken.gguf_shards import gguf_shard_paths

    shards = list(gguf_shard_paths(model_path))
    expected_shards = full_h2["model"]["shards"]
    observed = [(path.name, path.stat().st_size) for path in shards]
    expected = [(item["name"], int(item["size"])) for item in expected_shards]
    if observed != expected:
        raise RuntimeError(
            "current model shard names/sizes do not match canonical full-H2 evidence; "
            "content hashes were not recomputed"
        )
    base_ple = full_h2["ple_artifact"]
    manifest_path = ple_artifact_path / "manifest.json"
    manifest = _read_json(manifest_path, label="PLE manifest")
    payload = ple_artifact_path / str(manifest.get("payload", ""))
    if manifest.get("sha256") != base_ple.get("sha256") or payload.stat().st_size != int(
        base_ple.get("payload_bytes", -1)
    ):
        raise RuntimeError("current PLE manifest identity does not match full-H2 evidence")
    return full_h2, inventory


def canonical_identity(
    full_h2: Mapping[str, Any],
    *,
    full_h2_path: Path,
    ple_artifact_path: Path,
) -> dict[str, Any]:
    model = full_h2["model"]
    ple = full_h2["ple_artifact"]
    source_commit = str(full_h2["repository_commit"])
    manifest_path = ple_artifact_path / "manifest.json"
    codec = ple["codec"]
    codec_identity = f"{codec['id']}@{codec['version']}"
    identity_body = {
        "repository_commit": source_commit,
        "model": model,
        "ple_artifact": ple,
    }
    return {
        "scope": "full-h2-model-ple-repository",
        "source_schema_name": "qwen38-gguf-cache-zero-h2-evidence.schema.json",
        "source_schema_version": 1,
        "source_full_h2_evidence_sha256": _sha256_file(full_h2_path),
        "canonicalization": "canonical-json-sha256-v1",
        "canonical_identity_sha256": _sha256_bytes(_canonical_bytes(identity_body)),
        "repository_commit": source_commit,
        "repository_identity_sha256": _sha256_bytes(source_commit.encode("ascii")),
        "model": {
            "repository": model["repository"],
            "revision": model["revision"],
            "variant": model["variant"],
            "canonical_identity_sha256": _sha256_bytes(_canonical_bytes(model)),
            "shards": model["shards"],
        },
        "ple_artifact": {
            "path": str(ple_artifact_path),
            "canonical_identity_sha256": _sha256_bytes(_canonical_bytes(ple)),
            "payload_sha256": ple["sha256"],
            "manifest_sha256": _sha256_file(manifest_path),
            "payload_bytes": int(ple["payload_bytes"]),
            "rows": int(ple["rows"]),
            "row_bytes": int(ple["row_bytes"]),
            "codec": codec_identity,
        },
        "hash_reuse": {
            "source_identity": "full-h2-canonical",
            "repository_commit": "reused-from-full-h2",
            "model_shards": "reused-from-full-h2",
            "ple_payload": "reused-from-full-h2",
            "runtime_ple_integrity_hash": "performed",
            "model_shard_hashes_recomputed": False,
            "ple_identity_recomputed": False,
        },
    }


def _telemetry_row(index: int = 0) -> dict[str, Any]:
    query = (
        "index,name,uuid,compute_cap,memory.total,pci.bus_id,"
        "clocks.current.graphics,clocks.current.memory,temperature.gpu,"
        "power.draw,power.limit,ecc.mode.current"
    )
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 12:
        raise RuntimeError("nvidia-smi did not return the expected warm-H2 telemetry")
    return {
        "index": int(parts[0]),
        "name": parts[1],
        "uuid": parts[2],
        "compute_capability": parts[3],
        "memory_mib": int(parts[4]),
        "pci_bus_id": parts[5].lower().replace("00000000:", "0000:"),
        "clocks": {"graphics_mhz": int(parts[6]), "memory_mhz": int(parts[7])},
        "temperature_celsius": float(parts[8]),
        "power_watts": float(parts[9]),
        "power_limit_watts": float(parts[10]),
        "ecc_mode": parts[11].lower(),
    }


class _Monitor:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            try:
                self.samples.append(_telemetry_row())
            except BaseException as error:
                self.error = error
                return

    def start(self) -> None:
        self.samples.append(_telemetry_row())
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self.samples.append(_telemetry_row())
        if self.error is not None:
            raise RuntimeError("GPU telemetry monitor failed") from self.error


def _counter_delta(after: Mapping[str, Any], before: Mapping[str, Any], key: str) -> int:
    return max(0, int(after.get(key, 0)) - int(before.get(key, 0)))


def build_evidence(
    *,
    full_h2: Mapping[str, Any],
    identity: Mapping[str, Any],
    inventory: Mapping[str, Any],
    inventory_path: Path,
    telemetry_samples: list[Mapping[str, Any]],
    ple_before: Mapping[str, Any],
    ple_after: Mapping[str, Any],
    startup_seconds: float,
    warmup_seconds: float,
    request_seconds: float,
    output_token_ids: list[int],
    repository_commit: str,
) -> dict[str, Any]:
    if not telemetry_samples:
        raise ValueError("warm H2 requires measured GPU telemetry")
    current = telemetry_samples[-1]
    inventory_gpu = next((gpu for gpu in inventory["gpus"] if gpu["uuid"] == current["uuid"]), None)
    if inventory_gpu is None or inventory_gpu["pci_bus_id"] != current["pci_bus_id"]:
        raise ValueError("visible GPU identity does not match the bound inventory")
    profile = inventory["profile_id"]
    expected_ecc = "disabled" if profile == "ecc-off" else "enabled"
    if current["ecc_mode"] != expected_ecc:
        raise ValueError("visible GPU ECC mode does not match the bound inventory profile")
    total_seconds = startup_seconds + warmup_seconds + request_seconds
    if warmup_seconds > 300 or request_seconds > 300 or total_seconds > 300:
        raise ValueError("warm H2 exceeded its 300-second hard bound")
    logical = _counter_delta(ple_after, ple_before, "lookup_rows")
    unique = _counter_delta(ple_after, ple_before, "batch_unique_rows")
    reads = _counter_delta(ple_after, ple_before, "application_reads")
    packed = _counter_delta(ple_after, ple_before, "packed_bytes_read")
    physical = _counter_delta(ple_after, ple_before, "storage_read_bytes")
    major = _counter_delta(ple_after, ple_before, "major_faults")
    if min(logical, unique, reads, packed) <= 0:
        raise ValueError("warm H2 PLE telemetry did not advance during the measured request")
    amplification = physical / packed
    peak_temp = max(float(sample["temperature_celsius"]) for sample in telemetry_samples)
    peak_power = max(float(sample["power_watts"]) for sample in telemetry_samples)
    power_limit = min(float(sample["power_limit_watts"]) for sample in telemetry_samples)
    topology = inventory_gpu["topology"]
    if len(repository_commit) != 40 or any(
        character not in "0123456789abcdef" for character in repository_commit
    ):
        raise ValueError("repository_commit must be a lowercase 40-character commit")
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": 1,
        "evidence_status": "measured",
        "evidence_kind": "single-p4-warm-cache",
        "claim_status": "bounded-correctness-only",
        "repository_commit": repository_commit,
        "identity": dict(identity),
        "hardware_inventory": {
            "path": str(inventory_path),
            "sha256": _sha256_file(inventory_path),
            "profile_id": profile,
            "device": {
                "uuid": current["uuid"],
                "pci_bus_id": current["pci_bus_id"],
                "pci_root": topology["pci_root"],
                "numa_node": topology["numa_node"],
                "ecc_profile": profile,
                "clocks": current["clocks"],
                "temperature_celsius": current["temperature_celsius"],
                "power_watts": current["power_watts"],
                "power_limit_watts": current["power_limit_watts"],
                "throttle_status": "not-assessed",
            },
        },
        "request": {
            "kind": "single-request",
            "request_count": 1,
            "concurrency": 1,
            "prompt_tokens": 5,
            "context_tokens": 5,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "timeout_seconds": 300,
        },
        "warmup": {
            "cache_state": "warm",
            "iterations": 1,
            "elapsed_seconds": warmup_seconds,
            "max_duration_seconds": 300,
        },
        "timing": {
            "scope": "bounded-warm-cache-request",
            "startup_seconds": startup_seconds,
            "warmup_seconds": warmup_seconds,
            "total_request_seconds": request_seconds,
            "total_seconds": total_seconds,
            "measured_iterations": 1,
        },
        "thermal": {
            "qualification": "unqualified",
            "peak_celsius": peak_temp,
            "observation_seconds": warmup_seconds + request_seconds,
        },
        "power": {
            "qualification": "observed-only",
            "peak_watts": peak_power,
            "power_limit_watts": power_limit,
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
            "decode_tokens_per_second": None,
            "prompt_tokens_per_second": None,
        },
        "ple": {
            "backend": ple_after["backend"],
            "cache_state": "warm",
            "source_kind": "dedicated-artifact",
            "logical_rows": logical,
            "unique_rows": unique,
            "application_reads": reads,
            "packed_bytes": packed,
            "physical_read_bytes": physical,
            "major_page_faults": major,
            "read_amplification": amplification,
            "identity_reused": True,
            "runtime_integrity_hash": "performed",
            "runtime_integrity_sha256": full_h2["ple_artifact"]["sha256"],
        },
        "execution": {
            "device_name": "Tesla P4",
            "compute_capability": "6.1",
            "gpu_count": 1,
            "model_loaded": True,
            "model_forward": True,
            "expert_execution": "cpu",
            "serving_scope": "single-p4-reference-only",
        },
        "claims": {
            "model_execution": True,
            "single_p4_only": True,
            "steady_state_tps": False,
            "thermal_qualification": False,
            "dual_p4_serving": False,
        },
        "output_token_ids": output_token_ids,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    full_h2, inventory = load_inputs(
        full_h2_path=args.full_h2,
        inventory_path=args.inventory,
        expected_profile=args.expected_profile,
        model_path=args.model,
        ple_artifact_path=args.ple_artifact,
    )
    identity = canonical_identity(
        full_h2, full_h2_path=args.full_h2, ple_artifact_path=args.ple_artifact
    )
    import torch
    from freetoken.core import SamplingParams
    from freetoken.llm import LLM

    llm = None
    monitor = _Monitor()
    started = time.monotonic()
    try:
        llm = LLM(
            str(args.model),
            dtype=torch.bfloat16,
            max_running_req=1,
            max_extend_tokens=8,
            max_seq_len_override=128,
            num_token_override=128,
            attention_backend="auto",
            moe_backend="cpu",
            moe_cache_size=0,
            moe_cache_auto=False,
            moe_cache_rate=None,
            moe_cpu_threads=args.cpu_threads,
            cuda_graph_bs=[],
            cuda_graph_max_bs=0,
            cache_type="naive",
            ple_artifact_path=str(args.ple_artifact),
            ple_backend=args.ple_backend,
            ple_warm_mode="cold",
            ple_planner_mode="vectorized",
        )
        startup_seconds = time.monotonic() - started
        monitor.start()
        params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, ignore_eos=True)
        warm_started = time.monotonic()
        warm = llm.generate([PROMPT], params)
        warmup_seconds = time.monotonic() - warm_started
        if len(warm) != 1 or len(warm[0]["token_ids"]) != MAX_NEW_TOKENS:
            raise RuntimeError("warmup request did not return the bounded token count")
        before = dict(next(iter(llm.engine.model.host_weight_telemetry().values())))
        request_started = time.monotonic()
        measured = llm.generate([PROMPT], params)
        request_seconds = time.monotonic() - request_started
        if len(measured) != 1 or len(measured[0]["token_ids"]) != MAX_NEW_TOKENS:
            raise RuntimeError("measured request did not return the bounded token count")
        if measured[0]["token_ids"] != warm[0]["token_ids"]:
            raise RuntimeError("warmup and measured deterministic token IDs disagree")
        after = dict(next(iter(llm.engine.model.host_weight_telemetry().values())))
        monitor.stop()
        document = build_evidence(
            full_h2=full_h2,
            identity=identity,
            inventory=inventory,
            inventory_path=args.inventory,
            telemetry_samples=monitor.samples,
            ple_before=before,
            ple_after=after,
            startup_seconds=startup_seconds,
            warmup_seconds=warmup_seconds,
            request_seconds=request_seconds,
            output_token_ids=list(measured[0]["token_ids"]),
            repository_commit=args.repository_commit,
        )
        validator = _load_module(
            "validate_generated_warm_h2", ROOT / "scripts/validate_evidence.py"
        )
        errors = validator.validate_document(document, schema_dir=ROOT / "schemas")
        if errors:
            raise RuntimeError("generated warm-H2 evidence is invalid: " + "; ".join(errors))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return document
    finally:
        if monitor._thread.is_alive():
            monitor.stop()
        if llm is not None:
            llm.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded single-P4 warm-cache evidence")
    parser.add_argument("--full-h2", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ple-artifact", type=Path, required=True)
    parser.add_argument("--ple-backend", choices=("mmap", "pread"), default="pread")
    parser.add_argument("--expected-profile", choices=("ecc-on", "ecc-off"), required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    args = parser.parse_args(argv)
    run(args)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
