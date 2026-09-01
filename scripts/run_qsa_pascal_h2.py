#!/usr/bin/env python3
"""Run a bounded one-P4 QSA context sweep over the registered backend.

This producer intentionally uses one tiny Qwen4-Exp QSA layer and one request.  It records
raw CUDA-event phase observations for one prefill and one decode at contexts 128, 512, and
2048.  The observer never synchronizes; the forward returns before the producer synchronizes
and reads event timings.  The result is a context-scaling investigation, not a throughput,
thermal, model-quality, or sustained-load qualification.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_NAME = "qwen38-qsa-h2-evidence.schema.json"
CONTEXTS = (128, 512, 2048)
PHASES = ("prefill", "decode")
REPEATS = 1
HARD_TIMEOUT_SECONDS = 300.0
QSA_PHASES = (
    "store_kv",
    "index_cache_composite",
    "selection_composite",
    "selected_row_attention",
)
DONOR = {
    "id": "freetoken-qwen4-pr257-merge",
    "repository": "https://github.com/FlashML-org/FreeToken",
    "pull_request": 257,
    "ref": "bd8f3d519a48777bf22ee5c7c8f58f4f3ff31b40",
    "license": "Apache-2.0",
    "method": "upstream-qsa-context-sweep-reference",
}


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _strict_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"unable to read JSON {path}: {error}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(value: Any) -> str:
    import torch

    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def capture_telemetry(
    index: int = 0,
    *,
    check_output: Callable[..., str] = subprocess.check_output,
) -> dict[str, Any]:
    """Capture one identity-bearing instantaneous nvidia-smi row."""
    query = (
        "index,name,uuid,compute_cap,memory.total,pci.bus_id,"
        "clocks.current.graphics,clocks.current.memory,temperature.gpu,"
        "power.draw,power.limit,ecc.mode.current"
    )
    raw = check_output(
        [
            "nvidia-smi",
            f"--id={index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = [row for row in raw.splitlines() if row.strip()]
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
        memory_mhz,
        temperature,
        power,
        power_limit,
        ecc_mode,
    ) = fields
    if int(reported_index) != index:
        raise RuntimeError("nvidia-smi index mismatch")
    mode = ecc_mode.lower()
    if mode not in {"enabled", "disabled"}:
        raise RuntimeError(f"unsupported ECC mode {ecc_mode!r}")
    numbers = [float(value) for value in (temperature, power, power_limit)]
    if not all(math.isfinite(value) for value in numbers):
        raise RuntimeError("nvidia-smi returned non-finite QSA telemetry")
    return {
        "captured_at": _now(),
        "index": index,
        "name": name,
        "uuid": uuid,
        "compute_capability": compute_capability,
        "memory_mib": int(memory_mib),
        "pci_bus_id": pci_bus_id.lower().replace("00000000:", "0000:"),
        "ecc_mode": mode,
        "clocks": {"graphics_mhz": int(float(graphics_mhz)), "memory_mhz": int(float(memory_mhz))},
        "temperature_celsius": numbers[0],
        "power_watts": numbers[1],
        "power_limit_watts": numbers[2],
    }


def _identity_gpu(gpu: Mapping[str, Any]) -> dict[str, Any]:
    topology = gpu.get("topology")
    if not isinstance(topology, Mapping):
        raise RuntimeError("hardware inventory GPU lacks measured topology")
    return {
        "index": int(gpu["index"]),
        "name": gpu["name"],
        "compute_capability": gpu["compute_capability"],
        "memory_mib": int(gpu["memory_mib"]),
        "uuid": gpu["uuid"],
        "pci_bus_id": gpu["pci_bus_id"],
        "ecc_mode": gpu["ecc_mode"],
        "pci_root": topology["pci_root"],
        "numa_node": int(topology["numa_node"]),
    }


def _load_inventory(path: Path, *, expected_profile: str, gpu_index: int) -> dict[str, Any]:
    inventory = _strict_json(path)
    checker = _load_module(
        "check_hardware_inventory_for_qsa_h2", ROOT / "scripts/check_hardware_inventory.py"
    )
    schema = _strict_json(ROOT / "schemas/hardware-inventory.schema.json")
    errors = [
        error.message
        for error in checker.Draft202012Validator(
            schema, format_checker=checker.FORMAT_CHECKER
        ).iter_errors(inventory)
    ]
    errors.extend(
        checker.validate_pascal_inventory(
            inventory, minimum_gpus=1, expected_profile=expected_profile
        )
    )
    if errors:
        raise RuntimeError("hardware inventory is not accepted: " + "; ".join(errors))
    if inventory.get("profile_id") != expected_profile:
        raise RuntimeError("QSA H2 requires an explicitly matching ECC profile")
    matches = [
        gpu
        for gpu in inventory.get("gpus", [])
        if isinstance(gpu, Mapping) and gpu.get("index") == gpu_index
    ]
    if len(matches) != 1:
        raise RuntimeError(f"inventory must contain exactly one GPU index {gpu_index}")
    return inventory


class QSAPhaseEventCollector:
    """Record CUDA event pairs from observer callbacks without synchronizing in callbacks."""

    def __init__(self, torch: Any) -> None:
        self._torch = torch
        self._active: dict[tuple[str, int, int], tuple[Any, dict[str, Any]]] = {}
        self._finished: list[tuple[Any, Any, dict[str, Any]]] = []

    def begin_forward(self) -> None:
        if self._active or self._finished:
            raise RuntimeError("QSA phase collector must be reset between forwards")

    def __call__(self, event: str, metadata: Mapping[str, Any]) -> None:
        phase = str(metadata["phase"])
        key = (phase, int(metadata["layer_id"]), int(metadata["slot"]))
        if event == "begin":
            if key in self._active:
                raise RuntimeError(f"duplicate QSA phase begin for {key}")
            start = self._torch.cuda.Event(enable_timing=True)
            start.record()
            self._active[key] = (start, dict(metadata))
            return
        if event != "end":
            raise RuntimeError(f"unknown QSA phase observer event {event!r}")
        active = self._active.pop(key, None)
        if active is None:
            raise RuntimeError(f"QSA phase end without begin for {key}")
        finish = self._torch.cuda.Event(enable_timing=True)
        finish.record()
        self._finished.append((active[0], finish, active[1]))

    def finish(self) -> list[dict[str, Any]]:
        if self._active:
            raise RuntimeError(f"QSA phase observer left open events: {sorted(self._active)}")
        records = []
        for start, finish, metadata in self._finished:
            elapsed_ms = float(start.elapsed_time(finish))
            if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
                raise RuntimeError("QSA CUDA event returned invalid elapsed time")
            records.append(
                {
                    **metadata,
                    "elapsed_ns": max(0, round(elapsed_ms * 1_000_000)),
                }
            )
        self._finished = []
        return records


def _allocator_snapshot(torch: Any, device: Any) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    stats = torch.cuda.memory_stats(device)
    return {
        "driver_total_bytes": int(total),
        "driver_free_bytes": int(free),
        "allocator_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "allocator_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "allocator_high_water_bytes": int(torch.cuda.max_memory_allocated(device)),
        "allocation_count": int(stats.get("allocation.all.current", 0)),
        "allocation_retries": int(stats.get("num_alloc_retries", 0)),
        "allocation_failures": int(stats.get("num_ooms", 0)),
    }


def _sample_snapshot(snapshot: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: int(snapshot[key])
        for key in (
            "driver_total_bytes",
            "driver_free_bytes",
            "allocator_allocated_bytes",
            "allocator_reserved_bytes",
            "allocator_high_water_bytes",
        )
    }


def _phase_statistics(samples: list[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [int(sample[field]) for sample in samples]
    average = mean(values)
    return {
        "sample_count": len(values),
        "min_ns": min(values),
        "median_ns": int(median(values)),
        "max_ns": max(values),
        "coefficient_of_variation": 0.0 if average == 0 else pstdev(values) / average,
    }


def _workspace_record(fixture: Any, context_tokens: int, phase: str) -> dict[str, Any]:
    from freetoken.attention.qsa_workspace import QSAWorkspaceInputs, calculate_qsa_workspace

    args = fixture.config.qwen4_args
    group = next(group for group in fixture.config.attention_groups if group.name == "full")
    selected_path = fixture.backend.selected_path
    topk_backend = "torch" if selected_path != "triton" else "triton"
    rows = context_tokens if phase == "prefill" else 1
    request = QSAWorkspaceInputs(
        context_tokens=context_tokens,
        token_rows=rows,
        page_table_width=int(fixture.page_table.shape[1]),
        page_size=fixture.page_size,
        index_heads=args.index_n_heads,
        query_heads=fixture.config.num_qo_heads,
        kv_heads=fixture.config.num_kv_heads,
        head_dim=fixture.config.head_dim,
        index_head_dim=args.index_head_dim,
        top_k=args.index_budget,
        compression_ratio=args.index_ratio,
        num_index_layers=len(group.layer_ids),
        num_req_slots=fixture.num_req_slots,
        ring_capacity=fixture.pool.ring_capacity,
        # The cache pool includes one physical dummy page beyond the usable page table.
        num_pages=int(fixture.page_table.shape[1] // fixture.page_size) + 1,
        max_position=group.rotary_config.max_position,
        rotary_dim=group.rotary_config.rotary_dim,
        batch_size=1,
        phase="eager",
        topk_backend=topk_backend,
        qsa_selection_path=selected_path,
    )
    plan = calculate_qsa_workspace(request)
    return {
        "context_tokens": context_tokens,
        "phase": phase,
        "selected_path": selected_path,
        "topk_backend": topk_backend,
        "plan": plan.as_dict(),
    }


def _run_forward(
    *,
    fixture: Any,
    attn: Any,
    request: Any,
    phase: str,
    context_tokens: int,
    repeat: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    generator = torch.Generator(device=fixture.device).manual_seed(seed)
    length = request.extend_len
    x = (
        torch.randn(
            length,
            fixture.config.hidden_size,
            device=fixture.device,
            dtype=fixture.dtype,
            generator=generator,
        )
        * 0.05
    )
    batch = fixture.batch([request], phase)
    before = _allocator_snapshot(torch, fixture.device)
    collector = QSAPhaseEventCollector(torch)
    fixture.backend.set_phase_observer(collector)
    collector.begin_forward()
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    host_started = time.perf_counter_ns()
    output = attn.forward(x, batch)
    total_end.record()
    host_elapsed_ns = max(0, time.perf_counter_ns() - host_started)
    # Observer callbacks only record events.  Synchronize once after the complete forward,
    # then read the phase and total event timings.
    torch.cuda.synchronize(fixture.device)
    total_elapsed_ms = float(total_start.elapsed_time(total_end))
    if not math.isfinite(total_elapsed_ms) or total_elapsed_ms < 0:
        raise RuntimeError("QSA total CUDA event returned invalid elapsed time")
    phase_events = collector.finish()
    after = _allocator_snapshot(torch, fixture.device)
    if set(event["phase"] for event in phase_events) != set(QSA_PHASES):
        raise RuntimeError("registered QSA backend did not report all four phases")
    phase_elapsed = {
        phase_name: sum(
            int(event["elapsed_ns"]) for event in phase_events if event["phase"] == phase_name
        )
        for phase_name in QSA_PHASES
    }
    return {
        "id": f"context-{context_tokens}-{phase}-repeat-{repeat}",
        "context_tokens": context_tokens,
        "phase": phase,
        "repeat": repeat,
        "status": "passed",
        "synchronized": True,
        "total_elapsed_ns": max(0, round(total_elapsed_ms * 1_000_000)),
        "host_elapsed_ns": host_elapsed_ns,
        "phase_elapsed_ns": phase_elapsed,
        "phase_event_count": len(phase_events),
        "phase_events": phase_events,
        "output_sha256": _sha256_tensor(output),
        "output_shape": [int(value) for value in output.shape],
        "allocator_before": _sample_snapshot(before),
        "allocator_after": _sample_snapshot(after),
    }


def _checkpoint(
    name: str,
    snapshot: Mapping[str, Any] | None,
    *,
    context_tokens: int | None,
    reason: str | None = None,
) -> dict[str, Any]:
    if snapshot is None:
        return {
            "name": name,
            "status": "unmeasured",
            "context_tokens": context_tokens,
            "driver_total_bytes": None,
            "driver_free_bytes": None,
            "allocator_allocated_bytes": None,
            "allocator_reserved_bytes": None,
            "allocator_high_water_bytes": None,
            "allocation_count": None,
            "allocation_retries": None,
            "allocation_failures": None,
            "reason": reason,
        }
    return {
        "name": name,
        "status": "measured",
        "context_tokens": context_tokens,
        **{
            key: int(snapshot[key])
            for key in (
                "driver_total_bytes",
                "driver_free_bytes",
                "allocator_allocated_bytes",
                "allocator_reserved_bytes",
                "allocator_high_water_bytes",
                "allocation_count",
                "allocation_retries",
                "allocation_failures",
            )
        },
        "reason": reason,
    }


def _identity_checks(
    *,
    inventory: Mapping[str, Any],
    telemetry: list[Mapping[str, Any]],
    gpu_index: int,
    profile: str,
) -> None:
    gpu = next(gpu for gpu in inventory["gpus"] if gpu["index"] == gpu_index)
    expected_ecc = "disabled" if profile == "ecc-off" else "enabled"
    for row in telemetry:
        if (
            row["index"] != gpu_index
            or row["uuid"] != gpu["uuid"]
            or row["pci_bus_id"] != gpu["pci_bus_id"]
        ):
            raise RuntimeError("instantaneous QSA telemetry does not match inventory identity")
        if row["ecc_mode"] != expected_ecc:
            raise RuntimeError("instantaneous QSA telemetry does not match ECC profile")


def build_evidence(
    *,
    inventory: Mapping[str, Any],
    inventory_path: Path,
    telemetry: list[Mapping[str, Any]],
    samples: list[Mapping[str, Any]],
    workspace_plans: list[Mapping[str, Any]],
    allocator_checkpoints: list[Mapping[str, Any]],
    selected_path: str,
    topk_backend: str,
    duration_seconds: float,
    repository_commit: str,
    repeats: int = REPEATS,
) -> dict[str, Any]:
    if len(repository_commit) != 40 or any(
        char not in "0123456789abcdef" for char in repository_commit
    ):
        raise ValueError("repository_commit must be a lowercase 40-character commit")
    if not samples:
        raise ValueError("QSA H2 requires raw samples")
    if not telemetry:
        raise ValueError("QSA H2 requires measured GPU telemetry")
    if duration_seconds <= 0 or duration_seconds > HARD_TIMEOUT_SECONDS:
        raise ValueError("QSA H2 exceeded its 300-second hard bound")
    gpu_index = int(inventory["gpus"][0]["index"])
    profile = str(inventory["profile_id"])
    _identity_checks(inventory=inventory, telemetry=telemetry, gpu_index=gpu_index, profile=profile)
    contexts = list(CONTEXTS)
    phases = list(PHASES)
    matrix_complete = {(s["context_tokens"], s["phase"], s["repeat"]) for s in samples} == {
        (context, phase, repeat)
        for context in contexts
        for phase in phases
        for repeat in range(repeats)
    }
    phase_stats: list[dict[str, Any]] = []
    for context in contexts:
        for phase in phases:
            selected = [
                sample
                for sample in samples
                if sample["context_tokens"] == context and sample["phase"] == phase
            ]
            if not selected:
                continue
            composite: dict[str, Any] = {}
            for phase_name in QSA_PHASES:
                composite[phase_name] = _phase_statistics(
                    [{"value": sample["phase_elapsed_ns"][phase_name]} for sample in selected],
                    "value",
                )
            phase_stats.append(
                {
                    "context_tokens": context,
                    "phase": phase,
                    "total": _phase_statistics(selected, "total_elapsed_ns"),
                    "composite_phases": composite,
                }
            )
    inventory_gpu = next(gpu for gpu in inventory["gpus"] if gpu["index"] == gpu_index)
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": 1,
        "evidence_status": "measured",
        "evidence_kind": "qwen38-pascal-qsa-context-sweep",
        "claim_status": "bounded-qsa-context-sweep-only",
        "repository_commit": repository_commit,
        "donor": DONOR,
        "software": {
            "python": sys.version.split()[0],
            "torch": "unknown",
            "cuda_runtime": "12.6",
            "triton": "unknown",
        },
        "hardware_inventory": {
            "path": str(inventory_path),
            "sha256": _sha256_file(inventory_path),
            "profile_id": profile,
            "gpu_index": gpu_index,
            "gpu_identity": _identity_gpu(inventory_gpu),
        },
        "profile": {
            "attention_backend": "qsa_sparse",
            "selected_path": selected_path,
            "reference_only": selected_path != "triton",
            "topk_backend": topk_backend,
            "default_changed": False,
        },
        "workload": {
            "model": "Qwen3.8-Flash-Next",
            "geometry": "tiny-qwen4-qsa",
            "contexts": contexts,
            "phases": phases,
            "batch_size": 1,
            "repeats": repeats,
            "matrix_complete": matrix_complete,
            "max_duration_seconds": HARD_TIMEOUT_SECONDS,
            "sustained_load": False,
        },
        "workspace_plans": [dict(item) for item in workspace_plans],
        "allocator_checkpoints": [dict(item) for item in allocator_checkpoints],
        "samples": [dict(sample) for sample in samples],
        "telemetry": [dict(row) for row in telemetry],
        "unmeasured": {
            "phases": ["startup_canary", "cancellation_state_restore"],
            "limitations": [
                "tiny one-layer geometry only; no full-model serving claim",
                "one request and one sample per context/phase; no sustained-load "
                "or thermal qualification",
                "startup canary, cancellation, checkpoint restore, truncation, chunked "
                "prefill, and graph capture were not measured",
                "batch construction, host-to-device metadata copies, CPU scalar extraction, "
                "host synchronization, and allocation/copy subcosts outside the four observed "
                "composites were not isolated",
                "workspace plans are advisory accounting only; capacity validation, reservation, "
                "reuse, controlled exhaustion, and backoff were not exercised",
                "the selected Pascal path is the eager Torch FP32 reference and does not "
                "qualify an optimized kernel",
            ],
        },
        "duration_seconds": float(duration_seconds),
        "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
        "summary": {
            "sample_count": len(samples),
            "context_count": len(contexts),
            "phase_count": len(phases),
            "all_samples_passed": all(sample["status"] == "passed" for sample in samples),
            "performance_claim": False,
            "end_to_end_claim": False,
            "thermal_qualification": False,
            "default_changed": False,
        },
        "claims": {
            "qsa_only": True,
            "context_sweep": True,
            "raw_samples": True,
            "phase_timings": True,
            "allocator_high_water": True,
            "performance": False,
            "end_to_end": False,
            "thermal_qualification": False,
            "sustained_load": False,
            "default_changed": False,
        },
        "timing_statistics": phase_stats,
    }


def _validate(document: Mapping[str, Any]) -> None:
    validator = _load_module("validate_qsa_h2_evidence", ROOT / "scripts/validate_evidence.py")
    errors = validator.validate_document(document, schema_dir=ROOT / "schemas")
    if errors:
        raise RuntimeError("generated QSA H2 evidence is invalid: " + "; ".join(errors))


def run_probe(
    *,
    inventory_path: Path,
    output_path: Path,
    expected_profile: str,
    gpu_index: int,
    repository_commit: str,
    selection_path: str = "auto",
) -> dict[str, Any]:
    if gpu_index != 0:
        raise ValueError("qsa-p4 currently requires inventory GPU index 0")
    inventory = _load_inventory(
        inventory_path, expected_profile=expected_profile, gpu_index=gpu_index
    )
    if inventory.get("commit") != repository_commit:
        raise RuntimeError("hardware inventory commit must match measured repository commit")
    import torch
    import triton

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("qsa-p4 requires exactly one visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    if properties.name != "Tesla P4" or (properties.major, properties.minor) != (6, 1):
        raise RuntimeError("qsa-p4 requires a Tesla P4 sm_61 device")
    from freetoken.attention import create_attention_backend

    from tests.models.qwen4_exp.common import Fixture, parsed_config

    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    workspace_plans: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for context in CONTEXTS:
        num_pages = max(64, (context + 1 + 63) // 64 + 4)
        config = dataclasses.replace(
            parsed_config(num_layers=4), qsa_selection_path=selection_path
        )
        fixture = Fixture(config, num_pages=num_pages)
        # Exercise the public registry entry; direct class construction is not the producer path.
        backend = create_attention_backend("qsa_sparse", fixture.config)
        fixture.backend = backend
        fixture.ctx.attn_backend = backend
        attn = fixture.layer(3, seed=3800 + context)
        if context == CONTEXTS[0]:
            checkpoints.append(
                _checkpoint(
                    "post_load", _allocator_snapshot(torch, fixture.device), context_tokens=None
                )
            )
        workspace_plans.extend(_workspace_record(fixture, context, phase) for phase in PHASES)
        request = fixture.req(0, 0, context)
        samples.append(
            _run_forward(
                fixture=fixture,
                attn=attn,
                request=request,
                phase="prefill",
                context_tokens=context,
                repeat=0,
                seed=10000 + context,
            )
        )
        if context == CONTEXTS[0]:
            checkpoints.append(
                _checkpoint(
                    "first_small_prefill",
                    _allocator_snapshot(torch, fixture.device),
                    context_tokens=context,
                )
            )
        if context == CONTEXTS[-1]:
            checkpoints.append(
                _checkpoint(
                    "first_large_prefill",
                    _allocator_snapshot(torch, fixture.device),
                    context_tokens=context,
                )
            )
        fixture.step(request)
        samples.append(
            _run_forward(
                fixture=fixture,
                attn=attn,
                request=request,
                phase="decode",
                context_tokens=context,
                repeat=0,
                seed=20000 + context,
            )
        )
        if context == CONTEXTS[-1]:
            checkpoints.append(
                _checkpoint(
                    "steady_decode",
                    _allocator_snapshot(torch, fixture.device),
                    context_tokens=context,
                )
            )
        fixture.backend.set_phase_observer(None)
        del attn, fixture
        torch.cuda.empty_cache()
    checkpoints.extend(
        [
            _checkpoint(
                "startup_canary",
                None,
                context_tokens=None,
                reason="not part of this context-only sweep",
            ),
            _checkpoint(
                "cancellation_state_restore",
                None,
                context_tokens=None,
                reason="not part of this context-only sweep",
            ),
        ]
    )
    telemetry = [capture_telemetry(gpu_index)]
    duration = time.monotonic() - started
    gpu = next(gpu for gpu in inventory["gpus"] if gpu["index"] == gpu_index)
    if telemetry[0]["uuid"] != gpu["uuid"] or telemetry[0]["pci_bus_id"] != gpu["pci_bus_id"]:
        raise RuntimeError("QSA telemetry identity does not match inventory")
    selected_path = str(samples[0]["phase_events"][0]["path"])
    topk_backend = "torch" if selected_path != "triton" else "triton"
    document = build_evidence(
        inventory=inventory,
        inventory_path=inventory_path,
        telemetry=telemetry,
        samples=samples,
        workspace_plans=workspace_plans,
        allocator_checkpoints=checkpoints,
        selected_path=selected_path,
        topk_backend=topk_backend,
        duration_seconds=duration,
        repository_commit=repository_commit,
    )
    document["software"].update(
        {
            "torch": torch.__version__,
            "cuda_runtime": str(torch.version.cuda),
            "triton": triton.__version__,
        }
    )
    _validate(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded one-P4 QSA context-sweep evidence")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-profile", choices=("ecc-on", "ecc-off"), required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument(
        "--selection-path",
        choices=("auto", "torch-fp32-reference", "torch-fp32-vectorized-reference"),
        default="auto",
        help="Explicit QSA selector for bounded candidate evidence; auto preserves dispatch.",
    )
    args = parser.parse_args(argv)
    args.output.unlink(missing_ok=True)
    args.output.with_name(args.output.name + ".tmp").unlink(missing_ok=True)
    if len(args.repository_commit) != 40 or any(
        char not in "0123456789abcdef" for char in args.repository_commit
    ):
        parser.error("--repository-commit must be a 40-character lowercase commit SHA")
    run_probe(
        inventory_path=args.inventory,
        output_path=args.output,
        expected_profile=args.expected_profile,
        gpu_index=args.gpu_index,
        repository_commit=args.repository_commit,
        selection_path=args.selection_path,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
