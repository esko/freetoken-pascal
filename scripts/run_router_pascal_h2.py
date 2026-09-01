#!/usr/bin/env python3
"""Run the bounded Qwen3.8 arbitrary-K router comparison on one Pascal GPU.

The candidate is deliberately forced for this probe.  Process-wide ``auto``
dispatch remains unchanged and is recorded as the current Torch reference
fallback.  A candidate compile or launch failure is evidence, not a reason to
silently substitute a different implementation.
"""

from __future__ import annotations

import argparse
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_NAME = "qwen38-router-h2-evidence.schema.json"
DONOR = {
    "id": "freetoken-qwen4-pr257-merge",
    "repository": "https://github.com/FlashML-org/FreeToken",
    "pull_request": 257,
    "ref": "bd8f3d519a48777bf22ee5c7c8f58f4f3ff31b40",
    "license": "Apache-2.0",
    "method": "upstream-arbitrary-k-router-reference",
}
DTYPES = ("bfloat16", "float16", "float32")
TOKEN_COUNTS = (1, 2, 4, 8, 32, 128)
RENORMALIZE = (False, True)
REPEATS = 5
NUM_EXPERTS = 512
TOPK = 10
HARD_TIMEOUT_SECONDS = 300.0


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strict_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"unable to read JSON {path}: {error}") from error


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_tensor(value: Any) -> str:
    """Hash tensor bytes only after an explicit synchronized host copy."""
    detached = value.detach().contiguous().cpu()
    return _sha256_bytes(detached.numpy().tobytes())


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_smi_number(value: str, field: str) -> float:
    value = value.strip()
    if value.upper() in {"N/A", "NA", "NOT SUPPORTED"}:
        raise RuntimeError(f"nvidia-smi returned no {field} telemetry")
    try:
        number = float(value)
    except ValueError as error:
        raise RuntimeError(f"invalid nvidia-smi {field} value {value!r}") from error
    if not math.isfinite(number):
        raise RuntimeError(f"nvidia-smi returned non-finite {field} telemetry")
    return number


def capture_telemetry(
    index: int,
    *,
    check_output: Callable[..., str] = subprocess.check_output,
) -> dict[str, Any]:
    """Capture one instantaneous, identity-bearing nvidia-smi row."""
    query = (
        "index,name,uuid,compute_cap,memory.total,pci.bus_id,"
        "clocks.current.graphics,clocks.current.memory,temperature.gpu,"
        "power.draw,power.limit,ecc.mode.current"
    )
    text = check_output(
        [
            "nvidia-smi",
            f"--id={index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = [row for row in text.splitlines() if row.strip()]
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
        temperature_celsius,
        power_watts,
        power_limit_watts,
        ecc_mode,
    ) = fields
    if int(reported_index) != index:
        raise RuntimeError(
            f"nvidia-smi index mismatch: requested {index}, reported {reported_index!r}"
        )
    mode = ecc_mode.lower()
    if mode not in {"enabled", "disabled"}:
        raise RuntimeError(f"unsupported instantaneous ECC mode {ecc_mode!r}")
    return {
        "captured_at": _now(),
        "index": index,
        "name": name,
        "uuid": uuid,
        "compute_capability": compute_capability,
        "memory_mib": int(memory_mib),
        "pci_bus_id": pci_bus_id.lower().replace("00000000:", "0000:"),
        "ecc_mode": mode,
        "clocks": {
            "graphics_mhz": int(_parse_smi_number(graphics_mhz, "graphics clock")),
            "memory_mhz": int(_parse_smi_number(memory_mhz, "memory clock")),
        },
        "temperature_celsius": _parse_smi_number(temperature_celsius, "temperature"),
        "power_watts": _parse_smi_number(power_watts, "power"),
        "power_limit_watts": _parse_smi_number(power_limit_watts, "power limit"),
    }


def _load_inventory(path: Path, *, expected_profile: str, gpu_index: int) -> dict[str, Any]:
    inventory = _strict_json(path)
    checker = _load_module(
        "check_hardware_inventory_for_router_h2", ROOT / "scripts/check_hardware_inventory.py"
    )
    schema = _strict_json(ROOT / "schemas/hardware-inventory.schema.json")
    schema_errors = list(
        checker.Draft202012Validator(schema, format_checker=checker.FORMAT_CHECKER).iter_errors(
            inventory
        )
    )
    errors = [error.message for error in schema_errors]
    errors.extend(
        checker.validate_pascal_inventory(
            inventory, minimum_gpus=1, expected_profile=expected_profile
        )
    )
    if errors:
        raise RuntimeError("hardware inventory is not accepted: " + "; ".join(errors))
    if not isinstance(inventory.get("profile_id"), str):
        raise RuntimeError("router-p4 requires a profiled inventory")
    gpus = inventory.get("gpus", [])
    matches = [gpu for gpu in gpus if isinstance(gpu, Mapping) and gpu.get("index") == gpu_index]
    if len(matches) != 1:
        raise RuntimeError(f"inventory must contain exactly one GPU index {gpu_index}")
    return inventory


def _dtype(torch: Any, name: str) -> Any:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _seeded_logits(torch: Any, *, token_count: int, dtype: Any, seed: int, device: str) -> Any:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        (token_count, NUM_EXPERTS), generator=generator, device=device, dtype=dtype
    )


def _case_input(
    torch: Any,
    *,
    kind: str,
    token_count: int,
    dtype: Any,
    seed: int,
    device: str,
) -> tuple[Any, Any, int | None]:
    if kind == "random":
        logits = _seeded_logits(
            torch, token_count=token_count, dtype=dtype, seed=seed, device=device
        )
        limit = None
    elif kind == "padding":
        logits = _seeded_logits(
            torch, token_count=token_count, dtype=dtype, seed=seed, device=device
        )
        limit = max(0, token_count - max(1, token_count // 4))
    elif kind == "exceptional":
        logits = torch.zeros((token_count, NUM_EXPERTS), device=device, dtype=dtype)
        logits[:, 0] = float("nan")
        logits[:, 1] = float("-inf")
        logits[:, 2] = float("inf")
        logits[:, 3] = float("inf")
        limit = None
    else:  # pragma: no cover - internal call contract
        raise ValueError(f"unknown router case kind {kind!r}")
    hidden = torch.zeros((token_count, 8), device=device, dtype=dtype)
    return hidden, logits, limit


def _limit_tensor(torch: Any, limit: int | None, *, device: str) -> Any:
    if limit is None:
        return None
    return torch.tensor(limit, dtype=torch.int32, device=device)


def _run_router(
    fused_topk: Callable[..., tuple[Any, Any]],
    *,
    hidden: Any,
    logits: Any,
    limit: Any,
    renormalize: bool,
    mode: str,
) -> tuple[Any, Any]:
    return fused_topk(
        hidden,
        logits,
        TOPK,
        renormalize,
        limit,
        router_mode=mode,
        triton_candidate_available=True,
        triton_kernels_available=False,
    )


def _timed_sync(torch: Any, call: Callable[[], tuple[Any, Any]]) -> tuple[tuple[Any, Any], int]:
    started = time.perf_counter_ns()
    value = call()
    torch.cuda.synchronize()
    return value, max(0, time.perf_counter_ns() - started)


def _check_output_contract(torch: Any, output: tuple[Any, Any], *, logits: Any) -> None:
    weights, ids = output
    expected_shape = (logits.shape[0], TOPK)
    if tuple(weights.shape) != expected_shape or tuple(ids.shape) != expected_shape:
        raise RuntimeError(f"router output shape must be {expected_shape}")
    if weights.dtype != torch.float32 or ids.dtype != torch.int32:
        raise RuntimeError("router output must use float32 weights and int32 expert IDs")
    if weights.device != logits.device or ids.device != logits.device:
        raise RuntimeError("router output must remain on the logits device")
    if not weights.is_contiguous() or not ids.is_contiguous():
        raise RuntimeError("router output must be contiguous")


def _failure(error: BaseException, *, phase: str) -> dict[str, str]:
    message = f"{type(error).__name__}: {error}"
    return {"phase": phase, "exception_type": type(error).__name__, "message": message[:2048]}


def _comparison(reference: tuple[Any, Any], candidate: tuple[Any, Any]) -> dict[str, Any]:
    ref_weights, ref_ids = reference
    candidate_weights, candidate_ids = candidate
    ids_exact = bool(torch_equal(ref_ids, candidate_ids))
    weight_error = candidate_weights.float() - ref_weights.float()
    max_abs = float(weight_error.abs().max().item())
    relative_rms = float(
        torch_sqrt(
            (weight_error.square().mean())
            / ref_weights.float().square().mean().clamp_min(1e-30)
        ).item()
    )
    weights_within = bool(max_abs <= 1e-5 and relative_rms <= 1e-5)
    return {
        "ids_exact": ids_exact,
        "max_abs_error": max_abs,
        "relative_rms": relative_rms,
        "atol": 1e-5,
        "rtol": 1e-5,
        "weights_within_tolerance": weights_within,
        "passed": ids_exact and weights_within,
    }


def torch_equal(left: Any, right: Any) -> Any:
    # Kept as a helper so H0 callers can replace it while testing the builder seam.
    import torch

    return torch.equal(left, right)


def torch_sqrt(value: Any) -> Any:
    import torch

    return torch.sqrt(value)


def _timing_sample(
    status: str,
    elapsed_ns: int | None,
    *,
    ids_exact: bool = False,
    weights: bool = False,
) -> dict[str, Any]:
    return {
        "status": status,
        "elapsed_ns": elapsed_ns,
        "synchronized": True,
        "ids_exact": ids_exact,
        "weights_within_tolerance": weights,
    }


def _case(
    *,
    torch: Any,
    fused_topk: Callable[..., tuple[Any, Any]],
    kind: str,
    dtype_name: str,
    token_count: int,
    renormalize: bool,
    seed: int,
    device: str,
    deadline: float,
) -> dict[str, Any]:
    dtype = _dtype(torch, dtype_name)
    hidden, logits, token_limit = _case_input(
        torch,
        kind=kind,
        token_count=token_count,
        dtype=dtype,
        seed=seed,
        device=device,
    )
    limit = _limit_tensor(torch, token_limit, device=device)
    input_sha256 = _sha256_tensor(logits)
    reference, reference_initial_ns = _timed_sync(
        torch,
        lambda: _run_router(
            fused_topk,
            hidden=hidden,
            logits=logits,
            limit=limit,
            renormalize=renormalize,
            mode="torch-reference",
        ),
    )
    _check_output_contract(torch, reference, logits=logits)
    reference_record = {
        "mode": "torch-reference",
        "status": "passed",
        "initial_call_elapsed_ns": reference_initial_ns,
        "may_include_jit": False,
        "ids_sha256": _sha256_tensor(reference[1]),
        "weights_sha256": _sha256_tensor(reference[0]),
    }
    candidate_record: dict[str, Any]
    candidate: tuple[Any, Any] | None = None
    try:
        candidate, candidate_initial_ns = _timed_sync(
            torch,
            lambda: _run_router(
                fused_topk,
                hidden=hidden,
                logits=logits,
                limit=limit,
                renormalize=renormalize,
                mode="triton-candidate",
            ),
        )
        _check_output_contract(torch, candidate, logits=logits)
        candidate_record = {
            "eligible": True,
            "mode": "triton-candidate",
            "status": "passed",
            "initial_call_elapsed_ns": candidate_initial_ns,
            "may_include_jit": True,
            "ids_sha256": _sha256_tensor(candidate[1]),
            "weights_sha256": _sha256_tensor(candidate[0]),
            "failure": None,
        }
        comparison = _comparison(reference, candidate)
    except Exception as error:
        candidate_record = {
            "eligible": True,
            "mode": "triton-candidate",
            "status": "failed",
            "initial_call_elapsed_ns": None,
            "may_include_jit": True,
            "ids_sha256": None,
            "weights_sha256": None,
            "failure": _failure(error, phase="compile_or_launch"),
        }
        comparison = None

    steady_samples: list[dict[str, Any]] = []
    if candidate is None:
        return {
            "id": f"{kind}-{dtype_name}-{token_count}-{'renorm' if renormalize else 'raw'}",
            "kind": kind,
            "dtype": dtype_name,
            "token_count": token_count,
            "renormalize": renormalize,
            "padded": token_limit is not None,
            "num_token_non_padded": token_limit,
            "input_sha256": input_sha256,
            "reference": reference_record,
            "candidate": candidate_record,
            "comparison": comparison,
            "steady_samples": steady_samples,
        }
    for repeat in range(REPEATS):
        if time.monotonic() > deadline:
            raise TimeoutError("router H2 hard timeout exceeded")
        reference_first = repeat % 2 == 0
        order = "reference_then_candidate" if reference_first else "candidate_then_reference"
        observations: dict[str, dict[str, Any]] = {}
        calls = ("reference", "candidate") if reference_first else ("candidate", "reference")
        for name in calls:
            if name == "candidate" and candidate is None:
                observations[name] = _timing_sample("not-run", None)
                continue
            mode = "torch-reference" if name == "reference" else "triton-candidate"
            try:
                value, elapsed_ns = _timed_sync(
                    torch,
                    lambda mode=mode: _run_router(
                        fused_topk,
                        hidden=hidden,
                        logits=logits,
                        limit=limit,
                        renormalize=renormalize,
                        mode=mode,
                    ),
                )
                _check_output_contract(torch, value, logits=logits)
                if name == "reference":
                    expected = reference
                else:
                    expected = candidate
                assert expected is not None
                parity = _comparison(expected, value)
                observations[name] = _timing_sample(
                    "passed",
                    elapsed_ns,
                    ids_exact=parity["ids_exact"],
                    weights=parity["weights_within_tolerance"],
                )
            except Exception as error:
                raise RuntimeError(
                    f"{kind}/{dtype_name}/{token_count}/{renormalize} "
                    f"steady repeat {repeat} {name} failed"
                ) from error
        steady_samples.append(
            {
                "repeat": repeat,
                "order": order,
                "reference": observations["reference"],
                "candidate": observations["candidate"],
            }
        )
    return {
        "id": f"{kind}-{dtype_name}-{token_count}-{'renorm' if renormalize else 'raw'}",
        "kind": kind,
        "dtype": dtype_name,
        "token_count": token_count,
        "renormalize": renormalize,
        "padded": token_limit is not None,
        "num_token_non_padded": token_limit,
        "input_sha256": input_sha256,
        "reference": reference_record,
        "candidate": candidate_record,
        "comparison": comparison,
        "steady_samples": steady_samples,
    }


def _identity_gpu(inventory_gpu: Mapping[str, Any]) -> dict[str, Any]:
    topology = inventory_gpu.get("topology")
    if not isinstance(topology, Mapping):
        raise RuntimeError("inventory GPU lacks measured topology")
    return {
        "index": int(inventory_gpu["index"]),
        "name": inventory_gpu["name"],
        "compute_capability": inventory_gpu["compute_capability"],
        "memory_mib": int(inventory_gpu["memory_mib"]),
        "uuid": inventory_gpu["uuid"],
        "pci_bus_id": inventory_gpu["pci_bus_id"],
        "ecc_mode": inventory_gpu["ecc_mode"],
        "pci_root": topology["pci_root"],
        "numa_node": int(topology["numa_node"]),
    }


def _summary(cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [case for case in cases if case["candidate"]["eligible"]]
    passed = [case for case in eligible if case["candidate"]["status"] == "passed"]
    failed = [case for case in eligible if case["candidate"]["status"] == "failed"]
    not_run = [case for case in cases if case["candidate"]["status"] == "not-run"]
    parity = [case["comparison"] for case in cases if case["comparison"] is not None]
    steady_passed = all(
        sample[implementation]["status"] == "passed"
        and sample[implementation]["ids_exact"]
        and sample[implementation]["weights_within_tolerance"]
        for case in passed
        for sample in case["steady_samples"]
        for implementation in ("reference", "candidate")
    )
    return {
        "case_count": len(cases),
        "reference_case_count": len(cases),
        "candidate_eligible_case_count": len(eligible),
        "candidate_passed_case_count": len(passed),
        "candidate_failed_case_count": len(failed),
        "candidate_not_run_case_count": len(not_run),
        "all_candidate_parity_passed": not failed
        and bool(parity)
        and all(item["passed"] for item in parity)
        and steady_passed,
        "performance_claim": False,
        "end_to_end_claim": False,
        "default_changed": False,
    }


def _validate(document: Mapping[str, Any]) -> None:
    validator = _load_module(
        "validate_evidence_for_router_h2", ROOT / "scripts/validate_evidence.py"
    )
    errors = validator.validate_document(document, schema_dir=ROOT / "schemas")
    if errors:
        raise RuntimeError("generated router evidence is invalid: " + "; ".join(errors))


def run_probe(
    *,
    inventory_path: Path,
    output_path: Path,
    expected_profile: str,
    gpu_index: int,
    repository_commit: str,
) -> dict[str, Any]:
    if gpu_index != 0:
        raise ValueError(
            "router-p4 currently requires inventory GPU index 0 to match the sole "
            "container-visible CUDA device"
        )
    inventory = _load_inventory(
        inventory_path, expected_profile=expected_profile, gpu_index=gpu_index
    )
    if inventory.get("commit") != repository_commit:
        raise RuntimeError("hardware inventory commit must match the measured repository commit")
    import torch
    import triton

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("router-p4 requires exactly one visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    if properties.name != "Tesla P4" or (properties.major, properties.minor) != (6, 1):
        raise RuntimeError("router-p4 requires a Tesla P4 sm_61 device")
    from freetoken.moe.fused import fused_topk
    started = time.monotonic()
    deadline = started + HARD_TIMEOUT_SECONDS
    cases: list[dict[str, Any]] = []
    case_index = 0
    for kind in ("random", "padding", "exceptional"):
        for dtype_name in DTYPES:
            for token_count in TOKEN_COUNTS:
                for renormalize in RENORMALIZE:
                    cases.append(
                        _case(
                            torch=torch,
                            fused_topk=fused_topk,
                            kind=kind,
                            dtype_name=dtype_name,
                            token_count=token_count,
                            renormalize=renormalize,
                            seed=3800 + case_index,
                            device="cuda:0",
                            deadline=deadline,
                        )
                    )
                    case_index += 1
    telemetry = [capture_telemetry(0)]
    duration_seconds = time.monotonic() - started
    if duration_seconds > HARD_TIMEOUT_SECONDS:
        raise TimeoutError("router H2 hard timeout exceeded")
    inventory_gpu = next(gpu for gpu in inventory["gpus"] if gpu["index"] == gpu_index)
    if (
        telemetry[0]["uuid"] != inventory_gpu["uuid"]
        or telemetry[0]["pci_bus_id"] != inventory_gpu["pci_bus_id"]
    ):
        raise RuntimeError("instantaneous router telemetry does not match inventory identity")
    profile_mode = {"ecc-on": "enabled", "ecc-off": "disabled"}[expected_profile]
    if telemetry[0]["ecc_mode"] != profile_mode:
        raise RuntimeError("instantaneous router telemetry does not match ECC profile")
    document = {
        "schema_name": SCHEMA_NAME,
        "schema_version": 1,
        "evidence_status": "measured",
        "evidence_kind": "qwen38-pascal-router-ab",
        "claim_status": "bounded-router-only",
        "repository_commit": repository_commit,
        "donor": DONOR,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "triton": triton.__version__,
        },
        "hardware_inventory": {
            "path": str(inventory_path),
            "sha256": _sha256_file(inventory_path),
            "profile_id": expected_profile,
            "gpu_index": gpu_index,
            "gpu_identity": _identity_gpu(inventory_gpu),
        },
        "workload": {
            "model": "Qwen3.8-Flash-Next",
            "num_experts": NUM_EXPERTS,
            "topk": TOPK,
            "dtypes": list(DTYPES),
            "token_counts": list(TOKEN_COUNTS),
            "renormalize": list(RENORMALIZE),
            "repeats": REPEATS,
            "matrix_complete": True,
            "max_duration_seconds": HARD_TIMEOUT_SECONDS,
        },
        "auto_control": {
            "requested_mode": "auto",
            "selected_implementation": "torch-reference",
            "topk": TOPK,
            "num_experts": NUM_EXPERTS,
            "renormalize": True,
            "fallback_reason": "candidate-not-qualified",
            "default_changed": False,
            "end_to_end_claim": False,
        },
        "cases": cases,
        "telemetry": telemetry,
        "duration_seconds": duration_seconds,
        "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
        "summary": _summary(cases),
        "claims": {
            "router_only": True,
            "exact_ids": all(
                case["comparison"] is not None and case["comparison"]["ids_exact"] for case in cases
            ),
            "weight_tolerance": all(
                case["comparison"] is not None
                and case["comparison"]["weights_within_tolerance"]
                for case in cases
            ),
            "auto_enabled": False,
            "end_to_end_performance": False,
            "thermal_qualification": False,
            "dual_p4_policy": False,
        },
    }
    _validate(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded Pascal Qwen router H2 evidence")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-profile", choices=("ecc-on", "ecc-off"), required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--repository-commit", required=True)
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
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
