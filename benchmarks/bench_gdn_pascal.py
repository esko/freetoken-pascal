#!/usr/bin/env python3
"""Bounded H2 benchmark for the standalone Pascal FP32 GDN recurrence.

The benchmark deliberately measures only the explicit ``pascal-fp32`` recurrence.  It
first compares that candidate with an independent, readable Torch recurrence using the
same non-zero state, and only then records CUDA-event timings.  It is not a release
benchmark: the report is marked thermally constrained until the host cooling and clocks
have been qualified.

The script has no ``nvidia-smi`` dependency.  CUDA, Torch, and device properties are
queried through the Torch CUDA API so it can run in the project's CUDA container.

Example::

    PYTHONPATH=python python benchmarks/bench_gdn_pascal.py \
        --tokens 8 --head-dim 64 --key-heads 1 --value-heads 2 \
        --output results/hardware/gdn-pascal-t8-d64.json
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

DEFAULT_WARMUPS = 2
DEFAULT_REPEATS = 5
# These are deliberately small hard limits.  A benchmark invocation must not turn into
# an accidental thermal soak while the P4 cooling setup is still being tuned.
MAX_WARMUPS = 16
MAX_REPEATS = 32
SUPPORTED_TOKENS = (1, 8, 32)
SUPPORTED_HEAD_DIMS = (64, 128)
MAX_HEADS = 64
QUALIFICATION = "thermally-constrained-non-release"


class BenchmarkConfigError(ValueError):
    """Raised when a benchmark request is outside the bounded H2 contract."""


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Validated shape and sampling parameters for one bounded benchmark report."""

    tokens: int = 1
    head_dim: int = 128
    key_heads: int = 16
    value_heads: int = 48
    warmups: int = DEFAULT_WARMUPS
    repeats: int = DEFAULT_REPEATS
    seed: int = 93
    device: int = 0


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def validate_config(config: BenchmarkConfig) -> BenchmarkConfig:
    """Validate shape, GQA, and bounded timing controls without importing Torch."""

    if config.tokens not in SUPPORTED_TOKENS:
        raise BenchmarkConfigError(f"tokens must be one of {SUPPORTED_TOKENS}, got {config.tokens}")
    if config.head_dim not in SUPPORTED_HEAD_DIMS:
        raise BenchmarkConfigError(
            f"head_dim must be one of {SUPPORTED_HEAD_DIMS}, got {config.head_dim}"
        )
    if config.key_heads <= 0 or config.value_heads <= 0:
        raise BenchmarkConfigError("key_heads and value_heads must be positive")
    if config.key_heads > MAX_HEADS or config.value_heads > MAX_HEADS:
        raise BenchmarkConfigError(f"head counts must not exceed {MAX_HEADS}")
    if config.value_heads % config.key_heads:
        raise BenchmarkConfigError(
            "value_heads must be a positive multiple of key_heads for valid GQA"
        )
    if config.warmups < 0 or config.warmups > MAX_WARMUPS:
        raise BenchmarkConfigError(
            f"warmups must be between 0 and {MAX_WARMUPS}, got {config.warmups}"
        )
    if config.repeats <= 0 or config.repeats > MAX_REPEATS:
        raise BenchmarkConfigError(
            f"repeats must be between 1 and {MAX_REPEATS}, got {config.repeats}"
        )
    if config.device < 0:
        raise BenchmarkConfigError("device must be non-negative")
    return config


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser separately so hosted tests need no CUDA device."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, choices=SUPPORTED_TOKENS, default=1)
    parser.add_argument("--head-dim", type=int, choices=SUPPORTED_HEAD_DIMS, default=128)
    parser.add_argument(
        "--key-heads",
        "--hk",
        dest="key_heads",
        type=_positive_int,
        default=16,
        help="number of key/query heads (HK); value heads must be a GQA multiple",
    )
    parser.add_argument(
        "--value-heads",
        "--hv",
        dest="value_heads",
        type=_positive_int,
        default=48,
        help="number of value heads (HV), configurable as a multiple of HK",
    )
    parser.add_argument(
        "--warmups",
        "--warmup",
        dest="warmups",
        type=_nonnegative_int,
        default=DEFAULT_WARMUPS,
        help=f"untimed warmups per implementation (0..{MAX_WARMUPS}, default {DEFAULT_WARMUPS})",
    )
    parser.add_argument(
        "--repeats",
        type=_positive_int,
        default=DEFAULT_REPEATS,
        help=f"timed samples per implementation (1..{MAX_REPEATS}, default {DEFAULT_REPEATS})",
    )
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--device", type=_nonnegative_int, default=0)
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    return parser


def _git_commit() -> str:
    """Return the exact repository commit used for the report."""

    injected = os.environ.get("FREETOKEN_BENCHMARK_COMMIT")
    if injected is not None:
        if len(injected) != 40 or any(
            character not in "0123456789abcdef" for character in injected
        ):
            raise RuntimeError(
                "FREETOKEN_BENCHMARK_COMMIT must be a 40-character lowercase Git SHA"
            )
        return injected

    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot determine exact Git commit for benchmark report") from error
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError(f"git returned an invalid commit: {commit!r}")
    return commit


def _torch_reference_recurrence(
    torch: Any,
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    state_pool: Any,
    request_slots: list[int],
    request_ranges: list[tuple[int, int]],
    output: Any,
) -> Any:
    """Run the repository's independent upstream-derived Torch GDN oracle.

    The implementation intentionally does not call the Pascal adapter or the runtime's
    model dispatch policy. ``state_pool`` and ``output`` are supplied by the caller so
    reset/allocation work can remain outside timed CUDA events.
    """

    from freetoken.models.qwen4_exp.gdn_reference import recurrent_gated_delta_rule

    if len(request_slots) != len(request_ranges):
        raise BenchmarkConfigError("request slot/range lengths differ")
    ratio = int(v.shape[1]) // int(q.shape[1])
    for request, slot in enumerate(request_slots):
        start, end = request_ranges[request]
        result, final_state = recurrent_gated_delta_rule(
            q[start:end].repeat_interleave(ratio, dim=1).unsqueeze(0),
            k[start:end].repeat_interleave(ratio, dim=1).unsqueeze(0),
            v[start:end].unsqueeze(0),
            g[start:end].unsqueeze(0),
            beta[start:end].unsqueeze(0),
            initial_state=state_pool[slot].unsqueeze(0),
        )
        output[start:end].copy_(result[0])
        state_pool[slot].copy_(final_state[0])
    return output


def _cuda_time(torch: Any, operation: Callable[[], Any]) -> tuple[float, Any]:
    """Time one already-prepared operation with synchronized CUDA events."""

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), result


def _summary(samples: list[dict[str, float]]) -> dict[str, float | int]:
    """Summarize raw event samples without losing the individual observations."""

    values = [float(sample["elapsed_ms"]) for sample in samples]
    if not values:
        raise ValueError("cannot summarize an empty sample list")
    return {
        "count": len(values),
        "median_ms": float(statistics.median(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def _device_metadata(torch: Any, device: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    major, minor = torch.cuda.get_device_capability(device)
    return {
        "index": int(device.index),
        "uuid": f"GPU-{properties.uuid}",
        "pci_address": (
            f"{int(properties.pci_domain_id):04x}:{int(properties.pci_bus_id):02x}:"
            f"{int(properties.pci_device_id):02x}.0"
        ),
        "name": str(properties.name),
        "compute_capability": f"{int(major)}.{int(minor)}",
        "major": int(major),
        "minor": int(minor),
        "total_memory_bytes": int(properties.total_memory),
        "multi_processor_count": int(properties.multi_processor_count),
    }


def run_benchmark(
    config: BenchmarkConfig,
    *,
    command: str | None = None,
    torch_module: Any | None = None,
    candidate: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run one bounded candidate/reference sweep and return a JSON-safe report."""

    config = validate_config(config)
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as error:  # pragma: no cover - target environment concern
            raise RuntimeError("Torch is required for the Pascal GDN benchmark") from error
    torch = torch_module
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required for the Pascal GDN benchmark")
    if config.device >= int(torch.cuda.device_count()):
        raise RuntimeError(
            f"CUDA device {config.device} is unavailable; "
            f"device count is {torch.cuda.device_count()}"
        )
    device = torch.device("cuda", config.device)
    major, minor = torch.cuda.get_device_capability(device)
    if (int(major), int(minor)) != (6, 1):
        raise RuntimeError(f"Pascal GDN benchmark requires sm_61, got sm_{int(major)}{int(minor)}")
    if candidate is None:
        from freetoken.kernel.gdn_pascal import pascal_gdn_recurrence

        candidate = pascal_gdn_recurrence

    generator = torch.Generator(device=device).manual_seed(config.seed)
    q = torch.randn(
        config.tokens, config.key_heads, config.head_dim, device=device, generator=generator
    )
    k = torch.randn(
        config.tokens, config.key_heads, config.head_dim, device=device, generator=generator
    )
    v = torch.randn(
        config.tokens, config.value_heads, config.head_dim, device=device, generator=generator
    )
    g = -torch.rand(config.tokens, config.value_heads, device=device, generator=generator) * 0.2
    beta = torch.sigmoid(
        torch.randn(config.tokens, config.value_heads, device=device, generator=generator)
    )
    # The additive term makes the non-zero-state invariant explicit even in the vanishingly
    # unlikely event that a pseudo-random draw contains only zeros.
    baseline_state = (
        torch.randn(
            1,
            config.value_heads,
            config.head_dim,
            config.head_dim,
            device=device,
            generator=generator,
        )
        * 0.01
        + 0.001
    )
    slot_indices = torch.zeros(1, device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, config.tokens], device=device, dtype=torch.int32)
    request_slots = [0]
    request_ranges = [(0, config.tokens)]

    def candidate_call(state: Any, output: Any) -> Any:
        return candidate(q, k, v, g, beta, state, slot_indices, cu_seqlens, output=output)

    def reference_call(state: Any, output: Any) -> Any:
        return _torch_reference_recurrence(
            torch,
            q,
            k,
            v,
            g,
            beta,
            state,
            request_slots,
            request_ranges,
            output,
        )

    # Correctness is deliberately first.  This also pays the candidate JIT compilation cost
    # before any timing event is recorded.
    candidate_state = baseline_state.clone()
    candidate_output = torch.empty_like(v)
    candidate_call(candidate_state, candidate_output)
    torch.cuda.synchronize()
    reference_state = baseline_state.clone()
    reference_output = torch.empty_like(v)
    reference_call(reference_state, reference_output)
    torch.cuda.synchronize()
    output_abs = float(torch.max(torch.abs(candidate_output - reference_output)).item())
    state_abs = float(torch.max(torch.abs(candidate_state - reference_state)).item())
    output_ref = float(torch.max(torch.abs(reference_output)).item())
    state_ref = float(torch.max(torch.abs(reference_state)).item())
    output_rel = output_abs / max(output_ref, 1.0e-12)
    state_rel = state_abs / max(state_ref, 1.0e-12)
    output_passed = bool(torch.allclose(candidate_output, reference_output, rtol=3e-5, atol=3e-5))
    state_passed = bool(torch.allclose(candidate_state, reference_state, rtol=3e-5, atol=3e-5))
    if not output_passed or not state_passed:
        raise RuntimeError(
            "Pascal GDN correctness failed before timing: "
            f"output_abs={output_abs:.6g}, state_abs={state_abs:.6g}"
        )

    candidate_samples: list[dict[str, float]] = []
    reference_samples: list[dict[str, float]] = []
    for _ in range(config.warmups):
        warmup_state = baseline_state.clone()
        warmup_output = torch.empty_like(v)
        candidate_call(warmup_state, warmup_output)
        torch.cuda.synchronize()
        warmup_state = baseline_state.clone()
        warmup_output = torch.empty_like(v)
        reference_call(warmup_state, warmup_output)
        torch.cuda.synchronize()

    for sample_index in range(config.repeats):
        sample_state = baseline_state.clone()
        sample_output = torch.empty_like(v)
        elapsed_ms, _ = _cuda_time(
            torch,
            lambda state=sample_state, output=sample_output: candidate_call(state, output),
        )
        candidate_samples.append({"index": sample_index, "elapsed_ms": elapsed_ms})

        sample_state = baseline_state.clone()
        sample_output = torch.empty_like(v)
        elapsed_ms, _ = _cuda_time(
            torch,
            lambda state=sample_state, output=sample_output: reference_call(state, output),
        )
        reference_samples.append({"index": sample_index, "elapsed_ms": elapsed_ms})

    command = command or shlex.join([sys.executable, *sys.argv])
    device_info = _device_metadata(torch, device)
    return {
        "schema_name": "pascal-gdn-recurrence-benchmark",
        "schema_version": 1,
        "qualification": QUALIFICATION,
        "workload": {
            "tokens": config.tokens,
            "head_dim": config.head_dim,
            "key_heads": config.key_heads,
            "value_heads": config.value_heads,
            "gqa_ratio": config.value_heads // config.key_heads,
            "requests": 1,
            "state_slots": 1,
            "state_initialization": "identical nonzero state reset before every sample",
            "warmups": config.warmups,
            "repeats": config.repeats,
            "seed": config.seed,
        },
        "selected_behavior": {
            "candidate": "pascal_gdn_recurrence",
            "candidate_backend": "pascal-fp32",
            "candidate_activation": "explicit-only",
            "reference": "torch_gdn_reference",
            "reference_backend": "independent-torch-equation",
            "default_path": "automatic GDN dispatch remains unchanged",
            "fallback_path": "reference fallback remains available",
            "fallback_used": False,
        },
        "correctness": {
            "checked_before_timing": True,
            "passed": True,
            "output_max_abs_error": output_abs,
            "output_max_relative_error": output_rel,
            "state_max_abs_error": state_abs,
            "state_max_relative_error": state_rel,
            "rtol": 3e-5,
            "atol": 3e-5,
        },
        "samples": {
            "candidate": candidate_samples,
            "reference": reference_samples,
        },
        "statistics": {
            "candidate": _summary(candidate_samples),
            "reference": _summary(reference_samples),
        },
        "timing_scope": (
            "CUDA-event interval around each adapter call; includes device work and stream-idle "
            "time caused by synchronous validation/dispatch inside the call, but excludes input "
            "reset and allocation"
        ),
        "metadata": {
            "commit": _git_commit(),
            "command": command,
            "cuda": {
                "torch_cuda_version": str(torch.version.cuda),
                "device_count": int(torch.cuda.device_count()),
                "is_available": bool(torch.cuda.is_available()),
            },
            "torch": {"version": str(torch.__version__)},
            "device": device_info,
            "thermal": {
                "status": "unqualified",
                "reason": "P4 cooling/clocks are thermally constrained; no release claim",
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = BenchmarkConfig(
        tokens=args.tokens,
        head_dim=args.head_dim,
        key_heads=args.key_heads,
        value_heads=args.value_heads,
        warmups=args.warmups,
        repeats=args.repeats,
        seed=args.seed,
        device=args.device,
    )
    try:
        report = run_benchmark(
            config,
            command=shlex.join([sys.executable, sys.argv[0], *(argv or sys.argv[1:])]),
        )
    except (BenchmarkConfigError, RuntimeError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
