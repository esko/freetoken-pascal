#!/usr/bin/env python3
"""Collect raw Qwen3.8 mixed route/thread observations.

This is a synthetic observation harness for the Gorilla-relevant 2560/640
expert geometry.  It retains every raw repetition and selected-kernel
telemetry, but intentionally computes no speedup, threshold, or pass/fail
claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from freetoken.moe.cpu_abi import CpuExpertDescriptor, CpuExpertLayout
from freetoken.moe.ggml_reference import (
    Q5_1_BLOCK_BYTES,
    Q5_K_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES,
)
from freetoken.moe.q4_k import Q4K_BLOCK_BYTES, Q4KExecutor

_FORMATS = {
    "Q4_K": (256, Q4K_BLOCK_BYTES, 12),
    "Q5_K": (256, Q5_K_BLOCK_BYTES, 13),
    "Q5_1": (32, Q5_1_BLOCK_BYTES, 7),
    "Q8_0": (32, Q8_0_BLOCK_BYTES, 8),
}


class _PackedSource:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.range_offset = 0
        self.range_size = int(values.nbytes)
        self.source_address = int(values.__array_interface__["data"][0])

    def expert_packed(self, expert: int) -> np.ndarray:
        return self.values[expert]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(_positive_int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def _packed_values(
    experts: int, output_dim: int, input_dim: int, quant_name: str, *, seed: int
) -> _PackedSource:
    block_elements, block_bytes, _ = _FORMATS[quant_name]
    if input_dim % block_elements:
        raise ValueError(f"{quant_name} input_dim must be a multiple of {block_elements}")
    blocks = input_dim // block_elements
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 256, size=(experts, output_dim, blocks * block_bytes), dtype=np.uint8)
    scale = np.frombuffer(np.asarray(np.float16(0.125), dtype="<f2").tobytes(), dtype=np.uint8)
    values[..., :2] = scale
    if quant_name in {"Q4_K", "Q5_K"}:
        values[..., 2:4] = scale
    elif quant_name == "Q5_1":
        values[..., 2:4] = scale
    return _PackedSource(np.ascontiguousarray(values))


def _layout(
    *, experts: int, profile: str, seed: int, hidden_size: int, intermediate_size: int
) -> CpuExpertLayout:
    if profile not in {"normal", "promoted"}:
        raise ValueError(f"unknown profile {profile!r}")
    gate_up = "Q5_K" if profile == "promoted" else "Q4_K"
    down = "Q8_0" if profile == "promoted" else "Q5_1"
    shapes = {
        "gate": (intermediate_size, hidden_size),
        "up": (intermediate_size, hidden_size),
        "down": (hidden_size, intermediate_size),
    }
    descriptors = []
    for index, (projection, (output_dim, input_dim)) in enumerate(shapes.items()):
        quant_name = gate_up if projection in {"gate", "up"} else down
        block_elements, block_bytes, quant_type = _FORMATS[quant_name]
        source = _packed_values(experts, output_dim, input_dim, quant_name, seed=seed + index)
        row_stride = input_dim // block_elements * block_bytes
        descriptors.append(
            CpuExpertDescriptor(
                layer_id=2 if profile == "promoted" else 0,
                projection=projection,
                quant_type=quant_type,
                quant_name=quant_name,
                num_experts=experts,
                output_dim=output_dim,
                input_dim=input_dim,
                rows_per_expert=output_dim,
                row_stride_bytes=row_stride,
                expert_stride_bytes=output_dim * row_stride,
                tensor_bytes=experts * output_dim * row_stride,
                source=source,
            )
        )
    return CpuExpertLayout(tuple(descriptors), top_k=10)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40


def collect(args: argparse.Namespace) -> dict[str, object]:
    if max(args.route_counts) > 10:
        raise ValueError("route counts must be at most 10")
    if args.experts < 1:
        raise ValueError("experts must be positive")
    layout = _layout(
        experts=args.experts,
        profile=args.profile,
        seed=args.seed,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
    )
    layer_id = layout.layers[0]
    executor = Q4KExecutor(
        layout,
        mode=args.mode,
        num_threads=max(args.thread_counts),
        required_alignment=args.required_alignment,
    )
    try:
        executor.prepare(args.tokens, max(10, max(args.route_counts)))
        rng = np.random.default_rng(args.seed + 10_000)
        hidden = rng.normal(size=(args.tokens, args.hidden_size)).astype(np.float32)
        expert_ids = rng.integers(0, args.experts, size=(args.tokens, 10), dtype=np.int32)
        routing_weights = rng.normal(size=(args.tokens, 10)).astype(np.float32)
        executed_threads = args.thread_counts if executor.parallel_enabled else (1,)
        benchmark_args: dict[str, object] = {
            "repeats": args.repeats,
            "route_counts": args.route_counts,
        }
        if executor.parallel_enabled:
            benchmark_args["thread_counts"] = executed_threads
        samples = executor.microbenchmark(
            layer_id, hidden, expert_ids, routing_weights, **benchmark_args
        )
        return {
            "schema_name": "qwen38-mixed-threaded-raw-samples",
            "schema_version": 1,
            "evidence_status": "synthetic",
            "claim_status": "observation_only",
            "commit": _git_commit(),
            "command": " ".join(sys.argv),
            "workload": {
                "profile": args.profile,
                "seed": args.seed,
                "tokens": args.tokens,
                "experts": args.experts,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "route_counts_requested": list(args.route_counts),
                "thread_counts_requested": list(args.thread_counts),
                "thread_counts_executed": list(executed_threads),
                "repeats": args.repeats,
            },
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "selected_behavior": {
                "parallel_enabled": executor.parallel_enabled,
                "backend": executor.backend,
                "kernel_census": list(executor._kernel_census(layer_id)),
                "required_alignment": args.required_alignment,
            },
            "raw_samples": [
                {**sample.as_dict(), "thread_count": sample.thread_count or 1} for sample in samples
            ],
        }
    finally:
        executor.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "scalar", "avx2"), default="auto")
    parser.add_argument("--profile", choices=("normal", "promoted"), default="normal")
    parser.add_argument("--experts", type=_positive_int, default=10)
    parser.add_argument("--tokens", type=_positive_int, default=4)
    parser.add_argument("--hidden-size", type=_positive_int, default=2560)
    parser.add_argument("--intermediate-size", type=_positive_int, default=640)
    parser.add_argument("--seed", type=int, default=3815)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument(
        "--route-counts",
        type=_csv_ints,
        default=(1, 2, 4, 8, 10),
        help="comma-separated route widths (default: 1,2,4,8,10)",
    )
    parser.add_argument(
        "--thread-counts",
        type=_csv_ints,
        default=(1, 2, 4, 8, 16, 24),
        help="comma-separated worker counts (default: 1,2,4,8,16,24)",
    )
    parser.add_argument(
        "--required-alignment",
        type=_positive_int,
        default=1,
        help="source alignment gate in bytes (default: 1 for synthetic observations)",
    )
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = collect(args)
    except (ValueError, RuntimeError) as error:
        _parser().error(str(error))
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
