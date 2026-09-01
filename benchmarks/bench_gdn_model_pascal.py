#!/usr/bin/env python3
"""Bounded H2 A/B benchmark for one Qwen GDN layer boundary on Pascal.

The benchmark compares the explicit ``pascal-fp32`` model path with the explicit
``torch-reference`` path over the same BF16 Qwen3.8 geometry.  Each case includes
the production projection, causal convolution, gate parameter calculation, recurrent
state update, gated RMSNorm, and output projection.  It checks prefill and decode
outputs and recurrent/conv state before recording CUDA-event and host wall timings.

This is observational, thermally constrained H2 evidence.  It does not enable factory
or ``auto`` dispatch, change model defaults, or qualify a release backend.

Each timed sample also records ``phase_timings`` and aggregate ``phase_statistics`` for
the layer total, projections, width-4 convolution, qkv preparation, gate, recurrence,
norm, and output projection.  Pascal samples additionally expose synchronous metadata
validation and adapter/launch host overhead.  The adapter host interval is intentionally
reported as combined overhead; its CUDA interval overlaps ``recurrence_device`` and is
not an independent device-work bucket.

Example (run inside the CUDA 12.6 image)::

    FREETOKEN_BENCHMARK_COMMIT=$(git rev-parse HEAD) \\
    PYTHONPATH=python python benchmarks/bench_gdn_model_pascal.py \\
        --prefill-tokens 8 --warmups 2 --repeats 5 \\
        --output results/hardware/gdn-model-pascal-t8.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

MODEL_HEAD_DIM = 128
MODEL_KEY_HEADS = 16
MODEL_VALUE_HEADS = 48
MODEL_HIDDEN_SIZE = 2560
MODEL_CONV_KERNEL = 4
MODEL_RMS_EPS = 1e-6
MODEL_LAYER_ID = 0
SUPPORTED_PREFILL_TOKENS = (1, 8, 32)
DEFAULT_WARMUPS = 2
DEFAULT_REPEATS = 5
MAX_WARMUPS = 8
MAX_REPEATS = 16
QUALIFICATION = "thermally-constrained-non-release"
# Match the existing BF16 Qwen GDN model/reference gate. Recurrence-state parity remains much
# tighter because it compares the FP32 state before the BF16 norm/output projection.
MODEL_RTOL = 2e-2
MODEL_ATOL = 2e-2
STATE_RTOL = 3e-5
STATE_ATOL = 3e-5
COMMON_PHASE_NAMES = (
    "projection",
    "convolution",
    "qkv_prepare",
    "gate",
    "recurrence_device",
    "norm",
    "output_projection",
)
OPTIONAL_PHASE_NAMES = ("metadata_validation", "adapter_host_overhead")
PHASE_NAMES = ("layer_total", *COMMON_PHASE_NAMES, *OPTIONAL_PHASE_NAMES)
PHASE_EVENTS = ("begin", "end")


class BenchmarkConfigError(ValueError):
    """Raised when a benchmark request is outside the bounded model contract."""


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Validated controls for one paired prefill/decode model-boundary sweep."""

    prefill_tokens: int = 1
    warmups: int = DEFAULT_WARMUPS
    repeats: int = DEFAULT_REPEATS
    seed: int = 9301
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
    """Validate the fixed Qwen geometry and bounded timing controls without Torch."""

    if config.prefill_tokens not in SUPPORTED_PREFILL_TOKENS:
        raise BenchmarkConfigError(
            f"prefill_tokens must be one of {SUPPORTED_PREFILL_TOKENS}, got {config.prefill_tokens}"
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
    parser.add_argument(
        "--prefill-tokens",
        "--tokens",
        dest="prefill_tokens",
        type=int,
        choices=SUPPORTED_PREFILL_TOKENS,
        default=1,
        help="prefill sequence length; decode is always one token (default: 1)",
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
    parser.add_argument("--seed", type=int, default=9301)
    parser.add_argument("--device", type=_nonnegative_int, default=0)
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    return parser


def _checkout_commit(*, required: bool) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        if required:
            raise RuntimeError("cannot determine exact Git commit for benchmark report") from error
        return None
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError(f"git returned an invalid commit: {commit!r}")
    return commit


def _git_commit() -> str:
    """Return an exact commit, supporting the git-less CUDA container."""

    injected = os.environ.get("FREETOKEN_BENCHMARK_COMMIT")
    if injected is None:
        commit = _checkout_commit(required=True)
        assert commit is not None
        return commit
    if len(injected) != 40 or any(character not in "0123456789abcdef" for character in injected):
        raise RuntimeError("FREETOKEN_BENCHMARK_COMMIT must be a 40-character lowercase Git SHA")
    checkout = _checkout_commit(required=False)
    if checkout is not None and checkout != injected:
        raise RuntimeError(
            "FREETOKEN_BENCHMARK_COMMIT does not match the mounted checkout: "
            f"{injected} != {checkout}"
        )
    return injected


def _device_metadata(torch: Any, device: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    major, minor = torch.cuda.get_device_capability(device)
    uuid = str(properties.uuid)
    if not uuid.startswith("GPU-"):
        uuid = f"GPU-{uuid}"
    return {
        "index": int(device.index),
        "uuid": uuid,
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


def _summary(samples: list[dict[str, float]]) -> dict[str, float | int]:
    """Summarize both timing clocks while retaining raw samples in the report."""

    if not samples:
        raise ValueError("cannot summarize an empty sample list")
    event_values = [float(sample["cuda_event_ms"]) for sample in samples]
    wall_values = [float(sample["host_wall_ms"]) for sample in samples]
    return {
        "count": len(samples),
        "cuda_event_median_ms": float(statistics.median(event_values)),
        "cuda_event_min_ms": float(min(event_values)),
        "cuda_event_max_ms": float(max(event_values)),
        "host_wall_median_ms": float(statistics.median(wall_values)),
        "host_wall_min_ms": float(min(wall_values)),
        "host_wall_max_ms": float(max(wall_values)),
    }


class PhaseTelemetryError(RuntimeError):
    """Raised when the model phase observer reports an invalid or unbalanced interval."""


class _PhaseRecorder:
    """Record CUDA and host intervals emitted by one explicit model-boundary invocation."""

    def __init__(self, torch: Any, device: Any, *, required_phases: tuple[str, ...]) -> None:
        self._torch = torch
        self._stream = torch.cuda.current_stream(device)
        self._required_phases = required_phases
        self._active = False
        self._pending: dict[str, tuple[Any, float]] = {}
        self._completed: dict[str, list[tuple[Any, Any, float, float]]] = {}

    def reset(self) -> None:
        self._active = True
        self._pending.clear()
        self._completed.clear()

    def disable(self) -> None:
        self._active = False
        self._pending.clear()
        self._completed.clear()

    def __call__(self, phase: str, event: str) -> None:
        if not self._active:
            return
        if phase not in PHASE_NAMES:
            raise PhaseTelemetryError(f"unknown GDN model phase: {phase!r}")
        if event not in PHASE_EVENTS:
            raise PhaseTelemetryError(f"unknown GDN phase event: {event!r}")
        if event == "begin":
            if phase in self._pending:
                raise PhaseTelemetryError(f"duplicate begin event for GDN phase {phase!r}")
            start = self._torch.cuda.Event(enable_timing=True)
            start.record(self._stream)
            self._pending[phase] = (start, time.perf_counter())
            return
        pending = self._pending.pop(phase, None)
        if pending is None:
            raise PhaseTelemetryError(f"end event without begin for GDN phase {phase!r}")
        start, host_start = pending
        end = self._torch.cuda.Event(enable_timing=True)
        end.record(self._stream)
        self._completed.setdefault(phase, []).append((start, end, host_start, time.perf_counter()))

    def collect(self) -> dict[str, dict[str, float]]:
        if not self._active:
            raise PhaseTelemetryError("phase recorder is not active")
        if self._pending:
            raise PhaseTelemetryError(f"unclosed GDN phases: {sorted(self._pending)}")
        missing = [phase for phase in self._required_phases if phase not in self._completed]
        if missing:
            raise PhaseTelemetryError(f"missing GDN phases: {missing}")
        result = {}
        for phase in self._completed:
            intervals = self._completed[phase]
            result[phase] = {
                "cuda_event_ms": float(
                    sum(start.elapsed_time(end) for start, end, _begin, _finish in intervals)
                ),
                "host_wall_ms": float(
                    sum((finish - begin) * 1000.0 for _start, _end, begin, finish in intervals)
                ),
            }
        self.disable()
        return result


def _phase_summary(samples: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Summarize each raw phase interval without discarding per-sample observations."""

    phases = {phase for sample in samples for phase in sample["phase_timings"]}

    def values(phase: str, clock: str):
        return [
            float(sample["phase_timings"][phase][clock])
            for sample in samples
            if phase in sample["phase_timings"]
        ]

    return {
        phase: {
            "count": len(values(phase, "cuda_event_ms")),
            "cuda_event_median_ms": float(statistics.median(values(phase, "cuda_event_ms"))),
            "cuda_event_min_ms": float(min(values(phase, "cuda_event_ms"))),
            "cuda_event_max_ms": float(max(values(phase, "cuda_event_ms"))),
            "host_wall_median_ms": float(statistics.median(values(phase, "host_wall_ms"))),
            "host_wall_min_ms": float(min(values(phase, "host_wall_ms"))),
            "host_wall_max_ms": float(max(values(phase, "host_wall_ms"))),
        }
        for phase in PHASE_NAMES
        if phase in phases
    }


def validate_report_format(report: dict[str, Any]) -> None:
    """Validate the stable, JSON-safe shape emitted by this observational benchmark."""

    required = {
        "format_name",
        "format_version",
        "qualification",
        "geometry",
        "workload",
        "selected_behavior",
        "correctness",
        "timings",
        "timing_scope",
        "metadata",
    }
    missing = required.difference(report)
    if missing:
        raise ValueError(f"benchmark report missing required fields: {sorted(missing)}")
    if report["format_name"] != "raw-pascal-gdn-model-boundary-observation":
        raise ValueError("unexpected model-boundary benchmark format name")
    if report["format_version"] != 1:
        raise ValueError("unsupported model-boundary benchmark format version")
    if report["qualification"] != QUALIFICATION:
        raise ValueError("model-boundary benchmark must remain non-release qualified")
    geometry = report["geometry"]
    expected = {
        "hidden_size": MODEL_HIDDEN_SIZE,
        "head_dim": MODEL_HEAD_DIM,
        "key_heads": MODEL_KEY_HEADS,
        "value_heads": MODEL_VALUE_HEADS,
        "gqa_ratio": MODEL_VALUE_HEADS // MODEL_KEY_HEADS,
        "conv_kernel": MODEL_CONV_KERNEL,
        "model_dtype": "bfloat16",
        "recurrent_state_dtype": "float32",
    }
    if geometry != expected:
        raise ValueError(f"unexpected model-boundary geometry: {geometry!r}")
    timings = report["timings"]
    if not isinstance(timings, dict):
        raise ValueError("timings must be a mapping")
    for case in ("prefill", "decode"):
        block = timings.get(case)
        if not isinstance(block, dict):
            raise ValueError(f"timings missing {case} case")
        statistics_by_side = block.get("phase_statistics")
        if not isinstance(statistics_by_side, dict):
            raise ValueError(f"timings {case} missing phase_statistics")
        for side in ("candidate", "reference"):
            samples = block.get(side)
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"timings {case} missing {side} samples")
            required_phases = {"layer_total", *COMMON_PHASE_NAMES}
            if side == "candidate":
                required_phases.update(OPTIONAL_PHASE_NAMES)
            for sample in samples:
                if not isinstance(sample, dict):
                    raise ValueError(f"timings {case} {side} sample must be a mapping")
                phase_timings = sample.get("phase_timings")
                if not isinstance(phase_timings, dict):
                    raise ValueError(f"timings {case} {side} sample missing phase_timings")
                missing_phases = required_phases.difference(phase_timings)
                if missing_phases:
                    raise ValueError(
                        f"timings {case} {side} missing phases: {sorted(missing_phases)}"
                    )
                for phase in required_phases:
                    timing = phase_timings[phase]
                    if not isinstance(timing, dict):
                        raise ValueError(f"timing for phase {phase!r} must be a mapping")
                    for clock in ("cuda_event_ms", "host_wall_ms"):
                        value = float(timing.get(clock, float("nan")))
                        if not math.isfinite(value) or value < 0:
                            raise ValueError(f"invalid {clock} for phase {phase!r}")
            side_statistics = statistics_by_side.get(side)
            if not isinstance(side_statistics, dict):
                raise ValueError(f"timings {case} missing {side} phase statistics")
            missing_statistics = required_phases.difference(side_statistics)
            if missing_statistics:
                raise ValueError(
                    f"timings {case} {side} missing phase statistics: {sorted(missing_statistics)}"
                )
    proof_timings = report.get("metadata_proof_timings")
    if not isinstance(proof_timings, dict):
        raise ValueError("benchmark report missing metadata_proof_timings")
    for case in ("prefill", "decode"):
        block = proof_timings.get(case)
        if not isinstance(block, dict):
            raise ValueError(f"metadata proof timings missing {case} case")
        for phase in (
            "first_cold_proof_construction",
            "allocator_warm_proof_reissue",
            "warm_proof_validation",
        ):
            timing = block.get(phase)
            if not isinstance(timing, dict):
                raise ValueError(f"metadata proof timings {case} missing {phase}")
            samples = timing.get("samples")
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"metadata proof timings {case} {phase} missing samples")
            for sample in samples:
                if not isinstance(sample, dict):
                    raise ValueError(
                        f"metadata proof timing {case} {phase} sample must be a mapping"
                    )
                for clock in ("cuda_event_ms", "host_wall_ms"):
                    value = float(sample.get(clock, float("nan")))
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(
                            f"invalid {clock} for metadata proof timing {case} {phase}"
                        )
            statistics_block = timing.get("statistics")
            if not isinstance(statistics_block, dict):
                raise ValueError(f"metadata proof timings {case} {phase} missing statistics")
    json.dumps(report, allow_nan=False)


def _make_weights(torch: Any, device: Any, seed: int) -> dict[str, Any]:
    """Create one deterministic BF16 checkpoint view shared by both model operators."""

    from freetoken.models.qwen4_exp.gdn_reference import Qwen4ExpGatedDeltaNetReference

    torch.manual_seed(seed)
    source = (
        Qwen4ExpGatedDeltaNetReference(
            hidden_size=MODEL_HIDDEN_SIZE,
            num_k_heads=MODEL_KEY_HEADS,
            num_v_heads=MODEL_VALUE_HEADS,
            head_k_dim=MODEL_HEAD_DIM,
            head_v_dim=MODEL_HEAD_DIM,
            conv_kernel_size=MODEL_CONV_KERNEL,
            rms_norm_eps=MODEL_RMS_EPS,
            output_gate="sigmoid",
        )
        .to(device)
        .float()
        .eval()
    )
    with torch.no_grad():
        source.A_log.uniform_(0.01, 16.0).log_()
        source.dt_bias.uniform_(-1.0, 1.0)
        source.norm.weight.normal_(1.0, 0.1)
    return {
        "in_proj.weight": torch.cat(
            [
                source.in_proj_qkv.weight,
                source.in_proj_z.weight,
                source.in_proj_b.weight,
                source.in_proj_a.weight,
            ],
            dim=0,
        )
        .to(dtype=torch.bfloat16)
        .contiguous(),
        "conv1d.weight": source.conv1d.weight.to(dtype=torch.bfloat16).contiguous(),
        "dt_bias": source.dt_bias.detach().to(dtype=torch.float32).contiguous(),
        "A_log": source.A_log.detach().to(dtype=torch.float32).contiguous(),
        "norm.weight": source.norm.weight.detach().to(dtype=torch.bfloat16).contiguous(),
        "out_proj.weight": source.out_proj.weight.detach().to(dtype=torch.bfloat16).contiguous(),
    }


def _ensure_tp1() -> None:
    """Initialize the standalone process exactly as a TP1 Engine would."""

    from freetoken.distributed import set_tp_info, try_get_tp_info

    current = try_get_tp_info()
    if current is None:
        set_tp_info(rank=0, size=1)
    elif (current.rank, current.size) != (0, 1):
        raise RuntimeError(
            "Pascal GDN model benchmark requires TP1 rank 0, "
            f"got rank={current.rank}, size={current.size}"
        )


def _make_operator(
    torch: Any,
    mode: str,
    weights: dict[str, Any],
    *,
    phase_observer: Callable[[str, str], None] | None = None,
) -> Any:
    """Build a model op with explicit dispatch and no factory/auto selection."""

    from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet
    from freetoken.utils import torch_dtype

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        operator = Qwen4ExpGatedDeltaNet(
            hidden_size=MODEL_HIDDEN_SIZE,
            num_k_heads=MODEL_KEY_HEADS,
            num_v_heads=MODEL_VALUE_HEADS,
            head_k_dim=MODEL_HEAD_DIM,
            head_v_dim=MODEL_HEAD_DIM,
            conv_kernel_size=MODEL_CONV_KERNEL,
            rms_norm_eps=MODEL_RMS_EPS,
            layer_id=MODEL_LAYER_ID,
            output_gate="sigmoid",
            gdn_mode=mode,
            gdn_fla_available=False,
            gdn_candidate_available=False,
            gdn_pascal_available=mode == "pascal-fp32",
            gdn_phase_observer=phase_observer,
        )
    operator.load_state_dict({name: value.clone() for name, value in weights.items()})
    return operator


def _make_context(torch: Any, device: Any) -> Any:
    from freetoken.core import Context
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(MODEL_LAYER_ID,),
        num_key_heads=MODEL_KEY_HEADS,
        num_value_heads=MODEL_VALUE_HEADS,
        key_head_dim=MODEL_HEAD_DIM,
        value_head_dim=MODEL_HEAD_DIM,
        conv_kernel_dim=MODEL_CONV_KERNEL,
        output_gate="sigmoid",
    )
    context = Context(page_size=64)
    context.linear_state_pool = LinearStatePool(
        group, num_slots=4, dtype=torch.bfloat16, device=device, tp_size=1
    )
    if context.linear_state_pool.recurrent_states.dtype != torch.float32:
        raise RuntimeError(
            "model-boundary Pascal benchmark requires an FP32 recurrent state pool; "
            "set FREETOKEN_MAMBA_SSM_DTYPE=fp32"
        )
    return context


def _make_request(torch: Any, token_count: int) -> Any:
    from freetoken.core import Req, SamplingParams

    # cached_len=1 makes the non-zero initial conv/recurrent state an explicit continuation.
    return Req(
        input_ids=torch.zeros(token_count + 1, dtype=torch.int32),
        table_idx=1,
        cached_len=1,
        output_len=2,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )


def _make_batch(
    torch: Any, core: Any, context: Any, device: Any, *, phase: str, tokens: int
) -> Any:
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.core import Batch

    request = _make_request(torch, tokens)
    batch = Batch(reqs=[request], phase=phase)
    batch.padded_reqs = [request]
    if phase == "decode":
        batch.linear_table_idx = torch.tensor([1], dtype=torch.int32, device=device)
        # Mirror scheduler-owned eager metadata. A direct caller that omits this host tuple
        # remains on the cold/unproven validation path and is covered by hosted tests.
        batch.linear_table_idx_host = (1,)
    core._GLOBAL_CTX = context
    batch.fla_metadata = build_fla_metadata(batch, device)
    return batch


def _metadata_validation_mode(batch: Any) -> str:
    """Return the visible Pascal metadata validation mode for benchmark evidence."""

    metadata = getattr(batch, "fla_metadata", None)
    if metadata is None:
        return "unbuilt"
    if getattr(metadata, "pascal_metadata_proof", None) is not None:
        return "scheduler-issued-proof"
    return "synchronous-fallback"


def _set_global_context(core: Any, context: Any) -> None:
    # Direct model-boundary A/B uses two independent state pools.  The public setter intentionally
    # rejects replacement, while this isolated benchmark never runs nested forwards.
    core._GLOBAL_CTX = context


def _invoke(torch: Any, core: Any, context: Any, operator: Any, batch: Any, hidden: Any) -> Any:
    _set_global_context(core, context)
    with torch.inference_mode(), context.forward_batch(batch):
        return operator.forward(hidden)


def _reset_state(context: Any, conv: Any, recurrent: Any) -> None:
    pool = context.linear_state_pool
    pool.conv_states[0].copy_(conv)
    pool.recurrent_states[0].copy_(recurrent)


def _state_snapshot(context: Any) -> tuple[Any, Any]:
    pool = context.linear_state_pool
    return pool.conv_states[0, 1].clone(), pool.recurrent_states[0, 1].clone()


def _compare(
    torch: Any, candidate: Any, reference: Any, *, rtol: float, atol: float
) -> dict[str, Any]:
    difference = torch.abs(candidate.float() - reference.float())
    reference_abs = torch.abs(reference.float())
    return {
        "max_abs_error": float(difference.max().item()),
        "max_relative_error": float((difference / reference_abs.clamp_min(1e-12)).max().item()),
        "passed": bool(torch.allclose(candidate, reference, rtol=rtol, atol=atol)),
        "rtol": rtol,
        "atol": atol,
    }


def _correctness(
    torch: Any,
    core: Any,
    pascal_context: Any,
    reference_context: Any,
    pascal: Any,
    reference: Any,
    prefill_batch_pascal: Any,
    prefill_batch_reference: Any,
    decode_batch_pascal: Any,
    decode_batch_reference: Any,
    prefill_hidden: Any,
    decode_hidden: Any,
    initial_conv: Any,
    initial_recurrent: Any,
) -> tuple[dict[str, Any], tuple[Any, Any], tuple[Any, Any]]:
    """Run both model cases before any timing and return post-prefill state snapshots."""

    _reset_state(pascal_context, initial_conv, initial_recurrent)
    pascal_prefill = _invoke(
        torch, core, pascal_context, pascal, prefill_batch_pascal, prefill_hidden
    )
    torch.cuda.synchronize(prefill_hidden.device)
    pascal_post_prefill = _state_snapshot(pascal_context)

    _reset_state(reference_context, initial_conv, initial_recurrent)
    reference_prefill = _invoke(
        torch, core, reference_context, reference, prefill_batch_reference, prefill_hidden
    )
    torch.cuda.synchronize(prefill_hidden.device)
    reference_post_prefill = _state_snapshot(reference_context)

    prefill_result = {
        "output": _compare(
            torch, pascal_prefill, reference_prefill, rtol=MODEL_RTOL, atol=MODEL_ATOL
        ),
        "conv_state": _compare(
            torch,
            pascal_post_prefill[0],
            reference_post_prefill[0],
            rtol=0.0,
            atol=0.0,
        ),
        "recurrent_state": _compare(
            torch,
            pascal_post_prefill[1],
            reference_post_prefill[1],
            rtol=STATE_RTOL,
            atol=STATE_ATOL,
        ),
    }
    prefill_result["passed"] = all(item["passed"] for item in prefill_result.values())
    if not prefill_result["passed"]:
        raise RuntimeError(f"Pascal model prefill correctness failed: {prefill_result}")

    _reset_state(pascal_context, *pascal_post_prefill)
    pascal_decode = _invoke(torch, core, pascal_context, pascal, decode_batch_pascal, decode_hidden)
    torch.cuda.synchronize(decode_hidden.device)
    pascal_post_decode = _state_snapshot(pascal_context)

    _reset_state(reference_context, *reference_post_prefill)
    reference_decode = _invoke(
        torch, core, reference_context, reference, decode_batch_reference, decode_hidden
    )
    torch.cuda.synchronize(decode_hidden.device)
    reference_post_decode = _state_snapshot(reference_context)

    decode_result = {
        "output": _compare(
            torch, pascal_decode, reference_decode, rtol=MODEL_RTOL, atol=MODEL_ATOL
        ),
        "conv_state": _compare(
            torch,
            pascal_post_decode[0],
            reference_post_decode[0],
            rtol=0.0,
            atol=0.0,
        ),
        "recurrent_state": _compare(
            torch,
            pascal_post_decode[1],
            reference_post_decode[1],
            rtol=STATE_RTOL,
            atol=STATE_ATOL,
        ),
    }
    decode_result["passed"] = all(item["passed"] for item in decode_result.values())
    if not decode_result["passed"]:
        raise RuntimeError(f"Pascal model decode correctness failed: {decode_result}")

    return (
        {"prefill": prefill_result, "decode": decode_result, "passed": True},
        pascal_post_prefill,
        reference_post_prefill,
    )


def _timed_call(torch: Any, device: Any, operation: Callable[[], Any]) -> tuple[float, float]:
    """Return CUDA stream time and synchronized host wall time for one model forward."""

    torch.cuda.synchronize(device)
    start_host = time.perf_counter()
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        result = operation()
        end_event.record(stream)
    end_event.synchronize()
    torch.cuda.synchronize(device)
    host_wall_ms = (time.perf_counter() - start_host) * 1000.0
    cuda_event_ms = float(start_event.elapsed_time(end_event))
    del result
    return cuda_event_ms, host_wall_ms


def _measure_metadata_proof(
    torch: Any,
    device: Any,
    *,
    operator: Any,
    context: Any,
    batch: Any,
    num_tokens: int,
    repeats: int,
) -> dict[str, Any]:
    """Measure first cold construction, allocator-warm reissue, and warm validation."""

    metadata = getattr(batch, "fla_metadata", None)
    if metadata is None:
        raise RuntimeError("metadata proof benchmark requires prebuilt FLA metadata")
    slots = int(context.linear_state_pool.recurrent_states.shape[1])

    def construct() -> None:
        with torch.inference_mode():
            proof = operator._ensure_pascal_metadata_proof(
                metadata,
                device,
                pool=context.linear_state_pool,
                phase=batch.phase,
            )
        if proof is None:
            raise RuntimeError("scheduler metadata did not produce a Pascal proof")

    # Empty the CUDA allocator before the first issue so this sample is genuinely cold rather
    # than merely proof-object cold. Subsequent proof reissues intentionally observe allocator
    # reuse and are reported separately.
    metadata.pascal_metadata_proof = None
    torch.cuda.empty_cache()
    cuda_ms, wall_ms = _timed_call(torch, device, construct)
    first_cold_samples = [{"index": 0, "cuda_event_ms": cuda_ms, "host_wall_ms": wall_ms}]

    allocator_warm_samples: list[dict[str, float | int]] = []
    for index in range(max(1, repeats)):
        metadata.pascal_metadata_proof = None
        cuda_ms, wall_ms = _timed_call(torch, device, construct)
        allocator_warm_samples.append(
            {"index": index, "cuda_event_ms": cuda_ms, "host_wall_ms": wall_ms}
        )

    if getattr(metadata, "pascal_metadata_proof", None) is None:
        raise RuntimeError("cold metadata proof construction did not leave a proof")
    warm_samples: list[dict[str, float | int]] = []
    for index in range(repeats):

        def validate() -> None:
            with torch.inference_mode():
                operator._validate_pascal_metadata(
                    metadata,
                    num_slots=slots,
                    num_tokens=num_tokens,
                    device=device,
                    phase=batch.phase,
                    pool=context.linear_state_pool,
                )

        cuda_ms, wall_ms = _timed_call(torch, device, validate)
        warm_samples.append({"index": index, "cuda_event_ms": cuda_ms, "host_wall_ms": wall_ms})
    return {
        "first_cold_proof_construction": {
            "samples": first_cold_samples,
            "statistics": _summary(first_cold_samples),
        },
        "allocator_warm_proof_reissue": {
            "samples": allocator_warm_samples,
            "statistics": _summary(allocator_warm_samples),
        },
        "warm_proof_validation": {
            "samples": warm_samples,
            "statistics": _summary(warm_samples),
        },
    }


def _time_case(
    torch: Any,
    device: Any,
    *,
    config: BenchmarkConfig,
    case: str,
    pascal_context: Any,
    reference_context: Any,
    pascal: Any,
    reference: Any,
    pascal_batch: Any,
    reference_batch: Any,
    hidden: Any,
    pascal_state: tuple[Any, Any],
    reference_state: tuple[Any, Any],
    pascal_phase_recorder: _PhaseRecorder,
    reference_phase_recorder: _PhaseRecorder,
    core: Any,
) -> dict[str, Any]:
    def invoke(
        context: Any,
        operator: Any,
        batch: Any,
        state: tuple[Any, Any],
        phase_recorder: _PhaseRecorder,
        *,
        record_phases: bool,
    ) -> tuple[float, float, dict[str, dict[str, float]] | None]:
        _reset_state(context, *state)
        if record_phases:
            phase_recorder.reset()
        try:
            cuda_ms, wall_ms = _timed_call(
                torch,
                device,
                lambda: _invoke(torch, core, context, operator, batch, hidden),
            )
            phase_ms = phase_recorder.collect() if record_phases else None
            if phase_ms is not None:
                phase_ms["layer_total"] = {
                    "cuda_event_ms": cuda_ms,
                    "host_wall_ms": wall_ms,
                }
        finally:
            if record_phases:
                phase_recorder.disable()
        return cuda_ms, wall_ms, phase_ms

    for _ in range(config.warmups):
        invoke(
            pascal_context,
            pascal,
            pascal_batch,
            pascal_state,
            pascal_phase_recorder,
            record_phases=False,
        )
        invoke(
            reference_context,
            reference,
            reference_batch,
            reference_state,
            reference_phase_recorder,
            record_phases=False,
        )

    candidate_samples: list[dict[str, Any]] = []
    reference_samples: list[dict[str, Any]] = []
    pair_order: list[str] = []
    for sample_index in range(config.repeats):
        order = "candidate-reference" if sample_index % 2 == 0 else "reference-candidate"
        pair_order.append(order)

        def record_candidate(index: int = sample_index) -> None:
            cuda_ms, wall_ms, phase_ms = invoke(
                pascal_context,
                pascal,
                pascal_batch,
                pascal_state,
                pascal_phase_recorder,
                record_phases=True,
            )
            candidate_samples.append(
                {
                    "index": index,
                    "cuda_event_ms": cuda_ms,
                    "host_wall_ms": wall_ms,
                    "phase_timings": phase_ms,
                }
            )

        def record_reference(index: int = sample_index) -> None:
            cuda_ms, wall_ms, phase_ms = invoke(
                reference_context,
                reference,
                reference_batch,
                reference_state,
                reference_phase_recorder,
                record_phases=True,
            )
            reference_samples.append(
                {
                    "index": index,
                    "cuda_event_ms": cuda_ms,
                    "host_wall_ms": wall_ms,
                    "phase_timings": phase_ms,
                }
            )

        if sample_index % 2 == 0:
            record_candidate()
            record_reference()
        else:
            record_reference()
            record_candidate()
    return {
        "case": case,
        "pair_order": pair_order,
        "candidate": candidate_samples,
        "reference": reference_samples,
        "statistics": {
            "candidate": _summary(candidate_samples),
            "reference": _summary(reference_samples),
        },
        "phase_statistics": {
            "candidate": _phase_summary(candidate_samples),
            "reference": _phase_summary(reference_samples),
        },
    }


def run_benchmark(
    config: BenchmarkConfig,
    *,
    command: str | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Run correctness-first prefill/decode model-boundary timing on one P4."""

    config = validate_config(config)
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as error:  # pragma: no cover - target environment concern
            raise RuntimeError("Torch is required for the Pascal GDN model benchmark") from error
    torch = torch_module
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required for the Pascal GDN model benchmark")
    if config.device >= int(torch.cuda.device_count()):
        raise RuntimeError(
            f"CUDA device {config.device} is unavailable; "
            f"device count is {torch.cuda.device_count()}"
        )
    device = torch.device("cuda", config.device)
    capability = tuple(int(part) for part in torch.cuda.get_device_capability(device))
    if capability != (6, 1):
        raise RuntimeError(
            f"Pascal GDN model benchmark requires sm_61, got sm_{capability[0]}{capability[1]}"
        )

    import freetoken.core as core

    _ensure_tp1()
    pascal_phase_recorder = _PhaseRecorder(
        torch, device, required_phases=(*COMMON_PHASE_NAMES, *OPTIONAL_PHASE_NAMES)
    )
    reference_phase_recorder = _PhaseRecorder(torch, device, required_phases=COMMON_PHASE_NAMES)
    weights = _make_weights(torch, device, config.seed)
    pascal = _make_operator(torch, "pascal-fp32", weights, phase_observer=pascal_phase_recorder)
    reference = _make_operator(torch, "reference", weights, phase_observer=reference_phase_recorder)
    pascal_context = _make_context(torch, device)
    reference_context = _make_context(torch, device)

    generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    prefill_hidden = torch.randn(
        config.prefill_tokens,
        MODEL_HIDDEN_SIZE,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    decode_hidden = torch.randn(
        1, MODEL_HIDDEN_SIZE, device=device, dtype=torch.bfloat16, generator=generator
    )
    initial_conv = (
        torch.randn(
            2 * MODEL_KEY_HEADS * MODEL_HEAD_DIM + MODEL_VALUE_HEADS * MODEL_HEAD_DIM,
            MODEL_CONV_KERNEL - 1,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
        + 0.001
    )
    initial_recurrent = (
        torch.randn(
            MODEL_VALUE_HEADS,
            MODEL_HEAD_DIM,
            MODEL_HEAD_DIM,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
        + 0.001
    )
    _reset_state(pascal_context, initial_conv, initial_recurrent)
    _reset_state(reference_context, initial_conv, initial_recurrent)

    prefill_batch_pascal = _make_batch(
        torch, core, pascal_context, device, phase="prefill", tokens=config.prefill_tokens
    )
    prefill_batch_reference = _make_batch(
        torch, core, reference_context, device, phase="prefill", tokens=config.prefill_tokens
    )
    decode_batch_pascal = _make_batch(
        torch, core, pascal_context, device, phase="decode", tokens=config.prefill_tokens
    )
    decode_batch_reference = _make_batch(
        torch, core, reference_context, device, phase="decode", tokens=config.prefill_tokens
    )

    metadata_proof_timings = {
        "prefill": _measure_metadata_proof(
            torch,
            device,
            operator=pascal,
            context=pascal_context,
            batch=prefill_batch_pascal,
            num_tokens=config.prefill_tokens,
            repeats=config.repeats,
        ),
        "decode": _measure_metadata_proof(
            torch,
            device,
            operator=pascal,
            context=pascal_context,
            batch=decode_batch_pascal,
            num_tokens=1,
            repeats=config.repeats,
        ),
    }

    correctness, pascal_post_prefill, reference_post_prefill = _correctness(
        torch,
        core,
        pascal_context,
        reference_context,
        pascal,
        reference,
        prefill_batch_pascal,
        prefill_batch_reference,
        decode_batch_pascal,
        decode_batch_reference,
        prefill_hidden,
        decode_hidden,
        initial_conv,
        initial_recurrent,
    )
    torch.cuda.synchronize(device)

    prefill_timing = _time_case(
        torch,
        device,
        config=config,
        case="prefill",
        pascal_context=pascal_context,
        reference_context=reference_context,
        pascal=pascal,
        reference=reference,
        pascal_batch=prefill_batch_pascal,
        reference_batch=prefill_batch_reference,
        hidden=prefill_hidden,
        pascal_state=(initial_conv, initial_recurrent),
        reference_state=(initial_conv, initial_recurrent),
        pascal_phase_recorder=pascal_phase_recorder,
        reference_phase_recorder=reference_phase_recorder,
        core=core,
    )
    decode_timing = _time_case(
        torch,
        device,
        config=config,
        case="decode",
        pascal_context=pascal_context,
        reference_context=reference_context,
        pascal=pascal,
        reference=reference,
        pascal_batch=decode_batch_pascal,
        reference_batch=decode_batch_reference,
        hidden=decode_hidden,
        pascal_state=pascal_post_prefill,
        reference_state=reference_post_prefill,
        pascal_phase_recorder=pascal_phase_recorder,
        reference_phase_recorder=reference_phase_recorder,
        core=core,
    )

    command = command or shlex.join([sys.executable, *sys.argv])
    report = {
        "format_name": "raw-pascal-gdn-model-boundary-observation",
        "format_version": 1,
        "qualification": QUALIFICATION,
        "geometry": {
            "hidden_size": MODEL_HIDDEN_SIZE,
            "head_dim": MODEL_HEAD_DIM,
            "key_heads": MODEL_KEY_HEADS,
            "value_heads": MODEL_VALUE_HEADS,
            "gqa_ratio": MODEL_VALUE_HEADS // MODEL_KEY_HEADS,
            "conv_kernel": MODEL_CONV_KERNEL,
            "model_dtype": "bfloat16",
            "recurrent_state_dtype": "float32",
        },
        "workload": {
            "prefill_tokens": config.prefill_tokens,
            "decode_tokens": 1,
            "requests": 1,
            "state_slot": 1,
            "state_initialization": "identical nonzero conv and recurrent state",
            "warmups": config.warmups,
            "repeats": config.repeats,
            "seed": config.seed,
        },
        "selected_behavior": {
            "candidate": "pascal_gdn_model_boundary",
            "candidate_backend": "pascal-fp32",
            "candidate_activation": "explicit-only",
            "reference": "qwen4_exp_model_boundary",
            "reference_backend": "torch-reference",
            "reference_activation": "explicit-only",
            "factory_or_auto_enabled": False,
            "default_path": "automatic GDN dispatch remains unchanged",
            "fallback_path": "torch-reference remains available",
            "metadata_validation": {
                "prefill_candidate": _metadata_validation_mode(prefill_batch_pascal),
                "prefill_reference": _metadata_validation_mode(prefill_batch_reference),
                "decode_candidate": _metadata_validation_mode(decode_batch_pascal),
                "decode_reference": _metadata_validation_mode(decode_batch_reference),
            },
        },
        "correctness": correctness,
        "timings": {"prefill": prefill_timing, "decode": decode_timing},
        "metadata_proof_timings": metadata_proof_timings,
        "timing_scope": (
            "Single-layer model boundary. The device-scoped CUDA-event interval includes GPU "
            "work plus stream-idle time caused by synchronous validation/dispatch; host wall "
            "time also includes Python dispatch. Both exclude input/state reset and batch "
            "construction and include projection, width-4 causal convolution, gate, recurrence, "
            "norm, and output projection. Per-phase CUDA events are emitted by the optional "
            "model observer; whole-call timings include the small event-recording overhead."
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
            "device": _device_metadata(torch, device),
            "thermal": {
                "status": "unqualified",
                "reason": "P4 cooling/clocks are thermally constrained; no release claim",
            },
        },
    }
    validate_report_format(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = BenchmarkConfig(
        prefill_tokens=args.prefill_tokens,
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
    except (BenchmarkConfigError, RuntimeError, ValueError) as error:
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
