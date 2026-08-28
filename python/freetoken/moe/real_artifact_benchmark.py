"""Target-CPU benchmark for one bounded real Qwen3.8 expert.

This is deliberately a small H0 benchmark, not a serving benchmark.  It uses the
range/cache/layout hand-off from :mod:`real_artifact_probe`, fixes the workload to one
token and one selected route, and compares packed native execution with an independent
``gguf-py`` dequantize-plus-FP32 dense reference.  The reference dequantizes on every
timed repetition; native setup and range fetches happen before timing.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from freetoken.gguf_census import model_sha256 as canonical_model_sha256
from freetoken.moe import real_artifact_probe as probe
from freetoken.moe.real_artifact_probe import (
    DEFAULT_EXPERT,
    DEFAULT_LAYER,
    DEFAULT_REPEATS,
    DEFAULT_SEED,
    DEFAULT_VARIANT,
    ArtifactProbeError,
    RangeResponse,
    RangeTransport,
)

BENCHMARK_SCHEMA = "qwen38-real-artifact-target-cpu-benchmark.schema.json"
BENCHMARK_SCHEMA_VERSION = 1
SUPPORTED_LAYERS = frozenset({0, 2})
MIN_WARMUP = 5
_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _git_commit() -> str:
    """Return the exact repository commit, or fail closed if it is unavailable."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArtifactProbeError("target CPU benchmark cannot determine git commit") from error
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ArtifactProbeError(f"target CPU benchmark got invalid git commit {commit!r}")
    return commit


def _cpu_flags() -> tuple[str, ...]:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="ascii", errors="ignore")
    except OSError:
        return ()
    for line in text.splitlines():
        if line.lower().startswith("flags") or line.lower().startswith("features"):
            _, _, value = line.partition(":")
            return tuple(sorted(set(value.split())))
    return ()


def _host_metadata() -> dict[str, Any]:
    """Capture CPU and ISA facts needed to interpret a target-host sample."""
    flags = _cpu_flags()
    isa = tuple(name for name in ("avx2", "fma") if name in flags)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "flags": list(flags),
        "isa": list(isa),
    }


def _blas_metadata() -> dict[str, Any]:
    """Capture BLAS thread controls and the process CPU affinity.

    Every supported BLAS thread control must be explicitly set to one.  An unset
    variable is just as unreproducible as a value greater than one because a linked
    BLAS may otherwise create an uncontrolled worker pool.
    """
    values = {name: os.environ.get(name) for name in _BLAS_THREAD_VARS}
    invalid = {name: value for name, value in values.items() if value != "1"}
    if invalid:
        rendered = ", ".join(f"{name}={value!r}" for name, value in invalid.items())
        raise ArtifactProbeError(
            "target CPU benchmark requires BLAS thread variables to be 1; "
            f"non-single-thread settings: {rendered}"
        )
    try:
        affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {
        "thread_env": values,
        "all_values_are_one": not invalid,
        "process_affinity": affinity,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ArtifactProbeError(f"cannot hash file {path}: {error}") from error
    return digest.hexdigest()


def _native_library_metadata(build_metadata_path: Path | None) -> dict[str, Any]:
    """Capture native helper paths/hashes and require matching build metadata."""
    if build_metadata_path is None:
        raise ArtifactProbeError("native build metadata is required for measured evidence")
    libraries: dict[str, Any] = {}
    for name, env_name in (
        ("q4_k", "FREETOKEN_Q4K_NATIVE_LIB"),
        ("mixed_gemv", "FREETOKEN_MIXED_GEMV_NATIVE_LIB"),
    ):
        raw_path = os.environ.get(env_name)
        if not raw_path:
            raise ArtifactProbeError(f"{env_name} must name the measured native helper library")
        path = Path(raw_path)
        if not path.is_file():
            raise ArtifactProbeError(f"native helper {env_name} does not name a file: {path}")
        entry: dict[str, Any] = {
            "environment": env_name,
            "path": raw_path,
            "sha256": _sha256_file(path),
        }
        libraries[name] = entry
    try:
        build = json.loads(Path(build_metadata_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ArtifactProbeError(
            f"cannot read native build metadata {build_metadata_path}: {error}"
        ) from error
    if not isinstance(build, dict) or build.get("commit") != _git_commit():
        raise ArtifactProbeError("native build metadata commit must match the benchmark commit")
    if (
        build.get("schema_name") != "qwen38-target-cpu-native-build"
        or build.get("schema_version") != 1
    ):
        raise ArtifactProbeError("native build metadata has an unsupported schema identity")
    compiler = build.get("compiler")
    compile_flags = build.get("compile_flags")
    if (
        not isinstance(compiler, dict)
        or not all(compiler.get(key) for key in ("command", "version"))
        or not isinstance(compile_flags, dict)
        or not all(
            isinstance(compile_flags.get(key), list) for key in ("common", "baseline", "avx2")
        )
    ):
        raise ArtifactProbeError("native build metadata omits compiler or compile flags")
    for name, library in libraries.items():
        try:
            built_library = build["libraries"][name]
            built_sha256 = built_library["sha256"]
            built_path = built_library["path"]
        except (KeyError, TypeError) as error:
            raise ArtifactProbeError(
                f"native build metadata omits {name} library identity"
            ) from error
        if built_sha256 != library["sha256"]:
            raise ArtifactProbeError(f"native build metadata hash mismatch for {name}")
        if Path(str(built_path)).resolve() != Path(str(library["path"])).resolve():
            raise ArtifactProbeError(f"native build metadata path mismatch for {name}")
    return {"libraries": libraries, "build": build}


def _census_identity(census: Mapping[str, Any]) -> dict[str, str]:
    """Return the census-pinned model identity without overstating partial verification."""
    model_sha256 = census.get("model_sha256")
    if (
        not isinstance(model_sha256, str)
        or len(model_sha256) != 64
        or any(character not in "0123456789abcdef" for character in model_sha256)
    ):
        raise ArtifactProbeError("census must declare a lowercase 64-character model_sha256")
    shards = census.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ArtifactProbeError("census must contain shard identities")
    try:
        calculated = canonical_model_sha256(shards)
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactProbeError(f"census has invalid shard identities: {error}") from error
    if calculated != model_sha256:
        raise ArtifactProbeError("census model_sha256 does not match canonical shard identities")
    return {"model_sha256": model_sha256, "model_sha256_status": "declared"}


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _dense_swiglu(
    dense: Mapping[str, np.ndarray],
    hidden: np.ndarray,
    expert_ids: np.ndarray,
    routing_weights: np.ndarray,
    active_tokens: int,
) -> np.ndarray:
    """Compute the independent FP32 dense expert operation for validated arrays."""
    gate = dense["gate"]
    up = dense["up"]
    down = dense["down"]
    output = np.zeros((hidden.shape[0], gate.shape[1]), dtype=np.float32)
    activated = np.empty(gate.shape[0], dtype=np.float32)
    for token in range(active_tokens):
        for route in range(expert_ids.shape[1]):
            expert = int(expert_ids[token, route])
            if expert == -1:
                continue
            # The selected range is remapped to expert zero.  Keeping the route loop
            # here preserves the ABI's duplicate-route and weight-accumulation rules.
            gate_values = np.matmul(gate, hidden[token]).astype(np.float32, copy=False)
            up_values = np.matmul(up, hidden[token]).astype(np.float32, copy=False)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                activated[:] = gate_values / (1.0 + np.exp(-gate_values))
            np.multiply(activated, up_values, out=activated)
            contribution = np.matmul(down, activated).astype(np.float32, copy=False)
            np.multiply(contribution, np.float32(routing_weights[token, route]), out=contribution)
            np.add(output[token], contribution, out=output[token])
    return output


def _timed_cold_reference(
    layout: Any,
    fetched_sources: Mapping[str, bytes],
    gguf: Any,
    hidden: np.ndarray,
    expert_ids: np.ndarray,
    routing_weights: np.ndarray,
    active_tokens: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one reference observation, timing dequantization and dense execution together."""
    started = time.perf_counter_ns()
    dequant_started = started
    dense, packed_hashes = probe._oracle_dense_projections(layout, fetched_sources, gguf)
    dequant_elapsed = time.perf_counter_ns() - dequant_started
    dense_started = time.perf_counter_ns()
    output = _dense_swiglu(dense, hidden, expert_ids, routing_weights, active_tokens)
    dense_elapsed = time.perf_counter_ns() - dense_started
    elapsed = time.perf_counter_ns() - started
    return output, {
        "elapsed_ns": int(elapsed),
        "source_validation_and_dequant_elapsed_ns": int(dequant_elapsed),
        "dense_elapsed_ns": int(dense_elapsed),
        "dequantized": True,
        "timed_components": [
            "range/source validation",
            "packed bytes/view setup",
            "sha256 hashing",
            "gguf.dequantize",
            "dense_fp32_swiglu",
        ],
        "packed_source_sha256": packed_hashes,
        "dense_projection_sha256": {
            projection: _hash_array(dense[projection]) for projection in ("gate", "up", "down")
        },
        "output_sha256": _hash_array(output),
    }


def _timed_resident_reference(
    dense: Mapping[str, np.ndarray],
    hidden: np.ndarray,
    expert_ids: np.ndarray,
    routing_weights: np.ndarray,
    active_tokens: int,
    dense_projection_hashes: Mapping[str, str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the resident dense reference after dequantization has been prepared."""
    started = time.perf_counter_ns()
    output = _dense_swiglu(dense, hidden, expert_ids, routing_weights, active_tokens)
    elapsed = time.perf_counter_ns() - started
    return output, {
        "elapsed_ns": int(elapsed),
        "dequantized": False,
        "timed_components": ["dense_fp32_swiglu"],
        "dense_projection_sha256": dict(dense_projection_hashes),
        "output_sha256": _hash_array(output),
    }


def _telemetry_dict(telemetry: Any) -> dict[str, Any]:
    rendered = telemetry.as_dict()
    if not isinstance(rendered, dict):
        raise ArtifactProbeError("native executor telemetry must serialize as an object")
    return dict(rendered)


def _native_observation(
    executor: Any,
    layer: int,
    hidden: np.ndarray,
    expert_ids: np.ndarray,
    routing_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Time only the executor call after its output and workspace are prepared."""
    started = time.perf_counter_ns()
    result = executor.execute(layer, hidden, expert_ids, routing_weights)
    elapsed = time.perf_counter_ns() - started
    telemetry = _telemetry_dict(result.telemetry)
    return np.array(result.output, dtype=np.float32, copy=True), {
        "elapsed_ns": int(elapsed),
        "timed_operation": "executor.execute",
        "output_sha256": _hash_array(np.asarray(result.output, dtype=np.float32)),
        "telemetry": telemetry,
    }


def _assert_native_selection(executor: Any, layer: int) -> dict[str, Any]:
    """Reject a forced-AVX2 request that selected a scalar/reference path."""
    primitive = executor.primitive
    mixed = executor.mixed_primitive
    kernels = tuple(executor._kernel_census(layer))
    backend = str(executor._backend_for(layer))
    if primitive.isa != "avx2" or mixed.isa != "avx2":
        reasons = [item for item in (primitive.fallback_reason, mixed.fallback_reason) if item]
        reason = ", ".join(reasons) or "AVX2 primitive was not selected"
        raise ArtifactProbeError(f"native AVX2 benchmark unavailable: {reason}")
    if "avx2" not in backend or not kernels or any("avx2" not in kernel for kernel in kernels):
        raise ArtifactProbeError(
            "native AVX2 benchmark unavailable: selected backend/kernel census is "
            f"backend={backend!r}, kernels={kernels!r}"
        )
    return {
        "requested_mode": "forced_avx2",
        "backend": backend,
        "kernel_census": list(kernels),
        "q4k_isa": primitive.isa,
        "q4k_fallback_reason": primitive.fallback_reason,
        "mixed_isa": mixed.isa,
        "mixed_fallback_reason": mixed.fallback_reason,
    }


def _assert_native_telemetry(telemetry: Mapping[str, Any]) -> None:
    backend = str(telemetry.get("backend", ""))
    kernels = tuple(str(item) for item in telemetry.get("kernel_census", ()))
    fallback = telemetry.get("fallback_reason")
    if "avx2" not in backend or not kernels or any("avx2" not in item for item in kernels):
        raise ArtifactProbeError(
            "native AVX2 benchmark selected a non-native executor observation: "
            f"backend={backend!r}, kernels={kernels!r}"
        )
    if fallback:
        raise ArtifactProbeError(f"native AVX2 benchmark reported fallback: {fallback}")


def _stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    values = [int(sample["elapsed_ns"]) for sample in samples]
    if not values or any(value <= 0 for value in values):
        raise ArtifactProbeError("benchmark timings must be positive")
    mean = statistics.fmean(values)
    return {
        "sample_count": len(values),
        "median_elapsed_ns": statistics.median(values),
        "min_elapsed_ns": min(values),
        "max_elapsed_ns": max(values),
        "coefficient_of_variation": (statistics.pstdev(values) / mean if mean else None),
    }


def benchmark_qwen38_expert(
    *,
    manifest_path: Path,
    census_path: Path,
    variant: str = DEFAULT_VARIANT,
    layer: int = DEFAULT_LAYER,
    expert: int = DEFAULT_EXPERT,
    repeats: int = DEFAULT_REPEATS,
    warmup: int = MIN_WARMUP,
    seed: int = DEFAULT_SEED,
    transport: RangeTransport | Callable[[str, int, int], RangeResponse] | None = None,
    cache_dir: Path | None = None,
    offline: bool = False,
    command: str | None = None,
    native_build_metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Benchmark one selected real expert with strict native/correctness gates."""
    if layer not in SUPPORTED_LAYERS:
        raise ArtifactProbeError(
            f"target CPU benchmark supports actual Qwen3.8 layers {sorted(SUPPORTED_LAYERS)}, "
            f"got {layer}"
        )
    if repeats <= 0 or warmup < MIN_WARMUP:
        raise ArtifactProbeError(
            f"repeats must be positive and warmup must be at least {MIN_WARMUP}"
        )

    blas = _blas_metadata()
    native_metadata = _native_library_metadata(native_build_metadata_path)

    # Resolve the oracle and commit before fetching bytes, matching the bounded probe's
    # fail-early behavior.  Setup, metadata checks and range fetches are never timed.
    gguf = probe._load_gguf_oracle()
    artifact = probe.load_qwen38_expert_artifact(
        manifest_path=Path(manifest_path),
        census_path=Path(census_path),
        variant=variant,
        layer=layer,
        expert=expert,
        transport=transport,
        cache_dir=cache_dir,
        offline=offline,
    )
    layout = artifact["layout"]
    sources = artifact["sources"]
    census_identity = _census_identity(artifact["census"])
    census_sha256 = str(artifact.get("census_sha256", ""))
    if len(census_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in census_sha256
    ):
        raise ArtifactProbeError("artifact loader must provide the exact census file SHA-256")
    hidden_rng = np.random.default_rng(seed)
    hidden = hidden_rng.standard_normal((1, layout.descriptor(layer, "gate").input_dim)).astype(
        np.float32
    )
    expert_ids = np.array([[0]], dtype=np.int32)
    routing_weights = np.array([[1.0]], dtype=np.float32)
    hidden, expert_ids, routing_weights, active_tokens = probe._validate_oracle_arrays(
        layout, hidden, expert_ids, routing_weights, 1
    )
    if active_tokens != 1:
        raise ArtifactProbeError("target CPU benchmark requires exactly one active token")

    executor = probe.Q4KExecutor(layout, mode="forced_avx2", required_alignment=32)
    warmup_cold: list[dict[str, Any]] = []
    warmup_resident: list[dict[str, Any]] = []
    warmup_native: list[dict[str, Any]] = []
    cold_samples: list[dict[str, Any]] = []
    resident_samples: list[dict[str, Any]] = []
    native_samples: list[dict[str, Any]] = []
    cold_outputs: list[np.ndarray] = []
    resident_outputs: list[np.ndarray] = []
    native_outputs: list[np.ndarray] = []
    try:
        # Prepare is intentionally before the first native timer boundary.
        executor.prepare(1, 1)
        selected_behavior = _assert_native_selection(executor, layer)

        # Dense-resident setup is outside all reference timers.  The cold reference
        # repeats this exact independent dequantization inside every sample timer.
        resident_setup_started = time.perf_counter_ns()
        resident_dense, packed_hashes = probe._oracle_dense_projections(layout, sources, gguf)
        resident_setup_elapsed = time.perf_counter_ns() - resident_setup_started
        resident_dense_hashes = {
            projection: _hash_array(resident_dense[projection])
            for projection in ("gate", "up", "down")
        }
        for _ in range(warmup):
            cold_output, observation = _timed_cold_reference(
                layout,
                sources,
                gguf,
                hidden,
                expert_ids,
                routing_weights,
                active_tokens,
            )
            warmup_cold.append(observation)
            resident_output, observation = _timed_resident_reference(
                resident_dense,
                hidden,
                expert_ids,
                routing_weights,
                active_tokens,
                resident_dense_hashes,
            )
            warmup_resident.append(observation)
            native_output, observation = _native_observation(
                executor, layer, hidden, expert_ids, routing_weights
            )
            _assert_native_telemetry(observation["telemetry"])
            warmup_native.append(observation)
            # Warmup outputs are retained as hashes and still act as smoke checks.
            for reference_output, reference_name in (
                (cold_output, "cold gguf-py oracle"),
                (resident_output, "dense-resident gguf-py oracle"),
            ):
                comparison = probe._compare_outputs(
                    reference_output,
                    native_output,
                    expected_name=reference_name,
                    actual_name="native executor",
                )
                if not comparison["correct"]:
                    raise ArtifactProbeError(
                        "correctness mismatch during benchmark warmup: " + str(comparison)
                    )

        for _ in range(repeats):
            cold_output, observation = _timed_cold_reference(
                layout,
                sources,
                gguf,
                hidden,
                expert_ids,
                routing_weights,
                active_tokens,
            )
            cold_samples.append(observation)
            cold_outputs.append(cold_output)
            resident_output, observation = _timed_resident_reference(
                resident_dense,
                hidden,
                expert_ids,
                routing_weights,
                active_tokens,
                resident_dense_hashes,
            )
            resident_samples.append(observation)
            resident_outputs.append(resident_output)
            native_output, observation = _native_observation(
                executor, layer, hidden, expert_ids, routing_weights
            )
            _assert_native_telemetry(observation["telemetry"])
            native_samples.append(observation)
            native_outputs.append(native_output)
    finally:
        executor.close()

    cold_comparisons = [
        probe._compare_outputs(
            expected,
            actual,
            expected_name="cold gguf-py oracle",
            actual_name="native executor",
        )
        for expected, actual in zip(cold_outputs, native_outputs, strict=True)
    ]
    resident_comparisons = [
        probe._compare_outputs(
            expected,
            actual,
            expected_name="dense-resident gguf-py oracle",
            actual_name="native executor",
        )
        for expected, actual in zip(resident_outputs, native_outputs, strict=True)
    ]
    if (
        not cold_comparisons
        or not all(item["correct"] for item in cold_comparisons)
        or not resident_comparisons
        or not all(item["correct"] for item in resident_comparisons)
    ):
        raise ArtifactProbeError(
            "correctness mismatch between native AVX2 and gguf-py references: "
            f"cold={cold_comparisons}, dense_resident={resident_comparisons}"
        )

    cold_stats = _stats(cold_samples)
    resident_stats = _stats(resident_samples)
    native_stats = _stats(native_samples)
    native_median = float(native_stats["median_elapsed_ns"])
    if native_median <= 0:
        raise ArtifactProbeError("native median timing is not positive")
    ranges = artifact["ranges"]
    range_hashes = {str(item["projection"]): str(item["sha256"]) for item in ranges}
    selected_behavior["fallbacks"] = {
        "q4k": selected_behavior["q4k_fallback_reason"],
        "mixed": selected_behavior["mixed_fallback_reason"],
    }
    return {
        "schema_name": BENCHMARK_SCHEMA,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "evidence_status": "measured",
        "claim_status": "correctness_passed",
        "validation_class": "H0/no-P4",
        "metadata": {
            "commit": _git_commit(),
            "command": command or "<python API>",
            "cpu": _host_metadata(),
            "blas": blas,
            "native": native_metadata,
            "manifest_revision": artifact["source"]["revision"],
            "manifest_repository": artifact["source"]["repository"],
            "variant": variant,
            "census_sha256": census_sha256,
            **census_identity,
            "identity_scope": "declared full-model identity; only selected expert ranges were read",
            "range_hashes": range_hashes,
            "ranges": ranges,
        },
        "source": artifact["source"],
        "fetch": artifact["fetch"],
        "workload": {
            "layer": int(layer),
            "expert": int(expert),
            "seed": int(seed),
            "tokens": 1,
            "active_tokens": 1,
            "route_count": 1,
            "expert_ids": expert_ids.tolist(),
            "routing_weights": routing_weights.tolist(),
            "hidden_size": int(hidden.shape[1]),
            "hidden_sha256": _hash_array(hidden),
            "expert_ids_sha256": _hash_array(expert_ids),
            "routing_weights_sha256": _hash_array(routing_weights),
        },
        "selected_behavior": selected_behavior,
        "oracle": {
            **probe.gguf_oracle_identity(),
            "operation": "dequantize + FP32 dense SwiGLU",
            "cold_timed_reference": (
                "range/source validation + packed bytes/view setup + SHA-256 hashing + "
                "dequantize + FP32 dense SwiGLU on every repetition"
            ),
            "resident_setup_elapsed_ns": int(resident_setup_elapsed),
            "resident_setup_includes": ["gguf.dequantize"],
            "packed_source_sha256": packed_hashes,
            "dense_projection_sha256": resident_dense_hashes,
        },
        "warmups": {
            "count": int(warmup),
            "reference_cold_dequant_dense": warmup_cold,
            "reference_dense_resident": warmup_resident,
            "native": warmup_native,
        },
        "samples": {
            "reference_cold_dequant_dense": cold_samples,
            "reference_dense_resident": resident_samples,
            "native": native_samples,
        },
        "correctness": {
            "correct": True,
            "cold_dequant_dense": {
                "comparison_count": len(cold_comparisons),
                "comparisons": cold_comparisons,
                "rtol": probe.ORACLE_RTOL,
                "atol": probe.ORACLE_ATOL,
            },
            "dense_resident": {
                "comparison_count": len(resident_comparisons),
                "comparisons": resident_comparisons,
                "rtol": probe.ORACLE_RTOL,
                "atol": probe.ORACLE_ATOL,
            },
        },
        "statistics": {
            "scope": (
                "descriptive one-token/one-route selected-expert comparison only; no "
                "route/thread/full-bank/NUMA/full-engine performance claim"
            ),
            "dense_resident": {
                "scope": (
                    "prepared dense matrices; dequantization is setup, outside each timed sample"
                ),
                "reference": resident_stats,
                "native": native_stats,
                "reference_to_native_median_ratio": float(resident_stats["median_elapsed_ns"])
                / native_median,
            },
            "cold_dequant_dense": {
                "scope": (
                    "range/source validation, packed bytes/view setup, SHA-256 hashing, "
                    "dequantization and dense FP32 SwiGLU inside every reference timed sample"
                ),
                "reference": cold_stats,
                "native": native_stats,
                "reference_to_native_median_ratio": float(cold_stats["median_elapsed_ns"])
                / native_median,
            },
        },
        "warnings": [],
        "limitations": [
            "selected expert ranges only; no complete shard download or checksum",
            "one token and one route; no batch, top-k, route/thread sweep, cache, hybrid "
            "split or full-engine claim",
            "H0 target-CPU evidence only; no Tesla P4 or dual-P4 evidence",
        ],
    }


run_target_cpu_benchmark = benchmark_qwen38_expert


__all__ = ["BENCHMARK_SCHEMA", "benchmark_qwen38_expert", "run_target_cpu_benchmark"]
