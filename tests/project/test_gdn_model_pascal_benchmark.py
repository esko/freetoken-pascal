"""Hosted validation for the bounded Pascal GDN model-boundary benchmark."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "bench_gdn_model_pascal", ROOT / "benchmarks" / "bench_gdn_model_pascal.py"
)
assert SPEC and SPEC.loader
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def _geometry() -> dict[str, object]:
    return {
        "hidden_size": 256,
        "head_dim": 128,
        "key_heads": 16,
        "value_heads": 48,
        "gqa_ratio": 3,
        "conv_kernel": 4,
        "model_dtype": "bfloat16",
        "recurrent_state_dtype": "float32",
    }


def _minimal_report() -> dict[str, object]:
    return {
        "format_name": "raw-pascal-gdn-model-boundary-observation",
        "format_version": 1,
        "qualification": BENCHMARK.QUALIFICATION,
        "geometry": _geometry(),
        "workload": {},
        "selected_behavior": {},
        "correctness": {},
        "timings": {},
        "timing_scope": "test",
        "metadata": {},
    }


def test_parser_defaults_are_bounded_and_qwen_geometry_is_fixed() -> None:
    args = BENCHMARK.build_parser().parse_args([])

    assert args.prefill_tokens == 1
    assert args.warmups == 2
    assert args.repeats == 5
    assert BENCHMARK.validate_config(
        BENCHMARK.BenchmarkConfig(
            prefill_tokens=args.prefill_tokens,
            warmups=args.warmups,
            repeats=args.repeats,
            seed=args.seed,
            device=args.device,
        )
    )
    assert BENCHMARK.MODEL_HEAD_DIM == 128
    assert BENCHMARK.MODEL_KEY_HEADS == 16
    assert BENCHMARK.MODEL_VALUE_HEADS == 48


@pytest.mark.parametrize(
    "config, message",
    [
        (BENCHMARK.BenchmarkConfig(prefill_tokens=2), "prefill_tokens"),
        (BENCHMARK.BenchmarkConfig(warmups=BENCHMARK.MAX_WARMUPS + 1), "warmups"),
        (BENCHMARK.BenchmarkConfig(repeats=BENCHMARK.MAX_REPEATS + 1), "repeats"),
        (BENCHMARK.BenchmarkConfig(repeats=0), "repeats"),
    ],
)
def test_config_rejects_invalid_or_unbounded_requests(config, message) -> None:
    with pytest.raises(BENCHMARK.BenchmarkConfigError, match=message):
        BENCHMARK.validate_config(config)


def test_parser_rejects_values_outside_bounded_choices() -> None:
    with pytest.raises(SystemExit):
        BENCHMARK.build_parser().parse_args(["--prefill-tokens", "64"])
    args = BENCHMARK.build_parser().parse_args(["--repeats", str(BENCHMARK.MAX_REPEATS + 1)])
    with pytest.raises(BENCHMARK.BenchmarkConfigError, match="repeats"):
        BENCHMARK.validate_config(
            BENCHMARK.BenchmarkConfig(
                prefill_tokens=args.prefill_tokens,
                warmups=args.warmups,
                repeats=args.repeats,
                seed=args.seed,
                device=args.device,
            )
        )


def test_summary_retains_two_clock_statistics() -> None:
    samples = [
        {"index": 0, "cuda_event_ms": 3.0, "host_wall_ms": 4.0},
        {"index": 1, "cuda_event_ms": 1.0, "host_wall_ms": 2.0},
        {"index": 2, "cuda_event_ms": 2.0, "host_wall_ms": 3.0},
    ]

    assert BENCHMARK._summary(samples) == {
        "count": 3,
        "cuda_event_median_ms": 2.0,
        "cuda_event_min_ms": 1.0,
        "cuda_event_max_ms": 3.0,
        "host_wall_median_ms": 3.0,
        "host_wall_min_ms": 2.0,
        "host_wall_max_ms": 4.0,
    }


def test_report_format_enforces_geometry_and_json_safety() -> None:
    report = _minimal_report()
    BENCHMARK.validate_report_format(report)
    json.dumps(report, allow_nan=False)

    report["geometry"] = {**_geometry(), "head_dim": 64}
    with pytest.raises(ValueError, match="geometry"):
        BENCHMARK.validate_report_format(report)


def test_report_format_rejects_non_release_qualification_drift() -> None:
    report = _minimal_report()
    report["qualification"] = "release"
    with pytest.raises(ValueError, match="non-release"):
        BENCHMARK.validate_report_format(report)


def test_injected_commit_supports_gitless_runtime(monkeypatch) -> None:
    commit = "a" * 40
    monkeypatch.setenv("FREETOKEN_BENCHMARK_COMMIT", commit)
    monkeypatch.setattr(BENCHMARK, "_checkout_commit", lambda *, required: None)

    assert BENCHMARK._git_commit() == commit
