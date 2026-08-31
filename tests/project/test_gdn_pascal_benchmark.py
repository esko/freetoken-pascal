"""Hosted validation for the bounded Pascal GDN benchmark CLI."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "bench_gdn_pascal", ROOT / "benchmarks" / "bench_gdn_pascal.py"
)
assert SPEC and SPEC.loader
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def test_parser_defaults_are_bounded_and_pascal_shapes_are_explicit() -> None:
    args = BENCHMARK.build_parser().parse_args([])

    assert args.tokens == 1
    assert args.head_dim == 128
    assert args.key_heads == 16
    assert args.value_heads == 48
    assert args.warmups == 2
    assert args.repeats == 5
    BENCHMARK.validate_config(
        BENCHMARK.BenchmarkConfig(
            tokens=args.tokens,
            head_dim=args.head_dim,
            key_heads=args.key_heads,
            value_heads=args.value_heads,
            warmups=args.warmups,
            repeats=args.repeats,
            seed=args.seed,
            device=args.device,
        )
    )


@pytest.mark.parametrize(
    "config",
    [
        BENCHMARK.BenchmarkConfig(tokens=8, head_dim=128, key_heads=2, value_heads=8),
        BENCHMARK.BenchmarkConfig(tokens=32, head_dim=64, key_heads=4, value_heads=4),
        BENCHMARK.BenchmarkConfig(tokens=1, head_dim=128, key_heads=1, value_heads=32),
    ],
)
def test_valid_gqa_shape_choices_are_accepted(config) -> None:
    assert BENCHMARK.validate_config(config) == config


@pytest.mark.parametrize(
    "config, message",
    [
        (BENCHMARK.BenchmarkConfig(tokens=2), "tokens"),
        (BENCHMARK.BenchmarkConfig(head_dim=96), "head_dim"),
        (BENCHMARK.BenchmarkConfig(key_heads=3, value_heads=4), "multiple"),
        (BENCHMARK.BenchmarkConfig(key_heads=0), "positive"),
        (BENCHMARK.BenchmarkConfig(warmups=BENCHMARK.MAX_WARMUPS + 1), "warmups"),
        (BENCHMARK.BenchmarkConfig(repeats=BENCHMARK.MAX_REPEATS + 1), "repeats"),
        (BENCHMARK.BenchmarkConfig(repeats=0), "repeats"),
    ],
)
def test_validation_rejects_invalid_shapes_and_excessive_timing(config, message) -> None:
    with pytest.raises(BENCHMARK.BenchmarkConfigError, match=message):
        BENCHMARK.validate_config(config)


@pytest.mark.parametrize(
    "argv",
    [
        ["--tokens", "2"],
        ["--head-dim", "96"],
        ["--warmups", str(BENCHMARK.MAX_WARMUPS + 1)],
        ["--repeats", str(BENCHMARK.MAX_REPEATS + 1)],
        ["--key-heads", "3", "--value-heads", "4"],
    ],
)
def test_parser_rejects_invalid_or_unbounded_requests(argv) -> None:
    with pytest.raises((SystemExit, BENCHMARK.BenchmarkConfigError)):
        args = BENCHMARK.build_parser().parse_args(argv)
        BENCHMARK.validate_config(
            BENCHMARK.BenchmarkConfig(
                tokens=args.tokens,
                head_dim=args.head_dim,
                key_heads=args.key_heads,
                value_heads=args.value_heads,
                warmups=args.warmups,
                repeats=args.repeats,
                seed=args.seed,
                device=args.device,
            )
        )


def test_summary_keeps_raw_samples_and_reports_median_min_max() -> None:
    samples = [
        {"index": 0, "elapsed_ms": 3.0},
        {"index": 1, "elapsed_ms": 1.0},
        {"index": 2, "elapsed_ms": 2.0},
    ]

    assert BENCHMARK._summary(samples) == {
        "count": 3,
        "median_ms": 2.0,
        "min_ms": 1.0,
        "max_ms": 3.0,
    }


def test_injected_commit_supports_gitless_runtime(monkeypatch) -> None:
    commit = "a" * 40
    monkeypatch.setenv("FREETOKEN_BENCHMARK_COMMIT", commit)
    monkeypatch.setattr(BENCHMARK, "_checkout_commit", lambda *, required: None)

    assert BENCHMARK._git_commit() == commit


@pytest.mark.parametrize("commit", ["A" * 40, "a" * 39, "z" * 40])
def test_injected_commit_rejects_noncanonical_sha(monkeypatch, commit) -> None:
    monkeypatch.setenv("FREETOKEN_BENCHMARK_COMMIT", commit)

    with pytest.raises(RuntimeError, match="40-character lowercase Git SHA"):
        BENCHMARK._git_commit()


def test_injected_commit_must_match_checkout_when_git_is_available(monkeypatch) -> None:
    monkeypatch.setenv("FREETOKEN_BENCHMARK_COMMIT", "a" * 40)
    monkeypatch.setattr(BENCHMARK, "_checkout_commit", lambda *, required: "b" * 40)

    with pytest.raises(RuntimeError, match="does not match the mounted checkout"):
        BENCHMARK._git_commit()
