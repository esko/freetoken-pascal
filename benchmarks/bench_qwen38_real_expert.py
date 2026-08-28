#!/usr/bin/env python3
"""Benchmark one real Qwen3.8 expert on the target CPU (H0/no-P4).

The benchmark fetches only the selected gate/up/down expert ranges (or reads them
from the bounded range cache), then compares native packed AVX2 execution with two
separately reported references: dense-resident and cold dequantize-plus-dense.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from freetoken.moe.real_artifact_benchmark import (  # noqa: E402
    MIN_WARMUP,
    SUPPORTED_LAYERS,
    benchmark_qwen38_expert,
)
from freetoken.moe.real_artifact_probe import (  # noqa: E402
    DEFAULT_EXPERT,
    DEFAULT_LAYER,
    DEFAULT_SEED,
    DEFAULT_VARIANT,
    ArtifactProbeError,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/qwen38-gguf.json")
    parser.add_argument(
        "--census",
        type=Path,
        default=ROOT / "tests/fixtures/results/qwen38-q4-census.metadata.json",
    )
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument(
        "--layer",
        type=int,
        choices=sorted(SUPPORTED_LAYERS),
        default=DEFAULT_LAYER,
        help="actual artifact layer: 0 (normal) or 2 (Q5_K/Q8_0 promoted)",
    )
    parser.add_argument("--expert", type=_nonnegative_int, default=DEFAULT_EXPERT)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument(
        "--warmup",
        type=_positive_int,
        default=MIN_WARMUP,
        help=f"untallied warmups for each path (minimum {MIN_WARMUP})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache/freetoken/qwen38-range",
        help="cache only selected expert ranges; no complete shard is stored",
    )
    parser.add_argument(
        "--offline", action="store_true", help="require all selected ranges to already be cached"
    )
    parser.add_argument(
        "--native-build-metadata",
        type=Path,
        help="optional JSON manifest from build_target_cpu_native.py",
    )
    parser.add_argument("--output", type=Path, help="write JSON report to this path")
    args = parser.parse_args(argv)
    try:
        report = benchmark_qwen38_expert(
            manifest_path=args.manifest,
            census_path=args.census,
            variant=args.variant,
            layer=args.layer,
            expert=args.expert,
            repeats=args.repeats,
            warmup=args.warmup,
            seed=args.seed,
            cache_dir=args.cache_dir,
            offline=args.offline,
            native_build_metadata_path=args.native_build_metadata,
            command=shlex.join([sys.argv[0], *(argv if argv is not None else sys.argv[1:])]),
        )
    except ArtifactProbeError as error:
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
