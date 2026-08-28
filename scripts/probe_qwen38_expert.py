#!/usr/bin/env python3
"""Probe one real Qwen3.8 GGUF expert using bounded HTTP ranges (H0/no-P4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from freetoken.moe.real_artifact_probe import (  # noqa: E402
    DEFAULT_EXPERT,
    DEFAULT_LAYER,
    DEFAULT_REPEATS,
    DEFAULT_SEED,
    DEFAULT_VARIANT,
    DEFAULT_WARMUP,
    ArtifactProbeError,
    probe_qwen38_expert,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/qwen38-gguf.json")
    parser.add_argument(
        "--census",
        type=Path,
        default=ROOT / "tests/fixtures/results/qwen38-q4-census.metadata.json",
    )
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--expert", type=int, default=DEFAULT_EXPERT)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache/freetoken/qwen38-range",
        help="cache only selected ranges; no complete shard is ever stored",
    )
    parser.add_argument(
        "--offline", action="store_true", help="require all selected ranges in cache"
    )
    parser.add_argument("--output", type=Path, help="write JSON report to this path")
    args = parser.parse_args(argv)
    try:
        report = probe_qwen38_expert(
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
