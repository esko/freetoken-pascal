#!/usr/bin/env python3
"""Run the bounded Qwen3.8 Q3 real-byte reference triad (H0/no-P4)."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from freetoken.moe.real_artifact_probe import (  # noqa: E402
    DEFAULT_Q3_TRIAD,
    DEFAULT_Q3_VARIANT,
    DEFAULT_REPEATS,
    DEFAULT_SEED,
    DEFAULT_WARMUP,
    ArtifactProbeError,
    probe_qwen38_q3_triad,
)


def _probe(value: str) -> tuple[int, int]:
    try:
        layer_text, expert_text = value.split(":", 1)
        layer, expert = int(layer_text), int(expert_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "probe must use LAYER:EXPERT, for example 23:255"
        ) from error
    if layer < 0 or expert < 0:
        raise argparse.ArgumentTypeError("probe layer and expert must be non-negative")
    return layer, expert


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/qwen38-gguf.json")
    parser.add_argument(
        "--census",
        type=Path,
        default=ROOT / "tests/fixtures/results/qwen38-q3-census.metadata.json",
    )
    parser.add_argument("--variant", default=DEFAULT_Q3_VARIANT)
    parser.add_argument(
        "--probe",
        dest="probes",
        type=_probe,
        action="append",
        help="one triad point as LAYER:EXPERT (repeat exactly three times)",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache/freetoken/qwen38-range",
        help="cache selected ranges only; complete shards are never stored",
    )
    parser.add_argument("--offline", action="store_true", help="require selected ranges in cache")
    parser.add_argument("--commit", help="full source commit SHA-1 (default: git HEAD)")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="atomically checkpoint each completed point in this directory",
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume completed points from --checkpoint-dir"
    )
    parser.add_argument(
        "--watchdog-seconds",
        type=float,
        help="fail at a point boundary when one point exceeds this duration",
    )
    parser.add_argument("--output", type=Path, help="write the aggregate JSON report to this path")
    args = parser.parse_args(argv)
    probes = tuple(args.probes) if args.probes is not None else DEFAULT_Q3_TRIAD
    command_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    try:

        def progress(event: dict[str, object]) -> None:
            if event.get("status") in {"started", "completed", "resumed"}:
                print(
                    f"Q3 triad {event.get('status')}: {event.get('probe_id')}",
                    file=sys.stderr,
                )

        report = probe_qwen38_q3_triad(
            manifest_path=args.manifest,
            census_path=args.census,
            variant=args.variant,
            probes=probes,
            repeats=args.repeats,
            warmup=args.warmup,
            seed=args.seed,
            cache_dir=args.cache_dir,
            offline=args.offline,
            command=shlex.join(command_argv),
            commit=args.commit,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            watchdog_seconds=args.watchdog_seconds,
            progress=progress,
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
