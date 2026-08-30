#!/usr/bin/env python3
"""Run the dedicated-artifact PLE mmap/pread phase evidence harness.

This is an H0 evidence run, not a throughput benchmark.  It requires the
explicit ``--linux-probe`` opt-in so physical block-device bytes can never be
silently inferred from process telemetry.  Unit tests inject deterministic
physical snapshots through the same library API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from freetoken.ple_io_evidence import (  # noqa: E402
    LinuxPLEBlockCounterProbe,
    PLEIOEvidenceError,
)
from freetoken.ple_phase_harness import run_ple_io_phase_harness  # noqa: E402


def _batch(value: str) -> list[int]:
    try:
        values = [int(part.strip(), 10) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("row batches must be comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("row batches cannot be empty")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="dedicated PLE artifact directory containing manifest.json and ple.bin",
    )
    parser.add_argument(
        "--batch",
        dest="batches",
        action="append",
        type=_batch,
        help="comma-separated row IDs; repeat for deterministic batches",
    )
    parser.add_argument(
        "--linux-probe",
        action="store_true",
        help="opt in to the read-only Linux sysfs block-device counter probe",
    )
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument(
        "--planner-mode",
        choices=("direct", "vectorized", "adaptive"),
        default="vectorized",
    )
    parser.add_argument("--planner-direct-threshold", type=int, default=8)
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.linux_probe:
        raise SystemExit(
            "ERROR: physical evidence is disabled by default; pass --linux-probe explicitly"
        )
    manifest_path = args.artifact / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = int(manifest["rows"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"ERROR: cannot read dedicated PLE artifact geometry: {error}") from error
    batches = args.batches or [[0, rows // 2, rows - 1, rows // 2]]
    if any(row < 0 or row >= rows for batch in batches for row in batch):
        raise SystemExit("ERROR: requested row ID is outside the dedicated artifact")
    try:
        report = run_ple_io_phase_harness(
            args.artifact,
            row_batches=batches,
            linux_probe=LinuxPLEBlockCounterProbe(args.artifact / "ple.bin"),
            prefetch=not args.no_prefetch,
            planner_mode=args.planner_mode,
            planner_direct_threshold=args.planner_direct_threshold,
        )
    except PLEIOEvidenceError as error:
        raise SystemExit(f"ERROR: {error}") from error
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
