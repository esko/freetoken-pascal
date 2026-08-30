#!/usr/bin/env python3
"""Emit a deterministic quant-census document for a GGUF shard set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from freetoken.gguf_census import build_quant_census  # noqa: E402


def _declared_variant(manifest: Path, variant: str) -> dict[str, dict[str, object]]:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    try:
        shards = document["variants"][variant]["shards"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"manifest {manifest} has no variant {variant!r}") from error
    return {entry["name"]: entry for entry in shards}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path, help="any shard in the complete local shard set")
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--variant")
    parser.add_argument(
        "--profile",
        default="unassigned",
        help="precision-policy profile identity to embed (for example reference-q4)",
    )
    parser.add_argument(
        "--conversion-provenance",
        default="gguf_census:direct-tensor-metadata",
        help="immutable source/converter provenance recorded in sensitive records",
    )
    parser.add_argument(
        "--trust-declared-sha256",
        action="store_true",
        help="do not hash payloads; output is explicitly artifact-metadata evidence",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (args.artifact_manifest is None) != (args.variant is None):
        parser.error("--artifact-manifest and --variant must be used together")
    if args.trust_declared_sha256 and args.artifact_manifest is None:
        parser.error("--trust-declared-sha256 requires --artifact-manifest")

    declared = None
    if args.artifact_manifest is not None:
        declared = _declared_variant(args.artifact_manifest, args.variant)
    census = build_quant_census(
        args.gguf,
        declared_shards=declared,
        verify_sha256=not args.trust_declared_sha256,
        profile=args.profile,
        conversion_provenance=args.conversion_provenance,
    )
    rendered = json.dumps(census, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
