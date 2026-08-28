#!/usr/bin/env python3
"""Compare same-model Qwen3.8 semantic observations under explicit tolerances."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from freetoken.reference_correctness import (  # noqa: E402
    Tolerance,
    compare_observation_bundles,
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r} is forbidden")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--measured", action="store_true")
    args = parser.parse_args()
    contract = json.loads(
        args.contract.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
    )
    if contract.get("schema_version") != 1:
        raise ValueError("comparison contract schema_version must be 1")
    tolerances = {name: Tolerance(**values) for name, values in contract["tolerances"].items()}
    evidence = compare_observation_bundles(
        args.subject,
        args.reference,
        tolerances=tolerances,
        exact_observations=set(contract["exact_observations"]),
        require_independent=bool(contract.get("require_independent", True)),
        evidence_status="measured" if args.measured else "synthetic",
    )
    rendered = json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
