#!/usr/bin/env python3
"""Pack semantic Qwen3.8 arrays and immutable run identity into an observation bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from freetoken.reference_correctness import write_observation_bundle  # noqa: E402


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r} is forbidden")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True, help="non-pickle NumPy NPZ")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = json.loads(
        args.identity.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
    )
    with np.load(args.arrays, allow_pickle=False) as source:
        observations = {name: source[name] for name in source.files}
    write_observation_bundle(args.output, identity, observations)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
