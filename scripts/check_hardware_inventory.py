from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def validate_pascal_inventory(data: Any, *, minimum_gpus: int = 1) -> list[str]:
    if not isinstance(data, dict):
        return ["inventory root must be an object"]
    gpus = data.get("gpus")
    if not isinstance(gpus, list):
        return ["gpus must be an array"]
    errors: list[str] = []
    if data.get("evidence_status") != "measured":
        errors.append("hardware gate requires evidence_status 'measured'")
    if len(gpus) < minimum_gpus:
        errors.append(f"expected at least {minimum_gpus} GPUs, found {len(gpus)}")
    for index, gpu in enumerate(gpus):
        if not isinstance(gpu, dict):
            errors.append(f"gpus[{index}] must be an object")
        elif gpu.get("compute_capability") != "6.1":
            errors.append(
                f"gpus[{index}] must have compute capability 6.1, "
                f"found {gpu.get('compute_capability')!r}"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reject non-Pascal hardware evidence")
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--minimum-gpus", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.inventory.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"ERROR: unable to read hardware inventory: {error}", file=sys.stderr)
        return 1
    errors = validate_pascal_inventory(data, minimum_gpus=args.minimum_gpus)
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"validated {len(data['gpus'])} sm_61 GPU(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
