#!/usr/bin/env python3
"""Upgrade checked-in quant censuses with current derived host-layout sections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from freetoken.gguf_census import add_host_layout_sections  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("census", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.census:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["schema_version"] = 3
        add_host_layout_sections(document)
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"upgraded {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
