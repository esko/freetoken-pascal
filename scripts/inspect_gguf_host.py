#!/usr/bin/env python3
"""Inspect Qwen3.8 heterogeneous expert pools and file-backed PLE placement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from freetoken.gguf_host import host_layout_document, inspect_qwen_host_layout  # noqa: E402
from freetoken.gguf_types import MOE_VEC_TYPES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path, help="any shard in the complete local shard set")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    layout = inspect_qwen_host_layout(
        args.gguf,
        supported_expert_types=MOE_VEC_TYPES,
    )
    rendered = json.dumps(host_layout_document(layout), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
