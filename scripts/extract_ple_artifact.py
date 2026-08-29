#!/usr/bin/env python3
"""Extract the GGUF PLE range into an atomic serving artifact directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from freetoken.gguf_host import convert_gguf_ple_to_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = convert_gguf_ple_to_artifact(args.source, args.output)
    print((result / "manifest.json").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
