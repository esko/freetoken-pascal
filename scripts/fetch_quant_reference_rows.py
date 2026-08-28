#!/usr/bin/env python3
"""Fetch pinned GGUF byte ranges and write real-row dequantization references."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import urllib.request
from pathlib import Path

import gguf
import numpy as np

REVISION = "c8b5954a88c2775c546b92593eda40ea041d3176"
BASE_URL = (
    "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/"
    f"{REVISION}"
)
ROWS = (
    {
        "variant": "UD-Q4_K_XL",
        "shard": "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
        "tensor": "blk.0.ffn_gate_exps.weight",
        "quant_type": 12,
        "offset": 30833472800,
        "row_bytes": 1440,
        "elements": 2560,
        "packed_sha256": "f95f924ff5028deb29b1b3317cb396d4ec063d087419df9a2360bff8b53da564",
    },
    {
        "variant": "UD-Q4_K_XL",
        "shard": "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
        "tensor": "blk.0.ffn_down_exps.weight",
        "quant_type": 7,
        "offset": 30202586400,
        "row_bytes": 480,
        "elements": 640,
        "packed_sha256": "26b22a8c3ca1ef4ac00c7dea435e9efc8aeb84f5cfd85d5c401b822e862ba0bf",
    },
    {
        "variant": "UD-Q3_K_XL",
        "shard": "Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00003.gguf",
        "tensor": "blk.0.ffn_gate_exps.weight",
        "quant_type": 18,
        "offset": 30522234688,
        "row_bytes": 980,
        "elements": 2560,
        "packed_sha256": "9dcb100f585e45d2097ea0886118d96d0e3f2d168760b4858a2a65b4eeac28ba",
    },
    {
        "variant": "UD-Q3_K_XL",
        "shard": "Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00003.gguf",
        "tensor": "blk.0.ffn_down_exps.weight",
        "quant_type": 20,
        "offset": 30048634688,
        "row_bytes": 360,
        "elements": 640,
        "packed_sha256": "797f0645b85f31c7fcde09536f48100de2c338713e98f124ef5aefaefc5a9b31",
    },
)


def _fetch(entry: dict[str, object]) -> bytes:
    variant = entry["variant"]
    shard = entry["shard"]
    start = int(entry["offset"])
    size = int(entry["row_bytes"])
    request = urllib.request.Request(
        f"{BASE_URL}/{variant}/{shard}",
        headers={"Range": f"bytes={start}-{start + size - 1}"},
    )
    with urllib.request.urlopen(request) as response:
        packed = response.read()
    if len(packed) != size:
        raise RuntimeError(f"{shard}: fetched {len(packed)} bytes, expected {size}")
    digest = hashlib.sha256(packed).hexdigest()
    if digest != entry["packed_sha256"]:
        raise RuntimeError(f"{shard}: packed row digest changed: {digest}")
    return packed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/gguf/qwen38-reference-rows.json"),
    )
    args = parser.parse_args(argv)
    rows = []
    for source in ROWS:
        packed = _fetch(source)
        quant_type = gguf.GGMLQuantizationType(int(source["quant_type"]))
        values = gguf.dequantize(np.frombuffer(packed, dtype=np.uint8)[None, :], quant_type)
        values = values.astype("<f4", copy=False)
        rows.append(
            {
                **source,
                "packed_base64": base64.b64encode(packed).decode("ascii"),
                "reference_f32_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "reference_min": float(values.min()),
                "reference_max": float(values.max()),
                "reference_sum": float(values.sum()),
            }
        )
    document = {
        "schema_version": 1,
        "source_repository": "unsloth/Qwen3.8-Flash-Next-GGUF",
        "source_revision": REVISION,
        "oracle": "gguf-py dequantize",
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
