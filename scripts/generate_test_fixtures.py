from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GGUF_DIR = ROOT / "tests" / "fixtures" / "gguf"
MANIFEST = GGUF_DIR / "manifest.json"


def _write_valid(path: Path) -> None:
    writer = gguf.GGUFWriter(path, "qwen4")
    writer.add_name("freetoken-pascal-synthetic-tiny")
    writer.add_uint32("general.alignment", 32)
    q4 = np.arange(36, dtype=np.uint8).reshape(2, 18)
    q6 = np.arange(420, dtype=np.uint16).astype(np.uint8).reshape(2, 210)
    writer.add_tensor("blk.0.ffn_gate_exps.weight", q4, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
    writer.add_tensor("blk.0.ffn_down_exps.weight", q6, raw_dtype=gguf.GGMLQuantizationType.Q6_K)
    writer.add_tensor("output_norm.weight", np.arange(8, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _field_part_offset(reader: gguf.GGUFReader, tensor_index: int, part_index: int) -> int:
    return int(
        reader.tensors[tensor_index].field.parts[part_index].ctypes.data - reader.data.ctypes.data
    )


def _patched(source: Path, destination: Path, offset: int, value: bytes) -> None:
    data = bytearray(source.read_bytes())
    data[offset : offset + len(value)] = value
    destination.write_bytes(data)


def generate() -> dict[str, object]:
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    valid = GGUF_DIR / "valid-heterogeneous.gguf"
    _write_valid(valid)
    reader = gguf.GGUFReader(valid)
    q4_type_offset = _field_part_offset(reader, 0, 4)
    q4_dimension_offset = _field_part_offset(reader, 0, 3)
    q4_data_offset = _field_part_offset(reader, 0, 5)

    _patched(valid, GGUF_DIR / "unsupported-known-quant.gguf", q4_type_offset, struct.pack("<I", 6))
    _patched(valid, GGUF_DIR / "unknown-quant.gguf", q4_type_offset, struct.pack("<I", 999))
    _patched(
        valid, GGUF_DIR / "malformed-fastest-dim.gguf", q4_dimension_offset, struct.pack("<Q", 31)
    )
    _patched(valid, GGUF_DIR / "malformed-offset.gguf", q4_data_offset, struct.pack("<Q", 1))
    _patched(valid, GGUF_DIR / "out-of-range-offset.gguf", q4_data_offset, struct.pack("<Q", 4096))
    bad_magic = GGUF_DIR / "bad-magic.gguf"
    shutil.copyfile(valid, bad_magic)
    _patched(bad_magic, bad_magic, 0, b"NOPE")
    (GGUF_DIR / "truncated-metadata.gguf").write_bytes(valid.read_bytes()[:40])

    cases = {
        "valid-heterogeneous.gguf": "valid",
        "unsupported-known-quant.gguf": "unsupported quant type",
        "unknown-quant.gguf": "unknown quant type",
        "malformed-fastest-dim.gguf": "invalid quant block stride",
        "malformed-offset.gguf": "misaligned tensor offset",
        "out-of-range-offset.gguf": "tensor offset beyond file",
        "bad-magic.gguf": "invalid magic",
        "truncated-metadata.gguf": "truncated metadata",
    }
    files = []
    for name, expectation in cases.items():
        path = GGUF_DIR / name
        files.append(
            {
                "name": name,
                "expectation": expectation,
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "generator": "scripts/generate_test_fixtures.py",
        "license": "Apache-2.0",
        "files": files,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    generated = generate()
    print(f"generated {len(generated['files'])} GGUF fixtures")
