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


def _write_qwen_host_layout(path: Path) -> None:
    writer = gguf.GGUFWriter(path, "qwen4exp")
    writer.add_name("freetoken-pascal-qwen-host-layout")
    writer.add_uint32("general.alignment", 32)
    writer.add_array("qwen4exp.ple.layer_multipliers", [3, 5, 7])
    writer.add_array("qwen4exp.ple.head_offsets", [0, 8])
    writer.add_array("qwen4exp.ple.head_vocab_sizes", [8, 8])

    def quant_tensor(name: str, shape: tuple[int, ...], quant_type) -> None:
        _, type_size = gguf.GGML_QUANT_SIZES[quant_type]
        block, _ = gguf.GGML_QUANT_SIZES[quant_type]
        row_bytes = shape[-1] // block * type_size
        raw_shape = (*shape[:-1], row_bytes)
        count = int(np.prod(raw_shape))
        raw = (np.arange(count, dtype=np.uint32) * 17 + 11).astype(np.uint8).reshape(raw_shape)
        writer.add_tensor(name, raw, raw_dtype=quant_type)

    # Two layers intentionally use incompatible gate/up and down formats.
    for projection in ("gate", "up"):
        quant_tensor(
            f"blk.0.ffn_{projection}_exps.weight",
            (3, 64, 256),
            gguf.GGMLQuantizationType.Q4_K,
        )
    quant_tensor(
        "blk.0.ffn_down_exps.weight",
        (3, 256, 64),
        gguf.GGMLQuantizationType.Q5_1,
    )
    for projection in ("gate", "up"):
        quant_tensor(
            f"blk.1.ffn_{projection}_exps.weight",
            (3, 64, 256),
            gguf.GGMLQuantizationType.Q5_K,
        )
    quant_tensor(
        "blk.1.ffn_down_exps.weight",
        (3, 256, 64),
        gguf.GGMLQuantizationType.Q8_0,
    )
    quant_tensor(
        "per_layer_token_embd.weight",
        (32, 160),
        gguf.GGMLQuantizationType.IQ4_NL,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _q4_k_block(seed: int) -> np.ndarray:
    """Return one finite Q4_K block with deterministic, non-zero values."""
    block = np.zeros(144, dtype=np.uint8)
    block[:2] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)
    block[2:4] = np.frombuffer(np.float16(0.0).tobytes(), dtype=np.uint8)
    block[4:16] = 1
    for group in range(4):
        lanes = np.arange(32, dtype=np.int16)
        low = ((lanes + seed + group) % 16).astype(np.uint8)
        high = ((lanes + seed + group + 3) % 16).astype(np.uint8)
        block[16 + group * 32 : 48 + group * 32] = low | (high << 4)
    return block


def _write_qwen_tiny_experts(path: Path) -> None:
    """Write a block-valid, two-layer Qwen4-Exp routed-expert fixture."""
    writer = gguf.GGUFWriter(path, "qwen4exp")
    writer.add_name("freetoken-pascal-qwen4-tiny-experts")
    writer.add_uint32("general.alignment", 32)
    writer.add_array("qwen4exp.ple.layer_multipliers", [3, 5, 7])
    writer.add_array("qwen4exp.ple.head_offsets", [0, 8, 16, 24])
    writer.add_array("qwen4exp.ple.head_vocab_sizes", [8, 8, 8, 8])

    def q4_tensor(name: str, shape: tuple[int, int, int], seed: int) -> None:
        experts, outputs, inputs = shape
        blocks_per_row = inputs // 256
        raw = np.empty((experts, outputs, blocks_per_row * 144), dtype=np.uint8)
        for expert in range(experts):
            for row in range(outputs):
                for block in range(blocks_per_row):
                    begin = block * 144
                    raw[expert, row, begin : begin + 144] = _q4_k_block(
                        seed + expert * 17 + row * 3 + block
                    )
        writer.add_tensor(name, raw, raw_dtype=gguf.GGMLQuantizationType.Q4_K)

    for layer in range(2):
        q4_tensor(f"blk.{layer}.ffn_gate_exps.weight", (4, 256, 256), 11 + layer)
        q4_tensor(f"blk.{layer}.ffn_up_exps.weight", (4, 256, 256), 23 + layer)
        q4_tensor(f"blk.{layer}.ffn_down_exps.weight", (4, 256, 256), 37 + layer)
    ple = np.zeros((32, 90), dtype=np.uint8)
    for row in range(32):
        for block in range(5):
            begin = block * 18
            ple[row, begin : begin + 2] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)
            ple[row, begin + 2 : begin + 18] = np.arange(16, dtype=np.uint8) + row + block
    writer.add_tensor(
        "per_layer_token_embd.weight",
        ple,
        raw_dtype=gguf.GGMLQuantizationType.IQ4_NL,
    )
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
    qwen_host = GGUF_DIR / "qwen-host-layout.gguf"
    _write_qwen_host_layout(qwen_host)
    qwen_tiny_experts = GGUF_DIR / "qwen4-tiny-experts.gguf"
    _write_qwen_tiny_experts(qwen_tiny_experts)
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
        "qwen-host-layout.gguf": "valid qwen heterogeneous host layout",
        "qwen4-tiny-experts.gguf": "valid block-valid Qwen4 tiny expert layout",
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
