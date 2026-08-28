"""Keep host GGML layout/capability tables synchronized with pinned CUDA code."""

from __future__ import annotations

import re
from pathlib import Path

import gguf
import pytest
from freetoken.gguf_types import (
    BLOCK_SHAPE,
    DEQUANT_TYPES,
    GGML_NAME,
    GGML_UNQUANTIZED,
    MMQ_TYPES,
    MMVQ_TYPES,
    MOE_MMQ_TYPES,
    MOE_VEC_TYPES,
    row_bytes,
)

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "python/freetoken/kernel/csrc/gguf/gguf_kernel.cu"


def _switch_cases(function_name: str) -> set[int]:
    source = KERNEL.read_text(encoding="utf-8")
    function = re.search(rf"\b{function_name}\s*\([^)]*\)\s*\{{", source)
    assert function is not None, f"missing CUDA function {function_name}"
    switch = source.find("switch (type)", function.end())
    assert function.end() <= switch < function.end() + 5000
    opening = source.find("{", switch)
    depth = 1
    cursor = opening + 1
    while depth and cursor < len(source):
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    assert depth == 0, f"unterminated switch in {function_name}"
    cases = re.findall(r"\bcase\s+([0-9]+)\s*:", source[opening:cursor])
    return {int(value) for value in cases}


def test_block_layouts_match_gguf_python_oracle() -> None:
    oracle = {
        int(quant_type): (int(block), int(size))
        for quant_type, (block, size) in gguf.GGML_QUANT_SIZES.items()
        if int(quant_type) in BLOCK_SHAPE
    }
    assert oracle == BLOCK_SHAPE
    assert set(GGML_NAME) == set(BLOCK_SHAPE)


def test_row_bytes_is_exact_and_fails_closed() -> None:
    for quant_type, (block, size) in BLOCK_SHAPE.items():
        assert row_bytes(block * 7, quant_type) == size * 7
        if block > 1:
            with pytest.raises(ValueError, match="not a multiple"):
                row_bytes(block + 1, quant_type)
    with pytest.raises(ValueError, match="unknown GGML type"):
        row_bytes(256, 999)
    with pytest.raises(ValueError, match="positive"):
        row_bytes(0, 0)


def test_capability_sets_match_cuda_switches() -> None:
    assert _switch_cases("ggml_mul_mat_vec_a8") == MMVQ_TYPES
    assert _switch_cases("ggml_mul_mat_a8") == MMQ_TYPES
    assert _switch_cases("ggml_moe_a8") == MOE_MMQ_TYPES
    assert _switch_cases("ggml_moe_a8_vec") == MOE_VEC_TYPES


def test_capability_sets_are_consistent() -> None:
    assert MMVQ_TYPES == DEQUANT_TYPES == MOE_VEC_TYPES
    assert MMQ_TYPES == MOE_MMQ_TYPES
    assert MMQ_TYPES < MMVQ_TYPES
    assert DEQUANT_TYPES.isdisjoint(GGML_UNQUANTIZED)
    assert DEQUANT_TYPES | GGML_UNQUANTIZED == set(BLOCK_SHAPE)
