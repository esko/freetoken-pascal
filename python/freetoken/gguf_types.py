"""GGML type and block-layout contracts shared by host tools and runtime code."""

from __future__ import annotations

GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q4_1 = 3
GGML_Q5_0 = 6
GGML_Q5_1 = 7
GGML_Q8_0 = 8
GGML_Q2_K = 10
GGML_Q3_K = 11
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14
GGML_IQ2_XXS = 16
GGML_IQ2_XS = 17
GGML_IQ3_XXS = 18
GGML_IQ1_S = 19
GGML_IQ4_NL = 20
GGML_IQ3_S = 21
GGML_IQ2_S = 22
GGML_IQ4_XS = 23
GGML_IQ1_M = 29
GGML_BF16 = 30

# (elements per block, bytes per block), pinned to llama.cpp/ggml layouts.
BLOCK_SHAPE: dict[int, tuple[int, int]] = {
    GGML_F32: (1, 4),
    GGML_F16: (1, 2),
    GGML_Q4_0: (32, 18),
    GGML_Q4_1: (32, 20),
    GGML_Q5_0: (32, 22),
    GGML_Q5_1: (32, 24),
    GGML_Q8_0: (32, 34),
    GGML_Q2_K: (256, 84),
    GGML_Q3_K: (256, 110),
    GGML_Q4_K: (256, 144),
    GGML_Q5_K: (256, 176),
    GGML_Q6_K: (256, 210),
    GGML_IQ2_XXS: (256, 66),
    GGML_IQ2_XS: (256, 74),
    GGML_IQ3_XXS: (256, 98),
    GGML_IQ1_S: (256, 50),
    GGML_IQ4_NL: (32, 18),
    GGML_IQ3_S: (256, 110),
    GGML_IQ2_S: (256, 82),
    GGML_IQ4_XS: (256, 136),
    GGML_IQ1_M: (256, 56),
    GGML_BF16: (1, 2),
}

GGML_NAME = {
    GGML_F32: "F32",
    GGML_F16: "F16",
    GGML_Q4_0: "Q4_0",
    GGML_Q4_1: "Q4_1",
    GGML_Q5_0: "Q5_0",
    GGML_Q5_1: "Q5_1",
    GGML_Q8_0: "Q8_0",
    GGML_Q2_K: "Q2_K",
    GGML_Q3_K: "Q3_K",
    GGML_Q4_K: "Q4_K",
    GGML_Q5_K: "Q5_K",
    GGML_Q6_K: "Q6_K",
    GGML_IQ2_XXS: "IQ2_XXS",
    GGML_IQ2_XS: "IQ2_XS",
    GGML_IQ3_XXS: "IQ3_XXS",
    GGML_IQ1_S: "IQ1_S",
    GGML_IQ4_NL: "IQ4_NL",
    GGML_IQ3_S: "IQ3_S",
    GGML_IQ2_S: "IQ2_S",
    GGML_IQ4_XS: "IQ4_XS",
    GGML_IQ1_M: "IQ1_M",
    GGML_BF16: "BF16",
}

GGML_UNQUANTIZED = frozenset({GGML_F32, GGML_F16, GGML_BF16})
DEQUANT_TYPES = frozenset(BLOCK_SHAPE) - GGML_UNQUANTIZED
MMVQ_TYPES = DEQUANT_TYPES
MMQ_TYPES = frozenset(
    {
        GGML_Q4_0,
        GGML_Q4_1,
        GGML_Q5_0,
        GGML_Q5_1,
        GGML_Q8_0,
        GGML_Q2_K,
        GGML_Q3_K,
        GGML_Q4_K,
        GGML_Q5_K,
        GGML_Q6_K,
    }
)
MOE_VEC_TYPES = DEQUANT_TYPES
MOE_MMQ_TYPES = MMQ_TYPES


def row_bytes(numel: int, ggml_type: int) -> int:
    """Return the fail-closed packed byte length for one GGML row."""
    try:
        block, type_size = BLOCK_SHAPE[ggml_type]
    except KeyError as error:
        raise ValueError(f"unknown GGML type {ggml_type}") from error
    if numel <= 0:
        raise ValueError(f"row length must be positive, got {numel}")
    if numel % block:
        raise ValueError(
            f"{numel} not a multiple of block {block} for "
            f"{GGML_NAME.get(ggml_type, ggml_type)}"
        )
    return numel // block * type_size


__all__ = [
    "BLOCK_SHAPE",
    "DEQUANT_TYPES",
    "GGML_BF16",
    "GGML_F16",
    "GGML_F32",
    "GGML_IQ1_M",
    "GGML_IQ1_S",
    "GGML_IQ2_S",
    "GGML_IQ2_XS",
    "GGML_IQ2_XXS",
    "GGML_IQ3_S",
    "GGML_IQ3_XXS",
    "GGML_IQ4_NL",
    "GGML_IQ4_XS",
    "GGML_NAME",
    "GGML_Q2_K",
    "GGML_Q3_K",
    "GGML_Q4_0",
    "GGML_Q4_1",
    "GGML_Q4_K",
    "GGML_Q5_0",
    "GGML_Q5_1",
    "GGML_Q5_K",
    "GGML_Q6_K",
    "GGML_Q8_0",
    "GGML_UNQUANTIZED",
    "MMQ_TYPES",
    "MMVQ_TYPES",
    "MOE_MMQ_TYPES",
    "MOE_VEC_TYPES",
    "row_bytes",
]
