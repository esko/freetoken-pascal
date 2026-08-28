"""Host-only structural validation for GGUF fixtures and input preflight."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

SUPPORTED_GGML_TYPES = frozenset({0, 1, 2, 8, 14, 30})


def inspect_gguf(
    path: str | Path,
    *,
    supported_quant_types: Collection[int] = SUPPORTED_GGML_TYPES,
) -> dict[str, Any]:
    import gguf

    source = Path(path)
    try:
        reader = gguf.GGUFReader(source)
    except Exception as error:
        raise ValueError(f"invalid GGUF {source}: {error}") from error

    alignment = int(reader.alignment)
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"invalid GGUF alignment {alignment}")
    if "general.architecture" not in reader.fields:
        raise ValueError("GGUF metadata lacks general.architecture")

    file_size = source.stat().st_size
    previous_end = int(reader.data_offset)
    names: set[str] = set()
    tensors: list[dict[str, Any]] = []
    for tensor in reader.tensors:
        if tensor.name in names:
            raise ValueError(f"duplicate GGUF tensor name {tensor.name!r}")
        names.add(tensor.name)
        quant_type = int(tensor.tensor_type)
        if quant_type not in supported_quant_types:
            raise ValueError(f"unsupported GGUF quant type {quant_type} for {tensor.name}")
        try:
            block, type_size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
        except KeyError as error:
            raise ValueError(f"unknown GGUF quant type {quant_type} for {tensor.name}") from error

        dimensions = tuple(int(value) for value in tensor.shape)
        if not dimensions or any(value <= 0 for value in dimensions):
            raise ValueError(f"invalid dimensions for {tensor.name}: {dimensions}")
        fastest = dimensions[0]
        if fastest % block:
            raise ValueError(
                f"{tensor.name}: fastest dimension {fastest} is not divisible by block {block}"
            )
        rows = 1
        for dimension in dimensions[1:]:
            rows *= dimension
        expected_bytes = rows * (fastest // block) * type_size
        if tensor.data.nbytes != expected_bytes:
            raise ValueError(
                f"{tensor.name}: data size {tensor.data.nbytes} does not match {expected_bytes}"
            )

        offset = int(tensor.data_offset)
        end = offset + expected_bytes
        if offset % alignment:
            raise ValueError(f"{tensor.name}: data offset {offset} is not {alignment}-byte aligned")
        if offset < previous_end:
            raise ValueError(f"{tensor.name}: data overlaps the previous tensor")
        if end > file_size:
            raise ValueError(f"{tensor.name}: data ends beyond file size")
        if not tensor.data.flags.c_contiguous:
            raise ValueError(f"{tensor.name}: data has a non-contiguous stride")
        previous_end = end
        tensors.append(
            {
                "name": tensor.name,
                "dimensions": list(dimensions),
                "quant_type": quant_type,
                "offset": offset,
                "nbytes": expected_bytes,
            }
        )

    return {
        "version": int(reader.fields["GGUF.version"].contents()),
        "alignment": alignment,
        "tensor_count": len(tensors),
        "tensors": tensors,
    }


__all__ = ["SUPPORTED_GGML_TYPES", "inspect_gguf"]
