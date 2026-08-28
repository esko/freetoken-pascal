"""Host-only, fail-closed structural validation for GGUF files and shard sets."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

from freetoken.gguf_shards import gguf_shard_paths
from freetoken.gguf_types import BLOCK_SHAPE, GGML_NAME, row_bytes

SUPPORTED_GGML_TYPES = frozenset(BLOCK_SHAPE)
_MAX_U64 = (1 << 64) - 1


def _checked_product(values: tuple[int, ...], *, tensor_name: str) -> int:
    product = 1
    for value in values:
        if value <= 0:
            raise ValueError(f"invalid dimensions for {tensor_name}: {values}")
        if product > _MAX_U64 // value:
            raise ValueError(f"dimension product overflows uint64 for {tensor_name}")
        product *= value
    return product


def inspect_gguf(
    path: str | Path,
    *,
    supported_quant_types: Collection[int] = SUPPORTED_GGML_TYPES,
) -> dict[str, Any]:
    """Validate a GGUF or complete shard set without materializing weight data."""
    import gguf

    source = Path(path)
    try:
        shards = gguf_shard_paths(source)
    except Exception as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"invalid GGUF {source}: {error}") from error

    names: set[str] = set()
    tensors: list[dict[str, Any]] = []
    shard_results: list[dict[str, Any]] = []
    version: int | None = None
    architecture: str | None = None
    for shard_index, shard in enumerate(shards):
        try:
            reader = gguf.GGUFReader(shard)
        except Exception as error:
            raise ValueError(f"invalid GGUF {shard}: {error}") from error

        alignment = int(reader.alignment)
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(f"invalid GGUF alignment {alignment} in {shard}")
        arch_field = reader.fields.get("general.architecture")
        if shard_index == 0 and arch_field is None:
            raise ValueError(f"{shard}: GGUF metadata lacks general.architecture")
        shard_architecture = None if arch_field is None else str(arch_field.contents())
        shard_version = int(reader.fields["GGUF.version"].contents())
        if version is None:
            version = shard_version
            architecture = shard_architecture
        else:
            if shard_version != version:
                raise ValueError(f"{shard}: version disagrees with the first shard")
            if shard_architecture is not None and shard_architecture != architecture:
                raise ValueError(f"{shard}: architecture disagrees with the first shard")

        file_size = shard.stat().st_size
        ranges: list[tuple[int, int, str]] = []
        shard_tensor_count = 0
        for tensor in reader.tensors:
            if tensor.name in names:
                raise ValueError(f"duplicate GGUF tensor name {tensor.name!r} across shards")
            names.add(tensor.name)
            quant_type = int(tensor.tensor_type)
            if quant_type not in BLOCK_SHAPE:
                raise ValueError(f"unknown GGUF quant type {quant_type} for {tensor.name}")
            if quant_type not in supported_quant_types:
                raise ValueError(f"unsupported GGUF quant type {quant_type} for {tensor.name}")

            dimensions = tuple(int(value) for value in tensor.shape)
            _checked_product(dimensions, tensor_name=tensor.name)
            fastest = dimensions[0]
            packed_row_bytes = row_bytes(fastest, quant_type)
            rows = _checked_product(dimensions[1:], tensor_name=tensor.name)
            if rows > _MAX_U64 // packed_row_bytes:
                raise ValueError(f"packed byte count overflows uint64 for {tensor.name}")
            expected_bytes = rows * packed_row_bytes
            if tensor.data.nbytes != expected_bytes:
                raise ValueError(
                    f"{tensor.name}: data size {tensor.data.nbytes} does not match {expected_bytes}"
                )

            offset = int(tensor.data_offset)
            if offset < int(reader.data_offset):
                raise ValueError(f"{tensor.name}: data offset precedes the data section")
            if offset % alignment:
                raise ValueError(
                    f"{tensor.name}: data offset {offset} is not {alignment}-byte aligned"
                )
            if offset > _MAX_U64 - expected_bytes:
                raise ValueError(f"data end overflows uint64 for {tensor.name}")
            end = offset + expected_bytes
            if end > file_size:
                raise ValueError(f"{tensor.name}: data ends beyond file size")
            if not tensor.data.flags.c_contiguous:
                raise ValueError(f"{tensor.name}: data has a non-contiguous stride")
            ranges.append((offset, end, tensor.name))
            shard_tensor_count += 1
            tensors.append(
                {
                    "name": tensor.name,
                    "shape": list(reversed(dimensions)),
                    "ggml_dimensions": list(dimensions),
                    "quant_type": quant_type,
                    "quant_name": GGML_NAME[quant_type],
                    "shard_index": shard_index,
                    "offset": offset,
                    "nbytes": expected_bytes,
                    "rows": rows,
                    "row_bytes": packed_row_bytes,
                }
            )

        previous_end = int(reader.data_offset)
        previous_name = "GGUF header"
        for offset, end, name in sorted(ranges):
            if offset < previous_end:
                raise ValueError(f"{name}: data overlaps {previous_name}")
            previous_end = end
            previous_name = name
        shard_results.append(
            {
                "index": shard_index,
                "path": str(shard),
                "size": file_size,
                "alignment": alignment,
                "tensor_count": shard_tensor_count,
            }
        )

    return {
        "version": version,
        "architecture": architecture,
        "alignment": shard_results[0]["alignment"],
        "shard_count": len(shards),
        "tensor_count": len(tensors),
        "shards": shard_results,
        "tensors": tensors,
    }


__all__ = ["SUPPORTED_GGML_TYPES", "inspect_gguf"]
