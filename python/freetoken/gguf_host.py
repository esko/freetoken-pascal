"""File-backed heterogeneous GGUF expert and PLE host storage.

This module is deliberately torch-free.  It validates pool geometry and maps only
the source tensor ranges; optimized CPU/GPU executors consume these descriptors in
later layers without changing the source addressing contract.
"""

from __future__ import annotations

import ctypes
import hashlib
import mmap
import os
import re
import resource
from collections.abc import Collection
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from freetoken.gguf_types import GGML_IQ4_NL, GGML_NAME
from freetoken.gguf_validation import inspect_gguf

_EXPERT_RE = re.compile(r"^blk\.(?P<layer>[0-9]+)\.ffn_(?P<projection>gate|up|down)_exps\.weight$")
_PLE_TENSOR = "per_layer_token_embd.weight"
_PROJECTIONS = ("gate", "up", "down")


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _proc_read_bytes() -> int | None:
    try:
        with open("/proc/self/io", encoding="utf-8") as source:
            values = dict(line.rstrip().split(": ", 1) for line in source)
        return int(values["read_bytes"])
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return None


def _warm_model_files(paths: tuple[str, ...]) -> int:
    warmed = 0
    for raw_path in paths:
        path = Path(raw_path)
        size = path.stat().st_size
        with (
            path.open("rb") as source,
            mmap.mmap(
                source.fileno(),
                0,
                access=mmap.ACCESS_READ,
            ) as mapped,
        ):
            pages = np.frombuffer(mapped, dtype=np.uint8)[:: mmap.PAGESIZE]
            _ = int(pages.sum(dtype=np.uint64))
            del pages
            if size:
                _ = mapped[size - 1]
        warmed += size
    return warmed


@dataclass(frozen=True)
class ExpertBankDescriptor:
    layer: int
    projection: str
    tensor_name: str
    quant_type: int
    quant_name: str
    experts: int
    output_dim: int
    input_dim: int
    row_bytes: int
    bytes_per_expert: int
    tensor_bytes: int
    shard_index: int
    shard_path: str
    data_offset: int
    pool_id: int = -1

    def source_offset(self, expert: int) -> int:
        if not 0 <= expert < self.experts:
            raise IndexError(f"expert {expert} outside [0, {self.experts})")
        return self.data_offset + expert * self.bytes_per_expert


@dataclass(frozen=True)
class ExpertSlotPool:
    pool_id: int
    projection: str
    quant_type: int
    quant_name: str
    output_dim: int
    input_dim: int
    row_bytes: int
    bytes_per_slot: int
    layers: tuple[int, ...]

    def slot_offset(self, slot: int, *, num_slots: int) -> int:
        if num_slots < 0:
            raise ValueError(f"num_slots must be non-negative, got {num_slots}")
        if not 0 <= slot < num_slots:
            raise IndexError(f"slot {slot} outside [0, {num_slots})")
        return slot * self.bytes_per_slot


@dataclass(frozen=True)
class GGUFExpertLayout:
    descriptors: tuple[ExpertBankDescriptor, ...]
    slot_pools: tuple[ExpertSlotPool, ...]
    num_layers: int
    num_experts: int

    def descriptor(self, layer: int, projection: str) -> ExpertBankDescriptor:
        for descriptor in self.descriptors:
            if descriptor.layer == layer and descriptor.projection == projection:
                return descriptor
        raise KeyError(f"no GGUF expert bank for layer={layer}, projection={projection!r}")


@dataclass(frozen=True)
class PLEDescriptor:
    tensor_name: str
    quant_type: int
    quant_name: str
    rows: int
    elements_per_row: int
    row_bytes: int
    tensor_bytes: int
    shard_index: int
    shard_path: str
    data_offset: int


@dataclass(frozen=True)
class QwenHostLayout:
    experts: GGUFExpertLayout
    ple: PLEDescriptor
    total_tensor_bytes: int
    shard_paths: tuple[str, ...]


def _expert_layout_from_records(
    records: list[dict[str, Any]],
    *,
    shard_paths: tuple[str, ...],
    supported_expert_types: Collection[int] | None,
) -> GGUFExpertLayout:
    descriptors: list[ExpertBankDescriptor] = []
    unsupported: list[str] = []
    for record in records:
        match = _EXPERT_RE.match(str(record["name"]))
        if match is None:
            continue
        quant_type = int(record["quant_type"])
        quant_name = str(record.get("quant_name", GGML_NAME.get(quant_type, quant_type)))
        if supported_expert_types is not None and quant_type not in supported_expert_types:
            unsupported.append(f"{record['name']}: {quant_name}")
        shape = tuple(int(value) for value in record["shape"])
        if len(shape) != 3:
            raise ValueError(f"{record['name']}: expected [experts, output, input], got {shape}")
        experts, output_dim, input_dim = shape
        row_bytes = int(record["row_bytes"])
        bytes_per_expert = output_dim * row_bytes
        tensor_bytes = int(record.get("nbytes", record.get("bytes", 0)))
        if tensor_bytes != experts * bytes_per_expert:
            raise ValueError(
                f"{record['name']}: {tensor_bytes} bytes disagree with "
                f"{experts} experts x {bytes_per_expert} bytes"
            )
        shard_index = int(record.get("shard_index", 0))
        if not 0 <= shard_index < len(shard_paths):
            raise ValueError(f"{record['name']}: invalid shard index {shard_index}")
        descriptors.append(
            ExpertBankDescriptor(
                layer=int(match.group("layer")),
                projection=match.group("projection"),
                tensor_name=str(record["name"]),
                quant_type=quant_type,
                quant_name=quant_name,
                experts=experts,
                output_dim=output_dim,
                input_dim=input_dim,
                row_bytes=row_bytes,
                bytes_per_expert=bytes_per_expert,
                tensor_bytes=tensor_bytes,
                shard_index=shard_index,
                shard_path=shard_paths[shard_index],
                data_offset=int(record.get("offset", 0)),
            )
        )
    if unsupported:
        raise ValueError("unsupported GGUF expert banks: " + "; ".join(sorted(unsupported)))
    if not descriptors:
        raise ValueError("GGUF contains no routed expert banks")

    descriptors.sort(key=lambda item: (item.layer, _PROJECTIONS.index(item.projection)))
    layers = sorted({item.layer for item in descriptors})
    if layers != list(range(layers[-1] + 1)):
        raise ValueError(f"GGUF expert layers are not contiguous: {layers}")
    expert_counts = {item.experts for item in descriptors}
    if len(expert_counts) != 1:
        raise ValueError(f"GGUF expert banks disagree on expert count: {sorted(expert_counts)}")

    for layer in layers:
        by_projection = {item.projection: item for item in descriptors if item.layer == layer}
        missing = set(_PROJECTIONS) - set(by_projection)
        extra = set(by_projection) - set(_PROJECTIONS)
        if missing or extra:
            raise ValueError(
                f"layer {layer} expert banks incomplete: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        gate = by_projection["gate"]
        up = by_projection["up"]
        down = by_projection["down"]
        if (gate.output_dim, gate.input_dim) != (up.output_dim, up.input_dim):
            raise ValueError(f"layer {layer} gate/up shapes disagree")
        if (down.output_dim, down.input_dim) != (gate.input_dim, gate.output_dim):
            raise ValueError(f"layer {layer} down shape is not the transpose geometry")

    geometry_keys = sorted(
        {
            (
                item.projection,
                item.quant_type,
                item.output_dim,
                item.input_dim,
                item.row_bytes,
                item.bytes_per_expert,
            )
            for item in descriptors
        }
    )
    pool_for_key = {key: index for index, key in enumerate(geometry_keys)}
    descriptors = [
        replace(
            item,
            pool_id=pool_for_key[
                (
                    item.projection,
                    item.quant_type,
                    item.output_dim,
                    item.input_dim,
                    item.row_bytes,
                    item.bytes_per_expert,
                )
            ],
        )
        for item in descriptors
    ]
    pools = []
    for pool_id, key in enumerate(geometry_keys):
        projection, quant_type, output_dim, input_dim, row_bytes, bytes_per_slot = key
        pools.append(
            ExpertSlotPool(
                pool_id=pool_id,
                projection=projection,
                quant_type=quant_type,
                quant_name=GGML_NAME[quant_type],
                output_dim=output_dim,
                input_dim=input_dim,
                row_bytes=row_bytes,
                bytes_per_slot=bytes_per_slot,
                layers=tuple(item.layer for item in descriptors if item.pool_id == pool_id),
            )
        )
    return GGUFExpertLayout(
        descriptors=tuple(descriptors),
        slot_pools=tuple(pools),
        num_layers=len(layers),
        num_experts=expert_counts.pop(),
    )


def expert_layout_from_census(
    census: dict[str, Any],
    *,
    supported_expert_types: Collection[int] | None = None,
) -> GGUFExpertLayout:
    shards = tuple(str(shard["name"]) for shard in census["shards"])
    return _expert_layout_from_records(
        list(census["tensors"]),
        shard_paths=shards,
        supported_expert_types=supported_expert_types,
    )


def host_memory_report_from_census(census: dict[str, Any]) -> dict[str, int]:
    layout = expert_layout_from_census(census)
    expert_bytes = sum(descriptor.tensor_bytes for descriptor in layout.descriptors)
    ple_records = [record for record in census["tensors"] if record["name"] == _PLE_TENSOR]
    if len(ple_records) != 1:
        raise ValueError(f"expected exactly one {_PLE_TENSOR}, found {len(ple_records)}")
    ple_bytes = int(ple_records[0]["nbytes"])
    total = int(census["total_bytes"])
    if expert_bytes + ple_bytes > total:
        raise ValueError("expert and PLE byte accounting exceeds total tensor bytes")
    return {
        "total_tensor_bytes": total,
        "ordinary_tensor_bytes": total - expert_bytes - ple_bytes,
        "expert_mapped_bytes": expert_bytes,
        "ple_mapped_bytes": ple_bytes,
        "maximum_file_backed_resident_bytes": total,
        "anonymous_host_source_bytes": 0,
        "pinned_host_source_bytes": 0,
    }


def _layout_from_inspection(
    inspected: dict[str, Any],
    *,
    supported_expert_types: Collection[int] | None,
) -> QwenHostLayout:
    if inspected["architecture"] != "qwen4exp":
        raise ValueError(f"expected qwen4exp GGUF, got {inspected['architecture']!r}")
    shard_paths = tuple(str(shard["path"]) for shard in inspected["shards"])
    experts = _expert_layout_from_records(
        inspected["tensors"],
        shard_paths=shard_paths,
        supported_expert_types=supported_expert_types,
    )
    matches = [record for record in inspected["tensors"] if record["name"] == _PLE_TENSOR]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {_PLE_TENSOR}, found {len(matches)}")
    record = matches[0]
    shape = tuple(int(value) for value in record["shape"])
    if len(shape) != 2:
        raise ValueError(f"{_PLE_TENSOR}: expected [rows, width], got {shape}")
    if int(record["quant_type"]) != GGML_IQ4_NL:
        raise ValueError(f"{_PLE_TENSOR}: expected IQ4_NL, got {record['quant_name']}")
    shard_index = int(record["shard_index"])
    ple = PLEDescriptor(
        tensor_name=_PLE_TENSOR,
        quant_type=int(record["quant_type"]),
        quant_name=str(record["quant_name"]),
        rows=shape[0],
        elements_per_row=shape[1],
        row_bytes=int(record["row_bytes"]),
        tensor_bytes=int(record["nbytes"]),
        shard_index=shard_index,
        shard_path=shard_paths[shard_index],
        data_offset=int(record["offset"]),
    )
    return QwenHostLayout(
        experts=experts,
        ple=ple,
        total_tensor_bytes=sum(int(record["nbytes"]) for record in inspected["tensors"]),
        shard_paths=shard_paths,
    )


def inspect_qwen_host_layout(
    path: str | Path,
    *,
    supported_expert_types: Collection[int] | None = None,
) -> QwenHostLayout:
    return _layout_from_inspection(
        inspect_gguf(path),
        supported_expert_types=supported_expert_types,
    )


def host_memory_report(layout: QwenHostLayout) -> dict[str, int]:
    expert_bytes = sum(descriptor.tensor_bytes for descriptor in layout.experts.descriptors)
    return {
        "total_tensor_bytes": layout.total_tensor_bytes,
        "ordinary_tensor_bytes": layout.total_tensor_bytes - expert_bytes - layout.ple.tensor_bytes,
        "expert_mapped_bytes": expert_bytes,
        "ple_mapped_bytes": layout.ple.tensor_bytes,
        "maximum_file_backed_resident_bytes": layout.total_tensor_bytes,
        "anonymous_host_source_bytes": 0,
        "pinned_host_source_bytes": 0,
    }


def host_layout_document(layout: QwenHostLayout) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "expert_layers": layout.experts.num_layers,
        "experts_per_layer": layout.experts.num_experts,
        "expert_banks": len(layout.experts.descriptors),
        "slot_pools": [
            {
                "pool_id": pool.pool_id,
                "projection": pool.projection,
                "quant_type": pool.quant_name,
                "output_dim": pool.output_dim,
                "input_dim": pool.input_dim,
                "packed_row_bytes": pool.row_bytes,
                "bytes_per_slot": pool.bytes_per_slot,
                "layers": list(pool.layers),
            }
            for pool in layout.experts.slot_pools
        ],
        "ple": {
            "tensor": layout.ple.tensor_name,
            "quant_type": layout.ple.quant_name,
            "rows": layout.ple.rows,
            "elements_per_row": layout.ple.elements_per_row,
            "packed_row_bytes": layout.ple.row_bytes,
            "mapped_bytes": layout.ple.tensor_bytes,
            "shard": layout.ple.shard_path,
            "offset": layout.ple.data_offset,
        },
        "memory": host_memory_report(layout),
    }


class MappedFileRange:
    """Private, read-only-by-contract mapping of one validated file range."""

    def __init__(
        self,
        path: str | Path,
        *,
        offset: int,
        length: int,
        rows: int,
        row_bytes: int,
        expected_file_sha256: str | None = None,
        verify_file_sha256: bool = False,
    ) -> None:
        self.path = Path(path)
        if offset < 0 or length <= 0 or rows <= 0 or row_bytes <= 0:
            raise ValueError("mapped tensor range dimensions must be positive")
        if rows * row_bytes != length:
            raise ValueError(f"mapped rows {rows} x {row_bytes} do not equal {length} bytes")
        file_size = self.path.stat().st_size
        if offset > file_size or length > file_size - offset:
            raise ValueError(
                f"mapped tensor range [{offset}, {offset + length}) exceeds "
                f"{self.path} size {file_size}"
            )
        if expected_file_sha256 is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256):
                raise ValueError(f"invalid expected sha256 {expected_file_sha256!r}")
            if verify_file_sha256:
                actual = _sha256_file(self.path)
                if actual != expected_file_sha256:
                    raise ValueError(
                        f"{self.path}: sha256 {actual} does not match {expected_file_sha256}"
                    )
        elif verify_file_sha256:
            raise ValueError("verify_file_sha256 requires expected_file_sha256")

        page_size = mmap.PAGESIZE
        map_offset = offset - offset % page_size
        prefix = offset - map_offset
        map_length = prefix + length
        self._fd = os.open(self.path, os.O_RDONLY)
        try:
            self._mapping = mmap.mmap(
                self._fd,
                map_length,
                access=mmap.ACCESS_COPY,
                offset=map_offset,
            )
        except BaseException:
            os.close(self._fd)
            raise
        self._map_offset = map_offset
        self._prefix = prefix
        self._length = length
        self._page_size = page_size
        self._address = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
        self.rows = np.frombuffer(
            self._mapping,
            dtype=np.uint8,
            count=length,
            offset=prefix,
        ).reshape(rows, row_bytes)
        self.rows.flags.writeable = False
        self.file_backed = True

    @property
    def length(self) -> int:
        return self._length

    def advise(self, advice: int) -> None:
        if hasattr(self._mapping, "madvise"):
            self._mapping.madvise(advice)

    def resident_pages(self) -> int | None:
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            page_count = (len(self._mapping) + self._page_size - 1) // self._page_size
            vector = (ctypes.c_ubyte * page_count)()
            result = libc.mincore(
                ctypes.c_void_p(self._address),
                ctypes.c_size_t(len(self._mapping)),
                vector,
            )
            if result != 0:
                return None
            return sum(1 for value in vector if value & 1)
        except (AttributeError, OSError):
            return None

    def close(self) -> None:
        rows = self.rows
        self.rows = np.empty((0, 0), dtype=np.uint8)
        del rows
        self._mapping.close()
        os.close(self._fd)

    def __enter__(self) -> MappedFileRange:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class MappedExpertBank:
    def __init__(self, descriptor: ExpertBankDescriptor) -> None:
        self.descriptor = descriptor
        self.mapping = MappedFileRange(
            descriptor.shard_path,
            offset=descriptor.data_offset,
            length=descriptor.tensor_bytes,
            rows=descriptor.experts * descriptor.output_dim,
            row_bytes=descriptor.row_bytes,
        )

    def expert_packed(self, expert: int) -> np.ndarray:
        self.descriptor.source_offset(expert)
        start = expert * self.descriptor.output_dim
        result = self.mapping.rows[start : start + self.descriptor.output_dim]
        result.flags.writeable = False
        return result

    def close(self) -> None:
        self.mapping.close()


class MappedExpertBanks:
    def __init__(self, layout: GGUFExpertLayout) -> None:
        self.layout = layout
        self._banks: dict[tuple[int, str], MappedExpertBank] = {}
        try:
            for descriptor in layout.descriptors:
                self._banks[(descriptor.layer, descriptor.projection)] = MappedExpertBank(
                    descriptor
                )
        except BaseException:
            self.close()
            raise

    def bank(self, layer: int, projection: str) -> MappedExpertBank:
        try:
            return self._banks[(layer, projection)]
        except KeyError as error:
            raise KeyError(
                f"no mapped GGUF bank for layer={layer}, projection={projection!r}"
            ) from error

    def close(self) -> None:
        for bank in self._banks.values():
            bank.close()


def dequantize_iq4_nl(packed: np.ndarray) -> np.ndarray:
    raw = np.asarray(packed, dtype=np.uint8)
    if raw.ndim < 1 or raw.shape[-1] % 18:
        raise ValueError(f"IQ4_NL packed width must be a multiple of 18, got {raw.shape}")
    original = raw.shape[:-1]
    blocks_per_row = raw.shape[-1] // 18
    blocks = raw.reshape(-1, 18)
    scales = blocks[:, :2].copy().view("<f2").astype(np.float32).reshape(-1, 1)
    codes = blocks[:, 2:]
    indices = np.concatenate((codes & 0x0F, codes >> 4), axis=1)
    codebook = np.array(
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
        dtype=np.float32,
    )
    values = scales * codebook[indices]
    return values.reshape(*original, blocks_per_row * 32)


class MappedPLETable:
    _MODES = frozenset({"cold", "page-cache-warm", "targeted", "full-model-warm"})

    def __init__(
        self,
        descriptor: PLEDescriptor,
        mapping: MappedFileRange,
        *,
        model_shard_paths: tuple[str, ...],
    ) -> None:
        self.descriptor = descriptor
        self.mapping = mapping
        self._model_shard_paths = model_shard_paths
        self.mode = "cold"
        self._lookup_calls = 0
        self._lookup_rows = 0
        self._packed_bytes_read = 0
        self._output_bytes = 0
        self._minor_faults = 0
        self._major_faults = 0
        self._storage_read_bytes = 0
        self._targeted_warm_rows = 0
        self._full_model_warm_bytes = 0

    @classmethod
    def open_from_gguf(
        cls,
        path: str | Path,
        *,
        expected_file_sha256: str | None = None,
        verify_file_sha256: bool = False,
        warm_mode: str = "cold",
    ) -> MappedPLETable:
        layout = inspect_qwen_host_layout(path)
        descriptor = layout.ple
        mapping = MappedFileRange(
            descriptor.shard_path,
            offset=descriptor.data_offset,
            length=descriptor.tensor_bytes,
            rows=descriptor.rows,
            row_bytes=descriptor.row_bytes,
            expected_file_sha256=expected_file_sha256,
            verify_file_sha256=verify_file_sha256,
        )
        table = cls(
            descriptor,
            mapping,
            model_shard_paths=layout.shard_paths,
        )
        try:
            table.set_warm_mode(warm_mode)
        except BaseException:
            table.close()
            raise
        return table

    def set_warm_mode(self, mode: str) -> None:
        if mode not in self._MODES:
            raise ValueError(f"unknown PLE warm mode {mode!r}; expected {sorted(self._MODES)}")
        self.mode = mode
        if mode == "cold" and hasattr(mmap, "MADV_DONTNEED"):
            self.mapping.advise(mmap.MADV_DONTNEED)
        elif mode == "page-cache-warm" and hasattr(mmap, "MADV_WILLNEED"):
            self.mapping.advise(mmap.MADV_WILLNEED)
        if mode == "full-model-warm":
            # Explicitly touch every source shard, including ordinary tensors and
            # expert banks.  This is intentionally never the default.
            self._full_model_warm_bytes += _warm_model_files(self._model_shard_paths)

    def _validate_ids(self, ids: np.ndarray) -> np.ndarray:
        values = np.asarray(ids)
        if values.dtype.kind not in "iu" or values.dtype.kind == "b":
            raise TypeError(f"PLE row ids must be integers, got {values.dtype}")
        flat = values.astype(np.int64, copy=False).reshape(-1)
        if flat.size and (int(flat.min()) < 0 or int(flat.max()) >= self.descriptor.rows):
            bad = int(flat.min()) if int(flat.min()) < 0 else int(flat.max())
            raise IndexError(f"PLE row {bad} outside [0, {self.descriptor.rows})")
        return flat

    def warm_rows(self, ids: np.ndarray) -> None:
        flat = self._validate_ids(ids)
        unique = np.unique(flat)
        if unique.size:
            _ = int(self.mapping.rows[unique, 0].sum(dtype=np.uint64))
        self._targeted_warm_rows += int(unique.size)

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        original = np.asarray(ids).shape
        flat = self._validate_ids(ids)
        before_usage = resource.getrusage(resource.RUSAGE_SELF)
        before_storage = _proc_read_bytes()
        if self.mode == "targeted":
            self.warm_rows(flat)
        packed = self.mapping.rows[flat].copy()
        result = dequantize_iq4_nl(packed).reshape(
            *original,
            self.descriptor.elements_per_row,
        )
        after_usage = resource.getrusage(resource.RUSAGE_SELF)
        after_storage = _proc_read_bytes()
        self._lookup_calls += 1
        self._lookup_rows += int(flat.size)
        self._packed_bytes_read += int(flat.size) * self.descriptor.row_bytes
        self._output_bytes += int(result.nbytes)
        self._minor_faults += max(0, int(after_usage.ru_minflt - before_usage.ru_minflt))
        self._major_faults += max(0, int(after_usage.ru_majflt - before_usage.ru_majflt))
        if before_storage is not None and after_storage is not None:
            self._storage_read_bytes += max(0, after_storage - before_storage)
        return result

    def telemetry(self) -> dict[str, int | str | None]:
        resident_pages = self.mapping.resident_pages()
        return {
            "mode": self.mode,
            "mapped_bytes": self.mapping.length,
            "resident_pages": resident_pages,
            "resident_bytes": (None if resident_pages is None else resident_pages * mmap.PAGESIZE),
            "lookup_calls": self._lookup_calls,
            "lookup_rows": self._lookup_rows,
            "packed_bytes_read": self._packed_bytes_read,
            "output_bytes": self._output_bytes,
            "minor_faults": self._minor_faults,
            "major_faults": self._major_faults,
            "storage_read_bytes": self._storage_read_bytes,
            "targeted_warm_rows": self._targeted_warm_rows,
            "full_model_warm_bytes": self._full_model_warm_bytes,
        }

    def close(self) -> None:
        self.mapping.close()

    def __enter__(self) -> MappedPLETable:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class QwenGGUFHostWeights:
    def __init__(
        self,
        layout: QwenHostLayout,
        experts: MappedExpertBanks,
        ple: MappedPLETable,
    ) -> None:
        self.layout = layout
        self.experts = experts
        self.ple = ple

    def memory_report(self) -> dict[str, int]:
        report = host_memory_report(self.layout)
        return {
            **report,
            "anonymous_model_bytes": report["anonymous_host_source_bytes"],
            "pinned_bytes": report["pinned_host_source_bytes"],
        }

    def close(self) -> None:
        self.ple.close()
        self.experts.close()

    def __enter__(self) -> QwenGGUFHostWeights:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def open_qwen_host_weights(
    path: str | Path,
    *,
    supported_expert_types: Collection[int] | None = None,
    ple_warm_mode: str = "cold",
) -> QwenGGUFHostWeights:
    layout = inspect_qwen_host_layout(
        path,
        supported_expert_types=supported_expert_types,
    )
    experts = MappedExpertBanks(layout.experts)
    try:
        ple_mapping = MappedFileRange(
            layout.ple.shard_path,
            offset=layout.ple.data_offset,
            length=layout.ple.tensor_bytes,
            rows=layout.ple.rows,
            row_bytes=layout.ple.row_bytes,
        )
        ple = MappedPLETable(
            layout.ple,
            ple_mapping,
            model_shard_paths=layout.shard_paths,
        )
        ple.set_warm_mode(ple_warm_mode)
    except BaseException:
        experts.close()
        raise
    return QwenGGUFHostWeights(layout, experts, ple)


__all__ = [
    "ExpertBankDescriptor",
    "ExpertSlotPool",
    "GGUFExpertLayout",
    "MappedPLETable",
    "QwenGGUFHostWeights",
    "QwenHostLayout",
    "dequantize_iq4_nl",
    "expert_layout_from_census",
    "host_layout_document",
    "host_memory_report",
    "host_memory_report_from_census",
    "inspect_qwen_host_layout",
    "open_qwen_host_weights",
]
