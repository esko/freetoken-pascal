"""File-backed heterogeneous GGUF expert and PLE host storage.

This module is deliberately torch-free.  It validates pool geometry and maps only
the source tensor ranges; optimized CPU/GPU executors consume these descriptors in
later layers without changing the source addressing contract.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import mmap
import os
import re
import resource
import shutil
import tempfile
import threading
import time
from collections.abc import Collection, Mapping
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from freetoken.gguf_types import GGML_IQ4_NL, GGML_NAME
from freetoken.gguf_validation import inspect_gguf

_EXPERT_RE = re.compile(r"^blk\.(?P<layer>[0-9]+)\.ffn_(?P<projection>gate|up|down)_exps\.weight$")
_PLE_TENSOR = "per_layer_token_embd.weight"
_PROJECTIONS = ("gate", "up", "down")
_PLE_DESCRIPTOR_FIELDS = (
    "tensor_name",
    "quant_type",
    "quant_name",
    "rows",
    "elements_per_row",
    "row_bytes",
    "tensor_bytes",
    "codec",
)


@dataclass(frozen=True)
class PLECodecDescriptor:
    """Immutable identity and geometry contract for one packed PLE row codec.

    ``parameters`` intentionally remains an open, JSON-compatible mapping.  This
    lets future row codecs describe group sizes, scale encodings, or codebook
    revisions without adding branches to the storage and lookup APIs.
    """

    codec_id: str
    version: int
    packed_dtype: str
    decoded_dtype: str
    elements_per_block: int
    bytes_per_block: int
    parameters: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.codec_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_.-]*", self.codec_id
        ):
            raise ValueError(f"invalid PLE codec identity {self.codec_id!r}")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version <= 0:
            raise ValueError(f"invalid PLE codec version {self.version!r}")
        for name, value in (
            ("packed_dtype", self.packed_dtype),
            ("decoded_dtype", self.decoded_dtype),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"invalid PLE codec {name} {value!r}")
            try:
                np.dtype(value)
            except TypeError as error:
                raise ValueError(f"invalid PLE codec {name} {value!r}") from error
        for name, value in (
            ("elements_per_block", self.elements_per_block),
            ("bytes_per_block", self.bytes_per_block),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"invalid PLE codec {name} {value!r}")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("PLE codec parameters must be a mapping")
        parameters = dict(self.parameters)
        for name, value in parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"invalid PLE codec parameter name {name!r}")
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"invalid PLE codec parameter {name!r}")
        object.__setattr__(self, "parameters", MappingProxyType(parameters))

    @property
    def identity(self) -> str:
        return f"{self.codec_id}@{self.version}"

    def to_manifest(self) -> dict[str, Any]:
        return {
            "id": self.codec_id,
            "version": self.version,
            "packed_dtype": self.packed_dtype,
            "decoded_dtype": self.decoded_dtype,
            "elements_per_block": self.elements_per_block,
            "bytes_per_block": self.bytes_per_block,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> PLECodecDescriptor:
        if not isinstance(value, Mapping):
            raise ValueError("PLE artifact codec descriptor must be an object")
        required = (
            "id",
            "version",
            "packed_dtype",
            "decoded_dtype",
            "elements_per_block",
            "bytes_per_block",
            "parameters",
        )
        if any(key not in value for key in required):
            raise ValueError("PLE artifact codec descriptor is incomplete")
        parameters = value["parameters"]
        if not isinstance(parameters, Mapping):
            raise ValueError("PLE artifact codec parameters must be an object")
        return cls(
            codec_id=value["id"],
            version=value["version"],
            packed_dtype=value["packed_dtype"],
            decoded_dtype=value["decoded_dtype"],
            elements_per_block=value["elements_per_block"],
            bytes_per_block=value["bytes_per_block"],
            parameters=parameters,
        )

    def validate_row_geometry(self, elements_per_row: int, row_bytes: int) -> None:
        if (
            isinstance(elements_per_row, bool)
            or not isinstance(elements_per_row, int)
            or elements_per_row <= 0
            or isinstance(row_bytes, bool)
            or not isinstance(row_bytes, int)
            or row_bytes <= 0
            or elements_per_row % self.elements_per_block
            or row_bytes % self.bytes_per_block
            or elements_per_row // self.elements_per_block != row_bytes // self.bytes_per_block
        ):
            raise ValueError(
                f"invalid PLE codec row geometry for {self.identity}: "
                f"{elements_per_row} elements, {row_bytes} bytes"
            )


class PLECodec(Protocol):
    """Decoder contract consumed by :class:`MappedPLETable`."""

    descriptor: PLECodecDescriptor

    def decode(
        self,
        packed: np.ndarray,
        *,
        rows: int,
        elements_per_row: int,
    ) -> np.ndarray: ...


class PLECodecRegistry:
    """Immutable registry resolving a manifest descriptor to one decoder."""

    def __init__(self, codecs: Mapping[tuple[str, int], PLECodec]) -> None:
        resolved: dict[tuple[str, int], PLECodec] = {}
        for key, codec in codecs.items():
            if key != (codec.descriptor.codec_id, codec.descriptor.version):
                raise ValueError("PLE codec registry key disagrees with descriptor")
            resolved[key] = codec
        self._codecs = MappingProxyType(resolved)

    @property
    def codecs(self) -> Mapping[tuple[str, int], PLECodec]:
        return self._codecs

    def resolve(
        self,
        descriptor_or_id: PLECodecDescriptor | str,
        version: int | None = None,
    ) -> PLECodec:
        if isinstance(descriptor_or_id, PLECodecDescriptor):
            descriptor = descriptor_or_id
        elif isinstance(descriptor_or_id, str) and version is not None:
            key = (descriptor_or_id, version)
            versions = {item[1] for item in self._codecs if item[0] == descriptor_or_id}
            if not versions:
                raise ValueError(f"unknown PLE codec {descriptor_or_id!r}")
            try:
                return self._codecs[key]
            except KeyError as error:
                raise ValueError(
                    f"unsupported PLE codec version {descriptor_or_id!r}/{version}"
                ) from error
        else:
            raise TypeError("resolve requires a PLECodecDescriptor or codec id and version")
        versions = {key[1] for key in self._codecs if key[0] == descriptor.codec_id}
        if not versions:
            raise ValueError(f"unknown PLE codec {descriptor.codec_id!r}")
        key = (descriptor.codec_id, descriptor.version)
        try:
            codec = self._codecs[key]
        except KeyError as error:
            raise ValueError(
                f"unsupported PLE codec version {descriptor.codec_id!r}/{descriptor.version}"
            ) from error
        if codec.descriptor != descriptor:
            raise ValueError(f"PLE codec descriptor mismatch for {descriptor.identity}")
        return codec


IQ4_NL_CODEC_DESCRIPTOR = PLECodecDescriptor(
    codec_id="iq4_nl",
    version=1,
    packed_dtype="uint8",
    decoded_dtype="float32",
    elements_per_block=32,
    bytes_per_block=18,
    parameters={
        "codebook": "ggml-iq4-nl",
        "codebook_version": 1,
        "scale_dtype": "float16",
    },
)


@dataclass(frozen=True)
class _IQ4NLCodec:
    descriptor: PLECodecDescriptor = IQ4_NL_CODEC_DESCRIPTOR

    def decode(
        self,
        packed: np.ndarray,
        *,
        rows: int,
        elements_per_row: int,
    ) -> np.ndarray:
        raw = np.asarray(packed)
        if raw.dtype != np.dtype(self.descriptor.packed_dtype):
            raise ValueError(
                f"PLE codec {self.descriptor.identity} expects packed dtype "
                f"{self.descriptor.packed_dtype}, got {raw.dtype}"
            )
        if raw.ndim != 2 or raw.shape[0] != rows:
            raise ValueError(
                f"PLE codec {self.descriptor.identity} expects {rows} packed rows, "
                f"got shape {raw.shape}"
            )
        self.descriptor.validate_row_geometry(elements_per_row, int(raw.shape[1]))
        decoded = dequantize_iq4_nl(raw)
        expected_shape = (rows, elements_per_row)
        if decoded.shape != expected_shape:
            raise ValueError(
                f"PLE codec decoder output shape {decoded.shape} disagrees with {expected_shape}"
            )
        if decoded.dtype != np.dtype(self.descriptor.decoded_dtype):
            raise ValueError(
                f"PLE codec decoder output dtype {decoded.dtype} disagrees with "
                f"{self.descriptor.decoded_dtype}"
            )
        return decoded


IQ4_NL_CODEC: PLECodec = _IQ4NLCodec()
PLE_CODEC_REGISTRY = PLECodecRegistry(
    {(IQ4_NL_CODEC_DESCRIPTOR.codec_id, IQ4_NL_CODEC_DESCRIPTOR.version): IQ4_NL_CODEC}
)


@dataclass(frozen=True)
class PLELookupPlannerConfig:
    """Validated policy for selecting a PLE lookup read plan."""

    mode: str = "vectorized"
    direct_threshold: int = 8

    def validate(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {
            "vectorized",
            "direct",
            "adaptive",
        }:
            raise ValueError(
                "invalid PLE planner mode "
                f"{self.mode!r}; expected 'vectorized', 'direct', or 'adaptive'"
            )
        if isinstance(self.direct_threshold, bool) or not isinstance(self.direct_threshold, int):
            raise TypeError("planner_direct_threshold must be an integer")
        if self.direct_threshold <= 0:
            raise ValueError("planner_direct_threshold must be positive")


def _resolve_planner_config(
    planner_config: PLELookupPlannerConfig | None,
    *,
    planner_mode: str = "vectorized",
    planner_direct_threshold: int = 8,
) -> PLELookupPlannerConfig:
    if planner_config is not None:
        if not isinstance(planner_config, PLELookupPlannerConfig):
            raise TypeError("planner_config must be a PLELookupPlannerConfig")
        if planner_mode != "vectorized" or planner_direct_threshold != 8:
            raise ValueError(
                "planner_config cannot be combined with planner_mode or planner_direct_threshold"
            )
        resolved = planner_config
    else:
        resolved = PLELookupPlannerConfig(
            mode=planner_mode,
            direct_threshold=planner_direct_threshold,
        )
    resolved.validate()
    return resolved


def _sha256_file(path: Path | int, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    if isinstance(path, int):
        size = os.fstat(path).st_size
        os.lseek(path, 0, os.SEEK_SET)
        offset = 0
        while offset < size:
            chunk = os.read(path, min(chunk_size, size - offset))
            if not chunk:
                raise ValueError("file shortened while calculating sha256")
            digest.update(chunk)
            offset += len(chunk)
        return digest.hexdigest()
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
        if size == 0:
            continue
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


def _warm_pread_file(
    fd: int,
    size: int,
    chunk_size: int = 8 << 20,
    *,
    start_offset: int = 0,
) -> int:
    if isinstance(fd, bool) or not isinstance(fd, int) or fd < 0:
        raise ValueError("invalid PLE positional warm file descriptor")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("invalid PLE positional warm size")
    if isinstance(start_offset, bool) or not isinstance(start_offset, int) or start_offset < 0:
        raise ValueError("invalid PLE positional warm start offset")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("invalid PLE positional warm chunk size")
    offset = 0
    while offset < size:
        requested = min(chunk_size, size - offset)
        chunk = os.pread(fd, requested, start_offset + offset)
        if len(chunk) != requested:
            raise ValueError("short PLE positional warm read")
        offset += requested
    return offset


def _warm_pread_files(paths: tuple[str, ...]) -> int:
    warmed = 0
    for raw_path in paths:
        fd = os.open(raw_path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            warmed += _warm_pread_file(fd, size)
        finally:
            os.close(fd)
    return warmed


def _validate_expected_file_sha256(
    path: Path,
    expected_file_sha256: str | None,
    verify_file_sha256: bool,
) -> None:
    if expected_file_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256):
            raise ValueError(f"invalid expected sha256 {expected_file_sha256!r}")
        if verify_file_sha256:
            actual = _sha256_file(path)
            if actual != expected_file_sha256:
                raise ValueError(f"{path}: sha256 {actual} does not match {expected_file_sha256}")
    elif verify_file_sha256:
        raise ValueError("verify_file_sha256 requires expected_file_sha256")


def _open_validated_pread_fd(
    path: str | Path,
    *,
    offset: int,
    length: int,
    expected_file_identity: PLEFileIdentity | None = None,
) -> int:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("invalid PLE positional range offset")
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("invalid PLE positional range length")
    fd = os.open(path, os.O_RDONLY)
    try:
        if expected_file_identity is not None:
            expected_file_identity.assert_fd(fd, label="PLE payload")
        file_size = os.fstat(fd).st_size
        if offset > file_size or length > file_size - offset:
            raise ValueError(
                f"PLE positional range [{offset}, {offset + length}) exceeds file size {file_size}"
            )
    except BaseException:
        os.close(fd)
        raise
    return fd


def _publish_directory_noreplace(staging: Path, output: Path) -> None:
    """Atomically publish a directory without replacing a concurrent destination."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is required for atomic PLE publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(staging),
        -100,
        os.fsencode(output),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(output)
        raise OSError(error, os.strerror(error), output)
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def convert_gguf_ple_to_artifact(source: str | Path, output: str | Path) -> Path:
    """Atomically extract the IQ4_NL PLE tensor into a serving-only artifact."""
    source = Path(source)
    output = Path(output)
    if output == Path("/") or not output.name:
        raise ValueError("output must be an explicit artifact directory")
    layout = inspect_qwen_host_layout(source)
    descriptor = layout.ple
    source_size = Path(descriptor.shard_path).stat().st_size
    if descriptor.data_offset < 0 or descriptor.data_offset + descriptor.tensor_bytes > source_size:
        raise ValueError("PLE source range is truncated")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        payload = staging / "ple.bin"
        digest = hashlib.sha256()
        with Path(descriptor.shard_path).open("rb") as source_file, payload.open("wb") as target:
            source_file.seek(descriptor.data_offset)
            remaining = descriptor.tensor_bytes
            while remaining:
                chunk = source_file.read(min(8 << 20, remaining))
                if not chunk:
                    raise ValueError("PLE source range is truncated")
                target.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            target.flush()
            os.fsync(target.fileno())
        manifest = {
            "format": "freetoken-pascal-ple-v1",
            "version": 1,
            "payload": "ple.bin",
            # The serving artifact is a raw tensor payload.  Keep this explicit so an
            # artifact descriptor can never accidentally inherit the source GGUF offset.
            "data_offset": 0,
            "tensor_name": descriptor.tensor_name,
            "quant_type": descriptor.quant_type,
            "quant_name": descriptor.quant_name,
            "codec": descriptor.codec.to_manifest(),
            "rows": descriptor.rows,
            "elements_per_row": descriptor.elements_per_row,
            "row_bytes": descriptor.row_bytes,
            "tensor_bytes": descriptor.tensor_bytes,
            "sha256": digest.hexdigest(),
            "source": {
                "path": descriptor.shard_path,
                "offset": descriptor.data_offset,
            },
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as target:
            json.dump(manifest, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        _publish_directory_noreplace(staging, output)
        return output
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


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
    codec: PLECodecDescriptor = IQ4_NL_CODEC_DESCRIPTOR


@dataclass(frozen=True)
class PLEFileIdentity:
    """Stable file identity captured while validating a PLE artifact.

    The identity is carried with a preflight result so the serving opener can
    reject replacement/truncation between the (expensive) checksum and mapping.
    It is intentionally metadata only; it is not a substitute for the manifest
    checksum performed during validation.
    """

    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def from_path(cls, path: str | Path) -> PLEFileIdentity:
        stat = Path(path).stat()
        return cls(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    @classmethod
    def from_fd(cls, fd: int) -> PLEFileIdentity:
        stat = os.fstat(fd)
        return cls(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def assert_path(self, path: str | Path, *, label: str) -> None:
        try:
            current = type(self).from_path(path)
        except OSError as error:
            raise ValueError(f"{label} disappeared after validation") from error
        if current != self:
            raise ValueError(f"{label} changed after validation")

    def assert_fd(self, fd: int, *, label: str) -> None:
        try:
            current = type(self).from_fd(fd)
        except OSError as error:
            raise ValueError(f"{label} could not be inspected after open") from error
        if current != self:
            raise ValueError(f"{label} changed between validation and open")


@dataclass(frozen=True)
class ValidatedPLEArtifact:
    """Immutable handoff from pre-CUDA PLE validation to resource acquisition."""

    artifact_path: str
    descriptor: PLEDescriptor
    payload_sha256: str
    manifest_identity: PLEFileIdentity
    payload_identity: PLEFileIdentity

    def assert_current(self) -> None:
        root = Path(self.artifact_path)
        manifest_path = root / "manifest.json"
        self.manifest_identity.assert_path(manifest_path, label="PLE manifest")
        self.payload_identity.assert_path(self.descriptor.shard_path, label="PLE payload")
        try:
            current_sha256 = json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"]
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError("PLE manifest became unreadable after validation") from error
        if current_sha256 != self.payload_sha256:
            raise ValueError("PLE manifest checksum changed after validation")


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
    IQ4_NL_CODEC_DESCRIPTOR.validate_row_geometry(
        int(shape[1]),
        int(record["row_bytes"]),
    )
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
        codec=IQ4_NL_CODEC_DESCRIPTOR,
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


def _read_validated_ple_artifact(
    path: str | Path,
) -> tuple[PLEDescriptor, str, PLEFileIdentity]:
    """Read and validate a dedicated PLE artifact without opening any mapping.

    This is deliberately a pure file/metadata operation.  It is used by Engine
    startup before model construction and by the mapping backend itself.  Keeping
    the complete manifest, geometry, payload-size, codec, and checksum checks in
    one helper prevents the preflight and serving paths from accepting different
    artifact identities.
    """
    root = Path(path)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"invalid PLE artifact manifest: {root}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("PLE artifact manifest must be a JSON object")
    if manifest.get("format") != "freetoken-pascal-ple-v1" or manifest.get("version") != 1:
        raise ValueError("unsupported PLE artifact format")
    required = ("rows", "elements_per_row", "row_bytes", "tensor_bytes", "sha256")
    if any(key not in manifest for key in required):
        raise ValueError("PLE artifact manifest missing geometry or checksum")
    if manifest.get("payload") != "ple.bin":
        raise ValueError("invalid PLE artifact payload name")
    artifact_data_offset = manifest.get("data_offset", manifest.get("offset", 0))
    if (
        isinstance(artifact_data_offset, bool)
        or not isinstance(artifact_data_offset, int)
        or artifact_data_offset != 0
    ):
        raise ValueError("PLE artifact data_offset must be zero")
    codec_value = manifest.get("codec")
    if codec_value is None:
        # The original v1 artifact schema identified IQ4_NL through the quant
        # fields alone. Keep those immutable artifacts readable while newly
        # emitted manifests carry the explicit descriptor.
        codec_descriptor = IQ4_NL_CODEC_DESCRIPTOR
    else:
        try:
            codec_descriptor = PLECodecDescriptor.from_manifest(codec_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid PLE artifact codec descriptor: {error}") from error
    PLE_CODEC_REGISTRY.resolve(codec_descriptor)
    if (
        manifest.get("tensor_name") != _PLE_TENSOR
        or manifest.get("quant_type") != GGML_IQ4_NL
        or manifest.get("quant_name") != "IQ4_NL"
    ):
        raise ValueError("unsupported PLE artifact tensor")
    values = tuple(manifest[key] for key in required[:4])
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("invalid PLE artifact geometry")
    rows, elements, row_bytes, tensor_bytes = values
    if min(rows, elements, row_bytes, tensor_bytes) <= 0 or tensor_bytes != rows * row_bytes:
        raise ValueError("invalid PLE artifact geometry")
    try:
        codec_descriptor.validate_row_geometry(elements, row_bytes)
    except ValueError as error:
        raise ValueError(f"invalid PLE artifact geometry: {error}") from error
    payload = root / str(manifest.get("payload", "ple.bin"))
    try:
        payload_fd = os.open(payload, os.O_RDONLY)
    except OSError as error:
        raise ValueError("PLE artifact payload is missing") from error
    try:
        payload_identity = PLEFileIdentity.from_fd(payload_fd)
        if payload_identity.size != tensor_bytes:
            raise ValueError("PLE artifact payload is truncated or has a gap")
        if not payload.is_file():
            raise ValueError("PLE artifact payload is not a regular file")
    except BaseException:
        os.close(payload_fd)
        raise
    if not isinstance(manifest["sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", manifest["sha256"]
    ):
        os.close(payload_fd)
        raise ValueError("invalid PLE artifact sha256")
    try:
        # Hash bytes through the already-open descriptor.  The before/after
        # fstats bind the digest to one inode and catch in-place replacement,
        # truncation, or writes that alter the file metadata during the stream.
        actual_sha256 = _sha256_file(payload_fd)
        if PLEFileIdentity.from_fd(payload_fd) != payload_identity:
            raise ValueError("PLE artifact payload changed while hashing")
        if actual_sha256 != manifest["sha256"]:
            raise ValueError("PLE artifact sha256 mismatch")
        descriptor = PLEDescriptor(
            tensor_name=str(manifest.get("tensor_name", _PLE_TENSOR)),
            quant_type=GGML_IQ4_NL,
            quant_name="IQ4_NL",
            rows=rows,
            elements_per_row=elements,
            row_bytes=row_bytes,
            tensor_bytes=tensor_bytes,
            shard_index=0,
            shard_path=str(payload),
            data_offset=0,
            codec=codec_descriptor,
        )
        return descriptor, actual_sha256, payload_identity
    finally:
        os.close(payload_fd)


def validate_ple_artifact(
    path: str | Path,
    *,
    source_path: str | Path | None = None,
) -> PLEDescriptor:
    """Validate a dedicated PLE artifact and optionally its source GGUF identity.

    No mmap, file descriptor, tensor, or CUDA resource is acquired.  When a
    ``source_path`` is supplied, its parsed PLE descriptor must match every
    semantic field of the artifact, including codec and row geometry.
    """
    descriptor, _, _ = _read_validated_ple_artifact(path)
    if source_path is not None:
        source = inspect_qwen_host_layout(source_path).ple
        mismatches = [
            name
            for name in _PLE_DESCRIPTOR_FIELDS
            if getattr(descriptor, name) != getattr(source, name)
        ]
        if mismatches:
            raise ValueError(
                "dedicated PLE artifact does not match source GGUF descriptor: "
                + ", ".join(mismatches)
            )
    return descriptor


def validate_ple_artifact_handoff(
    path: str | Path,
    *,
    source_path: str | Path | None = None,
) -> ValidatedPLEArtifact:
    """Validate once and return a safe, immutable opener handoff.

    The payload checksum is intentionally performed here, before Engine model
    construction.  Consumers must pass this object to the artifact opener;
    they must not reconstruct it from a path or cache it globally.
    """
    root = Path(path).resolve(strict=True)
    manifest = root / "manifest.json"
    manifest_before = PLEFileIdentity.from_path(manifest)
    descriptor, payload_sha256, payload_identity = _read_validated_ple_artifact(root)
    manifest_identity = PLEFileIdentity.from_path(manifest)
    if manifest_identity != manifest_before:
        raise ValueError("PLE artifact manifest changed during validation")
    if source_path is not None:
        source = inspect_qwen_host_layout(source_path).ple
        mismatches = [
            name
            for name in _PLE_DESCRIPTOR_FIELDS
            if getattr(descriptor, name) != getattr(source, name)
        ]
        if mismatches:
            raise ValueError(
                "dedicated PLE artifact does not match source GGUF descriptor: "
                + ", ".join(mismatches)
            )
    return ValidatedPLEArtifact(
        artifact_path=str(root),
        descriptor=descriptor,
        payload_sha256=payload_sha256,
        manifest_identity=manifest_identity,
        payload_identity=payload_identity,
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
            "codec": layout.ple.codec.to_manifest(),
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
        expected_file_identity: PLEFileIdentity | None = None,
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
        _validate_expected_file_sha256(
            self.path,
            expected_file_sha256,
            verify_file_sha256,
        )

        page_size = mmap.PAGESIZE
        map_offset = offset - offset % page_size
        prefix = offset - map_offset
        map_length = prefix + length
        self._fd = os.open(self.path, os.O_RDONLY)
        try:
            if expected_file_identity is not None:
                expected_file_identity.assert_fd(self._fd, label="PLE payload")
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
        self._mapping_closed = False
        self._fd_closed = False
        self._closed = False
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

    def warm(self) -> int:
        """Touch every byte-range page represented by this mapping."""
        if self._closed:
            raise RuntimeError("mapped file range is closed")
        pages = np.frombuffer(
            self._mapping,
            dtype=np.uint8,
            count=self._length,
            offset=self._prefix,
        )[:: mmap.PAGESIZE]
        try:
            _ = int(pages.sum(dtype=np.uint64))
        finally:
            del pages
        if self._length:
            _ = self._mapping[self._prefix + self._length - 1]
        return self._length

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
        if self._closed:
            return
        rows = self.rows
        self.rows = np.empty((0, 0), dtype=np.uint8)
        del rows
        failure: BaseException | None = None
        if not self._mapping_closed:
            try:
                self._mapping.close()
            except BaseException as error:
                failure = error
            else:
                self._mapping_closed = True
        if not self._fd_closed:
            try:
                os.close(self._fd)
            except BaseException as error:
                failure = failure or error
            else:
                self._fd_closed = True
        if failure is not None:
            raise failure
        self._closed = True

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
        self._closed = False

    def expert_packed(self, expert: int) -> np.ndarray:
        self.descriptor.source_offset(expert)
        start = expert * self.descriptor.output_dim
        result = self.mapping.rows[start : start + self.descriptor.output_dim]
        result.flags.writeable = False
        return result

    def close(self) -> None:
        if self._closed:
            return
        self.mapping.close()
        self._closed = True


class MappedExpertBanks:
    def __init__(self, layout: GGUFExpertLayout) -> None:
        self.layout = layout
        self._banks: dict[tuple[int, str], MappedExpertBank] = {}
        self._closed = False
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
        if self._closed:
            return
        failure: BaseException | None = None
        for bank in self._banks.values():
            try:
                bank.close()
            except BaseException as error:
                failure = failure or error
        if failure is not None:
            raise failure
        self._closed = True


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


class _PrefetchCancelled(CancelledError):
    def __init__(self, warmed_rows: int) -> None:
        super().__init__("PLE prefetch cancelled")
        self.warmed_rows = warmed_rows


class _PrefetchWorkerError(Exception):
    def __init__(self, cause: BaseException, warmed_rows: int) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.warmed_rows = warmed_rows


class PLEPrefetchHandle:
    """Lifecycle handle for one bounded, warming-only PLE prefetch."""

    def __init__(
        self,
        row_ids: tuple[int, ...],
        requested_rows: int,
        future: Future[int],
        cancel_event: threading.Event,
        finalized_event: threading.Event,
    ) -> None:
        self._row_ids = row_ids
        self.requested_rows = requested_rows
        self.unique_rows = len(row_ids)
        self._future = future
        self._cancel_event = cancel_event
        self._finalized_event = finalized_event
        self._lifecycle_lock = threading.RLock()

    @property
    def row_ids(self) -> tuple[int, ...]:
        return self._row_ids

    def done(self) -> bool:
        return self._finalized_event.is_set()

    def cancel(self) -> bool:
        """Request cancellation and report whether the request was still pending."""
        with self._lifecycle_lock:
            if self._finalized_event.is_set():
                return False
            self._cancel_event.set()
            self._future.cancel()
            return True

    def result(self, timeout: float | None = None) -> None:
        """Wait for completion without exposing any prefetched row data."""
        deadline = None if timeout is None else time.monotonic() + timeout
        failure: BaseException | None = None
        try:
            self._future.result(timeout=timeout)
        except BaseException as error:
            failure = error
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        if not self._finalized_event.wait(timeout=remaining):
            raise TimeoutError("PLE prefetch finalization timed out")
        if isinstance(failure, _PrefetchWorkerError):
            raise failure.cause from failure
        if failure is not None:
            raise failure
        if self._cancel_event.is_set():
            raise CancelledError("PLE prefetch cancelled")

    def wait(self, timeout: float | None = None) -> None:
        """Alias for :meth:`result` for explicit lifecycle-oriented callers."""
        self.result(timeout=timeout)


def _validate_prefetch_config(max_rows: int, chunk_rows: int) -> None:
    if isinstance(max_rows, bool) or not isinstance(max_rows, int):
        raise TypeError("prefetch_max_rows must be an integer")
    if max_rows < 0:
        raise ValueError("prefetch_max_rows must be non-negative")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int):
        raise TypeError("prefetch_chunk_rows must be an integer")
    if chunk_rows <= 0:
        raise ValueError("prefetch_chunk_rows must be positive")


class MappedPLETable:
    _MODES = frozenset({"cold", "page-cache-warm", "targeted", "full-model-warm", "full-ple-warm"})

    def __init__(
        self,
        descriptor: PLEDescriptor,
        mapping: MappedFileRange | None,
        *,
        model_shard_paths: tuple[str, ...],
        prefetch_max_rows: int = 4096,
        prefetch_chunk_rows: int = 64,
        planner_config: PLELookupPlannerConfig | None = None,
        planner_mode: str = "vectorized",
        planner_direct_threshold: int = 8,
    ) -> None:
        _validate_prefetch_config(prefetch_max_rows, prefetch_chunk_rows)
        resolved_planner = _resolve_planner_config(
            planner_config,
            planner_mode=planner_mode,
            planner_direct_threshold=planner_direct_threshold,
        )
        self.descriptor = descriptor
        self.mapping = mapping
        self._closed = False
        self._closing = False
        self._model_shard_paths = model_shard_paths
        self._prefetch_max_rows = prefetch_max_rows
        self._prefetch_chunk_rows = prefetch_chunk_rows
        self._prefetch_lock = threading.RLock()
        self._io_lock = threading.RLock()
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_active: PLEPrefetchHandle | None = None
        self._prefetch_submitted = 0
        self._prefetch_completed = 0
        self._prefetch_cancelled = 0
        self._prefetch_failed = 0
        self._prefetch_requested_rows = 0
        self._prefetch_unique_rows = 0
        self._prefetch_warmed_rows = 0
        self.mode = "cold"
        self._planner_config = resolved_planner
        self._planner_selected_mode = "none"
        self._planner_calls = 0
        self._planner_time_ns = 0
        self._direct_calls = 0
        self._direct_rows = 0
        self._vectorized_calls = 0
        self._vectorized_rows = 0
        self._application_reads = 0
        self._application_bytes_read = 0
        self._lookup_calls = 0
        self._lookup_rows = 0
        self._packed_bytes_read = 0
        self._output_bytes = 0
        self._minor_faults = 0
        self._major_faults = 0
        self._storage_read_bytes = 0
        self._targeted_warm_rows = 0
        self._full_model_warm_bytes = 0
        self.backend = "mmap"
        self.source_kind = "source-gguf"
        self._pread_fd: int | None = None
        self._batch_calls = self._batch_requested_rows = self._batch_unique_rows = 0
        self._batch_positional_reads = self._batch_duplicate_rows = 0
        self._batch_sorted_rows = self._batch_bytes_read = 0
        self._targeted_positional_warm_reads = 0
        self._short_reads = 0
        self._advice = "not-requested"
        self._advice_applied = False
        self._advice_error: str | None = None
        self.codec = PLE_CODEC_REGISTRY.resolve(descriptor.codec)
        descriptor.codec.validate_row_geometry(descriptor.elements_per_row, descriptor.row_bytes)

    def _apply_random_advice(self) -> None:
        try:
            if self.mapping is not None and hasattr(mmap, "MADV_RANDOM"):
                self._advice = "madv-random"
                self.mapping.advise(mmap.MADV_RANDOM)
            elif self._pread_fd is not None and hasattr(os, "POSIX_FADV_RANDOM"):
                self._advice = "posix-fadv-random"
                os.posix_fadvise(
                    self._pread_fd,
                    self.descriptor.data_offset,
                    self.descriptor.tensor_bytes,
                    os.POSIX_FADV_RANDOM,
                )
            else:
                self._advice = "unsupported"
                return
            self._advice_applied = True
        except (OSError, TypeError, ValueError) as error:
            self._advice_error = f"{type(error).__name__}: {error}"

    def _require_pread_fd(self) -> int:
        fd = self._pread_fd
        if fd is None:
            raise RuntimeError("PLE positional backend file descriptor is unavailable")
        if isinstance(fd, bool) or not isinstance(fd, int) or fd < 0:
            raise RuntimeError("PLE positional backend file descriptor is invalid")
        try:
            os.fstat(fd)
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError("PLE positional backend file descriptor is invalid") from error
        return fd

    def _warm_ple_range(self) -> int:
        if self.mapping is not None:
            return self.mapping.warm()
        fd = self._require_pread_fd()
        return _warm_pread_file(
            fd,
            self.descriptor.tensor_bytes,
            start_offset=self.descriptor.data_offset,
        )

    def _apply_range_advice(self, advice: int) -> None:
        if self.mapping is not None:
            self.mapping.advise(advice)
            return
        if not hasattr(os, "posix_fadvise"):
            return
        fd = self._require_pread_fd()
        os.posix_fadvise(
            fd,
            self.descriptor.data_offset,
            self.descriptor.tensor_bytes,
            advice,
        )

    @classmethod
    def open_from_gguf(
        cls,
        path: str | Path,
        *,
        expected_file_sha256: str | None = None,
        verify_file_sha256: bool = False,
        warm_mode: str = "cold",
        backend: str = "mmap",
        prefetch_max_rows: int = 4096,
        prefetch_chunk_rows: int = 64,
        planner_config: PLELookupPlannerConfig | None = None,
        planner_mode: str = "vectorized",
        planner_direct_threshold: int = 8,
    ) -> MappedPLETable:
        _validate_prefetch_config(prefetch_max_rows, prefetch_chunk_rows)
        resolved_planner = _resolve_planner_config(
            planner_config,
            planner_mode=planner_mode,
            planner_direct_threshold=planner_direct_threshold,
        )
        if backend not in {"mmap", "pread"}:
            raise ValueError(f"unknown PLE backend {backend!r}")
        layout = inspect_qwen_host_layout(path)
        descriptor = layout.ple
        mapping: MappedFileRange | None = None
        table: MappedPLETable | None = None
        pread_fd: int | None = None
        try:
            if backend == "mmap":
                mapping = MappedFileRange(
                    descriptor.shard_path,
                    offset=descriptor.data_offset,
                    length=descriptor.tensor_bytes,
                    rows=descriptor.rows,
                    row_bytes=descriptor.row_bytes,
                    expected_file_sha256=expected_file_sha256,
                    verify_file_sha256=verify_file_sha256,
                )
            else:
                _validate_expected_file_sha256(
                    Path(descriptor.shard_path),
                    expected_file_sha256,
                    verify_file_sha256,
                )
                pread_fd = _open_validated_pread_fd(
                    descriptor.shard_path,
                    offset=descriptor.data_offset,
                    length=descriptor.tensor_bytes,
                )
            table = cls(
                descriptor,
                mapping,
                model_shard_paths=layout.shard_paths,
                prefetch_max_rows=prefetch_max_rows,
                prefetch_chunk_rows=prefetch_chunk_rows,
                planner_config=resolved_planner,
            )
            table.backend = backend
            table.source_kind = f"gguf-{backend}"
            table._pread_fd = pread_fd
            table._apply_random_advice()
            table.set_warm_mode(warm_mode)
        except BaseException:
            if table is not None:
                try:
                    table.close()
                except BaseException:
                    pass
            elif mapping is not None:
                try:
                    mapping.close()
                except BaseException:
                    pass
            if table is None and pread_fd is not None:
                try:
                    os.close(pread_fd)
                except BaseException:
                    pass
            raise
        assert table is not None
        return table

    @classmethod
    def open_from_artifact(
        cls,
        path: str | Path,
        *,
        validated_artifact: ValidatedPLEArtifact | None = None,
        warm_mode: str = "cold",
        backend: str = "mmap",
        prefetch_max_rows: int = 4096,
        prefetch_chunk_rows: int = 64,
        planner_config: PLELookupPlannerConfig | None = None,
        planner_mode: str = "vectorized",
        planner_direct_threshold: int = 8,
    ) -> MappedPLETable:
        _validate_prefetch_config(prefetch_max_rows, prefetch_chunk_rows)
        resolved_planner = _resolve_planner_config(
            planner_config,
            planner_mode=planner_mode,
            planner_direct_threshold=planner_direct_threshold,
        )
        if backend not in {"mmap", "pread"}:
            raise ValueError(f"unknown PLE backend {backend!r}")
        if validated_artifact is None:
            validated_artifact = validate_ple_artifact_handoff(path)
        elif not isinstance(validated_artifact, ValidatedPLEArtifact):
            raise TypeError("validated_artifact must be a ValidatedPLEArtifact")
        expected_root = Path(path).resolve()
        if Path(validated_artifact.artifact_path) != expected_root:
            raise ValueError("validated PLE artifact handoff path does not match opener path")
        # Check the manifest and payload immediately before opening any mapping
        # or descriptor.  The mapping/fd constructors repeat the payload identity
        # check after open to close the remaining pathname race.
        validated_artifact.assert_current()
        descriptor = validated_artifact.descriptor
        payload = Path(descriptor.shard_path)
        tensor_bytes = descriptor.tensor_bytes
        mapping: MappedFileRange | None = None
        table: MappedPLETable | None = None
        pread_fd: int | None = None
        try:
            if backend == "mmap":
                mapping = MappedFileRange(
                    str(payload),
                    offset=0,
                    length=tensor_bytes,
                    rows=descriptor.rows,
                    row_bytes=descriptor.row_bytes,
                    # ``validate_ple_artifact`` has already verified the immutable
                    # payload before any mapping is acquired.
                    verify_file_sha256=False,
                    expected_file_identity=validated_artifact.payload_identity,
                )
            else:
                pread_fd = _open_validated_pread_fd(
                    payload,
                    offset=0,
                    length=tensor_bytes,
                    expected_file_identity=validated_artifact.payload_identity,
                )
            table = cls(
                descriptor,
                mapping,
                model_shard_paths=(str(payload),),
                prefetch_max_rows=prefetch_max_rows,
                prefetch_chunk_rows=prefetch_chunk_rows,
                planner_config=resolved_planner,
            )
            table.source_kind = "dedicated-artifact"
            table.backend = backend
            table._pread_fd = pread_fd
            table._apply_random_advice()
            if warm_mode == "full-model-warm":
                raise ValueError("artifact warm mode is full-ple-warm")
            table.set_warm_mode(warm_mode)
        except BaseException:
            if table is not None:
                try:
                    table.close()
                except BaseException:
                    pass
            elif mapping is not None:
                try:
                    mapping.close()
                except BaseException:
                    pass
            if table is None and pread_fd is not None:
                try:
                    os.close(pread_fd)
                except BaseException:
                    pass
            raise
        assert table is not None
        return table

    def _prefetch_rows(
        self,
        row_ids: tuple[int, ...],
        cancel_event: threading.Event,
    ) -> int:
        warmed = 0
        if self.mapping is not None:
            for start in range(0, len(row_ids), self._prefetch_chunk_rows):
                if cancel_event.is_set():
                    raise _PrefetchCancelled(warmed)
                stop = min(start + self._prefetch_chunk_rows, len(row_ids))
                chunk = np.fromiter(row_ids[start:stop], dtype=np.int64)
                # Touch the mapped rows to warm the page cache, but never expose
                # the packed bytes through the prefetch handle.
                try:
                    _ = int(self.mapping.rows[chunk].sum(dtype=np.uint64))
                except BaseException as error:
                    raise _PrefetchWorkerError(error, warmed) from error
                warmed += stop - start
        else:
            fd = self._require_pread_fd()
            for row in row_ids:
                if cancel_event.is_set():
                    raise _PrefetchCancelled(warmed)
                try:
                    chunk = os.pread(
                        fd,
                        self.descriptor.row_bytes,
                        self.descriptor.data_offset + row * self.descriptor.row_bytes,
                    )
                except BaseException as error:
                    raise _PrefetchWorkerError(error, warmed) from error
                if len(chunk) != self.descriptor.row_bytes:
                    with self._prefetch_lock:
                        self._short_reads += 1
                    raise _PrefetchWorkerError(
                        ValueError("short PLE positional prefetch read"), warmed
                    )
                warmed += 1
        if cancel_event.is_set():
            raise _PrefetchCancelled(warmed)
        return warmed

    def _finish_prefetch(self, handle: PLEPrefetchHandle, future: Future[int]) -> None:
        with handle._lifecycle_lock:
            warmed = 0
            status = "completed"
            if future.cancelled():
                status = "cancelled"
            else:
                error = future.exception()
                if isinstance(error, _PrefetchCancelled):
                    status = "cancelled"
                    warmed = error.warmed_rows
                elif isinstance(error, _PrefetchWorkerError):
                    status = "failed"
                    warmed = error.warmed_rows
                elif error is not None:
                    status = "failed"
                else:
                    warmed = int(future.result())
                    if handle._cancel_event.is_set():
                        status = "cancelled"
            with self._prefetch_lock:
                if self._prefetch_active is handle:
                    self._prefetch_active = None
                self._prefetch_warmed_rows += warmed
                if status == "completed":
                    self._prefetch_completed += 1
                elif status == "cancelled":
                    self._prefetch_cancelled += 1
                else:
                    self._prefetch_failed += 1
            handle._finalized_event.set()

    def prefetch(self, ids: np.ndarray) -> PLEPrefetchHandle:
        """Warm a bounded set of PLE rows asynchronously.

        The returned handle owns the request lifecycle and never exposes packed
        row data.  At most one request may be active for a table.
        """
        values = np.asarray(ids)
        flat = self._validate_ids(values)
        unique = np.unique(flat)
        row_ids = tuple(int(row) for row in unique)
        requested_rows = int(flat.size)
        with self._prefetch_lock:
            if self._closed or self._closing:
                raise RuntimeError("PLE prefetch table is closed")
            active = self._prefetch_active
            if active is not None and not active.done():
                raise RuntimeError("PLE prefetch already active")
            if len(row_ids) > self._prefetch_max_rows:
                raise ValueError(
                    f"PLE prefetch request has {len(row_ids)} unique rows, "
                    f"exceeds hard bound of {self._prefetch_max_rows} rows"
                )
            self._prefetch_submitted += 1
            self._prefetch_requested_rows += requested_rows
            self._prefetch_unique_rows += len(row_ids)
            cancel_event = threading.Event()
            finalized_event = threading.Event()
            if row_ids:
                if self._prefetch_executor is None:
                    self._prefetch_executor = ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix="ple-prefetch",
                    )
                future = self._prefetch_executor.submit(self._prefetch_rows, row_ids, cancel_event)
            else:
                future = Future()
            handle = PLEPrefetchHandle(
                row_ids, requested_rows, future, cancel_event, finalized_event
            )
            self._prefetch_active = handle
            future.add_done_callback(lambda completed: self._finish_prefetch(handle, completed))
            if not row_ids:
                future.set_result(0)
            return handle

    def set_warm_mode(self, mode: str) -> None:
        self._ensure_io_open()
        if not isinstance(mode, str) or mode not in self._MODES:
            raise ValueError(f"unknown PLE warm mode {mode!r}; expected {sorted(self._MODES)}")
        if mode == "full-model-warm" and self.source_kind == "dedicated-artifact":
            raise ValueError("artifact warm mode is full-ple-warm")

        warmed = 0
        if mode == "cold":
            if self.mapping is not None and hasattr(mmap, "MADV_DONTNEED"):
                self._apply_range_advice(mmap.MADV_DONTNEED)
            elif self.mapping is None and hasattr(os, "POSIX_FADV_DONTNEED"):
                self._apply_range_advice(os.POSIX_FADV_DONTNEED)
        elif mode == "page-cache-warm":
            if self.mapping is not None and hasattr(mmap, "MADV_WILLNEED"):
                self._apply_range_advice(mmap.MADV_WILLNEED)
            elif self.mapping is None and hasattr(os, "POSIX_FADV_WILLNEED"):
                self._apply_range_advice(os.POSIX_FADV_WILLNEED)
        elif mode == "full-ple-warm":
            if self.source_kind == "dedicated-artifact" and self.mapping is not None:
                warmed = _warm_model_files(self._model_shard_paths)
            else:
                warmed = self._warm_ple_range()
        elif mode == "full-model-warm":
            # Explicitly touch every source shard, including ordinary tensors and
            # expert banks.  This is intentionally never the default.
            if self.mapping is None:
                warmed = _warm_pread_files(self._model_shard_paths)
            else:
                warmed = _warm_model_files(self._model_shard_paths)
        self.mode = mode
        if warmed:
            self._full_model_warm_bytes += warmed

    def _validate_ids(self, ids: np.ndarray) -> np.ndarray:
        values = np.asarray(ids)
        if values.dtype.kind not in "iu" or values.dtype.kind == "b":
            raise TypeError(f"PLE row ids must be integers, got {values.dtype}")
        flat = values.astype(np.int64, copy=False).reshape(-1)
        if flat.size and (int(flat.min()) < 0 or int(flat.max()) >= self.descriptor.rows):
            bad = int(flat.min()) if int(flat.min()) < 0 else int(flat.max())
            raise IndexError(f"PLE row {bad} outside [0, {self.descriptor.rows})")
        return flat

    def _ensure_io_open(self) -> None:
        with self._prefetch_lock:
            if self._closed or self._closing:
                raise RuntimeError("PLE table is closed")

    def warm_rows(self, ids: np.ndarray) -> None:
        with self._io_lock:
            self._ensure_io_open()
            self._warm_rows_impl(ids)

    def _warm_rows_impl(self, ids: np.ndarray) -> None:
        flat = self._validate_ids(ids)
        unique = np.unique(flat)
        if unique.size and self.mapping is not None:
            _ = int(self.mapping.rows[unique, 0].sum(dtype=np.uint64))
        elif unique.size:
            fd = self._require_pread_fd()
            for row in unique:
                chunk = os.pread(
                    fd,
                    1,
                    self.descriptor.data_offset + int(row) * self.descriptor.row_bytes,
                )
                if len(chunk) != 1:
                    with self._prefetch_lock:
                        self._short_reads += 1
                    raise ValueError("short PLE positional warm read")
                with self._prefetch_lock:
                    self._targeted_positional_warm_reads += 1
                    self._targeted_warm_rows += 1
            return
        with self._prefetch_lock:
            self._targeted_warm_rows += int(unique.size)

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        return self.lookup_batch(ids)

    def lookup_batch(self, ids: np.ndarray) -> np.ndarray:
        with self._io_lock:
            self._ensure_io_open()
            return self._lookup_batch_impl(ids)

    def _lookup_batch_impl(self, ids: np.ndarray) -> np.ndarray:
        original = np.asarray(ids).shape
        flat = self._validate_ids(ids)
        if not flat.size:
            return np.empty((*original, self.descriptor.elements_per_row), dtype=np.float32)

        planner_started = time.perf_counter_ns()
        if self._planner_config.mode == "adaptive":
            selected_planner = (
                "direct" if flat.size <= self._planner_config.direct_threshold else "vectorized"
            )
        else:
            selected_planner = self._planner_config.mode
        if selected_planner == "vectorized":
            unique, inverse = np.unique(flat, return_inverse=True)
            read_rows = unique
        else:
            unique = None
            inverse = None
            read_rows = flat
        planner_elapsed = max(0, time.perf_counter_ns() - planner_started)
        with self._prefetch_lock:
            self._planner_calls += 1
            self._planner_time_ns += planner_elapsed
            self._planner_selected_mode = selected_planner
            if selected_planner == "direct":
                self._direct_calls += 1
                self._direct_rows += int(flat.size)
            else:
                self._vectorized_calls += 1
                self._vectorized_rows += int(flat.size)

        before_usage = resource.getrusage(resource.RUSAGE_SELF)
        before_storage = _proc_read_bytes()
        if self.mode == "targeted":
            self._warm_rows_impl(flat)
        if self.backend == "pread":
            fd = self._require_pread_fd()
            chunks = []
            for row in read_rows:
                with self._prefetch_lock:
                    self._batch_positional_reads += 1
                    self._application_reads += 1
                chunk = os.pread(
                    fd,
                    self.descriptor.row_bytes,
                    self.descriptor.data_offset + int(row) * self.descriptor.row_bytes,
                )
                with self._prefetch_lock:
                    self._batch_bytes_read += len(chunk)
                    self._application_bytes_read += len(chunk)
                if len(chunk) != self.descriptor.row_bytes:
                    with self._prefetch_lock:
                        self._short_reads += 1
                    raise ValueError("short PLE positional read")
                chunks.append(np.frombuffer(chunk, dtype=np.uint8))
            packed_unique = np.asarray(chunks, dtype=np.uint8).reshape(
                -1, self.descriptor.row_bytes
            )
            if selected_planner == "direct":
                packed = packed_unique
            else:
                packed = packed_unique[inverse].copy()
        else:
            packed = self.mapping.rows[read_rows].copy()
            if selected_planner == "vectorized":
                packed = packed[inverse]
        decoded = self.codec.decode(
            packed,
            rows=int(packed.shape[0]),
            elements_per_row=self.descriptor.elements_per_row,
        )
        expected_shape = (int(packed.shape[0]), self.descriptor.elements_per_row)
        if not isinstance(decoded, np.ndarray) or decoded.shape != expected_shape:
            actual_shape = getattr(decoded, "shape", None)
            raise ValueError(
                f"PLE codec decoder output shape {actual_shape} disagrees with {expected_shape}"
            )
        expected_dtype = np.dtype(self.codec.descriptor.decoded_dtype)
        if decoded.dtype != expected_dtype:
            raise ValueError(
                f"PLE codec decoder output dtype {decoded.dtype} disagrees with {expected_dtype}"
            )
        result = decoded.reshape(*original, self.descriptor.elements_per_row)
        unique_rows = int(unique.size) if unique is not None else len(set(flat.tolist()))
        after_usage = resource.getrusage(resource.RUSAGE_SELF)
        after_storage = _proc_read_bytes()
        with self._prefetch_lock:
            self._lookup_calls += 1
            self._lookup_rows += int(flat.size)
            self._packed_bytes_read += int(flat.size) * self.descriptor.row_bytes
            self._output_bytes += int(result.nbytes)
            self._batch_calls += 1
            self._batch_requested_rows += int(flat.size)
            self._batch_unique_rows += unique_rows
            self._batch_duplicate_rows += int(flat.size - unique_rows)
            if selected_planner == "vectorized":
                self._batch_sorted_rows += unique_rows
            if self.backend != "pread":
                self._batch_bytes_read += int(read_rows.size) * self.descriptor.row_bytes
                self._application_reads += int(read_rows.size)
                self._application_bytes_read += int(read_rows.size) * self.descriptor.row_bytes
            self._minor_faults += max(0, int(after_usage.ru_minflt - before_usage.ru_minflt))
            self._major_faults += max(0, int(after_usage.ru_majflt - before_usage.ru_majflt))
            if before_storage is not None and after_storage is not None:
                self._storage_read_bytes += max(0, after_storage - before_storage)
        return result

    def telemetry(self) -> dict[str, int | str | bool | None]:
        with self._io_lock, self._prefetch_lock:
            resident_pages = (
                None if self.mapping is None or self._closed else self.mapping.resident_pages()
            )
            active = self._prefetch_active is not None and not self._prefetch_active.done()
            return {
                "mode": self.mode,
                "source_kind": self.source_kind,
                "mapped_bytes": 0 if self.mapping is None else self.mapping.length,
                "resident_pages": resident_pages,
                "resident_bytes": (
                    None if resident_pages is None else resident_pages * mmap.PAGESIZE
                ),
                "lookup_calls": self._lookup_calls,
                "lookup_rows": self._lookup_rows,
                "packed_bytes_read": self._packed_bytes_read,
                "output_bytes": self._output_bytes,
                "minor_faults": self._minor_faults,
                "major_faults": self._major_faults,
                "storage_read_bytes": self._storage_read_bytes,
                "targeted_warm_rows": self._targeted_warm_rows,
                "full_model_warm_bytes": self._full_model_warm_bytes,
                "backend": self.backend,
                "codec_id": self.codec.descriptor.codec_id,
                "codec_version": self.codec.descriptor.version,
                "codec_identity": self.codec.descriptor.identity,
                "planner_mode": self._planner_config.mode,
                "planner_direct_threshold": self._planner_config.direct_threshold,
                "planner_selected_mode": self._planner_selected_mode,
                "planner_calls": self._planner_calls,
                "planner_time_ns": self._planner_time_ns,
                "direct_calls": self._direct_calls,
                "direct_rows": self._direct_rows,
                "vectorized_calls": self._vectorized_calls,
                "vectorized_rows": self._vectorized_rows,
                "application_reads": self._application_reads,
                "application_bytes_read": self._application_bytes_read,
                "batch_calls": self._batch_calls,
                "batch_requested_rows": self._batch_requested_rows,
                "batch_unique_rows": self._batch_unique_rows,
                "batch_positional_reads": self._batch_positional_reads,
                "batch_duplicate_rows": self._batch_duplicate_rows,
                "batch_sorted_rows": self._batch_sorted_rows,
                "batch_bytes_read": self._batch_bytes_read,
                "short_reads": self._short_reads,
                "targeted_positional_warm_reads": self._targeted_positional_warm_reads,
                "advice": self._advice,
                "advice_applied": self._advice_applied,
                "advice_error": self._advice_error,
                "prefetch_active": active,
                "prefetch_submitted": self._prefetch_submitted,
                "prefetch_completed": self._prefetch_completed,
                "prefetch_cancelled": self._prefetch_cancelled,
                "prefetch_failed": self._prefetch_failed,
                "prefetch_requested_rows": self._prefetch_requested_rows,
                "prefetch_unique_rows": self._prefetch_unique_rows,
                "prefetch_warmed_rows": self._prefetch_warmed_rows,
            }

    def close(self) -> None:
        with self._prefetch_lock:
            if self._closed:
                return
            if self._closing:
                raise RuntimeError("PLE table close is already in progress")
            self._closing = True
            active = self._prefetch_active
            executor = self._prefetch_executor
            if active is not None and not active.done():
                active.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            with self._prefetch_lock:
                self._prefetch_executor = None
        failure: BaseException | None = None
        with self._io_lock:
            if self.mapping is not None:
                try:
                    self.mapping.close()
                except BaseException as error:
                    failure = error
            if self._pread_fd is not None:
                try:
                    os.close(self._pread_fd)
                except BaseException as error:
                    failure = failure or error
                self._pread_fd = None
        with self._prefetch_lock:
            # Match the pre-prefetch close contract: resource cleanup is a
            # one-shot operation even when one close reports an error.
            self._closed = True
            self._closing = False
        if failure is not None:
            raise failure

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
        self._closed = False
        self._cpu_bridge_claimed = False
        self._cpu_bridge_owner_token: object | None = None
        self._cpu_bridge_lock = threading.Lock()

    @property
    def closed(self) -> bool:
        """Whether the owned expert and PLE mappings have been released."""
        with self._cpu_bridge_lock:
            return self._closed

    @property
    def cpu_bridge_claimed(self) -> bool:
        """Whether ownership was permanently transferred to the CPU GGUF bridge."""
        with self._cpu_bridge_lock:
            return self._cpu_bridge_claimed

    def claim_cpu_bridge(self) -> object:
        """Permanently transfer this host's ownership to one CPU GGUF bundle."""
        with self._cpu_bridge_lock:
            if self._cpu_bridge_claimed:
                raise RuntimeError("Qwen GGUF host is already claimed by a CPU expert bundle")
            if self._closed:
                raise RuntimeError("Qwen GGUF host mappings are closed")
            token = object()
            self._cpu_bridge_claimed = True
            self._cpu_bridge_owner_token = token
            return token

    def memory_report(self) -> dict[str, int]:
        report = host_memory_report(self.layout)
        return {
            **report,
            "anonymous_model_bytes": report["anonymous_host_source_bytes"],
            "pinned_bytes": report["pinned_host_source_bytes"],
        }

    def close(self) -> None:
        with self._cpu_bridge_lock:
            if self._closed:
                return
            if self._cpu_bridge_claimed:
                raise RuntimeError(
                    "Qwen GGUF host is owned by a CPU expert bundle; use its close method"
                )
            self._close_resources_locked()

    def close_cpu_bridge(self, owner_token: object) -> None:
        """Close host mappings using the token issued by :meth:`claim_cpu_bridge`."""
        with self._cpu_bridge_lock:
            if not self._cpu_bridge_claimed or owner_token is not self._cpu_bridge_owner_token:
                raise RuntimeError("invalid CPU expert bundle owner token")
            self._close_resources_locked()

    def _close_resources_locked(self) -> None:
        if self._closed:
            return
        failure: BaseException | None = None
        for owned_resource in (self.ple, self.experts):
            try:
                owned_resource.close()
            except BaseException as error:
                failure = failure or error
        if failure is not None:
            raise failure
        self._closed = True

    def __enter__(self) -> QwenGGUFHostWeights:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def open_qwen_host_weights(
    path: str | Path,
    *,
    supported_expert_types: Collection[int] | None = None,
    ple_artifact_path: str | Path | None = None,
    ple_artifact_validation: ValidatedPLEArtifact | None = None,
    ple_backend: str = "mmap",
    ple_warm_mode: str = "cold",
    ple_prefetch_max_rows: int = 4096,
    ple_prefetch_chunk_rows: int = 64,
    ple_planner_config: PLELookupPlannerConfig | None = None,
    ple_planner_mode: str = "vectorized",
    ple_planner_direct_threshold: int = 8,
) -> QwenGGUFHostWeights:
    """Open expert mappings and exactly one selected PLE mapping.

    ``ple_artifact_path`` selects the dedicated serving artifact and is validated against the
    source GGUF descriptor before the host owner is returned. When omitted, the embedded GGUF
    PLE range remains available to standalone compatibility callers.
    """
    _validate_prefetch_config(ple_prefetch_max_rows, ple_prefetch_chunk_rows)
    if ple_artifact_validation is not None and ple_artifact_path is None:
        raise ValueError("PLE artifact validation requires ple_artifact_path")
    resolved_planner = _resolve_planner_config(
        ple_planner_config,
        planner_mode=ple_planner_mode,
        planner_direct_threshold=ple_planner_direct_threshold,
    )
    layout = inspect_qwen_host_layout(
        path,
        supported_expert_types=supported_expert_types,
    )
    experts = MappedExpertBanks(layout.experts)
    ple: MappedPLETable | None = None
    try:
        if ple_artifact_path is None:
            ple = MappedPLETable.open_from_gguf(
                path,
                warm_mode=ple_warm_mode,
                backend=ple_backend,
                prefetch_max_rows=ple_prefetch_max_rows,
                prefetch_chunk_rows=ple_prefetch_chunk_rows,
                planner_config=resolved_planner,
            )
        else:
            ple = MappedPLETable.open_from_artifact(
                ple_artifact_path,
                validated_artifact=ple_artifact_validation,
                warm_mode=ple_warm_mode,
                backend=ple_backend,
                prefetch_max_rows=ple_prefetch_max_rows,
                prefetch_chunk_rows=ple_prefetch_chunk_rows,
                planner_config=resolved_planner,
            )
            source = layout.ple
            mapped = ple.descriptor
            identity = (
                "tensor_name",
                "quant_type",
                "quant_name",
                "rows",
                "elements_per_row",
                "row_bytes",
                "tensor_bytes",
                "codec",
            )
            mismatches = [
                name
                for name in identity
                if getattr(mapped, name) != getattr(source, name)
            ]
            if mismatches:
                raise ValueError(
                    "dedicated PLE artifact does not match source GGUF descriptor: "
                    + ", ".join(mismatches)
                )
    except BaseException:
        try:
            if ple is not None:
                try:
                    ple.close()
                except BaseException:
                    pass
        finally:
            try:
                experts.close()
            except BaseException:
                pass
        raise
    assert ple is not None
    return QwenGGUFHostWeights(layout, experts, ple)


__all__ = [
    "IQ4_NL_CODEC",
    "IQ4_NL_CODEC_DESCRIPTOR",
    "PLE_CODEC_REGISTRY",
    "ExpertBankDescriptor",
    "ExpertSlotPool",
    "GGUFExpertLayout",
    "MappedPLETable",
    "PLECodec",
    "PLECodecDescriptor",
    "PLECodecRegistry",
    "PLEFileIdentity",
    "PLELookupPlannerConfig",
    "PLEPrefetchHandle",
    "QwenGGUFHostWeights",
    "QwenHostLayout",
    "ValidatedPLEArtifact",
    "dequantize_iq4_nl",
    "expert_layout_from_census",
    "host_layout_document",
    "host_memory_report",
    "host_memory_report_from_census",
    "inspect_qwen_host_layout",
    "open_qwen_host_weights",
    "validate_ple_artifact",
    "validate_ple_artifact_handoff",
]
