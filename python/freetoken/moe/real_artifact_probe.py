"""H0 probe for selected Qwen3.8 GGUF expert byte ranges.

The probe is intentionally narrower than a model loader.  It fetches one expert
from each of the selected layer's gate, up and down banks with HTTP Range, builds
the existing CPU expert ABI over those three byte ranges, and records scalar and
native A/B runs against an independent gguf-py dense FP32 SwiGLU oracle.  It never
downloads or hashes a complete shard.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen

import numpy as np

from freetoken.gguf_types import BLOCK_SHAPE, GGML_NAME
from freetoken.moe.cpu_abi import CpuExpertDescriptor, CpuExpertLayout
from freetoken.moe.q4_k import Q4KExecutor

DEFAULT_VARIANT = "UD-Q4_K_XL"
DEFAULT_LAYER = 0
DEFAULT_EXPERT = 0
DEFAULT_REPEATS = 3
DEFAULT_WARMUP = 1
DEFAULT_SEED = 1600
DEFAULT_Q3_VARIANT = "UD-Q3_K_XL"
DEFAULT_Q3_TRIAD = ((0, 0), (23, 255), (47, 511))
SUPPORTED_EXPERT_TYPES = frozenset({7, 8, 12, 13, 18, 20, 23})
_CONTENT_RANGE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)
_PROJECTIONS = ("gate", "up", "down")
GGUF_ORACLE_PACKAGE = "gguf"
GGUF_ORACLE_NAME = "gguf-py"
GGUF_ORACLE_VERSION = "0.19.0"
GGUF_ORACLE_OPERATION = "dequantize + FP32 dense SwiGLU"
ORACLE_RTOL = 5e-4
ORACLE_ATOL = 5e-4


class ArtifactProbeError(RuntimeError):
    """Raised when an artifact range or descriptor cannot be proven safe."""


def gguf_oracle_identity() -> dict[str, str]:
    """Return the pinned independent oracle identity without importing gguf."""
    return {
        "name": GGUF_ORACLE_NAME,
        "package": GGUF_ORACLE_PACKAGE,
        "version": GGUF_ORACLE_VERSION,
        "operation": GGUF_ORACLE_OPERATION,
    }


def _load_gguf_oracle() -> Any:
    """Load exactly the dependency version used to establish the oracle contract."""
    try:
        import gguf
    except (ImportError, ModuleNotFoundError) as error:
        raise ArtifactProbeError(
            "gguf-py==0.19.0 is required for a real expert probe; install the pinned "
            "requirements/cpu.lock environment"
        ) from error
    try:
        version = importlib.metadata.version(GGUF_ORACLE_PACKAGE)
    except importlib.metadata.PackageNotFoundError as error:
        raise ArtifactProbeError(
            "gguf-py==0.19.0 is required for a real expert probe, but its distribution "
            "metadata is unavailable"
        ) from error
    if version != GGUF_ORACLE_VERSION:
        raise ArtifactProbeError(
            f"real expert probe requires gguf-py=={GGUF_ORACLE_VERSION}, got {version}"
        )
    if not callable(getattr(gguf, "dequantize", None)):
        raise ArtifactProbeError(f"gguf-py=={GGUF_ORACLE_VERSION} does not expose gguf.dequantize")
    if not hasattr(gguf, "GGMLQuantizationType"):
        raise ArtifactProbeError(
            f"gguf-py=={GGUF_ORACLE_VERSION} does not expose GGMLQuantizationType"
        )
    return gguf


@dataclass(frozen=True)
class RangeResponse:
    """A transport response with HTTP status, headers and the exact body bytes."""

    status: int
    headers: Mapping[str, str]
    body: bytes


class RangeTransport(Protocol):
    """Inclusive-end range transport used by :class:`RangeFetcher`."""

    def fetch(self, url: str, start: int, end: int) -> RangeResponse: ...


class UrllibRangeTransport:
    """Small standard-library HTTP Range transport for the immutable HF URL."""

    def __init__(self, *, opener: Callable[..., Any] = urlopen, timeout: float = 60.0) -> None:
        self.opener = opener
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = float(timeout)

    def fetch(self, url: str, start: int, end: int) -> RangeResponse:
        request = Request(url, headers={"Range": f"bytes={start}-{end}"})
        try:
            response = self.opener(request, timeout=self.timeout)
            with response:
                headers = {str(key): str(value) for key, value in response.headers.items()}
                status_value = getattr(response, "status", None)
                status = int(response.getcode() if status_value is None else status_value)
                if status != 206:
                    raise ArtifactProbeError(
                        f"{url}: expected HTTP 206 for Range bytes={start}-{end}, got {status}"
                    )
                content_range = _header(headers, "Content-Range")
                if content_range is None:
                    raise ArtifactProbeError(f"{url}: response is missing Content-Range")
                match = _CONTENT_RANGE.fullmatch(content_range.strip())
                if match is None:
                    raise ArtifactProbeError(f"{url}: malformed Content-Range {content_range!r}")
                actual_start, actual_end, actual_total = (int(value) for value in match.groups())
                if (actual_start, actual_end) != (start, end) or actual_total <= end:
                    raise ArtifactProbeError(
                        f"{url}: Content-Range {content_range!r} does not equal "
                        f"bytes {start}-{end}/<total>"
                    )
                content_length = _header(headers, "Content-Length")
                if content_length is not None:
                    try:
                        declared_length = int(content_length)
                    except ValueError as error:
                        raise ArtifactProbeError(
                            f"{url}: invalid Content-Length {content_length!r}"
                        ) from error
                    if declared_length > end - start + 1:
                        raise ArtifactProbeError(
                            f"{url}: Content-Length exceeds requested range length "
                            f"{end - start + 1}"
                        )
                # Never allow a server that ignores Range to stream a complete shard.
                body = response.read(end - start + 2)
        except ArtifactProbeError:
            raise
        except Exception as error:
            raise ArtifactProbeError(f"HTTP Range request failed for {url}: {error}") from error
        return RangeResponse(status=status, headers=headers, body=body)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value)
    return None


def _validated_content_range(
    response: RangeResponse,
    *,
    url: str,
    start: int,
    size: int,
    expected_total: int,
) -> bytes:
    if response.status != 206:
        raise ArtifactProbeError(
            f"{url}: expected HTTP 206 for Range bytes={start}-{start + size - 1}, "
            f"got {response.status}"
        )
    content_range = _header(response.headers, "Content-Range")
    if content_range is None:
        raise ArtifactProbeError(f"{url}: response is missing Content-Range")
    match = _CONTENT_RANGE.fullmatch(content_range.strip())
    if match is None:
        raise ArtifactProbeError(f"{url}: malformed Content-Range {content_range!r}")
    actual_start, actual_end, actual_total = (int(value) for value in match.groups())
    expected_end = start + size - 1
    if (actual_start, actual_end, actual_total) != (start, expected_end, expected_total):
        raise ArtifactProbeError(
            f"{url}: Content-Range {content_range!r} does not equal "
            f"bytes {start}-{expected_end}/{expected_total}"
        )
    content_length = _header(response.headers, "Content-Length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as error:
            raise ArtifactProbeError(f"{url}: invalid Content-Length {content_length!r}") from error
        if declared_length != size:
            raise ArtifactProbeError(
                f"{url}: content length {declared_length} does not equal requested {size}"
            )
    body = bytes(response.body)
    if len(body) != size:
        raise ArtifactProbeError(
            f"{url}: range body length {len(body)} does not equal requested {size}"
        )
    return body


class RangeFetcher:
    """Fetch and verify exact inclusive HTTP ranges, optionally cached offline."""

    def __init__(
        self,
        transport: RangeTransport | Callable[[str, int, int], RangeResponse],
        *,
        cache_dir: Path | None = None,
    ) -> None:
        self.transport = transport
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_cache_hit = False
        self._cleanup_cache_temps()

    def _cleanup_cache_temps(self) -> None:
        """Remove only this fetcher's interrupted temporary cache files."""
        if self.cache_dir is None or not self.cache_dir.exists():
            return
        try:
            entries = tuple(self.cache_dir.iterdir())
        except OSError:
            return
        for entry in entries:
            name = entry.name
            if entry.is_file() and re.fullmatch(r"\.[0-9a-f]{64}\.[A-Za-z0-9_-]+(?:\.tmp)?", name):
                entry.unlink(missing_ok=True)

    def _cache_stem(self, url: str, start: int, size: int, expected_total: int) -> str:
        key = f"{url}\0{start}\0{size}\0{expected_total}".encode()
        return hashlib.sha256(key).hexdigest()

    def _load_cache(
        self, url: str, start: int, size: int, expected_total: int
    ) -> RangeResponse | None:
        self._cleanup_cache_temps()
        if self.cache_dir is None:
            return None
        stem = self._cache_stem(url, start, size, expected_total)
        body_path = self.cache_dir / f"{stem}.bin"
        metadata_path = self.cache_dir / f"{stem}.json"
        if not body_path.exists() and not metadata_path.exists():
            return None
        if not body_path.exists() or not metadata_path.exists():
            raise ArtifactProbeError(f"range cache entry {stem} is incomplete")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            body_size = body_path.stat().st_size
            if body_size != size:
                raise ArtifactProbeError(
                    f"range cache entry {stem} length {body_size} does not equal requested {size}"
                )
            body = body_path.read_bytes()
        except ArtifactProbeError:
            raise
        except (OSError, ValueError) as error:
            raise ArtifactProbeError(f"range cache entry {stem} cannot be read") from error
        expected_digest = hashlib.sha256(body).hexdigest()
        if metadata.get("url") != url or metadata.get("start") != start:
            raise ArtifactProbeError(f"range cache entry {stem} has mismatched request metadata")
        if metadata.get("size") != size or metadata.get("total") != expected_total:
            raise ArtifactProbeError(f"range cache entry {stem} has mismatched range metadata")
        if metadata.get("sha256") != expected_digest:
            raise ArtifactProbeError(f"range cache entry {stem} checksum failure")
        if len(body) != size:
            raise ArtifactProbeError(
                f"range cache entry {stem} length {len(body)} does not equal requested {size}"
            )
        end = start + size - 1
        return RangeResponse(
            status=206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{expected_total}",
                "Content-Length": str(size),
            },
            body=body,
        )

    def _write_cache(
        self, url: str, start: int, size: int, expected_total: int, body: bytes
    ) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        stem = self._cache_stem(url, start, size, expected_total)
        metadata = {
            "url": url,
            "start": start,
            "size": size,
            "total": expected_total,
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        # A cache write is best-effort transactional: a killed probe leaves either
        # no entry or a complete body/metadata pair, never a partial model range.
        with tempfile.NamedTemporaryFile(
            dir=self.cache_dir, prefix=f".{stem}.", delete=False
        ) as tmp:
            tmp.write(body)
            body_tmp = Path(tmp.name)
        metadata_tmp = self.cache_dir / f".{stem}.json.tmp"
        try:
            metadata_tmp.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(body_tmp, self.cache_dir / f"{stem}.bin")
            os.replace(metadata_tmp, self.cache_dir / f"{stem}.json")
        except OSError as error:
            # A body replace followed by a metadata failure must not leave a
            # permanently incomplete pair that poisons later online probes.
            body_path = self.cache_dir / f"{stem}.bin"
            metadata_path = self.cache_dir / f"{stem}.json"
            # Remove both sides even when an older metadata file exists: a
            # failed replacement must never leave a mixed old/new pair.
            body_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise ArtifactProbeError(f"cannot commit range cache entry {stem}: {error}") from error
        finally:
            body_tmp.unlink(missing_ok=True)
            metadata_tmp.unlink(missing_ok=True)

    def _discard_cache_entry(self, url: str, start: int, size: int, expected_total: int) -> None:
        """Remove an incomplete or stale pair before an online re-fetch."""
        if self.cache_dir is None:
            return
        stem = self._cache_stem(url, start, size, expected_total)
        for suffix in (".bin", ".json"):
            (self.cache_dir / f"{stem}{suffix}").unlink(missing_ok=True)

    def fetch(
        self,
        url: str,
        start: int,
        size: int,
        *,
        expected_total: int,
        offline: bool = False,
    ) -> RangeResponse:
        if start < 0 or size <= 0 or expected_total <= 0 or start + size > expected_total:
            raise ArtifactProbeError(
                f"invalid range [{start}, {start + size}) for shard size {expected_total}"
            )
        try:
            cached = self._load_cache(url, start, size, expected_total)
        except ArtifactProbeError:
            if offline:
                raise
            self._discard_cache_entry(url, start, size, expected_total)
            cached = None
        if cached is not None:
            self.cache_hits += 1
            self.last_cache_hit = True
            return cached
        if offline:
            raise ArtifactProbeError(
                f"offline range cache miss for {url} bytes={start}-{start + size - 1}"
            )
        self.cache_misses += 1
        self.last_cache_hit = False
        try:
            if callable(self.transport):
                response = self.transport(url, start, start + size - 1)
            else:
                response = self.transport.fetch(url, start, start + size - 1)
        except ArtifactProbeError:
            raise
        except Exception as error:
            raise ArtifactProbeError(f"range transport failed for {url}: {error}") from error
        if not isinstance(response, RangeResponse):
            raise ArtifactProbeError("range transport must return RangeResponse")
        body = _validated_content_range(
            response,
            url=url,
            start=start,
            size=size,
            expected_total=expected_total,
        )
        self._write_cache(url, start, size, expected_total, body)
        return RangeResponse(response.status, response.headers, body)


class _PackedRangeSource:
    """One-expert source whose range metadata remains tied to the artifact offset."""

    def __init__(
        self, packed: bytes, *, artifact_offset: int, output_dim: int, row_bytes: int
    ) -> None:
        packed_values = np.frombuffer(packed, dtype=np.uint8)
        expected = output_dim * row_bytes
        if packed_values.size != expected:
            raise ArtifactProbeError(
                f"packed expert source has {packed_values.size} bytes, expected {expected}"
            )
        # HTTP response bytes are not required to have a SIMD-friendly address.
        # Copy only this selected expert range into an aligned, read-only view;
        # complete shards are never materialized.
        storage = np.empty(expected + 31, dtype=np.uint8)
        aligned = storage[(-int(storage.ctypes.data)) % 32 :][:expected]
        np.copyto(aligned, packed_values)
        aligned.setflags(write=False)
        values = aligned.reshape(1, output_dim, row_bytes)
        self.values = values
        self._storage = storage
        self.range_offset = artifact_offset
        self.range_size = expected
        self.source_address = int(values.__array_interface__["data"][0])

    def expert_packed(self, expert: int) -> np.ndarray:
        if expert != 0:
            raise IndexError(f"selected range contains only remapped expert 0, got {expert}")
        return self.values[0]


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ArtifactProbeError(f"cannot read {label} metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactProbeError(f"{label} metadata must be a JSON object")
    return value


def _validate_metadata(
    manifest: Mapping[str, Any], census: Mapping[str, Any], variant: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        revision = str(manifest["revision"])
        repository = str(manifest["repository"])
        manifest_variant = manifest["variants"][variant]
        manifest_shard_list = tuple(manifest_variant["shards"])
        census_shard_list = tuple(census["shards"])
        manifest_shards = {str(item["name"]): item for item in manifest_shard_list}
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactProbeError(
            f"invalid manifest/census metadata for {variant}: {error}"
        ) from error
    if census.get("evidence_status") != "artifact-metadata":
        raise ArtifactProbeError(
            "Qwen census must be the pinned artifact-metadata fixture; payload census is not "
            "accepted"
        )
    if len(manifest_shard_list) != len(census_shard_list):
        raise ArtifactProbeError("manifest and census shard counts disagree")
    if len(manifest_shards) != len(manifest_shard_list):
        raise ArtifactProbeError("manifest contains duplicate shard names")
    census_names = {str(item["name"]) for item in census_shard_list}
    if len(census_names) != len(census_shard_list):
        raise ArtifactProbeError("census contains duplicate shard names")
    for index, (declared, observed) in enumerate(
        zip(manifest_shard_list, census_shard_list, strict=True)
    ):
        if str(declared["name"]) != str(observed["name"]):
            raise ArtifactProbeError(f"manifest and census shard name/index {index} disagree")
        for key in ("size", "sha256"):
            if str(declared[key]) != str(observed[key]):
                raise ArtifactProbeError(
                    f"shard {declared['name']}: manifest and census {key} disagree"
                )
    if int(census.get("shard_count", -1)) != len(manifest_shards):
        raise ArtifactProbeError("census shard_count disagrees with manifest")
    source = {
        "repository": repository,
        "revision": revision,
        "variant": variant,
        "base_url": f"https://huggingface.co/{repository}/resolve/{revision}/{variant}",
    }
    return source, manifest_shards


def _tensor_records(census: Mapping[str, Any], layer: int) -> dict[str, dict[str, Any]]:
    wanted = {
        f"blk.{layer}.ffn_{projection}_exps.weight": projection for projection in _PROJECTIONS
    }
    records: dict[str, dict[str, Any]] = {}
    for record in census.get("tensors", ()):
        name = str(record.get("name"))
        if name not in wanted:
            continue
        projection = wanted[name]
        if projection in records:
            raise ArtifactProbeError(f"layer {layer}: duplicate tensor record {name}")
        records[projection] = record
    if set(records) != set(_PROJECTIONS):
        missing = sorted(set(_PROJECTIONS) - set(records))
        raise ArtifactProbeError(f"layer {layer}: missing expert projections {missing}")
    return records


def _record_geometry(
    record: Mapping[str, Any], *, shard_size: int, layer: int, projection: str, expert: int
) -> dict[str, Any]:
    try:
        quant_type = int(record["quant_type"])
        quant_name = str(record["quant_name"])
        shape = tuple(int(value) for value in record["shape"])
        row_bytes = int(record["row_bytes"])
        tensor_bytes = int(record["nbytes"])
        tensor_offset = int(record["offset"])
        shard_index = int(record["shard_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactProbeError(f"layer {layer} {projection}: malformed census record") from error
    if quant_type not in SUPPORTED_EXPERT_TYPES or GGML_NAME.get(quant_type) != quant_name:
        raise ArtifactProbeError(
            f"layer {layer} {projection}: unsupported or inconsistent quant "
            f"{quant_type}/{quant_name}"
        )
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ArtifactProbeError(f"layer {layer} {projection}: invalid expert shape {shape}")
    experts, output_dim, input_dim = shape
    try:
        block_elements, block_bytes = BLOCK_SHAPE[quant_type]
    except KeyError as error:
        raise ArtifactProbeError(
            f"layer {layer} {projection}: unknown quant type {quant_type}"
        ) from error
    expected_row_bytes = input_dim // block_elements * block_bytes
    if input_dim % block_elements or row_bytes != expected_row_bytes:
        raise ArtifactProbeError(
            f"layer {layer} {projection}: row stride {row_bytes} does not match "
            f"{input_dim} elements of {quant_name}"
        )
    bytes_per_expert = output_dim * row_bytes
    try:
        declared_rows = int(record.get("rows", experts * output_dim))
    except (TypeError, ValueError) as error:
        raise ArtifactProbeError(f"layer {layer} {projection}: malformed rows field") from error
    if declared_rows != experts * output_dim:
        raise ArtifactProbeError(
            f"layer {layer} {projection}: rows {declared_rows} disagree with shape {shape}"
        )
    if tensor_bytes != experts * bytes_per_expert:
        raise ArtifactProbeError(
            f"layer {layer} {projection}: tensor bytes disagree with shape/stride"
        )
    if tensor_offset < 0 or tensor_offset + tensor_bytes > shard_size:
        raise ArtifactProbeError(
            f"layer {layer} {projection}: tensor bounds [{tensor_offset}, "
            f"{tensor_offset + tensor_bytes}) exceed shard size {shard_size}"
        )
    if not 0 <= expert < experts:
        raise ArtifactProbeError(f"expert {expert} outside layer {layer} range [0, {experts})")
    expert_offset = tensor_offset + expert * bytes_per_expert
    if expert_offset + bytes_per_expert > tensor_offset + tensor_bytes:
        raise ArtifactProbeError(
            f"layer {layer} {projection}: selected expert offset is out of bounds"
        )
    return {
        "quant_type": quant_type,
        "quant_name": quant_name,
        "experts": experts,
        "output_dim": output_dim,
        "input_dim": input_dim,
        "row_bytes": row_bytes,
        "bytes_per_expert": bytes_per_expert,
        "tensor_bytes": tensor_bytes,
        "tensor_offset": tensor_offset,
        "expert_offset": expert_offset,
        "shard_index": shard_index,
    }


def build_probe_layout(
    manifest: Mapping[str, Any],
    census: Mapping[str, Any],
    *,
    layer: int,
    expert: int,
    sources: Mapping[str, bytes],
    variant: str = DEFAULT_VARIANT,
) -> CpuExpertLayout:
    """Validate selected metadata/ranges and construct a one-expert CPU ABI layout."""
    _, manifest_shards = _validate_metadata(manifest, census, variant)
    if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)) or int(layer) < 0:
        raise ArtifactProbeError(f"layer must be a non-negative integer, got {layer!r}")
    if isinstance(expert, bool) or not isinstance(expert, (int, np.integer)) or int(expert) < 0:
        raise ArtifactProbeError(f"expert must be a non-negative integer, got {expert!r}")
    layer = int(layer)
    expert = int(expert)
    records = _tensor_records(census, layer)
    descriptors: list[CpuExpertDescriptor] = []
    geometries: dict[str, dict[str, Any]] = {}
    for projection in _PROJECTIONS:
        record = records[projection]
        try:
            shard_index = int(record["shard_index"])
            if shard_index < 0:
                raise IndexError(shard_index)
            shard = census["shards"][shard_index]
            shard_size = int(manifest_shards[str(shard["name"])]["size"])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ArtifactProbeError(f"layer {layer} {projection}: invalid shard index") from error
        geometry = _record_geometry(
            record,
            shard_size=shard_size,
            layer=layer,
            projection=projection,
            expert=expert,
        )
        geometries[projection] = geometry
        if projection not in sources:
            raise ArtifactProbeError(f"missing fetched source for {projection}")
        packed = bytes(sources[projection])
        if len(packed) != geometry["bytes_per_expert"]:
            raise ArtifactProbeError(
                f"layer {layer} {projection}: fetched {len(packed)} bytes, expected "
                f"{geometry['bytes_per_expert']}"
            )
        source = _PackedRangeSource(
            packed,
            artifact_offset=geometry["expert_offset"],
            output_dim=geometry["output_dim"],
            row_bytes=geometry["row_bytes"],
        )
        descriptors.append(
            CpuExpertDescriptor(
                layer_id=layer,
                projection=projection,
                quant_type=geometry["quant_type"],
                quant_name=geometry["quant_name"],
                num_experts=1,
                output_dim=geometry["output_dim"],
                input_dim=geometry["input_dim"],
                rows_per_expert=geometry["output_dim"],
                row_stride_bytes=geometry["row_bytes"],
                expert_stride_bytes=geometry["bytes_per_expert"],
                tensor_bytes=geometry["bytes_per_expert"],
                source_offset=geometry["expert_offset"],
                source=source,
            )
        )
    gate, up, down = (geometries[projection] for projection in _PROJECTIONS)
    if gate["experts"] != up["experts"] or gate["experts"] != down["experts"]:
        raise ArtifactProbeError(f"layer {layer}: projection expert counts disagree")
    if (gate["output_dim"], gate["input_dim"]) != (up["output_dim"], up["input_dim"]):
        raise ArtifactProbeError(f"layer {layer}: gate/up geometry disagrees")
    if (down["output_dim"], down["input_dim"]) != (gate["input_dim"], gate["output_dim"]):
        raise ArtifactProbeError(f"layer {layer}: down geometry is not transposed")
    return CpuExpertLayout(tuple(descriptors), top_k=2)


def _descriptor_summary(layout: CpuExpertLayout, *, expert: int) -> list[dict[str, Any]]:
    return [
        {
            "layer": descriptor.layer_id,
            "projection": descriptor.projection,
            "quant_type": descriptor.quant_type,
            "quant_name": descriptor.quant_name,
            "selected_expert": expert,
            "num_experts_remapped": descriptor.num_experts,
            "output_dim": descriptor.output_dim,
            "input_dim": descriptor.input_dim,
            "rows_per_expert": descriptor.rows_per_expert,
            "row_stride_bytes": descriptor.row_stride_bytes,
            "expert_stride_bytes": descriptor.expert_stride_bytes,
            "tensor_bytes": descriptor.tensor_bytes,
            "artifact_offset": descriptor.source_offset,
            "artifact_end": descriptor.source_offset + descriptor.tensor_bytes,
        }
        for descriptor in layout.descriptors
    ]


def _run_mode(
    layout: CpuExpertLayout,
    hidden: np.ndarray,
    expert_ids: np.ndarray,
    weights: np.ndarray,
    *,
    mode: str,
    repeats: int,
    warmup: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(layout.layers) != 1:
        raise ArtifactProbeError(
            f"real-artifact probe requires exactly one layer, got {layout.layers}"
        )
    layer_id = layout.layers[0]
    executor = Q4KExecutor(layout, mode=mode, required_alignment=32)
    try:
        executor.prepare(hidden.shape[0], expert_ids.shape[1])
        for _ in range(warmup):
            executor.execute(layer_id, hidden, expert_ids, weights)
        elapsed: list[int] = []
        telemetry: list[dict[str, Any]] = []
        output: np.ndarray | None = None
        for _ in range(repeats):
            started = time.perf_counter_ns()
            result = executor.execute(layer_id, hidden, expert_ids, weights)
            elapsed.append(time.perf_counter_ns() - started)
            telemetry.append(result.telemetry.as_dict())
            if output is None:
                output = np.array(result.output, dtype=np.float32, copy=True)
        assert output is not None
        primitive = executor.primitive
        mixed_primitive = executor.mixed_primitive
        actual_avx2 = any(
            "avx2" in item["backend"] or any("avx2" in kernel for kernel in item["kernel_census"])
            for item in telemetry
        )
        return output, {
            "requested_mode": mode,
            "repeats": int(repeats),
            "warmup": int(warmup),
            "selected_backend": executor.backend,
            "q4k_isa": primitive.isa,
            "q4k_fallback_reason": primitive.fallback_reason,
            "mixed_isa": mixed_primitive.isa,
            "mixed_fallback_reason": mixed_primitive.fallback_reason,
            "actual_avx2": actual_avx2,
            "raw_elapsed_ns": elapsed,
            "telemetry": telemetry,
        }
    finally:
        executor.close()


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _packed_source_bytes(
    layout: CpuExpertLayout,
    projection: str,
    fetched: bytes,
) -> bytes:
    """Validate that the oracle sees precisely the bytes returned by RangeFetcher."""
    descriptor = layout.descriptor(layout.layers[0], projection)
    expected_size = descriptor.output_dim * descriptor.row_stride_bytes
    if len(fetched) != expected_size:
        raise ArtifactProbeError(
            f"oracle {projection}: fetched {len(fetched)} bytes, expected {expected_size}"
        )
    source = descriptor.source
    if isinstance(source, np.ndarray):
        expected_source_shape = (
            descriptor.num_experts,
            descriptor.output_dim,
            descriptor.row_stride_bytes,
        )
        if source.shape != expected_source_shape:
            raise ArtifactProbeError(
                f"oracle {projection}: layout source shape {source.shape} disagrees with "
                f"expected {expected_source_shape}"
            )
        source_bytes = np.ascontiguousarray(source[0]).tobytes()
    else:
        getter = getattr(source, "expert_packed", None)
        if not callable(getter):
            raise ArtifactProbeError(
                f"oracle {projection}: layout source has no expert_packed accessor"
            )
        try:
            source_bytes = np.ascontiguousarray(getter(0)).tobytes()
        except (IndexError, TypeError, ValueError) as error:
            raise ArtifactProbeError(
                f"oracle {projection}: layout source cannot expose selected expert bytes"
            ) from error
    if source_bytes != fetched:
        raise ArtifactProbeError(
            f"oracle {projection}: fetched bytes disagree with the validated layout source"
        )
    return fetched


def _oracle_dense_projections(
    layout: CpuExpertLayout,
    fetched_sources: Mapping[str, bytes],
    gguf: Any,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Dequantize each fetched projection with the independent gguf-py decoder."""
    if len(layout.layers) != 1:
        raise ArtifactProbeError(
            f"real-artifact oracle requires exactly one layer, got {layout.layers}"
        )
    layer_id = layout.layers[0]
    dense: dict[str, np.ndarray] = {}
    packed_hashes: dict[str, str] = {}
    packed_rows: dict[str, np.ndarray] = {}
    for projection in _PROJECTIONS:
        descriptor = layout.descriptor(layer_id, projection)
        if projection not in fetched_sources:
            raise ArtifactProbeError(f"oracle missing fetched source for {projection}")
        fetched = _packed_source_bytes(layout, projection, bytes(fetched_sources[projection]))
        packed_rows[projection] = np.frombuffer(fetched, dtype=np.uint8).reshape(
            descriptor.output_dim, descriptor.row_stride_bytes
        )
        packed_hashes[projection] = hashlib.sha256(fetched).hexdigest()
    for projection in _PROJECTIONS:
        descriptor = layout.descriptor(layer_id, projection)
        packed = packed_rows[projection]
        try:
            quant_type = gguf.GGMLQuantizationType(int(descriptor.quant_type))
        except (TypeError, ValueError) as error:
            raise ArtifactProbeError(
                f"oracle {projection}: unsupported GGML quant type {descriptor.quant_type!r}"
            ) from error
        try:
            values = gguf.dequantize(packed, quant_type)
        except Exception as error:
            raise ArtifactProbeError(
                f"gguf-py dequantize failed for {projection} {quant_type.name}: {error}"
            ) from error
        values = np.asarray(values, dtype=np.float32)
        expected_shape = (descriptor.output_dim, descriptor.input_dim)
        if values.shape != expected_shape:
            raise ArtifactProbeError(
                f"oracle {projection}: gguf-py returned shape {values.shape}, expected "
                f"{expected_shape}"
            )
        if not np.isfinite(values).all():
            raise ArtifactProbeError(f"oracle {projection}: gguf-py returned non-finite values")
        dense[projection] = np.ascontiguousarray(values, dtype=np.float32)
    gate = layout.descriptor(layer_id, "gate")
    up = layout.descriptor(layer_id, "up")
    down = layout.descriptor(layer_id, "down")
    if (gate.input_dim, gate.output_dim) != (up.input_dim, up.output_dim):
        raise ArtifactProbeError("oracle gate/up geometry disagrees")
    if (down.output_dim, down.input_dim) != (gate.input_dim, gate.output_dim):
        raise ArtifactProbeError("oracle down geometry is not transposed")
    return dense, packed_hashes


def _validate_oracle_arrays(
    layout: CpuExpertLayout,
    hidden: np.ndarray,
    expert_ids: np.ndarray,
    routing_weights: np.ndarray,
    num_token_non_padded: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    layer_id = layout.layers[0]
    descriptor = layout.descriptor(layer_id, "gate")
    hidden = np.asarray(hidden)
    expert_ids = np.asarray(expert_ids)
    routing_weights = np.asarray(routing_weights)
    if hidden.ndim != 2 or hidden.shape[1] != descriptor.input_dim:
        raise ArtifactProbeError(
            f"oracle hidden shape {hidden.shape} does not match (*, {descriptor.input_dim})"
        )
    if expert_ids.ndim != 2 or routing_weights.ndim != 2:
        raise ArtifactProbeError("oracle expert_ids and routing_weights must be rank 2")
    if expert_ids.shape != routing_weights.shape or expert_ids.shape[0] != hidden.shape[0]:
        raise ArtifactProbeError("oracle hidden and routing arrays have incompatible shapes")
    if not np.issubdtype(hidden.dtype, np.floating):
        raise ArtifactProbeError(f"oracle hidden must be floating point, got {hidden.dtype}")
    if not np.issubdtype(expert_ids.dtype, np.integer):
        raise ArtifactProbeError(f"oracle expert_ids must be integer, got {expert_ids.dtype}")
    if not np.issubdtype(routing_weights.dtype, np.floating):
        raise ArtifactProbeError(
            f"oracle routing_weights must be floating point, got {routing_weights.dtype}"
        )
    active_tokens = hidden.shape[0] if num_token_non_padded is None else int(num_token_non_padded)
    if num_token_non_padded is not None and not 0 <= active_tokens <= hidden.shape[0]:
        raise ArtifactProbeError(
            f"oracle num_token_non_padded={active_tokens} outside [0, {hidden.shape[0]}]"
        )
    if not np.isfinite(hidden).all() or not np.isfinite(routing_weights).all():
        raise ArtifactProbeError("oracle hidden and routing weights must be finite")
    invalid = (expert_ids < -1) | (expert_ids > 0)
    if np.any(invalid[:active_tokens]):
        token, route = np.argwhere(invalid[:active_tokens])[0]
        raise ArtifactProbeError(
            f"oracle expert id {int(expert_ids[token, route])} is invalid at "
            f"token={token}, route={route}; selected ranges remap only expert 0"
        )
    return (
        hidden.astype(np.float32, copy=False),
        expert_ids,
        routing_weights.astype(np.float32, copy=False),
        active_tokens,
    )


def _run_gguf_oracle(
    layout: CpuExpertLayout,
    fetched_sources: Mapping[str, bytes],
    hidden: np.ndarray,
    expert_ids: np.ndarray,
    routing_weights: np.ndarray,
    *,
    num_token_non_padded: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run dense FP32 SwiGLU using gguf-py's independent dequantization."""
    if len(layout.layers) != 1:
        raise ArtifactProbeError(
            f"real-artifact oracle requires exactly one layer, got {layout.layers}"
        )
    gguf = _load_gguf_oracle()
    hidden, expert_ids, routing_weights, active_tokens = _validate_oracle_arrays(
        layout, hidden, expert_ids, routing_weights, num_token_non_padded
    )
    dequant_started = time.perf_counter_ns()
    dense, packed_hashes = _oracle_dense_projections(layout, fetched_sources, gguf)
    dequant_elapsed_ns = time.perf_counter_ns() - dequant_started
    gate = dense["gate"]
    up = dense["up"]
    down = dense["down"]
    output = np.zeros((hidden.shape[0], gate.shape[1]), dtype=np.float32)
    activated = np.empty(gate.shape[0], dtype=np.float32)
    execution_started = time.perf_counter_ns()
    for token in range(active_tokens):
        for route in range(expert_ids.shape[1]):
            expert = int(expert_ids[token, route])
            if expert == -1:
                continue
            # Selected ranges contain one remapped expert, so every valid route uses
            # the same independently dequantized matrices.  Duplicate IDs remain
            # separate routes and their weights are accumulated independently.
            gate_values = np.matmul(gate, hidden[token]).astype(np.float32, copy=False)
            up_values = np.matmul(up, hidden[token]).astype(np.float32, copy=False)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                activated[:] = gate_values / (1.0 + np.exp(-gate_values))
            np.multiply(activated, up_values, out=activated)
            contribution = np.matmul(down, activated).astype(np.float32, copy=False)
            np.multiply(contribution, np.float32(routing_weights[token, route]), out=contribution)
            np.add(output[token], contribution, out=output[token])
    execution_elapsed_ns = time.perf_counter_ns() - execution_started
    return output, {
        **gguf_oracle_identity(),
        "packed_source_sha256": packed_hashes,
        "dense_projection_sha256": {
            projection: _hash_array(dense[projection]) for projection in _PROJECTIONS
        },
        "output_sha256": _hash_array(output),
        "raw_elapsed_ns": {
            "dequantize": dequant_elapsed_ns,
            "dense_expert": execution_elapsed_ns,
        },
    }


def run_gguf_oracle(
    layout: CpuExpertLayout,
    fetched_sources: Mapping[str, bytes],
    hidden: np.ndarray,
    expert_ids: np.ndarray,
    routing_weights: np.ndarray,
    *,
    num_token_non_padded: int | None = None,
) -> np.ndarray:
    """Return an independent gguf-py dense FP32 expert result for exact fetched bytes."""
    output, _ = _run_gguf_oracle(
        layout,
        fetched_sources,
        hidden,
        expert_ids,
        routing_weights,
        num_token_non_padded=num_token_non_padded,
    )
    return output


def _compare_outputs(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    expected_name: str,
    actual_name: str,
    rtol: float = ORACLE_RTOL,
    atol: float = ORACLE_ATOL,
) -> dict[str, Any]:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    shape_match = expected.shape == actual.shape
    expected_finite = bool(np.isfinite(expected).all())
    actual_finite = bool(np.isfinite(actual).all())
    finite = expected_finite and actual_finite
    max_abs: float | None = None
    relative_rms: float | None = None
    max_tolerance_violation: float | None = None
    correct = False
    if shape_match and finite:
        difference = np.abs(expected.astype(np.float32) - actual.astype(np.float32))
        max_abs = float(difference.max(initial=0.0))
        rms = float(np.sqrt(np.mean(np.square(difference), dtype=np.float64)))
        denominator = float(np.sqrt(np.mean(np.square(expected), dtype=np.float64)))
        # NumPy 2.5 returns a scalar NumPy type when the denominator fallback
        # is selected.  Cast the durable metric back to a builtin float so the
        # JSON/report schema and checkpoint validator see the same type across
        # supported NumPy versions.
        relative_rms = float(rms / max(denominator, np.finfo(np.float32).tiny))
        tolerance = atol + rtol * np.abs(actual.astype(np.float32))
        max_tolerance_violation = float(np.max(difference - tolerance, initial=0.0))
        correct = bool(np.allclose(expected, actual, rtol=rtol, atol=atol))
    error: str | None = None
    if not shape_match:
        error = f"shape mismatch: {actual_name} {actual.shape}, {expected_name} {expected.shape}"
    elif not finite:
        error = f"non-finite values in {actual_name if not actual_finite else expected_name}"
    elif not correct:
        error = f"max_abs_error={max_abs} exceeds rtol={rtol}, atol={atol}"
    return {
        "expected": expected_name,
        "actual": actual_name,
        "expected_output_sha256": _hash_array(expected),
        "actual_output_sha256": _hash_array(actual),
        "shape_match": shape_match,
        "finite": finite,
        "max_abs_error": max_abs,
        "relative_rms_error": relative_rms,
        "max_tolerance_violation": max_tolerance_violation,
        "rtol": rtol,
        "atol": atol,
        "correct": correct,
        "error": error,
    }


def load_qwen38_expert_artifact(
    *,
    manifest_path: Path,
    census_path: Path,
    variant: str = DEFAULT_VARIANT,
    layer: int = DEFAULT_LAYER,
    expert: int = DEFAULT_EXPERT,
    transport: RangeTransport | Callable[[str, int, int], RangeResponse] | None = None,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> dict[str, Any]:
    """Fetch one expert's bounded ranges and build its validated CPU layout.

    The returned object is the shared hand-off between the bounded artifact probe and
    target-host benchmarks.  It deliberately retains the manifest/census source
    identity, each selected range hash, and cache counters while keeping complete
    shards out of memory and disk.
    """
    manifest = _load_json(Path(manifest_path), "manifest")
    census = _load_json(Path(census_path), "census")
    try:
        census_sha256 = hashlib.sha256(Path(census_path).read_bytes()).hexdigest()
    except OSError as error:
        raise ArtifactProbeError(f"cannot hash census metadata {census_path}: {error}") from error
    source, manifest_shards = _validate_metadata(manifest, census, variant)
    records = _tensor_records(census, int(layer))
    range_transport = transport or UrllibRangeTransport()
    fetcher = RangeFetcher(range_transport, cache_dir=cache_dir)
    sources: dict[str, bytes] = {}
    ranges: list[dict[str, Any]] = []
    for projection in _PROJECTIONS:
        record = records[projection]
        try:
            shard_index = int(record["shard_index"])
            if shard_index < 0:
                raise IndexError(shard_index)
            shard_name = str(census["shards"][shard_index]["name"])
            shard_size = int(manifest_shards[shard_name]["size"])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ArtifactProbeError(
                f"layer {layer} {projection}: invalid shard metadata"
            ) from error
        geometry = _record_geometry(
            record,
            shard_size=shard_size,
            layer=int(layer),
            projection=projection,
            expert=int(expert),
        )
        url = f"{source['base_url']}/{shard_name}"
        response = fetcher.fetch(
            url,
            geometry["expert_offset"],
            geometry["bytes_per_expert"],
            expected_total=shard_size,
            offline=offline,
        )
        sources[projection] = response.body
        ranges.append(
            {
                "projection": projection,
                "url": url,
                "shard": shard_name,
                "shard_size": shard_size,
                "declared_shard_sha256": str(manifest_shards[shard_name]["sha256"]),
                "artifact_offset": geometry["expert_offset"],
                "length": len(response.body),
                "sha256": hashlib.sha256(response.body).hexdigest(),
                "content_range": _header(response.headers, "Content-Range"),
                "cache": "hit" if fetcher.last_cache_hit else "miss",
            }
        )
    layout = build_probe_layout(
        manifest,
        census,
        variant=variant,
        layer=int(layer),
        expert=int(expert),
        sources=sources,
    )
    return {
        "manifest": manifest,
        "census": census,
        "census_sha256": census_sha256,
        "source": source,
        "layout": layout,
        "sources": sources,
        "ranges": ranges,
        "fetch": {
            "transport": "http-range",
            "offline": bool(offline),
            "cache_dir": None if cache_dir is None else str(cache_dir),
            "cache_hits": fetcher.cache_hits,
            "cache_misses": fetcher.cache_misses,
            "range_count": len(ranges),
            "fetched_bytes": sum(item["length"] for item in ranges),
            "full_shard_bytes": 0,
        },
    }


def probe_qwen38_expert(
    *,
    manifest_path: Path,
    census_path: Path,
    variant: str = DEFAULT_VARIANT,
    layer: int = DEFAULT_LAYER,
    expert: int = DEFAULT_EXPERT,
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
    seed: int = DEFAULT_SEED,
    transport: RangeTransport | Callable[[str, int, int], RangeResponse] | None = None,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> dict[str, Any]:
    """Run the bounded real-byte H0 probe and return a JSON-serializable report."""
    if repeats <= 0 or warmup < 0:
        raise ArtifactProbeError("repeats must be positive and warmup must be non-negative")
    # Validate the independent reference before making any network request.  A real
    # probe without gguf-py would otherwise consume selected ranges and fail only after
    # the transport work had completed.
    _load_gguf_oracle()
    manifest = _load_json(Path(manifest_path), "manifest")
    census = _load_json(Path(census_path), "census")
    source, manifest_shards = _validate_metadata(manifest, census, variant)
    records = _tensor_records(census, int(layer))
    range_transport = transport or UrllibRangeTransport()
    fetcher = RangeFetcher(range_transport, cache_dir=cache_dir)
    sources: dict[str, bytes] = {}
    ranges: list[dict[str, Any]] = []
    for projection in _PROJECTIONS:
        record = records[projection]
        try:
            shard_index = int(record["shard_index"])
            if shard_index < 0:
                raise IndexError(shard_index)
            shard_name = str(census["shards"][shard_index]["name"])
            shard_size = int(manifest_shards[shard_name]["size"])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ArtifactProbeError(
                f"layer {layer} {projection}: invalid shard metadata"
            ) from error
        geometry = _record_geometry(
            record,
            shard_size=shard_size,
            layer=int(layer),
            projection=projection,
            expert=int(expert),
        )
        url = f"{source['base_url']}/{shard_name}"
        response = fetcher.fetch(
            url,
            geometry["expert_offset"],
            geometry["bytes_per_expert"],
            expected_total=shard_size,
            offline=offline,
        )
        sources[projection] = response.body
        ranges.append(
            {
                "projection": projection,
                "url": url,
                "shard": shard_name,
                "shard_size": shard_size,
                "declared_shard_sha256": str(manifest_shards[shard_name]["sha256"]),
                "artifact_offset": geometry["expert_offset"],
                "length": len(response.body),
                "sha256": hashlib.sha256(response.body).hexdigest(),
                "content_range": _header(response.headers, "Content-Range"),
                "cache": "hit" if fetcher.last_cache_hit else "miss",
            }
        )
    layout = build_probe_layout(
        manifest,
        census,
        variant=variant,
        layer=int(layer),
        expert=int(expert),
        sources=sources,
    )
    hidden_rng = np.random.default_rng(seed)
    hidden = hidden_rng.standard_normal(
        (2, layout.descriptor(int(layer), "gate").input_dim)
    ).astype(np.float32)
    expert_ids = np.array([[0, -1], [0, 0]], dtype=np.int32)
    weights = np.array([[0.25, 0.0], [0.5, 0.25]], dtype=np.float32)
    scalar_output, scalar = _run_mode(
        layout,
        hidden,
        expert_ids,
        weights,
        mode="forced_scalar",
        repeats=int(repeats),
        warmup=int(warmup),
    )
    avx_output, avx2 = _run_mode(
        layout,
        hidden,
        expert_ids,
        weights,
        mode="forced_avx2",
        repeats=int(repeats),
        warmup=int(warmup),
    )
    oracle_output, oracle = _run_gguf_oracle(
        layout,
        sources,
        hidden,
        expert_ids,
        weights,
    )
    scalar["output_sha256"] = _hash_array(scalar_output)
    avx2["output_sha256"] = _hash_array(avx_output)
    oracle_vs_scalar = _compare_outputs(
        oracle_output,
        scalar_output,
        expected_name="gguf-py oracle",
        actual_name="scalar",
    )
    oracle_vs_native = _compare_outputs(
        oracle_output,
        avx_output,
        expected_name="gguf-py oracle",
        actual_name="native executor",
    )
    scalar_vs_native = _compare_outputs(
        scalar_output,
        avx_output,
        expected_name="scalar",
        actual_name="native executor",
    )
    correct = bool(
        oracle_vs_scalar["correct"] and oracle_vs_native["correct"] and scalar_vs_native["correct"]
    )
    finite = bool(
        oracle_vs_scalar["finite"] and oracle_vs_native["finite"] and scalar_vs_native["finite"]
    )
    promoted = any(
        descriptor["quant_name"] in {"Q5_K", "Q8_0"}
        for descriptor in _descriptor_summary(layout, expert=int(expert))
    )
    return {
        "schema_version": 1,
        "evidence_status": "artifact-metadata",
        "range_evidence": "measured/artifact-byte",
        "validation_class": "H0/no-P4",
        "repeats": int(repeats),
        "warmup": int(warmup),
        "limitations": [
            "selected expert ranges only; no complete shard download or checksum",
            "CPU executor A/B only; no P4, cache, hybrid split or full-engine claim",
        ],
        "source": source,
        "oracle": oracle,
        "selection": {
            "variant": variant,
            "layer": int(layer),
            "expert": int(expert),
            "promoted": promoted,
            "seed": int(seed),
            "hidden_sha256": _hash_array(hidden),
            "expert_ids_sha256": _hash_array(expert_ids),
            "routing_weights_sha256": _hash_array(weights),
        },
        "fetch": {
            "transport": "http-range",
            "offline": bool(offline),
            "cache_dir": None if cache_dir is None else str(cache_dir),
            "cache_hits": fetcher.cache_hits,
            "cache_misses": fetcher.cache_misses,
            "range_count": len(ranges),
            "fetched_bytes": sum(item["length"] for item in ranges),
            "full_shard_bytes": 0,
            "ranges": ranges,
        },
        "layout": {
            "top_k": layout.top_k,
            "descriptors": _descriptor_summary(layout, expert=int(expert)),
        },
        "ab": {
            "comparison": (
                "gguf-py dense FP32 oracle versus scalar and native executor outputs; "
                "internal scalar/native A/B retained separately"
            ),
            "independent_oracle": True,
            "correct": correct,
            "finite": finite,
            "max_abs_error": scalar_vs_native["max_abs_error"],
            "rtol": ORACLE_RTOL,
            "atol": ORACLE_ATOL,
            "oracle": oracle,
            "oracle_vs_scalar": oracle_vs_scalar,
            "oracle_vs_native": oracle_vs_native,
            "scalar_vs_native": scalar_vs_native,
            "timing": {
                "scalar_raw_elapsed_ns": scalar["raw_elapsed_ns"],
                "native_raw_elapsed_ns": avx2["raw_elapsed_ns"],
                "oracle_raw_elapsed_ns": oracle.get("raw_elapsed_ns"),
                "comparison_claim": False,
            },
            "scalar": scalar,
            "avx2": avx2,
            "native": avx2,
        },
    }


def _probe_commit() -> str:
    """Resolve the immutable source commit used by a probe invocation."""
    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArtifactProbeError(
            f"cannot resolve repository commit for evidence: {error}"
        ) from error
    commit = result.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ArtifactProbeError(f"repository commit is not a full SHA-1: {commit!r}")
    return commit


def _probe_host_identity() -> dict[str, Any]:
    """Capture host facts that make a reference-only run reproducible."""
    try:
        affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {
        "hostname": platform.node() or "unknown",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "process_affinity": affinity,
    }


_Q3_TRIAD_CHECKPOINT = "qwen38-q3-triad.checkpoint.json"
_Q3_TRIAD_CHECKPOINT_SCHEMA_VERSION = 2


def _write_q3_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist completed triad points so a supervisor can resume."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _cleanup_q3_checkpoint_temps(path)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as tmp:
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            temporary = Path(tmp.name)
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)
        raise ArtifactProbeError(
            f"cannot atomically write Q3 triad checkpoint {path}: {error}"
        ) from error


def _read_q3_checkpoint(path: Path) -> dict[str, Any]:
    _cleanup_q3_checkpoint_temps(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ArtifactProbeError(f"cannot read Q3 triad checkpoint {path}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactProbeError(f"Q3 triad checkpoint {path} must contain an object")
    return value


def _cleanup_q3_checkpoint_temps(path: Path) -> None:
    """Remove only orphaned temporary files for this checkpoint name."""
    if not path.parent.exists():
        return
    prefix = f".{path.name}."
    try:
        entries = tuple(path.parent.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.is_file() and entry.name.startswith(prefix) and entry.name.endswith(".tmp"):
            entry.unlink(missing_ok=True)


def _q3_invocation_audit(
    *, command: str, host: Mapping[str, Any], offline: bool, cache_dir: Path | None
) -> dict[str, Any]:
    """Record invocation details without making them part of resume identity."""
    return {
        "command": command,
        "host": dict(host),
        "offline": bool(offline),
        "cache_dir": None if cache_dir is None else str(cache_dir),
    }


def _validate_q3_checkpoint_summary(
    item: Mapping[str, Any],
    *,
    census: Mapping[str, Any],
    manifest_shards: Mapping[str, Mapping[str, Any]],
    source: Mapping[str, Any],
    variant: str,
    expected_commit: str,
    expected_repeats: int,
    expected_warmup: int,
    audit_history: list[Mapping[str, Any]],
) -> None:
    """Validate a completed point before allowing it into a resumed aggregate."""
    required_ranges = {
        "projection",
        "url",
        "shard",
        "shard_size",
        "declared_shard_sha256",
        "artifact_offset",
        "start",
        "end",
        "length",
        "sha256",
        "content_range",
        "cache",
    }
    try:

        def require(condition: bool, message: str) -> None:
            if not condition:
                raise ArtifactProbeError(message)

        ranges = item["ranges"]
        if (
            not isinstance(ranges, list)
            or len(ranges) != 3
            or any(
                not isinstance(value, dict) or not required_ranges.issubset(value)
                for value in ranges
            )
        ):
            raise ArtifactProbeError("Q3 triad checkpoint contains invalid probe ranges")
        for value in ranges:
            numeric_fields = (
                "shard_size",
                "artifact_offset",
                "start",
                "end",
                "length",
            )
            if any(
                isinstance(value[key], bool) or not isinstance(value[key], int)
                for key in numeric_fields
            ):
                raise ArtifactProbeError("Q3 triad checkpoint contains invalid probe ranges")
            projection = value["projection"]
            if projection not in _PROJECTIONS:
                raise ArtifactProbeError("Q3 triad checkpoint contains invalid probe ranges")
            if any(
                not isinstance(value[key], str)
                for key in ("url", "shard", "declared_shard_sha256", "sha256", "content_range")
            ):
                raise ArtifactProbeError("Q3 triad checkpoint contains invalid probe ranges")
            if (
                value["shard_size"] <= 0
                or value["artifact_offset"] < 0
                or value["length"] <= 0
                or value["start"] != value["artifact_offset"]
                or value["end"] != value["start"] + value["length"] - 1
                or value["end"] >= value["shard_size"]
                or not re.fullmatch(r"[0-9a-f]{64}", value["declared_shard_sha256"])
                or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
                or not value["url"]
                or not value["shard"]
                or not value["content_range"]
                or value["cache"] not in ("hit", "miss")
            ):
                raise ArtifactProbeError("Q3 triad checkpoint contains invalid probe ranges")
        require(
            item["commit"] == expected_commit,
            "Q3 triad checkpoint completed probe commit disagrees with identity",
        )
        require(
            isinstance(item["command"], str) and item["command"],
            "Q3 triad checkpoint completed probe command is malformed",
        )
        require(
            isinstance(item["host"], dict) and item["host"],
            "Q3 triad checkpoint completed probe host is malformed",
        )
        raw = item["raw"]
        require(
            isinstance(raw, dict),
            "Q3 triad checkpoint contains invalid raw probe evidence",
        )
        try:
            canonical = _triad_probe_summary(
                raw,
                layer=int(item["layer"]),
                expert=int(item["expert"]),
                seed=int(item["seed"]),
                commit=str(item["commit"]),
                command=str(item["command"]),
                host=item["host"],
            )
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ArtifactProbeError(
                "Q3 triad checkpoint completed probe cannot be normalized"
            ) from error
        require(
            set(item) == set(canonical),
            "Q3 triad checkpoint completed probe has schema-invalid fields",
        )
        require(
            item == canonical,
            "Q3 triad checkpoint completed probe differs from canonical raw evidence",
        )
        for key, expected in (
            ("schema_version", 1),
            ("evidence_status", "artifact-metadata"),
            ("range_evidence", "measured/artifact-byte"),
            ("validation_class", "H0/no-P4"),
            ("repeats", expected_repeats),
            ("warmup", expected_warmup),
        ):
            require(
                raw.get(key) == expected,
                f"Q3 triad checkpoint raw.{key} disagrees with the probe contract",
            )
        raw_metadata = raw["triad_metadata"]
        require(
            isinstance(raw_metadata, dict)
            and raw_metadata.get("commit") == item["commit"]
            and raw_metadata.get("command") == item["command"]
            and raw_metadata.get("host") == item["host"],
            "Q3 triad checkpoint raw triad metadata disagrees with the completed probe",
        )
        raw_source = raw["source"]
        require(
            raw_source == source,
            "Q3 triad checkpoint raw source disagrees with the pinned source",
        )
        raw_selection = raw["selection"]
        require(
            isinstance(raw_selection, dict)
            and raw_selection.get("variant") == variant
            and raw_selection.get("layer") == item["layer"]
            and raw_selection.get("expert") == item["expert"]
            and raw_selection.get("seed") == item["seed"],
            "Q3 triad checkpoint raw selection disagrees with completed identity",
        )
        raw_fetch = raw["fetch"]
        require(
            isinstance(raw_fetch, dict)
            and raw_fetch.get("transport") == "http-range"
            and raw_fetch.get("range_count") == 3
            and raw_fetch.get("full_shard_bytes") == 0
            and isinstance(raw_fetch.get("offline"), bool)
            and (raw_fetch.get("cache_dir") is None or isinstance(raw_fetch.get("cache_dir"), str))
            and all(
                isinstance(raw_fetch.get(key), int)
                and not isinstance(raw_fetch.get(key), bool)
                and raw_fetch.get(key) >= 0
                for key in (
                    "cache_hits",
                    "cache_misses",
                    "range_count",
                    "fetched_bytes",
                    "full_shard_bytes",
                )
            ),
            "Q3 triad checkpoint raw fetch violates bounded range contract",
        )
        for audit in audit_history:
            require(
                isinstance(audit, Mapping)
                and set(audit) == {"command", "host", "offline", "cache_dir"}
                and isinstance(audit.get("command"), str)
                and bool(audit.get("command"))
                and isinstance(audit.get("host"), dict)
                and bool(audit.get("host"))
                and isinstance(audit.get("offline"), bool)
                and (audit.get("cache_dir") is None or isinstance(audit.get("cache_dir"), str)),
                "Q3 triad checkpoint audit history contains invalid invocation types",
            )
        require(
            any(
                isinstance(audit, Mapping)
                and audit.get("command") == item["command"]
                and audit.get("host") == item["host"]
                and audit.get("offline") == raw_fetch.get("offline")
                and audit.get("cache_dir") == raw_fetch.get("cache_dir")
                for audit in audit_history
            ),
            "Q3 triad checkpoint completed probe invocation is absent from audit history",
        )
        raw_layout = raw["layout"]
        require(
            isinstance(raw_layout, dict) and raw_layout.get("top_k") == 2,
            "Q3 triad checkpoint raw layout violates Qwen top-k contract",
        )
        raw_ab = raw["ab"]
        require(
            isinstance(raw_ab, dict),
            "Q3 triad checkpoint contains invalid raw AB evidence",
        )
        raw_oracle = raw_ab["oracle"]
        require(
            isinstance(raw_oracle, dict),
            "Q3 triad checkpoint contains invalid raw oracle evidence",
        )
        oracle_timing = raw_oracle.get("raw_elapsed_ns")
        require(
            isinstance(oracle_timing, dict)
            and all(
                isinstance(oracle_timing.get(key), int)
                and not isinstance(oracle_timing.get(key), bool)
                and oracle_timing.get(key) >= 0
                for key in ("dequantize", "dense_expert")
            ),
            "Q3 triad checkpoint oracle timing is malformed",
        )
        for hash_name in ("packed_source_sha256", "dense_projection_sha256"):
            hash_values = raw_oracle.get(hash_name)
            require(
                isinstance(hash_values, dict)
                and set(hash_values) == set(_PROJECTIONS)
                and all(
                    isinstance(hash_values.get(projection), str)
                    and re.fullmatch(r"[0-9a-f]{64}", hash_values[projection])
                    for projection in _PROJECTIONS
                ),
                f"Q3 triad checkpoint oracle {hash_name} is malformed",
            )
        require(
            raw.get("oracle") == raw_oracle,
            "Q3 triad checkpoint raw oracle duplicates disagree",
        )
        require(
            raw_ab.get("native") == raw_ab.get("avx2"),
            "Q3 triad checkpoint raw native and avx2 duplicates disagree",
        )
        require(
            raw_ab.get("independent_oracle") is True
            and raw_oracle.get("name") == GGUF_ORACLE_NAME
            and raw_oracle.get("package") == GGUF_ORACLE_PACKAGE
            and raw_oracle.get("version") == GGUF_ORACLE_VERSION
            and raw_oracle.get("operation") == GGUF_ORACLE_OPERATION,
            "Q3 triad checkpoint oracle is not the pinned independent reference",
        )
        require(
            isinstance(raw_ab.get("correct"), bool)
            and isinstance(raw_ab.get("finite"), bool)
            and raw_ab.get("rtol") == ORACLE_RTOL
            and raw_ab.get("atol") == ORACLE_ATOL,
            "Q3 triad checkpoint raw AB contract is malformed",
        )
        raw_descriptors = raw_layout["descriptors"]
        require(
            isinstance(raw_descriptors, list) and len(raw_descriptors) == 3,
            "Q3 triad checkpoint raw layout descriptors are malformed",
        )
        records = _tensor_records(census, int(item["layer"]))
        quant_names: dict[str, str] = {}
        promoted = False
        all_ranges: list[dict[str, Any]] = []
        for projection, range_item, descriptor in zip(
            _PROJECTIONS, ranges, raw_descriptors, strict=True
        ):
            record = records[projection]
            shard_index = int(record["shard_index"])
            shard_name = str(census["shards"][shard_index]["name"])
            shard_size = int(manifest_shards[shard_name]["size"])
            geometry = _record_geometry(
                record,
                shard_size=shard_size,
                layer=int(item["layer"]),
                projection=projection,
                expert=int(item["expert"]),
            )
            quant_name = str(record["quant_name"])
            quant_names[projection] = quant_name
            promoted = promoted or quant_name in {"Q5_K", "Q8_0"}
            expected_range = {
                "projection": projection,
                "url": f"{source['base_url']}/{shard_name}",
                "shard": shard_name,
                "shard_size": shard_size,
                "declared_shard_sha256": str(manifest_shards[shard_name]["sha256"]),
                "artifact_offset": geometry["expert_offset"],
                "start": geometry["expert_offset"],
                "end": geometry["expert_offset"] + geometry["bytes_per_expert"] - 1,
                "length": geometry["bytes_per_expert"],
            }
            for key, expected in expected_range.items():
                require(
                    range_item.get(key) == expected,
                    f"Q3 triad checkpoint {projection} range disagrees with census geometry",
                )
            content_range = _CONTENT_RANGE.fullmatch(range_item["content_range"].strip())
            require(
                content_range is not None
                and tuple(int(value) for value in content_range.groups())
                == (range_item["start"], range_item["end"], range_item["shard_size"]),
                f"Q3 triad checkpoint {projection} Content-Range is inconsistent",
            )
            require(
                range_item["sha256"] == item["range_hashes"][projection],
                f"Q3 triad checkpoint {projection} range hash is not bound",
            )
            raw_range = raw_fetch["ranges"][len(all_ranges)]
            require(
                isinstance(raw_range, dict),
                "Q3 triad checkpoint contains invalid raw fetch ranges",
            )
            for key in (
                "projection",
                "url",
                "shard",
                "shard_size",
                "declared_shard_sha256",
                "artifact_offset",
                "length",
                "sha256",
                "content_range",
                "cache",
            ):
                require(
                    range_item.get(key) == raw_range.get(key),
                    f"Q3 triad checkpoint {projection} range disagrees with raw fetch",
                )
            all_ranges.append(range_item)
            expected_descriptor = {
                "layer": int(item["layer"]),
                "projection": projection,
                "selected_expert": int(item["expert"]),
                "num_experts_remapped": 1,
                "output_dim": geometry["output_dim"],
                "input_dim": geometry["input_dim"],
                "rows_per_expert": geometry["output_dim"],
                "row_stride_bytes": geometry["row_bytes"],
                "expert_stride_bytes": geometry["bytes_per_expert"],
                "tensor_bytes": geometry["bytes_per_expert"],
                "artifact_offset": geometry["expert_offset"],
                "artifact_end": geometry["expert_offset"] + geometry["bytes_per_expert"],
                "quant_type": geometry["quant_type"],
                "quant_name": quant_name,
            }
            for key, expected in expected_descriptor.items():
                require(
                    descriptor.get(key) == expected,
                    f"Q3 triad checkpoint raw layout {projection}.{key} disagrees with census",
                )
        require(
            item["quant_names"] == quant_names,
            "Q3 triad checkpoint quant names disagree with census layout",
        )
        require(
            raw_selection.get("promoted") is promoted,
            "Q3 triad checkpoint selection promoted flag disagrees with quant family",
        )
        require(
            isinstance(raw_fetch.get("ranges"), list)
            and len(raw_fetch["ranges"]) == 3
            and raw_fetch.get("fetched_bytes") == sum(item["length"] for item in ranges)
            and raw_fetch.get("cache_hits")
            == sum(range_item["cache"] == "hit" for range_item in ranges)
            and raw_fetch.get("cache_misses")
            == sum(range_item["cache"] == "miss" for range_item in ranges),
            "Q3 triad checkpoint raw fetch counters disagree with ranges",
        )
        expected_workload_hashes = {
            key: raw_selection[key]
            for key in ("hidden_sha256", "expert_ids_sha256", "routing_weights_sha256")
        }
        require(
            item["workload_hashes"] == expected_workload_hashes,
            "Q3 triad checkpoint workload hashes disagree with raw selection",
        )
        expected_output_hashes = {
            "oracle": raw_oracle["output_sha256"],
            "scalar": raw_ab["scalar"]["output_sha256"],
            "native": raw_ab["avx2"]["output_sha256"],
        }
        require(
            item["output_hashes"] == expected_output_hashes,
            "Q3 triad checkpoint output hashes disagree with raw outputs",
        )
        require(
            raw_oracle["packed_source_sha256"]
            == {
                projection: range_item["sha256"]
                for projection, range_item in zip(_PROJECTIONS, ranges, strict=True)
            },
            "Q3 triad checkpoint oracle packed hashes disagree with ranges",
        )
        expected_timing = {
            "scalar_elapsed_ns": raw_ab["scalar"]["raw_elapsed_ns"],
            "native_elapsed_ns": raw_ab["avx2"]["raw_elapsed_ns"],
            "oracle_elapsed_ns": {
                "dequantize": raw_oracle["raw_elapsed_ns"]["dequantize"],
                "dense_expert": raw_oracle["raw_elapsed_ns"]["dense_expert"],
            },
        }
        require(
            item["timing"] == expected_timing,
            "Q3 triad checkpoint timing disagrees with raw telemetry",
        )
        require(
            raw_ab["timing"]
            == {
                "scalar_raw_elapsed_ns": raw_ab["scalar"]["raw_elapsed_ns"],
                "native_raw_elapsed_ns": raw_ab["avx2"]["raw_elapsed_ns"],
                "oracle_raw_elapsed_ns": raw_oracle["raw_elapsed_ns"],
                "comparison_claim": False,
            },
            "Q3 triad checkpoint raw timing duplicates disagree",
        )
        comparison_expectations = {
            "oracle_vs_scalar": ("oracle", "scalar", "gguf-py oracle", "scalar"),
            "oracle_vs_native": ("oracle", "native", "gguf-py oracle", "native executor"),
            "scalar_vs_native": ("scalar", "native", "scalar", "native executor"),
        }
        for comparison_name, (
            expected_key,
            actual_key,
            expected_name,
            actual_name,
        ) in comparison_expectations.items():
            comparison = raw_ab[comparison_name]
            require(
                item["correctness"][comparison_name] == comparison,
                f"Q3 triad checkpoint {comparison_name} summary disagrees with raw comparison",
            )
            require(
                comparison.get("correct") is True
                and comparison.get("error") is None
                and comparison.get("expected") == expected_name
                and comparison.get("actual") == actual_name
                and comparison.get("expected_output_sha256") == expected_output_hashes[expected_key]
                and comparison.get("actual_output_sha256") == expected_output_hashes[actual_key]
                and comparison.get("shape_match") is True
                and comparison.get("finite") is True,
                f"Q3 triad checkpoint {comparison_name} correctness is malformed",
            )
            metrics = (
                comparison.get("max_abs_error"),
                comparison.get("relative_rms_error"),
                comparison.get("max_tolerance_violation"),
                comparison.get("rtol"),
                comparison.get("atol"),
            )
            require(
                all(
                    not isinstance(value, bool)
                    and isinstance(value, int | float)
                    and math.isfinite(value)
                    for value in metrics
                )
                and metrics[0] >= 0
                and metrics[1] >= 0
                and comparison["correct"] == (metrics[2] <= 0),
                f"Q3 triad checkpoint {comparison_name} numeric predicate is malformed",
            )
        require(
            item["correctness"]["passed"] is True
            and item["correctness"]["finite"] is True
            and item["correctness"]["rtol"] == raw_ab["rtol"]
            and item["correctness"]["atol"] == raw_ab["atol"]
            and raw_ab["correct"] is True
            and raw_ab["finite"] is True
            and raw_ab["max_abs_error"] == raw_ab["scalar_vs_native"]["max_abs_error"],
            "Q3 triad checkpoint correctness aggregate is malformed",
        )
        for mode_name, mode, expected_request in (
            ("scalar", raw_ab["scalar"], "forced_scalar"),
            ("native", raw_ab["avx2"], "forced_avx2"),
        ):
            telemetry = mode["telemetry"]
            require(
                mode.get("requested_mode") == expected_request
                and mode.get("repeats") == expected_repeats
                and mode.get("warmup") == expected_warmup
                and isinstance(mode.get("raw_elapsed_ns"), list)
                and all(
                    isinstance(value, int) and not isinstance(value, bool) and value > 0
                    for value in mode["raw_elapsed_ns"]
                )
                and isinstance(telemetry, list)
                and all(
                    isinstance(telemetry_item, dict)
                    and isinstance(telemetry_item.get("kernel_census"), list)
                    and all(isinstance(kernel, str) for kernel in telemetry_item["kernel_census"])
                    for telemetry_item in telemetry
                )
                and len(mode["raw_elapsed_ns"]) == expected_repeats
                and len(telemetry) == expected_repeats,
                f"Q3 triad checkpoint {mode_name} timing metadata is malformed",
            )
            kernels = sorted(
                {
                    str(kernel)
                    for telemetry_item in telemetry
                    for kernel in telemetry_item["kernel_census"]
                }
            )
            backends = {telemetry_item["backend"] for telemetry_item in telemetry}
            fallbacks = {
                mode.get("q4k_fallback_reason"),
                mode.get("mixed_fallback_reason"),
                None,
            }
            require(
                mode.get("selected_backend") == "reference"
                and backends == {"reference"}
                and mode.get("actual_avx2") is False
                and mode.get("q4k_isa") in {"scalar", "avx2"}
                and mode.get("mixed_isa") in {"scalar", "avx2"}
                and all(
                    telemetry_item.get("fallback_reason") in fallbacks
                    or (
                        isinstance(telemetry_item.get("fallback_reason"), str)
                        and telemetry_item["fallback_reason"].startswith("reference_")
                    )
                    for telemetry_item in telemetry
                )
                and kernels
                and all(kernel.startswith("reference") for kernel in kernels),
                f"Q3 triad checkpoint {mode_name} backend contract is malformed",
            )
            expected_mode_kernels = item["kernel_census_by_mode"][mode_name]
            require(
                expected_mode_kernels == kernels,
                f"Q3 triad checkpoint {mode_name} kernel census disagrees with telemetry",
            )
        require(
            item["kernel_census"]
            == sorted(
                set(item["kernel_census_by_mode"]["scalar"])
                | set(item["kernel_census_by_mode"]["native"])
            ),
            "Q3 triad checkpoint kernel census aggregate disagrees with modes",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactProbeError(
            "Q3 triad checkpoint completed probe semantic validation failed"
        ) from error


def _triad_probe_summary(
    result: Mapping[str, Any],
    *,
    layer: int,
    expert: int,
    seed: int,
    commit: str,
    command: str,
    host: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract a stable, small summary while retaining the complete probe report."""
    try:
        fetch = result["fetch"]
        ranges = fetch["ranges"]
        ab = result["ab"]
        scalar = ab["scalar"]
        native = ab["avx2"]
        oracle = ab["oracle"]
    except (KeyError, TypeError) as error:
        raise ArtifactProbeError(
            f"probe result is missing aggregate evidence fields: {error}"
        ) from error
    if not isinstance(ranges, list) or len(ranges) != len(_PROJECTIONS):
        raise ArtifactProbeError("each Q3 triad probe must contain exactly three projection ranges")

    probe_id = f"layer-{layer:02d}-expert-{expert:03d}"
    normalized_ranges: list[dict[str, Any]] = []
    range_hashes: dict[str, str] = {}
    for item in ranges:
        try:
            projection = str(item["projection"])
            start = int(item["artifact_offset"])
            length = int(item["length"])
            digest = str(item["sha256"])
            shard_size = int(item["shard_size"])
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactProbeError(f"Q3 triad range metadata is malformed: {error}") from error
        if projection not in _PROJECTIONS or projection in range_hashes:
            raise ArtifactProbeError(f"Q3 triad range projections are not unique: {projection!r}")
        if start < 0 or length <= 0 or start + length > shard_size:
            raise ArtifactProbeError(
                f"Q3 triad {projection} range [{start}, {start + length}) exceeds shard"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ArtifactProbeError(f"Q3 triad {projection} range has invalid SHA-256")
        range_hashes[projection] = digest
        normalized_ranges.append(
            {
                **dict(item),
                "probe_id": probe_id,
                "layer": layer,
                "expert": expert,
                "start": start,
                "end": start + length - 1,
                "length": length,
            }
        )
    if tuple(item["projection"] for item in normalized_ranges) != _PROJECTIONS:
        raise ArtifactProbeError("Q3 triad ranges must be ordered gate, up, down")

    def _timings(mode: Mapping[str, Any]) -> list[int]:
        values = mode.get("raw_elapsed_ns")
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in values
            )
        ):
            raise ArtifactProbeError("Q3 triad probe timings must be positive integer samples")
        return [int(value) for value in values]

    def _kernels(mode: Mapping[str, Any]) -> list[str]:
        values: set[str] = set()
        for telemetry in mode.get("telemetry", ()):  # type: ignore[union-attr]
            if isinstance(telemetry, Mapping):
                values.update(str(kernel) for kernel in telemetry.get("kernel_census", ()))
        if not values:
            raise ArtifactProbeError("Q3 triad probe is missing kernel census telemetry")
        return sorted(values)

    scalar_timing = _timings(scalar)
    native_timing = _timings(native)
    scalar_kernels = _kernels(scalar)
    native_kernels = _kernels(native)
    output_hashes = {
        "oracle": str(oracle.get("output_sha256", "")),
        "scalar": str(scalar.get("output_sha256", "")),
        "native": str(native.get("output_sha256", "")),
    }
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in output_hashes.values()):
        raise ArtifactProbeError("Q3 triad probe is missing output SHA-256 hashes")
    correctness = {
        "passed": bool(ab.get("correct", False)),
        "finite": bool(ab.get("finite", False)),
        "rtol": float(ab.get("rtol", ORACLE_RTOL)),
        "atol": float(ab.get("atol", ORACLE_ATOL)),
        "oracle_vs_scalar": dict(ab.get("oracle_vs_scalar", {})),
        "oracle_vs_native": dict(ab.get("oracle_vs_native", {})),
        "scalar_vs_native": dict(ab.get("scalar_vs_native", {})),
    }
    oracle_timing = oracle.get("raw_elapsed_ns", {})
    if not isinstance(oracle_timing, Mapping):
        oracle_timing = {}
    selection = result.get("selection", {})
    workload_hashes = {
        key: str(selection.get(key, ""))
        for key in ("hidden_sha256", "expert_ids_sha256", "routing_weights_sha256")
    }
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in workload_hashes.values()):
        raise ArtifactProbeError("Q3 triad probe is missing workload SHA-256 hashes")
    raw = dict(result)
    raw["triad_metadata"] = {"commit": commit, "command": command, "host": dict(host)}
    return {
        "probe_id": probe_id,
        "layer": layer,
        "expert": expert,
        "seed": seed,
        "commit": commit,
        "command": command,
        "host": dict(host),
        "range_hashes": range_hashes,
        "ranges": normalized_ranges,
        "quant_names": {
            str(item["projection"]): str(item.get("quant_name", ""))
            for item in result["layout"]["descriptors"]
        },
        "workload_hashes": workload_hashes,
        "output_hashes": output_hashes,
        "timing": {
            "scalar_elapsed_ns": scalar_timing,
            "native_elapsed_ns": native_timing,
            "oracle_elapsed_ns": {
                "dequantize": int(oracle_timing.get("dequantize", 0)),
                "dense_expert": int(oracle_timing.get("dense_expert", 0)),
            },
        },
        "correctness": correctness,
        "kernel_census": sorted(set(scalar_kernels) | set(native_kernels)),
        "kernel_census_by_mode": {"scalar": scalar_kernels, "native": native_kernels},
        "raw": raw,
    }


def probe_qwen38_q3_triad(
    *,
    manifest_path: Path,
    census_path: Path,
    variant: str = DEFAULT_Q3_VARIANT,
    probes: tuple[tuple[int, int], ...] = DEFAULT_Q3_TRIAD,
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
    seed: int = DEFAULT_SEED,
    transport: RangeTransport | Callable[[str, int, int], RangeResponse] | None = None,
    cache_dir: Path | None = None,
    offline: bool = False,
    command: str | None = None,
    commit: str | None = None,
    host: Mapping[str, Any] | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    watchdog_seconds: float | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a bounded three-point Q3 real-byte reference probe.

    Each point delegates to :func:`probe_qwen38_expert`, so every projection is
    fetched with one inclusive HTTP range and the existing independent oracle.  The
    aggregate is deliberately reference-only evidence: it does not claim an AVX2
    speedup, P4 execution, full-shard integrity or serving performance.
    """
    if variant != DEFAULT_Q3_VARIANT:
        raise ArtifactProbeError(f"Q3 triad requires variant {DEFAULT_Q3_VARIANT}, got {variant!r}")
    if len(probes) != 3:
        raise ArtifactProbeError(f"Q3 triad requires exactly three probes, got {len(probes)}")
    normalized: list[tuple[int, int]] = []
    for item in probes:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ArtifactProbeError(f"Q3 triad probe must be (layer, expert), got {item!r}")
        layer, expert = item
        if (
            isinstance(layer, bool)
            or isinstance(expert, bool)
            or not isinstance(layer, int)
            or not isinstance(expert, int)
            or layer < 0
            or expert < 0
        ):
            raise ArtifactProbeError(f"Q3 triad probe must use non-negative integers, got {item!r}")
        normalized.append((int(layer), int(expert)))
    if len(set(normalized)) != len(normalized):
        raise ArtifactProbeError("Q3 triad probes must be unique")
    if resume and checkpoint_dir is None:
        raise ArtifactProbeError("resume requires checkpoint_dir")
    if watchdog_seconds is not None:
        try:
            watchdog_seconds = float(watchdog_seconds)
        except (TypeError, ValueError) as error:
            raise ArtifactProbeError(
                "watchdog_seconds must be a positive finite number when provided"
            ) from error
        if not math.isfinite(watchdog_seconds) or watchdog_seconds <= 0:
            raise ArtifactProbeError(
                "watchdog_seconds must be a positive finite number when provided"
            )
    if repeats <= 0 or warmup < 0:
        raise ArtifactProbeError("repeats must be positive and warmup must be non-negative")
    try:
        base_seed = int(seed)
    except (TypeError, ValueError) as error:
        raise ArtifactProbeError(f"seed must be an integer, got {seed!r}") from error

    manifest_path = Path(manifest_path)
    census_path = Path(census_path)
    manifest = _load_json(manifest_path, "manifest")
    census = _load_json(census_path, "census")
    source, manifest_shards = _validate_metadata(manifest, census, variant)
    manifest_variant = manifest["variants"][variant]
    try:
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        census_sha256 = hashlib.sha256(census_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ArtifactProbeError(f"cannot hash triad identity metadata: {error}") from error

    resolved_commit = _probe_commit() if commit is None else str(commit).lower()
    if not re.fullmatch(r"[0-9a-f]{40}", resolved_commit):
        raise ArtifactProbeError(f"commit must be a full lowercase SHA-1, got {resolved_commit!r}")
    resolved_command = "probe_qwen38_q3_triad" if command is None else str(command).strip()
    if not resolved_command:
        raise ArtifactProbeError("command must be non-empty")
    resolved_host = _probe_host_identity() if host is None else dict(host)
    if not resolved_host:
        raise ArtifactProbeError("host identity must be non-empty")
    current_audit = _q3_invocation_audit(
        command=resolved_command,
        host=resolved_host,
        offline=offline,
        cache_dir=cache_dir,
    )

    try:
        census_quant_counts = {
            str(name): int(values["tensors"]) for name, values in census["by_quant_type"].items()
        }
        census_quant_bytes = {
            str(name): int(values["bytes"]) for name, values in census["by_quant_type"].items()
        }
        census_tensor_count = int(census["tensor_count"])
        census_total_bytes = int(census["total_bytes"])
        census_model_sha256 = str(census["model_sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactProbeError(f"Q3 census totals/counts are malformed: {error}") from error

    checkpoint_path = (
        None if checkpoint_dir is None else Path(checkpoint_dir).resolve() / _Q3_TRIAD_CHECKPOINT
    )
    checkpoint_identity = {
        "commit": resolved_commit,
        "manifest_sha256": manifest_sha256,
        "census_sha256": census_sha256,
        "variant": variant,
        "seed": base_seed,
        "repeats": int(repeats),
        "warmup": int(warmup),
        "probes": [
            {
                "probe_id": f"layer-{layer:02d}-expert-{expert:03d}",
                "layer": layer,
                "expert": expert,
                "seed": base_seed + index,
            }
            for index, (layer, expert) in enumerate(normalized)
        ],
    }
    selected_by_id: dict[str, dict[str, Any]] = {}
    expected_points = {item["probe_id"]: item for item in checkpoint_identity["probes"]}
    resumed = False
    audit_history = [current_audit]
    if checkpoint_path is not None and checkpoint_path.exists():
        if not resume:
            raise ArtifactProbeError(
                f"Q3 triad checkpoint already exists at {checkpoint_path}; pass resume=True"
            )
        checkpoint = _read_q3_checkpoint(checkpoint_path)
        if (
            checkpoint.get("schema_version") != _Q3_TRIAD_CHECKPOINT_SCHEMA_VERSION
            or checkpoint.get("identity") != checkpoint_identity
        ):
            raise ArtifactProbeError("Q3 triad checkpoint identity does not match this invocation")
        checkpoint_audit = checkpoint.get("audit", {})
        if not isinstance(checkpoint_audit, dict):
            raise ArtifactProbeError("Q3 triad checkpoint audit must be an object")
        checkpoint_history = checkpoint_audit.get("history", [])
        if not isinstance(checkpoint_history, list) or not checkpoint_history:
            raise ArtifactProbeError("Q3 triad checkpoint audit history is missing")
        audit_keys = {"command", "host", "offline", "cache_dir"}
        for item in checkpoint_history:
            if not isinstance(item, dict) or not audit_keys.issubset(item):
                raise ArtifactProbeError(
                    "Q3 triad checkpoint audit history contains malformed entry"
                )
        if checkpoint_audit.get("current") != checkpoint_history[-1]:
            raise ArtifactProbeError("Q3 triad checkpoint audit current entry is inconsistent")
        audit_history = [dict(item) for item in checkpoint_history]
        if current_audit not in audit_history:
            audit_history.append(current_audit)
        completed = checkpoint.get("completed", [])
        if not isinstance(completed, list):
            raise ArtifactProbeError("Q3 triad checkpoint completed field must be a list")
        for item in completed:
            if not isinstance(item, dict) or not isinstance(item.get("probe_id"), str):
                raise ArtifactProbeError("Q3 triad checkpoint contains malformed completed probe")
            completed_keys = {
                "probe_id",
                "layer",
                "expert",
                "seed",
                "commit",
                "command",
                "host",
                "range_hashes",
                "ranges",
                "quant_names",
                "workload_hashes",
                "output_hashes",
                "timing",
                "correctness",
                "kernel_census",
                "kernel_census_by_mode",
                "raw",
            }
            if not completed_keys.issubset(item):
                raise ArtifactProbeError("Q3 triad checkpoint contains malformed completed probe")
            if item["probe_id"] in selected_by_id:
                raise ArtifactProbeError("Q3 triad checkpoint contains a duplicate completed probe")
            expected_point = expected_points.get(item["probe_id"])
            if expected_point is None or any(
                item[key] != expected_point[key] for key in ("layer", "expert", "seed")
            ):
                raise ArtifactProbeError(
                    "Q3 triad checkpoint contains an unexpected probe identity"
                )
            if item["commit"] != resolved_commit or not isinstance(item["command"], str):
                raise ArtifactProbeError("Q3 triad checkpoint contains an invalid probe audit")
            if not isinstance(item["host"], dict):
                raise ArtifactProbeError("Q3 triad checkpoint contains an invalid probe host audit")
            _validate_q3_checkpoint_summary(
                item,
                census=census,
                manifest_shards=manifest_shards,
                source=source,
                variant=variant,
                expected_commit=resolved_commit,
                expected_repeats=int(repeats),
                expected_warmup=int(warmup),
                audit_history=audit_history,
            )
            selected_by_id[item["probe_id"]] = item
        expected_ids = set(expected_points)
        if set(selected_by_id) - expected_ids:
            raise ArtifactProbeError("Q3 triad checkpoint contains an unexpected probe")
        resumed = True

    def _event(payload: Mapping[str, Any]) -> None:
        if progress is not None:
            progress(dict(payload))

    if checkpoint_path is not None and not resumed:
        _write_q3_checkpoint(
            checkpoint_path,
            {
                "schema_version": _Q3_TRIAD_CHECKPOINT_SCHEMA_VERSION,
                "status": "started",
                "identity": checkpoint_identity,
                "audit": {"current": current_audit, "history": audit_history},
                "completed": [],
            },
        )

    selected: list[dict[str, Any]] = []
    for index, (layer, expert) in enumerate(normalized):
        probe_id = f"layer-{layer:02d}-expert-{expert:03d}"
        if probe_id in selected_by_id:
            selected.append(selected_by_id[probe_id])
            _event({"status": "resumed", "index": index, "probe_id": probe_id})
            continue
        started = time.monotonic()
        _event({"status": "started", "index": index, "probe_id": probe_id})
        result = probe_qwen38_expert(
            manifest_path=manifest_path,
            census_path=census_path,
            variant=variant,
            layer=layer,
            expert=expert,
            repeats=int(repeats),
            warmup=int(warmup),
            seed=base_seed + index,
            transport=transport,
            cache_dir=cache_dir,
            offline=offline,
        )
        summary = _triad_probe_summary(
            result,
            layer=layer,
            expert=expert,
            seed=base_seed + index,
            commit=resolved_commit,
            command=resolved_command,
            host=resolved_host,
        )
        _validate_q3_checkpoint_summary(
            summary,
            census=census,
            manifest_shards=manifest_shards,
            source=source,
            variant=variant,
            expected_commit=resolved_commit,
            expected_repeats=int(repeats),
            expected_warmup=int(warmup),
            audit_history=audit_history,
        )
        selected.append(summary)
        selected_by_id[probe_id] = selected[-1]
        elapsed = time.monotonic() - started
        _event(
            {
                "status": "completed",
                "index": index,
                "probe_id": probe_id,
                "elapsed_seconds": elapsed,
            }
        )
        if checkpoint_path is not None:
            _write_q3_checkpoint(
                checkpoint_path,
                {
                    "schema_version": _Q3_TRIAD_CHECKPOINT_SCHEMA_VERSION,
                    "status": "in-progress",
                    "identity": checkpoint_identity,
                    "audit": {"current": current_audit, "history": audit_history},
                    "completed": [
                        selected_by_id[item["probe_id"]]
                        for item in checkpoint_identity["probes"]
                        if item["probe_id"] in selected_by_id
                    ],
                },
            )
        if watchdog_seconds is not None and elapsed > watchdog_seconds:
            raise ArtifactProbeError(
                f"Q3 triad probe {probe_id} exceeded watchdog {watchdog_seconds:.3f}s "
                f"({elapsed:.3f}s); resume from checkpoint"
            )

    try:
        ranges = [
            {
                **item,
                "probe_id": probe["probe_id"],
                "layer": probe["layer"],
                "expert": probe["expert"],
            }
            for probe in selected
            for item in probe["ranges"]
        ]
        full_shard_bytes = sum(
            int(probe["raw"]["fetch"].get("full_shard_bytes", 0)) for probe in selected
        )
        fetched_bytes = sum(int(item["length"]) for item in ranges)
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactProbeError(
            "Q3 triad checkpoint contains malformed completed probe"
        ) from error
    source_shards = [dict(item) for item in manifest_variant["shards"]]
    report = {
        "schema_name": "qwen38-q3-triad.schema.json",
        "schema_version": 1,
        "evidence_status": "artifact-metadata",
        "range_evidence": "measured/artifact-byte",
        "claim_status": "reference_only",
        "validation_class": "H0/no-P4",
        "scope": {
            "reference_only": True,
            "performance_claim": False,
            "p4_claim": False,
            "full_shard_download": False,
        },
        "metadata": {
            "commit": resolved_commit,
            "command": resolved_command,
            "host": resolved_host,
            "artifact": {
                "repository": source["repository"],
                "revision": source["revision"],
                "variant": variant,
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "model_sha256": census_model_sha256,
                "model_sha256_status": "declared-per-shard",
                "shards": source_shards,
            },
            "census": {
                "path": str(census_path),
                "sha256": census_sha256,
                "evidence_status": census.get("evidence_status"),
                "tensor_count": census_tensor_count,
                "total_bytes": census_total_bytes,
                "quant_type_counts": census_quant_counts,
                "quant_type_bytes": census_quant_bytes,
                "model_sha256": census_model_sha256,
            },
            "probe_count": len(selected),
            "seed": base_seed,
            "repeats": int(repeats),
            "warmup": int(warmup),
            "checkpoint": {
                "enabled": checkpoint_path is not None,
                "resumed": resumed,
                "path": None if checkpoint_path is None else str(checkpoint_path),
            },
            "audit": {"current": current_audit, "history": audit_history},
        },
        "source": source,
        "fetch": {
            "transport": "http-range",
            "offline": bool(offline),
            "cache_dir": None if cache_dir is None else str(cache_dir),
            "cache_hits": sum(
                int(probe["raw"]["fetch"].get("cache_hits", 0)) for probe in selected
            ),
            "cache_misses": sum(
                int(probe["raw"]["fetch"].get("cache_misses", 0)) for probe in selected
            ),
            "range_count": len(ranges),
            "fetched_bytes": fetched_bytes,
            "full_shard_bytes": full_shard_bytes,
            "ranges": ranges,
        },
        "probes": selected,
        "limitations": [
            "three selected expert ranges only; shard payloads remain unverified",
            "reference-only CPU correctness evidence; no AVX2 speed or throughput claim",
            "no Tesla P4, cache, hybrid split, full-model or serving claim",
        ],
    }
    if checkpoint_path is not None:
        _write_q3_checkpoint(
            checkpoint_path,
            {
                "schema_version": _Q3_TRIAD_CHECKPOINT_SCHEMA_VERSION,
                "status": "complete",
                "identity": checkpoint_identity,
                "audit": {"current": current_audit, "history": audit_history},
                "completed": selected,
            },
        )
    return report


__all__ = [
    "DEFAULT_Q3_TRIAD",
    "DEFAULT_Q3_VARIANT",
    "ArtifactProbeError",
    "RangeFetcher",
    "RangeResponse",
    "UrllibRangeTransport",
    "build_probe_layout",
    "gguf_oracle_identity",
    "load_qwen38_expert_artifact",
    "probe_qwen38_expert",
    "probe_qwen38_q3_triad",
    "run_gguf_oracle",
]
