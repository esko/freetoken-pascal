"""H0 probe for selected Qwen3.8 GGUF expert byte ranges.

The probe is intentionally narrower than a model loader.  It fetches one expert
from each of the selected layer's gate, up and down banks with HTTP Range, builds
the existing CPU expert ABI over those three byte ranges, and records a scalar
versus forced-AVX2 A/B run.  It never downloads or hashes a complete shard.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
SUPPORTED_EXPERT_TYPES = frozenset({7, 8, 12, 13})
_CONTENT_RANGE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)
_PROJECTIONS = ("gate", "up", "down")


class ArtifactProbeError(RuntimeError):
    """Raised when an artifact range or descriptor cannot be proven safe."""


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

    def _cache_stem(self, url: str, start: int, size: int, expected_total: int) -> str:
        key = f"{url}\0{start}\0{size}\0{expected_total}".encode()
        return hashlib.sha256(key).hexdigest()

    def _load_cache(
        self, url: str, start: int, size: int, expected_total: int
    ) -> RangeResponse | None:
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
        finally:
            body_tmp.unlink(missing_ok=True)
            metadata_tmp.unlink(missing_ok=True)

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
        cached = self._load_cache(url, start, size, expected_total)
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
    difference = np.abs(scalar_output - avx_output)
    finite = bool(np.isfinite(scalar_output).all() and np.isfinite(avx_output).all())
    max_abs = float(difference.max(initial=0.0))
    correct = finite and bool(np.allclose(scalar_output, avx_output, rtol=5e-4, atol=5e-4))
    promoted = any(
        descriptor["quant_name"] in {"Q5_K", "Q8_0"}
        for descriptor in _descriptor_summary(layout, expert=int(expert))
    )
    return {
        "schema_version": 1,
        "evidence_status": "artifact-metadata",
        "range_evidence": "measured/artifact-byte",
        "validation_class": "H0/no-P4",
        "limitations": [
            "selected expert ranges only; no complete shard download or checksum",
            "CPU executor A/B only; no P4, cache, hybrid split or full-engine claim",
        ],
        "source": source,
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
            "comparison": "internal Q4KExecutor scalar/reference versus forced-AVX2 paths",
            "independent_oracle": False,
            "correct": correct,
            "finite": finite,
            "max_abs_error": max_abs,
            "rtol": 5e-4,
            "atol": 5e-4,
            "scalar": scalar,
            "avx2": avx2,
        },
    }


__all__ = [
    "ArtifactProbeError",
    "RangeFetcher",
    "RangeResponse",
    "UrllibRangeTransport",
    "build_probe_layout",
    "probe_qwen38_expert",
]
