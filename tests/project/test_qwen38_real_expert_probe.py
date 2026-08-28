from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from freetoken.moe.real_artifact_probe import (
    ArtifactProbeError,
    RangeFetcher,
    RangeResponse,
    UrllibRangeTransport,
    build_probe_layout,
    probe_qwen38_expert,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "manifests/qwen38-gguf.json"
CENSUS = ROOT / "tests/fixtures/results/qwen38-q4-census.metadata.json"


def _metadata(*, layer: int = 0, expert: int = 0) -> tuple[dict, dict, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    records = {
        record["name"]: record
        for record in census["tensors"]
        if record["name"].startswith(f"blk.{layer}.ffn_")
    }
    return manifest, census, records


class _FakeTransport:
    def __init__(self, payloads: dict[tuple[int, int], bytes], *, total: int) -> None:
        self.payloads = payloads
        self.total = total
        self.requests: list[tuple[str, int, int]] = []
        self.mode = "ok"

    def fetch(self, url: str, start: int, end: int) -> RangeResponse:
        self.requests.append((url, start, end))
        body = self.payloads[(start, end)]
        if self.mode == "short":
            body = body[:-1]
        response_start = start
        response_end = end
        if self.mode == "wrong-range":
            response_start += 1
        headers = {
            "Content-Range": f"bytes {response_start}-{response_end}/{self.total}",
            "Content-Length": str(len(body)),
        }
        return RangeResponse(status=206, headers=headers, body=body)


def _ranges_for_layer(layer: int, *, expert: int = 0):
    _, census, records = _metadata(layer=layer, expert=expert)
    shard = census["shards"][1]
    payloads = {}
    for projection in ("gate", "up", "down"):
        record = records[f"blk.{layer}.ffn_{projection}_exps.weight"]
        size = int(record["shape"][1]) * int(record["row_bytes"])
        start = int(record["offset"]) + expert * size
        payloads[(start, start + size - 1)] = bytes((index * 13) % 256 for index in range(size))
    return census, payloads, int(shard["size"])


def test_range_fetcher_requires_exact_content_range_and_length(tmp_path: Path) -> None:
    transport = _FakeTransport({(10, 13): b"abcd"}, total=100)
    fetcher = RangeFetcher(transport, cache_dir=tmp_path)
    assert fetcher.fetch("https://example/model", 10, 4, expected_total=100).body == b"abcd"
    assert transport.requests == [("https://example/model", 10, 13)]

    transport.mode = "short"
    fetcher = RangeFetcher(transport)
    with pytest.raises(ArtifactProbeError, match="length"):
        fetcher.fetch("https://example/model", 10, 4, expected_total=100)

    transport.mode = "wrong-range"
    fetcher = RangeFetcher(transport)
    with pytest.raises(ArtifactProbeError, match="Content-Range"):
        fetcher.fetch("https://example/model", 10, 4, expected_total=100)

    transport = _FakeTransport({}, total=100)
    fetcher = RangeFetcher(transport)
    with pytest.raises(ArtifactProbeError, match="invalid range"):
        fetcher.fetch("https://example/model", 99, 2, expected_total=100)
    assert transport.requests == []


def test_range_fetcher_rejects_corrupt_cached_checksum(tmp_path: Path) -> None:
    transport = _FakeTransport({(10, 13): b"abcd"}, total=100)
    fetcher = RangeFetcher(transport, cache_dir=tmp_path)
    fetcher.fetch("https://example/model", 10, 4, expected_total=100)
    cache_file = next(tmp_path.glob("*.bin"))
    cache_file.write_bytes(b"abce")
    with pytest.raises(ArtifactProbeError, match="checksum"):
        fetcher.fetch("https://example/model", 10, 4, expected_total=100, offline=True)


def test_range_fetcher_serves_a_valid_range_from_offline_cache(tmp_path: Path) -> None:
    transport = _FakeTransport({(10, 13): b"abcd"}, total=100)
    fetcher = RangeFetcher(transport, cache_dir=tmp_path)
    fetcher.fetch("https://example/model", 10, 4, expected_total=100)
    transport.payloads.clear()
    response = fetcher.fetch("https://example/model", 10, 4, expected_total=100, offline=True)
    assert response.body == b"abcd"
    assert fetcher.cache_hits == 1


def test_urllib_transport_rejects_ignored_range_before_reading_body() -> None:
    class _Response:
        def __init__(self) -> None:
            self.status = 200
            self.headers = {"Content-Length": "50000000000"}
            self.read_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            del args

        def read(self, size: int = -1) -> bytes:
            self.read_calls += 1
            del size
            return b"x" * 4

    response = _Response()
    transport = UrllibRangeTransport(opener=lambda request, timeout: response)
    with pytest.raises(ArtifactProbeError, match="206"):
        transport.fetch("https://example/model", 10, 13)
    assert response.read_calls == 0


def test_urllib_transport_caps_body_read_to_range_plus_one() -> None:
    class _Response:
        def __init__(self) -> None:
            self.status = 206
            self.headers = {
                "Content-Range": "bytes 10-13/100",
                "Content-Length": "4",
            }
            self.read_size = None

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            del args

        def read(self, size: int = -1) -> bytes:
            self.read_size = size
            return b"abcde"

    response = _Response()
    transport = UrllibRangeTransport(opener=lambda request, timeout: response)
    with pytest.raises(ArtifactProbeError, match="length"):
        RangeFetcher(transport).fetch("https://example/model", 10, 4, expected_total=100)
    assert response.read_size == 5


def test_build_probe_layout_rejects_unknown_type_and_bad_offsets() -> None:
    manifest, census, records = _metadata()
    sources = {}
    for projection in ("gate", "up", "down"):
        record = records[f"blk.0.ffn_{projection}_exps.weight"]
        row_bytes = int(record["row_bytes"])
        output_dim = int(record["shape"][1])
        sources[projection] = bytes(output_dim * row_bytes)

    bad_type = json.loads(json.dumps(census))
    bad_type_record = next(
        record for record in bad_type["tensors"] if record["name"] == "blk.0.ffn_gate_exps.weight"
    )
    bad_type_record["quant_type"] = 999
    with pytest.raises(ArtifactProbeError, match="quant"):
        build_probe_layout(manifest, bad_type, layer=0, expert=0, sources=sources)

    bad_offset = json.loads(json.dumps(census))
    bad_offset_record = next(
        record for record in bad_offset["tensors"] if record["name"] == "blk.0.ffn_down_exps.weight"
    )
    bad_offset_record["offset"] = bad_offset["shards"][1]["size"]
    with pytest.raises(ArtifactProbeError, match="bounds"):
        build_probe_layout(manifest, bad_offset, layer=0, expert=0, sources=sources)


def test_probe_fetches_normal_and_promoted_layers_and_reports_ab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real-artifact run is intentionally delegated to the command-level test
    # harness; this unit-level transport fixture only verifies the range API's
    # shape and remains small enough for hosted CI.
    census, payloads, total = _ranges_for_layer(0)
    del census
    transport = _FakeTransport(payloads, total=total)
    monkeypatch.setattr(
        "freetoken.moe.real_artifact_probe._run_mode",
        lambda *args, **kwargs: (
            np.zeros((2, 2560), dtype=np.float32),
            {
                "requested_mode": kwargs["mode"],
                "selected_backend": "reference",
                "q4k_isa": "scalar",
                "q4k_fallback_reason": None,
                "mixed_isa": "scalar",
                "mixed_fallback_reason": None,
                "raw_elapsed_ns": [1],
                "telemetry": [],
            },
        ),
    )
    result = probe_qwen38_expert(
        manifest_path=MANIFEST,
        census_path=CENSUS,
        layer=0,
        expert=0,
        repeats=1,
        warmup=0,
        transport=transport,
        cache_dir=tmp_path,
    )
    assert result["evidence_status"] == "artifact-metadata"
    assert result["range_evidence"] == "measured/artifact-byte"
    assert result["validation_class"] == "H0/no-P4"
    assert result["selection"]["layer"] == 0
    assert result["fetch"]["full_shard_bytes"] == 0
    assert result["fetch"]["range_count"] == 3
    assert result["ab"]["correct"] is True
    assert len(result["ab"]["scalar"]["raw_elapsed_ns"]) == 1
    assert len(transport.requests) == 3


def test_promoted_layer_selection_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, payloads, total = _ranges_for_layer(2)
    transport = _FakeTransport(payloads, total=total)
    monkeypatch.setattr(
        "freetoken.moe.real_artifact_probe._run_mode",
        lambda *args, **kwargs: (
            np.zeros((2, 2560), dtype=np.float32),
            {
                "requested_mode": kwargs["mode"],
                "selected_backend": "reference",
                "q4k_isa": "scalar",
                "q4k_fallback_reason": None,
                "mixed_isa": "scalar",
                "mixed_fallback_reason": None,
                "raw_elapsed_ns": [1],
                "telemetry": [],
            },
        ),
    )
    result = probe_qwen38_expert(
        manifest_path=MANIFEST,
        census_path=CENSUS,
        layer=2,
        expert=0,
        repeats=1,
        warmup=0,
        transport=transport,
        cache_dir=tmp_path,
    )
    assert result["selection"]["promoted"] is True
    assert {item["quant_name"] for item in result["layout"]["descriptors"]} == {
        "Q5_K",
        "Q8_0",
    }
