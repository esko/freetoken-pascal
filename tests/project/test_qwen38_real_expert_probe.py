from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest
from freetoken.moe import real_artifact_probe
from freetoken.moe.real_artifact_probe import (
    ArtifactProbeError,
    RangeFetcher,
    RangeResponse,
    UrllibRangeTransport,
    build_probe_layout,
    probe_qwen38_expert,
    run_gguf_oracle,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "manifests/qwen38-gguf.json"
CENSUS = ROOT / "tests/fixtures/results/qwen38-q4-census.metadata.json"


def _q8_row(codes: np.ndarray) -> bytes:
    """Build one deterministic GGML Q8_0 row for the independent oracle tests."""
    assert codes.shape == (32,)
    scale = np.asarray(np.array([1.0], dtype="<f2").view(np.uint8), dtype=np.uint8).tobytes()
    return scale + np.asarray(codes, dtype=np.int8).tobytes()


def _oracle_layout_and_sources() -> tuple[object, dict[str, bytes]]:
    descriptors = []
    sources: dict[str, bytes] = {}
    for projection in ("gate", "up", "down"):
        rows = []
        for row in range(32):
            codes = np.zeros(32, dtype=np.int8)
            if projection == "gate":
                codes[0] = 1 if row == 1 else 0
            elif projection == "up":
                codes[0] = 2 if row == 1 else 0
            elif row == 0:
                codes[1] = 1
            rows.append(_q8_row(codes))
        packed = b"".join(rows)
        sources[projection] = packed
        descriptors.append(
            real_artifact_probe.CpuExpertDescriptor(
                layer_id=0,
                projection=projection,
                quant_type=8,
                quant_name="Q8_0",
                num_experts=1,
                output_dim=32,
                input_dim=32,
                rows_per_expert=32,
                row_stride_bytes=34,
                expert_stride_bytes=32 * 34,
                tensor_bytes=32 * 34,
                source=np.frombuffer(packed, dtype=np.uint8).reshape(1, 32, 34),
            )
        )
    return real_artifact_probe.CpuExpertLayout(tuple(descriptors), top_k=2), sources


def test_gguf_oracle_uses_projection_orientation_and_duplicate_route_weights() -> None:
    pytest.importorskip("gguf")
    layout, sources = _oracle_layout_and_sources()
    hidden = np.ones((1, 32), dtype=np.float32)
    expert_ids = np.array([[0, 0]], dtype=np.int32)
    weights = np.array([[0.25, 0.5]], dtype=np.float32)

    actual = run_gguf_oracle(layout, sources, hidden, expert_ids, weights)

    expected = np.zeros((1, 32), dtype=np.float32)
    expected[0, 0] = 0.75 * (2.0 / (1.0 + np.exp(-1.0)))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_gguf_oracle_rejects_fetched_source_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("freetoken.moe.real_artifact_probe._load_gguf_oracle", lambda: object())
    layout, sources = _oracle_layout_and_sources()
    sources["down"] = sources["down"][:-1]
    with pytest.raises(ArtifactProbeError, match=r"down.*bytes"):
        run_gguf_oracle(
            layout,
            sources,
            np.ones((1, 32), dtype=np.float32),
            np.array([[0]], dtype=np.int32),
            np.array([[1.0]], dtype=np.float32),
        )


def test_gguf_oracle_dependency_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def reject_gguf(name, *args, **kwargs):
        if name == "gguf":
            raise ModuleNotFoundError("simulated missing gguf")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_gguf)
    with pytest.raises(ArtifactProbeError, match=r"gguf-py==0.19.0.*cpu.lock"):
        real_artifact_probe._load_gguf_oracle()


def test_probe_reports_oracle_mismatch_separately_from_scalar_native_ab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, payloads, total = _ranges_for_layer(0)
    transport = _FakeTransport(payloads, total=total)
    monkeypatch.setattr("freetoken.moe.real_artifact_probe._load_gguf_oracle", lambda: object())

    def fake_mode(*args, **kwargs):
        output = np.zeros((2, 2560), dtype=np.float32)
        if kwargs["mode"] == "forced_avx2":
            output[0, 0] = 1.0
        return output, {
            "requested_mode": kwargs["mode"],
            "selected_backend": "reference",
            "q4k_isa": "scalar",
            "q4k_fallback_reason": None,
            "mixed_isa": "scalar",
            "mixed_fallback_reason": None,
            "raw_elapsed_ns": [1],
            "telemetry": [],
        }

    monkeypatch.setattr("freetoken.moe.real_artifact_probe._run_mode", fake_mode)
    monkeypatch.setattr(
        "freetoken.moe.real_artifact_probe._run_gguf_oracle",
        lambda *args, **kwargs: (
            np.zeros((2, 2560), dtype=np.float32),
            real_artifact_probe.gguf_oracle_identity(),
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

    assert result["ab"]["oracle_vs_scalar"]["correct"] is True
    assert result["ab"]["oracle_vs_native"]["correct"] is False
    assert result["ab"]["scalar_vs_native"]["correct"] is False
    assert result["ab"]["oracle_vs_native"]["max_abs_error"] == 1.0
    assert result["ab"]["correct"] is False


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

    cache_file.write_bytes(b"oversized")
    with pytest.raises(ArtifactProbeError, match="length"):
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

    duplicate = json.loads(json.dumps(census))
    duplicate["tensors"].append(
        next(
            record
            for record in duplicate["tensors"]
            if record["name"] == "blk.0.ffn_gate_exps.weight"
        )
    )
    with pytest.raises(ArtifactProbeError, match="duplicate tensor"):
        build_probe_layout(manifest, duplicate, layer=0, expert=0, sources=sources)

    negative_shard = json.loads(json.dumps(census))
    next(
        record
        for record in negative_shard["tensors"]
        if record["name"] == "blk.0.ffn_gate_exps.weight"
    )["shard_index"] = -1
    with pytest.raises(ArtifactProbeError, match="invalid shard index"):
        build_probe_layout(manifest, negative_shard, layer=0, expert=0, sources=sources)


def test_probe_fetches_normal_and_promoted_layers_and_reports_ab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real-artifact run is intentionally delegated to the command-level test
    # harness; this unit-level transport fixture only verifies the range API's
    # shape and remains small enough for hosted CI.
    census, payloads, total = _ranges_for_layer(0)
    del census
    transport = _FakeTransport(payloads, total=total)
    monkeypatch.setattr("freetoken.moe.real_artifact_probe._load_gguf_oracle", lambda: object())
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
    monkeypatch.setattr(
        "freetoken.moe.real_artifact_probe._run_gguf_oracle",
        lambda *args, **kwargs: (
            np.zeros((2, 2560), dtype=np.float32),
            real_artifact_probe.gguf_oracle_identity(),
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
    assert result["oracle"]["name"] == "gguf-py"
    assert result["oracle"]["version"] == "0.19.0"
    assert result["ab"]["oracle"]["operation"] == "dequantize + FP32 dense SwiGLU"
    assert len(result["ab"]["scalar"]["raw_elapsed_ns"]) == 1
    assert len(transport.requests) == 3


def test_promoted_layer_selection_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, payloads, total = _ranges_for_layer(2)
    transport = _FakeTransport(payloads, total=total)
    monkeypatch.setattr("freetoken.moe.real_artifact_probe._load_gguf_oracle", lambda: object())
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
    monkeypatch.setattr(
        "freetoken.moe.real_artifact_probe._run_gguf_oracle",
        lambda *args, **kwargs: (
            np.zeros((2, 2560), dtype=np.float32),
            real_artifact_probe.gguf_oracle_identity(),
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


def test_run_mode_executes_the_selected_nonzero_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class _Telemetry:
        backend = "reference"
        kernel_census = ("reference",)

        def as_dict(self) -> dict[str, object]:
            return {"backend": self.backend, "kernel_census": self.kernel_census}

    class _Result:
        output = np.zeros((1, 1), dtype=np.float32)
        telemetry = _Telemetry()

    class _Primitive:
        isa = "scalar"
        fallback_reason = "test"

    class _Executor:
        primitive = _Primitive()
        mixed_primitive = _Primitive()
        backend = "reference"

        def __init__(self, layout, **kwargs):
            del layout, kwargs

        def prepare(self, tokens: int, routes: int) -> None:
            del tokens, routes

        def execute(self, layer_id: int, hidden, expert_ids, weights):
            del hidden, expert_ids, weights
            calls.append(layer_id)
            return _Result()

        def close(self) -> None:
            pass

    class _Layout:
        layers = (2,)

    monkeypatch.setattr(real_artifact_probe, "Q4KExecutor", _Executor)
    real_artifact_probe._run_mode(
        _Layout(),
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.int32),
        np.ones((1, 1), dtype=np.float32),
        mode="forced_scalar",
        repeats=1,
        warmup=1,
    )
    assert calls == [2, 2]
