from __future__ import annotations

import itertools
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from freetoken.moe import real_artifact_benchmark as benchmark
from freetoken.moe.real_artifact_probe import ArtifactProbeError


class _Descriptor:
    def __init__(self, projection: str, quant_name: str = "Q4_K") -> None:
        self.layer_id = 0
        self.projection = projection
        self.quant_type = 12
        self.quant_name = quant_name
        self.num_experts = 1
        self.output_dim = 2 if projection != "down" else 4
        self.input_dim = 4 if projection != "down" else 2
        self.row_stride_bytes = 1
        self.expert_stride_bytes = self.output_dim
        self.tensor_bytes = self.output_dim
        self.source_offset = 100


class _Layout:
    layers = (0,)
    top_k = 2

    def __init__(self, *, down_quant: str = "Q4_K") -> None:
        self.descriptors = tuple(
            _Descriptor(projection, down_quant if projection == "down" else "Q4_K")
            for projection in ("gate", "up", "down")
        )

    def descriptor(self, layer: int, projection: str) -> _Descriptor:
        assert layer == 0
        return next(item for item in self.descriptors if item.projection == projection)


class _Telemetry:
    def __init__(self, backend: str = "mixed_avx2", fallback_reason: str | None = None) -> None:
        self.backend = backend
        self.fallback_reason = fallback_reason
        self.kernel_census = ("q4_k_avx2", "q4_k_avx2")

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "fallback_reason": self.fallback_reason,
            "kernel_census": list(self.kernel_census),
        }


class _Result:
    def __init__(self, output: np.ndarray, telemetry: _Telemetry) -> None:
        self.output = output
        self.telemetry = telemetry


class _Primitive:
    def __init__(self, isa: str = "avx2", fallback_reason: str | None = None) -> None:
        self.isa = isa
        self.fallback_reason = fallback_reason
        self.backend = f"q4_k_{isa}"


class _FakeExecutor:
    instances: ClassVar[list[_FakeExecutor]] = []
    output_delta = 0.0
    primitive_isa = "avx2"

    def __init__(self, layout, **kwargs) -> None:
        del layout, kwargs
        self.primitive = _Primitive(self.primitive_isa)
        self.mixed_primitive = _Primitive(self.primitive_isa)
        self.backend = "mixed_avx2"
        self.calls = 0
        self.prepared = False
        self.instances.append(self)

    def prepare(self, tokens: int, routes: int) -> None:
        assert (tokens, routes) == (1, 1)
        self.prepared = True

    def execute(self, layer: int, hidden, expert_ids, weights):
        assert self.prepared
        assert layer == 0
        assert hidden.shape == (1, 4)
        assert expert_ids.tolist() == [[0]]
        assert weights.tolist() == [[1.0]]
        self.calls += 1
        output = np.full((1, 4), self.output_delta, dtype=np.float32)
        return _Result(output, _Telemetry())

    def _backend_for(self, layer: int) -> str:
        assert layer == 0
        return self.backend

    def _kernel_census(self, layer: int) -> tuple[str, ...]:
        assert layer == 0
        return ("q4_k_avx2", "q4_k_avx2")

    def close(self) -> None:
        pass


def _artifact() -> dict[str, object]:
    return {
        "source": {
            "repository": "owner/model",
            "revision": "a" * 40,
            "variant": "UD-Q4_K_XL",
            "base_url": "https://example/model",
        },
        "layout": _Layout(),
        "sources": {projection: b"x" for projection in ("gate", "up", "down")},
        "ranges": [
            {
                "projection": projection,
                "artifact_offset": 100 + index,
                "length": 1,
                "sha256": f"{index + 1}" * 64,
                "cache": "hit",
            }
            for index, projection in enumerate(("gate", "up", "down"))
        ],
        "fetch": {
            "transport": "http-range",
            "offline": True,
            "cache_dir": "/cache",
            "cache_hits": 3,
            "cache_misses": 0,
            "range_count": 3,
            "fetched_bytes": 3,
            "full_shard_bytes": 0,
        },
    }


def _patch_common(monkeypatch: pytest.MonkeyPatch, *, output_delta: float = 0.0) -> list[int]:
    calls: list[int] = []
    monkeypatch.setattr(
        benchmark.probe, "load_qwen38_expert_artifact", lambda **kwargs: _artifact()
    )
    monkeypatch.setattr(benchmark.probe, "_load_gguf_oracle", lambda: object())
    monkeypatch.setattr(benchmark.probe, "Q4KExecutor", _FakeExecutor)
    _FakeExecutor.instances.clear()
    _FakeExecutor.output_delta = output_delta

    def fake_dequant(*args, **kwargs):
        del args, kwargs
        calls.append(1)
        return {
            "gate": np.zeros((2, 4), dtype=np.float32),
            "up": np.zeros((2, 4), dtype=np.float32),
            "down": np.zeros((4, 2), dtype=np.float32),
        }, {"gate": "1" * 64, "up": "2" * 64, "down": "3" * 64}

    monkeypatch.setattr(benchmark.probe, "_oracle_dense_projections", fake_dequant)
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "b" * 40)
    monkeypatch.setattr(benchmark, "_host_metadata", lambda: {"isa": ["avx2", "fma"]})
    return calls


def test_benchmark_times_dequant_each_reference_sample_and_excludes_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dequant_calls = _patch_common(monkeypatch)
    clock = itertools.count()
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", lambda: next(clock) * 10)

    report = benchmark.benchmark_qwen38_expert(
        manifest_path=Path("manifest.json"),
        census_path=Path("census.json"),
        layer=0,
        expert=0,
        repeats=2,
        warmup=5,
        offline=True,
        command="bench command",
    )

    assert len(dequant_calls) == 8
    assert _FakeExecutor.instances[0].calls == 7
    assert report["workload"]["tokens"] == 1
    assert report["workload"]["route_count"] == 1
    assert report["warmups"]["reference_cold_dequant_dense"][0]["dequantized"] is True
    assert report["warmups"]["native"][0]["timed_operation"] == "executor.execute"
    assert len(report["samples"]["reference_cold_dequant_dense"]) == 2
    assert len(report["samples"]["native"]) == 2
    assert report["oracle"]["resident_setup_elapsed_ns"] == 10
    assert report["samples"]["reference_cold_dequant_dense"][0]["elapsed_ns"] == 40
    assert report["samples"]["reference_dense_resident"][0]["elapsed_ns"] == 10
    assert report["samples"]["native"][0]["elapsed_ns"] == 10
    assert report["statistics"]["scope"]
    assert report["metadata"]["commit"] == "b" * 40
    assert report["metadata"]["manifest_revision"] == "a" * 40
    assert report["metadata"]["range_hashes"] == {
        "gate": "1" * 64,
        "up": "2" * 64,
        "down": "3" * 64,
    }


def test_benchmark_fails_closed_when_native_avx2_is_not_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common(monkeypatch)
    _FakeExecutor.primitive_isa = "scalar"
    try:
        with pytest.raises(ArtifactProbeError, match="native AVX2"):
            benchmark.benchmark_qwen38_expert(
                manifest_path=Path("manifest.json"),
                census_path=Path("census.json"),
                repeats=1,
                warmup=5,
                offline=True,
            )
    finally:
        _FakeExecutor.primitive_isa = "avx2"
    assert _FakeExecutor.instances[0].calls == 0


def test_benchmark_fails_closed_on_native_reference_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common(monkeypatch, output_delta=1.0)
    with pytest.raises(ArtifactProbeError, match="correctness"):
        benchmark.benchmark_qwen38_expert(
            manifest_path=Path("manifest.json"),
            census_path=Path("census.json"),
            repeats=1,
            warmup=5,
            offline=True,
        )


def test_benchmark_rejects_non_single_thread_blas(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(monkeypatch)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    with pytest.raises(ArtifactProbeError, match="BLAS thread variables"):
        benchmark.benchmark_qwen38_expert(
            manifest_path=Path("manifest.json"),
            census_path=Path("census.json"),
            repeats=1,
            warmup=5,
            offline=True,
        )
