from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from freetoken.moe import real_artifact_benchmark as benchmark
from freetoken.moe.real_artifact_probe import ArtifactProbeError

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_evidence", ROOT / "scripts" / "validate_evidence.py"
)
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
VALIDATE_EVIDENCE = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(VALIDATE_EVIDENCE)


class _Descriptor:
    def __init__(self, projection: str, quant_name: str = "Q4_K", *, layer: int = 0) -> None:
        self.layer_id = layer
        self.projection = projection
        self.quant_type = {"Q4_K": 12, "Q5_K": 13, "Q8_0": 8}[quant_name]
        self.quant_name = quant_name
        self.num_experts = 1
        self.output_dim = 2 if projection != "down" else 4
        self.input_dim = 4 if projection != "down" else 2
        self.row_stride_bytes = 1
        self.expert_stride_bytes = self.output_dim
        self.tensor_bytes = self.output_dim
        self.source_offset = 100


class _Layout:
    top_k = 2

    def __init__(self, *, layer: int = 0, down_quant: str = "Q4_K") -> None:
        self.layer = layer
        self.layers = (layer,)
        self.descriptors = tuple(
            _Descriptor(
                projection,
                down_quant if projection == "down" else ("Q5_K" if layer == 2 else "Q4_K"),
                layer=layer,
            )
            for projection in ("gate", "up", "down")
        )

    def descriptor(self, layer: int, projection: str) -> _Descriptor:
        assert layer == self.layer
        return next(item for item in self.descriptors if item.projection == projection)


class _Telemetry:
    def __init__(
        self,
        backend: str = "mixed_avx2",
        fallback_reason: str | None = None,
        kernels: tuple[str, ...] = ("q4_k_avx2", "q4_k_avx2"),
    ) -> None:
        self.backend = backend
        self.fallback_reason = fallback_reason
        self.kernel_census = kernels

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
        self.layout = layout
        del kwargs
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
        assert layer == self.layout.layer
        assert hidden.shape == (1, 4)
        assert expert_ids.tolist() == [[0]]
        assert weights.tolist() == [[1.0]]
        self.calls += 1
        output = np.full((1, 4), self.output_delta, dtype=np.float32)
        kernels = tuple(
            f"{descriptor.quant_name.lower()}_avx2" for descriptor in self.layout.descriptors
        )
        return _Result(output, _Telemetry(kernels=kernels))

    def _backend_for(self, layer: int) -> str:
        assert layer == self.layout.layer
        return self.backend

    def _kernel_census(self, layer: int) -> tuple[str, ...]:
        assert layer == self.layout.layer
        return tuple(
            f"{descriptor.quant_name.lower()}_avx2" for descriptor in self.layout.descriptors
        )

    def close(self) -> None:
        pass


def _artifact(*, layer: int = 0) -> dict[str, object]:
    return {
        "source": {
            "repository": "owner/model",
            "revision": "a" * 40,
            "variant": "UD-Q4_K_XL",
            "base_url": "https://example/model",
        },
        "census": {
            "model_sha256": "c" * 64,
            "evidence_status": "artifact-metadata",
            "shards": [{"sha256_status": "declared"}],
        },
        "census_sha256": "d" * 64,
        "layout": _Layout(layer=layer, down_quant="Q8_0" if layer == 2 else "Q4_K"),
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
    for variable in benchmark._BLAS_THREAD_VARS:
        monkeypatch.setenv(variable, "1")
    monkeypatch.setattr(
        benchmark.probe,
        "load_qwen38_expert_artifact",
        lambda **kwargs: _artifact(layer=kwargs["layer"]),
    )
    monkeypatch.setattr(benchmark.probe, "_load_gguf_oracle", lambda: object())
    monkeypatch.setattr(benchmark.probe, "Q4KExecutor", _FakeExecutor)
    monkeypatch.setattr(
        benchmark,
        "_native_library_metadata",
        lambda path: {
            "libraries": {
                "q4_k": {
                    "environment": "FREETOKEN_Q4K_NATIVE_LIB",
                    "path": "q4.so",
                    "sha256": "a" * 64,
                },
                "mixed_gemv": {
                    "environment": "FREETOKEN_MIXED_GEMV_NATIVE_LIB",
                    "path": "mixed.so",
                    "sha256": "b" * 64,
                },
            },
            "build": {"commit": "1" * 40, "libraries": {}},
        },
    )
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


def test_benchmark_records_promoted_layer2_kernel_census(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common(monkeypatch)
    report = benchmark.benchmark_qwen38_expert(
        manifest_path=Path("manifest.json"),
        census_path=Path("census.json"),
        layer=2,
        repeats=1,
        warmup=5,
        offline=True,
    )

    assert report["workload"]["layer"] == 2
    assert report["selected_behavior"]["kernel_census"] == [
        "q5_k_avx2",
        "q5_k_avx2",
        "q8_0_avx2",
    ]


def test_benchmark_report_schema_and_semantics_are_machine_validatable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common(monkeypatch)
    report = benchmark.benchmark_qwen38_expert(
        manifest_path=Path("manifest.json"),
        census_path=Path("census.json"),
        repeats=1,
        warmup=5,
        offline=True,
    )

    assert VALIDATE_EVIDENCE.validate_document(report, schema_dir=ROOT / "schemas") == []

    invalid = copy.deepcopy(report)
    invalid["samples"]["reference_cold_dequant_dense"][0]["timed_components"].remove(
        "sha256 hashing"
    )
    invalid["samples"]["reference_cold_dequant_dense"][0]["timed_components"].append(
        "unrecognized work"
    )
    errors = VALIDATE_EVIDENCE.validate_document(invalid, schema_dir=ROOT / "schemas")
    assert any("cold sample 0 omits timed components" in error for error in errors)

    invalid = copy.deepcopy(report)
    invalid["samples"]["native"][0]["telemetry"]["fallback_reason"] = "scalar fallback"
    errors = VALIDATE_EVIDENCE.validate_document(invalid, schema_dir=ROOT / "schemas")
    assert any("samples.native[0] reports fallback telemetry" in error for error in errors)


def test_native_build_metadata_is_required_and_hash_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "1" * 40)
    with pytest.raises(ArtifactProbeError, match="metadata is required"):
        benchmark._native_library_metadata(None)

    libraries = {}
    for name, env_name in (
        ("q4_k", "FREETOKEN_Q4K_NATIVE_LIB"),
        ("mixed_gemv", "FREETOKEN_MIXED_GEMV_NATIVE_LIB"),
    ):
        path = tmp_path / f"{name}.so"
        path.write_bytes(name.encode())
        monkeypatch.setenv(env_name, str(path))
        libraries[name] = {"sha256": hashlib.sha256(name.encode()).hexdigest()}
    build_path = tmp_path / "build.json"
    build_path.write_text(
        json.dumps({"commit": "1" * 40, "libraries": libraries}), encoding="utf-8"
    )

    metadata = benchmark._native_library_metadata(build_path)
    assert metadata["build"]["commit"] == "1" * 40
    libraries["q4_k"]["sha256"] = "0" * 64
    build_path.write_text(
        json.dumps({"commit": "1" * 40, "libraries": libraries}), encoding="utf-8"
    )
    with pytest.raises(ArtifactProbeError, match="hash mismatch for q4_k"):
        benchmark._native_library_metadata(build_path)
