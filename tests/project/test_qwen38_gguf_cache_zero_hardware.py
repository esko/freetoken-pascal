"""Opt-in H2 smoke for the pinned Qwen3.8 GGUF cache-zero path.

This test is intentionally separate from the ordinary CUDA smoke tests.  Engine startup
requires CUDA to be uninitialized, and a successful run must exercise the real Qwen GGUF
model, host expert banks, and dedicated PLE artifact rather than a tiny fixture.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODEL_MANIFEST = ROOT / "manifests/qwen38-gguf.json"
MODEL_VARIANT = "UD-Q4_K_XL"
EXPECTED_EXPERT_CENSUS = {
    "Q4_K": 94,
    "Q5_1": 43,
    "Q5_K": 2,
    "Q8_0": 5,
}


def _repository_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_evidence(evidence: dict[str, Any]) -> None:
    output = Path(
        os.environ.get(
            "FREETOKEN_PASCAL_H2_EVIDENCE",
            "results/hardware/qwen38-gguf-cache-zero-h2.json",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(evidence), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("QWEN38_GGUF_CACHE_ZERO_H2_EVIDENCE " + json.dumps(_jsonable(evidence), sort_keys=True))


def _sha256_file(path: Path) -> str:
    """Hash one pinned model shard without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned_model_identity(model_path: Path) -> tuple[dict[str, Any], list[Path]]:
    from freetoken.gguf_shards import gguf_shard_paths

    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    variant = manifest["variants"][MODEL_VARIANT]
    try:
        shards = list(gguf_shard_paths(model_path))
    except Exception as error:
        pytest.fail(f"pinned Q4 GGUF shard validation failed: {type(error).__name__}: {error}")
    expected_shards = variant["shards"]
    if [path.name for path in shards] != [item["name"] for item in expected_shards]:
        pytest.fail(
            "FREETOKEN_PASCAL_MODEL_PATH is not the pinned Q4 shard set: "
            f"got {[path.name for path in shards]!r}, "
            f"expected {[item['name'] for item in expected_shards]!r}"
        )
    observed_sizes = [path.stat().st_size for path in shards]
    declared_sizes = [int(item["size"]) for item in expected_shards]
    if observed_sizes != declared_sizes:
        pytest.fail(
            "pinned Q4 GGUF shard sizes do not match the manifest: "
            f"observed={observed_sizes!r}, declared={declared_sizes!r}"
        )
    observed_hashes = [_sha256_file(path) for path in shards]
    declared_hashes = [str(item["sha256"]) for item in expected_shards]
    if observed_hashes != declared_hashes:
        mismatches = [
            {
                "name": path.name,
                "expected": expected,
                "observed": observed,
            }
            for path, expected, observed in zip(
                shards, declared_hashes, observed_hashes, strict=True
            )
            if expected != observed
        ]
        pytest.fail(f"pinned Q4 GGUF shard SHA-256 mismatch: {mismatches!r}")
    return {
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "variant": MODEL_VARIANT,
        "shards": [
            {
                "name": item["name"],
                "size": int(item["size"]),
                "sha256": item["sha256"],
                "observed_sha256": observed,
                "sha256_status": "verified",
            }
            for item, observed in zip(expected_shards, observed_hashes, strict=True)
        ],
        "observed_sizes": observed_sizes,
        "quant_type_counts": variant["quant_type_counts"],
    }, shards


def _pinned_ple_identity(artifact_path: Path) -> dict[str, Any]:
    manifest_path = artifact_path / "manifest.json"
    payload_path = artifact_path / "ple.bin"
    if not manifest_path.is_file() or not payload_path.is_file():
        pytest.fail(
            "FREETOKEN_PASCAL_PLE_ARTIFACT must contain manifest.json and ple.bin; "
            f"got {artifact_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        pytest.fail(f"dedicated PLE manifest is unreadable: {type(error).__name__}: {error}")
    required = {
        "format",
        "version",
        "payload",
        "tensor_name",
        "quant_name",
        "codec",
        "rows",
        "elements_per_row",
        "row_bytes",
        "tensor_bytes",
        "sha256",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        pytest.fail(f"dedicated PLE manifest is missing fields: {missing!r}")
    if manifest["format"] != "freetoken-pascal-ple-v1" or manifest["version"] != 1:
        pytest.fail(
            "dedicated PLE artifact has unsupported identity: "
            f"format={manifest.get('format')!r}, version={manifest.get('version')!r}"
        )
    if manifest["payload"] != "ple.bin":
        pytest.fail(f"dedicated PLE manifest payload must be ple.bin, got {manifest['payload']!r}")
    if payload_path.stat().st_size != int(manifest["tensor_bytes"]):
        pytest.fail(
            "dedicated PLE payload size does not match its manifest: "
            f"observed={payload_path.stat().st_size}, declared={manifest['tensor_bytes']}"
        )
    return {
        "path": str(artifact_path),
        "sha256": manifest["sha256"],
        "payload": manifest["payload"],
        "payload_bytes": payload_path.stat().st_size,
        "tensor_name": manifest["tensor_name"],
        "quant_name": manifest["quant_name"],
        "codec": manifest["codec"],
        "rows": int(manifest["rows"]),
        "elements_per_row": int(manifest["elements_per_row"]),
        "row_bytes": int(manifest["row_bytes"]),
        "tensor_bytes": int(manifest["tensor_bytes"]),
    }


def _assert_p4(torch: Any, device: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    assert properties.name == "Tesla P4"
    assert (properties.major, properties.minor) == (6, 1)
    usable_mib = properties.total_memory // (1024 * 1024)
    assert 7580 <= usable_mib <= 8192
    return {
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "usable_memory_mib": usable_mib,
    }


@pytest.mark.sm61
@pytest.mark.large_model
def test_qwen38_gguf_cache_zero_real_engine_prefill_decode() -> None:
    """Run one bounded text request through the real Engine on one verified P4."""
    model_value = os.environ.get("FREETOKEN_PASCAL_MODEL_PATH")
    if not model_value:
        pytest.skip("FREETOKEN_PASCAL_MODEL_PATH is unset")
    ple_value = os.environ.get("FREETOKEN_PASCAL_PLE_ARTIFACT")
    if not ple_value:
        pytest.skip("FREETOKEN_PASCAL_PLE_ARTIFACT is unset")

    model_path = Path(model_value)
    artifact_path = Path(ple_value)
    ple_backend = os.environ.get("FREETOKEN_PASCAL_PLE_BACKEND", "mmap")
    if ple_backend not in {"mmap", "pread"}:
        pytest.fail(f"unsupported FREETOKEN_PASCAL_PLE_BACKEND={ple_backend!r}")
    model_identity, _shards = _pinned_model_identity(model_path)
    ple_identity = _pinned_ple_identity(artifact_path)

    from freetoken.gguf_host import inspect_qwen_host_layout

    try:
        layout = inspect_qwen_host_layout(model_path)
    except Exception as error:
        pytest.fail(f"pinned Q4 host-layout inspection failed: {type(error).__name__}: {error}")
    census = Counter(descriptor.quant_name for descriptor in layout.experts.descriptors)
    assert dict(census) == EXPECTED_EXPERT_CENSUS
    assert len(layout.experts.descriptors) == sum(EXPECTED_EXPERT_CENSUS.values())

    torch = pytest.importorskip("torch")
    from freetoken.core import SamplingParams
    from freetoken.llm import LLM

    llm = None
    try:
        try:
            # Do not probe CUDA before LLM/Engine construction: Engine intentionally requires
            # CUDA to be uninitialized while it binds the assigned P4.
            startup_started = time.monotonic()
            llm = LLM(
                str(model_path),
                dtype=torch.bfloat16,
                max_running_req=1,
                max_extend_tokens=8,
                max_seq_len_override=128,
                num_token_override=128,
                attention_backend="auto",
                moe_backend="cpu",
                moe_cache_size=0,
                moe_cache_auto=False,
                moe_cache_rate=None,
                moe_cpu_threads=int(os.environ.get("FREETOKEN_PASCAL_CPU_THREADS", "8")),
                cuda_graph_bs=[],
                cuda_graph_max_bs=0,
                cache_type="naive",
                ple_artifact_path=str(artifact_path),
                ple_backend=ple_backend,
                ple_warm_mode="cold",
                ple_planner_mode="vectorized",
            )
            startup_seconds = time.monotonic() - startup_started
        except Exception as error:
            pytest.fail(
                "Qwen GGUF cache-zero Engine construction failed; no fallback was attempted: "
                f"{type(error).__name__}: {error}"
            )

        engine = llm.engine
        p4_identity = _assert_p4(torch, engine.device)
        assert engine.config.tp_info.size == 1
        assert engine.config.max_running_req == 1
        assert engine.config.moe_backend == "cpu"
        assert engine.config.moe_cache_size == 0
        assert engine.config.cuda_graph_bs == []
        assert engine.config.cuda_graph_max_bs == 0
        assert engine.config.max_extend_tokens == 8
        assert engine.moe_offload_cache is None
        assert engine.graph_runner.graph_bs_list == []

        try:
            generation_started = time.monotonic()
            result = llm.generate(
                ["Write one short greeting."],
                SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
            )
            generation_seconds = time.monotonic() - generation_started
        except Exception as error:
            pytest.fail(
                "Qwen GGUF cache-zero prefill+decode failed; no fallback was attempted: "
                f"{type(error).__name__}: {error}"
            )
        assert len(result) == 1
        assert len(result[0]["token_ids"]) == 2

        ple_telemetry = engine.model.host_weight_telemetry()
        expert_telemetry = engine.gguf_cpu_expert_telemetry()
        assert ple_telemetry
        assert expert_telemetry
        assert {str(item.get("expert_source")) for item in expert_telemetry.values()} == {
            "gguf-host-mmap"
        }
        assert {str(item.get("expert_execution_device")) for item in expert_telemetry.values()} == {
            "cpu"
        }
        ple_backends = {str(item.get("backend")) for item in ple_telemetry.values()}
        ple_sources = {str(item.get("source_kind")) for item in ple_telemetry.values()}
        assert ple_backends == {ple_backend}
        assert ple_sources == {"dedicated-artifact"}
        expected_mapped_bytes = ple_identity["tensor_bytes"] if ple_backend == "mmap" else 0
        assert {int(item["mapped_bytes"]) for item in ple_telemetry.values()} == {
            expected_mapped_bytes
        }

        execution = [
            item.get("execution_telemetry")
            for item in expert_telemetry.values()
            if item.get("execution_telemetry") is not None
        ]
        assert execution, "CPU expert execution telemetry was not emitted"
        expected_cpu_threads = int(os.environ.get("FREETOKEN_PASCAL_CPU_THREADS", "8"))
        assert len(expert_telemetry) == 48
        direct_avx2_backends = {"q4_k_avx2", "mixed_gemv_avx2", "mixed_avx2"}
        assert all(str(item["backend"]) in direct_avx2_backends for item in execution)
        assert all(
            item["kernel_census"]
            and all(str(kernel).endswith("_avx2") for kernel in item["kernel_census"])
            for item in execution
        )
        assert all(int(item["thread_count"]) == expected_cpu_threads for item in execution)
        for layer_telemetry in expert_telemetry.values():
            affinity = layer_telemetry["affinity_telemetry"]
            assert affinity["affinity_status"] == "verified"
            assert affinity["verification_status"] == "verified"
            assert affinity["affinity_verified"] is True
            assert affinity["worker_affinity_errors"] == []
        assert all(str(item["backend"]).lower().find("ssd") < 0 for item in execution)
        assert all(str(item["backend"]).lower().find("gpu") < 0 for item in execution)
        assert any(int(item["routes_executed"]) > 0 for item in execution)
        assert any(int(item["bytes_read_packed"]) > 0 for item in execution)
        memory = [item["memory"] for item in expert_telemetry.values()]
        assert all(int(item["expert_mapped_bytes"]) > 0 for item in memory)
        assert all(int(item["anonymous_host_source_bytes"]) == 0 for item in memory)
        assert all(int(item["pinned_host_source_bytes"]) == 0 for item in memory)

        evidence = {
            "schema_name": "qwen38-gguf-cache-zero-h2-evidence.schema.json",
            "schema_version": 1,
            "evidence_status": "measured",
            "repository_commit": _repository_commit(),
            "model": model_identity,
            "ple_artifact": ple_identity,
            "hardware": p4_identity,
            "config": {
                "tp_size": engine.config.tp_info.size,
                "max_running_req": engine.config.max_running_req,
                "max_extend_tokens": engine.config.max_extend_tokens,
                "max_seq_len": engine.config.max_seq_len,
                "moe_backend": engine.config.moe_backend,
                "moe_cpu_threads": engine.config.moe_cpu_threads,
                "moe_cache_size": engine.config.moe_cache_size,
                "cuda_graph_bs": engine.config.cuda_graph_bs,
                "cuda_graph_max_bs": engine.config.cuda_graph_max_bs,
                "offload_moe_cache": False,
                "cache_type": engine.config.cache_type,
                "ple_backend": ple_backend,
            },
            "expert_quant_census": dict(census),
            "expert_source": {
                "kind": next(iter(expert_telemetry.values()))["expert_source"],
                "execution_device": next(iter(expert_telemetry.values()))[
                    "expert_execution_device"
                ],
                "ssd_execution": False,
                "memory": memory[0],
            },
            "ple_telemetry": ple_telemetry,
            "expert_telemetry": expert_telemetry,
            "generation": {
                "prompt": "Write one short greeting.",
                "output_token_count": len(result[0]["token_ids"]),
                "output_token_ids": result[0]["token_ids"],
                "elapsed_seconds": generation_seconds,
                "observed_tokens_per_second": len(result[0]["token_ids"]) / generation_seconds,
            },
            "startup_seconds": startup_seconds,
        }
        _write_evidence(evidence)
    finally:
        if llm is not None:
            llm.shutdown()
