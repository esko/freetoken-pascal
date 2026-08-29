from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from freetoken.moe import real_artifact_probe
from freetoken.moe.real_artifact_probe import ArtifactProbeError, RangeResponse

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "manifests/qwen38-gguf.json"
CENSUS = ROOT / "tests/fixtures/results/qwen38-q3-census.metadata.json"
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_evidence", ROOT / "scripts" / "validate_evidence.py"
)
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
VALIDATE_EVIDENCE = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(VALIDATE_EVIDENCE)
CLI_SPEC = importlib.util.spec_from_file_location(
    "probe_qwen38_q3_triad_cli", ROOT / "scripts" / "probe_qwen38_q3_triad.py"
)
assert CLI_SPEC and CLI_SPEC.loader
Q3_CLI = importlib.util.module_from_spec(CLI_SPEC)
CLI_SPEC.loader.exec_module(Q3_CLI)
_TEST_COMMIT = "a" * 40


@pytest.fixture(autouse=True)
def _stable_evidence_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep probe tests hermetic in pinned containers that omit Git."""
    monkeypatch.setattr(real_artifact_probe, "_probe_commit", lambda: _TEST_COMMIT)
    real_run = VALIDATE_EVIDENCE.subprocess.run

    def _run(command, *args, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, stdout=f"{_TEST_COMMIT}\n", stderr="")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(VALIDATE_EVIDENCE.subprocess, "run", _run)


class _TriadTransport:
    def __init__(
        self, expected: dict[tuple[int, int], int], *, total_by_url: dict[str, int]
    ) -> None:
        self.expected = expected
        self.total_by_url = total_by_url
        self.requests: list[tuple[str, int, int]] = []
        self.mode = "ok"

    def fetch(self, url: str, start: int, end: int) -> RangeResponse:
        self.requests.append((url, start, end))
        size = self.expected[(start, end)]
        body = bytes((start + index) % 251 for index in range(size))
        if self.mode == "short":
            body = body[:-1]
        response_start = start + 1 if self.mode == "wrong-range" else start
        total = self.total_by_url[url]
        return RangeResponse(
            status=206,
            headers={
                "Content-Range": f"bytes {response_start}-{end}/{total}",
                "Content-Length": str(len(body)),
            },
            body=body,
        )


def _range_transport() -> tuple[_TriadTransport, list[tuple[int, int, int, str]]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    shards = {item["name"]: item for item in manifest["variants"]["UD-Q3_K_XL"]["shards"]}
    expected: dict[tuple[int, int], int] = {}
    description: list[tuple[int, int, int, str]] = []
    for layer, expert in real_artifact_probe.DEFAULT_Q3_TRIAD:
        for projection in ("gate", "up", "down"):
            record = next(
                item
                for item in census["tensors"]
                if item["name"] == f"blk.{layer}.ffn_{projection}_exps.weight"
            )
            size = int(record["shape"][1]) * int(record["row_bytes"])
            start = int(record["offset"]) + expert * size
            end = start + size - 1
            shard = census["shards"][int(record["shard_index"])]
            expected[(start, end)] = size
            description.append((layer, start, end, shard["name"]))
    base_url = (
        "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/"
        "c8b5954a88c2775c546b92593eda40ea041d3176/UD-Q3_K_XL"
    )
    total_by_url = {f"{base_url}/{name}": int(item["size"]) for name, item in shards.items()}
    return _TriadTransport(expected, total_by_url=total_by_url), description


def _stub_probe_internals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(real_artifact_probe, "_load_gguf_oracle", lambda: object())

    def run_mode(layout, hidden, expert_ids, weights, *, mode, repeats, warmup):
        del expert_ids, weights
        output = np.zeros(
            (hidden.shape[0], layout.descriptor(layout.layers[0], "down").output_dim),
            dtype=np.float32,
        )
        kernel_census = ["reference_iq3_xxs", "reference_iq4_nl"]
        if any(item.quant_name == "Q8_0" for item in layout.descriptors):
            kernel_census.append("reference_q8_0")
        return output, {
            "requested_mode": mode,
            "repeats": repeats,
            "warmup": warmup,
            "selected_backend": "reference",
            "q4k_isa": "scalar",
            "q4k_fallback_reason": "reference_only_q3",
            "mixed_isa": "scalar",
            "mixed_fallback_reason": "reference_only_q3",
            "actual_avx2": False,
            "raw_elapsed_ns": [101],
            "telemetry": [
                {
                    "backend": "reference",
                    "kernel_census": kernel_census,
                    "fallback_reason": "reference_only_q3",
                }
            ],
        }

    monkeypatch.setattr(real_artifact_probe, "_run_mode", run_mode)

    def run_oracle(layout, sources, hidden, expert_ids, weights):
        del expert_ids, weights
        output = np.zeros(
            (hidden.shape[0], layout.descriptor(layout.layers[0], "down").output_dim),
            dtype=np.float32,
        )
        return output, {
            **real_artifact_probe.gguf_oracle_identity(),
            "packed_source_sha256": {
                name: hashlib.sha256(sources[name]).hexdigest() for name in ("gate", "up", "down")
            },
            "dense_projection_sha256": {name: "b" * 64 for name in ("gate", "up", "down")},
            "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
            "raw_elapsed_ns": {"dequantize": 102, "dense_expert": 103},
        }

    monkeypatch.setattr(real_artifact_probe, "_run_gguf_oracle", run_oracle)


def test_compare_outputs_emits_builtin_numeric_metrics() -> None:
    values = np.zeros((1, 4), dtype=np.float32)

    comparison = real_artifact_probe._compare_outputs(
        values,
        values,
        expected_name="expected",
        actual_name="actual",
    )

    assert all(
        type(comparison[name]) is float
        for name in (
            "max_abs_error",
            "relative_rms_error",
            "max_tolerance_violation",
            "rtol",
            "atol",
        )
    )


def test_q3_triad_uses_exact_inclusive_ranges_and_never_reads_full_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport, expected = _range_transport()
    _stub_probe_internals(monkeypatch)

    report = real_artifact_probe.probe_qwen38_q3_triad(
        manifest_path=MANIFEST,
        census_path=CENSUS,
        transport=transport,
        cache_dir=tmp_path,
        repeats=1,
        warmup=0,
        command="probe_qwen38_q3_triad --test",
        host={"hostname": "fake-host", "machine": "x86_64"},
    )

    assert [(start, end) for _layer, start, end, _name in expected] == [
        (start, end) for _url, start, end in transport.requests
    ]
    assert [_url.rsplit("/", 1)[-1] for _url, _start, _end in transport.requests] == [
        name for _layer, _start, _end, name in expected
    ]
    assert len(transport.requests) == 9
    assert report["fetch"]["range_count"] == 9
    assert report["fetch"]["full_shard_bytes"] == 0
    assert report["scope"] == {
        "reference_only": True,
        "performance_claim": False,
        "p4_claim": False,
        "full_shard_download": False,
    }
    assert [item["layer"] for item in report["probes"]] == [0, 23, 47]
    assert all(item["correctness"]["passed"] for item in report["probes"])
    assert all(item["kernel_census"] for item in report["probes"])
    assert VALIDATE_EVIDENCE.validate_document(report, schema_dir=ROOT / "schemas") == []


def test_q3_triad_rejects_duplicate_selection_before_any_range_request(
    tmp_path: Path,
) -> None:
    transport, _expected = _range_transport()
    with pytest.raises(ArtifactProbeError, match="unique"):
        real_artifact_probe.probe_qwen38_q3_triad(
            manifest_path=MANIFEST,
            census_path=CENSUS,
            probes=((0, 0), (0, 0), (47, 511)),
            transport=transport,
            cache_dir=tmp_path,
            command="probe_qwen38_q3_triad --test",
            host={"hostname": "fake-host"},
        )
    assert transport.requests == []


def test_q3_triad_propagates_short_range_failure_without_following_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport, _expected = _range_transport()
    transport.mode = "short"
    _stub_probe_internals(monkeypatch)
    with pytest.raises(ArtifactProbeError, match="length"):
        real_artifact_probe.probe_qwen38_q3_triad(
            manifest_path=MANIFEST,
            census_path=CENSUS,
            transport=transport,
            cache_dir=tmp_path,
            repeats=1,
            warmup=0,
            command="probe_qwen38_q3_triad --test",
            host={"hostname": "fake-host"},
        )
    assert len(transport.requests) == 1


def test_q3_triad_validator_rejects_tampered_range_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport, _expected = _range_transport()
    _stub_probe_internals(monkeypatch)
    report = real_artifact_probe.probe_qwen38_q3_triad(
        manifest_path=MANIFEST,
        census_path=CENSUS,
        transport=transport,
        cache_dir=tmp_path,
        repeats=1,
        warmup=0,
        command="probe_qwen38_q3_triad --test",
        host={"hostname": "fake-host"},
    )
    invalid = copy.deepcopy(report)
    invalid["fetch"]["ranges"][0]["end"] += 1
    errors = VALIDATE_EVIDENCE.validate_document(invalid, schema_dir=ROOT / "schemas")
    assert any("end" in error for error in errors)


def test_q3_triad_validator_binds_all_durable_evidence_to_raw_and_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport, _expected = _range_transport()
    _stub_probe_internals(monkeypatch)
    report = real_artifact_probe.probe_qwen38_q3_triad(
        manifest_path=MANIFEST,
        census_path=CENSUS,
        transport=transport,
        cache_dir=tmp_path,
        repeats=1,
        warmup=0,
        command="probe_qwen38_q3_triad --test",
        host={"hostname": "fake-host", "machine": "x86_64"},
    )
    cases = (
        (
            "aggregate range hash",
            lambda value: value["fetch"]["ranges"][0].update({"sha256": "0" * 64}),
            "fetch.ranges must equal",
        ),
        (
            "aggregate content range",
            lambda value: value["fetch"]["ranges"][0].update({"content_range": "bytes 0-1/2"}),
            "fetch.ranges must equal",
        ),
        (
            "raw range offset",
            lambda value: value["probes"][0]["raw"]["fetch"]["ranges"][0].update(
                {"artifact_offset": 1}
            ),
            "disagrees with raw probe range",
        ),
        (
            "raw content range",
            lambda value: value["probes"][0]["raw"]["fetch"]["ranges"][0].update(
                {"content_range": "bytes 0-1/2"}
            ),
            "disagrees with raw probe range",
        ),
        (
            "oracle packed range hash",
            lambda value: value["probes"][0]["raw"]["ab"]["oracle"]["packed_source_sha256"].update(
                {"gate": "0" * 64}
            ),
            "oracle packed bytes",
        ),
        (
            "range URL",
            lambda value: value["probes"][0]["ranges"][0].update({"url": "https://wrong"}),
            "derive from pinned source",
        ),
        (
            "raw transport",
            lambda value: value["probes"][0]["raw"]["fetch"].update({"transport": "full-download"}),
            "raw.fetch.transport",
        ),
        (
            "range cache counters",
            lambda value: value["probes"][0]["raw"]["fetch"]["ranges"][0].update({"cache": "hit"}),
            "cache_hits must match range cache states",
        ),
        (
            "census expert offset",
            lambda value: value["probes"][0]["ranges"][0].update(
                {"artifact_offset": value["probes"][0]["ranges"][0]["artifact_offset"] + 1}
            ),
            "census geometry",
        ),
        (
            "raw layout",
            lambda value: value["probes"][0]["raw"]["layout"]["descriptors"][0].update(
                {"artifact_offset": 1}
            ),
            "raw layout gate.artifact_offset",
        ),
        (
            "quant names",
            lambda value: value["probes"][0]["quant_names"].update({"gate": "Q4_K"}),
            "quant_names must match raw layout",
        ),
        (
            "workload hashes",
            lambda value: value["probes"][0]["raw"]["selection"].update(
                {"hidden_sha256": "0" * 64}
            ),
            "workload_hashes",
        ),
        (
            "promoted selection",
            lambda value: value["probes"][0]["raw"]["selection"].update({"promoted": True}),
            "promoted flag",
        ),
        (
            "output hashes",
            lambda value: value["probes"][0]["raw"]["ab"]["scalar"].update(
                {"output_sha256": "0" * 64}
            ),
            "output_hashes",
        ),
        (
            "correctness subclaim",
            lambda value: value["probes"][0]["correctness"]["oracle_vs_scalar"].update(
                {"correct": False}
            ),
            "correctness.oracle_vs_scalar",
        ),
        (
            "raw finite claim",
            lambda value: value["probes"][0]["raw"]["ab"].update({"finite": False}),
            "correctness.finite",
        ),
        (
            "oracle pin",
            lambda value: value["probes"][0]["raw"]["ab"]["oracle"].update({"version": "0.18.0"}),
            "oracle.version is not pinned",
        ),
        (
            "comparison output hash",
            lambda value: value["probes"][0]["raw"]["ab"]["oracle_vs_scalar"].update(
                {"expected_output_sha256": "0" * 64}
            ),
            "expected_output_sha256 is not bound",
        ),
        (
            "comparison numeric predicate",
            lambda value: value["probes"][0]["raw"]["ab"]["oracle_vs_scalar"].update(
                {"max_tolerance_violation": 1.0}
            ),
            "numeric predicate",
        ),
        (
            "comparison error",
            lambda value: value["probes"][0]["raw"]["ab"]["oracle_vs_scalar"].update(
                {"error": "tampered"}
            ),
            "must have a null error",
        ),
        (
            "raw repeats",
            lambda value: value["probes"][0]["raw"].update({"repeats": 2}),
            "raw.repeats",
        ),
        (
            "raw warmup",
            lambda value: value["probes"][0]["raw"].update({"warmup": 2}),
            "raw.warmup",
        ),
        (
            "raw schema version",
            lambda value: value["probes"][0]["raw"].update({"schema_version": 2}),
            "schema_version",
        ),
        (
            "raw evidence status",
            lambda value: value["probes"][0]["raw"].update({"evidence_status": "tampered"}),
            "evidence_status",
        ),
        (
            "raw range evidence",
            lambda value: value["probes"][0]["raw"].update({"range_evidence": "tampered"}),
            "range_evidence",
        ),
        (
            "raw validation class",
            lambda value: value["probes"][0]["raw"].update({"validation_class": "H2"}),
            "validation_class",
        ),
        (
            "timing samples",
            lambda value: value["probes"][0]["raw"]["ab"]["scalar"].update(
                {"raw_elapsed_ns": [999]}
            ),
            "timing.scalar_elapsed_ns",
        ),
        (
            "correctness tolerance",
            lambda value: value["probes"][0]["raw"]["ab"].update({"rtol": 0.25}),
            "correctness tolerances",
        ),
        (
            "selected backend",
            lambda value: value["probes"][0]["raw"]["ab"]["scalar"].update(
                {"selected_backend": "avx2"}
            ),
            "selected_backend",
        ),
        (
            "selected ISA",
            lambda value: value["probes"][0]["raw"]["ab"]["scalar"].update({"q4k_isa": "tampered"}),
            "ISA selection",
        ),
        (
            "selected fallback",
            lambda value: value["probes"][0]["raw"]["ab"]["scalar"]["telemetry"][0].update(
                {"fallback_reason": "tampered"}
            ),
            "telemetry fallback",
        ),
        (
            "telemetry backend",
            lambda value: value["probes"][0]["raw"]["ab"]["scalar"]["telemetry"][0].update(
                {"backend": "tampered"}
            ),
            "telemetry backends",
        ),
        (
            "actual AVX2",
            lambda value: value["probes"][0]["raw"]["ab"]["scalar"].update({"actual_avx2": True}),
            "actual_avx2",
        ),
        (
            "kernel census",
            lambda value: value["probes"][0]["raw"]["ab"]["scalar"]["telemetry"][0].update(
                {"kernel_census": ["tampered_kernel"]}
            ),
            "kernel_census_by_mode",
        ),
        (
            "aggregate timing",
            lambda value: value["probes"][0]["raw"]["ab"]["timing"].update(
                {"comparison_claim": True}
            ),
            "raw.ab.timing",
        ),
        (
            "aggregate error",
            lambda value: value["probes"][0]["raw"]["ab"].update({"max_abs_error": 1.0}),
            "raw.ab.max_abs_error",
        ),
        (
            "dense projection hashes",
            lambda value: value["probes"][0]["raw"].update(
                {
                    "oracle": {
                        **value["probes"][0]["raw"]["oracle"],
                        "dense_projection_sha256": {
                            **value["probes"][0]["raw"]["oracle"]["dense_projection_sha256"],
                            "gate": "0" * 64,
                        },
                    }
                }
            ),
            "dense projection hashes",
        ),
        (
            "layout top-k",
            lambda value: value["probes"][0]["raw"]["layout"].update({"top_k": 1}),
            "layout.top_k",
        ),
        (
            "layout selected expert",
            lambda value: value["probes"][0]["raw"]["layout"]["descriptors"][0].update(
                {"selected_expert": 1}
            ),
            "selected_expert",
        ),
        (
            "layout artifact end",
            lambda value: value["probes"][0]["raw"]["layout"]["descriptors"][0].update(
                {"artifact_end": 1}
            ),
            "artifact_end",
        ),
        (
            "layout remapped expert count",
            lambda value: value["probes"][0]["raw"]["layout"]["descriptors"][0].update(
                {"num_experts_remapped": 512}
            ),
            "num_experts_remapped",
        ),
        (
            "layout rows per expert",
            lambda value: value["probes"][0]["raw"]["layout"]["descriptors"][0].update(
                {"rows_per_expert": 1}
            ),
            "rows_per_expert",
        ),
        (
            "manifest hash",
            lambda value: value["metadata"]["artifact"].update({"manifest_sha256": "0" * 64}),
            "manifest_sha256",
        ),
        (
            "census hash",
            lambda value: value["metadata"]["census"].update({"sha256": "0" * 64}),
            "census.sha256",
        ),
        (
            "census count",
            lambda value: value["metadata"]["census"].update(
                {"tensor_count": value["metadata"]["census"]["tensor_count"] + 1}
            ),
            "tensor_count",
        ),
        (
            "census total bytes",
            lambda value: value["metadata"]["census"].update(
                {"total_bytes": value["metadata"]["census"]["total_bytes"] + 1}
            ),
            "total_bytes",
        ),
        (
            "census quant count",
            lambda value: value["metadata"]["census"]["quant_type_counts"].update(
                {"IQ3_XXS": value["metadata"]["census"]["quant_type_counts"]["IQ3_XXS"] + 1}
            ),
            "quant_type_counts",
        ),
        (
            "census quant bytes",
            lambda value: value["metadata"]["census"]["quant_type_bytes"].update(
                {"IQ3_XXS": value["metadata"]["census"]["quant_type_bytes"]["IQ3_XXS"] + 1}
            ),
            "quant_type_bytes",
        ),
        (
            "census model hash",
            lambda value: value["metadata"]["census"].update({"model_sha256": "0" * 64}),
            "census.model_sha256",
        ),
        (
            "nested commit",
            lambda value: value["probes"][0]["raw"]["triad_metadata"].update({"commit": "b" * 40}),
            "raw.triad_metadata.commit",
        ),
        (
            "command consistency",
            lambda value: value["probes"][0]["raw"]["triad_metadata"].update(
                {"command": "tampered-command"}
            ),
            "raw.triad_metadata.command",
        ),
        (
            "host consistency",
            lambda value: value["probes"][0]["raw"]["triad_metadata"].update(
                {"host": {"hostname": "tampered-host"}}
            ),
            "raw.triad_metadata.host",
        ),
        (
            "checkpoint enabled state",
            lambda value: value["metadata"]["checkpoint"].update({"enabled": True}),
            "checkpoint.enabled",
        ),
        (
            "checkpoint resumed state",
            lambda value: value["metadata"]["checkpoint"].update({"resumed": True}),
            "checkpoint.resumed",
        ),
    )
    for label, mutate, needle in cases:
        invalid = copy.deepcopy(report)
        mutate(invalid)
        errors = VALIDATE_EVIDENCE.validate_document(invalid, schema_dir=ROOT / "schemas")
        assert any(needle in error for error in errors), (label, errors)


def test_q3_triad_checkpoint_resumes_completed_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport, _expected = _range_transport()
    _stub_probe_internals(monkeypatch)
    original = real_artifact_probe.probe_qwen38_expert
    calls = 0

    def interrupting_probe(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated supervisor interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(real_artifact_probe, "probe_qwen38_expert", interrupting_probe)
    checkpoint_dir = tmp_path / "checkpoint"
    with pytest.raises(RuntimeError, match="interruption"):
        real_artifact_probe.probe_qwen38_q3_triad(
            manifest_path=MANIFEST,
            census_path=CENSUS,
            transport=transport,
            cache_dir=tmp_path / "ranges",
            checkpoint_dir=checkpoint_dir,
            repeats=1,
            warmup=0,
            command="probe_qwen38_q3_triad --test",
            host={"hostname": "fake-host"},
        )
    checkpoint_path = checkpoint_dir / "qwen38-q3-triad.checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["status"] == "in-progress"
    assert [item["probe_id"] for item in checkpoint["completed"]] == ["layer-00-expert-000"]

    calls = 0

    def completing_probe(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(real_artifact_probe, "probe_qwen38_expert", completing_probe)
    report = real_artifact_probe.probe_qwen38_q3_triad(
        manifest_path=MANIFEST,
        census_path=CENSUS,
        transport=transport,
        cache_dir=tmp_path / "ranges",
        checkpoint_dir=checkpoint_dir,
        resume=True,
        repeats=1,
        warmup=0,
        command="probe_qwen38_q3_triad --test --resume --output alternate.json",
        host={"hostname": "fake-host"},
    )
    assert calls == 2
    assert report["metadata"]["checkpoint"]["resumed"] is True
    assert report["metadata"]["command"].endswith("alternate.json")
    assert len(report["metadata"]["audit"]["history"]) == 2
    assert len(transport.requests) == 9
    assert json.loads(checkpoint_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert VALIDATE_EVIDENCE.validate_document(report, schema_dir=ROOT / "schemas") == []


def test_q3_triad_watchdog_persists_completed_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport, _expected = _range_transport()
    _stub_probe_internals(monkeypatch)
    monotonic = iter((10.0, 12.0))
    monkeypatch.setattr(real_artifact_probe.time, "monotonic", lambda: next(monotonic))
    with pytest.raises(ArtifactProbeError, match="watchdog"):
        real_artifact_probe.probe_qwen38_q3_triad(
            manifest_path=MANIFEST,
            census_path=CENSUS,
            transport=transport,
            cache_dir=tmp_path / "ranges",
            checkpoint_dir=tmp_path / "checkpoint",
            watchdog_seconds=1.0,
            repeats=1,
            warmup=0,
            command="probe_qwen38_q3_triad --test",
            host={"hostname": "fake-host"},
        )
    checkpoint = json.loads(
        (tmp_path / "checkpoint/qwen38-q3-triad.checkpoint.json").read_text(encoding="utf-8")
    )
    assert [item["probe_id"] for item in checkpoint["completed"]] == ["layer-00-expert-000"]


@pytest.mark.parametrize("watchdog", [float("nan"), float("inf")])
def test_q3_triad_rejects_nonfinite_watchdog(watchdog: float) -> None:
    with pytest.raises(ArtifactProbeError, match="positive finite"):
        real_artifact_probe.probe_qwen38_q3_triad(
            manifest_path=MANIFEST,
            census_path=CENSUS,
            watchdog_seconds=watchdog,
        )


def test_q3_triad_rejects_corrupt_checkpoint_completed_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport, _expected = _range_transport()
    _stub_probe_internals(monkeypatch)
    original = real_artifact_probe.probe_qwen38_expert
    calls = 0

    def interrupting_probe(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated supervisor interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(real_artifact_probe, "probe_qwen38_expert", interrupting_probe)
    checkpoint_dir = tmp_path / "checkpoint"
    with pytest.raises(RuntimeError, match="interruption"):
        real_artifact_probe.probe_qwen38_q3_triad(
            manifest_path=MANIFEST,
            census_path=CENSUS,
            transport=transport,
            cache_dir=tmp_path / "ranges",
            checkpoint_dir=checkpoint_dir,
            repeats=1,
            warmup=0,
            command="probe_qwen38_q3_triad --test",
            host={"hostname": "fake-host"},
        )
    checkpoint_path = checkpoint_dir / "qwen38-q3-triad.checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_clean = copy.deepcopy(checkpoint)
    checkpoint["completed"][0]["ranges"][0]["length"] = "not-an-integer"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    checkpoint_orphan = checkpoint_dir / ".qwen38-q3-triad.checkpoint.json.orphan.tmp"
    checkpoint_orphan.write_text("orphan", encoding="utf-8")
    with pytest.raises(ArtifactProbeError, match="invalid probe ranges"):
        real_artifact_probe.probe_qwen38_q3_triad(
            manifest_path=MANIFEST,
            census_path=CENSUS,
            transport=transport,
            cache_dir=tmp_path / "ranges",
            checkpoint_dir=checkpoint_dir,
            resume=True,
            repeats=1,
            warmup=0,
            command="probe_qwen38_q3_triad --test --resume",
            host={"hostname": "fake-host"},
        )

    def mutate_invocation(value: dict[str, object]) -> None:
        completed = value["completed"][0]
        completed.update({"command": "tampered-command", "host": {"hostname": "tampered-host"}})
        completed["raw"]["triad_metadata"].update(
            {"command": "tampered-command", "host": {"hostname": "tampered-host"}}
        )
        completed["raw"]["fetch"].update(
            {"offline": True, "cache_dir": str(tmp_path / "tampered-cache")}
        )

    def mutate_audit_offline(value: dict[str, object]) -> None:
        value["audit"]["history"][0]["offline"] = "yes"
        value["audit"]["current"]["offline"] = "yes"

    def mutate_timing(value: dict[str, object]) -> None:
        completed = value["completed"][0]
        completed["raw"]["ab"]["scalar"]["raw_elapsed_ns"] = [float("nan")]
        completed["timing"]["scalar_elapsed_ns"] = [float("nan")]

    def mutate_hashes(value: dict[str, object]) -> None:
        completed = value["completed"][0]
        completed["raw"]["selection"]["hidden_sha256"] = "bad"
        completed["workload_hashes"]["hidden_sha256"] = "bad"
        completed["raw"]["ab"]["scalar"]["output_sha256"] = "bad"
        completed["output_hashes"]["scalar"] = "bad"

    def mutate_schema(value: dict[str, object]) -> None:
        value["completed"][0]["unexpected"] = True

    for mutate, match in (
        (
            lambda value: value["completed"][0]["ranges"][0].update(
                {"artifact_offset": value["completed"][0]["ranges"][0]["artifact_offset"] + 1}
            ),
            "invalid probe ranges",
        ),
        (
            lambda value: value["completed"][0].update({"correctness": {}}),
            "canonical raw evidence",
        ),
        (mutate_invocation, "audit history"),
        (mutate_audit_offline, "invocation types"),
        (mutate_timing, "positive integer samples"),
        (mutate_hashes, "missing (output|workload)"),
        (mutate_schema, "schema-invalid fields"),
    ):
        candidate = copy.deepcopy(checkpoint_clean)
        mutate(candidate)
        checkpoint_path.write_text(json.dumps(candidate), encoding="utf-8")
        with pytest.raises(ArtifactProbeError, match=match):
            real_artifact_probe.probe_qwen38_q3_triad(
                manifest_path=MANIFEST,
                census_path=CENSUS,
                transport=transport,
                cache_dir=tmp_path / "ranges",
                checkpoint_dir=checkpoint_dir,
                resume=True,
                repeats=1,
                warmup=0,
                command="probe_qwen38_q3_triad --test --resume",
                host={"hostname": "fake-host"},
            )
    assert not checkpoint_orphan.exists()

    checkpoint["completed"][0]["ranges"][0] = {}
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ArtifactProbeError, match="invalid probe ranges"):
        real_artifact_probe.probe_qwen38_q3_triad(
            manifest_path=MANIFEST,
            census_path=CENSUS,
            transport=transport,
            cache_dir=tmp_path / "ranges",
            checkpoint_dir=checkpoint_dir,
            resume=True,
            repeats=1,
            warmup=0,
            command="probe_qwen38_q3_triad --test --resume",
            host={"hostname": "fake-host"},
        )


def test_q3_triad_cli_resume_allows_output_and_offline_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport, _expected = _range_transport()
    _stub_probe_internals(monkeypatch)
    original_triad = Q3_CLI.probe_qwen38_q3_triad
    original_probe = real_artifact_probe.probe_qwen38_expert
    calls = 0

    def counting_probe(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_probe(*args, **kwargs)

    monkeypatch.setattr(real_artifact_probe, "probe_qwen38_expert", counting_probe)

    def run_with_transport(**kwargs):
        kwargs["transport"] = transport
        return original_triad(**kwargs)

    monkeypatch.setattr(Q3_CLI, "probe_qwen38_q3_triad", run_with_transport)
    checkpoint_dir = tmp_path / "checkpoint"
    cache_dir = tmp_path / "ranges"
    common = [
        "--manifest",
        str(MANIFEST),
        "--census",
        str(CENSUS),
        "--cache-dir",
        str(cache_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--repeats",
        "1",
        "--warmup",
        "0",
    ]
    first_output = tmp_path / "first.json"
    Q3_CLI.main([*common, "--output", str(first_output)])
    assert first_output.exists()
    assert calls == 3
    assert len(transport.requests) == 9

    calls = 0
    second_output = tmp_path / "second.json"
    second_cache_dir = tmp_path / "alternate-ranges"
    resume_args = list(common)
    cache_index = resume_args.index("--cache-dir") + 1
    resume_args[cache_index] = str(second_cache_dir)
    Q3_CLI.main(
        [
            *resume_args,
            "--resume",
            "--offline",
            "--output",
            str(second_output),
        ]
    )
    report = json.loads(second_output.read_text(encoding="utf-8"))
    assert calls == 0
    assert len(transport.requests) == 9
    assert report["metadata"]["checkpoint"]["resumed"] is True
    assert report["metadata"]["command"].endswith(f"--output {second_output}")
    assert report["metadata"]["audit"]["current"]["offline"] is True
    assert len(report["metadata"]["audit"]["history"]) == 2
    assert VALIDATE_EVIDENCE.validate_document(report, schema_dir=ROOT / "schemas") == []
