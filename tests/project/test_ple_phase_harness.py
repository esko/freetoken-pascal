from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from freetoken.gguf_host import convert_gguf_ple_to_artifact
from freetoken.ple_io_evidence import (
    PLEIOCounters,
    PLEIOEvidenceError,
)
from freetoken.ple_phase_harness import PLEIOPhaseHarness, run_ple_io_phase_harness

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/gguf/qwen-host-layout.gguf"


def _physical_source() -> Callable[[], PLEIOCounters]:
    value = 10_000

    def sample() -> PLEIOCounters:
        nonlocal value
        value += 4_096
        return PLEIOCounters(
            0,
            0,
            0,
            0,
            physical_block_device_bytes=value,
            device_identity="test-nvme0n1",
            physical_source="block-device-stat",
            physical_source_detail="injected fixture counter",
        )

    return sample


def test_phase_harness_runs_identical_batches_through_both_backends(tmp_path: Path) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    batches = (
        np.array([31, 0, 31, 16], dtype=np.int64),
        np.array([16, 0], dtype=np.int64),
    )

    report = run_ple_io_phase_harness(
        artifact,
        row_batches=batches,
        physical_counter_source=_physical_source(),
        prefetch=True,
    )

    assert report["validation_class"] == "H0/no-P4"
    assert report["claim_status"] == "observation_only"
    assert report["physical_counter_status"] == "injected"
    assert report["row_batches"] == [[31, 0, 31, 16], [16, 0]]
    assert [item["backend"] for item in report["backends"]] == ["mmap", "pread"]
    for backend in report["backends"]:
        assert backend["source_kind"] == "dedicated-artifact"
        assert backend["artifact"]["payload_sha256"] == report["artifact"]["payload_sha256"]
        assert backend["codec"]["identity"]
        assert [item["phase"] for item in backend["phases"]] == ["cold", "warm", "steady"]
        assert len(backend["raw_phase_samples"]) == 3
        for phase, raw in zip(backend["phases"], backend["raw_phase_samples"], strict=True):
            assert phase["logical_rows"] == 6
            assert phase["unique_rows"] == 5
            assert phase["application_reads"] > 0
            assert phase["physical_block_device_bytes"] > 0
            assert phase["read_amplification"] > 0
            assert raw["before"]["counters"]["physical_source"] == "block-device-stat"
            assert raw["after"]["table_telemetry"]["backend"] == backend["backend"]
        assert backend["telemetry"]["prefetch_submitted"] == 1
        assert backend["telemetry"]["planner_calls"] == 6
        warm = backend["phases"][1]
        assert warm["telemetry_delta"]["prefetch_submitted"] == 0
        assert warm["telemetry_delta"]["prefetch_completed"] == 0
    json.dumps(report, allow_nan=False)


def test_phase_harness_can_use_explicit_object_api(tmp_path: Path) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    harness = PLEIOPhaseHarness(
        artifact,
        row_batches=[[0, 16, 31]],
        physical_counter_source=_physical_source(),
        prefetch=False,
    )

    report = harness.run()

    assert report["backends"][0]["phases"][0]["logical_rows"] == 3
    assert all(
        item["telemetry_delta"]["prefetch_submitted"] == 0
        for item in report["backends"][0]["phases"]
    )


def test_phase_harness_requires_explicit_physical_source(tmp_path: Path) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")

    with pytest.raises(PLEIOEvidenceError, match="physical"):
        run_ple_io_phase_harness(artifact, row_batches=[[0, 16, 31]])


def test_phase_harness_rejects_source_gguf_full_model_warm(tmp_path: Path) -> None:
    with pytest.raises(PLEIOEvidenceError, match=r"dedicated.*full-model-warm"):
        run_ple_io_phase_harness(
            FIXTURE,
            row_batches=[[0, 16, 31]],
            physical_counter_source=_physical_source(),
        )


def test_phase_runner_fails_closed_if_phase_record_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")

    class CorruptEvidence:
        phase = "corrupt"

    monkeypatch.setattr(
        "freetoken.ple_phase_harness.PLEIOEvidenceRecorder.end_phase",
        lambda *_args, **_kwargs: CorruptEvidence(),
    )

    with pytest.raises(PLEIOEvidenceError, match="phase"):
        run_ple_io_phase_harness(
            artifact,
            row_batches=[[0, 16, 31]],
            physical_counter_source=_physical_source(),
        )


def test_phase_harness_does_not_promote_storage_read_bytes(tmp_path: Path) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")

    def process_only() -> dict[str, object]:
        return {"storage_read_bytes": 1 << 30}

    with pytest.raises(PLEIOEvidenceError, match="physical"):
        run_ple_io_phase_harness(
            artifact,
            row_batches=[[0, 16, 31]],
            physical_counter_source=process_only,
        )
