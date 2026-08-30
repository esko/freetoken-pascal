from __future__ import annotations

import json

import pytest
from freetoken.ple_io_evidence import (
    PLEIOCounters,
    PLEIOEvidenceError,
    PLEIOEvidenceRecorder,
    PLEIOPhaseEvidence,
    measure_ple_io_phase,
)


def counters(**changes: object) -> PLEIOCounters:
    values: dict[str, object] = {
        "major_faults": 10,
        "logical_packed_bytes": 1_000,
        "application_bytes_read": 1_200,
        "application_reads": 4,
        "physical_block_device_bytes": 2_000,
        "device_identity": "nvme0n1",
        "physical_source": "block-device-stat",
    }
    values.update(changes)
    return PLEIOCounters(**values)


def test_measure_phase_reports_counter_deltas_and_read_amplification() -> None:
    evidence = measure_ple_io_phase(
        "cold",
        counters(),
        counters(
            major_faults=17,
            logical_packed_bytes=1_500,
            application_bytes_read=1_800,
            application_reads=7,
            physical_block_device_bytes=3_500,
        ),
    )

    assert evidence.phase == "cold"
    assert evidence.major_faults == 7
    assert evidence.logical_packed_bytes == 500
    assert evidence.application_bytes_read == 600
    assert evidence.application_reads == 3
    assert evidence.physical_block_device_bytes == 1_500
    assert evidence.read_amplification == pytest.approx(3.0)
    assert evidence.physical_evidence is True
    assert evidence.to_dict()["phase"] == "cold"
    json.dumps(evidence.to_dict(), allow_nan=False)


@pytest.mark.parametrize("phase", ["warm", "steady"])
def test_all_cache_temperature_phases_are_explicit(phase: str) -> None:
    evidence = measure_ple_io_phase(phase, counters(), counters(logical_packed_bytes=1_100))

    assert evidence.phase == phase


def test_injected_counter_recorder_captures_phase_lifecycle() -> None:
    snapshots = iter(
        [
            counters(),
            counters(logical_packed_bytes=1_500, physical_block_device_bytes=2_500),
        ]
    )
    recorder = PLEIOEvidenceRecorder(lambda: next(snapshots))

    recorder.begin_phase("cold")
    evidence = recorder.end_phase("cold")

    assert evidence.application_bytes_read == 0
    assert evidence.physical_block_device_bytes == 500
    assert recorder.completed == (evidence,)


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        (counters(logical_packed_bytes=2_000), counters(), "counter regression"),
        (counters(device_identity="nvme0n1"), counters(device_identity="nvme1n1"), "identity"),
        (counters(device_identity=("nvme0n1", "nvme1n1")), counters(), "ambiguous"),
        (
            counters(physical_block_device_bytes=None),
            counters(logical_packed_bytes=1_100, physical_block_device_bytes=None),
            "physical",
        ),
        (counters(physical_source="process-io"), counters(physical_source="process-io"), "process"),
        (
            counters(physical_source="opaque-counter"),
            counters(physical_source="opaque-counter"),
            "identified",
        ),
        (
            counters(physical_source="not-a-block-device-stat"),
            counters(physical_source="not-a-block-device-stat"),
            "identified",
        ),
        (
            counters(physical_source="block-device-stat "),
            counters(physical_source="block-device-stat "),
            "identified",
        ),
        (
            counters(physical_source="BLOCK-DEVICE-STAT"),
            counters(physical_source="BLOCK-DEVICE-STAT"),
            "identified",
        ),
    ],
)
def test_evidence_fails_closed_for_invalid_counter_provenance(
    before: PLEIOCounters,
    after: PLEIOCounters,
    message: str,
) -> None:
    with pytest.raises(PLEIOEvidenceError, match=message):
        measure_ple_io_phase("cold", before, after)


@pytest.mark.parametrize("phase", ["", "hot", "COLD", None])
def test_invalid_phase_fails_closed(phase: str | None) -> None:
    with pytest.raises(PLEIOEvidenceError, match="phase"):
        measure_ple_io_phase(phase, counters(), counters(logical_packed_bytes=1_100))


def test_zero_logical_bytes_cannot_produce_read_amplification() -> None:
    with pytest.raises(PLEIOEvidenceError, match="denominator"):
        measure_ple_io_phase(
            "steady",
            counters(logical_packed_bytes=4_000),
            counters(logical_packed_bytes=4_000),
        )


def test_process_telemetry_mapping_never_becomes_physical_evidence() -> None:
    before = {
        "major_faults": 2,
        "packed_bytes_read": 1_000,
        "application_bytes_read": 1_000,
        "application_reads": 1,
        "storage_read_bytes": 9_000,
    }
    after = {**before, "major_faults": 3, "packed_bytes_read": 1_100}

    with pytest.raises(PLEIOEvidenceError, match="physical"):
        measure_ple_io_phase("warm", before, after)


def test_recorder_requires_matching_active_phase() -> None:
    recorder = PLEIOEvidenceRecorder(lambda: counters())

    with pytest.raises(PLEIOEvidenceError, match="active phase"):
        recorder.end_phase("cold")

    recorder.begin_phase("cold")
    with pytest.raises(PLEIOEvidenceError, match="already active"):
        recorder.begin_phase("warm")
    with pytest.raises(PLEIOEvidenceError, match="does not match"):
        recorder.end_phase("warm")


def _evidence(**changes: object) -> PLEIOPhaseEvidence:
    values: dict[str, object] = {
        "phase": "cold",
        "major_faults": 1,
        "logical_packed_bytes": 100,
        "application_bytes_read": 200,
        "application_reads": 2,
        "physical_block_device_bytes": 300,
        "read_amplification": 3.0,
        "device_identity": "nvme0n1",
        "physical_source": "block-device-stat",
        "physical_source_detail": "/sys/block/nvme0n1/stat",
        "physical_evidence": True,
    }
    values.update(changes)
    return PLEIOPhaseEvidence(**values)


@pytest.mark.parametrize(
    "field",
    [
        "major_faults",
        "logical_packed_bytes",
        "application_bytes_read",
        "application_reads",
        "physical_block_device_bytes",
    ],
)
@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_phase_evidence_rejects_non_negative_integer_contract_violations(
    field: str, value: object
) -> None:
    with pytest.raises(PLEIOEvidenceError, match=field):
        _evidence(**{field: value})


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), float("-inf"), True, "3"])
def test_phase_evidence_rejects_invalid_amplification(value: object) -> None:
    with pytest.raises(PLEIOEvidenceError, match="read_amplification"):
        _evidence(read_amplification=value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("phase", "hot"),
        ("device_identity", ""),
        ("device_identity", None),
        ("physical_source", ""),
        ("physical_source", "trace-block-device-stat"),
        ("physical_source_detail", ""),
        ("physical_evidence", False),
        ("physical_evidence", 1),
    ],
)
def test_phase_evidence_rejects_invalid_phase_and_provenance(field: str, value: object) -> None:
    with pytest.raises(PLEIOEvidenceError):
        _evidence(**{field: value})


def test_recorder_begin_provider_failure_does_not_activate_phase() -> None:
    attempts = 0

    def flaky_source() -> PLEIOCounters:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("probe unavailable")
        return counters()

    recorder = PLEIOEvidenceRecorder(flaky_source)
    with pytest.raises(RuntimeError, match="probe unavailable"):
        recorder.begin_phase("cold")

    recorder.begin_phase("cold")
    assert recorder.completed == ()


def test_recorder_end_failure_keeps_phase_for_retry() -> None:
    recorder = PLEIOEvidenceRecorder(lambda: counters())
    recorder.begin_phase("cold")

    with pytest.raises(PLEIOEvidenceError, match="counter regression"):
        recorder.end_phase("cold", counters(logical_packed_bytes=900))

    evidence = recorder.end_phase(
        "cold",
        counters(logical_packed_bytes=1_100, physical_block_device_bytes=2_100),
    )
    assert evidence.logical_packed_bytes == 100
    assert recorder.completed == (evidence,)


def test_recorder_end_provider_failure_keeps_phase_for_retry() -> None:
    attempts = 0

    def flaky_source() -> PLEIOCounters:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("probe unavailable")
        return counters()

    recorder = PLEIOEvidenceRecorder(flaky_source)
    recorder.begin_phase("cold")
    with pytest.raises(RuntimeError, match="probe unavailable"):
        recorder.end_phase("cold")

    evidence = recorder.end_phase(
        "cold",
        counters(logical_packed_bytes=1_100, physical_block_device_bytes=2_100),
    )
    assert evidence.logical_packed_bytes == 100
