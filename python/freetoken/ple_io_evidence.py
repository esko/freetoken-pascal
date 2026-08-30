"""Fail-closed evidence accounting for dedicated PLE storage runs.

The runtime's process telemetry can report page faults and application reads, but
it cannot establish how many bytes the block device supplied.  This module keeps
those counters separate and accepts physical bytes only from an explicitly
identified external block-device counter source.  It is intentionally independent
of :mod:`freetoken.gguf_host` so a normal lookup cannot accidentally turn process
``read_bytes`` into a physical-I/O claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

PLE_IO_PHASES = frozenset({"cold", "warm", "steady"})
_PROCESS_ONLY_SOURCES = frozenset(
    {
        "application",
        "procfs",
        "process",
        "process-io",
        "rusage",
    }
)
_PHYSICAL_SOURCE_MARKERS = frozenset(
    {"block", "device", "disk", "iostat", "blktrace", "nvme", "sysfs"}
)
_COUNTER_FIELDS = frozenset(
    {
        "major_faults",
        "logical_packed_bytes",
        "application_bytes_read",
        "application_reads",
        "physical_block_device_bytes",
        "device_identity",
        "physical_source",
    }
)


class PLEIOEvidenceError(ValueError):
    """Raised when a phase cannot produce trustworthy PLE I/O evidence."""


@dataclass(frozen=True, slots=True)
class PLEIOCounters:
    """Monotonic cumulative counters sampled at one instant.

    ``physical_block_device_bytes`` is optional at the snapshot boundary so a
    process-only source can be represented for diagnostics.  Measurement rejects
    such snapshots; callers must provide both the counter and its independently
    identified block-device source before physical bytes can be reported.
    ``device_identity`` may be a scalar device name or a sequence supplied by a
    multi-device probe.  A sequence is valid only when it contains exactly one
    non-empty identity, and is otherwise rejected as ambiguous.
    """

    major_faults: int
    logical_packed_bytes: int
    application_bytes_read: int
    application_reads: int
    physical_block_device_bytes: int | None = None
    device_identity: str | tuple[str, ...] | Sequence[str] | None = None
    physical_source: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "major_faults",
            "logical_packed_bytes",
            "application_bytes_read",
            "application_reads",
        ):
            _validate_counter(name, getattr(self, name))
        if self.physical_block_device_bytes is not None:
            _validate_counter(
                "physical_block_device_bytes",
                self.physical_block_device_bytes,
            )
        identity = self.device_identity
        if identity is not None and not isinstance(identity, str):
            try:
                identity = tuple(identity)
            except TypeError:
                # Keep the invalid value available for the measurement error,
                # rather than making a process-only diagnostic impossible to log.
                pass
            else:
                object.__setattr__(self, "device_identity", identity)


@dataclass(frozen=True, slots=True)
class PLEIOPhaseEvidence:
    """Counter deltas and physical/logical read amplification for one phase."""

    phase: str
    major_faults: int
    logical_packed_bytes: int
    application_bytes_read: int
    application_reads: int
    physical_block_device_bytes: int
    read_amplification: float
    device_identity: str
    physical_source: str
    physical_evidence: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible evidence fields without hiding provenance."""
        return {
            "phase": self.phase,
            "major_faults": self.major_faults,
            "logical_packed_bytes": self.logical_packed_bytes,
            "application_bytes_read": self.application_bytes_read,
            "application_reads": self.application_reads,
            "physical_block_device_bytes": self.physical_block_device_bytes,
            "read_amplification": self.read_amplification,
            "device_identity": self.device_identity,
            "physical_source": self.physical_source,
            "physical_evidence": self.physical_evidence,
        }


def _validate_counter(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PLEIOEvidenceError(f"{name} must be a non-negative integer")


def _validate_phase(phase: object) -> str:
    if not isinstance(phase, str) or phase not in PLE_IO_PHASES:
        raise PLEIOEvidenceError(f"invalid PLE I/O phase {phase!r}; expected cold, warm, or steady")
    return phase


def _coerce_counters(value: PLEIOCounters | Mapping[str, Any]) -> PLEIOCounters:
    if isinstance(value, PLEIOCounters):
        return value
    if isinstance(value, Mapping):
        try:
            # Existing MappedPLETable telemetry contains many unrelated fields.
            # Select only this contract's fields; in particular, never reinterpret
            # its process ``storage_read_bytes`` field as physical I/O.
            selected = {key: value[key] for key in _COUNTER_FIELDS if key in value}
            if "logical_packed_bytes" not in selected and "packed_bytes_read" in value:
                selected["logical_packed_bytes"] = value["packed_bytes_read"]
            return PLEIOCounters(**selected)
        except TypeError as error:
            raise PLEIOEvidenceError(f"invalid PLE I/O counter mapping: {error}") from error
    raise TypeError("PLE I/O counters must be PLEIOCounters or a mapping")


def _identity(value: object) -> str:
    if isinstance(value, str):
        if value.strip():
            return value
        raise PLEIOEvidenceError("physical device identity is ambiguous")
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        identities = tuple(value)
        if len(identities) != 1 or not isinstance(identities[0], str) or not identities[0].strip():
            raise PLEIOEvidenceError("physical device identity is ambiguous")
        return identities[0]
    raise PLEIOEvidenceError("physical device identity is ambiguous")


def _physical_provenance(before: PLEIOCounters, after: PLEIOCounters) -> tuple[str, str]:
    if before.physical_block_device_bytes is None or after.physical_block_device_bytes is None:
        raise PLEIOEvidenceError(
            "physical block-device bytes are undefined; process I/O alone is not evidence"
        )
    before_source = before.physical_source
    after_source = after.physical_source
    if not isinstance(before_source, str) or not before_source.strip():
        raise PLEIOEvidenceError("physical block-device counter source is undefined")
    if not isinstance(after_source, str) or not after_source.strip():
        raise PLEIOEvidenceError("physical block-device counter source is undefined")
    if (
        before_source.casefold() in _PROCESS_ONLY_SOURCES
        or after_source.casefold() in _PROCESS_ONLY_SOURCES
    ):
        raise PLEIOEvidenceError("process I/O alone cannot establish physical block-device bytes")
    if not any(marker in before_source.casefold() for marker in _PHYSICAL_SOURCE_MARKERS):
        raise PLEIOEvidenceError("physical counter source is not an identified block-device source")
    if not any(marker in after_source.casefold() for marker in _PHYSICAL_SOURCE_MARKERS):
        raise PLEIOEvidenceError("physical counter source is not an identified block-device source")
    if before_source != after_source:
        raise PLEIOEvidenceError("physical block-device counter source changed")
    before_identity = _identity(before.device_identity)
    after_identity = _identity(after.device_identity)
    if before_identity != after_identity:
        raise PLEIOEvidenceError("physical device identity changed")
    return before_identity, before_source


def _delta(name: str, before: int, after: int) -> int:
    value = after - before
    if value < 0:
        raise PLEIOEvidenceError(f"counter regression in {name}")
    return value


def measure_ple_io_phase(
    phase: str,
    before: PLEIOCounters | Mapping[str, Any],
    after: PLEIOCounters | Mapping[str, Any],
) -> PLEIOPhaseEvidence:
    """Compute fail-closed PLE counters for one explicit cache-temperature phase."""
    phase = _validate_phase(phase)
    start = _coerce_counters(before)
    end = _coerce_counters(after)
    device_identity, physical_source = _physical_provenance(start, end)
    logical = _delta("logical_packed_bytes", start.logical_packed_bytes, end.logical_packed_bytes)
    if logical == 0:
        raise PLEIOEvidenceError(
            "read-amplification denominator is zero or undefined (logical packed bytes)"
        )
    physical = _delta(
        "physical_block_device_bytes",
        start.physical_block_device_bytes,
        end.physical_block_device_bytes,
    )
    amplification = physical / logical
    if not isfinite(amplification):
        raise PLEIOEvidenceError("read-amplification denominator is undefined")
    return PLEIOPhaseEvidence(
        phase=phase,
        major_faults=_delta("major_faults", start.major_faults, end.major_faults),
        logical_packed_bytes=logical,
        application_bytes_read=_delta(
            "application_bytes_read",
            start.application_bytes_read,
            end.application_bytes_read,
        ),
        application_reads=_delta(
            "application_reads", start.application_reads, end.application_reads
        ),
        physical_block_device_bytes=physical,
        read_amplification=amplification,
        device_identity=device_identity,
        physical_source=physical_source,
    )


class PLEIOEvidenceRecorder:
    """Capture phase boundaries from an injected cumulative-counter provider."""

    def __init__(self, counter_source: Callable[[], PLEIOCounters | Mapping[str, Any]]) -> None:
        if not callable(counter_source):
            raise TypeError("counter_source must be callable")
        self._counter_source = counter_source
        self._active_phase: str | None = None
        self._start: PLEIOCounters | Mapping[str, Any] | None = None
        self._completed: list[PLEIOPhaseEvidence] = []

    @property
    def completed(self) -> tuple[PLEIOPhaseEvidence, ...]:
        return tuple(self._completed)

    def begin_phase(
        self,
        phase: str,
        counters: PLEIOCounters | Mapping[str, Any] | None = None,
    ) -> None:
        phase = _validate_phase(phase)
        if self._active_phase is not None:
            raise PLEIOEvidenceError(f"a phase is already active: {self._active_phase}")
        self._active_phase = phase
        self._start = self._sample(counters)

    def end_phase(
        self,
        phase: str,
        counters: PLEIOCounters | Mapping[str, Any] | None = None,
    ) -> PLEIOPhaseEvidence:
        phase = _validate_phase(phase)
        if self._active_phase is None or self._start is None:
            raise PLEIOEvidenceError("no active phase to end")
        if phase != self._active_phase:
            raise PLEIOEvidenceError(
                f"phase {phase!r} does not match active phase {self._active_phase!r}"
            )
        start = self._start
        end = self._sample(counters)
        self._active_phase = None
        self._start = None
        evidence = measure_ple_io_phase(phase, start, end)
        self._completed.append(evidence)
        return evidence

    # Lifecycle aliases make call sites read naturally without introducing a second API.
    start_phase = begin_phase
    finish_phase = end_phase

    def _sample(
        self,
        counters: PLEIOCounters | Mapping[str, Any] | None,
    ) -> PLEIOCounters | Mapping[str, Any]:
        if counters is not None:
            return counters
        return self._counter_source()


__all__ = [
    "PLE_IO_PHASES",
    "PLEIOCounters",
    "PLEIOEvidenceError",
    "PLEIOEvidenceRecorder",
    "PLEIOPhaseEvidence",
    "measure_ple_io_phase",
]
