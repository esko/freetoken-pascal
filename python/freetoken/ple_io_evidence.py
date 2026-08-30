"""Fail-closed evidence accounting for dedicated PLE storage runs.

The runtime's process telemetry can report page faults and application reads, but
it cannot establish how many bytes the block device supplied.  This module keeps
those counters separate and accepts physical bytes only from an explicitly
identified external block-device counter source.  It is intentionally independent
of :mod:`freetoken.gguf_host` so a normal lookup cannot accidentally turn process
``read_bytes`` into a physical-I/O claim.
"""

from __future__ import annotations

import os
import platform as platform_module
import stat as stat_module
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any

PLE_IO_PHASES = frozenset({"cold", "warm", "steady"})
# Linux documents /sys/block/<dev>/stat sector counters in 512-byte units,
# independently of the device's logical block size.
LINUX_SYSFS_STAT_SECTOR_BYTES = 512
PLE_IO_PHYSICAL_SOURCE_KINDS = frozenset(
    {
        "blktrace",
        "block-device-stat",
        "iostat",
        "nvme-cli",
        "sysfs-block-stat",
    }
)
_PROCESS_ONLY_SOURCES = frozenset(
    {
        "application",
        "procfs",
        "process",
        "process-io",
        "rusage",
    }
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
        "physical_source_detail",
    }
)


class PLEIOEvidenceError(ValueError):
    """Raised when a phase cannot produce trustworthy PLE I/O evidence."""


class PLEBlockProbeError(PLEIOEvidenceError):
    """Raised when a dedicated PLE payload cannot map to one block device."""


@dataclass(frozen=True, slots=True)
class PLEBlockDevice:
    """One unambiguous terminal block device selected for PLE evidence."""

    name: str
    major: int
    minor: int
    sysfs_path: str
    stat_path: str


def _read_sysfs_text(path: str | os.PathLike[str]) -> str:
    try:
        return Path(path).read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise PLEBlockProbeError(f"cannot read Linux sysfs path {path}: {error}") from error


def _parse_device_number(value: object, *, label: str) -> tuple[int, int]:
    if not isinstance(value, str):
        raise PLEBlockProbeError(f"{label} must contain one major:minor device number")
    fields = value.strip().split(":")
    if len(fields) != 2:
        raise PLEBlockProbeError(f"{label} must contain one major:minor device number")
    try:
        major, minor = (int(field, 10) for field in fields)
    except ValueError as error:
        raise PLEBlockProbeError(f"{label} has malformed major:minor device number") from error
    if major < 0 or minor < 0:
        raise PLEBlockProbeError(f"{label} has a negative major:minor device number")
    return major, minor


class LinuxPLEBlockCounterProbe:
    """Sample physical PLE bytes from one Linux sysfs block-device counter.

    The payload path is deliberately explicit: its ``st_dev`` is mapped through
    ``sysfs_root/dev/block/<major>:<minor>`` and then reduced through a single
    ``slaves`` entry when the mapping is not already a partition.  A partition is
    a terminal device: its own stat is the correct filesystem-partition counter,
    and it must not be replaced with its whole-disk parent.  A multiple-slave
    graph is rejected because combining devices would make read amplification
    unreviewable.  The probe does not inspect process I/O and does not mutate page
    caches.

    ``counter_source`` can provide the non-physical fields required by
    :class:`PLEIOCounters` (for example, a separately collected process snapshot).
    When omitted, those fields are zero solely to make the physical probe usable as
    a standalone injected source; zero is not a process-I/O measurement.
    """

    def __init__(
        self,
        payload_path: str | os.PathLike[str],
        *,
        sysfs_root: str | os.PathLike[str] = "/sys",
        counter_source: Callable[[], PLEIOCounters | Mapping[str, Any]] | None = None,
        stat_fn: Callable[[str | os.PathLike[str]], object] | None = None,
        read_text_fn: Callable[[str | os.PathLike[str]], str] | None = None,
        system: str | None = None,
    ) -> None:
        selected_system = platform_module.system() if system is None else system
        if selected_system.lower() != "linux":
            raise PLEBlockProbeError(
                f"Linux sysfs block probing is unsupported on {selected_system!r}"
            )
        if counter_source is not None and not callable(counter_source):
            raise TypeError("counter_source must be callable")
        if stat_fn is not None and not callable(stat_fn):
            raise TypeError("stat_fn must be callable")
        if read_text_fn is not None and not callable(read_text_fn):
            raise TypeError("read_text_fn must be callable")
        self.payload_path = Path(payload_path)
        self.sysfs_root = Path(sysfs_root)
        self.counter_source = counter_source
        self._stat_fn = os.stat if stat_fn is None else stat_fn
        self._read_text_fn = _read_sysfs_text if read_text_fn is None else read_text_fn

    def _payload_device_number(self) -> tuple[int, int]:
        try:
            payload_stat = self._stat_fn(self.payload_path)
        except (OSError, ValueError, TypeError) as error:
            raise PLEBlockProbeError(
                f"cannot stat dedicated PLE payload {self.payload_path}: {error}"
            ) from error
        mode = getattr(payload_stat, "st_mode", None)
        if mode is not None and not stat_module.S_ISREG(mode):
            raise PLEBlockProbeError(
                f"dedicated PLE payload is not a regular file: {self.payload_path}"
            )
        device = getattr(payload_stat, "st_dev", None)
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise PLEBlockProbeError("dedicated PLE payload has no valid st_dev")
        try:
            major, minor = os.major(device), os.minor(device)
        except (OverflowError, ValueError) as error:
            raise PLEBlockProbeError("dedicated PLE payload st_dev is malformed") from error
        if major == 0:
            raise PLEBlockProbeError(
                "dedicated PLE payload is on a non-block filesystem (major 0; overlay/tmpfs?)"
            )
        return major, minor

    def _read_text(self, path: Path) -> str:
        try:
            value = self._read_text_fn(path)
        except PLEBlockProbeError:
            raise
        except (OSError, UnicodeError, ValueError, TypeError) as error:
            raise PLEBlockProbeError(f"cannot read Linux sysfs path {path}: {error}") from error
        if not isinstance(value, str):
            raise PLEBlockProbeError(f"Linux sysfs path {path} did not return text")
        return value

    def _validate_block_path(
        self,
        path: Path,
        *,
        expected_device: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        if not path.is_dir():
            raise PLEBlockProbeError(f"resolved sysfs block path is missing: {path}")
        dev_file = path / "dev"
        if not dev_file.is_file():
            raise PLEBlockProbeError(f"resolved sysfs path is not a block device: {path}")
        actual = _parse_device_number(self._read_text(dev_file), label=str(dev_file))
        if expected_device is not None and actual != expected_device:
            raise PLEBlockProbeError(
                f"sysfs dev mapping changed: expected {expected_device[0]}:{expected_device[1]}, "
                f"found {actual[0]}:{actual[1]}"
            )
        return actual

    def _resolve_target(self, major: int, minor: int) -> Path:
        try:
            root = self.sysfs_root.resolve(strict=True)
            dev_block = root / "dev" / "block"
            if not dev_block.is_dir():
                raise PLEBlockProbeError(f"Linux sysfs block mapping is missing: {dev_block}")
            link = dev_block / f"{major}:{minor}"
            target = link.resolve(strict=True)
        except PLEBlockProbeError:
            raise
        except (OSError, RuntimeError) as error:
            raise PLEBlockProbeError(
                f"cannot resolve Linux sysfs block mapping {major}:{minor}: {error}"
            ) from error
        try:
            target.relative_to(root)
        except ValueError as error:
            raise PLEBlockProbeError(
                f"Linux sysfs block mapping {major}:{minor} escapes {root}"
            ) from error
        self._validate_block_path(target, expected_device=(major, minor))
        return target

    def _next_block_path(self, path: Path) -> Path | None:
        partition = path / "partition"
        if partition.exists() or partition.is_symlink():
            text = self._read_text(partition).strip()
            try:
                partition_number = int(text, 10)
            except ValueError as error:
                raise PLEBlockProbeError(f"malformed partition marker: {partition}") from error
            if partition_number <= 0:
                raise PLEBlockProbeError(f"invalid partition marker: {partition}")
            # /sys/block/<dev>/stat is already scoped to this partition.  Walking
            # to the parent disk would combine unrelated filesystem partitions.
            return None

        slaves = path / "slaves"
        if not (slaves.exists() or slaves.is_symlink()):
            if "virtual" in path.parts:
                raise PLEBlockProbeError(
                    f"virtual block device {path.name} has no discoverable physical backing"
                )
            return None
        if not slaves.is_dir():
            raise PLEBlockProbeError(f"Linux sysfs slaves path is not a directory: {slaves}")
        try:
            entries = tuple(sorted(slaves.iterdir(), key=lambda item: item.name))
        except OSError as error:
            raise PLEBlockProbeError(
                f"cannot enumerate Linux sysfs slaves {slaves}: {error}"
            ) from error
        if len(entries) > 1:
            names = ", ".join(entry.name for entry in entries)
            raise PLEBlockProbeError(
                f"ambiguous Linux sysfs block backing for {path.name}: {names}"
            )
        if not entries:
            if "virtual" in path.parts:
                raise PLEBlockProbeError(
                    f"virtual block device {path.name} has no discoverable physical backing"
                )
            return None
        try:
            target = entries[0].resolve(strict=True)
        except RuntimeError as error:
            raise PLEBlockProbeError(
                f"cycle in Linux sysfs slave target {entries[0]}"
            ) from error
        except OSError as error:
            raise PLEBlockProbeError(
                f"missing Linux sysfs slave target {entries[0]}"
            ) from error
        try:
            target.relative_to(self.sysfs_root.resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as error:
            raise PLEBlockProbeError(
                f"Linux sysfs slave target is outside the sysfs root: {entries[0]}"
            ) from error
        self._validate_block_path(target)
        return target

    def resolve(self) -> PLEBlockDevice:
        """Resolve the payload's unique terminal block device."""
        major, minor = self._payload_device_number()
        current = self._resolve_target(major, minor)
        visited: set[Path] = set()
        terminal_device = (major, minor)
        while True:
            if current in visited:
                raise PLEBlockProbeError(f"cycle in Linux sysfs block backing at {current}")
            visited.add(current)
            terminal_device = self._validate_block_path(current)
            next_path = self._next_block_path(current)
            if next_path is None:
                break
            current = next_path

        stat_path = current / "stat"
        if not stat_path.is_file():
            raise PLEBlockProbeError(f"Linux block stat is missing: {stat_path}")
        return PLEBlockDevice(
            name=current.name,
            major=terminal_device[0],
            minor=terminal_device[1],
            sysfs_path=str(current),
            stat_path=str(stat_path),
        )

    def _read_sectors(self, device: PLEBlockDevice) -> int:
        fields = self._read_text(Path(device.stat_path)).strip().split()
        if len(fields) < 3:
            raise PLEBlockProbeError(
                f"Linux block stat {device.stat_path} has no documented sectors-read field 3"
            )
        try:
            values = [int(field, 10) for field in fields]
        except ValueError as error:
            raise PLEBlockProbeError(f"malformed Linux block stat: {device.stat_path}") from error
        if any(value < 0 for value in values):
            raise PLEBlockProbeError(f"negative Linux block stat counter: {device.stat_path}")
        # Linux /sys/block/<dev>/stat field 3 is sectors read (1-based), i.e. index 2.
        return values[2]

    def sample(
        self,
        counters: PLEIOCounters | Mapping[str, Any] | None = None,
    ) -> PLEIOCounters:
        """Return a PLE snapshot with physical bytes and explicit sysfs provenance."""
        if counters is None:
            counters = (
                self.counter_source()
                if self.counter_source is not None
                else PLEIOCounters(0, 0, 0, 0)
            )
        base = _coerce_counters(counters)
        device = self.resolve()
        physical_bytes = self._read_sectors(device) * LINUX_SYSFS_STAT_SECTOR_BYTES
        detail = (
            f"{device.major}:{device.minor}/{device.name}:"
            f"field=3(sectors-read):sector-size={LINUX_SYSFS_STAT_SECTOR_BYTES}"
        )
        return replace(
            base,
            physical_block_device_bytes=physical_bytes,
            device_identity=f"{device.major}:{device.minor}/{device.name}",
            physical_source="sysfs-block-stat",
            physical_source_detail=detail,
        )

    __call__ = sample
    probe = sample


def probe_linux_ple_io_counters(
    payload_path: str | os.PathLike[str],
    *,
    sysfs_root: str | os.PathLike[str] = "/sys",
    counters: PLEIOCounters | Mapping[str, Any] | None = None,
    counter_source: Callable[[], PLEIOCounters | Mapping[str, Any]] | None = None,
    stat_fn: Callable[[str | os.PathLike[str]], object] | None = None,
    read_text_fn: Callable[[str | os.PathLike[str]], str] | None = None,
    system: str | None = None,
) -> PLEIOCounters:
    """Sample one explicit PLE payload through its Linux sysfs block device."""
    return LinuxPLEBlockCounterProbe(
        payload_path,
        sysfs_root=sysfs_root,
        counter_source=counter_source,
        stat_fn=stat_fn,
        read_text_fn=read_text_fn,
        system=system,
    ).sample(counters)


@dataclass(frozen=True, slots=True)
class PLEIOCounters:
    """Monotonic cumulative counters sampled at one instant.

    ``physical_block_device_bytes`` is optional at the snapshot boundary so a
    process-only source can be represented for diagnostics.  Measurement rejects
    such snapshots; callers must provide both the counter and its independently
    identified block-device source before physical bytes can be reported.
    ``device_identity`` may be a scalar device name or a sequence supplied by a
    multi-device probe.  A sequence is valid only when it contains exactly one
    non-empty identity, and is otherwise rejected as ambiguous.  ``physical_source``
    is a machine-readable source kind and is checked against
    :data:`PLE_IO_PHYSICAL_SOURCE_KINDS`; free-form command or path information
    belongs in ``physical_source_detail``.
    """

    major_faults: int
    logical_packed_bytes: int
    application_bytes_read: int
    application_reads: int
    physical_block_device_bytes: int | None = None
    device_identity: str | tuple[str, ...] | Sequence[str] | None = None
    physical_source: str | None = None
    physical_source_detail: str | None = None

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
    physical_source_detail: str | None = None

    def __post_init__(self) -> None:
        """Reject evidence that cannot be serialized or proven physical."""
        _validate_phase(self.phase)
        for name in (
            "major_faults",
            "logical_packed_bytes",
            "application_bytes_read",
            "application_reads",
            "physical_block_device_bytes",
        ):
            _validate_counter(name, getattr(self, name))
        amplification = self.read_amplification
        if isinstance(amplification, bool) or not isinstance(amplification, (int, float)):
            raise PLEIOEvidenceError("read_amplification must be a finite non-negative number")
        amplification = float(amplification)
        if not isfinite(amplification) or amplification < 0:
            raise PLEIOEvidenceError("read_amplification must be a finite non-negative number")
        object.__setattr__(self, "read_amplification", amplification)
        if self.physical_evidence is not True:
            raise PLEIOEvidenceError("physical_evidence must be true")
        _validate_nonempty_string("device_identity", self.device_identity)
        _validate_source_kind(self.physical_source)
        if self.physical_source_detail is not None:
            _validate_nonempty_string("physical_source_detail", self.physical_source_detail)

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
            "physical_source_detail": self.physical_source_detail,
            "physical_evidence": self.physical_evidence,
        }


def _validate_counter(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PLEIOEvidenceError(f"{name} must be a non-negative integer")


def _validate_phase(phase: object) -> str:
    if not isinstance(phase, str) or phase not in PLE_IO_PHASES:
        raise PLEIOEvidenceError(f"invalid PLE I/O phase {phase!r}; expected cold, warm, or steady")
    return phase


def _validate_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PLEIOEvidenceError(f"{name} must be a non-empty string")
    return value


def _validate_source_kind(value: object) -> str:
    if isinstance(value, str) and value in _PROCESS_ONLY_SOURCES:
        raise PLEIOEvidenceError("process I/O alone cannot establish physical block-device bytes")
    if not isinstance(value, str) or value not in PLE_IO_PHYSICAL_SOURCE_KINDS:
        raise PLEIOEvidenceError(
            "physical counter source is not an identified block-device source kind"
        )
    return value


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
    if before_source is None or after_source is None:
        raise PLEIOEvidenceError("physical block-device counter source is undefined")
    _validate_source_kind(before_source)
    _validate_source_kind(after_source)
    if before_source != after_source:
        raise PLEIOEvidenceError("physical block-device counter source changed")
    before_detail = before.physical_source_detail
    after_detail = after.physical_source_detail
    if (before_detail is None) != (after_detail is None):
        raise PLEIOEvidenceError("physical counter source detail changed")
    if before_detail is not None:
        _validate_nonempty_string("physical_source_detail", before_detail)
        _validate_nonempty_string("physical_source_detail", after_detail)
        if before_detail != after_detail:
            raise PLEIOEvidenceError("physical counter source detail changed")
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
        physical_source_detail=start.physical_source_detail,
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
        start = _coerce_counters(self._sample(counters))
        self._active_phase = phase
        self._start = start

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
        end = _coerce_counters(self._sample(counters))
        evidence = measure_ple_io_phase(phase, start, end)
        # Keep a failed end retry-safe: lifecycle state is consumed only after
        # sampling, validation, and delta measurement have all succeeded.
        self._active_phase = None
        self._start = None
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
    "LINUX_SYSFS_STAT_SECTOR_BYTES",
    "PLE_IO_PHASES",
    "PLE_IO_PHYSICAL_SOURCE_KINDS",
    "LinuxPLEBlockCounterProbe",
    "PLEBlockDevice",
    "PLEBlockProbeError",
    "PLEIOCounters",
    "PLEIOEvidenceError",
    "PLEIOEvidenceRecorder",
    "PLEIOPhaseEvidence",
    "measure_ple_io_phase",
    "probe_linux_ple_io_counters",
]
