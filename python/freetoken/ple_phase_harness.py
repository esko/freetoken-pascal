"""Integrated H0 evidence runner for the dedicated PLE artifact."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from freetoken.ple_io_evidence import (
    PLE_IO_PHASE_ORDER,
    LinuxPLEBlockCounterProbe,
    PLEIOCounters,
    PLEIOEvidenceError,
    PLEIOEvidenceRecorder,
)

_TELEMETRY_COUNTERS = (
    "lookup_calls",
    "lookup_rows",
    "packed_bytes_read",
    "output_bytes",
    "minor_faults",
    "major_faults",
    "storage_read_bytes",
    "targeted_warm_rows",
    "full_model_warm_bytes",
    "planner_calls",
    "planner_time_ns",
    "direct_calls",
    "direct_rows",
    "vectorized_calls",
    "vectorized_rows",
    "application_reads",
    "application_bytes_read",
    "batch_calls",
    "batch_requested_rows",
    "batch_unique_rows",
    "batch_positional_reads",
    "batch_duplicate_rows",
    "batch_sorted_rows",
    "batch_bytes_read",
    "short_reads",
    "targeted_positional_warm_reads",
    "prefetch_submitted",
    "prefetch_completed",
    "prefetch_cancelled",
    "prefetch_failed",
    "prefetch_requested_rows",
    "prefetch_unique_rows",
    "prefetch_warmed_rows",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise PLEIOEvidenceError(f"cannot hash PLE artifact file {path}: {error}") from error
    return digest.hexdigest()


def _artifact_identity(root: Path) -> dict[str, Any]:
    """Read identity after ``MappedPLETable`` has validated the artifact."""
    try:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = root / manifest["payload"]
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise PLEIOEvidenceError("cannot read dedicated PLE artifact identity") from error
    codec = manifest.get("codec")
    if isinstance(codec, Mapping):
        codec_identity = f"{codec.get('id')}@{codec.get('version')}"
    else:
        codec_identity = "iq4_nl@1"
    return {
        "format": manifest.get("format"),
        "version": manifest.get("version"),
        "manifest_sha256": _sha256(manifest_path),
        "payload_sha256": _sha256(payload),
        "rows": manifest.get("rows"),
        "elements_per_row": manifest.get("elements_per_row"),
        "row_bytes": manifest.get("row_bytes"),
        "tensor_bytes": manifest.get("tensor_bytes"),
        "tensor_name": manifest.get("tensor_name"),
        "codec_identity": codec_identity,
    }


def _normalize_batches(row_batches: Sequence[object]) -> tuple[tuple[int, ...], ...]:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - required by the table
        raise PLEIOEvidenceError("numpy is required for PLE phase evidence") from error
    if isinstance(row_batches, np.ndarray):
        batches: tuple[object, ...] = (row_batches,)
    else:
        try:
            batches = tuple(row_batches)
        except TypeError as error:
            raise PLEIOEvidenceError("row_batches must be a non-empty sequence") from error
    if not batches:
        raise PLEIOEvidenceError("row_batches must contain at least one batch")
    normalized: list[tuple[int, ...]] = []
    for index, batch in enumerate(batches):
        values = np.asarray(batch)
        if values.dtype.kind not in "iu" or values.dtype.kind == "b":
            raise PLEIOEvidenceError(f"row batch {index} must contain integer IDs")
        flat = values.reshape(-1)
        if not flat.size:
            raise PLEIOEvidenceError(f"row batch {index} is empty")
        normalized.append(tuple(int(value) for value in flat.tolist()))
    return tuple(normalized)


def _telemetry_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for name in _TELEMETRY_COUNTERS:
        start, end = before.get(name), after.get(name)
        if isinstance(start, bool) or not isinstance(start, int):
            raise PLEIOEvidenceError(f"phase telemetry counter {name} is invalid")
        if isinstance(end, bool) or not isinstance(end, int):
            raise PLEIOEvidenceError(f"phase telemetry counter {name} is invalid")
        if end < start:
            raise PLEIOEvidenceError(f"phase telemetry counter regression in {name}")
        delta[name] = end - start
    delta["planner_selected_mode"] = after.get("planner_selected_mode")
    delta["mode"] = after.get("mode")
    return delta


def _join_table_and_physical(
    telemetry: Mapping[str, Any],
    physical: PLEIOCounters | Mapping[str, Any],
) -> PLEIOCounters:
    try:
        base = PLEIOCounters(
            telemetry["major_faults"],
            telemetry["packed_bytes_read"],
            telemetry["application_bytes_read"],
            telemetry["application_reads"],
        )
    except (KeyError, TypeError, PLEIOEvidenceError) as error:
        raise PLEIOEvidenceError("PLE table telemetry lacks application counters") from error
    if isinstance(physical, PLEIOCounters):
        values: Mapping[str, Any] = {
            "physical_block_device_bytes": physical.physical_block_device_bytes,
            "device_identity": physical.device_identity,
            "physical_source": physical.physical_source,
            "physical_source_detail": physical.physical_source_detail,
        }
    elif isinstance(physical, Mapping):
        # Explicitly ignore process fields such as storage_read_bytes.
        values = physical
    else:
        raise PLEIOEvidenceError("physical counter source returned an invalid value")
    try:
        return PLEIOCounters(
            base.major_faults,
            base.logical_packed_bytes,
            base.application_bytes_read,
            base.application_reads,
            physical_block_device_bytes=values.get("physical_block_device_bytes"),
            device_identity=values.get("device_identity"),
            physical_source=values.get("physical_source"),
            physical_source_detail=values.get("physical_source_detail"),
        )
    except (TypeError, PLEIOEvidenceError) as error:
        raise PLEIOEvidenceError(f"invalid physical counter source: {error}") from error


class PLEIOPhaseHarness:
    """Run identical batches through both production PLE storage backends.

    ``physical_counter_source`` is the deterministic H0 injection seam.  A real
    ``LinuxPLEBlockCounterProbe`` is opt-in and is the only path labelled as
    measured physical evidence.  Cache preparation is deliberately reported as
    synthetic/advisory unless the caller supplies a preparation callback or
    explicitly declares that an operator prepared the cache state.
    """

    def __init__(
        self,
        artifact: str | os.PathLike[str],
        *,
        row_batches: Sequence[object],
        physical_counter_source: Callable[[], PLEIOCounters | Mapping[str, Any]] | None = None,
        linux_probe: LinuxPLEBlockCounterProbe | None = None,
        prefetch: bool = True,
        prefetch_max_rows: int = 4096,
        prefetch_chunk_rows: int = 64,
        planner_mode: str = "vectorized",
        planner_direct_threshold: int = 8,
        phase_preparer: Callable[[str, Any], None] | None = None,
        operator_cache_control: bool = False,
    ) -> None:
        if (physical_counter_source is None) == (linux_probe is None):
            raise PLEIOEvidenceError(
                "provide exactly one injected physical source or opt-in Linux probe"
            )
        if physical_counter_source is not None and not callable(physical_counter_source):
            raise TypeError("physical_counter_source must be callable")
        if linux_probe is not None and not callable(linux_probe):
            raise TypeError("linux_probe must be callable")
        if phase_preparer is not None and not callable(phase_preparer):
            raise TypeError("phase_preparer must be callable")
        if not isinstance(prefetch, bool):
            raise TypeError("prefetch must be a boolean")
        self._root = Path(artifact)
        if self._root.is_file():
            raise PLEIOEvidenceError(
                "v1 PLE phase evidence requires a dedicated artifact; "
                "source-GGUF full-model-warm is forbidden"
            )
        self._batches = _normalize_batches(row_batches)
        self._physical_source = physical_counter_source
        self._linux_probe = linux_probe
        self._prefetch = prefetch
        self._prefetch_max_rows = prefetch_max_rows
        self._prefetch_chunk_rows = prefetch_chunk_rows
        self._planner_mode = planner_mode
        self._planner_direct_threshold = planner_direct_threshold
        self._phase_preparer = phase_preparer
        self._operator_cache_control = operator_cache_control
        self._artifact: dict[str, Any] | None = None

    def _sample(self, table: Any) -> tuple[PLEIOCounters, dict[str, Any]]:
        telemetry = dict(table.telemetry())
        if self._linux_probe is not None:
            physical = self._linux_probe.sample(PLEIOCounters(0, 0, 0, 0))
        else:
            if self._physical_source is None:
                raise PLEIOEvidenceError("physical counter source is unavailable")
            physical = self._physical_source()
        return _join_table_and_physical(telemetry, physical), telemetry

    def _cache_state(self, phase: str) -> str:
        if self._phase_preparer is not None:
            return "callback-prepared"
        if self._operator_cache_control:
            return "operator-declared"
        return "synthetic/advisory"

    def _run_backend(
        self, backend: str, expected_hashes: list[str] | None
    ) -> tuple[dict[str, Any], list[str]]:
        import numpy as np

        from freetoken.gguf_host import MappedPLETable

        table = MappedPLETable.open_from_artifact(
            self._root,
            backend=backend,
            warm_mode="cold",
            prefetch_max_rows=self._prefetch_max_rows,
            prefetch_chunk_rows=self._prefetch_chunk_rows,
            planner_mode=self._planner_mode,
            planner_direct_threshold=self._planner_direct_threshold,
        )
        snapshots: list[dict[str, Any]] = []

        def provider() -> PLEIOCounters:
            counters, telemetry = self._sample(table)
            snapshots.append({"counters": counters.to_dict(), "table_telemetry": telemetry})
            return counters

        recorder = PLEIOEvidenceRecorder(provider)
        phases: list[dict[str, Any]] = []
        raw_samples: list[dict[str, Any]] = []
        try:
            for phase in PLE_IO_PHASE_ORDER:
                if self._phase_preparer is not None:
                    self._phase_preparer(phase, table)
                elif phase == "warm":
                    table.set_warm_mode("page-cache-warm")
                before_count = len(snapshots)
                recorder.begin_phase(phase)
                if phase == "warm" and self._prefetch:
                    rows = np.asarray(
                        [row for batch in self._batches for row in batch], dtype=np.int64
                    )
                    table.prefetch(rows).result()
                result_hashes: list[str] = []
                for batch in self._batches:
                    output = table.lookup_batch(np.asarray(batch, dtype=np.int64))
                    result_hashes.append(
                        hashlib.sha256(
                            np.asarray(output, dtype="<f4").tobytes(order="C")
                        ).hexdigest()
                    )
                evidence = recorder.end_phase(phase)
                if len(snapshots) - before_count != 2 or evidence.phase != phase:
                    raise PLEIOEvidenceError(f"phase {phase} boundary evidence is corrupt")
                before, after = snapshots[-2:]
                telemetry_delta = _telemetry_delta(
                    before["table_telemetry"], after["table_telemetry"]
                )
                phase_record = evidence.to_dict()
                phase_record.update(
                    {
                        "cache_state": self._cache_state(phase),
                        "logical_rows": telemetry_delta["batch_requested_rows"],
                        "unique_rows": telemetry_delta["batch_unique_rows"],
                        "result_sha256": result_hashes,
                        "telemetry_delta": telemetry_delta,
                    }
                )
                phases.append(phase_record)
                raw_samples.append({"phase": phase, "before": before, "after": after})
            completed = recorder.completed
            if tuple(item.phase for item in completed) != PLE_IO_PHASE_ORDER:
                raise PLEIOEvidenceError("phase recorder completed phases are corrupt")
            telemetry = table.telemetry()
        finally:
            table.close()
        actual_hashes = [digest for phase in phases for digest in phase["result_sha256"]]
        if expected_hashes is not None and actual_hashes != expected_hashes:
            raise PLEIOEvidenceError("mmap and pread ordered PLE rows differ")
        return {
            "backend": backend,
            "source_kind": telemetry["source_kind"],
            "advice": telemetry["advice"],
            "advice_applied": telemetry["advice_applied"],
            "advice_error": telemetry["advice_error"],
            "codec": {
                "identity": telemetry["codec_identity"],
                "id": telemetry["codec_id"],
                "version": telemetry["codec_version"],
            },
            "artifact": dict(self._artifact),
            "phases": phases,
            "raw_phase_samples": raw_samples,
            "telemetry": telemetry,
        }, actual_hashes

    def run(self) -> dict[str, Any]:
        """Return one JSON-compatible H0 observation-only report."""
        if not self._root.is_dir():
            raise PLEIOEvidenceError(f"dedicated PLE artifact directory is missing: {self._root}")
        self._artifact = _artifact_identity(self._root)
        backends: list[dict[str, Any]] = []
        expected_hashes: list[str] | None = None
        for backend in ("mmap", "pread"):
            record, expected_hashes = self._run_backend(backend, expected_hashes)
            backends.append(record)
        report = {
            "format": "freetoken-pascal-ple-io-phase-evidence-v1",
            "evidence_status": "h0-observation",
            "physical_counter_status": (
                "measured" if self._linux_probe is not None else "injected"
            ),
            "claim_status": "observation_only",
            "validation_class": "H0/no-P4",
            "cache_control": (
                "callback-prepared"
                if self._phase_preparer is not None
                else "operator-declared"
                if self._operator_cache_control
                else "synthetic/advisory"
            ),
            "artifact": dict(self._artifact),
            "row_batches": [list(batch) for batch in self._batches],
            "backends": backends,
        }
        if [item["backend"] for item in backends] != ["mmap", "pread"]:
            raise PLEIOEvidenceError("phase report backend records are corrupt")
        if any(
            [item["phase"] for item in backend["phases"]] != list(PLE_IO_PHASE_ORDER)
            for backend in backends
        ):
            raise PLEIOEvidenceError("phase report records are missing or reordered")
        return report


def run_ple_io_phase_harness(
    artifact: str | os.PathLike[str],
    *,
    row_batches: Sequence[object],
    physical_counter_source: Callable[[], PLEIOCounters | Mapping[str, Any]] | None = None,
    linux_probe: LinuxPLEBlockCounterProbe | None = None,
    prefetch: bool = True,
    prefetch_max_rows: int = 4096,
    prefetch_chunk_rows: int = 64,
    planner_mode: str = "vectorized",
    planner_direct_threshold: int = 8,
    phase_preparer: Callable[[str, Any], None] | None = None,
    operator_cache_control: bool = False,
) -> dict[str, Any]:
    return PLEIOPhaseHarness(
        artifact,
        row_batches=row_batches,
        physical_counter_source=physical_counter_source,
        linux_probe=linux_probe,
        prefetch=prefetch,
        prefetch_max_rows=prefetch_max_rows,
        prefetch_chunk_rows=prefetch_chunk_rows,
        planner_mode=planner_mode,
        planner_direct_threshold=planner_direct_threshold,
        phase_preparer=phase_preparer,
        operator_cache_control=operator_cache_control,
    ).run()


__all__ = ["PLEIOPhaseHarness", "run_ple_io_phase_harness"]
