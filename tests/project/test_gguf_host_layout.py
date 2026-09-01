from __future__ import annotations

import base64
import hashlib
import json
import mmap
import os
import subprocess
import sys
import threading
from concurrent.futures import CancelledError
from pathlib import Path

import gguf
import numpy as np
import pytest
from freetoken.gguf_host import (
    MappedPLETable,
    PLELookupPlannerConfig,
    convert_gguf_ple_to_artifact,
    dequantize_iq4_nl,
    expert_layout_from_census,
    host_memory_report_from_census,
    inspect_qwen_host_layout,
    open_qwen_host_weights,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/gguf/qwen-host-layout.gguf"


def test_qwen_expert_layout_preserves_every_incompatible_pool() -> None:
    layout = inspect_qwen_host_layout(FIXTURE).experts

    assert layout.num_layers == 2
    assert layout.num_experts == 3
    assert len(layout.descriptors) == 6
    assert len(layout.slot_pools) == 6
    assert [layout.descriptor(0, name).quant_name for name in ("gate", "up", "down")] == [
        "Q4_K",
        "Q4_K",
        "Q5_1",
    ]
    assert [layout.descriptor(1, name).quant_name for name in ("gate", "up", "down")] == [
        "Q5_K",
        "Q5_K",
        "Q8_0",
    ]


def test_expert_and_slot_addressing_cover_first_middle_and_last() -> None:
    layout = inspect_qwen_host_layout(FIXTURE).experts

    for descriptor in layout.descriptors:
        for expert in (0, descriptor.experts // 2, descriptor.experts - 1):
            address = descriptor.source_offset(expert)
            assert descriptor.data_offset <= address
            assert address + descriptor.bytes_per_expert <= (
                descriptor.data_offset + descriptor.tensor_bytes
            )
        pool = layout.slot_pools[descriptor.pool_id]
        for slot in (0, 2, 4):
            assert pool.slot_offset(slot, num_slots=5) == slot * pool.bytes_per_slot

    with pytest.raises(IndexError, match="expert"):
        layout.descriptor(0, "gate").source_offset(3)
    with pytest.raises(IndexError, match="slot"):
        layout.slot_pools[0].slot_offset(5, num_slots=5)


def test_unsupported_expert_types_report_every_affected_bank() -> None:
    supported = {
        int(gguf.GGMLQuantizationType.Q4_K),
        int(gguf.GGMLQuantizationType.Q5_1),
    }

    with pytest.raises(ValueError) as raised:
        inspect_qwen_host_layout(FIXTURE, supported_expert_types=supported)

    message = str(raised.value)
    assert "blk.1.ffn_gate_exps.weight: Q5_K" in message
    assert "blk.1.ffn_up_exps.weight: Q5_K" in message
    assert "blk.1.ffn_down_exps.weight: Q8_0" in message


@pytest.mark.parametrize(
    "filename",
    ["qwen38-q4-census.metadata.json", "qwen38-q3-census.metadata.json"],
)
def test_real_census_layout_represents_all_48_layers(filename: str) -> None:
    census = json.loads((ROOT / "tests/fixtures/results" / filename).read_text(encoding="utf-8"))

    layout = expert_layout_from_census(census)

    assert layout.num_layers == 48
    assert layout.num_experts == 512
    assert len(layout.descriptors) == 144
    assert {descriptor.projection for descriptor in layout.descriptors} == {
        "gate",
        "up",
        "down",
    }
    assert sum(descriptor.tensor_bytes for descriptor in layout.descriptors) < 80 * (1 << 30)

    report = host_memory_report_from_census(census)
    assert report["total_tensor_bytes"] < 128 * (1 << 30)
    assert report["expert_mapped_bytes"] > report["ordinary_tensor_bytes"]
    assert report["ple_mapped_bytes"] == 28_800_138_240
    assert report["anonymous_host_source_bytes"] == 0
    assert report["pinned_host_source_bytes"] == 0


def test_mapped_expert_sources_are_file_backed_and_not_copied() -> None:
    with open_qwen_host_weights(FIXTURE) as weights:
        for descriptor in weights.layout.experts.descriptors:
            bank = weights.experts.bank(descriptor.layer, descriptor.projection)
            packed = bank.expert_packed(1)
            assert packed.shape == (descriptor.output_dim, descriptor.row_bytes)
            assert not packed.flags.writeable
            assert bank.mapping.file_backed
            del packed
        report = weights.memory_report()

    assert report["anonymous_model_bytes"] == 0
    assert report["expert_mapped_bytes"] > 0
    assert report["ple_mapped_bytes"] > 0
    assert report["pinned_bytes"] == 0


def test_open_qwen_host_weights_rolls_back_mmap_on_warm_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_warm_mode(self: MappedPLETable, mode: str) -> None:
        raise RuntimeError(f"injected warm setup failure: {mode}")

    def open_fd_count() -> int:
        return len(list(Path("/proc/self/fd").iterdir()))

    before = open_fd_count()
    monkeypatch.setattr(MappedPLETable, "set_warm_mode", fail_warm_mode)
    with pytest.raises(RuntimeError, match="warm setup failure"):
        open_qwen_host_weights(FIXTURE)
    assert open_fd_count() == before


def test_open_qwen_host_weights_rolls_back_ple_mapping_on_table_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_table_setup(self: MappedPLETable, *args: object, **kwargs: object) -> None:
        raise RuntimeError("injected table setup failure")

    def open_fd_count() -> int:
        return len(list(Path("/proc/self/fd").iterdir()))

    before = open_fd_count()
    monkeypatch.setattr(MappedPLETable, "__init__", fail_table_setup)
    with pytest.raises(RuntimeError, match="table setup failure"):
        open_qwen_host_weights(FIXTURE)
    assert open_fd_count() == before


def test_iq4_nl_ple_lookup_matches_independent_gguf_oracle() -> None:
    with open_qwen_host_weights(FIXTURE) as weights:
        ids = np.array([[0, 15], [16, 31]], dtype=np.int64)
        actual = weights.ple.lookup(ids)
        packed = weights.ple.mapping.rows[ids.reshape(-1)].copy()

    expected = gguf.dequantize(
        packed,
        gguf.GGMLQuantizationType.IQ4_NL,
    ).reshape(2, 2, 160)
    np.testing.assert_array_equal(actual, expected)


def test_real_ple_first_middle_last_rows_match_pinned_gguf_oracle() -> None:
    reference = json.loads(
        (ROOT / "tests/fixtures/gguf/qwen38-reference-rows.json").read_text(encoding="utf-8")
    )
    rows = [row for row in reference["rows"] if row["tensor"] == "per_layer_token_embd.weight"]

    assert [row["row_index"] for row in rows] == [0, 160000768, 320001535]
    for row in rows:
        packed = np.frombuffer(base64.b64decode(row["packed_base64"]), dtype=np.uint8)
        actual = dequantize_iq4_nl(packed.reshape(1, -1))
        assert (
            hashlib.sha256(actual.astype("<f4", copy=False).tobytes()).hexdigest()
            == row["reference_f32_sha256"]
        )


@pytest.mark.parametrize("ids", [np.array([-1]), np.array([32])])
def test_ple_invalid_indices_fail_without_partial_output(ids: np.ndarray) -> None:
    with open_qwen_host_weights(FIXTURE) as weights:
        with pytest.raises(IndexError, match="PLE row"):
            weights.ple.lookup(ids)
        assert weights.ple.telemetry()["lookup_rows"] == 0


def test_ple_hash_size_and_short_range_fail_closed(tmp_path: Path) -> None:
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    with MappedPLETable.open_from_gguf(
        FIXTURE,
        expected_file_sha256=digest,
        verify_file_sha256=True,
    ):
        pass

    with pytest.raises(ValueError, match="sha256"):
        MappedPLETable.open_from_gguf(
            FIXTURE,
            expected_file_sha256="0" * 64,
            verify_file_sha256=True,
        )

    short = tmp_path / "short.gguf"
    short.write_bytes(FIXTURE.read_bytes()[:-1])
    with pytest.raises(ValueError):
        MappedPLETable.open_from_gguf(short)
    with pytest.raises(ValueError):
        MappedPLETable.open_from_gguf(short, backend="pread")


def test_source_ple_rejects_unknown_backend_before_opening() -> None:
    with pytest.raises(ValueError, match="unknown PLE backend"):
        MappedPLETable.open_from_gguf(FIXTURE, backend="other")


def test_ple_warm_modes_and_fault_telemetry_are_observable() -> None:
    with open_qwen_host_weights(FIXTURE, ple_warm_mode="cold") as weights:
        before = weights.ple.telemetry()
        weights.ple.warm_rows(np.array([0, 16, 31]))
        weights.ple.lookup(np.array([0, 16, 31]))
        after = weights.ple.telemetry()

    assert before["mode"] == "cold"
    assert before["source_kind"] == "gguf-mmap"
    if hasattr(mmap, "MADV_RANDOM"):
        assert before["advice"] == "madv-random"
        assert before["advice_applied"] is True
    assert after["lookup_rows"] == 3
    assert after["packed_bytes_read"] == 3 * 90
    assert after["minor_faults"] >= 0
    assert after["major_faults"] >= 0
    assert after["storage_read_bytes"] >= 0
    assert after["targeted_warm_rows"] == 3

    with MappedPLETable.open_from_gguf(FIXTURE, warm_mode="full-model-warm") as table:
        assert table.telemetry()["full_model_warm_bytes"] == FIXTURE.stat().st_size


def test_source_ple_pread_targeted_warm_uses_descriptor_offset_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with MappedPLETable.open_from_gguf(FIXTURE, backend="pread") as table:
        real_pread = os.pread
        calls: list[tuple[int, int]] = []

        def recording_pread(fd: int, size: int, offset: int) -> bytes:
            calls.append((size, offset))
            return real_pread(fd, size, offset)

        monkeypatch.setattr("freetoken.gguf_host.os.pread", recording_pread)
        table.warm_rows(np.array([31, 0, 31, 16, 0]))
        telemetry = table.telemetry()

    descriptor = inspect_qwen_host_layout(FIXTURE).ple
    assert descriptor.data_offset > 0
    assert calls == [
        (1, descriptor.data_offset + row * descriptor.row_bytes) for row in (0, 16, 31)
    ]
    assert telemetry["targeted_warm_rows"] == 3
    assert telemetry["targeted_positional_warm_reads"] == 3


def test_dedicated_ple_pread_targeted_warm_keeps_zero_artifact_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    real_pread = os.pread
    calls: list[tuple[int, int]] = []

    def recording_pread(fd: int, size: int, offset: int) -> bytes:
        calls.append((size, offset))
        return real_pread(fd, size, offset)

    monkeypatch.setattr("freetoken.gguf_host.os.pread", recording_pread)
    with MappedPLETable.open_from_artifact(artifact, backend="pread") as table:
        table.warm_rows(np.array([31, 0, 31, 16]))
        telemetry = table.telemetry()

    assert calls == [(1, row * 90) for row in (0, 16, 31)]
    assert telemetry["targeted_warm_rows"] == 3
    assert telemetry["targeted_positional_warm_reads"] == 3


def test_source_ple_pread_async_prefetch_uses_descriptor_offset_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with MappedPLETable.open_from_gguf(FIXTURE, backend="pread") as table:
        real_pread = os.pread
        calls: list[tuple[int, int]] = []

        def recording_pread(fd: int, size: int, offset: int) -> bytes:
            calls.append((size, offset))
            return real_pread(fd, size, offset)

        monkeypatch.setattr("freetoken.gguf_host.os.pread", recording_pread)
        handle = table.prefetch(np.array([31, 0, 31, 16, 0]))
        handle.result()
        telemetry = table.telemetry()

    descriptor = inspect_qwen_host_layout(FIXTURE).ple
    assert calls == [
        (descriptor.row_bytes, descriptor.data_offset + row * descriptor.row_bytes)
        for row in (0, 16, 31)
    ]
    assert telemetry["prefetch_unique_rows"] == 3
    assert telemetry["prefetch_warmed_rows"] == 3


def test_source_ple_pread_full_ple_warm_uses_exact_descriptor_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_pread = os.pread
    calls: list[tuple[int, int]] = []

    def recording_pread(fd: int, size: int, offset: int) -> bytes:
        calls.append((size, offset))
        return real_pread(fd, size, offset)

    monkeypatch.setattr("freetoken.gguf_host.os.pread", recording_pread)
    with MappedPLETable.open_from_gguf(
        FIXTURE,
        backend="pread",
        warm_mode="full-ple-warm",
    ) as table:
        telemetry = table.telemetry()

    descriptor = inspect_qwen_host_layout(FIXTURE).ple
    assert calls == [(descriptor.tensor_bytes, descriptor.data_offset)]
    assert telemetry["full_model_warm_bytes"] == descriptor.tensor_bytes


def test_source_ple_pread_full_model_warm_reports_all_source_bytes() -> None:
    with MappedPLETable.open_from_gguf(
        FIXTURE,
        backend="pread",
        warm_mode="full-model-warm",
    ) as table:
        telemetry = table.telemetry()

    assert telemetry["full_model_warm_bytes"] == FIXTURE.stat().st_size


def test_source_ple_pread_advice_is_scoped_to_descriptor_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = inspect_qwen_host_layout(FIXTURE).ple
    calls: list[tuple[int, int, int, int]] = []

    def recording_fadvise(fd: int, offset: int, length: int, advice: int) -> None:
        calls.append((fd, offset, length, advice))

    monkeypatch.setattr("freetoken.gguf_host.os.posix_fadvise", recording_fadvise)
    with MappedPLETable.open_from_gguf(
        FIXTURE,
        backend="pread",
        warm_mode="page-cache-warm",
    ) as table:
        telemetry = table.telemetry()

    assert calls[0][1:3] == (descriptor.data_offset, descriptor.tensor_bytes)
    assert calls[1][1:3] == (descriptor.data_offset, descriptor.tensor_bytes)
    assert calls[0][3] == os.POSIX_FADV_RANDOM
    assert calls[1][3] == os.POSIX_FADV_WILLNEED
    assert telemetry["advice"] == "posix-fadv-random"
    assert telemetry["advice_applied"] is True


def test_source_ple_pread_lookup_matches_mmap_for_first_middle_last_and_duplicate() -> None:
    ids = np.array([0, 16, 31, 16, 0], dtype=np.int64)
    with MappedPLETable.open_from_gguf(FIXTURE) as mmap_table:
        expected = mmap_table.lookup_batch(ids)
    with MappedPLETable.open_from_gguf(FIXTURE, backend="pread") as pread_table:
        actual = pread_table.lookup_batch(ids)
        telemetry = pread_table.telemetry()
        source_kind = pread_table.source_kind

    np.testing.assert_array_equal(actual, expected)
    assert telemetry["backend"] == "pread"
    assert telemetry["source_kind"] == "gguf-pread"
    assert source_kind == "gguf-pread"
    assert telemetry["mapped_bytes"] == 0
    assert telemetry["batch_unique_rows"] == 3
    assert telemetry["batch_duplicate_rows"] == 2
    assert telemetry["batch_positional_reads"] == 3
    assert telemetry["batch_bytes_read"] == 3 * 90


def test_source_ple_pread_async_prefetch_partial_failure_is_public_and_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with MappedPLETable.open_from_gguf(FIXTURE, backend="pread") as table:
        real_pread = os.pread
        calls = 0

        def fail_second(fd: int, size: int, offset: int) -> bytes:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected second-row failure")
            return real_pread(fd, size, offset)

        monkeypatch.setattr("freetoken.gguf_host.os.pread", fail_second)
        handle = table.prefetch(np.array([31, 0, 16]))
        with pytest.raises(OSError, match="second-row"):
            handle.result()
        telemetry = table.telemetry()

    assert telemetry["prefetch_failed"] == 1
    assert telemetry["prefetch_warmed_rows"] == 1
    assert telemetry["prefetch_active"] is False


def test_source_ple_pread_targeted_warm_failure_does_not_claim_unwarmed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with MappedPLETable.open_from_gguf(FIXTURE, backend="pread") as table:
        real_pread = os.pread
        calls = 0

        def short_second_read(fd: int, size: int, offset: int) -> bytes:
            nonlocal calls
            calls += 1
            if calls == 2:
                return b""
            return real_pread(fd, size, offset)

        monkeypatch.setattr("freetoken.gguf_host.os.pread", short_second_read)
        with pytest.raises(ValueError, match="short PLE positional warm read"):
            table.warm_rows(np.array([0, 16, 31]))
        telemetry = table.telemetry()

    assert telemetry["short_reads"] == 1
    assert telemetry["targeted_positional_warm_reads"] == 1
    assert telemetry["targeted_warm_rows"] == 1


def test_source_ple_pread_invalid_fd_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "freetoken.gguf_host._open_validated_pread_fd",
        lambda *_args, **_kwargs: -1,
    )
    with pytest.raises(RuntimeError, match="file descriptor is invalid"):
        MappedPLETable.open_from_gguf(FIXTURE, backend="pread")


def test_dedicated_ple_artifact_round_trip_and_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "ple"
    convert_gguf_ple_to_artifact(FIXTURE, artifact)
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["format"] == "freetoken-pascal-ple-v1"
    assert manifest["data_offset"] == 0
    assert manifest["tensor_bytes"] == (artifact / "ple.bin").stat().st_size
    with MappedPLETable.open_from_artifact(artifact) as table:
        assert table.lookup(np.array([0, 31])).shape == (2, 160)


@pytest.mark.parametrize("backend", ["mmap", "pread"])
def test_dedicated_ple_maps_only_raw_ple_payload_at_offset_zero(
    tmp_path: Path, backend: str
) -> None:
    artifact = tmp_path / backend
    convert_gguf_ple_to_artifact(FIXTURE, artifact)

    with MappedPLETable.open_from_artifact(artifact, backend=backend) as table:
        assert table.descriptor.data_offset == 0
        assert table.descriptor.shard_path == str(artifact / "ple.bin")
        assert table._model_shard_paths == (str(artifact / "ple.bin"),)
        if backend == "mmap":
            assert table.mapping is not None
            assert table.mapping.path == artifact / "ple.bin"
            assert table.mapping._prefix == 0
            assert table.mapping.length == table.descriptor.tensor_bytes
        else:
            assert table.mapping is None


def test_dedicated_ple_rejects_nonzero_artifact_data_offset(tmp_path: Path) -> None:
    artifact = tmp_path / "bad-offset"
    convert_gguf_ple_to_artifact(FIXTURE, artifact)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["data_offset"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="data_offset must be zero"):
        MappedPLETable.open_from_artifact(artifact)


@pytest.mark.parametrize("backend", ["mmap", "pread"])
def test_dedicated_ple_artifact_constructor_failure_rolls_back_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    artifact = tmp_path / backend
    convert_gguf_ple_to_artifact(FIXTURE, artifact)

    def fail_table_setup(self: MappedPLETable, *args: object, **kwargs: object) -> None:
        raise RuntimeError("injected artifact table setup failure")

    def open_fd_count() -> int:
        return len(list(Path("/proc/self/fd").iterdir()))

    before = open_fd_count()
    monkeypatch.setattr(MappedPLETable, "__init__", fail_table_setup)
    with pytest.raises(RuntimeError, match="artifact table setup failure"):
        MappedPLETable.open_from_artifact(artifact, backend=backend)
    assert open_fd_count() == before


def test_dedicated_ple_artifact_constructor_failure_closes_fake_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "ple"
    convert_gguf_ple_to_artifact(FIXTURE, artifact)

    class FakeMapping:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise OSError("injected mapping close failure")

    mapping = FakeMapping()

    def fail_table_setup(self: MappedPLETable, *args: object, **kwargs: object) -> None:
        raise RuntimeError("injected artifact constructor failure")

    monkeypatch.setattr("freetoken.gguf_host.MappedFileRange", lambda *_args, **_kwargs: mapping)
    monkeypatch.setattr(MappedPLETable, "__init__", fail_table_setup)
    with pytest.raises(RuntimeError, match="artifact constructor failure"):
        MappedPLETable.open_from_artifact(artifact)
    assert mapping.close_calls == 1


@pytest.mark.parametrize("backend", ["mmap", "pread"])
@pytest.mark.parametrize("failure_method", ["_apply_random_advice", "set_warm_mode"])
def test_dedicated_ple_artifact_setup_failure_rolls_back_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    failure_method: str,
) -> None:
    artifact = tmp_path / f"{backend}-{failure_method}"
    convert_gguf_ple_to_artifact(FIXTURE, artifact)

    def fail_setup(self: MappedPLETable, *args: object, **kwargs: object) -> None:
        raise RuntimeError(f"injected artifact {failure_method} failure")

    def open_fd_count() -> int:
        return len(list(Path("/proc/self/fd").iterdir()))

    before = open_fd_count()
    monkeypatch.setattr(MappedPLETable, failure_method, fail_setup)
    with pytest.raises(RuntimeError, match=f"artifact {failure_method} failure"):
        MappedPLETable.open_from_artifact(artifact, backend=backend)
    assert open_fd_count() == before


def test_dedicated_ple_artifact_setup_failure_closes_fake_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "ple"
    convert_gguf_ple_to_artifact(FIXTURE, artifact)

    class FakeMapping:
        def __init__(self) -> None:
            self.close_calls = 0

        def advise(self, _advice: int) -> None:
            pass

        def close(self) -> None:
            self.close_calls += 1

    mapping = FakeMapping()

    def fail_setup(self: MappedPLETable, *args: object, **kwargs: object) -> None:
        raise RuntimeError("injected artifact warm setup failure")

    monkeypatch.setattr("freetoken.gguf_host.MappedFileRange", lambda *_args, **_kwargs: mapping)
    monkeypatch.setattr(MappedPLETable, "set_warm_mode", fail_setup)
    with pytest.raises(RuntimeError, match="artifact warm setup failure"):
        MappedPLETable.open_from_artifact(artifact)
    assert mapping.close_calls == 1


def test_dedicated_ple_artifact_rejects_tampering_and_bad_geometry(tmp_path: Path) -> None:
    artifact = tmp_path / "ple"
    convert_gguf_ple_to_artifact(FIXTURE, artifact)
    payload = artifact / "ple.bin"
    payload.write_bytes(payload.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="sha256"):
        MappedPLETable.open_from_artifact(artifact)

    convert_gguf_ple_to_artifact(FIXTURE, artifact := tmp_path / "bad")
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["row_bytes"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="geometry"):
        MappedPLETable.open_from_artifact(artifact)


def test_dedicated_full_warm_only_touches_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "ple"
    convert_gguf_ple_to_artifact(FIXTURE, artifact)
    touched: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "freetoken.gguf_host._warm_model_files", lambda paths: touched.append(paths) or 1
    )
    with MappedPLETable.open_from_artifact(artifact, warm_mode="full-ple-warm") as table:
        assert table.telemetry()["mode"] == "full-ple-warm"
    assert touched == [(str(artifact / "ple.bin"),)]


def test_dedicated_pread_batch_matches_mmap_and_deduplicates(tmp_path: Path) -> None:
    artifact = tmp_path / "ple"
    convert_gguf_ple_to_artifact(FIXTURE, artifact)
    ids = np.array([31, 0, 31, 16])
    with MappedPLETable.open_from_artifact(artifact) as mmap_table:
        expected = mmap_table.lookup_batch(ids)
    with MappedPLETable.open_from_artifact(artifact, backend="pread") as pread_table:
        actual = pread_table.lookup_batch(ids)
        np.testing.assert_array_equal(actual, expected)
        assert pread_table.telemetry()["batch_unique_rows"] == 3
        assert pread_table.telemetry()["backend"] == "pread"
        assert pread_table.telemetry()["mapped_bytes"] == 0
        assert pread_table.telemetry()["batch_positional_reads"] == 3
        assert pread_table.telemetry()["batch_sorted_rows"] == 3
        assert pread_table.telemetry()["batch_bytes_read"] == 3 * 90
        assert pread_table.telemetry()["advice"] == "posix-fadv-random"
        assert pread_table.telemetry()["advice_applied"] is True


@pytest.mark.parametrize("backend", ["mmap", "pread"])
def test_adaptive_ple_planner_uses_direct_path_at_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / backend)
    ids = np.array([31, 0, 31, 16], dtype=np.int64)
    with MappedPLETable.open_from_artifact(artifact) as reference_table:
        expected = reference_table.lookup_batch(ids)
    calls: list[int] = []
    real_pread = os.pread

    def recording_pread(fd: int, size: int, offset: int) -> bytes:
        calls.append(offset)
        return real_pread(fd, size, offset)

    monkeypatch.setattr("freetoken.gguf_host.os.pread", recording_pread)
    with MappedPLETable.open_from_artifact(
        artifact,
        backend=backend,
        planner_mode="adaptive",
        planner_direct_threshold=4,
    ) as table:
        actual = table.lookup_batch(ids)
        telemetry = table.telemetry()

    assert actual.shape == (4, 160)
    np.testing.assert_array_equal(actual, expected)
    assert telemetry["planner_mode"] == "adaptive"
    assert telemetry["planner_selected_mode"] == "direct"
    assert telemetry["planner_calls"] == 1
    assert telemetry["direct_calls"] == 1
    assert telemetry["direct_rows"] == 4
    assert telemetry["vectorized_calls"] == 0
    assert telemetry["planner_time_ns"] >= 0
    assert telemetry["application_reads"] == 4
    assert telemetry["application_bytes_read"] == 4 * 90
    if backend == "pread":
        assert calls == [31 * 90, 0, 31 * 90, 16 * 90]


@pytest.mark.parametrize("backend", ["mmap", "pread"])
def test_adaptive_ple_planner_uses_vectorized_path_above_threshold(
    tmp_path: Path, backend: str
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / backend)
    ids = np.array([31, 0, 31, 16], dtype=np.int64)
    with MappedPLETable.open_from_artifact(
        artifact,
        backend=backend,
        planner_mode="adaptive",
        planner_direct_threshold=3,
    ) as table:
        actual = table.lookup_batch(ids)
        telemetry = table.telemetry()

    assert actual.shape == (4, 160)
    assert telemetry["planner_selected_mode"] == "vectorized"
    assert telemetry["planner_calls"] == 1
    assert telemetry["direct_calls"] == 0
    assert telemetry["vectorized_calls"] == 1
    assert telemetry["vectorized_rows"] == 4
    assert telemetry["application_reads"] == 3
    assert telemetry["application_bytes_read"] == 3 * 90
    assert telemetry["batch_unique_rows"] == 3
    assert telemetry["batch_duplicate_rows"] == 1
    assert telemetry["batch_sorted_rows"] == 3


def test_ple_planner_config_is_explicit_and_fails_before_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    with pytest.raises(ValueError, match="planner mode"):
        MappedPLETable.open_from_artifact(artifact, planner_mode="unknown")
    with pytest.raises(TypeError, match="planner_direct_threshold"):
        MappedPLETable.open_from_artifact(artifact, planner_direct_threshold=True)
    with pytest.raises(ValueError, match="positive"):
        MappedPLETable.open_from_artifact(artifact, planner_direct_threshold=0)

    monkeypatch.setattr(
        "freetoken.gguf_host.MappedFileRange",
        lambda *_args, **_kwargs: pytest.fail("invalid planner config opened a mapping"),
    )
    with pytest.raises(ValueError, match="planner mode"):
        MappedPLETable.open_from_artifact(
            artifact,
            planner_config=PLELookupPlannerConfig(mode="unknown"),
        )


def test_ple_planner_empty_and_invalid_requests_do_no_io_or_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    with MappedPLETable.open_from_artifact(
        artifact,
        backend="pread",
        planner_mode="adaptive",
        planner_direct_threshold=1,
    ) as table:
        monkeypatch.setattr(
            "freetoken.gguf_host.os.pread",
            lambda *_args: pytest.fail("unexpected positional read"),
        )
        assert table.lookup_batch(np.array([], dtype=np.int64)).shape == (0, 160)
        with pytest.raises(IndexError):
            table.lookup_batch(np.array([table.descriptor.rows]))
        telemetry = table.telemetry()

    assert telemetry["planner_calls"] == 0
    assert telemetry["planner_time_ns"] == 0
    assert telemetry["direct_calls"] == 0
    assert telemetry["vectorized_calls"] == 0
    assert telemetry["application_reads"] == 0
    assert telemetry["application_bytes_read"] == 0


def test_dedicated_pread_full_warm_never_mmaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    monkeypatch.setattr(
        "freetoken.gguf_host.mmap.mmap",
        lambda *_args, **_kwargs: pytest.fail("unexpected mmap"),
    )
    with MappedPLETable.open_from_artifact(
        artifact, backend="pread", warm_mode="full-ple-warm"
    ) as table:
        assert table.telemetry()["full_model_warm_bytes"] == table.descriptor.tensor_bytes


def test_dedicated_pread_batch_fails_closed_on_short_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    with MappedPLETable.open_from_artifact(artifact, backend="pread") as table:
        monkeypatch.setattr("freetoken.gguf_host.os.pread", lambda *_args: b"")
        with pytest.raises(ValueError, match="short PLE positional read"):
            table.lookup_batch(np.array([0, 0]))
        telemetry = table.telemetry()
        assert telemetry["short_reads"] == 1
        assert telemetry["batch_calls"] == 0
        assert telemetry["lookup_rows"] == 0
        assert telemetry["batch_positional_reads"] == 1
        assert telemetry["application_reads"] == 1
        assert telemetry["batch_bytes_read"] == 0
        assert telemetry["application_bytes_read"] == 0


def test_dedicated_pread_batch_retains_partial_io_after_later_short_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    with MappedPLETable.open_from_artifact(artifact, backend="pread") as table:
        row_bytes = table.descriptor.row_bytes
        real_pread = os.pread
        calls = 0

        def short_second_read(fd: int, size: int, offset: int) -> bytes:
            nonlocal calls
            calls += 1
            if calls == 2:
                return b"partial"
            return real_pread(fd, size, offset)

        monkeypatch.setattr("freetoken.gguf_host.os.pread", short_second_read)
        with pytest.raises(ValueError, match="short PLE positional read"):
            table.lookup_batch(np.array([0, 16, 31]))
        telemetry = table.telemetry()

    assert telemetry["short_reads"] == 1
    assert telemetry["batch_positional_reads"] == 2
    assert telemetry["application_reads"] == 2
    assert telemetry["batch_bytes_read"] == row_bytes + len(b"partial")
    assert telemetry["application_bytes_read"] == row_bytes + len(b"partial")
    assert telemetry["batch_calls"] == 0
    assert telemetry["lookup_calls"] == 0


def test_dedicated_pread_batch_retains_partial_io_after_later_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    with MappedPLETable.open_from_artifact(artifact, backend="pread") as table:
        row_bytes = table.descriptor.row_bytes
        real_pread = os.pread
        calls = 0

        def failing_second_read(fd: int, size: int, offset: int) -> bytes:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected second-row failure")
            return real_pread(fd, size, offset)

        monkeypatch.setattr("freetoken.gguf_host.os.pread", failing_second_read)
        with pytest.raises(OSError, match="second-row"):
            table.lookup_batch(np.array([0, 16, 31]))
        telemetry = table.telemetry()

    assert telemetry["short_reads"] == 0
    assert telemetry["batch_positional_reads"] == 2
    assert telemetry["application_reads"] == 2
    assert telemetry["batch_bytes_read"] == row_bytes
    assert telemetry["application_bytes_read"] == row_bytes
    assert telemetry["batch_calls"] == 0
    assert telemetry["lookup_calls"] == 0


def test_dedicated_pread_empty_and_invalid_batches_do_no_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    with MappedPLETable.open_from_artifact(artifact, backend="pread") as table:
        monkeypatch.setattr(
            "freetoken.gguf_host.os.pread",
            lambda *_args: pytest.fail("unexpected positional read"),
        )
        assert table.lookup_batch(np.array([], dtype=np.int64)).shape == (0, 160)
        with pytest.raises(IndexError):
            table.lookup_batch(np.array([table.descriptor.rows]))
        assert table.telemetry()["batch_calls"] == 0


def test_ple_prefetch_deduplicates_ids_and_only_reports_warming(tmp_path: Path) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    with MappedPLETable.open_from_artifact(artifact, prefetch_max_rows=4) as table:
        handle = table.prefetch(np.array([31, 0, 31, 16]))

        assert handle.row_ids == (0, 16, 31)
        with pytest.raises(TypeError):
            handle.row_ids[0] = 99  # type: ignore[index]
        with pytest.raises(AttributeError):
            handle.row_ids = (99,)  # type: ignore[misc]
        assert handle.result() is None

        telemetry = table.telemetry()
        assert telemetry["prefetch_active"] is False
        assert telemetry["prefetch_submitted"] == 1
        assert telemetry["prefetch_completed"] == 1
        assert telemetry["prefetch_cancelled"] == 0
        assert telemetry["prefetch_failed"] == 0
        assert telemetry["prefetch_requested_rows"] == 4
        assert telemetry["prefetch_unique_rows"] == 3
        assert telemetry["prefetch_warmed_rows"] == 3


def test_ple_prefetch_is_bounded_and_rejects_a_second_active_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    with MappedPLETable.open_from_artifact(artifact, backend="pread", prefetch_max_rows=2) as table:
        with pytest.raises(ValueError, match=r"prefetch.*2 rows"):
            table.prefetch(np.array([0, 1, 2]))
        assert table.telemetry()["prefetch_submitted"] == 0

        real_pread = os.pread
        started = threading.Event()
        release = threading.Event()

        def blocked_pread(fd: int, size: int, offset: int) -> bytes:
            started.set()
            release.wait(timeout=2)
            return real_pread(fd, size, offset)

        monkeypatch.setattr("freetoken.gguf_host.os.pread", blocked_pread)
        handle = table.prefetch(np.array([0, 1]))
        assert started.wait(timeout=2)
        assert table.telemetry()["prefetch_active"] is True
        with pytest.raises(RuntimeError, match="prefetch already active"):
            table.prefetch(np.array([2]))
        release.set()
        handle.result()


def test_ple_prefetch_cancellation_stops_between_mmap_chunks(tmp_path: Path) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    with MappedPLETable.open_from_artifact(
        artifact, prefetch_max_rows=4, prefetch_chunk_rows=1
    ) as table:
        copied_rows = np.array(table.mapping.rows, copy=True)
        started = threading.Event()
        release = threading.Event()
        calls = 0

        class BlockingRows:
            def __getitem__(self, index):
                nonlocal calls
                calls += 1
                if calls == 1:
                    started.set()
                    release.wait(timeout=2)
                return copied_rows[index]

        table.mapping.rows = BlockingRows()
        handle = table.prefetch(np.array([0, 1, 2]))
        assert started.wait(timeout=2)
        assert handle.cancel() is True
        release.set()
        with pytest.raises(CancelledError):
            handle.wait()
        assert calls == 1
        telemetry = table.telemetry()
        assert telemetry["prefetch_cancelled"] == 1
        assert telemetry["prefetch_warmed_rows"] == 1


def test_ple_prefetch_close_cancels_and_joins_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    table = MappedPLETable.open_from_artifact(artifact, backend="pread")
    real_pread = os.pread
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def blocked_pread(fd: int, size: int, offset: int) -> bytes:
        started.set()
        release.wait(timeout=2)
        return real_pread(fd, size, offset)

    monkeypatch.setattr("freetoken.gguf_host.os.pread", blocked_pread)
    handle = table.prefetch(np.array([0, 1]))
    assert started.wait(timeout=2)

    def close_table() -> None:
        table.close()
        closed.set()

    closer = threading.Thread(target=close_table)
    closer.start()
    assert not closed.wait(timeout=0.05)
    release.set()
    closer.join(timeout=2)
    assert closed.is_set()
    with pytest.raises(CancelledError):
        handle.result()
    with pytest.raises(RuntimeError, match="closed"):
        table.prefetch(np.array([], dtype=np.int64))


def test_ple_prefetch_cancellation_stops_between_pread_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    with MappedPLETable.open_from_artifact(artifact, backend="pread") as table:
        real_pread = os.pread
        started = threading.Event()
        release = threading.Event()
        calls = 0

        def blocked_first_pread(fd: int, size: int, offset: int) -> bytes:
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                release.wait(timeout=2)
            return real_pread(fd, size, offset)

        monkeypatch.setattr("freetoken.gguf_host.os.pread", blocked_first_pread)
        handle = table.prefetch(np.array([0, 1, 2]))
        assert started.wait(timeout=2)
        assert handle.cancel() is True
        release.set()
        with pytest.raises(CancelledError):
            handle.wait()
        assert calls == 1
        telemetry = table.telemetry()
        assert telemetry["prefetch_cancelled"] == 1
        assert telemetry["prefetch_warmed_rows"] == 1


def test_ple_prefetch_failure_is_visible_and_does_not_poison_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    with MappedPLETable.open_from_artifact(artifact, backend="pread") as table:
        real_pread = os.pread
        failed = True

        def fail_once(fd: int, size: int, offset: int) -> bytes:
            nonlocal failed
            if failed:
                failed = False
                raise OSError("injected PLE prefetch failure")
            return real_pread(fd, size, offset)

        monkeypatch.setattr("freetoken.gguf_host.os.pread", fail_once)
        handle = table.prefetch(np.array([0, 1]))
        with pytest.raises(OSError, match="injected PLE prefetch failure"):
            handle.result()
        monkeypatch.setattr("freetoken.gguf_host.os.pread", real_pread)

        assert table.lookup(np.array([0])).shape == (1, 160)
        telemetry = table.telemetry()
        assert telemetry["prefetch_failed"] == 1
        assert telemetry["prefetch_active"] is False
        assert telemetry["lookup_rows"] == 1


def test_ple_prefetch_partial_failure_reports_rows_and_finalizes_before_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    with MappedPLETable.open_from_artifact(artifact, backend="pread") as table:
        real_pread = os.pread
        calls = 0

        def fail_second(fd: int, size: int, offset: int) -> bytes:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected second-row failure")
            return real_pread(fd, size, offset)

        monkeypatch.setattr("freetoken.gguf_host.os.pread", fail_second)
        handle = table.prefetch(np.array([0, 1, 2]))
        with pytest.raises(OSError, match="second-row"):
            handle.result()
        telemetry = table.telemetry()
        assert telemetry["prefetch_failed"] == 1
        assert telemetry["prefetch_warmed_rows"] == 1
        assert telemetry["prefetch_active"] is False


def test_ple_prefetch_done_and_cancel_share_finalization_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    with MappedPLETable.open_from_artifact(artifact, backend="pread") as table:
        entered_finish = threading.Event()
        release_finish = threading.Event()
        real_finish = table._finish_prefetch

        def delayed_finish(handle, future) -> None:
            entered_finish.set()
            release_finish.wait(timeout=2)
            real_finish(handle, future)

        monkeypatch.setattr(table, "_finish_prefetch", delayed_finish)
        handle = table.prefetch(np.array([0]))
        assert entered_finish.wait(timeout=2)
        assert handle.done() is False
        assert handle.cancel() is True
        release_finish.set()
        with pytest.raises(CancelledError):
            handle.result()
        assert handle.done() is True
        assert handle.cancel() is False
        telemetry = table.telemetry()
        assert telemetry["prefetch_cancelled"] == 1
        assert telemetry["prefetch_completed"] == 0


def test_ple_close_waits_for_synchronous_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    table = MappedPLETable.open_from_artifact(artifact, backend="pread")
    real_pread = os.pread
    started = threading.Event()
    release = threading.Event()
    lookup_done = threading.Event()
    close_done = threading.Event()

    def blocked_pread(fd: int, size: int, offset: int) -> bytes:
        started.set()
        release.wait(timeout=2)
        return real_pread(fd, size, offset)

    monkeypatch.setattr("freetoken.gguf_host.os.pread", blocked_pread)
    lookup = threading.Thread(target=lambda: (table.lookup_batch(np.array([0])), lookup_done.set()))
    lookup.start()
    assert started.wait(timeout=2)
    closer = threading.Thread(target=lambda: (table.close(), close_done.set()))
    closer.start()
    assert not close_done.wait(timeout=0.05)
    release.set()
    lookup.join(timeout=2)
    closer.join(timeout=2)
    assert lookup_done.is_set()
    assert close_done.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        table.lookup_batch(np.array([0]))


def test_ple_prefetch_empty_request_is_immediately_complete(tmp_path: Path) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    with MappedPLETable.open_from_artifact(artifact, prefetch_max_rows=0) as table:
        handle = table.prefetch(np.array([], dtype=np.int64))
        assert handle.done() is True
        assert handle.wait() is None
        telemetry = table.telemetry()
        assert telemetry["prefetch_active"] is False
        assert telemetry["prefetch_submitted"] == 1
        assert telemetry["prefetch_completed"] == 1
        assert telemetry["prefetch_requested_rows"] == 0
        assert telemetry["prefetch_unique_rows"] == 0
        assert telemetry["prefetch_warmed_rows"] == 0


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"prefetch_max_rows": True}, TypeError, "prefetch_max_rows"),
        ({"prefetch_max_rows": -1}, ValueError, "non-negative"),
        ({"prefetch_chunk_rows": 0}, ValueError, "positive"),
    ],
)
def test_ple_prefetch_configuration_fails_before_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    artifact = convert_gguf_ple_to_artifact(FIXTURE, tmp_path / "ple")
    monkeypatch.setattr(
        "freetoken.gguf_host.MappedFileRange",
        lambda *_args, **_kwargs: pytest.fail("invalid prefetch config opened a mapping"),
    )
    with pytest.raises(error, match=message):
        MappedPLETable.open_from_artifact(artifact, **kwargs)


def test_dedicated_loader_ignores_provenance_source_path(tmp_path: Path) -> None:
    artifact = tmp_path / "ple"
    convert_gguf_ple_to_artifact(FIXTURE, artifact)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["path"] = str(tmp_path / "missing-model.gguf")
    manifest_path.write_text(json.dumps(manifest))
    with MappedPLETable.open_from_artifact(artifact) as table:
        assert table.lookup(np.array([0])).shape == (1, 160)


def test_host_layout_cli_reports_selected_behavior() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/inspect_gguf_host.py", str(FIXTURE)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(result.stdout)

    assert document["expert_layers"] == 2
    assert len(document["slot_pools"]) == 6
    assert document["ple"]["quant_type"] == "IQ4_NL"
    assert document["memory"]["pinned_host_source_bytes"] == 0
