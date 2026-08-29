from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import gguf
import numpy as np
import pytest
from freetoken.gguf_host import (
    MappedPLETable,
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


def test_ple_warm_modes_and_fault_telemetry_are_observable() -> None:
    with open_qwen_host_weights(FIXTURE, ple_warm_mode="cold") as weights:
        before = weights.ple.telemetry()
        weights.ple.warm_rows(np.array([0, 16, 31]))
        weights.ple.lookup(np.array([0, 16, 31]))
        after = weights.ple.telemetry()

    assert before["mode"] == "cold"
    assert after["lookup_rows"] == 3
    assert after["packed_bytes_read"] == 3 * 90
    assert after["minor_faults"] >= 0
    assert after["major_faults"] >= 0
    assert after["storage_read_bytes"] >= 0
    assert after["targeted_warm_rows"] == 3

    with MappedPLETable.open_from_gguf(FIXTURE, warm_mode="full-model-warm") as table:
        assert table.telemetry()["full_model_warm_bytes"] == FIXTURE.stat().st_size


def test_dedicated_ple_artifact_round_trip_and_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "ple"
    convert_gguf_ple_to_artifact(FIXTURE, artifact)
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["format"] == "freetoken-pascal-ple-v1"
    assert manifest["tensor_bytes"] == (artifact / "ple.bin").stat().st_size
    with MappedPLETable.open_from_artifact(artifact) as table:
        assert table.lookup(np.array([0, 31])).shape == (2, 160)


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
        assert pread_table.telemetry()["batch_physical_reads"] == 3
        assert pread_table.telemetry()["batch_sorted_rows"] == 3
        assert pread_table.telemetry()["batch_bytes_read"] == 3 * 90


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
