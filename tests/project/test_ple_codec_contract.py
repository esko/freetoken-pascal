from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from freetoken.gguf_host import (
    IQ4_NL_CODEC_DESCRIPTOR,
    PLE_CODEC_REGISTRY,
    MappedPLETable,
    PLECodecDescriptor,
    convert_gguf_ple_to_artifact,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/gguf/qwen-host-layout.gguf"


def _artifact(tmp_path: Path) -> Path:
    output = tmp_path / "ple"
    convert_gguf_ple_to_artifact(FIXTURE, output)
    return output


def test_iq4_nl_codec_descriptor_and_registry_are_immutable() -> None:
    descriptor = IQ4_NL_CODEC_DESCRIPTOR

    assert descriptor.codec_id == "iq4_nl"
    assert descriptor.version == 1
    assert descriptor.packed_dtype == "uint8"
    assert descriptor.decoded_dtype == "float32"
    assert descriptor.elements_per_block == 32
    assert descriptor.bytes_per_block == 18
    assert descriptor.parameters["scale_dtype"] == "float16"
    assert PLE_CODEC_REGISTRY.resolve(descriptor).descriptor == descriptor

    with pytest.raises(TypeError):
        descriptor.parameters["scale_dtype"] = "float32"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        descriptor.codec_id = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        PLE_CODEC_REGISTRY.resolve(descriptor).descriptor = descriptor  # type: ignore[misc]


def test_descriptor_manifest_round_trip_preserves_codec_parameters() -> None:
    descriptor = IQ4_NL_CODEC_DESCRIPTOR
    restored = PLECodecDescriptor.from_manifest(descriptor.to_manifest())

    assert restored == descriptor
    assert descriptor.to_manifest() == {
        "id": "iq4_nl",
        "version": 1,
        "packed_dtype": "uint8",
        "decoded_dtype": "float32",
        "elements_per_block": 32,
        "bytes_per_block": 18,
        "parameters": {
            "codebook": "ggml-iq4-nl",
            "codebook_version": 1,
            "scale_dtype": "float16",
        },
    }


def test_extraction_is_reproducible_and_records_codec_identity(tmp_path: Path) -> None:
    first = _artifact(tmp_path / "first")
    second = _artifact(tmp_path / "second")

    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))

    assert first_manifest == second_manifest
    assert first_manifest["codec"] == IQ4_NL_CODEC_DESCRIPTOR.to_manifest()


def test_mapped_table_uses_codec_interface_and_reports_identity(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    with MappedPLETable.open_from_artifact(artifact) as table:
        expected = table.lookup(np.array([0, 31], dtype=np.int64))
        telemetry = table.telemetry()

    assert expected.shape == (2, 160)
    assert telemetry["codec_id"] == "iq4_nl"
    assert telemetry["codec_version"] == 1


def test_original_v1_artifact_without_codec_metadata_remains_readable(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["codec"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with MappedPLETable.open_from_artifact(artifact) as table:
        assert table.codec.descriptor == IQ4_NL_CODEC_DESCRIPTOR
        assert table.lookup(np.array([0])).shape == (1, 160)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "future_codec", "unknown.*codec"),
        ("version", 2, "version"),
        ("bytes_per_block", 19, "descriptor"),
    ],
)
def test_artifact_rejects_unknown_or_mismatched_codec(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    artifact = _artifact(tmp_path)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["codec"][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        MappedPLETable.open_from_artifact(artifact)


def test_artifact_rejects_decoder_shape_and_dtype_contract_violations(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    with MappedPLETable.open_from_artifact(artifact) as table:

        class BadCodec:
            descriptor = table.codec.descriptor

            def __init__(self, output: np.ndarray) -> None:
                self.output = output

            def decode(
                self,
                _packed: np.ndarray,
                *,
                rows: int,
                elements_per_row: int,
            ) -> np.ndarray:
                del rows, elements_per_row
                return self.output

        table.codec = BadCodec(np.empty((2, 159), dtype=np.float32))  # type: ignore[assignment]
        with pytest.raises(ValueError, match="decoder output shape"):
            table.lookup(np.array([0, 31], dtype=np.int64))

        table.codec = BadCodec(np.empty((2, 160), dtype=np.float16))  # type: ignore[assignment]
        with pytest.raises(ValueError, match="decoder output dtype"):
            table.lookup(np.array([0, 31], dtype=np.int64))
