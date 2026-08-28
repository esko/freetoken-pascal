from __future__ import annotations

import struct
from pathlib import Path

import gguf
import numpy as np
import pytest
from freetoken.gguf_shards import gguf_reader, gguf_shard_paths
from freetoken.gguf_validation import inspect_gguf


def _write_shards(directory: Path) -> tuple[Path, ...]:
    destination = directory / "tiny.gguf"
    writer = gguf.GGUFWriter(
        destination,
        "qwen4",
        split_max_tensors=1,
        small_first_shard=True,
    )
    writer.add_tensor("a.weight", np.arange(32, dtype=np.float32))
    writer.add_tensor("b.weight", np.arange(32, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return tuple(sorted(directory.glob("tiny-*-of-*.gguf")))


def _patch_scalar(path: Path, field_name: str, value: int, fmt: str) -> None:
    reader = gguf.GGUFReader(path)
    part = reader.fields[field_name].parts[-1]
    offset = int(part.ctypes.data - reader.data.ctypes.data)
    del reader
    with path.open("r+b") as output:
        output.seek(offset)
        output.write(struct.pack(fmt, value))
    gguf_reader.cache_clear()


def test_metadata_only_first_shard_and_tensor_shards_validate(tmp_path: Path) -> None:
    shards = _write_shards(tmp_path)

    assert len(shards) == 3
    result = inspect_gguf(shards[1])

    assert result["shard_count"] == 3
    assert [shard["tensor_count"] for shard in result["shards"]] == [0, 1, 1]
    assert [tensor["name"] for tensor in result["tensors"]] == ["a.weight", "b.weight"]
    assert gguf_shard_paths(shards[2]) == shards


def test_missing_shard_fails_closed(tmp_path: Path) -> None:
    shards = _write_shards(tmp_path)
    shards[1].unlink()

    with pytest.raises(ValueError, match="missing GGUF shard"):
        inspect_gguf(shards[0])


@pytest.mark.parametrize(
    ("field_name", "value", "message", "fmt"),
    [
        ("split.no", 9, "split.no", "<H"),
        ("split.count", 9, "split.count", "<H"),
        ("split.tensors.count", 9, "split.tensors.count", "<i"),
    ],
)
def test_inconsistent_split_metadata_fails_closed(
    tmp_path: Path,
    field_name: str,
    value: int,
    message: str,
    fmt: str,
) -> None:
    shards = _write_shards(tmp_path)
    _patch_scalar(shards[1], field_name, value, fmt)

    with pytest.raises(ValueError, match=message):
        inspect_gguf(shards[0])


def test_duplicate_tensor_name_across_shards_fails_closed(tmp_path: Path) -> None:
    shards = _write_shards(tmp_path)
    reader = gguf.GGUFReader(shards[2])
    name = reader.tensors[0].field.parts[1]
    offset = int(name.ctypes.data - reader.data.ctypes.data)
    del reader
    with shards[2].open("r+b") as output:
        output.seek(offset)
        output.write(b"a.weight")
    gguf_reader.cache_clear()

    with pytest.raises(ValueError, match="duplicate GGUF tensor name"):
        inspect_gguf(shards[0])
