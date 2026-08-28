"""Lightweight GGUF shard-set resolution with no torch dependency."""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

_SHARD_RE = re.compile(
    r"^(?P<stem>.+)-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.gguf$"
)


@functools.cache
def gguf_reader(path: str):
    import gguf

    return gguf.GGUFReader(path)


def field_value(reader, name: str) -> Any:
    field = reader.fields.get(name)
    return None if field is None else field.contents()


def _required_int_field(reader, name: str, path: Path) -> int:
    value = field_value(reader, name)
    if value is None:
        raise ValueError(f"{path}: split GGUF lacks {name}")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid {name} value {value!r}") from error


def gguf_shard_paths(model_path: str | Path) -> tuple[Path, ...]:
    """Resolve and validate the complete ordered shard set for ``model_path``."""
    source = Path(model_path)
    if not source.is_file() or source.suffix != ".gguf":
        raise ValueError(f"GGUF path is not a .gguf file: {source}")
    match = _SHARD_RE.match(source.name)
    if match is None:
        return (source,)

    count = int(match.group("count"))
    if count <= 1:
        raise ValueError(f"invalid GGUF split count {count} in {source.name}")
    stem = match.group("stem")
    paths = tuple(
        source.with_name(f"{stem}-{index:05d}-of-{count:05d}.gguf")
        for index in range(1, count + 1)
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing GGUF shard(s): {', '.join(missing)}")

    declared_total: int | None = None
    actual_total = 0
    for index, path in enumerate(paths):
        reader = gguf_reader(str(path))
        shard_no = _required_int_field(reader, "split.no", path)
        shard_count = _required_int_field(reader, "split.count", path)
        tensor_total = _required_int_field(reader, "split.tensors.count", path)
        if shard_no != index:
            raise ValueError(
                f"{path}: split.no {shard_no} does not match shard index {index}"
            )
        if shard_count != count:
            raise ValueError(
                f"{path}: split.count {shard_count} does not match filename count {count}"
            )
        if declared_total is None:
            declared_total = tensor_total
        elif tensor_total != declared_total:
            raise ValueError(
                f"{path}: split.tensors.count {tensor_total} does not match "
                f"{declared_total}"
            )
        actual_total += len(reader.tensors)
    if actual_total != declared_total:
        raise ValueError(
            f"GGUF shards contain {actual_total} tensors, expected {declared_total}"
        )
    return paths


__all__ = ["field_value", "gguf_reader", "gguf_shard_paths"]
