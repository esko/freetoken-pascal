"""Exact, schema-backed GGUF tensor and expert-layout census generation."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from freetoken.gguf_validation import inspect_gguf

_EXPERT_RE = re.compile(r"^blk\.(?P<layer>[0-9]+)\.ffn_(?P<projection>[^.]+)_exps\.weight$")


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def model_sha256(shards: list[dict[str, Any]]) -> str:
    """Canonical identity: payload SHA for one shard, manifest digest for many."""
    if len(shards) == 1:
        return str(shards[0]["sha256"])
    digest = hashlib.sha256()
    for shard in shards:
        digest.update(
            f"{shard['name']}\0{shard['size']}\0{shard['sha256']}\n".encode()
        )
    return digest.hexdigest()


def build_quant_census(
    path: str | Path,
    *,
    declared_shards: dict[str, dict[str, Any]] | None = None,
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Build a census, optionally using pinned declared identities for huge artifacts.

    Declared identities are never represented as measured hashes. Their sizes must
    still match the local logical files, enabling header-only sparse files to be
    audited without pretending their absent payload bytes were verified.
    """
    inspected = inspect_gguf(path)
    shard_identities: list[dict[str, Any]] = []
    for shard in inspected["shards"]:
        shard_path = Path(shard["path"])
        name = shard_path.name
        if declared_shards is None:
            identity = {
                "name": name,
                "size": shard["size"],
                "sha256": sha256_file(shard_path),
                "sha256_status": "verified",
            }
        else:
            declared = declared_shards.get(name)
            if declared is None:
                raise ValueError(f"no declared identity for GGUF shard {name}")
            if int(declared["size"]) != shard["size"]:
                raise ValueError(
                    f"{name}: local size {shard['size']} does not match declared "
                    f"{declared['size']}"
                )
            digest = str(declared["sha256"])
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"{name}: invalid declared sha256 {digest!r}")
            status = "declared"
            if verify_sha256:
                actual = sha256_file(shard_path)
                if actual != digest:
                    raise ValueError(
                        f"{name}: sha256 {actual} does not match declared {digest}"
                    )
                status = "verified"
            identity = {
                "name": name,
                "size": shard["size"],
                "sha256": digest,
                "sha256_status": status,
            }
        shard_identities.append(identity)

    by_quant: dict[str, dict[str, int]] = defaultdict(lambda: {"tensors": 0, "bytes": 0})
    pools: list[dict[str, Any]] = []
    layer_pools: dict[int, list[dict[str, Any]]] = defaultdict(list)
    tensors: list[dict[str, Any]] = []
    for tensor in inspected["tensors"]:
        quant_name = tensor["quant_name"]
        by_quant[quant_name]["tensors"] += 1
        by_quant[quant_name]["bytes"] += tensor["nbytes"]
        record = {
            key: tensor[key]
            for key in (
                "name",
                "shape",
                "quant_type",
                "quant_name",
                "shard_index",
                "offset",
                "nbytes",
                "rows",
                "row_bytes",
            )
        }
        tensors.append(record)
        match = _EXPERT_RE.match(tensor["name"])
        if match is not None:
            shape = tensor["shape"]
            if len(shape) < 2:
                raise ValueError(
                    f"expert tensor {tensor['name']} must have rank at least 2, got {shape}"
                )
            pool = {
                "layer": int(match.group("layer")),
                "projection": match.group("projection"),
                "tensor": tensor["name"],
                "experts": shape[0],
                "shape": shape,
                "quant_type": quant_name,
                "bytes": tensor["nbytes"],
            }
            pools.append(pool)
            layer_pools[pool["layer"]].append(pool)

    expert_layers = []
    for layer, layer_entries in sorted(layer_pools.items()):
        counts = {entry["experts"] for entry in layer_entries}
        if len(counts) != 1:
            raise ValueError(f"layer {layer} expert projections disagree on expert count")
        expert_layers.append(
            {
                "layer": layer,
                "experts": counts.pop(),
                "projections": sorted(entry["projection"] for entry in layer_entries),
                "quant_types": sorted({entry["quant_type"] for entry in layer_entries}),
            }
        )

    all_verified = all(
        shard["sha256_status"] == "verified" for shard in shard_identities
    )
    return {
        "schema_name": "quant-census.schema.json",
        "schema_version": 2,
        "evidence_status": "measured" if all_verified else "artifact-metadata",
        "architecture": inspected["architecture"],
        "model_sha256": model_sha256(shard_identities),
        "shard_count": len(shard_identities),
        "shards": shard_identities,
        "tensor_count": len(tensors),
        "total_bytes": sum(tensor["nbytes"] for tensor in tensors),
        "by_quant_type": dict(sorted(by_quant.items())),
        "expert_layers": expert_layers,
        "expert_pools": sorted(pools, key=lambda entry: (entry["layer"], entry["projection"])),
        "tensors": tensors,
    }


__all__ = ["build_quant_census", "model_sha256", "sha256_file"]
