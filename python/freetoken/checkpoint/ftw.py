"""FreeToken Weight (FTW) checkpoint: one O_DIRECT-friendly on-disk format for a whole model.

The format is a single *logical contiguous byte region* of all tensors, sliced *physically*
into shard files of at most ``shard_limit`` bytes (default 8 GiB, for HF/filesystem
friendliness). It exists because reading the original safetensors back fast is awkward:
tensors are packed with no alignment, so an individual tensor can't be O_DIRECT-read at an
arbitrary offset; and the earlier per-bank cache prototype worked around that by giving every expert bank
its own file -- which doesn't cover dense weights and turns a model's long tail of tiny
tensors (norms, biases, router) into hundreds of tiny I/Os.

FTW fixes both:

* **Aligned.** Every tensor starts at a 4096-aligned region offset and is padded to 4096;
  shards are cut at 4096-aligned boundaries. So any tensor (or any shard-local slice of one)
  is read with offset, length (rounded up to 4096), and destination all block-aligned --
  exactly what O_DIRECT requires. A tensor larger than a shard simply spans shards; because
  both its start and the shard boundary are aligned, each piece stays aligned.
* **Unified.** It holds dense weights as ``kind="weight"`` (exactly what a model's
  ``iter_weights`` yields -- post fusion/TP-shard, fed straight to ``load_state_dict``) and
  the offload expert state as ``kind="experts_bank"`` (post backend-repack -- the per-expert
  weight banks plus, distinguished only by their reserved names, the alpha scale vectors;
  the FTW content). The converter runs the per-model loaders once; this reader is
  model-agnostic.

Layout on disk::

    <dir>/freetoken_weight.json        # index: tensors[] + shards[] + meta
    <dir>/freetoken-00000.ftw         # the byte region, sliced <= shard_limit
    <dir>/freetoken-00001.ftw
    <dir>/config.json, tokenizer*, ...# copied so the dir is a self-contained checkpoint
"""

from __future__ import annotations

import json
import math
import mmap
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

INDEX_NAME = "freetoken_weight.json"
FORMAT_TAG = "freetoken_weight"
FORMAT_VERSION = 1
ALIGN = 4096  # O_DIRECT block alignment (== page size on this platform)
DEFAULT_SHARD_LIMIT = 8 << 30  # 8 GiB; must be a multiple of ALIGN
_SHARD_FMT = "freetoken-{:05d}.ftw"
_DEFAULT_CHUNK = 8 << 20
_BANK_CONCURRENCY = 4
_ALPHA_NAMES = ("gate_up_alpha", "down_alpha")
# Per-layer expert-bank entry name (converter streaming path, see checkpoint/convert.py):
# each layer of a bank is its own FTW tensor instead of one flat [num_layers*E, ...] region.
_LAYER_ENTRY_RE = re.compile(r"^(?P<base>.+)#L(?P<layer>\d{5})$")


def layer_bank_entry_name(bank_name: str, layer_id: int) -> str:
    """Name of one per-layer ``experts_bank`` FTW entry; :func:`load_ftw_banks` groups
    entries matching ``_LAYER_ENTRY_RE`` back into a per-layer bank list by base name."""
    return f"{bank_name}#L{layer_id:05d}"


def _pread_into(fd: int, mv: memoryview, offset: int) -> None:
    """POSIX positional read into ``mv`` at ``offset``, looping over any short preadv.

    preadv may return short (a signal, or the EOF-adjacent tail); the loop resumes
    at the running offset, which stays O_DIRECT-legal: the writer pads every tensor
    to ALIGN and cuts shards at ALIGN boundaries, so direct-IO short reads land on
    block boundaries. EOF before the buffer is filled raises ``OSError`` — a
    truncated shard must not silently load garbage weights."""
    done = 0
    total = len(mv)
    while done < total:
        n = os.preadv(fd, [mv[done:]], offset + done)
        if n == 0:
            raise OSError(
                f"unexpected EOF reading FTW: got {done}/{total} bytes at offset {offset}"
            )
        done += n


def _align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def _dtype_str(dt: torch.dtype) -> str:
    return str(dt).removeprefix("torch.")


def _dtype_of(s: str) -> torch.dtype:
    return getattr(torch, s)


def _elsize(dt: torch.dtype) -> int:
    return torch.empty((), dtype=dt).element_size()


def _index_int(index: dict, key: str, *, minimum: int | None = None) -> int | None:
    """Read an optional FTW integer without accepting JSON booleans."""
    value = index.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"FTW index has invalid {key}: expected an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"FTW index has invalid {key}: must be >= {minimum}")
    return value


def _validate_ftw_index(path: str, index, *, strict_storage: bool) -> int | None:
    """Validate one parsed FTW index snapshot and return its logical byte length.

    ``strict_storage`` is false only for the legacy metadata-only policy probe.  The
    reader and the loader always use strict validation, so they never allocate from an
    index whose shard geometry has not also been checked.
    """
    if not isinstance(index, dict):
        raise ValueError(f"FTW index must be an object: {path}")
    if index.get("format") != FORMAT_TAG:
        raise ValueError(f"not a {FORMAT_TAG}: {path}")
    version = _index_int(index, "version", minimum=0)
    if strict_storage and version is None:
        raise ValueError("FTW index must contain version")
    if version is not None and version != FORMAT_VERSION:
        raise ValueError(f"unsupported FTW index version {version}; expected {FORMAT_VERSION}")
    align = _index_int(index, "align", minimum=1)
    if strict_storage and align is None:
        raise ValueError("FTW index must contain align")
    if align is not None and align != ALIGN:
        raise ValueError(f"unsupported FTW alignment {align}; expected {ALIGN}")
    shard_limit = _index_int(index, "shard_limit", minimum=1)
    if strict_storage and shard_limit is None:
        raise ValueError("FTW index must contain shard_limit")
    if shard_limit is not None and shard_limit % ALIGN:
        raise ValueError(f"FTW shard_limit must be a positive {ALIGN}-byte multiple")
    total_bytes = _index_int(index, "total_bytes", minimum=0)
    _index_int(index, "expert_bank_num_layers", minimum=1)

    shards = index.get("shards")
    if shards is None:
        raise ValueError("FTW index must contain a shard list")
    else:
        if not isinstance(shards, list):
            raise ValueError("FTW index shards must be a list")
        if strict_storage and not shards:
            raise ValueError("FTW index must contain a non-empty shard list")
        if any(not isinstance(shard, dict) for shard in shards):
            raise ValueError("FTW shard entries must be objects")
        validated_shards = []
        files: set[str] = set()
        for shard in shards:
            file = shard.get("file")
            global_off = _index_int(shard, "global_off", minimum=0)
            nbytes = _index_int(shard, "nbytes", minimum=1)
            if not isinstance(file, str) or not file or file != os.path.basename(file):
                raise ValueError("FTW shard file must be a canonical basename")
            if global_off is None or nbytes is None:
                raise ValueError("FTW shard entries require global_off and nbytes")
            if file in (".", ".."):
                raise ValueError("FTW shard file must name a regular file")
            if file in files:
                raise ValueError(f"FTW shard file {file!r} is duplicated")
            files.add(file)
            if global_off % ALIGN:
                raise ValueError(f"FTW shard {file!r} has an unaligned global_off")
            if nbytes % ALIGN:
                raise ValueError(f"FTW shard {file!r} has an unaligned nbytes")
            if shard_limit is not None and nbytes > shard_limit:
                raise ValueError(f"FTW shard {file!r} exceeds shard_limit")
            validated_shards.append((global_off, nbytes, file))
        logical_end = 0
        for global_off, nbytes, file in sorted(validated_shards):
            if global_off != logical_end:
                raise ValueError(
                    "FTW shards must form one contiguous logical range; a gap or overlap "
                    "was found"
                )
            try:
                physical_bytes = os.stat(os.path.join(path, file)).st_size
            except OSError as exc:
                raise ValueError(f"FTW shard {file!r} is unavailable: {exc}") from exc
            if physical_bytes != nbytes:
                raise ValueError(
                    f"FTW shard {file!r} size mismatch: index={nbytes}, file={physical_bytes}"
                )
            logical_end += nbytes
        if strict_storage:
            if total_bytes is None:
                raise ValueError("FTW index must contain total_bytes")
            if total_bytes != logical_end:
                raise ValueError(
                    f"FTW total_bytes mismatch: index={total_bytes}, shards={logical_end}"
                )
        elif not validated_shards:
            logical_end = None

    tensors = index.get("tensors")
    if not isinstance(tensors, list):
        raise ValueError("FTW tensor entries must be an object list")
    names: set[str] = set()
    ranges: list[tuple[int, int, str]] = []
    for entry in tensors:
        if not isinstance(entry, dict):
            raise ValueError("FTW tensor entries must be objects")
        name = entry.get("name")
        kind = entry.get("kind")
        dtype_name = entry.get("dtype")
        shape = entry.get("shape")
        global_off = _index_int(entry, "global_off", minimum=0)
        nbytes = _index_int(entry, "nbytes", minimum=1)
        if not isinstance(name, str) or not name:
            raise ValueError("FTW tensor name must be a non-empty string")
        if global_off is None or nbytes is None:
            raise ValueError(f"FTW tensor {name!r} requires global_off and nbytes")
        if name in names:
            if kind == "experts_bank":
                layer_match = _LAYER_ENTRY_RE.match(name)
                if layer_match is not None:
                    raise ValueError(
                        f"FTW bank {layer_match.group('base')!r} has duplicate layer "
                        f"{int(layer_match.group('layer'))}"
                    )
                raise ValueError("FTW expert banks contain duplicate NUMA sample identity")
            raise ValueError(f"FTW tensor {name!r} is duplicated")
        names.add(name)
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"FTW tensor {name!r} kind must be a non-empty string")
        if not isinstance(dtype_name, str):
            raise ValueError(f"FTW tensor {name!r} dtype must be a string")
        try:
            dtype = _dtype_of(dtype_name)
            if not isinstance(dtype, torch.dtype):
                raise TypeError
            elsize = _elsize(dtype)
        except (AttributeError, TypeError, RuntimeError) as exc:
            raise ValueError(f"FTW tensor {name!r} has an invalid dtype") from exc
        if not isinstance(shape, list) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape
        ):
            # Empty shape is a valid scalar tensor; all dimensions that are present must
            # nevertheless be positive exact integers.
            if shape != []:
                raise ValueError(f"FTW tensor {name!r} has an invalid shape")
        if math.prod(shape) * elsize != nbytes:
            raise ValueError(f"FTW tensor {name!r} byte count does not match its shape")
        if global_off % ALIGN:
            raise ValueError(f"FTW tensor {name!r} has an unaligned global_off")
        end = global_off + _align_up(nbytes)
        ranges.append((global_off, end, name))
        if logical_end is not None and end > logical_end:
            raise ValueError(f"FTW tensor {name!r} exceeds shard ranges")
    prior_end = -1
    for global_off, end, name in sorted(ranges):
        if global_off < prior_end:
            raise ValueError(f"FTW tensor {name!r} overlaps another tensor range")
        prior_end = end
    return logical_end


def prepare_ftw_host_bank_policy(
    path: str,
    *,
    num_layers: int,
    policy,
    swap_probe=None,
    _index: dict | None = None,
):
    """Validate every FTW expert-bank identity and allocation from index metadata.

    Validation is independent of whether a host-bank policy was requested so the
    legacy/default loader cannot allocate from an ambiguous index.  When ``policy``
    is present, the same pass also prepares its bounded allocation plan.
    """
    from freetoken.moe.numa_memory import NumaSampleIdentity

    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError(f"num_layers must be a positive integer, got {num_layers}")
    # Swap admission is intentionally before opening the index: an explicit
    # no-swap policy must not inspect checkpoint metadata before it can fail.
    if policy is not None and policy.require_no_swap:
        swap_probe = policy.preflight_swap(probe=swap_probe)
    if _index is None:
        with open(os.path.join(path, INDEX_NAME), encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = _index
    logical_end = _validate_ftw_index(path, index, strict_storage=_index is not None)
    meta_layers = index.get("expert_bank_num_layers")
    if meta_layers is not None and meta_layers != num_layers:
        raise RuntimeError(
            f"{path!r} was converted with {meta_layers} expert-bank layers but the "
            f"model config says num_moe_layers={num_layers}; the checkpoint does not "
            "match its config"
        )
    tensors = index["tensors"]
    entries = [entry for entry in tensors if entry.get("kind") == "experts_bank"]
    if not entries and policy is None:
        # Preserve the legacy probe contract: a valid FTW checkpoint without
        # expert banks is not an expert-bank bundle.
        return None
    if not entries:
        raise ValueError(f"{path!r} contains no experts_bank entries")
    for entry in entries:
        shape = entry.get("shape")
        global_off = entry.get("global_off")
        nbytes = entry.get("nbytes")
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                for dim in shape
            )
        ):
            raise ValueError(f"FTW expert bank {entry.get('name')!r} has an invalid shape")
        if (
            isinstance(global_off, bool)
            or not isinstance(global_off, int)
            or global_off < 0
            or global_off % ALIGN
        ):
            raise ValueError(
                f"FTW expert bank {entry.get('name')!r} has an invalid global_off"
            )
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0:
            raise ValueError(f"FTW expert bank {entry.get('name')!r} has invalid nbytes")
        if logical_end is not None and global_off + _align_up(nbytes) > logical_end:
            raise ValueError(f"FTW expert bank {entry.get('name')!r} exceeds shard ranges")
    alpha_entries = [entry for entry in entries if entry["name"] in _ALPHA_NAMES]
    row_entries = [entry for entry in entries if entry["name"] not in _ALPHA_NAMES]
    flat_entries = []
    per_layer_groups: dict[str, dict[int, dict]] = {}
    for entry in row_entries:
        match = _LAYER_ENTRY_RE.match(entry["name"])
        if match is None:
            flat_entries.append(entry)
            continue
        by_layer = per_layer_groups.setdefault(match.group("base"), {})
        layer_id = int(match.group("layer"))
        if layer_id in by_layer:
            raise ValueError(
                f"FTW bank {match.group('base')!r} has duplicate layer {layer_id}"
            )
        by_layer[layer_id] = entry
    mixed = set(entry["name"] for entry in flat_entries) & per_layer_groups.keys()
    if mixed:
        raise ValueError(f"FTW bank(s) mix flat and per-layer row layouts: {sorted(mixed)}")

    sample_identities: set[NumaSampleIdentity] = set()

    def register_sample_identity(bank_name: str, layer_id: int | None) -> None:
        identity = NumaSampleIdentity(bank_name, layer_id)
        if identity in sample_identities:
            raise ValueError(
                "FTW expert banks contain duplicate NUMA sample identity "
                f"{identity.as_dict()}"
            )
        sample_identities.add(identity)

    for entry in alpha_entries:
        register_sample_identity(entry["name"], None)
    for entry in flat_entries:
        for layer_id in range(num_layers):
            register_sample_identity(entry["name"], layer_id)
    for base, by_layer in per_layer_groups.items():
        for layer_id in by_layer:
            register_sample_identity(base, layer_id)

    layer_bytes = [0] * num_layers
    for entry in alpha_entries:
        shape = tuple(entry["shape"])
        dtype = _dtype_of(entry["dtype"])
        expected = math.prod(shape) * _elsize(dtype)
        if expected != int(entry["nbytes"]):
            raise ValueError(f"FTW alpha {entry['name']!r} byte count is invalid")
        if int(entry["global_off"]) % ALIGN:
            raise ValueError(f"FTW alpha {entry['name']!r} offset is not {ALIGN}-byte aligned")
        layer_bytes[0] += max(ALIGN, _align_up(expected))
    for entry in flat_entries:
        shape = tuple(entry["shape"])
        if not shape:
            raise ValueError(f"FTW expert bank {entry['name']!r} has an empty shape")
        total, *row_shape = shape
        if total % num_layers:
            raise ValueError(
                f"FTW bank {entry['name']!r} has {total} rows, not divisible by "
                f"{num_layers} layers"
            )
        dtype = _dtype_of(entry["dtype"])
        if int(entry["global_off"]) % ALIGN:
            raise ValueError(f"FTW bank {entry['name']!r} offset is not {ALIGN}-byte aligned")
        row_bytes = (math.prod(row_shape) if row_shape else 1) * _elsize(dtype)
        per_layer = (total // num_layers) * row_bytes
        if per_layer * num_layers != int(entry["nbytes"]):
            raise ValueError(f"FTW bank {entry['name']!r} byte count does not match its shape")
        for layer_id in range(num_layers):
            off = int(entry["global_off"]) + layer_id * per_layer
            layer_bytes[layer_id] += _align_up(off + per_layer) - (off // ALIGN) * ALIGN
    for base, by_layer in per_layer_groups.items():
        if sorted(by_layer) != list(range(num_layers)):
            raise ValueError(
                f"FTW bank {base!r} has per-layer entries for layers {sorted(by_layer)}, "
                f"expected exactly range({num_layers})"
            )
        for layer_id in range(num_layers):
            entry = by_layer[layer_id]
            if int(entry["global_off"]) % ALIGN:
                raise ValueError(
                    f"FTW bank {base!r} layer {layer_id} offset is not {ALIGN}-byte aligned"
                )
            dtype = _dtype_of(entry["dtype"])
            expected = math.prod(tuple(entry["shape"])) * _elsize(dtype)
            if expected != int(entry["nbytes"]):
                raise ValueError(f"FTW bank {base!r} layer {layer_id} byte count is invalid")
            layer_bytes[layer_id] += max(ALIGN, _align_up(expected))
    if policy is None:
        return None
    return policy.prepare_layer_bytes(layer_bytes, swap_probe=swap_probe)


def is_ftw_checkpoint(path: str) -> bool:
    """True if ``path`` is a directory holding a FreeToken Weight (FTW) index."""
    return os.path.isfile(os.path.join(path, INDEX_NAME))


# ============================== writer ==============================
class FTWWriter:
    """Stream tensors into the FTW, rolling shard files at ``shard_limit``.

    Tensors are written in call order into one logical byte stream; each is padded to
    ``ALIGN`` so the next starts aligned. A tensor that doesn't fit the current shard's
    remaining room is split across shards (the split point is the shard boundary, which is
    aligned). Call :meth:`add_tensor` for each tensor, then :meth:`finalize`.
    """

    def __init__(self, out_dir: str, *, shard_limit: int = DEFAULT_SHARD_LIMIT):
        assert shard_limit % ALIGN == 0, "shard_limit must be a multiple of ALIGN"
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.shard_limit = shard_limit
        self._tensors: list[dict] = []
        self._shards: list[dict] = []
        self._global = 0  # running FTW offset (incl. padding)
        self._f = None  # current shard file handle
        self._shard_idx = -1
        self._shard_start = 0  # FTW offset where the current shard began
        self._cur = 0  # bytes written to the current shard

    def _roll(self) -> None:
        if self._f is not None:
            self._shards.append({"file": _SHARD_FMT.format(self._shard_idx),
                                 "global_off": self._shard_start, "nbytes": self._cur})
            self._f.close()
        self._shard_idx += 1
        self._shard_start = self._global
        self._cur = 0
        self._f = open(os.path.join(self.out_dir, _SHARD_FMT.format(self._shard_idx)), "wb")

    def _write_raw(self, data: memoryview) -> None:
        """Write ``data`` into the FTW byte stream, splitting across shards at the limit."""
        if self._f is None:
            self._roll()
        off = 0
        n = len(data)
        while off < n:
            if self._cur == self.shard_limit:
                self._roll()
            take = min(n - off, self.shard_limit - self._cur)
            self._f.write(data[off:off + take])
            off += take
            self._cur += take
            self._global += take

    def add_tensor(self, name: str, tensor: torch.Tensor, kind: str = "weight") -> None:
        t = tensor.detach().cpu().contiguous()
        raw = t.reshape(-1).view(torch.uint8)
        nbytes = int(raw.numel())
        # A small tensor (<= shard) never splits: roll early so it lands whole in one shard.
        if self._f is None or (nbytes <= self.shard_limit
                               and self._cur + nbytes > self.shard_limit):
            self._roll()
        global_off = self._global
        assert global_off % ALIGN == 0, "tensor start must be aligned (invariant)"
        self._write_raw(memoryview(raw.numpy()))
        self._tensors.append({"name": name, "kind": kind, "dtype": _dtype_str(t.dtype),
                              "shape": list(t.shape), "global_off": global_off, "nbytes": nbytes})
        # pad to ALIGN so the next tensor starts aligned
        pad = _align_up(self._global) - self._global
        if pad:
            self._write_raw(memoryview(bytes(pad)))

    def finalize(self, meta: dict) -> dict:
        if self._f is not None:
            self._shards.append({"file": _SHARD_FMT.format(self._shard_idx),
                                 "global_off": self._shard_start, "nbytes": self._cur})
            self._f.close()
            self._f = None
        index = {"format": FORMAT_TAG, "version": FORMAT_VERSION, "align": ALIGN,
                 "shard_limit": self.shard_limit, "total_bytes": self._global,
                 "tensors": self._tensors, "shards": self._shards, **meta}
        tmp = os.path.join(self.out_dir, INDEX_NAME + ".tmp")
        with open(tmp, "w") as f:
            json.dump(index, f)
        os.replace(tmp, os.path.join(self.out_dir, INDEX_NAME))
        return index


# ============================== reader ==============================
class FTWReader:
    """Random-access reader over an FTW checkpoint.

    Maps a tensor's logical byte range to one-or-more shard-file ranges (split at shard
    boundaries) and reads each piece with chunked multi-threaded O_DIRECT directly into the
    destination buffer. Offsets/lengths are all 4096-aligned (lengths rounded up into the
    rounded-up destination), so O_DIRECT is always legal -- including the tail of a tensor
    (the rounding reads into the region's padding, which is discarded by the tensor view)."""

    def __init__(self, path: str):
        with open(os.path.join(path, INDEX_NAME)) as f:
            self.index = json.load(f)
        # Validate the parsed object before deriving sorted views or opening any shard.
        # The loader later validates this same ``self.index`` object with its exact model
        # layer count; reopening the JSON there would reintroduce a TOCTOU window.
        _validate_ftw_index(path, self.index, strict_storage=True)
        self.dir = path
        self.shards = sorted(self.index["shards"], key=lambda s: s["global_off"])
        self.tensors = {t["name"]: t for t in self.index["tensors"]}
        self._fds: dict[str, int] = {}
        self._maps: dict[str, tuple[mmap.mmap, memoryview]] = {}
        # O_DIRECT (DMA straight from disk, bypassing the page cache) is the fast path but a
        # perf choice, not a correctness one. Some filesystems reject it at open with EINVAL
        # (tmpfs, many overlay/network mounts) and the flag is Linux-only; when it's absent
        # we fall back to mmap (below), NOT to chunked buffered preadv -- a whole-shard
        # mapping + kernel readahead copies far faster than per-chunk page-cache reads.
        # 0 here means "O_DIRECT unavailable -> use the mmap path".
        self._direct = getattr(os, "O_DIRECT", 0)
        self._probed = False
        self._lock = threading.Lock()  # load_ftw_banks calls read_into concurrently

    def meta(self, key: str, default=None):
        return self.index.get(key, default)

    def entries(self, *kinds: str) -> list[dict]:
        keep = set(kinds)
        return [t for t in self.index["tensors"] if not keep or t["kind"] in keep]

    def _ensure_mode(self) -> None:
        """Resolve the read backend once: keep O_DIRECT if the filesystem accepts it, else
        drop to the mmap fallback. Thread-safe -- ``_probed`` is published only after
        ``_direct`` is final, so a concurrent reader never races onto a stale direct path."""
        if self._probed:
            return
        with self._lock:
            if self._probed:
                return
            if self._direct and self.shards:
                try:
                    os.close(os.open(os.path.join(self.dir, self.shards[0]["file"]),
                                     os.O_RDONLY | self._direct))
                except OSError:
                    self._direct = 0
                    logger.warning("O_DIRECT unsupported on %s; using mmap fallback for "
                                   "FTW load", self.dir)
            self._probed = True

    def _fd(self, file: str) -> int:
        fd = self._fds.get(file)
        if fd is None:
            with self._lock:  # first-open only; chunk reads reuse the cached fd lock-free
                fd = self._fds.get(file)
                if fd is None:
                    fd = os.open(os.path.join(self.dir, file), os.O_RDONLY | self._direct)
                    self._fds[file] = fd
        return fd

    def _map(self, file: str) -> memoryview:
        entry = self._maps.get(file)
        if entry is None:
            with self._lock:
                entry = self._maps.get(file)
                if entry is None:
                    fd = os.open(os.path.join(self.dir, file), os.O_RDONLY)
                    try:
                        m = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
                    finally:
                        os.close(fd)  # the mapping keeps its own reference to the file
                    try:
                        m.madvise(mmap.MADV_SEQUENTIAL)  # kernel readahead for streaming
                    except (AttributeError, OSError):
                        pass
                    entry = (m, memoryview(m))
                    self._maps[file] = entry
        return entry[1]

    def close(self) -> None:
        first_error: BaseException | None = None
        for fd in tuple(self._fds.values()):
            try:
                os.close(fd)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._fds.clear()
        for m, mv in tuple(self._maps.values()):
            try:
                mv.release()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            try:
                m.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._maps.clear()
        if first_error is not None:
            raise first_error

    def _pieces(self, global_off: int, nbytes: int):
        """Yield (file, file_off, dest_off, length) covering [global_off, +nbytes),
        split at shard boundaries. All file_off/dest_off are ALIGN-aligned."""
        dest_off = 0
        remaining = nbytes
        pos = global_off
        for sh in self.shards:
            s0, s1 = sh["global_off"], sh["global_off"] + sh["nbytes"]
            if pos >= s1 or remaining <= 0:
                continue
            if pos < s0:  # regions are contiguous; a gap means a corrupt index
                raise ValueError("FTW gap / misordered shards")
            take = min(remaining, s1 - pos)
            yield sh["file"], pos - s0, dest_off, take
            pos += take
            dest_off += take
            remaining -= take
        if remaining:
            raise ValueError("tensor range exceeds FTW shards")

    def read_into(self, dest: memoryview, entry: dict, *, workers: int = 8,
                  chunk: int = _DEFAULT_CHUNK) -> None:
        """Read one tensor's bytes into ``dest`` (length >= entry nbytes rounded to ALIGN)."""
        self._ensure_mode()
        jobs = []  # (file, file_off, dest_off, length) all ALIGN-aligned
        for file, file_off, dest_off, length in self._pieces(entry["global_off"], entry["nbytes"]):
            rlen = _align_up(length)  # round the tail up; padding is in-region, harmless
            for c in range(0, rlen, chunk):
                jobs.append((file, file_off + c, dest_off + c, min(chunk, rlen - c)))

        # Open/map each distinct shard once, single-threaded, so the pool only reuses handles.
        touch = self._fd if self._direct else self._map
        for file in {j[0] for j in jobs}:
            touch(file)

        if self._direct:
            def rd(job):
                file, fo, do, ln = job
                try:
                    _pread_into(self._fd(file), dest[do:do + ln], fo)
                except OSError as e:
                    raise OSError(f"shard {file}: {e}") from e
        else:
            def rd(job):
                file, fo, do, ln = job
                mv = self._map(file)
                if fo + ln > len(mv):
                    raise OSError(
                        f"unexpected EOF reading FTW: shard {file} has "
                        f"{len(mv)} bytes, need {ln} at offset {fo}"
                    )
                dest[do:do + ln] = mv[fo:fo + ln]

        if len(jobs) <= 1:
            for j in jobs:
                rd(j)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, jobs))


def _transient_buffer(nbytes: int) -> mmap.mmap:
    return mmap.mmap(-1, _align_up(nbytes))


def iter_ftw_weights(path: str, *, kinds=("weight",), workers: int = 8,
                       chunk: int = _DEFAULT_CHUNK, prefetch: int = 2):
    """Yield ``(name, host_tensor)`` for the requested kinds, reading each tensor via
    chunked O_DIRECT. A background thread prefetches the next ``prefetch`` tensors so the
    disk stays busy while the consumer copies the current one to the GPU. Transient buffers
    are freed as the consumer advances (peak host mem ~ prefetch+1 tensors)."""
    import queue
    import threading

    from freetoken.utils.progress import byte_bar

    reader = FTWReader(path)
    entries = reader.entries(*kinds)
    q: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
    _DONE = object()
    err: list[BaseException] = []
    cancel = threading.Event()

    def _put(item) -> bool:
        # A plain q.put would deadlock teardown: if the consumer stops with the queue
        # full (early break out of the generator, or an exception mid-load), close()
        # runs the finally below, which joins this thread while it waits for queue
        # space forever. Poll the cancel flag instead of blocking indefinitely.
        while not cancel.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _producer():
        try:
            for e in entries:
                buf = _transient_buffer(e["nbytes"])
                reader.read_into(memoryview(buf), e, workers=workers, chunk=chunk)
                dt = _dtype_of(e["dtype"])
                t = torch.frombuffer(buf, dtype=dt, count=e["nbytes"] // _elsize(dt))
                if not _put((e["name"], t.view(*e["shape"]) if e["shape"] else t, buf, e["nbytes"])):
                    return
        except BaseException as ex:  # surface to consumer
            err.append(ex)
        finally:
            _put(_DONE)

    th = threading.Thread(target=_producer, name="FTW-prefetch", daemon=True)
    th.start()
    bar = byte_bar(sum(e["nbytes"] for e in entries), "Loading weights (FTW)")
    try:
        while True:
            item = q.get()
            if item is _DONE:
                break
            name, tensor, buf, nbytes = item
            yield name, tensor
            bar.update(nbytes)
            del tensor, buf  # buffer reclaimable once the consumer drops the tensor
    finally:
        bar.close()
        cancel.set()
        th.join()
        reader.close()
    if err:
        raise err[0]


def load_ftw_banks(
    path: str, *, num_layers: int, workers: int = 8, chunk: int = _DEFAULT_CHUNK,
    layer_residency: list[str] | None = None,
    host_bank_policy=None,
    swap_probe=None,
):
    """Reconstruct the offload :class:`ExpertBanks` from the FTW's ``experts_bank``
    entries, on the per-layer host bank contract (one ``[num_experts, ...]``
    HostBank per layer per bank; see ``moe.offload_cache.set_bank_sources``).

    ``layer_residency`` (default: all pinned) settles each layer's banks per its ``HostResidency`` label as reads complete: PINNED -> cudaHostRegister, LOCKED -> mlock (CPU-executor resident, no pin quota spent).
    The applied labels are echoed back on ``ExpertBanks.layer_residency``.

    Two on-disk row layouts, distinguished per bank name (a file never mixes them for
    the same name -- checked below):

    * **Flat region** (pre-existing files, and non-streamable formats): one entry per
      bank, ONE contiguous ``[num_layers * num_experts, ...]`` region. ``num_layers``
      isn't part of that region's shape, so the caller passes it
      (``ModelConfig.num_moe_layers`` -- FTW checkpoints carry the model's config.json);
      the ``expert_bank_num_layers`` index meta the converter records is used as a
      cross-check when present. A layer's byte range within the region generally is
      NOT 4096-aligned (only the whole region's start is guaranteed aligned) -- read it
      via its ALIGNED enclosing window ``[align_down(off), align_up(off+len))`` into a
      page-aligned scratch HostBank, and view the real per-layer tensor as a
      head-offset slice.
    * **Per-layer** (streamable-format conversion, see :mod:`freetoken.checkpoint.convert`):
      one entry per ``(bank, layer)``, name ``f"{bank_name}#L{layer_id:05d}"``. Each was
      written by its own ``add_tensor`` call, so its start is already ALIGN-aligned --
      no windowing/head-pad needed, read straight into a HostBank shaped like the entry.

    Alphas (``gate_up_alpha``/``down_alpha``) stay flat ``[num_layers*num_experts]``
    vectors, unaffected by the row split (fixed GPU residency; see
    ``cache_budget.expert_bytes_per_slot``).
    """
    from freetoken.moe.host_banks import HostBank, HostResidency, PinPipeline, born_pinned_default
    from freetoken.moe.numa_memory import NumaSampleIdentity
    from freetoken.utils.progress import byte_bar

    if host_bank_policy is not None and layer_residency is not None:
        raise ValueError("host_bank_policy and layer_residency cannot be combined")
    if layer_residency is not None and len(layer_residency) != num_layers:
        raise ValueError(
            f"layer_residency has {len(layer_residency)} entries, expected {num_layers}"
        )
    reader = None
    bar = None
    staging_ring = None
    committed = False
    cleanup_errors: list[BaseException] = []
    closed_resources: set[int] = set()
    sources: dict[str, list[torch.Tensor]] = {}
    alpha_kw: dict[str, torch.Tensor] = {}
    views: list[torch.Tensor] = []
    raw = None

    def _safe_close(resource, label: str) -> None:
        if resource is None:
            return
        resource_id = id(resource)
        if resource_id in closed_resources:
            return
        closed_resources.add(resource_id)
        try:
            resource.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
            logger.warning(f"{label} cleanup failed: {exc}")

    def _clear_derived_views() -> None:
        nonlocal raw
        sources.clear()
        alpha_kw.clear()
        views.clear()
        raw = None

    if host_bank_policy is not None and host_bank_policy.require_no_swap:
        # Preserve the admission ordering: inspect swap before opening checkpoint
        # metadata, then pass that exact observation into policy preparation.
        swap_probe = host_bank_policy.preflight_swap(probe=swap_probe)
    try:
        reader = FTWReader(path)
        prepare_ftw_host_bank_policy(
            path,
            num_layers=num_layers,
            policy=host_bank_policy,
            swap_probe=swap_probe,
            _index=reader.index,
        )
    except BaseException:
        _safe_close(reader, "FTW reader")
        raise
    if host_bank_policy is not None:
        residency = list(host_bank_policy.plan.layer_residency)
    else:
        residency = (
            [HostResidency.PINNED.value] * num_layers
            if layer_residency is None
            else list(layer_residency)
        )
    try:
        if (
            host_bank_policy is not None
            and host_bank_policy.strategy.value == "bounded-staging"
        ):
            staging_ring = host_bank_policy.staging_ring()
    except BaseException:
        _safe_close(reader, "FTW reader")
        raise

    allocated_banks: list[tuple[NumaSampleIdentity, HostBank]] = []
    numa_placement = (
        host_bank_policy.numa_placement if host_bank_policy is not None else None
    )

    def _new_bank(shape, dtype, backing, *, bank_name: str, layer_id: int | None):
        try:
            identity = NumaSampleIdentity(bank_name, layer_id)
            bank = HostBank(
                shape,
                dtype,
                backing=backing,
                numa_placement=numa_placement if backing == "mmap" else None,
            )
        except BaseException:
            # Allocation happens after the FTW reader is opened but before the
            # read-job ``try`` block; close it here so a failed NUMA mapping
            # cannot strand the descriptor while rolling back prior banks.
            _cleanup_owned()
            raise
        allocated_banks.append((identity, bank))
        return bank

    def _rollback_allocations() -> None:
        for _identity, bank in reversed(allocated_banks):
            try:
                bank.close()
            except BaseException as exc:
                logger.warning(f"FTW host-bank rollback failed: {exc}")
        if staging_ring is not None:
            try:
                staging_ring.close()
            except BaseException as exc:
                logger.warning(f"FTW staging-ring rollback failed: {exc}")

    def _cleanup_owned() -> None:
        """Drop aliases and release every resource owned by this acquisition."""
        _clear_derived_views()
        _rollback_allocations()
        _safe_close(bar, "FTW progress bar")
        _safe_close(reader, "FTW reader")

    # PINNED layers are born-pinned (cudaHostAlloc) where that wins (see
    # born_pinned_default); LOCKED/PAGEABLE layers stay lazy mmaps.
    try:
        born = host_bank_policy is None and born_pinned_default()
        bank_entries = reader.entries("experts_bank")
    except BaseException:
        _cleanup_owned()
        raise

    def _backing(layer_id: int) -> str:
        if born and residency[layer_id] == HostResidency.PINNED.value:
            return "cuda"
        return "mmap"

    if not bank_entries:
        _safe_close(reader, "FTW reader")
        if cleanup_errors:
            raise cleanup_errors[0]
        return None

    alpha_entries = [e for e in bank_entries if e["name"] in _ALPHA_NAMES]
    row_entries = [e for e in bank_entries if e["name"] not in _ALPHA_NAMES]

    meta_layers = reader.meta("expert_bank_num_layers")
    if meta_layers is not None and meta_layers != num_layers:
        _cleanup_owned()
        raise RuntimeError(
            f"{path!r} was converted with {meta_layers} expert-bank layers but the "
            f"model config says num_moe_layers={num_layers}; the checkpoint does not "
            "match its config"
        )

    # Alphas: unchanged, one flat HostBank per entry.
    alpha_specs = {e["name"]: (tuple(e["shape"]), _dtype_of(e["dtype"])) for e in alpha_entries}
    alpha_hb = {
        name: _new_bank(shape, dtype, _backing(0), bank_name=name, layer_id=None)
        for name, (shape, dtype) in alpha_specs.items()
    }

    # Split row entries into the two layouts by name.
    flat_entries: list[dict] = []
    per_layer_groups: dict[str, dict[int, dict]] = {}
    for e in row_entries:
        m = _LAYER_ENTRY_RE.match(e["name"])
        if m is None:
            flat_entries.append(e)
            continue
        per_layer_groups.setdefault(m.group("base"), {})[int(m.group("layer"))] = e

    mixed = {e["name"] for e in flat_entries} & per_layer_groups.keys()
    assert not mixed, f"FTW bank(s) mix flat and per-layer row layouts: {sorted(mixed)}"

    # Row banks: one padded-window HostBank per (name, layer_id) for the flat layout, plus
    # how to carve the real [num_experts, *row_shape] tensor out of its head; ``None`` marks
    # a per-layer entry (direct view, no carving needed).
    row_hb: dict[str, list] = {}
    row_view_args: dict[str, list] = {}
    row_jobs = []  # (name, HostBank, window_off, window_len, layer_bytes) -- flat layout
    layer_jobs = []  # (name, HostBank, entry) -- per-layer layout, direct aligned read

    for e in flat_entries:
        name = e["name"]
        total, *row_shape = e["shape"]
        assert total % num_layers == 0, (name, total, num_layers)
        num_experts = total // num_layers
        dtype = _dtype_of(e["dtype"])
        row_bytes = (math.prod(row_shape) if row_shape else 1) * _elsize(dtype)
        layer_bytes = num_experts * row_bytes
        assert layer_bytes * num_layers == e["nbytes"], (name, layer_bytes, num_layers, e["nbytes"])
        row_hb[name] = []
        row_view_args[name] = []
        for layer_id in range(num_layers):
            off = e["global_off"] + layer_id * layer_bytes
            win_off = (off // ALIGN) * ALIGN
            win_end = _align_up(off + layer_bytes)
            head_pad = off - win_off
            bank = _new_bank(
                (win_end - win_off,),
                torch.uint8,
                _backing(layer_id),
                bank_name=name,
                layer_id=layer_id,
            )
            row_hb[name].append(bank)
            row_view_args[name].append((head_pad, layer_bytes, num_experts, tuple(row_shape), dtype))
            row_jobs.append((name, bank, win_off, win_end - win_off, layer_bytes, layer_id))

    for base, by_layer in per_layer_groups.items():
        assert sorted(by_layer) == list(range(num_layers)), (
            f"FTW bank {base!r} has per-layer entries for layers {sorted(by_layer)}, "
            f"expected exactly range({num_layers})"
        )
        row_hb[base] = []
        row_view_args[base] = []
        for layer_id in range(num_layers):
            e = by_layer[layer_id]
            assert e["global_off"] % ALIGN == 0, (base, layer_id, e["global_off"])  # writer invariant
            bank = _new_bank(
                tuple(e["shape"]),
                _dtype_of(e["dtype"]),
                _backing(layer_id),
                bank_name=base,
                layer_id=layer_id,
            )
            row_hb[base].append(bank)
            row_view_args[base].append(None)
            layer_jobs.append((base, bank, e, layer_id))

    total_bytes = sum(e["nbytes"] for e in bank_entries)
    try:
        bar = byte_bar(total_bytes, "Loading expert banks (FTW)")
    except BaseException:
        _cleanup_owned()
        raise
    reads_completed = False
    futures = []
    f = None

    # Jobs are per (bank, layer) -- many small reads, so a wider pool; each bank pins
    # as its read completes, overlapping cudaHostRegister with the remaining reads.
    n_jobs = len(alpha_entries) + len(row_jobs) + len(layer_jobs)
    try:
        with PinPipeline() as pins:

            def _read_alpha(e):
                bank = alpha_hb[e["name"]]
                dest = bank.memoryview()
                try:
                    reader.read_into(dest, e, workers=workers, chunk=chunk)
                finally:
                    dest.release()
                pins.submit(
                    bank,
                    residency[0] if host_bank_policy is not None else HostResidency.PINNED.value,
                )
                bar.update(e["nbytes"])

            def _read_row(job):
                _name, bank, win_off, win_len, layer_bytes, layer_id = job
                dest = bank.memoryview()
                try:
                    reader.read_into(
                        dest,
                        {"global_off": win_off, "nbytes": win_len},
                        workers=workers,
                        chunk=chunk,
                    )
                finally:
                    dest.release()
                pins.submit(bank, residency[layer_id])
                bar.update(layer_bytes)

            def _read_layer(job):
                _name, bank, entry, layer_id = job
                dest = bank.memoryview()
                try:
                    reader.read_into(dest, entry, workers=workers, chunk=chunk)
                finally:
                    dest.release()
                pins.submit(bank, residency[layer_id])
                bar.update(entry["nbytes"])

            with ThreadPoolExecutor(min(max(_BANK_CONCURRENCY, 16), max(n_jobs, 1))) as ex:
                futures = [ex.submit(_read_alpha, e) for e in alpha_entries]
                futures += [ex.submit(_read_row, job) for job in row_jobs]
                futures += [ex.submit(_read_layer, job) for job in layer_jobs]
                for f in futures:
                    f.result()
        if host_bank_policy is not None and host_bank_policy.sample_numa_residency:
            # Reads and settles have completed, so the optional self-only
            # residency sample cannot race the loader's writes.
            for identity, bank in allocated_banks:
                host_bank_policy.sample_numa_bank(bank, identity=identity)
        reads_completed = True
    finally:
        # A failed worker future retains its traceback and call arguments; release
        # those exported memoryviews before closing mmap-backed HostBanks.
        futures.clear()
        f = None
        if not reads_completed:
            _cleanup_owned()
        error_count = len(cleanup_errors)
        _safe_close(bar, "FTW progress bar")
        _safe_close(reader, "FTW reader")
        new_errors = cleanup_errors[error_count:]
        if new_errors and reads_completed:
            _cleanup_owned()
        if new_errors and sys.exc_info()[1] is None:
            raise new_errors[0]

    from freetoken.moe.expert_banks import ExpertBanks

    sources: dict[str, list] = {}
    alpha_kw: dict[str, torch.Tensor] = {}
    try:
        # A failed mlock leaves a LOCKED layer pageable; the log and labels report what
        # the banks actually settled at.
        applied = list(residency)
        for banks in row_hb.values():
            for layer_id, bank in enumerate(banks):
                if (
                    applied[layer_id] == HostResidency.LOCKED.value
                    and bank.residency is not HostResidency.LOCKED
                ):
                    applied[layer_id] = HostResidency.PAGEABLE.value
        unpinned = [i for i, r in enumerate(applied) if r != HostResidency.PINNED.value]
        if unpinned:
            by_layer = [0] * num_layers
            for banks in row_hb.values():
                for layer_id, bank in enumerate(banks):
                    by_layer[layer_id] += bank.nbytes
            locked = [i for i in unpinned if applied[i] == HostResidency.LOCKED.value]
            pageable = [i for i in unpinned if i not in set(locked)]
            pinned_b = sum(b for i, b in enumerate(by_layer) if i not in set(unpinned))
            locked_b = sum(by_layer[i] for i in locked)
            pageable_part = ""
            if pageable:
                pageable_b = sum(by_layer[i] for i in pageable)
                pageable_part = (
                    f" + {pageable_b / 2**30:.2f} GiB pageable "
                    f"(lock failed, {len(pageable)} CPU layers: {pageable})"
                )
            logger.info(
                f"MoE bank split residency: {pinned_b / 2**30:.2f} GiB pinned "
                f"({'born-pinned cudaHostAlloc' if born else 'cudaHostRegister'}, "
                f"{num_layers - len(unpinned)} GPU layers) + "
                f"{locked_b / 2**30:.2f} GiB OS-locked ({len(locked)} CPU layers: {locked})"
                f"{pageable_part}"
            )

        owners = tuple(alpha_hb.values()) + tuple(
            bank for banks in row_hb.values() for bank in banks
        ) + (() if staging_ring is None else (staging_ring,))
        if host_bank_policy is not None:
            host_bank_policy.refresh_numa_accounting()
            host_bank_policy.accounting.applied_layers = tuple(applied)
            host_bank_policy.accounting.applied_pinned_bytes = sum(
                bank.allocated_nbytes
                for bank in owners
                if getattr(bank, "residency", None) is HostResidency.PINNED
            )
            accounting = host_bank_policy.accounting.as_dict()
            if host_bank_policy.strategy.value == "bounded-staging":
                accounting["applied_staging_bytes"] = host_bank_policy.plan.staging_bytes
        else:
            accounting = None

        # Materialize aliases only after accounting has succeeded.  On a later failure,
        # clear every derived view before closing its mmap owner.
        for name, banks in row_hb.items():
            views = []
            for bank, view_args in zip(banks, row_view_args[name], strict=True):
                if view_args is None:
                    views.append(bank.tensor)
                    continue
                head_pad, layer_bytes, num_experts, row_shape, dtype = view_args
                raw = bank.tensor[head_pad:head_pad + layer_bytes].view(dtype)
                views.append(
                    raw.view(num_experts, *row_shape) if row_shape else raw.view(num_experts)
                )
                raw = None
            sources[name] = views
            views = []
        alpha_kw = {n: alpha_hb[n].tensor for n in alpha_hb}
        retained_owners = owners if host_bank_policy is not None else ()
        result = ExpertBanks(
            reader.meta("quant_format"), sources, **alpha_kw,
            layer_residency=applied,
            host_bank_accounting=accounting,
            host_bank_owners=retained_owners,
        )
    except BaseException:
        if not committed:
            _cleanup_owned()
        raise
    committed = True
    return result


__all__ = [
    "INDEX_NAME", "FORMAT_TAG", "FORMAT_VERSION", "ALIGN", "DEFAULT_SHARD_LIMIT",
    "is_ftw_checkpoint", "FTWWriter", "FTWReader",
    "iter_ftw_weights", "load_ftw_banks", "layer_bank_entry_name",
    "prepare_ftw_host_bank_policy",
]
