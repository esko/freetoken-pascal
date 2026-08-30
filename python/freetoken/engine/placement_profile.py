"""Torch-free canonical identity and serialization for placement profiles.

This module is an H0 persistence boundary only.  It binds a checked model/runtime/topology
identity to one immutable :class:`PlacementPlan` and one ordered :class:`BackoffProfile`; it
does not inspect CUDA, load a profile at runtime, or select a serving configuration.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import operator
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .placement_plan import (
    MAX_PLACEMENT_BYTES,
    PLACEMENT_SCHEMA_VERSION,
    BackoffProfile,
    GPUPlacementPlan,
    PlacementInputError,
    PlacementPlan,
)

PLACEMENT_PROFILE_SCHEMA_NAME = "freetoken-placement-profile"
PLACEMENT_PROFILE_SCHEMA_VERSION = 1
PLACEMENT_PROFILE_IDENTITY_SCHEMA_NAME = "freetoken-placement-profile-identity"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PLACEHOLDERS = frozenset({"", "na", "n/a", "none", "null", "placeholder", "unknown"})
_CONTEXT_GEOMETRY_FIELDS = frozenset(
    {"context_tokens", "batch_size", "prefill_chunk_tokens", "microbatch_size"}
)
_STATE_GEOMETRY_FIELDS = frozenset(
    {
        "num_request_slots",
        "kv_page_size",
        "kv_pages",
        "gdn_state_bytes",
        "kv_state_bytes",
        "expert_cache_slots",
    }
)
_QSA_GEOMETRY_FIELDS = frozenset(
    {
        "context_tokens",
        "token_rows",
        "page_table_width",
        "page_size",
        "index_heads",
        "query_heads",
        "kv_heads",
        "head_dim",
        "index_head_dim",
        "top_k",
        "compression_ratio",
        "num_index_layers",
        "num_req_slots",
        "ring_capacity",
        "num_pages",
        "max_position",
        "rotary_dim",
        "batch_size",
        "capture_max_batch_size",
        "phase",
        "topk_backend",
        "dtype_bytes",
    }
)


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise PlacementInputError(f"{name} must be an integer, not a boolean")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise PlacementInputError(f"{name} must be an integer") from exc
    if result < 0 or result > MAX_PLACEMENT_BYTES:
        raise PlacementInputError(f"{name} is outside the supported integer range")
    if positive and result == 0:
        raise PlacementInputError(f"{name} must be positive")
    return result


def _serialized_integer(value: Any, name: str) -> int:
    """Validate an integer that came from a serialized profile document."""
    return _integer(value, name)


def _serialized_signed_integer(value: Any, name: str) -> int:
    """Validate a bounded signed integer that came from a serialized profile document."""
    if isinstance(value, bool):
        raise PlacementInputError(f"{name} must be an integer, not a boolean")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise PlacementInputError(f"{name} must be an integer") from exc
    if result < -MAX_PLACEMENT_BYTES or result > MAX_PLACEMENT_BYTES:
        raise PlacementInputError(f"{name} is outside the supported integer range")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlacementInputError(f"{name} must be non-empty text")
    result = value.strip()
    if result.casefold() in _PLACEHOLDERS:
        raise PlacementInputError(f"{name} must not be a placeholder value")
    return result


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PlacementInputError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _runtime_commit(value: Any) -> str:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise PlacementInputError("runtime_commit must be a 40-character lowercase commit")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlacementInputError(f"{name} must be a mapping")
    if not value:
        raise PlacementInputError(f"{name} must not be empty")
    return value


def _geometry_value(value: Any, path: str) -> Any:
    """Normalize JSON-like raw geometry into immutable values with bounded integers."""
    if isinstance(value, bool):
        raise PlacementInputError(f"{path} must not use a boolean as geometry")
    if isinstance(value, int):
        return _integer(value, path)
    if isinstance(value, float):
        raise PlacementInputError(f"{path} must not be a floating-point value")
    if isinstance(value, str):
        return _text(value, path)
    if isinstance(value, Mapping):
        if not value:
            raise PlacementInputError(f"{path} must not be empty")
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise PlacementInputError(f"{path} keys must be non-empty text")
            normalized_key = key.strip()
            if normalized_key in normalized:
                raise PlacementInputError(f"{path} contains duplicate key {normalized_key!r}")
            normalized[normalized_key] = _geometry_value(item, f"{path}.{key}")
        return MappingProxyType(normalized)
    if isinstance(value, (list, tuple)):
        if not value:
            raise PlacementInputError(f"{path} must not contain an empty sequence")
        return tuple(_geometry_value(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise PlacementInputError(f"{path} contains an unsupported value")


def _geometry(value: Any, name: str, required: tuple[str, ...]) -> Mapping[str, Any]:
    source = _mapping(value, name)
    normalized = _geometry_value(source, name)
    if not isinstance(normalized, Mapping):  # pragma: no cover - guarded by _mapping
        raise PlacementInputError(f"{name} must be a mapping")
    missing = set(required) - set(normalized)
    if missing:
        raise PlacementInputError(f"{name} missing required fields: {sorted(missing)}")
    return normalized


def _exact_geometry(value: Any, name: str, fields: frozenset[str]) -> Mapping[str, Any]:
    normalized = _geometry(value, name, ())
    actual = set(normalized)
    if actual != fields:
        raise PlacementInputError(
            f"{name} fields disagree: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return normalized


def _geometry_integer(value: Mapping[str, Any], key: str, name: str, *, positive: bool) -> int:
    if key not in value:
        raise PlacementInputError(f"{name} missing required field {key!r}")
    return _integer(value[key], f"{name}.{key}", positive=positive)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _canonical_value(value: Any, path: str = "$") -> Any:
    """Convert a JSON-compatible value while rejecting floats, NaN, and non-string keys."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value < -MAX_PLACEMENT_BYTES or value > MAX_PLACEMENT_BYTES:
            raise PlacementInputError(f"{path} integer is outside the supported range")
        return value
    if isinstance(value, float):
        raise PlacementInputError(f"{path} must not contain floating-point values")
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PlacementInputError(f"{path} keys must be strings")
            normalized[key] = _canonical_value(item, f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise PlacementInputError(f"{path} is not JSON-compatible")


def canonical_json_bytes(document: Any) -> bytes:
    """Serialize deterministic UTF-8 JSON with a single trailing newline.

    Digests in this module cover exactly these bytes.  Sorted object keys, compact separators,
    UTF-8 output and the newline are part of the profile schema contract.
    """
    normalized = _canonical_value(document)
    try:
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - _canonical_value prevalidates
        raise PlacementInputError(f"cannot serialize canonical profile JSON: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def _digest(document: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _qsa_workspace_inputs_type() -> type:
    """Load the torch-free QSA input class without importing runtime attention registries."""
    try:
        from freetoken.attention.qsa_workspace import QSAWorkspaceInputs

        return QSAWorkspaceInputs
    except ModuleNotFoundError:
        # ``freetoken.attention.__init__`` imports optional serving dependencies.  The H0
        # accounting module itself is standalone, so load that source directly when a hosted
        # minimal environment intentionally omits those optional packages.
        source = Path(__file__).resolve().parents[1] / "attention" / "qsa_workspace.py"
        spec = importlib.util.spec_from_file_location("freetoken._h0_qsa_workspace", source)
        if spec is None or spec.loader is None:
            raise PlacementInputError("cannot load torch-free QSA workspace schema") from None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.QSAWorkspaceInputs


def _exact_fields(value: Any, expected: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlacementInputError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise PlacementInputError(
            f"{name} fields disagree: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


@dataclass(frozen=True, slots=True)
class GPUProfileTopology:
    """Stable, measured topology identity for one placement-plan rank."""

    rank: int
    gpu_uuid: str
    capacity_bytes: int
    compute_capability: str
    pci_bus_id: str
    numa_node: int
    peer_ownership: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        rank = _integer(self.rank, "topology.rank")
        gpu_uuid = _text(self.gpu_uuid, "topology.gpu_uuid")
        capacity = _integer(self.capacity_bytes, "topology.capacity_bytes", positive=True)
        compute = _text(self.compute_capability, "topology.compute_capability")
        pci = _text(self.pci_bus_id, "topology.pci_bus_id")
        numa = _integer(self.numa_node, "topology.numa_node")
        try:
            peers = tuple(_integer(peer, "topology.peer_ownership") for peer in self.peer_ownership)
        except TypeError as exc:
            raise PlacementInputError("topology.peer_ownership must be iterable") from exc
        if len(set(peers)) != len(peers) or any(peer < 0 for peer in peers):
            raise PlacementInputError(
                "topology.peer_ownership must contain unique non-negative ranks"
            )
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "gpu_uuid", gpu_uuid)
        object.__setattr__(self, "capacity_bytes", capacity)
        object.__setattr__(self, "compute_capability", compute)
        object.__setattr__(self, "pci_bus_id", pci)
        object.__setattr__(self, "numa_node", numa)
        object.__setattr__(self, "peer_ownership", peers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "gpu_uuid": self.gpu_uuid,
            "capacity_bytes": self.capacity_bytes,
            "compute_capability": self.compute_capability,
            "pci_bus_id": self.pci_bus_id,
            "numa_node": self.numa_node,
            "peer_ownership": list(self.peer_ownership),
        }

    @classmethod
    def from_dict(cls, value: Any) -> GPUProfileTopology:
        fields = _exact_fields(
            value,
            frozenset(
                {
                    "rank",
                    "gpu_uuid",
                    "capacity_bytes",
                    "compute_capability",
                    "pci_bus_id",
                    "numa_node",
                    "peer_ownership",
                }
            ),
            "topology GPU",
        )
        return cls(**dict(fields))

    to_dict = as_dict


@dataclass(frozen=True, slots=True)
class PlacementProfileIdentity:
    """All immutable model, runtime, geometry and hardware inputs to a placement profile."""

    model_sha256: str
    quant_census_sha256: str
    binary_sha256: str
    toolchain_sha256: str
    runtime_commit: str
    runtime_version: str
    driver_version: str
    cuda_runtime_version: str
    cuda_toolchain_identity: str
    context_geometry: Mapping[str, Any]
    state_geometry: Mapping[str, Any]
    qsa_geometry: Mapping[str, Any]
    topology: tuple[GPUProfileTopology, ...]

    def __post_init__(self) -> None:
        values = {
            "model_sha256": _sha256(self.model_sha256, "model_sha256"),
            "quant_census_sha256": _sha256(self.quant_census_sha256, "quant_census_sha256"),
            "binary_sha256": _sha256(self.binary_sha256, "binary_sha256"),
            "toolchain_sha256": _sha256(self.toolchain_sha256, "toolchain_sha256"),
            "runtime_commit": _runtime_commit(self.runtime_commit),
            "runtime_version": _text(self.runtime_version, "runtime_version"),
            "driver_version": _text(self.driver_version, "driver_version"),
            "cuda_runtime_version": _text(self.cuda_runtime_version, "cuda_runtime_version"),
            "cuda_toolchain_identity": _text(
                self.cuda_toolchain_identity, "cuda_toolchain_identity"
            ),
            "context_geometry": _exact_geometry(
                self.context_geometry, "context_geometry", _CONTEXT_GEOMETRY_FIELDS
            ),
            "state_geometry": _exact_geometry(
                self.state_geometry, "state_geometry", _STATE_GEOMETRY_FIELDS
            ),
            "qsa_geometry": _exact_geometry(
                self.qsa_geometry, "qsa_geometry", _QSA_GEOMETRY_FIELDS
            ),
        }
        for key in _CONTEXT_GEOMETRY_FIELDS:
            _geometry_integer(values["context_geometry"], key, "context_geometry", positive=True)
        for key in ("num_request_slots", "kv_page_size", "kv_pages"):
            _geometry_integer(values["state_geometry"], key, "state_geometry", positive=True)
        for key in ("gdn_state_bytes", "kv_state_bytes", "expert_cache_slots"):
            _geometry_integer(values["state_geometry"], key, "state_geometry", positive=False)
        qsa = values["qsa_geometry"]
        for key in _QSA_GEOMETRY_FIELDS - {"phase", "topk_backend"}:
            _geometry_integer(qsa, key, "qsa_geometry", positive=True)
        _text(qsa["phase"], "qsa_geometry.phase")
        _text(qsa["topk_backend"], "qsa_geometry.topk_backend")
        try:
            _qsa_workspace_inputs_type()(**dict(qsa))
        except PlacementInputError:
            raise
        except (TypeError, ValueError) as exc:
            raise PlacementInputError(f"invalid QSA geometry: {exc}") from exc
        if (
            values["context_geometry"]["prefill_chunk_tokens"]
            > values["context_geometry"]["context_tokens"]
        ):
            raise PlacementInputError(
                "context_geometry prefill_chunk_tokens exceeds context_tokens"
            )
        if values["context_geometry"]["microbatch_size"] > values["context_geometry"]["batch_size"]:
            raise PlacementInputError("context_geometry microbatch_size exceeds batch_size")
        try:
            topology = tuple(self.topology)
        except TypeError as exc:
            raise PlacementInputError("topology must be an iterable of GPUProfileTopology") from exc
        if len(topology) not in {1, 2}:
            raise PlacementInputError("topology must describe exactly one or two GPUs")
        if any(not isinstance(item, GPUProfileTopology) for item in topology):
            raise PlacementInputError("topology must contain GPUProfileTopology values")
        if tuple(item.rank for item in topology) != tuple(range(len(topology))):
            raise PlacementInputError("topology ranks must be contiguous and ordered")
        if len({item.gpu_uuid for item in topology}) != len(topology):
            raise PlacementInputError("topology GPU UUIDs must be unique")
        for item in topology:
            if any(peer >= len(topology) for peer in item.peer_ownership):
                raise PlacementInputError("topology peer ownership rank is out of range")
        object.__setattr__(self, "topology", topology)
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_sha256": self.model_sha256,
            "quant_census_sha256": self.quant_census_sha256,
            "binary_sha256": self.binary_sha256,
            "toolchain_sha256": self.toolchain_sha256,
            "runtime_commit": self.runtime_commit,
            "runtime_version": self.runtime_version,
            "driver_version": self.driver_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "cuda_toolchain_identity": self.cuda_toolchain_identity,
            "context_geometry": _plain(self.context_geometry),
            "state_geometry": _plain(self.state_geometry),
            "qsa_geometry": _plain(self.qsa_geometry),
            "topology": [item.as_dict() for item in self.topology],
        }

    @property
    def digest(self) -> str:
        return _digest(
            {
                "schema_name": PLACEMENT_PROFILE_IDENTITY_SCHEMA_NAME,
                "schema_version": PLACEMENT_PROFILE_SCHEMA_VERSION,
                "identity": self.as_dict(),
            }
        )

    @classmethod
    def from_dict(cls, value: Any) -> PlacementProfileIdentity:
        fields = _exact_fields(
            value,
            frozenset(
                {
                    "model_sha256",
                    "quant_census_sha256",
                    "binary_sha256",
                    "toolchain_sha256",
                    "runtime_commit",
                    "runtime_version",
                    "driver_version",
                    "cuda_runtime_version",
                    "cuda_toolchain_identity",
                    "context_geometry",
                    "state_geometry",
                    "qsa_geometry",
                    "topology",
                }
            ),
            "profile identity",
        )
        try:
            topology = tuple(GPUProfileTopology.from_dict(item) for item in fields["topology"])
            kwargs = dict(fields)
            kwargs["topology"] = topology
            return cls(**kwargs)
        except PlacementInputError:
            raise
        except (TypeError, ValueError) as exc:
            raise PlacementInputError(f"invalid profile identity: {exc}") from exc

    def stale_reasons(self, other: PlacementProfileIdentity) -> tuple[str, ...]:
        if not isinstance(other, PlacementProfileIdentity):
            raise PlacementInputError("other identity must be a PlacementProfileIdentity")
        return tuple(_diff_paths(self.as_dict(), other.as_dict(), "identity"))

    def matches(self, other: PlacementProfileIdentity) -> bool:
        return not self.stale_reasons(other)

    to_dict = as_dict


def _diff_paths(left: Any, right: Any, path: str) -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left or key not in right:
                result.append(child)
            else:
                result.extend(_diff_paths(left[key], right[key], child))
        return result
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        result = []
        if len(left) != len(right):
            result.append(path)
        for index, (lhs, rhs) in enumerate(zip(left, right, strict=False)):
            result.extend(_diff_paths(lhs, rhs, f"{path}.{index}"))
        return result
    return [] if left == right else [path]


def _gpu_plan_from_dict(value: Any) -> GPUPlacementPlan:
    expected = frozenset(
        {
            "schema_version",
            "rank",
            "gpu_uuid",
            "capacity_bytes",
            "key",
            "status",
            "required_bytes",
            "non_qsa_required_bytes",
            "qsa_persistent_bytes",
            "qsa_transient_high_water_bytes",
            "qsa_required_bytes",
            "live_required_bytes",
            "peak_required_bytes",
            "available_bytes",
            "headroom_bytes",
            "deficit_bytes",
            "categories",
            "reasons",
        }
    )
    fields = _exact_fields(value, expected, "placement GPU")
    for field in (
        "schema_version",
        "rank",
        "capacity_bytes",
        "required_bytes",
        "non_qsa_required_bytes",
        "qsa_persistent_bytes",
        "qsa_transient_high_water_bytes",
        "qsa_required_bytes",
        "live_required_bytes",
        "peak_required_bytes",
        "available_bytes",
        "headroom_bytes",
        "deficit_bytes",
    ):
        if field == "headroom_bytes":
            _serialized_signed_integer(fields[field], f"placement GPU.{field}")
        else:
            _serialized_integer(fields[field], f"placement GPU.{field}")
    if fields["schema_version"] != PLACEMENT_SCHEMA_VERSION:
        raise PlacementInputError("unsupported placement GPU schema version")
    try:
        plan = GPUPlacementPlan(
            rank=fields["rank"],
            gpu_uuid=fields["gpu_uuid"],
            capacity_bytes=fields["capacity_bytes"],
            available_bytes=fields["available_bytes"],
            categories=fields["categories"],
            required_bytes=fields["required_bytes"],
            headroom_bytes=fields["headroom_bytes"],
            deficit_bytes=fields["deficit_bytes"],
            status=fields["status"],
            reasons=fields["reasons"],
        )
    except PlacementInputError:
        raise
    except (TypeError, ValueError) as exc:
        raise PlacementInputError(f"invalid placement GPU: {exc}") from exc
    planner_fields = dict(fields)
    planner_fields.pop("capacity_bytes")
    if canonical_json_bytes(plan.as_dict()) != canonical_json_bytes(planner_fields):
        raise PlacementInputError("placement GPU contains inconsistent derived fields")
    return plan


def _plan_from_dict(value: Any) -> PlacementPlan:
    fields = _exact_fields(
        value,
        frozenset({"schema_version", "gpu_count", "safety_reserve_bytes", "gpus", "gpus_by_key"}),
        "placement plan",
    )
    for field in ("schema_version", "gpu_count", "safety_reserve_bytes"):
        _serialized_integer(fields[field], f"placement plan.{field}")
    if fields["schema_version"] != PLACEMENT_SCHEMA_VERSION:
        raise PlacementInputError("unsupported placement plan schema version")
    try:
        gpus = tuple(_gpu_plan_from_dict(item) for item in fields["gpus"])
        plan = PlacementPlan(gpus=gpus, safety_reserve_bytes=fields["safety_reserve_bytes"])
    except PlacementInputError:
        raise
    except (TypeError, ValueError) as exc:
        raise PlacementInputError(f"invalid placement plan: {exc}") from exc
    if canonical_json_bytes(_plan_payload(plan)) != canonical_json_bytes(dict(fields)):
        raise PlacementInputError("placement plan contains inconsistent derived fields")
    if fields["gpu_count"] != plan.gpu_count:
        raise PlacementInputError("placement plan GPU count is inconsistent")
    return plan


def _gpu_plan_payload(plan: GPUPlacementPlan) -> dict[str, Any]:
    """Serialize the capacity omitted by the planner's telemetry view for profile identity."""
    result = plan.as_dict()
    result["capacity_bytes"] = plan.capacity_bytes
    return result


def _plan_payload(plan: PlacementPlan) -> dict[str, Any]:
    result = plan.as_dict()
    result["gpus"] = [_gpu_plan_payload(item) for item in plan.gpus]
    result["gpus_by_key"] = {key: _gpu_plan_payload(item) for key, item in plan.by_key.items()}
    return result


def _backoff_from_dict(value: Any) -> BackoffProfile:
    fields = _exact_fields(
        value,
        frozenset(
            {
                "schema_version",
                "name",
                "cache_slots",
                "context_tokens",
                "batch_size",
                "gpu_placement",
            }
        ),
        "backoff profile",
    )
    for field in ("schema_version", "cache_slots", "context_tokens", "batch_size"):
        _serialized_integer(fields[field], f"backoff profile.{field}")
    if fields["schema_version"] != PLACEMENT_SCHEMA_VERSION:
        raise PlacementInputError("unsupported backoff profile schema version")
    try:
        profile = BackoffProfile(
            name=fields["name"],
            cache_slots=fields["cache_slots"],
            context_tokens=fields["context_tokens"],
            batch_size=fields["batch_size"],
            gpu_placement=fields["gpu_placement"],
        )
    except PlacementInputError:
        raise
    except (TypeError, ValueError) as exc:
        raise PlacementInputError(f"invalid backoff profile: {exc}") from exc
    if canonical_json_bytes(profile.as_dict()) != canonical_json_bytes(dict(fields)):
        raise PlacementInputError("backoff profile contains inconsistent fields")
    return profile


@dataclass(frozen=True, slots=True)
class PlacementProfile:
    """Canonical identity envelope for one plan and one ordered backoff candidate."""

    identity: PlacementProfileIdentity
    plan: PlacementPlan
    backoff_profile: BackoffProfile

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PlacementProfileIdentity):
            raise PlacementInputError("profile identity must be a PlacementProfileIdentity")
        if not isinstance(self.plan, PlacementPlan):
            raise PlacementInputError("profile plan must be a PlacementPlan")
        if not isinstance(self.backoff_profile, BackoffProfile):
            raise PlacementInputError("profile backoff_profile must be a BackoffProfile")
        normalized_name = _text(self.backoff_profile.name, "backoff_profile.name")
        if normalized_name != self.backoff_profile.name:
            raise PlacementInputError(
                "backoff_profile.name must use canonical surrounding whitespace"
            )
        topology = self.identity.topology
        if len(topology) != self.plan.gpu_count:
            raise PlacementInputError("profile topology count must match placement plan")
        for topo, gpu in zip(topology, self.plan.gpus, strict=True):
            if (
                topo.rank != gpu.rank
                or topo.gpu_uuid != gpu.gpu_uuid
                or topo.capacity_bytes != gpu.capacity_bytes
            ):
                raise PlacementInputError(
                    "profile topology rank/UUID/capacity does not match placement plan "
                    f"at rank {gpu.rank}"
                )
        context = self.identity.context_geometry
        state = self.identity.state_geometry
        qsa = self.identity.qsa_geometry
        if context["context_tokens"] != self.backoff_profile.context_tokens:
            raise PlacementInputError(
                "profile context geometry does not match backoff context_tokens"
            )
        if context["batch_size"] != self.backoff_profile.batch_size:
            raise PlacementInputError("profile context geometry does not match backoff batch_size")
        if qsa["context_tokens"] != self.backoff_profile.context_tokens:
            raise PlacementInputError("profile QSA geometry does not match backoff context_tokens")
        if qsa["batch_size"] != self.backoff_profile.batch_size:
            raise PlacementInputError("profile QSA geometry does not match backoff batch_size")
        if state["expert_cache_slots"] != self.backoff_profile.cache_slots:
            raise PlacementInputError("profile state geometry does not match backoff cache_slots")
        for state_key, qsa_key, label in (
            ("num_request_slots", "num_req_slots", "request-slot"),
            ("kv_page_size", "page_size", "page-size"),
            ("kv_pages", "num_pages", "page-count"),
        ):
            if state[state_key] != qsa[qsa_key]:
                raise PlacementInputError(f"profile state/QSA {label} geometry does not match")

    @property
    def profile_name(self) -> str:
        return self.backoff_profile.name

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_name": PLACEMENT_PROFILE_SCHEMA_NAME,
            "schema_version": PLACEMENT_PROFILE_SCHEMA_VERSION,
            "identity": self.identity.as_dict(),
            "placement_plan": _plan_payload(self.plan),
            "backoff_profile": self.backoff_profile.as_dict(),
        }

    @property
    def digest(self) -> str:
        """SHA-256 of the canonical profile payload, excluding the digest itself."""
        return _digest(self._payload())

    def as_dict(self) -> dict[str, Any]:
        document = self._payload()
        document["digest"] = self.digest
        return document

    @classmethod
    def from_dict(cls, value: Any) -> PlacementProfile:
        # Validate the complete tree before parsing individual fields. This prevents a
        # float hidden in an ignored or derived field from entering the digest path.
        canonical_json_bytes(value)
        fields = _exact_fields(
            value,
            frozenset(
                {
                    "schema_name",
                    "schema_version",
                    "identity",
                    "placement_plan",
                    "backoff_profile",
                    "digest",
                }
            ),
            "placement profile",
        )
        if fields["schema_name"] != PLACEMENT_PROFILE_SCHEMA_NAME:
            raise PlacementInputError("unsupported placement profile schema name")
        _serialized_integer(fields["schema_version"], "placement profile.schema_version")
        if fields["schema_version"] != PLACEMENT_PROFILE_SCHEMA_VERSION:
            raise PlacementInputError("unsupported placement profile schema version")
        digest = fields["digest"]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise PlacementInputError("placement profile digest must be a lowercase SHA-256 digest")
        try:
            profile = cls(
                identity=PlacementProfileIdentity.from_dict(fields["identity"]),
                plan=_plan_from_dict(fields["placement_plan"]),
                backoff_profile=_backoff_from_dict(fields["backoff_profile"]),
            )
        except PlacementInputError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise PlacementInputError(f"invalid placement profile: {exc}") from exc
        if profile.digest != digest:
            raise PlacementInputError("placement profile digest mismatch")
        return profile

    def stale_reasons(self, other: PlacementProfile) -> tuple[str, ...]:
        if not isinstance(other, PlacementProfile):
            raise PlacementInputError("other profile must be a PlacementProfile")
        reasons = self.identity.stale_reasons(other.identity)
        reasons += tuple(
            _diff_paths(_plan_payload(self.plan), _plan_payload(other.plan), "placement_plan")
        )
        reasons += tuple(
            _diff_paths(
                self.backoff_profile.as_dict(),
                other.backoff_profile.as_dict(),
                "backoff_profile",
            )
        )
        return tuple(dict.fromkeys(reasons))

    def matches(self, other: PlacementProfile) -> bool:
        return not self.stale_reasons(other)

    to_dict = as_dict


__all__ = [
    "PLACEMENT_PROFILE_IDENTITY_SCHEMA_NAME",
    "PLACEMENT_PROFILE_SCHEMA_NAME",
    "PLACEMENT_PROFILE_SCHEMA_VERSION",
    "GPUProfileTopology",
    "PlacementProfile",
    "PlacementProfileIdentity",
    "canonical_json_bytes",
]
