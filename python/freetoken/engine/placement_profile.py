"""Torch-free canonical identity and serialization for placement profiles.

This module is an H0 persistence boundary only.  It binds a checked model/runtime/topology
identity to one immutable :class:`PlacementPlan` and one ordered :class:`BackoffProfile`; it
does not inspect CUDA, load a profile at runtime, or select a serving configuration.
"""

from __future__ import annotations

import hashlib
import json
import operator
import re
from collections.abc import Mapping
from dataclasses import dataclass
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


def _geometry_integer(value: Mapping[str, Any], key: str, name: str, *, positive: bool) -> int:
    if key not in value:
        raise PlacementInputError(f"{name} missing required field {key!r}")
    return _integer(value[key], f"{name}.{key}", positive=positive)


def _optional_geometry_integer(
    value: Mapping[str, Any], key: str, name: str, *, positive: bool
) -> None:
    if key in value:
        _geometry_integer(value, key, name, positive=positive)


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
            "context_geometry": _geometry(
                self.context_geometry,
                "context_geometry",
                (),
            ),
            "state_geometry": _geometry(self.state_geometry, "state_geometry", ()),
            "qsa_geometry": _geometry(self.qsa_geometry, "qsa_geometry", ()),
        }
        _optional_geometry_integer(
            values["context_geometry"], "context_tokens", "context_geometry", positive=True
        )
        _optional_geometry_integer(
            values["context_geometry"], "batch_size", "context_geometry", positive=True
        )
        _optional_geometry_integer(
            values["state_geometry"], "cache_slots", "state_geometry", positive=False
        )
        _optional_geometry_integer(
            values["qsa_geometry"], "context_tokens", "qsa_geometry", positive=True
        )
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
    if plan.as_dict() != planner_fields:
        raise PlacementInputError("placement GPU contains inconsistent derived fields")
    return plan


def _plan_from_dict(value: Any) -> PlacementPlan:
    fields = _exact_fields(
        value,
        frozenset({"schema_version", "gpu_count", "safety_reserve_bytes", "gpus", "gpus_by_key"}),
        "placement plan",
    )
    if fields["schema_version"] != PLACEMENT_SCHEMA_VERSION:
        raise PlacementInputError("unsupported placement plan schema version")
    try:
        gpus = tuple(_gpu_plan_from_dict(item) for item in fields["gpus"])
        plan = PlacementPlan(gpus=gpus, safety_reserve_bytes=fields["safety_reserve_bytes"])
    except PlacementInputError:
        raise
    except (TypeError, ValueError) as exc:
        raise PlacementInputError(f"invalid placement plan: {exc}") from exc
    if _plan_payload(plan) != dict(fields):
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
    if profile.as_dict() != dict(fields):
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
        for geometry_name, geometry in (("context", context), ("state", state), ("QSA", qsa)):
            for key, expected in (
                ("context_tokens", self.backoff_profile.context_tokens),
                ("batch_size", self.backoff_profile.batch_size),
                ("cache_slots", self.backoff_profile.cache_slots),
            ):
                if key in geometry and geometry[key] != expected:
                    raise PlacementInputError(
                        f"profile {geometry_name} geometry does not match backoff {key}"
                    )

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
