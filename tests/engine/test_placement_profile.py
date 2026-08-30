"""H0 tests for canonical, stale-safe placement profile serialization."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from freetoken.engine.placement_plan import (
    PLACEMENT_CATEGORIES,
    BackoffProfile,
    PlacementPlanInput,
    PlacementPlannerError,
    plan_placement,
)
from freetoken.engine.placement_profile import (
    PLACEMENT_PROFILE_SCHEMA_NAME,
    PLACEMENT_PROFILE_SCHEMA_VERSION,
    GPUProfileTopology,
    PlacementProfile,
    PlacementProfileIdentity,
    canonical_json_bytes,
)


def _categories(**overrides: int) -> dict[str, int]:
    values = {name: 0 for name in PLACEMENT_CATEGORIES}
    values.update(
        {
            "dense_resident_weights": 100,
            "shared_experts": 20,
            "gdn_kv_recurrent_state": 30,
            "qsa_persistent_score": 5,
            "qsa_persistent_top_k": 6,
            "qsa_persistent_expand_gather": 7,
            "qsa_persistent_attention": 8,
            "qsa_persistent_state": 9,
            "qsa_transient_score": 10,
            "qsa_transient_top_k": 11,
            "qsa_transient_expand_gather": 12,
            "qsa_transient_attention": 13,
            "qsa_transient_state": 14,
            "cuda_context": 15,
            "generic_workspaces": 16,
            "transfer_buffers": 17,
            "static_expert_cache_slots": 18,
            "dynamic_expert_cache_slots": 19,
            "safety_reserve": 50,
        }
    )
    values.update(overrides)
    return values


def _plan(*, two_gpus: bool = False, capacity: int = 500):
    inputs = [
        PlacementPlanInput(
            capacity_bytes=capacity,
            available_bytes=capacity,
            gpu_uuid="gpu-0",
            categories=_categories(),
        )
    ]
    if two_gpus:
        inputs.append(
            PlacementPlanInput(
                capacity_bytes=capacity + 100,
                available_bytes=capacity + 100,
                gpu_uuid="gpu-1",
                categories=_categories(),
            )
        )
    return plan_placement(inputs, safety_reserve_bytes=50)


def _identity(*, topology: tuple[GPUProfileTopology, ...] | None = None, **overrides):
    if topology is None:
        topology = (
            GPUProfileTopology(
                rank=0,
                gpu_uuid="gpu-0",
                capacity_bytes=500,
                compute_capability="6.1",
                pci_bus_id="0000:01:00.0",
                numa_node=0,
                peer_ownership=(),
            ),
        )
    values = {
        "model_sha256": "a" * 64,
        "quant_census_sha256": "b" * 64,
        "binary_sha256": "c" * 64,
        "toolchain_sha256": "d" * 64,
        "runtime_commit": "e" * 40,
        "runtime_version": "fixture-runtime-1",
        "driver_version": "550.1",
        "cuda_runtime_version": "12.6.3",
        "cuda_toolchain_identity": "cuda-12.6.3-sm61",
        "context_geometry": {
            "context_tokens": 1024,
            "batch_size": 1,
            "prefill_chunk_tokens": 512,
            "microbatch_size": 1,
        },
        "state_geometry": {
            "num_request_slots": 1,
            "kv_page_size": 16,
            "kv_pages": 4,
            "gdn_state_bytes": 1024,
            "kv_state_bytes": 2048,
            "expert_cache_slots": 0,
        },
        "qsa_geometry": {
            "context_tokens": 1024,
            "token_rows": 512,
            "page_table_width": 1024,
            "page_size": 16,
            "index_heads": 4,
            "query_heads": 8,
            "kv_heads": 4,
            "head_dim": 64,
            "index_head_dim": 32,
            "top_k": 8,
            "compression_ratio": 4,
            "num_index_layers": 2,
            "num_req_slots": 1,
            "ring_capacity": 4,
            "num_pages": 4,
            "max_position": 1024,
            "rotary_dim": 16,
            "batch_size": 1,
            "capture_max_batch_size": 1,
            "phase": "eager",
            "topk_backend": "triton",
            "dtype_bytes": 2,
        },
        "topology": topology,
    }
    values.update(overrides)
    return PlacementProfileIdentity(**values)


def _profile(*, two_gpus: bool = False, capacity: int = 500):
    plan = _plan(two_gpus=two_gpus, capacity=capacity)
    topology = tuple(
        GPUProfileTopology(
            rank=item.rank,
            gpu_uuid=item.gpu_uuid,
            capacity_bytes=item.capacity_bytes,
            compute_capability="6.1",
            pci_bus_id=f"0000:0{item.rank + 1}:00.0",
            numa_node=item.rank,
            peer_ownership=tuple(other.rank for other in plan.gpus if other.rank != item.rank),
        )
        for item in plan.gpus
    )
    identity = _identity(topology=topology)
    if two_gpus:
        identity = replace(
            identity,
            context_geometry={**identity.context_geometry, "batch_size": 2},
            state_geometry={
                **identity.state_geometry,
                "num_request_slots": 2,
                "expert_cache_slots": 0,
            },
            qsa_geometry={
                **identity.qsa_geometry,
                "token_rows": 1024,
                "batch_size": 2,
                "num_req_slots": 2,
                "capture_max_batch_size": 2,
            },
        )
    backoff = BackoffProfile("cache-zero", cache_slots=0, context_tokens=1024, batch_size=1)
    if two_gpus:
        backoff = replace(backoff, batch_size=2)
    return PlacementProfile(identity=identity, plan=plan, backoff_profile=backoff)


def test_profile_binds_identity_plan_and_backoff_and_is_immutable() -> None:
    profile = _profile()

    assert profile.profile_name == "cache-zero"
    assert len(profile.digest) == 64
    assert profile.digest == profile.digest
    assert profile.as_dict()["schema_name"] == PLACEMENT_PROFILE_SCHEMA_NAME
    assert profile.as_dict()["schema_version"] == PLACEMENT_PROFILE_SCHEMA_VERSION
    with pytest.raises((AttributeError, TypeError)):
        profile.identity.context_geometry["context_tokens"] = 2


def test_canonical_json_uses_sorted_compact_utf8_and_newline() -> None:
    assert canonical_json_bytes({"z": "é", "a": [1, 2]}) == '{"a":[1,2],"z":"é"}\n'.encode()
    assert canonical_json_bytes({"a": 1, "b": 2}) == canonical_json_bytes({"b": 2, "a": 1})


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_identity_mutation_changes_digest_and_reports_field(field: str) -> None:
    original = _profile()
    value = getattr(original.identity, field)
    if isinstance(value, dict) or hasattr(value, "items"):
        changed = dict(value)
        mutation_key = {
            "context_geometry": "prefill_chunk_tokens",
            "state_geometry": "gdn_state_bytes",
            "qsa_geometry": "token_rows",
        }[field]
        changed[mutation_key] += 1
    elif field.endswith("sha256"):
        changed = "f" * 64
    elif field == "runtime_commit":
        changed = "f" * 40
    else:
        changed = f"changed-{field}"
    changed_identity = replace(original.identity, **{field: changed})
    changed_profile = replace(original, identity=changed_identity)

    assert changed_profile.digest != original.digest
    assert any(
        path.startswith(f"identity.{field}") for path in original.stale_reasons(changed_profile)
    )
    assert not original.matches(changed_profile)


def test_plan_and_backoff_mutations_change_digest_and_report_paths() -> None:
    original = _profile()
    categories = dict(original.plan.gpus[0].categories)
    categories["generic_workspaces"] += 1
    changed_plan = plan_placement(
        (
            PlacementPlanInput(
                capacity_bytes=500,
                gpu_uuid="gpu-0",
                categories=categories,
            ),
        ),
        safety_reserve_bytes=50,
    )
    changed_plan_profile = replace(original, plan=changed_plan)
    assert changed_plan_profile.digest != original.digest
    assert any(
        path.startswith("placement_plan.") for path in original.stale_reasons(changed_plan_profile)
    )

    changed_backoff = replace(original.backoff_profile, cache_slots=1)
    changed_identity = replace(
        original.identity,
        state_geometry={**original.identity.state_geometry, "expert_cache_slots": 1},
    )
    changed_backoff_profile = replace(
        original, identity=changed_identity, backoff_profile=changed_backoff
    )
    assert changed_backoff_profile.digest != original.digest
    assert any(
        path.startswith("backoff_profile.")
        for path in original.stale_reasons(changed_backoff_profile)
    )


def test_topology_mutation_changes_digest_and_reports_path() -> None:
    original = _profile()
    gpu = replace(original.identity.topology[0], peer_ownership=(0,))
    changed = replace(original.identity, topology=(gpu,))
    changed_profile = replace(original, identity=changed)
    assert changed_profile.digest != original.digest
    assert "identity.topology.0.peer_ownership" in original.stale_reasons(changed_profile)


@pytest.mark.parametrize(
    "field", ["compute_capability", "pci_bus_id", "numa_node", "peer_ownership"]
)
def test_each_topology_attribute_is_digest_bound(field: str) -> None:
    original = _profile(two_gpus=True)
    current = original.identity.topology[0]
    value = {
        "compute_capability": "7.5",
        "pci_bus_id": "0000:03:00.0",
        "numa_node": 1,
        "peer_ownership": (),
    }[field]
    changed_topology = (replace(current, **{field: value}), *original.identity.topology[1:])
    changed = replace(original.identity, topology=changed_topology)
    changed_profile = replace(original, identity=changed)
    assert changed_profile.digest != original.digest
    assert any(
        path.startswith(f"identity.topology.0.{field}")
        for path in original.stale_reasons(changed_profile)
    )


def test_two_gpu_asymmetric_topology_must_match_plan() -> None:
    profile = _profile(two_gpus=True)
    assert profile.identity.topology[1].capacity_bytes == 600
    assert profile.plan.gpus[1].capacity_bytes == 600

    wrong = replace(
        profile.identity,
        topology=tuple(
            replace(item, capacity_bytes=item.capacity_bytes + (1 if item.rank == 1 else 0))
            for item in profile.identity.topology
        ),
    )
    with pytest.raises(PlacementPlannerError, match="capacity"):
        PlacementProfile(wrong, profile.plan, profile.backoff_profile)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.pop("identity"),
        lambda d: d.__setitem__("unknown", 1),
        lambda d: d.__setitem__("schema_version", 99),
        lambda d: d.__setitem__("digest", "0" * 64),
    ],
)
def test_profile_from_dict_rejects_missing_unknown_wrong_version_or_tamper(mutation) -> None:
    document = copy.deepcopy(_profile().as_dict())
    mutation(document)
    with pytest.raises(PlacementPlannerError):
        PlacementProfile.from_dict(document)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.__setitem__("schema_version", True),
        lambda d: d.__setitem__("schema_version", 1.0),
        lambda d: d["placement_plan"].__setitem__("gpu_count", True),
        lambda d: d["placement_plan"].__setitem__("gpu_count", 1.0),
        lambda d: d["placement_plan"]["gpus"][0].__setitem__("headroom_bytes", 0.0),
        lambda d: d["placement_plan"]["gpus"][0].__setitem__("deficit_bytes", False),
        lambda d: d["placement_plan"]["gpus_by_key"]["gpu-0:0"].__setitem__("required_bytes", True),
    ],
)
def test_profile_parser_rejects_numeric_aliases_without_mutating_original(mutation) -> None:
    profile = _profile()
    original_digest = profile.digest
    document = copy.deepcopy(profile.as_dict())
    mutation(document)
    with pytest.raises(PlacementPlannerError):
        PlacementProfile.from_dict(document)
    assert profile.digest == original_digest


def test_profile_rejects_noncanonical_backoff_names_and_placeholders() -> None:
    profile = _profile()
    with pytest.raises(PlacementPlannerError, match="canonical"):
        replace(
            profile,
            backoff_profile=replace(profile.backoff_profile, name="  cache-zero  "),
        )
    roundtrip = profile.as_dict()
    roundtrip["backoff_profile"]["name"] = " cache-zero "
    with pytest.raises(PlacementPlannerError, match="canonical"):
        PlacementProfile.from_dict(roundtrip)
    document = profile.as_dict()
    document["backoff_profile"]["name"] = " unknown "
    document["digest"] = profile.digest
    with pytest.raises(PlacementPlannerError, match="placeholder"):
        PlacementProfile.from_dict(document)


def test_overcommitted_plan_profile_roundtrips_and_rejects_tampered_headroom() -> None:
    profile = _profile(capacity=200)
    assert profile.plan.gpus[0].status == "insufficient-capacity"
    assert profile.plan.gpus[0].headroom_bytes < 0
    assert PlacementProfile.from_dict(profile.as_dict()) == profile

    document = profile.as_dict()
    document["placement_plan"]["gpus"][0]["headroom_bytes"] = 0
    with pytest.raises(PlacementPlannerError):
        PlacementProfile.from_dict(document)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.pop("model_sha256"),
        lambda d: d.__setitem__("unknown", 1),
        lambda d: d.__setitem__("model_sha256", "A" * 64),
        lambda d: d.__setitem__("runtime_commit", "0"),
        lambda d: d.__setitem__("context_geometry", {"tokens": True}),
        lambda d: d.__setitem__("topology", []),
    ],
)
def test_identity_from_dict_rejects_malformed_schema(mutation) -> None:
    document = copy.deepcopy(_profile().identity.as_dict())
    mutation(document)
    with pytest.raises(PlacementPlannerError):
        PlacementProfileIdentity.from_dict(document)


def test_profile_roundtrip_is_exact_and_rejects_noncanonical_json_values() -> None:
    profile = _profile(two_gpus=True)
    assert PlacementProfileIdentity.from_dict(profile.identity.as_dict()) == profile.identity
    assert PlacementProfile.from_dict(json.loads(json.dumps(profile.as_dict()))) == profile

    with pytest.raises(PlacementPlannerError, match="float"):
        canonical_json_bytes({"bad": 1.5})
    with pytest.raises(PlacementPlannerError, match="boolean"):
        replace(_identity(), context_geometry={"context_tokens": True, "batch_size": 1})


def test_geometry_and_topology_require_explicit_non_placeholder_values() -> None:
    with pytest.raises(PlacementPlannerError, match="placeholder"):
        replace(_identity(), driver_version="unknown")
    with pytest.raises(PlacementPlannerError, match="geometry"):
        replace(_identity(), qsa_geometry={})
    with pytest.raises(PlacementPlannerError, match="topology"):
        replace(_identity(), topology=())
    with pytest.raises(PlacementPlannerError, match="range"):
        GPUProfileTopology(
            rank=0,
            gpu_uuid="gpu-0",
            capacity_bytes=500,
            compute_capability="6.1",
            pci_bus_id="0000:01:00.0",
            numa_node=0,
            peer_ownership=(1 << 63,),
        )


def test_geometry_cannot_omit_required_fields_or_use_unknown_fields() -> None:
    identity = _identity()
    with pytest.raises(PlacementPlannerError, match="context_geometry"):
        replace(identity, context_geometry={"tag": "x"})
    with pytest.raises(PlacementPlannerError, match="state_geometry"):
        replace(identity, state_geometry={"tag": "x"})
    with pytest.raises(PlacementPlannerError, match="qsa_geometry"):
        replace(identity, qsa_geometry={"tag": "x"})


def test_qsa_geometry_is_reconstructed_and_validated() -> None:
    identity = _identity()
    with pytest.raises(PlacementPlannerError, match="phase"):
        replace(
            identity,
            qsa_geometry={
                **identity.qsa_geometry,
                "phase": "capture",
                "topk_backend": "torch",
            },
        )
    with pytest.raises(PlacementPlannerError, match="compression_ratio"):
        replace(identity, qsa_geometry={**identity.qsa_geometry, "compression_ratio": 3})


def test_geometry_cross_fields_must_match_plan_and_backoff() -> None:
    identity = _identity()
    with pytest.raises(PlacementPlannerError, match="context geometry"):
        PlacementProfile(
            replace(
                identity,
                context_geometry={**identity.context_geometry, "context_tokens": 2048},
            ),
            _plan(),
            BackoffProfile("cache-zero", cache_slots=0, context_tokens=1024, batch_size=1),
        )
    with pytest.raises(PlacementPlannerError, match="state geometry"):
        PlacementProfile(
            replace(
                identity,
                state_geometry={**identity.state_geometry, "expert_cache_slots": 1},
            ),
            _plan(),
            BackoffProfile("cache-zero", cache_slots=0, context_tokens=1024, batch_size=1),
        )
    with pytest.raises(PlacementPlannerError, match="request-slot"):
        PlacementProfile(
            replace(
                identity,
                state_geometry={**identity.state_geometry, "num_request_slots": 2},
            ),
            _plan(),
            BackoffProfile("cache-zero", cache_slots=0, context_tokens=1024, batch_size=1),
        )


def test_profile_module_is_torch_free() -> None:
    python_root = Path(__file__).resolve().parents[2] / "python"
    script = """
import json
import sys

before = set(sys.modules)
import freetoken.engine.placement_profile  # noqa: F401
added = sorted(
    name for name in set(sys.modules) - before if name == "torch" or name.startswith("torch.")
)
print(json.dumps({"torch_modules_added": added}))
if added:
    raise SystemExit(1)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(python_root)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=python_root.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "placement_profile imported torch modules in a clean interpreter or failed to import; "
        f"returncode={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}"
    )
