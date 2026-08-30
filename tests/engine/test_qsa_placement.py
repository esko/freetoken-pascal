"""H0 tests for binding QSA workspace accounting to placement plans."""

from __future__ import annotations

from dataclasses import replace

import pytest
from freetoken.engine.placement_plan import (
    PLACEMENT_CATEGORIES,
    BackoffProfile,
    PlacementPlanInput,
    PlacementPlannerError,
    plan_placement,
)
from freetoken.engine.placement_profile import (
    GPUProfileTopology,
    PlacementProfile,
    PlacementProfileIdentity,
)
from freetoken.engine.qsa_placement import (
    QSA_PLACEMENT_CATEGORIES,
    QSAPlacementBinding,
    QSAWorkspaceInputs,
    bind_qsa_workspace,
    calculate_qsa_workspace,
    derive_qsa_placement_categories,
)


def _inputs(**overrides: int) -> QSAWorkspaceInputs:
    values = {
        "context_tokens": 8,
        "token_rows": 2,
        "page_table_width": 4,
        "page_size": 8,
        "index_heads": 2,
        "query_heads": 4,
        "kv_heads": 2,
        "head_dim": 16,
        "index_head_dim": 8,
        "top_k": 4,
        "dtype_bytes": 2,
        "compression_ratio": 4,
        "num_index_layers": 2,
        "num_req_slots": 3,
        "ring_capacity": 4,
        "num_pages": 1,
        "max_position": 16,
        "rotary_dim": 8,
        "batch_size": 2,
    }
    values.update(overrides)
    return QSAWorkspaceInputs(**values)


def _qsa_categories(workspace) -> dict[str, int]:
    state = workspace.inventory["state"]
    persistent_state = workspace.persistent_bytes
    transient_state = state.bytes - persistent_state
    return {
        "qsa_persistent_score": 0,
        "qsa_persistent_top_k": 0,
        "qsa_persistent_expand_gather": 0,
        "qsa_persistent_attention": 0,
        "qsa_persistent_state": persistent_state,
        "qsa_transient_score": workspace.inventory["score"].bytes,
        "qsa_transient_top_k": workspace.inventory["top_k"].bytes,
        "qsa_transient_expand_gather": workspace.inventory["expand_gather"].bytes,
        "qsa_transient_attention": workspace.inventory["attention"].bytes,
        "qsa_transient_state": transient_state,
    }


def _gpu_categories(workspace, **overrides: int) -> dict[str, int]:
    values = {name: 0 for name in PLACEMENT_CATEGORIES}
    values.update(
        {
            "dense_resident_weights": 100,
            "shared_experts": 20,
            "gdn_kv_recurrent_state": 30,
            "cuda_context": 15,
            "generic_workspaces": 16,
            "transfer_buffers": 17,
            "static_expert_cache_slots": 18,
            "dynamic_expert_cache_slots": 19,
            "safety_reserve": 50,
        }
    )
    values.update(_qsa_categories(workspace))
    values.update(overrides)
    return values


def _gpu(workspace, **overrides: int):
    categories = _gpu_categories(workspace, **overrides)
    return plan_placement(
        (
            PlacementPlanInput(
                capacity_bytes=1 << 40,
                gpu_uuid="gpu-0",
                categories=categories,
            ),
        ),
        safety_reserve_bytes=50,
    ).gpus[0]


def _profile(workspace, *, qsa_geometry=None) -> PlacementProfile:
    identity = PlacementProfileIdentity(
        model_sha256="a" * 64,
        quant_census_sha256="b" * 64,
        binary_sha256="c" * 64,
        toolchain_sha256="d" * 64,
        runtime_commit="e" * 40,
        runtime_version="fixture-runtime-1",
        driver_version="550.1",
        cuda_runtime_version="12.6.3",
        cuda_toolchain_identity="cuda-12.6.3-sm61",
        context_geometry={
            "context_tokens": 8,
            "batch_size": 2,
            "prefill_chunk_tokens": 8,
            "microbatch_size": 2,
        },
        state_geometry={
            "num_request_slots": 3,
            "kv_page_size": 8,
            "kv_pages": 1,
            "gdn_state_bytes": 1024,
            "kv_state_bytes": 2048,
            "expert_cache_slots": 0,
        },
        qsa_geometry=qsa_geometry
        or {
            name: getattr(workspace.request, name)
            for name in workspace.request.__dataclass_fields__
        },
        topology=(
            GPUProfileTopology(
                rank=0,
                gpu_uuid="gpu-0",
                capacity_bytes=1 << 40,
                compute_capability="6.1",
                pci_bus_id="0000:01:00.0",
                numa_node=0,
            ),
        ),
    )
    return PlacementProfile(
        identity=identity,
        plan=plan_placement(
            (
                PlacementPlanInput(
                    capacity_bytes=1 << 40,
                    gpu_uuid="gpu-0",
                    categories=_gpu_categories(workspace),
                ),
            ),
            safety_reserve_bytes=50,
        ),
        backoff_profile=BackoffProfile("fixture", cache_slots=0, context_tokens=8, batch_size=2),
    )


def test_projection_uses_exact_ten_qsa_categories_and_lifetimes() -> None:
    workspace = calculate_qsa_workspace(_inputs())

    categories = derive_qsa_placement_categories(workspace)

    assert tuple(categories) == QSA_PLACEMENT_CATEGORIES
    assert categories["qsa_persistent_state"] == workspace.persistent_bytes
    assert categories["qsa_transient_state"] == (
        workspace.inventory["state"].bytes - workspace.persistent_bytes
    )
    assert sum(categories.values()) == (
        workspace.persistent_bytes
        + sum(
            workspace.inventory[name].bytes
            for name in ("score", "top_k", "expand_gather", "attention")
        )
        + categories["qsa_transient_state"]
    )


def test_binding_recomputes_workspace_and_preserves_live_peak_semantics() -> None:
    workspace = calculate_qsa_workspace(_inputs())
    binding = bind_qsa_workspace(workspace, _gpu(workspace))

    assert isinstance(binding, QSAPlacementBinding)
    assert binding.categories == _qsa_categories(workspace)
    assert binding.persistent_bytes == workspace.persistent_bytes
    assert binding.live_bytes == workspace.persistent_bytes + workspace.eager_transient_peak_bytes
    assert binding.peak_bytes == workspace.required_bytes
    assert binding.placement_qsa_required_bytes == sum(binding.categories.values())


def test_binding_rejects_arbitrary_qsa_bytes_and_plan_category_drift() -> None:
    workspace = calculate_qsa_workspace(_inputs())
    gpu = _gpu(workspace)

    with pytest.raises(PlacementPlannerError, match="QSA workspace"):
        bind_qsa_workspace({"required_bytes": workspace.required_bytes}, gpu)
    drifted = plan_placement(
        (
            PlacementPlanInput(
                capacity_bytes=1 << 40,
                gpu_uuid="gpu-0",
                categories={**gpu.categories, "qsa_transient_score": 1},
            ),
        ),
        safety_reserve_bytes=50,
    ).gpus[0]
    with pytest.raises(PlacementPlannerError, match="qsa_transient_score"):
        bind_qsa_workspace(workspace, drifted)


def test_binding_checks_profile_geometry_before_accepting_plan() -> None:
    workspace = calculate_qsa_workspace(_inputs())
    profile = _profile(workspace)
    assert bind_qsa_workspace(workspace, profile).profile_digest == profile.digest

    stale = dict(profile.identity.qsa_geometry)
    stale["token_rows"] += 1
    with pytest.raises(PlacementPlannerError, match="geometry"):
        bind_qsa_workspace(
            workspace,
            replace(profile, identity=replace(profile.identity, qsa_geometry=stale)),
        )
