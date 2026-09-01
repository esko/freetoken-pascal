"""H0 tests for binding QSA workspace accounting to placement plans."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
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
    return dict(derive_qsa_placement_categories(workspace))


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
    assert categories == {
        "qsa_persistent_score": 0,
        "qsa_persistent_top_k": 0,
        "qsa_persistent_expand_gather": 0,
        "qsa_persistent_attention": 0,
        "qsa_persistent_state": 1056,
        "qsa_transient_score": 52,
        "qsa_transient_top_k": 0,
        "qsa_transient_expand_gather": 56,
        "qsa_transient_attention": 256,
        "qsa_transient_state": 16,
    }
    assert sum(categories.values()) == workspace.required_bytes == 1436


def test_vectorized_projection_includes_per_head_score_tile() -> None:
    workspace = calculate_qsa_workspace(
        _inputs(
            context_tokens=8192,
            page_table_width=8192,
            num_pages=1024,
            qsa_selection_path="torch-fp32-vectorized-reference",
        )
    )

    categories = derive_qsa_placement_categories(workspace)
    score = workspace.inventory["score"].components

    assert score["vector_score_heads"] > 0
    assert categories["qsa_transient_score"] >= score["vector_score_heads"]


def test_capture_projection_partitions_graph_and_dynamic_buffers() -> None:
    workspace = calculate_qsa_workspace(_inputs(phase="capture", capture_max_batch_size=3))

    categories = derive_qsa_placement_categories(workspace)

    assert categories == {
        "qsa_persistent_score": 196,
        "qsa_persistent_top_k": 12,
        "qsa_persistent_expand_gather": 84,
        "qsa_persistent_attention": 0,
        "qsa_persistent_state": 1116,
        "qsa_transient_score": 8,
        "qsa_transient_top_k": 0,
        "qsa_transient_expand_gather": 0,
        "qsa_transient_attention": 256,
        "qsa_transient_state": 16,
    }
    assert sum(categories.values()) == workspace.required_bytes == 1688


def test_binding_recomputes_workspace_and_preserves_live_peak_semantics() -> None:
    workspace = calculate_qsa_workspace(_inputs())
    binding = bind_qsa_workspace(workspace, _gpu(workspace))

    assert isinstance(binding, QSAPlacementBinding)
    assert binding.categories == _qsa_categories(workspace)
    assert binding.persistent_bytes == workspace.persistent_bytes
    assert binding.live_bytes == workspace.persistent_bytes + workspace.eager_transient_peak_bytes
    assert binding.peak_bytes == workspace.required_bytes
    assert binding.placement_qsa_required_bytes == sum(binding.categories.values())
    assert (
        binding.placement_persistent_bytes + binding.placement_transient_high_water_bytes
        == workspace.required_bytes
    )


def test_binding_accepts_exact_capacity_and_rejects_one_byte_less() -> None:
    workspace = calculate_qsa_workspace(_inputs())
    categories = _gpu_categories(workspace)
    template = plan_placement(
        (PlacementPlanInput(capacity_bytes=1 << 40, gpu_uuid="gpu-0", categories=categories),),
        safety_reserve_bytes=50,
    )
    exact = plan_placement(
        (
            PlacementPlanInput(
                capacity_bytes=template.gpus[0].required_bytes,
                gpu_uuid="gpu-0",
                categories=categories,
            ),
        ),
        safety_reserve_bytes=50,
    ).gpus[0]
    assert bind_qsa_workspace(workspace, exact).placement_qsa_required_bytes == 1436

    insufficient = plan_placement(
        (
            PlacementPlanInput(
                capacity_bytes=template.gpus[0].required_bytes - 1,
                gpu_uuid="gpu-0",
                categories=categories,
            ),
        ),
        safety_reserve_bytes=50,
    ).gpus[0]
    with pytest.raises(PlacementPlannerError, match="capacity-safe"):
        bind_qsa_workspace(workspace, insufficient)


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


def test_binding_rejects_duplicate_full_qsa_bytes_on_two_gpus() -> None:
    workspace = calculate_qsa_workspace(_inputs())
    categories = _gpu_categories(workspace)
    plan = plan_placement(
        (
            PlacementPlanInput(capacity_bytes=1 << 40, gpu_uuid="gpu-0", categories=categories),
            PlacementPlanInput(capacity_bytes=1 << 40, gpu_uuid="gpu-1", categories=categories),
        ),
        safety_reserve_bytes=50,
    )
    with pytest.raises(PlacementPlannerError, match="ownership/partition"):
        bind_qsa_workspace(workspace, plan)


def test_direct_binding_rejects_tampering_invalid_digest_and_bad_gpu_set() -> None:
    workspace = calculate_qsa_workspace(_inputs())
    gpu = _gpu(workspace)
    categories = derive_qsa_placement_categories(workspace)
    binding = QSAPlacementBinding(workspace, categories, (gpu,), "a" * 64)

    tampered = replace(workspace, telemetry=replace(workspace.telemetry, status="ready"))
    with pytest.raises(PlacementPlannerError, match=r"caller-supplied|inconsistent"):
        QSAPlacementBinding(tampered, categories, (gpu,))
    with pytest.raises(PlacementPlannerError, match="SHA-256"):
        replace(binding, profile_digest="g" * 64)
    with pytest.raises(PlacementPlannerError, match="exactly one or two"):
        QSAPlacementBinding(workspace, categories, (gpu, gpu, gpu))
    with pytest.raises(PlacementPlannerError, match="contiguous"):
        QSAPlacementBinding(
            workspace,
            categories,
            (replace(gpu, rank=1, gpu_uuid="gpu-1"),),
        )
    with pytest.raises(PlacementPlannerError, match="unique"):
        QSAPlacementBinding(workspace, categories, (gpu, replace(gpu, rank=1)))
    with pytest.raises(PlacementPlannerError, match="ownership/partition"):
        QSAPlacementBinding(workspace, categories, (gpu, replace(gpu, rank=1, gpu_uuid="gpu-1")))


def test_binding_accepts_already_loaded_canonical_runtime_qsa_types() -> None:
    source = Path(__file__).resolve().parents[2] / "python/freetoken/attention/qsa_workspace.py"
    spec = spec_from_file_location("freetoken.attention.qsa_workspace", source)
    assert spec is not None and spec.loader is not None
    runtime_module = module_from_spec(spec)
    sys.modules[spec.name] = runtime_module
    spec.loader.exec_module(runtime_module)

    standalone = _inputs()
    runtime_inputs = runtime_module.QSAWorkspaceInputs(
        **{name: getattr(standalone, name) for name in standalone.__dataclass_fields__}
    )
    runtime_workspace = runtime_module.calculate_qsa_workspace(runtime_inputs)
    gpu = _gpu(standalone)
    categories = _qsa_categories(standalone)

    try:
        assert bind_qsa_workspace(runtime_inputs, gpu).peak_bytes == 1436
        binding = QSAPlacementBinding(runtime_workspace, categories, (gpu,))
        assert binding.peak_bytes == 1436
    finally:
        del sys.modules[spec.name]


def test_qsa_placement_module_imports_without_torch() -> None:
    script = """
import types
import sys
sys.modules['torch'] = types.ModuleType('torch')
import freetoken.engine.qsa_placement
assert not any(name.startswith('freetoken.attention') for name in sys.modules)
assert [name for name in sys.modules if name == 'torch' or name.startswith('torch.')] == ['torch']
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


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
