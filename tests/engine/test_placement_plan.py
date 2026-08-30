"""H0 tests for the immutable per-GPU placement planner and canary contract."""

from __future__ import annotations

from dataclasses import replace

import pytest
from freetoken.engine.placement_plan import (
    PLACEMENT_CATEGORIES,
    BackoffProfile,
    BackoffStateMachine,
    CanaryResult,
    PlacementObservation,
    PlacementPlanInput,
    PlacementPlannerError,
    evaluate_canary,
    plan_placement,
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


def _gpu(*, capacity: int = 500, available: int | None = None, **overrides: int):
    return PlacementPlanInput(
        capacity_bytes=capacity,
        available_bytes=available,
        categories=_categories(**overrides),
    )


def _observation(plan, *, rank: int = 0, **overrides):
    categories = dict(plan.gpus[rank].categories)
    categories.update(overrides.pop("categories", {}))
    transient = sum(
        categories[name] for name in PLACEMENT_CATEGORIES if name.startswith("qsa_transient_")
    )
    peak = sum(categories.values()) - categories["safety_reserve"]
    allocated = peak - transient
    allocated = overrides.pop("allocator_allocated_bytes", allocated)
    reserved = overrides.pop("allocator_reserved_bytes", peak)
    free = overrides.pop(
        "driver_free_bytes",
        plan.gpus[rank].capacity_bytes - peak,
    )
    return PlacementObservation(
        rank=rank,
        gpu_uuid=plan.gpus[rank].gpu_uuid,
        driver_total_bytes=plan.gpus[rank].capacity_bytes,
        driver_free_bytes=free,
        allocator_allocated_bytes=allocated,
        allocator_reserved_bytes=reserved,
        allocator_high_water_bytes=overrides.pop("allocator_high_water_bytes", peak),
        categories=categories,
        **overrides,
    )


def test_plan_has_exact_release_categories_per_rank_and_structured_telemetry() -> None:
    plan = plan_placement((_gpu(), _gpu(capacity=600)), safety_reserve_bytes=50)

    assert plan.gpu_count == 2
    assert tuple(plan.gpus[0].categories) == PLACEMENT_CATEGORIES
    assert tuple(plan.gpus[1].categories) == PLACEMENT_CATEGORIES
    assert plan.gpus[0].required_bytes == sum(_categories().values())
    assert plan.gpus[0].headroom_bytes == 500 - plan.gpus[0].required_bytes
    assert plan.telemetry[0].as_dict() == {
        "schema_version": 1,
        "rank": 0,
        "gpu_uuid": "unknown",
        "key": "unknown:0",
        "status": "ready",
        "required_bytes": plan.gpus[0].required_bytes,
        "non_qsa_required_bytes": plan.gpus[0].non_qsa_required_bytes,
        "qsa_persistent_bytes": plan.gpus[0].qsa_persistent_bytes,
        "qsa_transient_high_water_bytes": plan.gpus[0].qsa_transient_high_water_bytes,
        "qsa_required_bytes": plan.gpus[0].qsa_required_bytes,
        "live_required_bytes": plan.gpus[0].live_required_bytes,
        "peak_required_bytes": plan.gpus[0].peak_required_bytes,
        "available_bytes": 500,
        "headroom_bytes": plan.gpus[0].headroom_bytes,
        "deficit_bytes": 0,
        "categories": dict(plan.gpus[0].categories),
        "reasons": [],
    }
    with pytest.raises(TypeError):
        plan.gpus[0].categories["cuda_context"] = 1


def test_cache_zero_and_asymmetric_capacity_remain_usable() -> None:
    plan = plan_placement(
        (
            _gpu(static_expert_cache_slots=0, dynamic_expert_cache_slots=0),
            _gpu(capacity=600, static_expert_cache_slots=0, dynamic_expert_cache_slots=0),
        ),
        safety_reserve_bytes=50,
    )

    assert all(item.status == "ready" for item in plan.telemetry)
    assert plan.gpus[0].categories["static_expert_cache_slots"] == 0
    assert plan.gpus[0].categories["dynamic_expert_cache_slots"] == 0


def test_qsa_lifetimes_and_gpu_identity_are_explicit() -> None:
    first = PlacementPlanInput(capacity_bytes=500, categories=_categories(), gpu_uuid="p4-uuid-0")
    second = PlacementPlanInput(
        capacity_bytes=600,
        categories=_categories(qsa_transient_attention=23),
        gpu_uuid="p4-uuid-1",
    )
    plan = plan_placement((first, second), safety_reserve_bytes=50)

    item = plan.gpus[0]
    assert item.key == "p4-uuid-0:0"
    assert plan.gpus[1].key == "p4-uuid-1:1"
    assert tuple(plan.by_key) == ("p4-uuid-0:0", "p4-uuid-1:1")
    assert item.required_bytes == (
        item.non_qsa_required_bytes
        + item.qsa_persistent_bytes
        + item.qsa_transient_high_water_bytes
        + plan.safety_reserve_bytes
    )
    assert item.as_dict()["qsa_required_bytes"] == (
        item.qsa_persistent_bytes + item.qsa_transient_high_water_bytes
    )
    assert item.live_required_bytes == (item.non_qsa_required_bytes + item.qsa_persistent_bytes)
    assert item.peak_required_bytes == item.nonreserve_required_bytes


def test_input_categories_and_integer_values_fail_closed() -> None:
    complete = _categories()
    with pytest.raises(PlacementPlannerError, match="missing"):
        PlacementPlanInput(
            capacity_bytes=500,
            categories={key: value for key, value in complete.items() if key != "cuda_context"},
        )
    with pytest.raises(PlacementPlannerError, match="unknown"):
        PlacementPlanInput(capacity_bytes=500, categories={**complete, "other": 1})
    legacy = {key: value for key, value in complete.items() if key != "gdn_kv_recurrent_state"}
    legacy["gdn_qsa_kv_state"] = 30
    with pytest.raises(PlacementPlannerError, match=r"missing|unknown"):
        PlacementPlanInput(capacity_bytes=500, categories=legacy)

    for field, value in (("capacity_bytes", True), ("capacity_bytes", -1)):
        with pytest.raises(PlacementPlannerError, match=field):
            PlacementPlanInput(**{field: value, "categories": complete})
    with pytest.raises(PlacementPlannerError, match="1 or 2"):
        plan_placement(())
    with pytest.raises(PlacementPlannerError, match="1 or 2"):
        plan_placement((_gpu(), _gpu(), _gpu()))


def test_overcommit_is_reported_with_deficit_and_does_not_look_ready() -> None:
    plan = plan_placement((_gpu(capacity=300),), safety_reserve_bytes=50)
    telemetry = plan.telemetry[0]

    assert telemetry.status == "insufficient-capacity"
    assert telemetry.deficit_bytes == plan.gpus[0].required_bytes - 300
    assert telemetry.headroom_bytes < 0
    assert "insufficient" in telemetry.reasons[0]


def test_canary_accepts_bounded_category_drift_and_reports_checkpoint() -> None:
    plan = plan_placement((_gpu(),), safety_reserve_bytes=50)
    observed = _observation(plan, categories={"qsa_transient_score": 12})

    result = evaluate_canary(
        plan,
        checkpoint="post-first-large-prefill",
        observations=(observed,),
        tolerance_bytes=2,
    )

    assert result.status == "pass"
    assert result.checkpoint == "post-first-large-prefill"
    assert result.gpus[0].headroom_bytes == observed.driver_free_bytes - plan.safety_reserve_bytes
    assert result.gpus[0].allocator_allocated_bytes == plan.gpus[0].live_required_bytes
    assert result.gpus[0].allocator_high_water_bytes == observed.allocator_high_water_bytes
    assert result.as_dict()["checkpoint"] == "post-first-large-prefill"


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"managed_memory": True}, "managed-memory"),
        ({"host_spill": True}, "host-spill"),
        ({"fallback": True}, "fallback"),
        ({"allocation_retries": 1}, "allocation-retries"),
        ({"allocation_failures": 1}, "allocation-failures"),
        ({"cache_overcommit": True}, "cache-overcommit"),
        ({"unplanned_placement": True}, "unplanned-placement"),
        ({"retained_workspace_growth_bytes": 3}, "retained-workspace-growth"),
    ],
)
def test_canary_rejects_unsafe_observation_flags(kwargs, reason: str) -> None:
    plan = plan_placement((_gpu(),), safety_reserve_bytes=50)
    result = evaluate_canary(
        plan,
        checkpoint="post-load",
        observations=(_observation(plan, **kwargs),),
        tolerance_bytes=2,
    )

    assert result.status == "fail"
    assert any(reason in item for item in result.reasons)


def test_canary_rejects_allocator_inconsistency_missing_categories_and_low_reserve() -> None:
    plan = plan_placement((_gpu(),), safety_reserve_bytes=50)
    bad = _observation(plan, allocator_allocated_bytes=10, allocator_reserved_bytes=9)
    result = evaluate_canary(plan, checkpoint="post-load", observations=(bad,))
    assert result.status == "fail"
    assert any("allocated" in item and "reserved" in item for item in result.reasons)

    missing = dict(plan.gpus[0].categories)
    missing.pop("qsa_persistent_state")
    with pytest.raises(PlacementPlannerError, match="missing"):
        PlacementObservation(
            rank=0,
            gpu_uuid="unknown",
            driver_total_bytes=500,
            driver_free_bytes=499,
            allocator_allocated_bytes=1,
            allocator_reserved_bytes=1,
            allocator_high_water_bytes=1,
            categories=missing,
        )

    low = _observation(plan, driver_free_bytes=49)
    result = evaluate_canary(plan, checkpoint="post-load", observations=(low,))
    assert result.status == "fail"
    assert any("reserve" in item for item in result.reasons)


def test_canary_bounds_tolerance_and_requires_exact_driver_allocator_consistency() -> None:
    plan = plan_placement((_gpu(),), safety_reserve_bytes=50)
    with pytest.raises(PlacementPlannerError, match=r"tolerance.*reserve"):
        evaluate_canary(
            plan,
            checkpoint="post-load",
            observations=(_observation(plan),),
            tolerance_bytes=51,
        )

    inconsistent = _observation(plan, driver_free_bytes=171)
    result = evaluate_canary(
        plan,
        checkpoint="post-load",
        observations=(inconsistent,),
        tolerance_bytes=50,
    )
    assert result.status == "fail"
    assert any("consistency" in reason for reason in result.reasons)


def test_canary_telemetry_and_result_schema_fail_closed() -> None:
    plan = plan_placement((_gpu(),), safety_reserve_bytes=50)
    result = evaluate_canary(
        plan,
        checkpoint="post-load",
        observations=(_observation(plan),),
    )
    telemetry = result.gpus[0]

    with pytest.raises(PlacementPlannerError, match="rank"):
        replace(telemetry, rank=True)
    with pytest.raises(PlacementPlannerError, match="driver_free_bytes"):
        replace(telemetry, driver_free_bytes=-1)
    with pytest.raises(PlacementPlannerError, match=r"missing|unknown"):
        replace(telemetry, planned_categories={"dense_resident_weights": 1})
    with pytest.raises(PlacementPlannerError, match=r"status.*reasons"):
        replace(telemetry, status="pass", reasons=("unexpected",))

    with pytest.raises(PlacementPlannerError, match="exactly 1 or 2"):
        CanaryResult("post-load", "pass", ())
    with pytest.raises(PlacementPlannerError, match="status"):
        CanaryResult("post-load", "pass", (replace(telemetry, status="fail", reasons=("bad",)),))
    with pytest.raises(PlacementPlannerError, match="exactly 1 or 2"):
        CanaryResult("post-load", "fail", (telemetry, telemetry, telemetry))


def test_canary_requires_exact_checkpoint_and_rejects_unavailable_high_water() -> None:
    plan = plan_placement((_gpu(),), safety_reserve_bytes=50)
    with pytest.raises(PlacementPlannerError, match="unknown checkpoint"):
        evaluate_canary(plan, checkpoint="unknown", observations=(_observation(plan),))

    with pytest.raises(PlacementPlannerError, match="driver_total_bytes"):
        PlacementObservation(
            rank=0,
            gpu_uuid="unknown",
            driver_total_bytes=None,
            driver_free_bytes=100,
            allocator_allocated_bytes=1,
            allocator_reserved_bytes=1,
            allocator_high_water_bytes=1,
            categories=_categories(),
        )

    high = _observation(plan, allocator_high_water_bytes=501)
    result = evaluate_canary(plan, checkpoint="post-load", observations=(high,))
    assert result.status == "fail"
    assert any("high-water" in item for item in result.reasons)


def test_canary_distinguishes_live_allocation_from_qsa_transient_peak() -> None:
    plan = plan_placement((_gpu(),), safety_reserve_bytes=50)
    live = plan.gpus[0].live_required_bytes
    peak = plan.gpus[0].peak_required_bytes

    passing = _observation(plan, allocator_allocated_bytes=live, allocator_high_water_bytes=peak)
    assert (
        evaluate_canary(
            plan,
            checkpoint="post-first-large-prefill",
            observations=(passing,),
        ).status
        == "pass"
    )

    under_materialized = _observation(
        plan,
        allocator_allocated_bytes=live - 1,
        allocator_high_water_bytes=peak,
    )
    assert (
        evaluate_canary(
            plan,
            checkpoint="post-first-large-prefill",
            observations=(under_materialized,),
        ).status
        == "fail"
    )
    assert any(
        "live" in reason
        for reason in evaluate_canary(
            plan,
            checkpoint="post-first-large-prefill",
            observations=(under_materialized,),
        ).reasons
    )

    conflated = _observation(plan, allocator_allocated_bytes=peak, allocator_high_water_bytes=peak)
    result = evaluate_canary(
        plan,
        checkpoint="post-first-large-prefill",
        observations=(conflated,),
        tolerance_bytes=0,
    )
    assert result.status == "fail"
    assert any("live" in reason for reason in result.reasons)


def test_synthetic_observation_factory_keeps_resident_and_peak_distinct() -> None:
    plan = plan_placement((_gpu(),), safety_reserve_bytes=50)
    observation = PlacementObservation.from_plan(plan.gpus[0])

    assert observation.allocator_allocated_bytes == plan.gpus[0].live_required_bytes
    assert observation.allocator_reserved_bytes == plan.gpus[0].peak_required_bytes
    assert observation.allocator_high_water_bytes == plan.gpus[0].peak_required_bytes
    assert (
        evaluate_canary(
            plan,
            checkpoint="post-load",
            observations=(observation,),
        ).status
        == "pass"
    )


def test_backoff_starts_pending_and_becomes_safe_only_after_a_pass() -> None:
    plan = plan_placement((_gpu(),), safety_reserve_bytes=50)
    profiles = (
        BackoffProfile("full-cache", cache_slots=4, context_tokens=4096, batch_size=4),
        BackoffProfile("cache-zero", cache_slots=0, context_tokens=2048, batch_size=1),
    )
    machine = plan.backoff(profiles)
    assert machine.status == "pending"
    assert not machine.ready

    failed = evaluate_canary(
        plan,
        checkpoint="post-load",
        observations=(_observation(plan, fallback=True),),
    )
    decision = machine.observe(failed)
    assert decision.status == "backoff"
    assert machine.status == "pending"
    assert not machine.ready

    passed = evaluate_canary(
        plan,
        checkpoint="post-first-large-prefill",
        observations=(_observation(plan),),
    )
    decision = machine.observe(passed)
    assert decision.status == "safe"
    assert machine.status == "safe"
    assert machine.ready

    failed_later = evaluate_canary(
        plan,
        checkpoint="post-first-large-prefill",
        observations=(_observation(plan, fallback=True),),
    )
    decision = machine.observe(failed_later)
    assert decision.status == "fail-readiness"
    assert machine.status == "fail-readiness"
    assert not machine.ready


def test_backoff_rejects_identical_resource_profiles() -> None:
    with pytest.raises(PlacementPlannerError, match="strict progress"):
        BackoffStateMachine(
            (
                BackoffProfile("a", cache_slots=2, context_tokens=1024, batch_size=1),
                BackoffProfile("b", cache_slots=2, context_tokens=1024, batch_size=1),
            )
        )

    machine = BackoffStateMachine(
        (
            BackoffProfile("a", cache_slots=2, context_tokens=1024, batch_size=1),
            BackoffProfile("b", cache_slots=1, context_tokens=1024, batch_size=1),
        )
    )
    with pytest.raises(PlacementPlannerError, match="boolean or CanaryResult"):
        machine.observe(1)


def test_backoff_profiles_are_ordered_and_never_oscillate() -> None:
    plan = plan_placement((_gpu(),), safety_reserve_bytes=50)
    profiles = (
        BackoffProfile("full-cache", cache_slots=4, context_tokens=4096, batch_size=4),
        BackoffProfile("small-cache", cache_slots=2, context_tokens=4096, batch_size=4),
        BackoffProfile("cache-zero", cache_slots=0, context_tokens=2048, batch_size=1),
    )
    machine = plan.backoff(profiles)
    failed = evaluate_canary(
        plan,
        checkpoint="post-load",
        observations=(_observation(plan, fallback=True),),
    )

    first = machine.observe(failed)
    second = machine.observe(failed)
    third = machine.observe(failed)
    fourth = machine.observe(failed)

    assert (first.profile.name, second.profile.name, third.profile.name) == (
        "small-cache",
        "cache-zero",
        "cache-zero",
    )
    assert first.status == "backoff"
    assert second.status == "backoff"
    assert third.status == "fail-readiness"
    assert fourth.status == "fail-readiness"
