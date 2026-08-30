"""H0 tests for deterministic QSA workspace planning and capacity gates."""

from __future__ import annotations

import pytest
from freetoken.attention.qsa_workspace import (
    MAX_QSA_WORKSPACE_BYTES,
    QSAWorkspaceCapacityError,
    QSAWorkspaceInputError,
    QSAWorkspaceInputs,
    QSAWorkspaceInventory,
    calculate_qsa_workspace,
    validate_qsa_workspace_capacity,
)


def _inputs(**overrides: int) -> QSAWorkspaceInputs:
    values = {
        "context_tokens": 8,
        "token_rows": 2,
        "block_count": 2,
        "index_heads": 2,
        "query_heads": 4,
        "kv_heads": 2,
        "head_dim": 16,
        "index_head_dim": 8,
        "top_k": 4,
        "dtype_bytes": 2,
        "compression_ratio": 4,
        "attention_splits": 2,
        "num_index_layers": 2,
        "num_req_slots": 3,
        "ring_capacity": 4,
        "num_pages": 1,
        "page_size": 8,
    }
    values.update(overrides)
    return QSAWorkspaceInputs(**values)


def test_inventory_uses_actual_qsa_shapes_and_all_categories() -> None:
    plan = calculate_qsa_workspace(_inputs())

    assert set(plan.inventory) == {"score", "top_k", "expand_gather", "attention", "state"}
    assert plan.inventory.score.components["q_index"] == 2 * 2 * 8 * 2
    assert plan.inventory.score.components["logits"] == 2 * 2 * 4
    assert plan.inventory.top_k.components["blocks"] == 2 * (4 // 4) * 4
    assert plan.inventory.expand_gather.components["indices"] == 2 * (4 + 4 - 1) * 4
    assert plan.inventory.attention.components["partial_output"] == 2 * 2 * 4 * 16 * 4
    assert plan.inventory.attention.components["partial_lse"] == 2 * 2 * 4 * 4
    assert plan.inventory.state.components["compressed_slab"] == (2 * (8 // 4 + 3) * 8 * 2)
    assert plan.required_bytes == sum(category.bytes for category in plan.inventory.values())


def test_incomplete_group_keeps_complete_context_blocks_and_tail_shape() -> None:
    plan = calculate_qsa_workspace(_inputs(context_tokens=7, block_count=1))

    assert plan.request.context_tokens == 7
    assert plan.request.block_count == 1
    assert plan.inventory.top_k.components["blocks"] == 8
    assert plan.inventory.expand_gather.components["indices"] == 2 * 7 * 4


def test_zero_and_negative_dimensions_fail_closed() -> None:
    for field in ("context_tokens", "token_rows", "block_count", "index_heads", "dtype_bytes"):
        with pytest.raises(QSAWorkspaceInputError, match=field):
            QSAWorkspaceInputs(**{**_inputs().__dict__, field: 0})
        with pytest.raises(QSAWorkspaceInputError, match=field):
            QSAWorkspaceInputs(**{**_inputs().__dict__, field: -1})

    with pytest.raises(QSAWorkspaceInputError, match="divisible"):
        _inputs(top_k=5)
    with pytest.raises(QSAWorkspaceInputError, match="grouped-query"):
        _inputs(query_heads=3)


def test_missing_or_unknown_inventory_categories_fail_closed() -> None:
    complete = {name: 1 for name in ("score", "top_k", "expand_gather", "attention", "state")}
    with pytest.raises(QSAWorkspaceInputError, match="missing"):
        QSAWorkspaceInventory.from_mapping(
            {key: value for key, value in complete.items() if key != "state"}
        )
    with pytest.raises(QSAWorkspaceInputError, match="unknown"):
        QSAWorkspaceInventory.from_mapping({**complete, "other": 1})


def test_capacity_is_checked_before_launch_and_reports_controlled_error() -> None:
    plan = calculate_qsa_workspace(_inputs())
    with pytest.raises(QSAWorkspaceCapacityError) as raised:
        plan.validate_capacity(plan.required_bytes - 1)

    error = raised.value
    assert error.required_bytes == plan.required_bytes
    assert error.capacity_bytes == plan.required_bytes - 1
    assert error.deficit_bytes == 1
    assert error.telemetry.status == "insufficient-capacity"

    telemetry = validate_qsa_workspace_capacity(_inputs(), plan.required_bytes)
    assert telemetry.status == "ready"
    assert telemetry.headroom_bytes == 0


def test_capacity_zero_is_a_controlled_insufficient_capacity_error() -> None:
    with pytest.raises(QSAWorkspaceCapacityError, match="capacity"):
        validate_qsa_workspace_capacity(_inputs(), 0)


def test_maximum_values_reject_arithmetic_overflow_without_allocating() -> None:
    with pytest.raises(QSAWorkspaceInputError, match="overflow"):
        calculate_qsa_workspace(_inputs(token_rows=MAX_QSA_WORKSPACE_BYTES, index_heads=2))
    with pytest.raises(QSAWorkspaceInputError, match="overflow"):
        calculate_qsa_workspace(_inputs(num_pages=MAX_QSA_WORKSPACE_BYTES, page_size=8))


def test_plan_and_telemetry_are_structured_and_serializable() -> None:
    plan = calculate_qsa_workspace(_inputs())
    telemetry = plan.telemetry
    payload = telemetry.as_dict()

    assert payload["required_bytes"] == plan.required_bytes
    assert payload["status"] == "unvalidated"
    assert payload["categories"]["score"] == plan.inventory.score.bytes
    assert payload["shapes"]["score"]["logits"] == (2, 2)
    assert plan.as_dict()["request"]["context_tokens"] == 8
