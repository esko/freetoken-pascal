"""H0 tests for deterministic QSA workspace planning and capacity gates."""

from __future__ import annotations

import pytest
from freetoken.attention.qsa_workspace import (
    MAX_QSA_WORKSPACE_BYTES,
    QSA_LOGITS_WORKSPACE_BYTES,
    QSAWorkspaceCapacityError,
    QSAWorkspaceCategory,
    QSAWorkspaceInputError,
    QSAWorkspaceInputs,
    QSAWorkspaceInventory,
    calculate_qsa_workspace,
    qsa_topk_scratch_width,
    validate_qsa_workspace_capacity,
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


def test_inventory_uses_actual_qsa_shapes_and_all_categories() -> None:
    plan = calculate_qsa_workspace(_inputs())

    assert set(plan.inventory) == {"score", "top_k", "expand_gather", "attention", "state"}
    assert plan.request.page_count == 1
    assert plan.request.score_columns == 2
    assert plan.inventory["score"].components["q_index"] == 2 * 2 * 8 * 2
    assert plan.inventory["score"].components["logits"] == 2 * 2 * 4
    assert plan.inventory["score"].components["block_table"] == 2 * 1 * 4
    assert plan.inventory["score"].shapes["block_table"] == (2, 1)
    assert plan.inventory["top_k"].components["blocks"] == 2 * (4 // 4) * 4
    assert plan.inventory["expand_gather"].components["indices"] == 2 * (4 + 4 - 1) * 4
    assert plan.request.attention_splits == 1
    assert plan.inventory["attention"].components["partial_output"] == 0
    assert plan.inventory["attention"].components["partial_lse"] == 0
    assert plan.inventory["state"].components["compressed_slab"] == (2 * (8 // 4 + 3) * 8 * 2)
    assert plan.inventory["state"].components["index_rope"] == 16 * 8 * 4
    assert plan.inventory["state"].components["cmp_rows"] == 2 * 4
    assert plan.inventory["state"].components["ring_rows"] == 2 * 4
    assert plan.inventory["state"].shapes["cmp_rows"] == (2,)
    assert plan.inventory["state"].shapes["ring_rows"] == (2,)
    assert plan.persistent_bytes == (
        plan.inventory["state"].components["compressed_slab"]
        + plan.inventory["state"].components["pending_ring"]
        + plan.inventory["state"].components["index_rope"]
    )
    assert (
        plan.required_bytes
        == plan.persistent_bytes + plan.capture_resident_bytes + plan.eager_transient_peak_bytes
    )


def test_incomplete_group_keeps_complete_context_blocks_and_tail_shape() -> None:
    plan = calculate_qsa_workspace(_inputs(context_tokens=7, page_table_width=2))

    assert plan.request.context_tokens == 7
    assert plan.request.page_table_width == 2
    assert plan.request.page_count == 1
    assert plan.request.score_columns == 2
    assert plan.inventory["top_k"].components["blocks"] == 8
    assert plan.inventory["expand_gather"].components["indices"] == 2 * 7 * 4


def test_zero_and_negative_dimensions_fail_closed() -> None:
    for field in ("context_tokens", "token_rows", "page_table_width", "index_heads", "dtype_bytes"):
        with pytest.raises(QSAWorkspaceInputError, match=field):
            QSAWorkspaceInputs(**{**_inputs().__dict__, field: 0})
        with pytest.raises(QSAWorkspaceInputError, match=field):
            QSAWorkspaceInputs(**{**_inputs().__dict__, field: -1})

    with pytest.raises(QSAWorkspaceInputError, match="divisible"):
        _inputs(top_k=5)
    with pytest.raises(QSAWorkspaceInputError, match="divisible"):
        _inputs(query_heads=3)

    with pytest.raises(QSAWorkspaceInputError, match="capture_max_batch_size"):
        _inputs(batch_size=2, capture_max_batch_size=1)
    with pytest.raises(QSAWorkspaceInputError, match="batch_size"):
        _inputs(batch_size=5, num_req_slots=4)

    for field, value in (("phase", "unknown"), ("topk_backend", "other"), ("dtype_bytes", 4)):
        with pytest.raises(QSAWorkspaceInputError):
            _inputs(**{field: value})
    with pytest.raises(QSAWorkspaceInputError, match=r"capture.*Triton|Triton.*capture"):
        _inputs(phase="capture", topk_backend="torch")


def test_missing_or_unknown_inventory_categories_fail_closed() -> None:
    complete = {name: 1 for name in ("score", "top_k", "expand_gather", "attention", "state")}
    with pytest.raises(QSAWorkspaceInputError, match="missing"):
        QSAWorkspaceInventory({key: value for key, value in complete.items() if key != "state"})
    with pytest.raises(QSAWorkspaceInputError, match="unknown"):
        QSAWorkspaceInventory({**complete, "other": 1})
    valid = QSAWorkspaceCategory("state", 1, {"one": 1}, {"one": (1,)})
    with pytest.raises(QSAWorkspaceInputError, match="malformed"):
        QSAWorkspaceInventory({**{key: valid for key in complete}, "state": valid})


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
    assert payload["categories"]["score"] == plan.inventory["score"].bytes
    assert payload["shapes"]["score"]["logits"] == (2, 2)
    assert plan.as_dict()["request"]["context_tokens"] == 8


def test_ragged_metadata_uses_batch_size_not_request_slot_capacity() -> None:
    plan = calculate_qsa_workspace(
        _inputs(
            batch_size=2,
            num_req_slots=32,
            token_rows=7,
            page_table_width=17,
            page_size=8,
            context_tokens=16,
        )
    )

    assert plan.request.page_count == 3
    assert plan.request.score_columns == 6
    assert plan.inventory["score"].shapes["token_to_req"] == (7,)
    assert plan.inventory["score"].shapes["cu_seqlens"] == (3,)
    assert plan.inventory["score"].shapes["block_table"] == (2, 3)
    metadata = plan.inventory["score"].components
    assert metadata["token_to_req"] == 7 * 4
    assert metadata["block_table"] == 2 * 3 * 4
    assert metadata["ring_slots"] == 2 * 4


def test_score_chunk_matches_runtime_128_mib_budget() -> None:
    columns = (QSA_LOGITS_WORKSPACE_BYTES // 4) + 1
    plan = calculate_qsa_workspace(
        _inputs(
            token_rows=9,
            context_tokens=columns * 4,
            page_table_width=columns * 4,
            page_size=4,
            compression_ratio=4,
        )
    )
    assert plan.request.chunk_rows == 1
    assert plan.inventory["score"].shapes["logits"] == (1, columns)


def test_eager_peak_retains_indices_and_overlaps_current_score_chunk() -> None:
    plan = calculate_qsa_workspace(
        _inputs(
            batch_size=2,
            token_rows=17,
            page_table_width=128,
            page_size=8,
            context_tokens=128,
            top_k=4,
        )
    )
    score = plan.inventory["score"].components
    topk = plan.inventory["top_k"].components
    expand = plan.inventory["expand_gather"].components
    metadata = sum(
        score[name]
        for name in (
            "last_indices",
            "token_to_req",
            "cu_seqlens",
            "seq_lens",
            "ring_slots",
            "block_table",
        )
    )
    expected_select = metadata + score["q_index"] + expand["indices"]
    expected_select += (
        score["logits"] + score["visible"] + topk["blocks"] + topk["candidate_scratch"]
    )
    assert plan.eager_transient_peak_bytes >= expected_select
    assert plan.required_bytes == plan.persistent_bytes + plan.eager_transient_peak_bytes


def test_scatter_rows_are_retained_through_eager_selection_and_attention() -> None:
    plan = calculate_qsa_workspace(
        _inputs(
            token_rows=17,
            batch_size=2,
            page_table_width=128,
            page_size=8,
            context_tokens=128,
            top_k=4,
        )
    )
    state = plan.inventory["state"]
    scatter = state.components["cmp_rows"] + state.components["ring_rows"]
    assert scatter == 17 * 4 * 2
    # The fixture has enough selection/attention work that the per-forward scatter plan is
    # part of both retained-phase lower bounds, rather than merely an index-update allocation.
    score = plan.inventory["score"].components
    topk = plan.inventory["top_k"].components
    metadata = sum(
        score[name]
        for name in (
            "last_indices",
            "token_to_req",
            "cu_seqlens",
            "seq_lens",
            "ring_slots",
            "block_table",
        )
    )
    select_floor = (
        metadata
        + scatter
        + score["q_index"]
        + plan.inventory["expand_gather"].components["indices"]
        + score["logits"]
        + score["visible"]
        + topk["blocks"]
        + topk["candidate_scratch"]
    )
    attention_floor = (
        metadata
        + scatter
        + plan.inventory["expand_gather"].components["indices"]
        + plan.inventory["attention"].bytes
    )
    assert plan.eager_transient_peak_bytes >= max(select_floor, attention_floor)


def test_scatter_rows_are_accounted_in_capture_residency() -> None:
    plan = calculate_qsa_workspace(
        _inputs(
            phase="capture",
            batch_size=2,
            capture_max_batch_size=8,
            num_req_slots=8,
            token_rows=2,
            page_table_width=128,
            page_size=8,
            context_tokens=128,
        )
    )
    scatter = (
        plan.inventory["state"].components["cmp_rows"]
        + plan.inventory["state"].components["ring_rows"]
    )
    assert scatter == 2 * 4 * 2
    assert plan.capture_resident_bytes >= scatter


def test_capture_resident_accounts_all_graph_buffers_and_active_attention() -> None:
    plan = calculate_qsa_workspace(
        _inputs(
            phase="capture",
            batch_size=2,
            capture_max_batch_size=8,
            num_req_slots=8,
            token_rows=2,
            page_table_width=128,
            page_size=8,
            context_tokens=128,
        )
    )
    request = plan.request
    bs = request.capture_max_batch_size
    columns = request.score_columns
    chunk = request.capture_chunk_rows
    block_top_k = request.top_k // request.compression_ratio
    graph = (
        bs * request.page_count * 4
        + bs * 4
        + bs * 4
        + bs * 4
        + (bs + 1) * 4
        + chunk * columns * 4
        + bs * 4
        + bs * block_top_k * 4
        + bs * request.selection_width * 4
        + bs * request.index_head_dim * 2
        + bs * 4
        + bs * request.index_heads * request.index_head_dim * 2
        + chunk * qsa_topk_scratch_width(columns, block_top_k, request.topk_backend)
    )
    # The fixture uses only two compressed score columns, so Triton's one-program top-k path
    # has no candidate scratch.  Capture also retains the active output and partial buffers.
    assert plan.capture_resident_bytes >= graph
    assert plan.required_bytes == plan.persistent_bytes + plan.capture_resident_bytes
    assert plan.capture_resident_bytes > plan.eager_transient_peak_bytes


def test_torch_topk_accounts_python_visible_fallback_temporaries() -> None:
    plan = calculate_qsa_workspace(
        _inputs(
            topk_backend="torch",
            page_table_width=20_000,
            context_tokens=8,
            token_rows=3,
        )
    )
    request = plan.request
    topk = plan.inventory["top_k"]
    columns = request.score_columns
    chunk = request.chunk_rows
    width = min(request.top_k // request.compression_ratio, columns)
    assert columns > 4096
    assert topk.components["torch_columns"] == columns * 4
    assert topk.components["torch_visibility_mask"] == chunk * columns
    assert topk.components["torch_values"] == chunk * width * 4
    assert topk.components["torch_chosen"] == chunk * width * 8
    assert topk.components["torch_valid"] == chunk * width
    assert topk.components["torch_chosen_i32"] == chunk * width * 4
    assert topk.components["torch_where"] == chunk * width * 4
    assert topk.shapes["torch_columns"] == (columns,)
    assert topk.shapes["torch_values"] == (chunk, width)
    assert topk.shapes["torch_chosen"] == (chunk, width)
    assert topk.shapes["torch_where"] == (chunk, width)
    assert plan.as_dict()["categories"]["top_k"] == topk.bytes
