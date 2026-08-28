from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    sm61_verified = os.environ.get("FREETOKEN_SM61_RUNNER_VERIFIED") == "1"
    dual_verified = os.environ.get("FREETOKEN_DUAL_P4_RUNNER_VERIFIED") == "1"
    model_available = bool(os.environ.get("FREETOKEN_PASCAL_MODEL_PATH"))
    for item in items:
        if item.get_closest_marker("sm61") and not sm61_verified:
            item.add_marker(
                pytest.mark.skip(reason="H2 deferred: no inventory-verified sm_61 runner")
            )
        if item.get_closest_marker("dual_p4") and not dual_verified:
            item.add_marker(
                pytest.mark.skip(reason="H3 deferred: no inventory-verified dual-P4 runner")
            )
        if item.get_closest_marker("large_model") and not model_available:
            item.add_marker(
                pytest.mark.skip(reason="H4 deferred: FREETOKEN_PASCAL_MODEL_PATH is unset")
            )
