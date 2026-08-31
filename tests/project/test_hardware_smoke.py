from __future__ import annotations

import os

import pytest


def _assert_expected_p4(properties: object) -> None:
    assert properties.name == "Tesla P4"
    assert (properties.major, properties.minor) == (6, 1)
    # nvidia-smi reports the nominal 7,680 MiB. Torch exposes about 7,599 MiB
    # while ECC is enabled because the reservation is not allocatable.
    usable_mib = properties.total_memory // (1024 * 1024)
    assert 7580 <= usable_mib <= 8192


@pytest.mark.sm61
def test_pascal_smoke_exposes_exactly_one_device() -> None:
    """A one-allocation identity check, never a sustained-load qualification."""
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "inventory-verified runner has no CUDA runtime"
    assert torch.cuda.device_count() == 1
    properties = torch.cuda.get_device_properties(0)
    _assert_expected_p4(properties)

    with torch.cuda.device(0):
        result = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32) + 4
        assert result.cpu().tolist() == [5, 6, 7]


@pytest.mark.dual_p4
def test_dual_pascal_smoke_keeps_device_allocations_isolated() -> None:
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "inventory-verified runner has no CUDA runtime"
    assert torch.cuda.device_count() == 2
    for index in range(2):
        properties = torch.cuda.get_device_properties(index)
        _assert_expected_p4(properties)
        with torch.cuda.device(index):
            result = torch.tensor([index], device=f"cuda:{index}") + 1
            assert result.cpu().tolist() == [index + 1]


def test_smoke_contract_has_at_most_one_visible_device() -> None:
    """The single-card gate accepts either physical P4 while rejecting broad masks."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible is None or (visible.isdigit() and "," not in visible)
