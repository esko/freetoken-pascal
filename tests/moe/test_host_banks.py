from __future__ import annotations

import pytest
import torch


def _specs() -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {
        "gate_up": ((4, 8), torch.uint8),
        "down": ((4, 8), torch.uint8),
    }


def _swap_reader(*, vm_swap: str = "0 kB", total: str = "0 kB", free: str = "0 kB"):
    files = {
        "/proc/self/status": f"Name:\tworker\nVmSwap: {vm_swap}\n",
        "/proc/meminfo": f"SwapTotal: {total}\nSwapFree: {free}\n",
        "/proc/swaps": "Filename Type Size Used Priority\n",
    }
    return files.__getitem__


def test_default_host_bank_policy_is_pageable_and_reports_per_layer_accounting():
    from freetoken.moe.host_banks import (
        HostBankPolicy,
        HostBankStrategy,
        alloc_layer_banks,
    )

    policy = HostBankPolicy()
    plan = policy.prepare(_specs(), num_layers=3)

    assert policy.strategy is HostBankStrategy.PAGEABLE
    assert plan.layer_residency == ("pageable", "pageable", "pageable")
    assert plan.pinned_bytes == 0
    assert plan.staging_bytes == 0
    assert plan.source_bytes == 3 * 2 * 4096

    banks = alloc_layer_banks(_specs(), 3, policy=policy)
    assert all(
        bank.residency.value == "pageable"
        for banks_by_layer in banks.values()
        for bank in banks_by_layer
    )
    report = policy.accounting.as_dict()
    assert report["strategy"] == "pageable"
    assert report["source_bytes"] == 3 * 2 * 4096
    assert report["pinned_bytes"] == 0
    assert report["layers"] == ["pageable", "pageable", "pageable"]


def test_no_swap_preflight_reports_probe_and_allows_clear_system():
    from freetoken.moe.host_banks import HostBankPolicy

    policy = HostBankPolicy(
        strategy="pageable",
        require_no_swap=True,
        swap_probe_reader=_swap_reader(),
    )
    policy.prepare(_specs(), num_layers=1)

    report = policy.accounting.as_dict()
    assert report["require_no_swap"] is True
    assert report["swap_status"] == "clear"
    assert report["no_swap_observed"] is True
    assert report["swap_probe_source"] == "procfs"
    assert report["vm_swap_bytes"] == 0


def test_no_swap_preflight_fails_before_host_bank_allocation(monkeypatch):
    from freetoken.moe.host_banks import HostBank, HostBankPolicy, alloc_layer_banks

    policy = HostBankPolicy(
        strategy="pageable",
        require_no_swap=True,
        swap_probe_reader=_swap_reader(total="1 MiB", free="1 MiB"),
    )
    called = False

    def unexpected_init(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("swap failure must happen before HostBank allocation")

    monkeypatch.setattr(HostBank, "__init__", unexpected_init)
    with pytest.raises(ValueError, match=r"require_no_swap.*swap-active"):
        alloc_layer_banks(_specs(), 1, policy=policy)
    assert not called


def test_default_swap_policy_reports_pressure_without_rejecting():
    from freetoken.moe.host_banks import HostBankPolicy

    policy = HostBankPolicy(
        strategy="pageable",
        swap_probe_reader=_swap_reader(vm_swap="2 kB", total="1 MiB", free="1 MiB"),
    )
    policy.prepare_layer_bytes([4096])

    assert policy.accounting.swap_status == "swap-active"
    assert policy.accounting.as_dict()["no_swap_observed"] is False


def test_default_policy_does_not_probe_swap():
    reads = []

    def unexpected_read(path):
        reads.append(path)
        raise AssertionError("swap probing is opt-in")

    from freetoken.moe.host_banks import HostBankPolicy

    policy = HostBankPolicy(strategy="pageable", swap_probe_reader=unexpected_read)
    policy.prepare_layer_bytes([4096])

    assert reads == []
    report = policy.accounting.as_dict()
    assert report["swap_status"] == "not-requested"
    assert report["no_swap_observed"] is None


def test_no_swap_failure_invalidates_prior_plan_and_accounting():
    from freetoken.moe.host_banks import HostBankPolicy

    state = {"active": False}

    def reader(path):
        if path == "/proc/self/status":
            return "VmSwap: 0 kB\n"
        if path == "/proc/meminfo":
            return f"SwapTotal: {'1' if state['active'] else '0'} kB\nSwapFree: 0 kB\n"
        return "Filename Type Size Used Priority\n"

    policy = HostBankPolicy(
        strategy="pageable", require_no_swap=True, swap_probe_reader=reader
    )
    policy.prepare_layer_bytes([4096])
    assert policy.plan.layer_bytes == (4096,)

    state["active"] = True
    with pytest.raises(ValueError, match=r"require_no_swap.*swap-active"):
        policy.prepare_layer_bytes([4096])
    with pytest.raises(RuntimeError, match="must be prepared"):
        _ = policy.plan
    assert policy.accounting.source_bytes == 0
    assert policy.accounting.swap_status == "swap-active"


def test_pinned_policy_rejects_budget_before_allocating_any_bank(monkeypatch):
    from freetoken.moe.host_banks import HostBank, HostBankPolicy

    policy = HostBankPolicy(strategy="pinned", max_pinned_bytes=4096)
    with pytest.raises(ValueError, match=r"pinned.*4096"):
        policy.prepare(_specs(), num_layers=2)

    called = False

    def unexpected_init(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("budget failure must happen before HostBank allocation")

    monkeypatch.setattr(HostBank, "__init__", unexpected_init)
    with pytest.raises(ValueError, match="pinned"):
        from freetoken.moe.host_banks import alloc_layer_banks

        alloc_layer_banks(_specs(), 2, policy=policy)
    assert not called


def test_selective_pinned_policy_only_registers_selected_layers():
    from freetoken.moe.host_banks import HostBankPolicy, HostResidency

    policy = HostBankPolicy(
        strategy="pinned",
        selected_layers=(1,),
        max_pinned_bytes=8192,
    )
    plan = policy.prepare(_specs(), num_layers=3)
    assert plan.layer_residency == (
        HostResidency.PAGEABLE.value,
        HostResidency.PINNED.value,
        HostResidency.PAGEABLE.value,
    )
    assert plan.pinned_bytes == 8192

    class FakeBank:
        def __init__(self):
            self.residency = HostResidency.PAGEABLE
            self.allocated_nbytes = 4096

        def pin(self):
            self.residency = HostResidency.PINNED

    fake_banks = {name: [FakeBank() for _ in range(3)] for name in _specs()}
    policy.settle(fake_banks)
    policy.settle(fake_banks)
    report = policy.accounting.as_dict()
    assert report["applied_pinned_bytes"] == 8192
    assert report["applied_layers"] == ["pageable", "pinned", "pageable"]


def test_legacy_scalar_pin_banks_stays_pinned_under_pageable_layer_plan():
    from freetoken.moe.host_banks import HostResidency, pin_banks, requested_residency

    class FakeBank:
        allocated_nbytes = 4096
        residency = HostResidency.PAGEABLE

        def __init__(self):
            self.pin_calls = 0

        def pin(self):
            self.pin_calls += 1
            self.residency = HostResidency.PINNED

    bank = FakeBank()
    with requested_residency(["pageable"]):
        pin_banks({"scalar": bank})

    assert bank.pin_calls == 1
    assert bank.residency is HostResidency.PINNED


def test_registered_bank_cleanup_unregisters_exactly_once(monkeypatch):
    from freetoken.kernel import pinned
    from freetoken.moe.host_banks import _LIVE_BUFFERS, HostBank

    calls = []
    monkeypatch.setattr(
        pinned, "host_register", lambda addr, size: calls.append(("pin", addr, size))
    )
    monkeypatch.setattr(
        pinned, "host_unregister", lambda addr, size: calls.append(("unpin", addr, size))
    )
    bank = HostBank((8,), torch.uint8, backing="mmap")
    backing = bank._buf
    bank.pin()

    bank.close()
    bank.close()

    assert [call[0] for call in calls] == ["pin", "unpin"]
    assert backing not in _LIVE_BUFFERS


def test_bank_cleanup_with_exported_view_is_retryable():
    from freetoken.moe.host_banks import HostBank

    bank = HostBank((8,), torch.uint8, backing="mmap")
    exported = bank.memoryview()
    with pytest.raises(BufferError):
        bank.close()
    assert bank.tensor is not None

    exported.release()
    bank.close()


def test_staging_cleanup_closes_every_slot_after_failure():
    from freetoken.moe.host_banks import HostStagingRing

    closed = []

    class Slot(bytearray):
        def __init__(self, size, number):
            super().__init__(size)
            self.number = number

        def close(self):
            closed.append(self.number)
            if self.number == 0:
                raise RuntimeError("close failed")

    next_slot = iter(range(2))
    ring = HostStagingRing(8192, slots=2, allocator=lambda size: Slot(size, next(next_slot)))
    with pytest.raises(RuntimeError, match="close failed"):
        ring.close()
    assert closed == [0, 1]


def test_staging_constructor_rollback_closes_all_allocated_slots():
    from freetoken.moe.host_banks import HostStagingRing

    closed = []

    class Slot(bytearray):
        def __init__(self, size, number):
            super().__init__(size)
            self.number = number

        def close(self):
            closed.append(self.number)
            if self.number == 0:
                raise RuntimeError("cleanup failed")

    allocations = iter((Slot(4096, 0), Slot(4096, 1)))

    def allocate(_size):
        try:
            return next(allocations)
        except StopIteration as error:
            raise RuntimeError("allocation failed") from error

    with pytest.raises(RuntimeError, match="allocation failed"):
        HostStagingRing(12288, slots=3, allocator=allocate)
    assert closed == [0, 1]


def test_staging_constructor_closes_wrong_sized_current_slot():
    from freetoken.moe.host_banks import HostStagingRing

    closed = []

    class Slot(bytearray):
        def __init__(self, size, number):
            super().__init__(size)
            self.number = number

        def close(self):
            closed.append(self.number)

    allocations = iter((Slot(4096, 0), Slot(2048, 1)))

    with pytest.raises(ValueError, match="require exactly"):
        HostStagingRing(8192, slots=2, allocator=lambda _size: next(allocations))

    assert sorted(closed) == [0, 1]


def test_policy_rejects_empty_specs_before_allocation():
    from freetoken.moe.host_banks import HostBankPolicy

    with pytest.raises(ValueError, match="at least one bank spec"):
        HostBankPolicy().prepare({}, num_layers=1)


def test_explicit_policy_keeps_source_mmap_even_when_born_pinned_is_requested(monkeypatch):
    from freetoken.moe.host_banks import HostBankPolicy, alloc_layer_banks

    monkeypatch.setenv("FREETOKEN_BANK_CUDA_ALLOC", "1")
    policy = HostBankPolicy(strategy="pageable")
    banks = alloc_layer_banks(_specs(), 1, policy=policy)
    assert all(
        bank.residency.value == "pageable" for per_layer in banks.values() for bank in per_layer
    )


def test_bounded_staging_requires_explicit_capacity_and_has_fixed_ring_shape():
    from freetoken.moe.host_banks import HostBankPolicy, HostStagingRing

    with pytest.raises(ValueError, match="staging_bytes"):
        HostBankPolicy(strategy="bounded-staging").prepare(_specs(), num_layers=2)

    policy = HostBankPolicy(
        strategy="bounded-staging",
        staging_bytes=8192,
        max_staging_bytes=8192,
        staging_slots=2,
    )
    plan = policy.prepare(_specs(), num_layers=4)
    assert plan.layer_residency == ("pageable",) * 4
    assert plan.staging_bytes == 8192
    assert plan.source_bytes == 4 * 2 * 4096

    ring = HostStagingRing(8192, slots=2, allocator=lambda size: bytearray(size))
    assert ring.slot_bytes == 4096
    assert len(ring.slots) == 2
    ring.close()
    ring.close()
    with pytest.raises(ValueError, match="require exactly"):
        HostStagingRing(8192, slots=2, allocator=lambda size: bytearray(size + 1))


def test_policy_rejects_invalid_layer_selection_and_limits():
    from freetoken.moe.host_banks import HostBankPolicy

    with pytest.raises(ValueError, match="selected_layers"):
        HostBankPolicy(strategy="pinned", selected_layers=(3,)).prepare(_specs(), num_layers=2)
    with pytest.raises(ValueError, match="non-negative"):
        HostBankPolicy(max_pinned_bytes=-1)
    for invalid_limit in (float("inf"), float("nan"), True, 1.5):
        with pytest.raises(ValueError, match="non-negative integer"):
            HostBankPolicy(max_pinned_bytes=invalid_limit)
    with pytest.raises(ValueError, match="unknown host-bank strategy"):
        HostBankPolicy(strategy="surprise")
    with pytest.raises(ValueError, match="finite max_pinned_bytes"):
        HostBankPolicy(strategy="pinned", max_pinned_bytes=None).prepare(_specs(), 1)
    with pytest.raises(ValueError, match="finite max_staging_bytes"):
        HostBankPolicy(
            strategy="bounded-staging", staging_bytes=4096, max_staging_bytes=None
        ).prepare(_specs(), 1)


def test_failed_reprepare_invalidates_plan_and_partial_settlement_is_rejected():
    from freetoken.moe.host_banks import HostBankPolicy, HostResidency

    policy = HostBankPolicy(strategy="pinned", max_pinned_bytes=8192)
    policy.prepare(_specs(), num_layers=1)
    with pytest.raises(ValueError, match="outside"):
        policy.selected_layers = (2,)
        policy.prepare(_specs(), num_layers=1)
    with pytest.raises(RuntimeError, match="must be prepared"):
        _ = policy.plan

    policy.selected_layers = (0,)
    policy.prepare(_specs(), num_layers=1)

    class FakeBank:
        residency = HostResidency.PAGEABLE
        allocated_nbytes = 4096

        def pin(self):
            self.residency = HostResidency.PINNED

    with pytest.raises(ValueError, match="complete bank set"):
        policy.settle({"gate_up": [FakeBank()]})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("max_pinned_bytes", -1, "max_pinned_bytes"),
        ("staging_slots", 0, "staging_slots"),
        ("strategy", "surprise", "unknown host-bank strategy"),
    ),
)
def test_prepare_revalidates_mutated_policy_configuration(field, value, message):
    from freetoken.moe.host_banks import HostBankPolicy

    policy = HostBankPolicy(strategy="pinned", max_pinned_bytes=8192)
    policy.prepare(_specs(), num_layers=1)
    setattr(policy, field, value)

    with pytest.raises(ValueError, match=message):
        policy.prepare(_specs(), num_layers=1)
    with pytest.raises(RuntimeError, match="must be prepared"):
        _ = policy.plan
    assert policy.accounting.source_bytes == 0
    assert policy.accounting.pinned_bytes == 0
    assert policy.accounting.applied_pinned_bytes == 0
