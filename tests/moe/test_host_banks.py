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
        swap_probe_reader=_swap_reader(total="1024 kB", free="1024 kB"),
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


def test_default_swap_policy_does_not_probe_pressure():
    from freetoken.moe.host_banks import HostBankPolicy

    policy = HostBankPolicy(
        strategy="pageable",
        swap_probe_reader=lambda _path: (_ for _ in ()).throw(
            AssertionError("default policy must not probe procfs")
        ),
    )
    policy.prepare_layer_bytes([4096])

    assert policy.accounting.swap_status == "not-requested"
    assert policy.accounting.as_dict()["no_swap_observed"] is None


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

    policy = HostBankPolicy(strategy="pageable", require_no_swap=True, swap_probe_reader=reader)
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


def _numa_reader(path):
    return {
        "/proc/self/status": "Name:\tworker\nMems_allowed_list:\t0-1\n",
        "/sys/devices/system/node/online": "0-1\n",
    }[path]


def test_enforced_numa_policy_applies_to_policy_owned_mmap_banks():
    from freetoken.moe.host_banks import HostBankPolicy, alloc_layer_banks

    class Backend:
        def __init__(self):
            self.calls = []

        def mbind(self, addr, nbytes, mode, nodes):
            self.calls.append((addr, nbytes, mode, tuple(nodes)))

    backend = Backend()
    policy = HostBankPolicy(
        strategy="pageable",
        numa_policy="interleave",
        enforce_numa_placement=True,
        numa_backend=backend,
        numa_reader=_numa_reader,
    )
    banks = alloc_layer_banks({"x": ((1,), torch.uint8)}, 1, policy=policy)
    assert len(backend.calls) == 1
    assert backend.calls[0][2:] == (3, (0, 1))
    assert policy.accounting.as_dict()["numa_status"] == "applied"
    banks["x"][0].close()


def test_host_bank_accounting_exposes_cross_node_residency():
    from freetoken.moe.host_banks import HostBankPolicy, alloc_layer_banks

    class Backend:
        def mbind(self, *args):
            return None

        def move_pages(self, addresses):
            return tuple(0 if index == 0 else 1 for index, _ in enumerate(addresses))

    policy = HostBankPolicy(
        strategy="pageable",
        numa_policy="preferred",
        numa_node=0,
        enforce_numa_placement=True,
        sample_numa_residency=True,
        numa_sample_max_pages=2,
        numa_backend=Backend(),
        numa_reader=_numa_reader,
    )
    banks = alloc_layer_banks({"x": ((8192,), torch.uint8)}, 1, policy=policy)
    policy.sample_numa_bank(banks["x"][0])
    report = policy.accounting.as_dict()
    assert report["numa_sample_target_pages"] == 1
    assert report["numa_sample_off_target_pages"] == 1
    assert report["numa_sample_off_target_counts"] == {"1": 1}
    assert report["numa_sample_target_fraction"] == 0.5
    assert report["numa_sample_placement_match"] is False
    banks["x"][0].close()


def test_ftw_numa_samples_retain_global_and_per_layer_bank_identity(tmp_path):
    import json

    from freetoken.checkpoint.ftw import FTWWriter, layer_bank_entry_name, load_ftw_banks
    from freetoken.moe.host_banks import HostBankPolicy

    checkpoint = tmp_path / "ftw"
    writer = FTWWriter(str(checkpoint))
    writer.add_tensor("gate_up_alpha", torch.ones(4), kind="experts_bank")
    writer.add_tensor("gate_up", torch.ones((4, 8), dtype=torch.uint8), kind="experts_bank")
    for bank_name in ("down",):
        for layer_id in range(2):
            writer.add_tensor(
                layer_bank_entry_name(bank_name, layer_id),
                torch.full((2, 8), layer_id, dtype=torch.uint8),
                kind="experts_bank",
            )
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 2})

    class Backend:
        def mbind(self, *args):
            return None

        def move_pages(self, addresses):
            return (0,) * len(addresses)

    policy = HostBankPolicy(
        strategy="pageable",
        numa_policy="preferred",
        numa_node=0,
        enforce_numa_placement=True,
        sample_numa_residency=True,
        numa_sample_max_pages=1,
        numa_backend=Backend(),
        numa_reader=lambda path: {
            "/proc/self/status": "Mems_allowed_list:\t0\n",
            "/sys/devices/system/node/online": "0\n",
        }[path],
    )
    banks = load_ftw_banks(str(checkpoint), num_layers=2, workers=1, host_bank_policy=policy)
    assert banks is not None
    samples = banks.host_bank_accounting["numa_sample_banks"]
    identities = {
        (sample["identity"]["bank_name"], sample["identity"]["layer_id"]) for sample in samples
    }
    assert len(samples) == len(identities)
    assert identities == {
        ("gate_up_alpha", None),
        ("gate_up", 0),
        ("gate_up", 1),
        ("down", 0),
        ("down", 1),
    }
    banks.close()

    index_path = checkpoint / "freetoken_weight.json"
    index = json.loads(index_path.read_text())
    index["tensors"].reverse()
    index_path.write_text(json.dumps(index))
    reordered = load_ftw_banks(str(checkpoint), num_layers=2, workers=1, host_bank_policy=policy)
    assert reordered is not None
    reordered_identities = {
        (sample["identity"]["bank_name"], sample["identity"]["layer_id"])
        for sample in reordered.host_bank_accounting["numa_sample_banks"]
    }
    assert reordered_identities == identities
    reordered.close()


def test_ftw_invalid_sample_identity_rolls_back_prior_mappings(tmp_path):
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks
    from freetoken.moe.host_banks import _LIVE_BUFFERS, HostBankPolicy

    checkpoint = tmp_path / "ftw-invalid-name"
    writer = FTWWriter(str(checkpoint))
    writer.add_tensor("gate_up_alpha", torch.ones(2), kind="experts_bank")
    writer.add_tensor(" ", torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})
    before = tuple(_LIVE_BUFFERS)
    policy = HostBankPolicy(strategy="pageable")

    with pytest.raises(ValueError, match="bank_name"):
        load_ftw_banks(str(checkpoint), num_layers=1, workers=1, host_bank_policy=policy)

    assert tuple(_LIVE_BUFFERS) == before


@pytest.mark.parametrize("layout", ["duplicate-flat", "mixed"])
def test_ftw_ambiguous_sample_identities_fail_before_allocation(tmp_path, layout):
    from freetoken.checkpoint.ftw import FTWWriter, layer_bank_entry_name, load_ftw_banks
    from freetoken.moe.host_banks import _LIVE_BUFFERS, HostBankPolicy

    checkpoint = tmp_path / layout
    writer = FTWWriter(str(checkpoint))
    writer.add_tensor("gate_up", torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
    if layout == "duplicate-flat":
        writer.add_tensor("gate_up", torch.zeros((2, 8), dtype=torch.uint8), kind="experts_bank")
        message = "duplicate NUMA sample identity"
    else:
        writer.add_tensor(
            layer_bank_entry_name("gate_up", 0),
            torch.zeros((2, 8), dtype=torch.uint8),
            kind="experts_bank",
        )
        message = "mix flat and per-layer"
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})
    before = tuple(_LIVE_BUFFERS)

    with pytest.raises(ValueError, match=message):
        load_ftw_banks(
            str(checkpoint),
            num_layers=1,
            workers=1,
            host_bank_policy=HostBankPolicy(strategy="pageable"),
        )

    assert tuple(_LIVE_BUFFERS) == before


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        ("duplicate-alpha", "duplicate NUMA sample identity"),
        ("duplicate-flat", "duplicate NUMA sample identity"),
        ("duplicate-per-layer", "duplicate layer 0"),
        ("mixed", "mix flat and per-layer"),
        ("malformed-name", "bank_name"),
    ],
)
def test_ftw_default_path_rejects_ambiguous_identities_before_allocation(
    tmp_path, monkeypatch, layout, message
):
    import freetoken.moe.host_banks as host_banks
    from freetoken.checkpoint.ftw import FTWWriter, layer_bank_entry_name, load_ftw_banks

    checkpoint = tmp_path / f"default-{layout}"
    writer = FTWWriter(str(checkpoint))
    if layout == "duplicate-alpha":
        writer.add_tensor("gate_up_alpha", torch.ones(2), kind="experts_bank")
        writer.add_tensor("gate_up_alpha", torch.zeros(2), kind="experts_bank")
    elif layout == "malformed-name":
        writer.add_tensor(" ", torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
    else:
        writer.add_tensor("gate_up", torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
        if layout == "duplicate-flat":
            writer.add_tensor(
                "gate_up", torch.zeros((2, 8), dtype=torch.uint8), kind="experts_bank"
            )
        elif layout == "duplicate-per-layer":
            name = layer_bank_entry_name("down", 0)
            writer.add_tensor(name, torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
            writer.add_tensor(name, torch.zeros((2, 8), dtype=torch.uint8), kind="experts_bank")
        else:
            writer.add_tensor(
                layer_bank_entry_name("gate_up", 0),
                torch.zeros((2, 8), dtype=torch.uint8),
                kind="experts_bank",
            )
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})
    before = tuple(host_banks._LIVE_BUFFERS)

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("invalid FTW layout reached HostBank allocation")

    monkeypatch.setattr(host_banks, "HostBank", forbidden_allocation)
    with pytest.raises(ValueError, match=message):
        load_ftw_banks(str(checkpoint), num_layers=1, workers=1)

    assert tuple(host_banks._LIVE_BUFFERS) == before


def test_ftw_default_path_rolls_back_owned_prefix_on_early_allocation_failure(
    tmp_path, monkeypatch
):
    import freetoken.moe.host_banks as host_banks
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks

    checkpoint = tmp_path / "default-allocation-failure"
    writer = FTWWriter(str(checkpoint))
    writer.add_tensor("gate_up_alpha", torch.ones(2), kind="experts_bank")
    writer.add_tensor("gate_up", torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})

    original = host_banks.HostBank
    calls = 0

    def fail_second_allocation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError("injected HostBank allocation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(host_banks, "born_pinned_default", lambda: False)
    monkeypatch.setattr(host_banks, "HostBank", fail_second_allocation)
    before = tuple(host_banks._LIVE_BUFFERS)

    with pytest.raises(MemoryError, match="injected HostBank allocation failure"):
        load_ftw_banks(str(checkpoint), num_layers=1, workers=1)

    assert calls == 2
    assert tuple(host_banks._LIVE_BUFFERS) == before


def test_ftw_default_path_rolls_back_derived_views_when_bundle_construction_fails(
    tmp_path, monkeypatch
):
    import freetoken.moe.expert_banks as expert_banks
    import freetoken.moe.host_banks as host_banks
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks

    checkpoint = tmp_path / "default-bundle-failure"
    writer = FTWWriter(str(checkpoint))
    writer.add_tensor("gate_up_alpha", torch.ones(2), kind="experts_bank")
    writer.add_tensor("gate_up", torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})

    class FailingExpertBanks:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("injected ExpertBanks construction failure")

    monkeypatch.setattr(host_banks, "born_pinned_default", lambda: False)
    monkeypatch.setattr(expert_banks, "ExpertBanks", FailingExpertBanks)
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    before = tuple(host_banks._LIVE_BUFFERS)

    with pytest.raises(RuntimeError, match="injected ExpertBanks construction failure"):
        load_ftw_banks(str(checkpoint), num_layers=1, workers=1)

    assert tuple(host_banks._LIVE_BUFFERS) == before


def test_ftw_default_path_preserves_empty_expert_bundle_probe(tmp_path, monkeypatch):
    import freetoken.moe.host_banks as host_banks
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks

    checkpoint = tmp_path / "no-expert-banks"
    writer = FTWWriter(str(checkpoint))
    writer.add_tensor("model.embed_tokens.weight", torch.ones(2), kind="weight")
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("empty expert-bank probe reached HostBank allocation")

    monkeypatch.setattr(host_banks, "HostBank", forbidden_allocation)
    before = tuple(host_banks._LIVE_BUFFERS)
    assert load_ftw_banks(str(checkpoint), num_layers=1, workers=1) is None
    assert tuple(host_banks._LIVE_BUFFERS) == before


def test_ftw_default_path_preserves_fully_empty_checkpoint_probe(tmp_path, monkeypatch):
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks
    from freetoken.moe import host_banks

    checkpoint = tmp_path / "fully-empty"
    FTWWriter(str(checkpoint)).finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("fully empty checkpoint reached HostBank allocation")

    monkeypatch.setattr(host_banks, "HostBank", forbidden_allocation)
    before = tuple(host_banks._LIVE_BUFFERS)
    assert load_ftw_banks(str(checkpoint), num_layers=1, workers=1) is None
    assert tuple(host_banks._LIVE_BUFFERS) == before


def test_ftw_policy_rejects_fully_empty_checkpoint(tmp_path):
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks
    from freetoken.moe.host_banks import HostBankPolicy

    checkpoint = tmp_path / "fully-empty-policy"
    FTWWriter(str(checkpoint)).finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})

    with pytest.raises(ValueError, match="no experts_bank entries"):
        load_ftw_banks(
            str(checkpoint),
            num_layers=1,
            workers=1,
            host_bank_policy=HostBankPolicy(strategy="pageable"),
        )


def test_ftw_default_path_keeps_successful_allocations_live(tmp_path, monkeypatch):
    import freetoken.moe.host_banks as host_banks
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks

    checkpoint = tmp_path / "valid-default"
    writer = FTWWriter(str(checkpoint))
    writer.add_tensor("gate_up_alpha", torch.ones(2), kind="experts_bank")
    writer.add_tensor(
        "gate_up", torch.arange(16, dtype=torch.uint8).view(2, 8), kind="experts_bank"
    )
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})

    original = host_banks.HostBank
    allocated = []

    def track_allocation(*args, **kwargs):
        bank = original(*args, **kwargs)
        allocated.append(bank)
        return bank

    monkeypatch.setattr(host_banks, "born_pinned_default", lambda: False)
    monkeypatch.setattr(host_banks, "HostBank", track_allocation)
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    before = tuple(host_banks._LIVE_BUFFERS)
    result = load_ftw_banks(str(checkpoint), num_layers=1, workers=1)
    assert result is not None
    assert result.sources["gate_up"][0].tolist() == [list(range(8)), list(range(8, 16))]
    assert all(bank.tensor is not None for bank in allocated)

    # The returned bundle owns the wrappers and keeps the aliases live until close.
    result.close()
    assert tuple(host_banks._LIVE_BUFFERS) == before
    result.close()
    assert tuple(host_banks._LIVE_BUFFERS) == before


@pytest.mark.parametrize("use_policy", [False, True])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("global_off", "0", "invalid global_off"),
        ("global_off", -4096, "invalid global_off"),
        ("nbytes", -1, "invalid nbytes"),
        ("shape", [0, 8], "invalid shape"),
        ("shape", [2.0, 8], "invalid shape"),
    ],
)
def test_ftw_malformed_bank_geometry_fails_before_allocation(
    tmp_path, monkeypatch, use_policy, field, value, message
):
    import json

    import freetoken.moe.host_banks as host_banks
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks

    checkpoint = tmp_path / f"bad-{field}-{value!s}-{use_policy}"
    writer = FTWWriter(str(checkpoint))
    writer.add_tensor("gate_up", torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})
    index_path = checkpoint / "freetoken_weight.json"
    index = json.loads(index_path.read_text())
    index["tensors"][0][field] = value
    index_path.write_text(json.dumps(index))

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("malformed FTW geometry reached HostBank allocation")

    monkeypatch.setattr(host_banks, "HostBank", forbidden_allocation)
    before = tuple(host_banks._LIVE_BUFFERS)
    policy = host_banks.HostBankPolicy(strategy="pageable") if use_policy else None
    with pytest.raises(ValueError, match=message):
        load_ftw_banks(str(checkpoint), num_layers=1, workers=1, host_bank_policy=policy)
    assert tuple(host_banks._LIVE_BUFFERS) == before


@pytest.mark.parametrize("corruption", ["shard-gap", "shard-size", "tensor-range"])
def test_ftw_corrupt_storage_geometry_fails_before_allocation(tmp_path, monkeypatch, corruption):
    import json

    import freetoken.moe.host_banks as host_banks
    from freetoken.checkpoint.ftw import ALIGN, FTWWriter, load_ftw_banks

    checkpoint = tmp_path / corruption
    writer = FTWWriter(str(checkpoint))
    writer.add_tensor("gate_up", torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})
    index_path = checkpoint / "freetoken_weight.json"
    index = json.loads(index_path.read_text())
    if corruption == "shard-gap":
        index["shards"][0]["global_off"] = ALIGN
        message = "contiguous logical range"
    elif corruption == "shard-size":
        index["shards"][0]["nbytes"] += ALIGN
        message = "size mismatch"
    else:
        index["tensors"][0]["global_off"] = index["shards"][0]["nbytes"]
        message = "exceeds shard ranges"
    index_path.write_text(json.dumps(index))

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("corrupt FTW geometry reached HostBank allocation")

    monkeypatch.setattr(host_banks, "HostBank", forbidden_allocation)
    before = tuple(host_banks._LIVE_BUFFERS)
    with pytest.raises(ValueError, match=message):
        load_ftw_banks(str(checkpoint), num_layers=1, workers=1)
    assert tuple(host_banks._LIVE_BUFFERS) == before


@pytest.mark.parametrize("use_policy", [False, True])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("align", 0),
        ("shard_limit", 1.5),
        ("total_bytes", False),
        ("expert_bank_num_layers", 1.0),
    ],
)
def test_ftw_exact_index_metadata_fails_closed_before_allocation(
    tmp_path, monkeypatch, use_policy, field, value
):
    import json

    import freetoken.moe.host_banks as host_banks
    from freetoken.checkpoint.ftw import FTWWriter, load_ftw_banks

    checkpoint = tmp_path / f"bad-index-{field}-{use_policy}"
    writer = FTWWriter(str(checkpoint))
    writer.add_tensor("gate_up", torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})
    index_path = checkpoint / "freetoken_weight.json"
    index = json.loads(index_path.read_text())
    index[field] = value
    index_path.write_text(json.dumps(index))

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("malformed FTW index reached HostBank allocation")

    monkeypatch.setattr(host_banks, "HostBank", forbidden_allocation)
    before = tuple(host_banks._LIVE_BUFFERS)
    policy = host_banks.HostBankPolicy(strategy="pageable") if use_policy else None
    with pytest.raises(ValueError):
        load_ftw_banks(str(checkpoint), num_layers=1, workers=1, host_bank_policy=policy)
    assert tuple(host_banks._LIVE_BUFFERS) == before


def test_ftw_reader_rejects_non_object_index_with_controlled_error(tmp_path):
    from freetoken.checkpoint.ftw import INDEX_NAME, FTWReader

    checkpoint = tmp_path / "non-object-index"
    checkpoint.mkdir()
    (checkpoint / INDEX_NAME).write_text("[]")
    with pytest.raises(ValueError, match="index must be an object"):
        FTWReader(str(checkpoint))


@pytest.mark.parametrize("corruption", ["shard-overlap", "shard-missing", "shard-truncated"])
def test_ftw_missing_overlap_and_truncated_shards_fail_before_allocation(
    tmp_path, monkeypatch, corruption
):
    import json

    import freetoken.moe.host_banks as host_banks
    from freetoken.checkpoint.ftw import ALIGN, FTWWriter, load_ftw_banks

    checkpoint = tmp_path / corruption
    writer = FTWWriter(str(checkpoint), shard_limit=ALIGN)
    writer.add_tensor("gate_up", torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
    writer.add_tensor("down", torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})
    index_path = checkpoint / "freetoken_weight.json"
    index = json.loads(index_path.read_text())
    if corruption == "shard-overlap":
        index["shards"][1]["global_off"] = 0
        message = "contiguous logical range"
    elif corruption == "shard-missing":
        (checkpoint / index["shards"][1]["file"]).unlink()
        message = "unavailable"
    else:
        shard = checkpoint / index["shards"][1]["file"]
        shard.write_bytes(shard.read_bytes()[:-1])
        message = "size mismatch"
    index_path.write_text(json.dumps(index))

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("corrupt FTW storage reached HostBank allocation")

    monkeypatch.setattr(host_banks, "HostBank", forbidden_allocation)
    before = tuple(host_banks._LIVE_BUFFERS)
    with pytest.raises(ValueError, match=message):
        load_ftw_banks(str(checkpoint), num_layers=1, workers=1)
    assert tuple(host_banks._LIVE_BUFFERS) == before


def test_ftw_read_failure_rolls_back_and_closes_reader(tmp_path, monkeypatch):
    import freetoken.checkpoint.ftw as ftw
    import freetoken.moe.host_banks as host_banks

    checkpoint = tmp_path / "read-failure"
    writer = ftw.FTWWriter(str(checkpoint))
    writer.add_tensor("gate_up", torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    monkeypatch.setattr(host_banks, "born_pinned_default", lambda: False)

    def fail_read(*args, **kwargs):
        raise OSError("injected FTW read failure")

    monkeypatch.setattr(ftw.FTWReader, "read_into", fail_read)
    before = tuple(host_banks._LIVE_BUFFERS)
    with pytest.raises(OSError, match="injected FTW read failure"):
        ftw.load_ftw_banks(str(checkpoint), num_layers=1, workers=1)
    assert tuple(host_banks._LIVE_BUFFERS) == before


def test_ftw_setup_failure_rolls_back_allocated_banks(tmp_path, monkeypatch):
    import freetoken.checkpoint.ftw as ftw
    import freetoken.moe.host_banks as host_banks

    checkpoint = tmp_path / "setup-failure"
    writer = ftw.FTWWriter(str(checkpoint))
    writer.add_tensor("gate_up", torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    monkeypatch.setattr(host_banks, "born_pinned_default", lambda: False)

    class FailingPipeline:
        def __init__(self):
            raise RuntimeError("injected FTW setup failure")

    monkeypatch.setattr(host_banks, "PinPipeline", FailingPipeline)
    before = tuple(host_banks._LIVE_BUFFERS)
    with pytest.raises(RuntimeError, match="injected FTW setup failure"):
        ftw.load_ftw_banks(str(checkpoint), num_layers=1, workers=1)
    assert tuple(host_banks._LIVE_BUFFERS) == before


def test_ftw_reader_close_failure_still_rolls_back_before_return(tmp_path, monkeypatch):
    import freetoken.checkpoint.ftw as ftw
    import freetoken.moe.host_banks as host_banks

    checkpoint = tmp_path / "close-failure"
    writer = ftw.FTWWriter(str(checkpoint))
    writer.add_tensor("gate_up", torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    monkeypatch.setattr(host_banks, "born_pinned_default", lambda: False)
    original_close = ftw.FTWReader.close

    def fail_close(self):
        original_close(self)
        raise RuntimeError("injected FTW reader close failure")

    monkeypatch.setattr(ftw.FTWReader, "close", fail_close)
    before = tuple(host_banks._LIVE_BUFFERS)
    with pytest.raises(RuntimeError, match="injected FTW reader close failure"):
        ftw.load_ftw_banks(str(checkpoint), num_layers=1, workers=1)
    assert tuple(host_banks._LIVE_BUFFERS) == before


def test_ftw_derived_view_failure_clears_aliases_before_rollback(tmp_path, monkeypatch):
    import freetoken.checkpoint.ftw as ftw
    import freetoken.moe.host_banks as host_banks

    checkpoint = tmp_path / "derived-view-failure"
    writer = ftw.FTWWriter(str(checkpoint))
    writer.add_tensor("gate_up", torch.ones((2, 8), dtype=torch.uint8), kind="experts_bank")
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 1})
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    monkeypatch.setattr(host_banks, "born_pinned_default", lambda: False)
    original_view = torch.Tensor.view
    calls = 0

    def fail_derived_view(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("injected derived-view failure")
        return original_view(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "view", fail_derived_view)
    before = tuple(host_banks._LIVE_BUFFERS)
    with pytest.raises(RuntimeError, match="injected derived-view failure"):
        ftw.load_ftw_banks(str(checkpoint), num_layers=1, workers=1)
    assert tuple(host_banks._LIVE_BUFFERS) == before


def test_default_policy_does_not_call_numa_backend():
    from freetoken.moe.host_banks import HostBankPolicy, alloc_layer_banks

    class Backend:
        def mbind(self, *args):
            raise AssertionError("default policy must not issue mbind")

    policy = HostBankPolicy(strategy="pageable", numa_backend=Backend())
    banks = alloc_layer_banks({"x": ((1,), torch.uint8)}, 1, policy=policy)
    assert policy.accounting.as_dict()["numa_status"] == "not-requested"
    banks["x"][0].close()


def test_enforced_numa_policy_applies_to_scalar_policy_bank():
    from freetoken.moe.host_banks import HostBankPolicy, alloc_banks

    class Backend:
        def __init__(self):
            self.calls = 0

        def mbind(self, *args):
            self.calls += 1

    backend = Backend()
    policy = HostBankPolicy(
        strategy="pageable",
        numa_policy="interleave",
        enforce_numa_placement=True,
        numa_backend=backend,
        numa_reader=_numa_reader,
    )
    banks = alloc_banks({"x": ((1,), torch.uint8)}, policy=policy)
    assert backend.calls == 1
    banks["x"].close()


def test_bind_numa_failure_rolls_back_all_policy_mmaps():
    from freetoken.moe.host_banks import _LIVE_BUFFERS, HostBankPolicy, alloc_layer_banks

    class Backend:
        def __init__(self):
            self.calls = 0

        def mbind(self, *args):
            self.calls += 1
            if self.calls == 2:
                raise OSError("placement denied")

    backend = Backend()
    before = tuple(_LIVE_BUFFERS)
    policy = HostBankPolicy(
        strategy="pageable",
        numa_policy="bind",
        numa_node=0,
        enforce_numa_placement=True,
        numa_backend=backend,
        numa_reader=_numa_reader,
    )
    with pytest.raises(ValueError, match="mbind failed"):
        alloc_layer_banks({"x": ((1,), torch.uint8)}, 2, policy=policy)
    assert backend.calls == 2
    assert tuple(_LIVE_BUFFERS) == before


def test_policy_mmap_uses_private_anonymous_read_write_flags(monkeypatch):
    import freetoken.moe.host_banks as host_banks

    original = host_banks.mmap.mmap
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(host_banks.mmap, "mmap", capture)
    bank = host_banks.HostBank((1,), torch.uint8, backing="mmap")
    bank.close()
    assert calls
    _args, kwargs = calls[-1]
    assert kwargs["flags"] == host_banks.mmap.MAP_PRIVATE | host_banks.mmap.MAP_ANONYMOUS
    assert kwargs["prot"] == host_banks.mmap.PROT_READ | host_banks.mmap.PROT_WRITE
