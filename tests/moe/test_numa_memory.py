import ctypes
import errno
import mmap
import os

import pytest
from freetoken.moe.numa_memory import (
    AllowedNumaNodes,
    NumaPlacementController,
    NumaPlacementError,
    NumaSample,
    NumaSampleIdentity,
    NumaSyscallBackend,
    linux_syscall_numbers,
    resolve_allowed_numa_nodes,
    resolve_numa_placement,
)


def _procfs(files):
    def read(path):
        return files[path]

    return read


def test_allowed_nodes_intersect_online_and_process_mask():
    result = resolve_allowed_numa_nodes(
        read_text=_procfs(
            {
                "/proc/self/status": "Name:\ttest\nMems_allowed_list:\t0-2,4\n",
                "/sys/devices/system/node/online": "1-4\n",
            }
        )
    )
    assert result == AllowedNumaNodes(
        (1, 2, 4), "available", "/proc/self/status,/sys/devices/system/node/online"
    )


def test_missing_or_malformed_topology_is_unavailable():
    result = resolve_allowed_numa_nodes(
        read_text=_procfs(
            {
                "/proc/self/status": "Mems_allowed_list: 0-a\n",
                "/sys/devices/system/node/online": "0\n",
            }
        )
    )
    assert result.status == "unavailable"
    assert result.nodes == ()
    assert result.errors


def test_disabled_resolution_does_not_read_procfs():
    calls = []

    def read(path):
        calls.append(path)
        raise AssertionError("disabled NUMA placement must not probe topology")

    plan = resolve_numa_placement("interleave", None, read_text=read)
    assert plan.status == "not-requested"
    assert calls == []


def test_policy_node_masks_are_explicit():
    allowed = AllowedNumaNodes((0, 2, 4), "available", "fixture")
    assert resolve_numa_placement("preferred", 2, enforce=True, allowed=allowed).target_nodes == (
        2,
    )
    assert resolve_numa_placement(
        "interleave", None, enforce=True, allowed=allowed
    ).target_nodes == (
        0,
        2,
        4,
    )
    assert resolve_numa_placement("interleave", 4, enforce=True, allowed=allowed).target_nodes == (
        4,
    )
    assert resolve_numa_placement("preferred", None, enforce=True, allowed=allowed).status == (
        "fallback"
    )
    with pytest.raises(NumaPlacementError, match="requires numa_node"):
        resolve_numa_placement("bind", None, enforce=True, allowed=allowed)
    with pytest.raises(NumaPlacementError, match="not online"):
        resolve_numa_placement("bind", 3, enforce=True, allowed=allowed)


class FakeBackend:
    def __init__(self, *, error=None, statuses=(0, 0)):
        self.calls = []
        self.error = error
        self.statuses = statuses

    def mbind(self, addr, nbytes, mode, nodes):
        self.calls.append((addr, nbytes, mode, tuple(nodes)))
        if self.error:
            raise self.error

    def move_pages(self, addresses):
        self.calls.append(("move_pages", tuple(addresses)))
        return self.statuses[: len(addresses)]


def test_mbind_happens_only_for_private_anonymous_mapping_before_touch():
    backend = FakeBackend()
    plan = resolve_numa_placement(
        "preferred", 2, enforce=True, allowed=AllowedNumaNodes((2,), "available", "fixture")
    )
    controller = NumaPlacementController(plan, backend)
    controller.apply(0x1000, 8192, private_anonymous=True, before_touch=True)
    assert backend.calls == [(0x1000, 8192, 1, (2,))]
    with pytest.raises(NumaPlacementError, match="private-anonymous"):
        controller.apply(0x1000, 8192, private_anonymous=False, before_touch=True)


def test_preferred_application_error_falls_back_but_bind_fails_closed():
    allowed = AllowedNumaNodes((0,), "available", "fixture")
    preferred = NumaPlacementController(
        resolve_numa_placement("preferred", 0, enforce=True, allowed=allowed),
        FakeBackend(error=OSError("denied")),
    )
    preferred.apply(0x1000, 4096, private_anonymous=True, before_touch=True)
    assert preferred.status == "fallback"
    assert preferred.telemetry()["fallback_reason"]

    bound = NumaPlacementController(
        resolve_numa_placement("bind", 0, enforce=True, allowed=allowed),
        FakeBackend(error=OSError("denied")),
    )
    with pytest.raises(NumaPlacementError, match="mbind failed"):
        bound.apply(0x1000, 4096, private_anonymous=True, before_touch=True)
    assert bound.status == "failed"


def test_false_mbind_result_is_an_application_failure():
    class FalseBackend:
        def mbind(self, *args):
            return False

    controller = NumaPlacementController(
        resolve_numa_placement("interleave", None, enforce=True, allowed=(0,)),
        FalseBackend(),
    )
    controller.apply(0x1000, 4096, private_anonymous=True, before_touch=True)
    assert controller.status == "fallback"


def test_sample_reports_counts_and_unknown_pages():
    backend = FakeBackend(statuses=(0, -14))
    controller = NumaPlacementController(
        resolve_numa_placement("interleave", None, enforce=True, allowed=(0, 1)), backend
    )
    controller.apply(0x2000, 8192, private_anonymous=True, before_touch=True)
    sample = controller.sample(0x2000, 8192, stride=4096, max_pages=2)
    assert sample == NumaSample("partial", ((0, 1),), 1, sampled_pages=2)
    assert controller.telemetry()["sample"]["status"] == "partial"


def test_sample_classifies_target_and_cross_node_pages():
    backend = FakeBackend(statuses=(0, 1, 1, -14))
    controller = NumaPlacementController(
        resolve_numa_placement(
            "preferred",
            0,
            enforce=True,
            allowed=AllowedNumaNodes((0, 1), "available", "fixture"),
        ),
        backend,
    )
    controller.apply(0x2000, 16384, private_anonymous=True, before_touch=True)
    controller.sample(0x2000, 16384, stride=4096, max_pages=4)

    sample = controller.telemetry()["sample"]
    assert sample["target_nodes"] == (0,)
    assert sample["target_pages"] == 1
    assert sample["off_target_pages"] == 2
    assert sample["off_target_counts"] == {"1": 2}
    assert sample["known_pages"] == 3
    assert sample["target_fraction"] == pytest.approx(1 / 3)
    assert sample["placement_match"] is False


def test_per_bank_sample_retains_typed_identity_but_aggregate_does_not():
    controller = NumaPlacementController(
        resolve_numa_placement(
            "bind", 0, enforce=True, allowed=AllowedNumaNodes((0,), "available", "fixture")
        ),
        FakeBackend(statuses=(0,)),
    )
    controller.apply(0x2000, 4096, private_anonymous=True, before_touch=True)
    identity = NumaSampleIdentity(bank_name="gate_up", layer_id=7)
    controller.sample(0x2000, 4096, max_pages=1, identity=identity)
    telemetry = controller.telemetry()
    assert telemetry["sample"]["identity"] is None
    assert telemetry["sample_banks"][0]["identity"] == {
        "bank_name": "gate_up",
        "layer_id": 7,
    }


@pytest.mark.parametrize(
    ("bank_name", "layer_id", "message"),
    [
        ("", 0, "bank_name"),
        (" gate_up", 0, "bank_name"),
        ("gate_up", -1, "layer_id"),
        ("gate_up", True, "layer_id"),
    ],
)
def test_sample_identity_rejects_ambiguous_values(bank_name, layer_id, message):
    with pytest.raises(ValueError, match=message):
        NumaSampleIdentity(bank_name=bank_name, layer_id=layer_id)


def test_sample_rejects_untyped_identity():
    with pytest.raises(ValueError, match="NumaSampleIdentity"):
        NumaSample("verified", identity=object())


def test_sample_target_match_is_unknown_without_known_pages():
    backend = FakeBackend(statuses=(-14,))
    controller = NumaPlacementController(
        resolve_numa_placement(
            "bind", 0, enforce=True, allowed=AllowedNumaNodes((0,), "available", "fixture")
        ),
        backend,
    )
    controller.apply(0x2000, 4096, private_anonymous=True, before_touch=True)
    controller.sample(0x2000, 4096, max_pages=1)
    sample = controller.telemetry()["sample"]
    assert sample["known_pages"] == 0
    assert sample["target_fraction"] is None
    assert sample["placement_match"] is None


def test_complete_on_target_sample_reports_verified_match():
    backend = FakeBackend(statuses=(1, 1))
    controller = NumaPlacementController(
        resolve_numa_placement(
            "bind", 1, enforce=True, allowed=AllowedNumaNodes((0, 1), "available", "fixture")
        ),
        backend,
    )
    controller.apply(0x2000, 8192, private_anonymous=True, before_touch=True)
    controller.sample(0x2000, 8192, max_pages=2)
    sample = controller.telemetry()["sample"]
    assert sample["target_pages"] == 2
    assert sample["off_target_pages"] == 0
    assert sample["target_fraction"] == 1.0
    assert sample["placement_match"] is True


def test_unknown_pages_are_excluded_from_fraction_and_prevent_positive_match():
    backend = FakeBackend(statuses=(0, -14))
    controller = NumaPlacementController(
        resolve_numa_placement(
            "preferred", 0, enforce=True, allowed=AllowedNumaNodes((0,), "available", "fixture")
        ),
        backend,
    )
    controller.apply(0x2000, 8192, private_anonymous=True, before_touch=True)
    controller.sample(0x2000, 8192, max_pages=2)
    sample = controller.telemetry()["sample"]
    assert sample["known_pages"] == 1
    assert sample["unknown"] == 1
    assert sample["target_fraction"] == 1.0
    assert sample["placement_match"] is None


def test_fallback_plan_never_claims_a_sample_match():
    controller = NumaPlacementController(
        resolve_numa_placement(
            "preferred", None, enforce=True, allowed=AllowedNumaNodes((0,), "available", "fixture")
        ),
        FakeBackend(),
    )
    sample = controller.telemetry()["sample"]
    assert sample["sampled_pages"] == 0
    assert sample["target_nodes"] == ()
    assert sample["placement_match"] is None


def test_sampling_aggregates_counts_unknown_and_errors_across_banks():
    class BanksBackend:
        def __init__(self):
            self.calls = 0

        def mbind(self, *args):
            return None

        def move_pages(self, addresses):
            self.calls += 1
            if self.calls == 1:
                return (0, -14)
            return (1, 1)

    controller = NumaPlacementController(
        resolve_numa_placement("interleave", None, enforce=True, allowed=(0, 1)),
        BanksBackend(),
    )
    controller.apply(0x1000, 8192, private_anonymous=True, before_touch=True)
    controller.sample(0x1000, 8192, stride=4096, max_pages=2)
    controller.sample(0x3000, 8192, stride=4096, max_pages=2)
    sample = controller.telemetry()["sample"]
    assert sample["status"] == "partial"
    assert sample["counts"] == {"0": 1, "1": 2}
    assert sample["unknown"] == 1
    assert sample["sampled_pages"] == 4
    assert len(controller.telemetry()["sample_banks"]) == 2


def test_sampling_aggregation_is_order_independent():
    class ReverseBackend:
        def __init__(self):
            self.calls = 0

        def mbind(self, *args):
            return None

        def move_pages(self, addresses):
            self.calls += 1
            return (1, 1) if self.calls == 1 else (0, -14)

    controller = NumaPlacementController(
        resolve_numa_placement("interleave", None, enforce=True, allowed=(0, 1)),
        ReverseBackend(),
    )
    controller.apply(0x1000, 8192, private_anonymous=True, before_touch=True)
    controller.sample(0x1000, 8192, stride=4096, max_pages=2)
    controller.sample(0x3000, 8192, stride=4096, max_pages=2)
    sample = controller.telemetry()["sample"]
    assert sample["status"] == "partial"
    assert sample["counts"] == {"0": 1, "1": 2}
    assert sample["unknown"] == 1
    assert sample["sampled_pages"] == 4


def test_sampling_status_is_unavailable_only_when_every_bank_is_unavailable():
    class UnavailableBackend:
        def mbind(self, *args):
            return None

        def move_pages(self, addresses):
            raise OSError("not permitted")

    controller = NumaPlacementController(
        resolve_numa_placement("interleave", None, enforce=True, allowed=(0,)),
        UnavailableBackend(),
    )
    controller.apply(0x1000, 4096, private_anonymous=True, before_touch=True)
    controller.sample(0x1000, 4096, max_pages=1)
    controller.sample(0x2000, 4096, max_pages=1)
    sample = controller.telemetry()["sample"]
    assert sample["status"] == "unavailable"
    assert sample["sampled_pages"] == 0
    assert len(sample["errors"]) == 2


def test_move_pages_low_level_uses_int_status_slots():
    class FakeLibc:
        def syscall(self, number, pid, count, pages, nodes, statuses, flags):
            count = int(count.value)
            pointer = ctypes.cast(statuses, ctypes.POINTER(ctypes.c_int))
            pointer[0] = 0
            pointer[1] = -14
            assert count == 2
            return 0

    backend = NumaSyscallBackend(system="Linux", machine="x86_64", libc=FakeLibc())
    assert backend.move_pages((0x1000, 0x2000)) == (0, -14)


def test_syscall_arch_map_is_fail_closed():
    assert linux_syscall_numbers(system="Linux", machine="x86_64") == {
        "mbind": 237,
        "move_pages": 279,
    }
    assert linux_syscall_numbers(system="Linux", machine="aarch64") is None
    backend = NumaSyscallBackend(system="Linux", machine="aarch64", libc=object())
    with pytest.raises(OSError) as error:
        backend.mbind(1, 4096, 1, (0,))
    assert error.value.errno == errno.ENOSYS


@pytest.mark.skipif(
    os.environ.get("FREETOKEN_RUN_GORILLA_NUMA_DIAGNOSTIC") != "1",
    reason="set FREETOKEN_RUN_GORILLA_NUMA_DIAGNOSTIC=1 for a live host diagnostic",
)
def test_gorilla_live_numa_diagnostic():
    result = resolve_allowed_numa_nodes()
    if not result.available:
        pytest.skip(f"live NUMA topology unavailable: {result.errors}")
    assert result.nodes
    mapping = mmap.mmap(
        -1,
        8192,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    try:
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        controller = NumaPlacementController(
            resolve_numa_placement(
                "preferred",
                result.nodes[0],
                enforce=True,
                allowed=result,
            ),
            NumaSyscallBackend(),
        )
        controller.apply(address, 8192, private_anonymous=True, before_touch=True)
        mapping[0] = 1  # first touch follows the placement request
        mapping[4096] = 1  # second page is touched before the residency query
        sample = controller.sample(address, 8192, stride=4096, max_pages=2)
        assert sample.status in {"verified", "partial", "unavailable"}
    finally:
        mapping.close()
