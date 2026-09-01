from __future__ import annotations

import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_qsa_pascal_h2.py"
SPEC = importlib.util.spec_from_file_location("run_qsa_pascal_h2", SCRIPT)
assert SPEC and SPEC.loader
QSA_H2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QSA_H2)


def _telemetry_output(*_args: object, **_kwargs: object) -> str:
    return (
        "0, Tesla P4, GPU-7af15080-238c-fd1d-66da-4f8984053664, 6.1, 8192, "
        "00000000:02:00.0, 607, 3003, 35, 7.19, 75.00, Disabled\n"
    )


def test_capture_telemetry_normalizes_identity_and_ecc() -> None:
    sample = QSA_H2.capture_telemetry(check_output=_telemetry_output)

    assert sample["pci_bus_id"] == "0000:02:00.0"
    assert sample["ecc_mode"] == "disabled"
    assert sample["compute_capability"] == "6.1"
    assert sample["power_limit_watts"] == 75.0


def test_capture_telemetry_rejects_wrong_gpu_index() -> None:
    def wrong_index(*_args: object, **_kwargs: object) -> str:
        return _telemetry_output().replace("0, Tesla", "1, Tesla", 1)

    with pytest.raises(RuntimeError, match="index mismatch"):
        QSA_H2.capture_telemetry(check_output=wrong_index)


class _FakeEvent:
    def __init__(self) -> None:
        self.recorded = False

    def record(self) -> None:
        self.recorded = True

    def elapsed_time(self, other: _FakeEvent) -> float:
        assert self.recorded and other.recorded
        return 1.25


class _FakeCuda:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    @staticmethod
    def Event(*, enable_timing: bool = False) -> _FakeEvent:
        assert enable_timing
        return _FakeEvent()

    def synchronize(self, *_args: object) -> None:
        self.synchronize_calls += 1


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()


def test_allocator_snapshot_preserves_cuda_free_total_order() -> None:
    class _AllocatorCuda:
        @staticmethod
        def mem_get_info(_device: object) -> tuple[int, int]:
            return (3_000, 8_000)

        @staticmethod
        def memory_stats(_device: object) -> dict[str, int]:
            return {
                "allocation.all.current": 4,
                "num_alloc_retries": 1,
                "num_ooms": 0,
            }

        @staticmethod
        def memory_allocated(_device: object) -> int:
            return 1_000

        @staticmethod
        def memory_reserved(_device: object) -> int:
            return 2_000

        @staticmethod
        def max_memory_allocated(_device: object) -> int:
            return 2_500

    snapshot = QSA_H2._allocator_snapshot(
        SimpleNamespace(cuda=_AllocatorCuda()), object()
    )

    assert snapshot["driver_free_bytes"] == 3_000
    assert snapshot["driver_total_bytes"] == 8_000


def test_phase_event_collector_defers_synchronization_until_forward_finishes() -> None:
    fake_torch = _FakeTorch()
    collector = QSA_H2.QSAPhaseEventCollector(fake_torch)

    collector.begin_forward()
    collector(
        "begin",
        {"phase": "selection_composite", "layer_id": 3, "slot": 0, "path": "torch-fp32-reference"},
    )
    assert fake_torch.cuda.synchronize_calls == 0
    collector(
        "end",
        {"phase": "selection_composite", "layer_id": 3, "slot": 0, "path": "torch-fp32-reference"},
    )
    assert fake_torch.cuda.synchronize_calls == 0

    records = collector.finish()

    assert fake_torch.cuda.synchronize_calls == 0
    assert records == [
        {
            "phase": "selection_composite",
            "layer_id": 3,
            "slot": 0,
            "path": "torch-fp32-reference",
            "elapsed_ns": 1_250_000,
        }
    ]


def test_phase_event_collector_rejects_unbalanced_callbacks() -> None:
    collector = QSA_H2.QSAPhaseEventCollector(_FakeTorch())

    collector.begin_forward()
    with pytest.raises(RuntimeError, match="without begin"):
        collector(
            "end", {"phase": "store_kv", "layer_id": 3, "slot": 0, "path": "torch-fp32-reference"}
        )


def test_main_removes_stale_qsa_evidence_before_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "evidence.json"
    temporary = tmp_path / "evidence.json.tmp"
    output.write_text("stale", encoding="utf-8")
    temporary.write_text("partial", encoding="utf-8")

    def fake_probe(**kwargs: object) -> dict[str, object]:
        assert not output.exists()
        assert not temporary.exists()
        return {}

    monkeypatch.setattr(QSA_H2, "run_probe", fake_probe)

    assert (
        QSA_H2.main(
            [
                "--inventory",
                str(tmp_path / "inventory.json"),
                "--output",
                str(output),
                "--expected-profile",
                "ecc-off",
                "--repository-commit",
                "1" * 40,
            ]
        )
        == 0
    )


def test_fixture_is_bound_to_qsa_registry_contract() -> None:
    fixture = json.loads(
        (ROOT / "tests/fixtures/results/qwen38-qsa-h2-evidence.json").read_text(encoding="utf-8")
    )
    assert fixture["profile"]["attention_backend"] == "qsa_sparse"
    assert all(
        item["selected_path"] == fixture["profile"]["selected_path"]
        for item in fixture["workspace_plans"]
    )
    assert all(
        {event["phase"] for event in sample["phase_events"]}
        == {"store_kv", "index_cache_composite", "selection_composite", "selected_row_attention"}
        for sample in fixture["samples"]
    )


def test_workspace_record_uses_resolved_qsa_rotary_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Plan:
        @staticmethod
        def as_dict() -> dict[str, object]:
            return {}

    class _Inputs:
        def __init__(self, **values: object) -> None:
            self.__dict__.update(values)

    def fake_calculate(request: object) -> _Plan:
        captured["max_position"] = getattr(request, "max_position")
        return _Plan()

    workspace_module = SimpleNamespace(
        QSAWorkspaceInputs=_Inputs,
        calculate_qsa_workspace=fake_calculate,
    )
    monkeypatch.setitem(sys.modules, "freetoken", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "freetoken.attention", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "freetoken.attention.qsa_workspace", workspace_module)
    rotary_config = SimpleNamespace(max_position=262_144, rotary_dim=128)
    group = SimpleNamespace(name="full", layer_ids=(3,), rotary_config=rotary_config)
    qwen_args = SimpleNamespace(
        index_n_heads=4,
        index_head_dim=128,
        index_budget=2048,
        index_ratio=16,
    )
    config = SimpleNamespace(
        qwen4_args=qwen_args,
        attention_groups=(group,),
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=256,
        rotary_config=rotary_config,
    )
    fake_fixture = type(
        "Fixture",
        (),
        {
            "config": config,
            "backend": type("Backend", (), {"selected_path": "torch-fp32-reference"})(),
            "page_table": type("Table", (), {"shape": (9, 4096)})(),
            "page_size": 64,
            "num_req_slots": 9,
            "pool": type("Pool", (), {"ring_capacity": 4096})(),
        },
    )()

    record = QSA_H2._workspace_record(fake_fixture, 128, "prefill")

    assert record["plan"] == {}
    assert captured["max_position"] == group.rotary_config.max_position


def test_hardware_gate_has_a_profile_bound_bounded_qsa_level() -> None:
    gate = (ROOT / "scripts/ci/hardware_gate.sh").read_text(encoding="utf-8")

    assert "qsa-p4)" in gate
    branch = gate.split("run_qsa_p4()", 1)[1].split('case "$level"', 1)[0]
    assert "timeout --foreground --signal=TERM --kill-after=5s 300s" in branch
    assert "scripts/run_qsa_pascal_h2.py" in branch
    assert '--inventory "$inventory_path"' in branch
    assert '--expected-profile "$profile_id"' in branch
    assert "qwen38-qsa-h2-${profile_id}.json" in branch
