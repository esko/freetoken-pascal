from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_router_pascal_h2.py"
SPEC = importlib.util.spec_from_file_location("run_router_pascal_h2", SCRIPT)
assert SPEC and SPEC.loader
ROUTER_H2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUTER_H2)


def _telemetry_output(*_args: object, **_kwargs: object) -> str:
    return (
        "0, Tesla P4, GPU-7af15080-238c-fd1d-66da-4f8984053664, 6.1, 8192, "
        "00000000:02:00.0, 607, 3003, 35, 7.19, 75.00, Disabled\n"
    )


def test_capture_telemetry_normalizes_identity_and_ecc() -> None:
    sample = ROUTER_H2.capture_telemetry(0, check_output=_telemetry_output)

    assert sample["pci_bus_id"] == "0000:02:00.0"
    assert sample["ecc_mode"] == "disabled"
    assert sample["compute_capability"] == "6.1"
    assert sample["power_limit_watts"] == 75.0


def test_tensor_hash_accepts_bfloat16_and_hashes_exact_bytes() -> None:
    torch = pytest.importorskip("torch")
    value = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)

    assert ROUTER_H2._sha256_tensor(value) == ROUTER_H2._sha256_bytes(
        value.view(torch.uint8).numpy().tobytes()
    )


def test_capture_telemetry_fails_closed_on_wrong_device() -> None:
    def wrong_index(*_args: object, **_kwargs: object) -> str:
        return _telemetry_output().replace("0, Tesla", "1, Tesla", 1)

    with pytest.raises(RuntimeError, match="index mismatch"):
        ROUTER_H2.capture_telemetry(0, check_output=wrong_index)


def test_summary_requires_every_steady_observation_to_pass() -> None:
    samples = [
        {
            "reference": {
                "status": "passed",
                "ids_exact": True,
                "weights_within_tolerance": True,
            },
            "candidate": {
                "status": "passed",
                "ids_exact": True,
                "weights_within_tolerance": True,
            },
        }
    ]
    case = {
        "candidate": {"eligible": True, "status": "passed"},
        "comparison": {"passed": True},
        "steady_samples": samples,
    }
    assert ROUTER_H2._summary([case])["all_candidate_parity_passed"] is True

    samples[0]["candidate"]["weights_within_tolerance"] = False
    assert ROUTER_H2._summary([case])["all_candidate_parity_passed"] is False


def test_probe_rejects_nonzero_inventory_index_before_hardware_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires inventory GPU index 0"):
        ROUTER_H2.run_probe(
            inventory_path=tmp_path / "unused.json",
            output_path=tmp_path / "unused-output.json",
            expected_profile="ecc-off",
            gpu_index=1,
            repository_commit="1" * 40,
        )


def test_main_removes_stale_evidence_before_probe(
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

    monkeypatch.setattr(ROUTER_H2, "run_probe", fake_probe)

    assert ROUTER_H2.main(
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
    ) == 0
