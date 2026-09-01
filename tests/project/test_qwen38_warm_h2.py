from __future__ import annotations

import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_qwen38_warm_h2.py"
SPEC = importlib.util.spec_from_file_location("run_qwen38_warm_h2", SCRIPT)
assert SPEC and SPEC.loader
WARM_H2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WARM_H2)


def _fixture() -> dict:
    return json.loads(
        (ROOT / "tests/fixtures/results/qwen38-gguf-cache-zero-warm-h2.json").read_text(
            encoding="utf-8"
        )
    )


def _inputs(tmp_path: Path) -> dict:
    fixture = _fixture()
    device = fixture["hardware_inventory"]["device"]
    inventory = {
        "profile_id": "ecc-off",
        "gpus": [
            {
                "uuid": device["uuid"],
                "pci_bus_id": device["pci_bus_id"],
                "topology": {
                    "pci_root": device["pci_root"],
                    "numa_node": device["numa_node"],
                },
            }
        ],
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    telemetry = {
        **device,
        "name": "Tesla P4",
        "compute_capability": "6.1",
        "memory_mib": 8192,
        "ecc_mode": "disabled",
    }
    return {
        "full_h2": {"ple_artifact": {"sha256": fixture["ple"]["runtime_integrity_sha256"]}},
        "identity": fixture["identity"],
        "inventory": inventory,
        "inventory_path": inventory_path,
        "telemetry_samples": [telemetry],
        "ple_before": {
            "lookup_rows": 0,
            "batch_unique_rows": 0,
            "application_reads": 0,
            "packed_bytes_read": 0,
            "storage_read_bytes": 0,
            "major_faults": 0,
        },
        "ple_after": {
            "backend": "pread",
            "lookup_rows": 96,
            "batch_unique_rows": 96,
            "application_reads": 96,
            "packed_bytes_read": 8640,
            "storage_read_bytes": 0,
            "major_faults": 0,
        },
        "startup_seconds": 1.0,
        "warmup_seconds": 2.0,
        "request_seconds": 3.0,
        "output_token_ids": [201519, 8691],
        "repository_commit": "1" * 40,
    }


def test_builder_emits_schema_valid_bounded_non_performance_evidence(tmp_path: Path) -> None:
    evidence = WARM_H2.build_evidence(**_inputs(tmp_path))
    validator = WARM_H2._load_module("validate_test_warm_h2", ROOT / "scripts/validate_evidence.py")

    assert validator.validate_document(evidence, schema_dir=ROOT / "schemas") == []
    assert evidence["performance"]["claim"] is False
    assert evidence["claims"]["thermal_qualification"] is False
    assert evidence["claims"]["dual_p4_serving"] is False
    assert evidence["identity"]["hash_reuse"]["model_shard_hashes_recomputed"] is True
    assert evidence["identity"]["hash_reuse"]["runtime_ple_integrity_hash"] == "performed"


def test_builder_rejects_request_over_thermal_bound(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["request_seconds"] = 301.0

    with pytest.raises(ValueError, match="300-second hard bound"):
        WARM_H2.build_evidence(**inputs)


def test_hard_deadline_can_be_cancelled_without_killing_process() -> None:
    class FakeProcess:
        terminated = False

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, *, timeout: int) -> int:
            assert timeout == 5
            return 0

    process = FakeProcess()
    commands: list[list[str]] = []

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        commands.append(command)
        assert kwargs
        return process

    deadline = WARM_H2._HardDeadline(1, popen=fake_popen)

    deadline.start()
    deadline.cancel()

    assert process.terminated is True
    assert commands[0][0] == sys.executable
    assert commands[0][1] == str(SCRIPT)
    assert commands[0][2] == "--watchdog"
    assert commands[0][-1] == "1"


def test_hard_deadline_expires_in_a_separate_victim_process() -> None:
    code = f"""
import importlib.util
import time
spec = importlib.util.spec_from_file_location('warm_deadline_victim', {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
deadline = module._HardDeadline(1)
deadline.start()
time.sleep(10)
"""

    victim = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    returncode = victim.wait(timeout=5)

    assert returncode == -signal.SIGKILL


def test_watchdog_exits_when_parent_dies_before_deadline() -> None:
    code = f"""
import importlib.util
spec = importlib.util.spec_from_file_location('warm_parent_death_victim', {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
deadline = module._HardDeadline(30)
deadline.start()
print(deadline._process.pid, flush=True)
"""
    victim = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    watchdog_pid = int(victim.stdout.strip())

    deadline = time.monotonic() + 3
    while Path(f"/proc/{watchdog_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.05)

    assert not Path(f"/proc/{watchdog_pid}").exists()


def test_model_shard_verification_rejects_same_size_content_drift(tmp_path: Path) -> None:
    shard = tmp_path / "model-00001-of-00001.gguf"
    shard.write_bytes(b"actual")
    expected = [{"name": shard.name, "size": 6, "sha256": "0" * 64}]

    with pytest.raises(RuntimeError, match="names/sizes/hashes"):
        WARM_H2._verify_model_shards([shard], expected)


def test_builder_rejects_inventory_identity_drift(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["telemetry_samples"][0]["uuid"] = "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"

    with pytest.raises(ValueError, match="visible GPU identity"):
        WARM_H2.build_evidence(**inputs)


def test_builder_rejects_missing_measured_ple_activity(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["ple_after"] = {"backend": "pread"}

    with pytest.raises(ValueError, match="telemetry did not advance"):
        WARM_H2.build_evidence(**inputs)
