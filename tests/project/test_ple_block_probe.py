from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from freetoken.ple_io_evidence import (
    LinuxPLEBlockCounterProbe,
    PLEBlockProbeError,
    PLEIOCounters,
    PLEIOEvidenceError,
    PLEIOEvidenceRecorder,
)


def _payload_stat(path: Path, *, major: int, minor: int) -> os.stat_result:
    """Return a real regular-file stat with an injected device identity."""
    result = os.stat(path)
    values = list(result)
    # os.stat_result uses st_mode at index 0 and st_dev at index 2.
    values[0] = stat.S_IFREG | 0o644
    values[2] = os.makedev(major, minor)
    return os.stat_result(values)


def _write_block_device(
    sysfs: Path,
    *,
    major: int,
    minor: int,
    name: str = "nvme0n1",
    sectors_read: int = 10,
    logical_block_size: int = 512,
    virtual: bool = False,
) -> Path:
    target_root = "virtual" if virtual else "pci"
    target = sysfs / "devices" / target_root / "block" / name
    target.mkdir(parents=True)
    (target / "dev").write_text(f"{major}:{minor}\n", encoding="ascii")
    (target / "stat").write_text(
        f"8 2 {sectors_read} 4 5 6 7 8 9 10 11\n", encoding="ascii"
    )
    queue = target / "queue"
    queue.mkdir()
    (queue / "logical_block_size").write_text(f"{logical_block_size}\n", encoding="ascii")
    dev_link = sysfs / "dev" / "block"
    dev_link.mkdir(parents=True, exist_ok=True)
    (dev_link / f"{major}:{minor}").symlink_to(target)
    return target


def _base(**changes: object) -> PLEIOCounters:
    values: dict[str, object] = {
        "major_faults": 2,
        "logical_packed_bytes": 100,
        "application_bytes_read": 100,
        "application_reads": 1,
    }
    values.update(changes)
    return PLEIOCounters(**values)


def test_probe_resolves_payload_device_and_converts_sectors(tmp_path: Path) -> None:
    payload = tmp_path / "ple.bin"
    payload.write_bytes(b"payload")
    sysfs = tmp_path / "sys"
    _write_block_device(sysfs, major=259, minor=0, sectors_read=17, logical_block_size=4096)

    probe = LinuxPLEBlockCounterProbe(
        payload,
        sysfs_root=sysfs,
        counter_source=lambda: _base(),
        stat_fn=lambda path: _payload_stat(Path(path), major=259, minor=0),
    )

    counters = probe.sample()

    assert counters.physical_block_device_bytes == 17 * 4096
    assert counters.device_identity == "nvme0n1"
    assert counters.physical_source == "sysfs-block-stat"
    assert counters.physical_source_detail is not None
    assert "sectors-read" in counters.physical_source_detail
    assert "logical-block-size=4096" in counters.physical_source_detail
    assert counters.major_faults == 2
    assert counters.logical_packed_bytes == 100


def test_probe_accepts_explicit_sector_size_when_queue_metadata_is_unavailable(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "ple.bin"
    payload.write_bytes(b"payload")
    sysfs = tmp_path / "sys"
    target = _write_block_device(sysfs, major=8, minor=1, sectors_read=3)
    (target / "queue" / "logical_block_size").unlink()

    probe = LinuxPLEBlockCounterProbe(
        payload,
        sysfs_root=sysfs,
        sector_size=1024,
        counter_source=lambda: _base(),
        stat_fn=lambda path: _payload_stat(Path(path), major=8, minor=1),
    )

    assert probe.sample().physical_block_device_bytes == 3 * 1024


def test_probe_follows_partition_parent_and_single_slave(tmp_path: Path) -> None:
    payload = tmp_path / "ple.bin"
    payload.write_bytes(b"payload")
    sysfs = tmp_path / "sys"
    parent = _write_block_device(sysfs, major=259, minor=0, sectors_read=23)
    partition = parent / "nvme0n1p1"
    partition.mkdir()
    (partition / "dev").write_text("259:1\n", encoding="ascii")
    (partition / "partition").write_text("1\n", encoding="ascii")
    (sysfs / "dev" / "block" / "259:1").symlink_to(partition)

    probe = LinuxPLEBlockCounterProbe(
        payload,
        sysfs_root=sysfs,
        counter_source=lambda: _base(),
        stat_fn=lambda path: _payload_stat(Path(path), major=259, minor=1),
    )
    assert probe.sample().device_identity == "nvme0n1"
    assert probe.sample().physical_block_device_bytes == 23 * 512

    dm = _write_block_device(sysfs, major=253, minor=0, name="dm-0", sectors_read=1)
    slaves = dm / "slaves"
    slaves.mkdir()
    (slaves / "nvme0n1").symlink_to(parent)
    (sysfs / "dev" / "block" / "253:0").unlink()
    (sysfs / "dev" / "block" / "253:0").symlink_to(dm)
    dm_probe = LinuxPLEBlockCounterProbe(
        payload,
        sysfs_root=sysfs,
        counter_source=lambda: _base(),
        stat_fn=lambda path: _payload_stat(Path(path), major=253, minor=0),
    )
    assert dm_probe.sample().device_identity == "nvme0n1"
    assert dm_probe.sample().physical_block_device_bytes == 23 * 512


@pytest.mark.parametrize(
    "mutator",
    [
        lambda sysfs, target: (target / "stat").write_text("8 2 nope 4\n", encoding="ascii"),
        lambda sysfs, target: (target / "queue" / "logical_block_size").write_text(
            "0\n", encoding="ascii"
        ),
    ],
)
def test_probe_rejects_malformed_sysfs_counters(tmp_path: Path, mutator) -> None:
    payload = tmp_path / "ple.bin"
    payload.write_bytes(b"payload")
    sysfs = tmp_path / "sys"
    target = _write_block_device(sysfs, major=8, minor=1)
    mutator(sysfs, target)
    probe = LinuxPLEBlockCounterProbe(
        payload,
        sysfs_root=sysfs,
        counter_source=lambda: _base(),
        stat_fn=lambda path: _payload_stat(Path(path), major=8, minor=1),
    )

    with pytest.raises(PLEBlockProbeError):
        probe.sample()


@pytest.mark.parametrize("count", [0, 2])
def test_probe_rejects_missing_or_ambiguous_slaves(tmp_path: Path, count: int) -> None:
    payload = tmp_path / "ple.bin"
    payload.write_bytes(b"payload")
    sysfs = tmp_path / "sys"
    target = _write_block_device(sysfs, major=253, minor=0, name="dm-0", virtual=True)
    slaves = target / "slaves"
    slaves.mkdir()
    for index in range(count):
        slave = _write_block_device(
            sysfs,
            major=8,
            minor=index + 1,
            name=f"sda{index + 1}",
        )
        (slaves / slave.name).symlink_to(slave)
    probe = LinuxPLEBlockCounterProbe(
        payload,
        sysfs_root=sysfs,
        counter_source=lambda: _base(),
        stat_fn=lambda path: _payload_stat(Path(path), major=253, minor=0),
    )

    if count == 0:
        # A virtual block device with no discoverable backing is not physical evidence.
        with pytest.raises(PLEBlockProbeError, match="backing"):
            probe.sample()
    else:
        with pytest.raises(PLEBlockProbeError, match="ambiguous"):
            probe.sample()


def test_probe_rejects_cycle_and_missing_mapping(tmp_path: Path) -> None:
    payload = tmp_path / "ple.bin"
    payload.write_bytes(b"payload")
    sysfs = tmp_path / "sys"
    target = _write_block_device(sysfs, major=253, minor=0, name="dm-0")
    slaves = target / "slaves"
    slaves.mkdir()
    (slaves / "dm-0").symlink_to(target)
    probe = LinuxPLEBlockCounterProbe(
        payload,
        sysfs_root=sysfs,
        counter_source=lambda: _base(),
        stat_fn=lambda path: _payload_stat(Path(path), major=253, minor=0),
    )
    with pytest.raises(PLEBlockProbeError, match="cycle"):
        probe.sample()

    missing = LinuxPLEBlockCounterProbe(
        payload,
        sysfs_root=sysfs / "missing",
        counter_source=lambda: _base(),
        stat_fn=lambda path: _payload_stat(Path(path), major=8, minor=0),
    )
    with pytest.raises(PLEBlockProbeError, match="mapping"):
        missing.sample()


@pytest.mark.parametrize("major", [0])
def test_probe_rejects_non_block_payload_filesystem(tmp_path: Path, major: int) -> None:
    payload = tmp_path / "ple.bin"
    payload.write_bytes(b"payload")
    probe = LinuxPLEBlockCounterProbe(
        payload,
        sysfs_root=tmp_path / "sys",
        stat_fn=lambda path: _payload_stat(Path(path), major=major, minor=42),
    )
    with pytest.raises(PLEBlockProbeError, match="non-block"):
        probe.sample()

    directory = tmp_path / "payload-dir"
    directory.mkdir()
    probe = LinuxPLEBlockCounterProbe(
        directory,
        sysfs_root=tmp_path / "sys",
        stat_fn=os.stat,
    )
    with pytest.raises(PLEBlockProbeError, match="regular file"):
        probe.sample()


def test_probe_rejects_non_linux_platform(tmp_path: Path) -> None:
    with pytest.raises(PLEBlockProbeError, match="unsupported"):
        LinuxPLEBlockCounterProbe(tmp_path / "ple.bin", system="Darwin")


def test_recorder_detects_regressing_sysfs_counter(tmp_path: Path) -> None:
    payload = tmp_path / "ple.bin"
    payload.write_bytes(b"payload")
    sysfs = tmp_path / "sys"
    target = _write_block_device(sysfs, major=8, minor=1, sectors_read=5)
    snapshots = iter((5, 4))
    calls = 0

    def base_source():
        nonlocal calls
        sectors = next(snapshots)
        (target / "stat").write_text(f"8 2 {sectors} 4 5\n", encoding="ascii")
        calls += 1
        return _base(logical_packed_bytes=100 + calls * 10)

    probe = LinuxPLEBlockCounterProbe(
        payload,
        sysfs_root=sysfs,
        counter_source=base_source,
        stat_fn=lambda path: _payload_stat(Path(path), major=8, minor=1),
    )
    recorder = PLEIOEvidenceRecorder(probe.sample)
    recorder.begin_phase("cold")
    with pytest.raises(PLEIOEvidenceError, match="regression"):
        recorder.end_phase("cold")


@pytest.mark.skipif(
    os.environ.get("FREETOKEN_RUN_GORILLA_PLE_BLOCK_PROBE") != "1",
    reason="set FREETOKEN_RUN_GORILLA_PLE_BLOCK_PROBE=1 for a live Gorilla probe",
)
def test_gorilla_live_ple_block_probe_is_non_destructive() -> None:
    payload_name = os.environ.get("FREETOKEN_GORILLA_PLE_PAYLOAD")
    if not payload_name:
        pytest.skip("FREETOKEN_GORILLA_PLE_PAYLOAD is not set")
    payload = Path(payload_name)
    if not payload.is_file():
        pytest.skip(f"PLE payload is not a regular file: {payload}")
    try:
        counters = LinuxPLEBlockCounterProbe(payload).sample()
    except PLEBlockProbeError as error:
        pytest.skip(f"live payload filesystem is unsupported: {error}")
    assert counters.physical_source == "sysfs-block-stat"
    assert counters.device_identity
    assert counters.physical_block_device_bytes is not None
