from __future__ import annotations

import pytest
from freetoken.moe.host_memory import probe_swap

_CLEAR = {
    "/status": "Name:\tworker\nVmSwap:    0 kB\n",
    "/meminfo": "MemTotal: 8 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n",
    "/swaps": "Filename\tType\tSize\tUsed\tPriority\n",
}


def _reader(files: dict[str, str]):
    def read(path: str) -> str:
        return files[path]

    return read


def _probe(files: dict[str, str]):
    return probe_swap(
        read_text=_reader(files),
        status_path="/status",
        meminfo_path="/meminfo",
        swaps_path="/swaps",
    )


def test_clear_probe_parses_whitespace_and_reports_bytes():
    result = _probe(
        {
            "/status": "VmSwap:\t 1 kB\n",
            "/meminfo": "SwapTotal: 2048 kB\nSwapFree: 1024 kB\n",
            "/swaps": "Filename Type Size Used Priority\n",
        }
    )

    assert result.status == "swap-active"
    assert result.vm_swap_bytes == 1024
    assert result.swap_total_bytes == 2 * 1024**2
    assert result.swap_free_bytes == 1024**2
    assert result.active_swap_devices == ()
    assert result.as_dict()["no_swap_observed"] is False


def test_clear_probe_requires_zero_process_and_system_swap():
    result = _probe(_CLEAR)

    assert result.status == "clear"
    assert result.no_swap
    assert result.as_dict()["process_vm_swap_bytes"] == 0


def test_active_swap_device_is_reported_even_when_meminfo_is_zero():
    files = dict(_CLEAR)
    files["/swaps"] = "Filename Type Size Used Priority\n/dev/sda2 partition 1024 0 -2\n"

    result = _probe(files)

    assert result.status == "swap-active"
    assert result.active_swap_devices == ("/dev/sda2",)


def test_process_swap_is_reported_when_system_swap_is_empty():
    files = dict(_CLEAR)
    files["/status"] = "VmSwap: 4 kB\n"

    result = _probe(files)

    assert result.status == "process-swapped"
    assert result.vm_swap_bytes == 4096


@pytest.mark.parametrize(
    ("path", "text", "expected"),
    (
        ("/status", "VmSwap: nope\n", "ambiguous"),
        ("/status", "VmSwap: 0\n", "ambiguous"),
        ("/meminfo", "SwapTotal: 1 MiB\nSwapFree: 0 kB\n", "ambiguous"),
        ("/meminfo", "SwapTotal: 3 kB\nSwapFree: 4 kB\n", "ambiguous"),
        ("/swaps", "not a proc swaps file\n", "ambiguous"),
        ("/status", "VmSwap: 999999999999999999999999999999999999999999 kB\n", "ambiguous"),
    ),
)
def test_malformed_or_overflow_probe_is_ambiguous(path, text, expected):
    files = dict(_CLEAR)
    files[path] = text

    result = _probe(files)

    assert result.status == expected
    assert result.errors


def test_missing_or_permission_denied_procfs_is_unavailable():
    def denied(_path: str) -> str:
        raise PermissionError("denied")

    result = probe_swap(
        read_text=denied,
        status_path="/status",
        meminfo_path="/meminfo",
        swaps_path="/swaps",
    )

    assert result.status == "unavailable"
    assert len(result.errors) == 3


def test_unknown_units_are_ambiguous():
    files = dict(_CLEAR)
    files["/meminfo"] = "SwapTotal: 1 furlong\nSwapFree: 0 kB\n"

    result = _probe(files)

    assert result.status == "ambiguous"
    assert "exact kB unit" in result.errors[0]


def test_swap_row_with_used_bytes_above_size_is_ambiguous():
    files = dict(_CLEAR)
    files["/swaps"] = "Filename Type Size Used Priority\n/dev/sda2 partition 1 2 -2\n"

    result = _probe(files)

    assert result.status == "ambiguous"
    assert "used exceeds size" in result.errors[0]


def test_swap_row_with_embedded_unit_is_ambiguous():
    files = dict(_CLEAR)
    files["/swaps"] = "Filename Type Size Used Priority\n/dev/sda2 partition 1 kB 0 -2\n"

    result = _probe(files)

    assert result.status == "ambiguous"
    assert "malformed active swap row" in result.errors[0]


def _raise_runtime(_path: str) -> str:
    raise RuntimeError("boom")


@pytest.mark.parametrize("bad_reader", (lambda _path: 1, _raise_runtime))
def test_bad_reader_results_in_unavailable_or_ambiguous_without_swallowing_base_exceptions(
    bad_reader,
):
    result = probe_swap(
        read_text=bad_reader,
        status_path="/status",
        meminfo_path="/meminfo",
        swaps_path="/swaps",
    )

    assert result.status in {"unavailable", "ambiguous"}
    assert result.errors


def test_probe_does_not_swallow_keyboard_interrupt():
    def interrupt(_path: str) -> str:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        probe_swap(read_text=interrupt)
