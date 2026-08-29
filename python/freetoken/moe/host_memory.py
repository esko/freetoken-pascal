"""Read-only host-memory pressure probes used by explicit residency policies.

The probe intentionally observes Linux procfs only.  It never changes swap state,
memory policy, process limits, or page residency.  A missing or ambiguous field is
reported as a failed probe so callers that require a no-swap admission can fail closed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SwapProbeStatus = Literal["clear", "swap-active", "process-swapped", "unavailable", "ambiguous"]
_MAX_BYTES = (1 << 63) - 1
_QUANTITY_RE = re.compile(r"^(?P<value>[0-9]+)(?:\s*(?P<unit>[A-Za-z]+))?$")


@dataclass(frozen=True, slots=True)
class SwapProbe:
    """A point-in-time, read-only swap observation."""

    vm_swap_bytes: int | None
    swap_total_bytes: int | None
    swap_free_bytes: int | None
    active_swap_devices: tuple[str, ...]
    source: str
    errors: tuple[str, ...]
    status: SwapProbeStatus

    @property
    def no_swap(self) -> bool:
        """Whether procfs established that this process and system have no swap."""
        return self.status == "clear"

    @property
    def no_swap_observed(self) -> bool | None:
        """Whether no swap was observed, or ``None`` when observation failed."""
        if self.status == "clear":
            return True
        if self.status in {"swap-active", "process-swapped"}:
            return False
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "source": self.source,
            "errors": list(self.errors),
            "vm_swap_bytes": self.vm_swap_bytes,
            "process_vm_swap_bytes": self.vm_swap_bytes,
            "swap_total_bytes": self.swap_total_bytes,
            "swap_free_bytes": self.swap_free_bytes,
            "active_swap_devices": list(self.active_swap_devices),
            "no_swap_observed": self.no_swap_observed,
        }


def _read_default(path: str) -> str:
    return Path(path).read_text(encoding="ascii")


def _quantity(
    raw: str,
    *,
    default_unit: str,
    field: str,
    required_unit: str | None = None,
) -> int:
    match = _QUANTITY_RE.fullmatch(raw.strip())
    if match is None:
        raise ValueError(f"{field} is not a non-negative quantity: {raw!r}")
    value = int(match.group("value"))
    raw_unit = match.group("unit")
    if required_unit is not None and raw_unit != required_unit:
        raise ValueError(f"{field} must use the exact {required_unit} unit")
    unit = (raw_unit or default_unit).lower()
    multipliers = {
        "b": 1,
        "kb": 1024,
        "kib": 1024,
        "mb": 1024**2,
        "mib": 1024**2,
        "gb": 1024**3,
        "gib": 1024**3,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    multiplier = multipliers.get(unit)
    if multiplier is None:
        raise ValueError(f"{field} has unsupported unit {unit!r}")
    if value > _MAX_BYTES // multiplier:
        raise ValueError(f"{field} exceeds supported byte range")
    return value * multiplier


def _field(
    text: str,
    name: str,
    *,
    default_unit: str,
    source: str,
    required_unit: str | None = None,
) -> int:
    matches = []
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == name:
            matches.append(value)
    if len(matches) != 1:
        if not matches:
            raise ValueError(f"{source} is missing {name}")
        raise ValueError(f"{source} contains duplicate {name}")
    return _quantity(
        matches[0],
        default_unit=default_unit,
        field=f"{source} {name}",
        required_unit=required_unit,
    )


def _active_devices(text: str, *, source: str) -> tuple[str, ...]:
    lines = [line.split() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{source} is empty")
    if lines[0] != ["Filename", "Type", "Size", "Used", "Priority"]:
        raise ValueError(f"{source} has an invalid header")
    devices: list[str] = []
    for columns in lines[1:]:
        if len(columns) != 5:
            raise ValueError(f"{source} has a malformed active swap row")
        device = columns[0]
        size = _swap_row_quantity(columns[2], field=f"{source} {device} size")
        used = _swap_row_quantity(columns[3], field=f"{source} {device} used")
        if used > size:
            raise ValueError(f"{source} {device} used exceeds size")
        try:
            int(columns[4])
        except ValueError as error:
            raise ValueError(f"{source} {device} has an invalid priority") from error
        devices.append(device)
    return tuple(devices)


def _swap_row_quantity(raw: str, *, field: str) -> int:
    """Parse the unitless kB columns used by the exact ``/proc/swaps`` format."""
    if not re.fullmatch(r"[0-9]+", raw.strip()):
        raise ValueError(f"{field} must be an unadorned non-negative kB quantity")
    return _quantity(f"{raw.strip()} kB", default_unit="kB", field=field, required_unit="kB")


def probe_swap(
    *,
    read_text: Callable[[str], str] | None = None,
    status_path: str = "/proc/self/status",
    meminfo_path: str = "/proc/meminfo",
    swaps_path: str = "/proc/swaps",
) -> SwapProbe:
    """Read process/system swap state from procfs using an injectable reader."""
    reader = _read_default if read_text is None else read_text
    errors: list[str] = []
    ambiguous = False
    values: dict[str, int | None] = {
        "vm_swap_bytes": None,
        "swap_total_bytes": None,
        "swap_free_bytes": None,
    }

    try:
        status = reader(status_path)
    except Exception as error:
        errors.append(f"{status_path}: unavailable ({error})")
    else:
        try:
            values["vm_swap_bytes"] = _field(
                status,
                "VmSwap",
                default_unit="kB",
                source=status_path,
                required_unit="kB",
            )
        except Exception as error:
            errors.append(f"{status_path}: ambiguous ({error})")
            ambiguous = True

    try:
        meminfo = reader(meminfo_path)
    except Exception as error:
        errors.append(f"{meminfo_path}: unavailable ({error})")
    else:
        for name, key in (("SwapTotal", "swap_total_bytes"), ("SwapFree", "swap_free_bytes")):
            try:
                values[key] = _field(
                    meminfo,
                    name,
                    default_unit="kB",
                    source=meminfo_path,
                    required_unit="kB",
                )
            except Exception as error:
                errors.append(f"{meminfo_path}: ambiguous ({error})")
                ambiguous = True

    devices: tuple[str, ...] = ()
    try:
        swaps = reader(swaps_path)
    except Exception as error:
        errors.append(f"{swaps_path}: unavailable ({error})")
    else:
        try:
            devices = _active_devices(swaps, source=swaps_path)
        except Exception as error:
            errors.append(f"{swaps_path}: ambiguous ({error})")
            ambiguous = True

    total = values["swap_total_bytes"]
    free = values["swap_free_bytes"]
    if total is not None and free is not None:
        if free > total:
            errors.append(f"{meminfo_path} SwapFree exceeds SwapTotal")
            ambiguous = True
        if total == 0 and free != 0:
            errors.append(f"{meminfo_path} reports free swap with zero total")
            ambiguous = True

    if errors:
        status: SwapProbeStatus = "ambiguous" if ambiguous else "unavailable"
    elif devices or (total is not None and total > 0):
        status = "swap-active"
    elif values["vm_swap_bytes"] is not None and values["vm_swap_bytes"] > 0:
        status = "process-swapped"
    else:
        status = "clear"
    return SwapProbe(
        vm_swap_bytes=values["vm_swap_bytes"],
        swap_total_bytes=total,
        swap_free_bytes=free,
        active_swap_devices=devices,
        source="procfs",
        errors=tuple(errors),
        status=status,
    )


__all__ = ["SwapProbe", "SwapProbeStatus", "probe_swap"]
