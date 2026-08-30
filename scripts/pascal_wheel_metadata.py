"""Build the small JSON profile embedded in FreeToken wheel packages."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "freetoken-pascal-bundle-v1"


def canonical_architectures(raw: str | None = None) -> list[str]:
    values = re.split(r"[\s,]+", raw.strip()) if raw and raw.strip() else ["6.1"]
    architectures: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if not normalized:
            continue
        if normalized.startswith("sm_"):
            normalized = f"{normalized[3:-1]}.{normalized[-1]}"
        if normalized not in architectures:
            architectures.append(normalized)
    return architectures


def canonical_cuda_version(raw: str | None) -> str:
    if not raw:
        return "unknown"
    match = re.match(r"^(\d+\.\d+)", raw.strip())
    return match.group(1) if match else "unknown"


def write_wheel_metadata(
    path: Path,
    *,
    role: str,
    version: str,
    cuda: str,
    architectures: list[str],
    runtime_version: str,
) -> None:
    profile = "pascal" if cuda == "12.6" and "6.1" in architectures else "modern"
    metadata: dict[str, Any] = {
        "schema": SCHEMA,
        "profile": profile,
        "role": role,
        "version": version,
        "runtime_version": runtime_version,
        "cuda": cuda,
        "architectures": architectures,
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
