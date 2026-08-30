#!/usr/bin/env python3
"""Verify the local wheel pair accepted by the Pascal installer.

This module deliberately uses only the Python standard library so the installer can run it
before uv, a virtual environment, or any project dependency exists.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path
from typing import Any

SCHEMA = "freetoken-pascal-bundle-v1"
RUNTIME_NAME = "freetoken-pascal"
CACHE_NAME = "freetoken-kernel-cache"
RUNTIME_METADATA = "freetoken/_pascal_build_meta.json"
CACHE_METADATA = "freetoken_kernel_cache/_pascal_build_meta.json"


class WheelBundleError(ValueError):
    """Raised when a wheel pair cannot prove Pascal compatibility."""


def _local_wheel(path: str | Path, label: str) -> Path:
    raw = str(path)
    if "://" in raw:
        raise WheelBundleError(f"{label} must be a local wheel path, not a URL")
    wheel = Path(raw)
    if wheel.suffix != ".whl":
        raise WheelBundleError(f"{label} is not a wheel path: {wheel}")
    if not wheel.is_file():
        raise WheelBundleError(f"{label} is not a readable local wheel: {wheel}")
    return wheel


def _metadata_file(archive: zipfile.ZipFile, wheel: Path) -> tuple[str, Any]:
    names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    if len(names) != 1:
        raise WheelBundleError(f"{wheel.name}: missing or ambiguous wheel METADATA")
    try:
        message = BytesParser(policy=compat32).parsebytes(archive.read(names[0]))
    except (OSError, ValueError, KeyError) as exc:
        raise WheelBundleError(f"{wheel.name}: malformed wheel METADATA ({exc})") from exc
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise WheelBundleError(f"{wheel.name}: wheel METADATA needs Name and Version")
    return name, version


def _read_json(archive: zipfile.ZipFile, wheel: Path, member: str) -> dict[str, Any]:
    try:
        raw = archive.read(member)
    except KeyError as exc:
        raise WheelBundleError(f"{wheel.name}: missing Pascal build metadata ({member})") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WheelBundleError(f"{wheel.name}: malformed Pascal build metadata ({member})") from exc
    if not isinstance(value, dict):
        raise WheelBundleError(f"{wheel.name}: Pascal build metadata must be a JSON object")
    return value


def _check_profile(wheel: Path, profile: dict[str, Any], role: str) -> None:
    if profile.get("schema") != SCHEMA:
        raise WheelBundleError(f"{wheel.name}: metadata schema must be {SCHEMA}")
    if profile.get("role") != role:
        raise WheelBundleError(f"{wheel.name}: metadata role must be {role}")
    if profile.get("profile") != "pascal":
        raise WheelBundleError(f"{wheel.name}: metadata profile must be pascal")
    if profile.get("cuda") != "12.6":
        raise WheelBundleError(f"{wheel.name}: metadata must prove CUDA 12.6")
    architectures = profile.get("architectures")
    if not isinstance(architectures, list) or not all(
        isinstance(item, str) for item in architectures
    ):
        raise WheelBundleError(f"{wheel.name}: metadata architectures must be a list")
    if "6.1" not in architectures and "sm_61" not in architectures:
        raise WheelBundleError(f"{wheel.name}: metadata architectures must include 6.1")
    if not isinstance(profile.get("version"), str) or not profile["version"]:
        raise WheelBundleError(f"{wheel.name}: metadata must contain a version")
    if not isinstance(profile.get("runtime_version"), str) or not profile["runtime_version"]:
        raise WheelBundleError(f"{wheel.name}: metadata must contain runtime_version")


def _inspect_wheel(
    path: str | Path, *, expected_name: str, role: str, metadata_member: str
) -> tuple[str, dict[str, Any]]:
    wheel = _local_wheel(path, role)
    try:
        with zipfile.ZipFile(wheel) as archive:
            name, version = _metadata_file(archive, wheel)
            if name != expected_name:
                raise WheelBundleError(
                    f"{wheel.name}: expected package {expected_name}, got {name}"
                )
            profile = _read_json(archive, wheel, metadata_member)
    except zipfile.BadZipFile as exc:
        raise WheelBundleError(f"{wheel.name}: not a readable wheel ZIP") from exc
    _check_profile(wheel, profile, role)
    if profile["version"] != version:
        raise WheelBundleError(f"{wheel.name}: embedded version does not match wheel METADATA")
    return version, profile


def verify_pascal_wheel_bundle(runtime: str | Path, cache: str | Path) -> dict[str, str]:
    """Verify and return the identity of an installable local Pascal wheel pair."""

    runtime_version, runtime_profile = _inspect_wheel(
        runtime,
        expected_name=RUNTIME_NAME,
        role="runtime",
        metadata_member=RUNTIME_METADATA,
    )
    cache_version, cache_profile = _inspect_wheel(
        cache,
        expected_name=CACHE_NAME,
        role="kernel-cache",
        metadata_member=CACHE_METADATA,
    )
    if cache_profile.get("runtime_version") != runtime_version:
        raise WheelBundleError(
            "kernel-cache metadata runtime_version does not match the runtime wheel version"
        )
    if runtime_profile["runtime_version"] != runtime_version:
        raise WheelBundleError("runtime metadata runtime_version does not match the runtime wheel")
    return {
        "runtime_version": runtime_version,
        "cache_version": cache_version,
        "cuda": "12.6",
        "architecture": "6.1",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, help="local freetoken runtime wheel")
    parser.add_argument("--kernel-cache", required=True, help="local freetoken kernel-cache wheel")
    args = parser.parse_args(argv)
    try:
        identity = verify_pascal_wheel_bundle(args.runtime, args.kernel_cache)
    except WheelBundleError as exc:
        print(f"[error] Pascal wheel bundle rejected: {exc}", file=sys.stderr)
        return 1
    print(
        "Pascal wheel bundle accepted: "
        f"runtime={identity['runtime_version']} cache={identity['cache_version']} "
        f"CUDA {identity['cuda']} arch {identity['architecture']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
