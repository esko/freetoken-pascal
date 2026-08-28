#!/usr/bin/env python3
"""Build the standalone Q4/mixed AVX2 helpers used by the target-CPU benchmark.

The command mirrors the split baseline/AVX2 g++ invocations in the native-kernel
tests and does not import Torch or require CUDA.  It writes both shared libraries
and a JSON manifest containing compiler flags and SHA-256 values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIR = ROOT / "python/freetoken/moe"
BASELINE_FLAGS = ("-mno-avx", "-mno-avx2", "-mno-fma")
AVX2_FLAGS = ("-mavx2", "-mfma")
COMMON_FLAGS = ("-std=c++17", "-O2", "-fPIC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _compile_library(
    *,
    cxx: str,
    output_dir: Path,
    stem: str,
    dispatch: Path,
    scalar: Path,
    avx2: Path,
) -> dict[str, object]:
    object_specs = (
        ("dispatch", dispatch, BASELINE_FLAGS),
        ("scalar", scalar, BASELINE_FLAGS),
        ("avx2", avx2, AVX2_FLAGS),
    )
    objects: dict[str, dict[str, object]] = {}
    for object_name, source, isa_flags in object_specs:
        output = output_dir / f"{stem}_{object_name}.o"
        command = [
            cxx,
            *COMMON_FLAGS,
            *isa_flags,
            "-I",
            str(INCLUDE_DIR),
            "-c",
            str(source),
            "-o",
            str(output),
        ]
        _run(command)
        objects[object_name] = {
            "path": str(output),
            "source": str(source),
            "source_sha256": _sha256(source),
            "flags": list((*COMMON_FLAGS, *isa_flags)),
        }
    library = output_dir / f"{stem}.so"
    link_command = [
        cxx,
        "-shared",
        *(str(objects[name]["path"]) for name in ("dispatch", "scalar", "avx2")),
        "-o",
        str(library),
    ]
    _run(link_command)
    return {
        "path": str(library),
        "sha256": _sha256(library),
        "objects": objects,
        "link_flags": ["-shared"],
    }


def build(*, cxx: str, output_dir: Path, command: str) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    q4 = _compile_library(
        cxx=cxx,
        output_dir=output_dir,
        stem="q4_k_native",
        dispatch=ROOT / "python/freetoken/moe/q4_k_native.cpp",
        scalar=ROOT / "python/freetoken/moe/q4_k_scalar.cpp",
        avx2=ROOT / "python/freetoken/moe/q4_k_avx2.cpp",
    )
    mixed = _compile_library(
        cxx=cxx,
        output_dir=output_dir,
        stem="mixed_gemv_native",
        dispatch=ROOT / "python/freetoken/moe/mixed_gemv_native.cpp",
        scalar=ROOT / "python/freetoken/moe/mixed_gemv_scalar.cpp",
        avx2=ROOT / "python/freetoken/moe/mixed_gemv_avx2.cpp",
    )
    try:
        compiler_version = subprocess.check_output(
            [cxx, "--version"], text=True, stderr=subprocess.STDOUT
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError) as error:
        raise RuntimeError(f"unable to read compiler version from {cxx}: {error}") from error
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"unable to determine git commit: {error}") from error
    return {
        "schema_name": "qwen38-target-cpu-native-build",
        "schema_version": 1,
        "commit": commit,
        "command": command,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "compiler": {"command": cxx, "version": compiler_version},
        "compile_flags": {
            "common": list(COMMON_FLAGS),
            "baseline": list(BASELINE_FLAGS),
            "avx2": list(AVX2_FLAGS),
        },
        "libraries": {"q4_k": q4, "mixed_gemv": mixed},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++", help="C++ compiler (default: g++)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".cache/freetoken/target-cpu-native",
    )
    parser.add_argument(
        "--output", type=Path, help="build manifest path (default: output-dir/build.json)"
    )
    args = parser.parse_args(argv)
    command = shlex.join([sys.argv[0], *(argv if argv is not None else sys.argv[1:])])
    try:
        manifest = build(cxx=args.cxx, output_dir=args.output_dir, command=command)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"ERROR: native helper build failed: {error}\n")
    output = args.output or args.output_dir / "build.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
