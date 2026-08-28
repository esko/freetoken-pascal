from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sysconfig
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "manifests" / "cuda-sources.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, check: bool = True) -> tuple[int, str]:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if check and result.returncode:
        raise RuntimeError(
            f"command failed with exit {result.returncode}: {' '.join(command)}\n{output}"
        )
    return result.returncode, output


def _include_profiles() -> dict[str, list[Path]]:
    import tvm_ffi
    from torch.utils.cpp_extension import include_paths

    tvm_root = Path(tvm_ffi.__file__).resolve().parent
    return {
        "torch": [
            *(Path(path) for path in include_paths()),
            Path(sysconfig.get_paths()["include"]),
        ],
        "tvm-ffi": [
            ROOT / "python" / "freetoken" / "kernel" / "csrc" / "include",
            tvm_root / "include",
        ],
    }


def validate_manifest(data: Any) -> list[str]:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        return ["CUDA source manifest must have schema_version 1"]
    if data.get("target") != "sm_61":
        return ["CUDA source manifest target must be sm_61"]
    units = data.get("translation_units")
    if not isinstance(units, list) or not units:
        return ["translation_units must be a non-empty array"]
    errors: list[str] = []
    declared: set[str] = set()
    for index, unit in enumerate(units):
        if not isinstance(unit, dict):
            errors.append(f"translation_units[{index}] must be an object")
            continue
        path = unit.get("path")
        if not isinstance(path, str) or not path.endswith(".cu"):
            errors.append(f"translation_units[{index}].path must name a .cu file")
            continue
        if path in declared:
            errors.append(f"duplicate CUDA source {path}")
        declared.add(path)
        if not (ROOT / path).is_file():
            errors.append(f"declared CUDA source does not exist: {path}")
        if unit.get("include_profile") not in {"torch", "tvm-ffi"}:
            errors.append(f"{path}: unknown include_profile")
        if not isinstance(unit.get("expects_device_code"), bool):
            errors.append(f"{path}: expects_device_code must be boolean")

    discovered = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "python" / "freetoken" / "kernel" / "csrc").rglob("*.cu")
    }
    for path in sorted(discovered - declared):
        errors.append(f"shipping CUDA source is absent from manifest: {path}")
    for path in sorted(declared - discovered):
        errors.append(f"manifest lists a non-shipping CUDA source: {path}")
    return errors


def compile_sources(data: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    import torch

    profiles = _include_profiles()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for unit in data["translation_units"]:
        source = ROOT / unit["path"]
        object_path = output_dir / f"{source.stem}.o"
        command = [
            "nvcc",
            "--compile",
            "--std=c++20",
            "--expt-relaxed-constexpr",
            "--generate-code",
            "arch=compute_61,code=sm_61",
            "--compiler-options=-fPIC",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
            "-DTORCH_EXTENSION_NAME=freetoken_cuda_census",
            "-DTORCH_API_INCLUDE_EXTENSION_H",
        ]
        for include in profiles[unit["include_profile"]]:
            command.extend(["--include-path", str(include)])
        if unit["include_profile"] == "torch":
            command.extend(
                [
                    "--include-path",
                    str(source.parent),
                ]
            )
        command.extend([str(source), "--output-file", str(object_path)])
        _, compile_output = _run(command)
        elf_returncode, elf_output = _run(
            ["cuobjdump", "--list-elf", str(object_path)], check=False
        )
        has_sm61 = "sm_61" in elf_output
        if elf_returncode and unit["expects_device_code"]:
            raise RuntimeError(
                f"cuobjdump failed for device-bearing source {unit['path']}: {elf_output}"
            )
        if unit["expects_device_code"] and not has_sm61:
            raise RuntimeError(f"{unit['path']} compiled without sm_61 device code")
        results.append(
            {
                "path": unit["path"],
                "source_sha256": _sha256(source),
                "object": object_path.name,
                "object_sha256": _sha256(object_path),
                "expects_device_code": unit["expects_device_code"],
                "sm_61_device_code": has_sm61,
                "compile_output": compile_output,
                "cuobjdump": elf_output,
            }
        )
    return {
        "schema_version": 1,
        "target": data["target"],
        "translation_units": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile and census shipping CUDA sources")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate_manifest(data)
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    census = compile_sources(data, output_dir=args.output_dir)
    census_path = args.output_dir / "cuda-source-census.json"
    census_path.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"compiled {len(census['translation_units'])} CUDA translation units for sm_61")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
