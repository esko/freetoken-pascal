from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "toolchain.json"
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
VERSION_RE = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?")


def _locked_version(lock_text: str, package: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(package)}==([^ \\\n]+)", lock_text)
    return match.group(1) if match else None


def validate_toolchain(data: Any, *, root: Path) -> list[str]:
    if not isinstance(data, dict):
        return ["manifest root must be an object"]

    errors: list[str] = []
    required_strings = ("platform", "python", "uv", "cuda", "torch", "triton", "cmake", "ninja")
    for field in required_strings:
        if not isinstance(data.get(field), str) or not data[field]:
            errors.append(f"{field} must be a non-empty string")

    cuda_match = VERSION_RE.match(str(data.get("cuda", "")))
    if not cuda_match or cuda_match.group(1) != "12" or cuda_match.group(2) != "6":
        errors.append("cuda must be pinned to the 12.6 release line")
    if "+cu126" not in str(data.get("torch", "")):
        errors.append("torch must be an explicit +cu126 build")

    targets = data.get("target_architectures")
    if targets != ["sm_61"]:
        errors.append("target_architectures must contain exactly sm_61")
    pascal_arches = data.get("torch_pascal_architectures")
    if not isinstance(pascal_arches, list) or not {"sm_60", "sm_61"}.intersection(pascal_arches):
        errors.append("torch_pascal_architectures must include sm_60 or sm_61")
    if data.get("maximum_host_gcc_major") != 13:
        errors.append("maximum_host_gcc_major must be 13 for CUDA 12.6")

    images = data.get("images")
    if not isinstance(images, dict):
        errors.append("images must be an object")
    else:
        for name in ("python", "cuda"):
            image = images.get(name)
            if not isinstance(image, dict):
                errors.append(f"images.{name} must be an object")
                continue
            if not isinstance(image.get("reference"), str) or ":" not in image["reference"]:
                errors.append(f"images.{name}.reference must use a versioned tag")
            if not isinstance(image.get("digest"), str) or not SHA256_RE.fullmatch(image["digest"]):
                errors.append(f"images.{name}.digest must be a sha256 digest")

    lock_versions = {
        "uv": (root / "requirements" / "cpu.lock", "uv"),
        "cuda": (root / "requirements" / "cuda126.lock", "cuda-toolkit"),
        "torch": (root / "requirements" / "cuda126.lock", "torch"),
        "triton": (root / "requirements" / "cuda126.lock", "triton"),
        "cmake": (root / "requirements" / "cuda126.lock", "cmake"),
        "ninja": (root / "requirements" / "cuda126.lock", "ninja"),
    }
    for field, (path, package) in lock_versions.items():
        try:
            actual = _locked_version(path.read_text(encoding="utf-8"), package)
        except OSError as error:
            errors.append(f"unable to read {path.relative_to(root)}: {error}")
            continue
        if actual != data.get(field):
            errors.append(f"{package} lock is {actual!r}, expected {data.get(field)!r}")

    dockerfiles = {
        "python": root / "containers" / "cpu.Dockerfile",
        "cuda": root / "containers" / "cuda126.Dockerfile",
    }
    if isinstance(images, dict):
        for name, path in dockerfiles.items():
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as error:
                errors.append(f"unable to read {path.relative_to(root)}: {error}")
                continue
            image = images.get(name)
            if isinstance(image, dict):
                expected = f"{image.get('reference')}@{image.get('digest')}"
                if expected not in text:
                    errors.append(f"{path.relative_to(root)} does not use pinned image {expected}")

    cuda_dockerfile = root / "containers" / "cuda126.Dockerfile"
    try:
        cuda_text = cuda_dockerfile.read_text(encoding="utf-8")
    except OSError:
        pass
    else:
        expected_arch_settings = (
            "TORCH_CUDA_ARCH_LIST=6.1",
            "CMAKE_CUDA_ARCHITECTURES=61",
            "CUDAARCHS=61",
        )
        for setting in expected_arch_settings:
            if setting not in cuda_text:
                errors.append(f"containers/cuda126.Dockerfile must set {setting}")

    compile_gate = root / "scripts" / "ci" / "verify_cuda126.sh"
    try:
        compile_text = compile_gate.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"unable to read {compile_gate.relative_to(root)}: {error}")
    else:
        if "arch=compute_61,code=sm_61" not in compile_text:
            errors.append("CUDA compile gate must explicitly generate sm_61 code")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Pascal toolchain contract")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        data = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"ERROR: unable to read toolchain manifest: {error}", file=sys.stderr)
        return 1
    errors = validate_toolchain(data, root=args.root)
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1
    print(
        f"validated CUDA {data['cuda']}, Torch {data['torch']}, "
        f"and targets {','.join(data['target_architectures'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
