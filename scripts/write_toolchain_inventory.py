from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _command_version(command: str, *args: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    result = subprocess.run(
        [executable, *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip() or None


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_inventory() -> dict[str, Any]:
    inventory: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "gcc": _command_version("gcc", "--version"),
        "nvcc": _command_version("nvcc", "--version"),
        "cmake": _package_version("cmake"),
        "ninja": _package_version("ninja"),
        "torch": _package_version("torch"),
        "triton": _package_version("triton"),
        "target_architectures": {
            "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
            "CMAKE_CUDA_ARCHITECTURES": os.environ.get("CMAKE_CUDA_ARCHITECTURES"),
            "CUDAARCHS": os.environ.get("CUDAARCHS"),
        },
    }
    try:
        import torch

        inventory["torch_cuda"] = torch.version.cuda
        arch_flags = torch._C._cuda_getArchFlags()
        inventory["torch_arch_list"] = arch_flags.split() if arch_flags else []
    except (ImportError, OSError):
        inventory["torch_cuda"] = None
        inventory["torch_arch_list"] = []
    return inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the resolved build toolchain inventory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rendered = json.dumps(collect_inventory(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
