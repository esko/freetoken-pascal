#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_dir="${1:-${repo_root}/build/toolchain}"
cd "${repo_root}"
mkdir -p "${artifact_dir}"

python "${repo_root}/scripts/check_toolchain.py"

python - <<'PY'
import json
import re
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
manifest = json.loads((root / "manifests/toolchain.json").read_text(encoding="utf-8"))

if not sys.version.startswith("3.12."):
    raise SystemExit(f"Python 3.12 required, got {sys.version}")

nvcc = subprocess.run(["nvcc", "--version"], check=True, capture_output=True, text=True)
if "release 12.6" not in nvcc.stdout:
    raise SystemExit(f"CUDA 12.6 nvcc required:\n{nvcc.stdout}")

gcc = subprocess.run(["gcc", "-dumpfullversion"], check=True, capture_output=True, text=True)
gcc_major = int(gcc.stdout.strip().split(".")[0])
if gcc_major > manifest["maximum_host_gcc_major"]:
    raise SystemExit(f"GCC {gcc_major} is unsupported by this CUDA 12.6 contract")

import torch
import triton

if torch.__version__ != manifest["torch"]:
    raise SystemExit(f"Torch {manifest['torch']} required, got {torch.__version__}")
if torch.version.cuda != "12.6":
    raise SystemExit(f"Torch CUDA 12.6 required, got {torch.version.cuda}")
if triton.__version__ != manifest["triton"]:
    raise SystemExit(f"Triton {manifest['triton']} required, got {triton.__version__}")

arch_flags = torch._C._cuda_getArchFlags()
available = set(arch_flags.split()) if arch_flags else set()
allowed = set(manifest["torch_pascal_architectures"])
if not available.intersection(allowed):
    raise SystemExit(f"Torch wheel has no Pascal cubins: {sorted(available)}")
PY

nvcc \
    --cubin \
    --generate-code arch=compute_61,code=sm_61 \
    "${repo_root}/tools/toolchain/sm61_probe.cu" \
    --output-file "${artifact_dir}/sm61_probe.cubin"

cuobjdump --list-elf "${artifact_dir}/sm61_probe.cubin" > "${artifact_dir}/sm61_probe.elf.txt"
if ! grep -q 'sm_61' "${artifact_dir}/sm61_probe.elf.txt"; then
    echo "ERROR: compiled probe does not contain sm_61 code" >&2
    cat "${artifact_dir}/sm61_probe.elf.txt" >&2
    exit 1
fi

python "${repo_root}/scripts/write_toolchain_inventory.py" \
    --output "${artifact_dir}/inventory.json"
cp "${repo_root}/manifests/toolchain.json" "${artifact_dir}/toolchain.json"
cat "${artifact_dir}/inventory.json"
