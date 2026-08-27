#!/usr/bin/env bash
set -euo pipefail

level="${FREETOKEN_PASCAL_TEST_LEVEL:-smoke}"
mkdir -p results/hardware

python - <<'PY'
import json
import subprocess
from pathlib import Path

query = [
    "nvidia-smi",
    "--query-gpu=index,name,compute_cap,memory.total,pci.bus_id,driver_version",
    "--format=csv,noheader,nounits",
]
rows = subprocess.check_output(query, text=True).strip().splitlines()
gpus = []
for row in rows:
    index, name, capability, memory, bus, driver = [part.strip() for part in row.split(",")]
    gpus.append(
        {
            "index": int(index),
            "name": name,
            "compute_capability": capability,
            "memory_mib": int(memory),
            "pci_bus_id": bus,
            "driver": driver,
        }
    )
Path("results/hardware/inventory.json").write_text(json.dumps({"gpus": gpus}, indent=2) + "\n")
if not gpus:
    raise SystemExit("No NVIDIA GPU found")
if any(gpu["compute_capability"] != "6.1" for gpu in gpus):
    raise SystemExit(f"Expected only sm_61 GPUs, got {gpus}")
PY

case "$level" in
  smoke)
    pytest -m sm61 -q
    ;;
  single-p4)
    pytest -m "sm61 and not dual_p4 and not large_model and not benchmark" -q
    ;;
  dual-p4)
    test "$(nvidia-smi -L | wc -l)" -ge 2
    pytest -m "sm61 or dual_p4" -q
    ;;
  release)
    test "$(nvidia-smi -L | wc -l)" -ge 2
    pytest -m "sm61 or dual_p4 or large_model" -q
    if [[ -x scripts/run_release_benchmarks.sh ]]; then
      scripts/run_release_benchmarks.sh
    else
      echo "release benchmark runner is not implemented" >&2
      exit 1
    fi
    ;;
  *)
    echo "unknown hardware test level: $level" >&2
    exit 2
    ;;
esac
