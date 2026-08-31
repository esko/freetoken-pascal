#!/usr/bin/env bash
set -euo pipefail

level="${FREETOKEN_PASCAL_TEST_LEVEL:-smoke}"
image="${FREETOKEN_PASCAL_CUDA_IMAGE:-freetoken-pascal:cuda126}"
smoke_gpu="${FREETOKEN_SMOKE_GPU:-0}"
host_python="${FREETOKEN_HOST_PYTHON:-python3}"
repo_root="$(git rev-parse --show-toplevel)"
container_root=/workspace/freetoken-pascal
mkdir -p results/hardware

docker image inspect "$image" >/dev/null
software_probe="$({
  docker run --rm -i --gpus all "$image" python - <<'PY'
import json
import torch
import triton

print(json.dumps({
    "cuda_runtime": torch.version.cuda,
    "torch": torch.__version__,
    "device_count": torch.cuda.device_count(),
    "triton": triton.__version__,
}))
PY
})"
readarray -t software_fields < <("$host_python" -c '
import json, sys
data = json.loads(sys.argv[1])
print(data["cuda_runtime"])
print(data["torch"])
print(data["device_count"])
print(data["triton"])
' "$software_probe")

# PCI/NUMA/NVMe capture runs on the host; CUDA runtime identity is measured in
# the exact container used by the bounded device tests.
"$host_python" scripts/capture_hardware_inventory.py \
  results/hardware/inventory.json \
  --cuda-runtime "${software_fields[0]}" \
  --torch-version "${software_fields[1]}" \
  --torch-device-count "${software_fields[2]}" \
  --triton-version "${software_fields[3]}"

minimum_gpus=1
case "$level" in
  dual-p4|release) minimum_gpus=2 ;;
esac
docker run --rm \
  -v "$repo_root:$container_root" \
  -w "$container_root" \
  "$image" \
  python scripts/check_hardware_inventory.py \
    results/hardware/inventory.json \
    --minimum-gpus "$minimum_gpus"

run_single_smoke() {
  docker run --rm --gpus "device=$smoke_gpu" \
    -e FREETOKEN_SM61_RUNNER_VERIFIED=1 \
    -v "$repo_root:$container_root" \
    -w "$container_root" \
    "$image" \
    bash -lc 'PYTHONPATH=python pytest -q tests/project/test_hardware_smoke.py -m "sm61 and not dual_p4"'
}

run_dual_smoke() {
  docker run --rm --gpus all \
    -e FREETOKEN_SM61_RUNNER_VERIFIED=1 \
    -e FREETOKEN_DUAL_P4_RUNNER_VERIFIED=1 \
    -v "$repo_root:$container_root" \
    -w "$container_root" \
    "$image" \
    bash -lc 'PYTHONPATH=python pytest -q tests/project/test_hardware_smoke.py -m dual_p4'
}

case "$level" in
  smoke|single-p4)
    run_single_smoke
    ;;
  dual-p4)
    run_single_smoke
    run_dual_smoke
    ;;
  release)
    test -n "${FREETOKEN_PASCAL_MODEL_PATH:-}" || {
      echo "release level requires FREETOKEN_PASCAL_MODEL_PATH" >&2
      exit 1
    }
    echo "release benchmark runner is not implemented" >&2
    exit 1
    ;;
  *)
    echo "unknown hardware test level: $level" >&2
    exit 2
    ;;
esac
