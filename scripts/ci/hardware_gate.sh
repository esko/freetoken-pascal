#!/usr/bin/env bash
set -euo pipefail

level="${FREETOKEN_PASCAL_TEST_LEVEL:-smoke}"
image="${FREETOKEN_PASCAL_CUDA_IMAGE:-freetoken-pascal:cuda126}"
smoke_gpu="${FREETOKEN_SMOKE_GPU:-0}"
host_python="${FREETOKEN_HOST_PYTHON:-python3}"
repo_root="$(git rev-parse --show-toplevel)"
repository_commit="$(git rev-parse HEAD)"
container_root=/workspace/freetoken-pascal
mkdir -p results/hardware

profile_id="${FREETOKEN_PASCAL_PROFILE_ID:-}"
inventory_profile_args=()
if [[ -n "$profile_id" ]]; then
  case "$profile_id" in
    ecc-on|ecc-off)
      inventory_profile_args+=(--profile-id "$profile_id")
      ;;
    *)
      echo "FREETOKEN_PASCAL_PROFILE_ID must be ecc-on or ecc-off" >&2
      exit 2
      ;;
  esac
fi
inventory_path=results/hardware/inventory.json
case "$level" in
  warm-p4|dual-p4-short|router-p4|qsa-p4)
    if [[ -n "$profile_id" ]]; then
      inventory_path="results/hardware/inventory-${profile_id}-${level}.json"
    fi
    ;;
esac

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
if not isinstance(data, dict):
    raise SystemExit("software probe did not return an object")
print(data["cuda_runtime"])
print(data["torch"])
print(data["device_count"])
print(data["triton"])
' "$(printf '%s\n' "$software_probe" | tail -n 1)")

# PCI/NUMA/NVMe capture runs on the host; CUDA runtime identity is measured in
# the exact container used by the bounded device tests.
"$host_python" scripts/capture_hardware_inventory.py \
  "$inventory_path" \
  --cuda-runtime "${software_fields[0]}" \
  --torch-version "${software_fields[1]}" \
  --torch-device-count "${software_fields[2]}" \
  --triton-version "${software_fields[3]}" \
  "${inventory_profile_args[@]}"

minimum_gpus=1
case "$level" in
  dual-p4|dual-p4-short|release) minimum_gpus=2 ;;
esac
inventory_check_profile_args=()
if [[ -n "$profile_id" ]]; then
  inventory_check_profile_args+=(--expected-profile "$profile_id")
fi
docker run --rm --gpus all \
  -v "$repo_root:$container_root" \
  -w "$container_root" \
  "$image" \
  python scripts/check_hardware_inventory.py \
    "$inventory_path" \
    --minimum-gpus "$minimum_gpus" \
    "${inventory_check_profile_args[@]}"

run_single_smoke() {
  docker run --rm --gpus "device=$smoke_gpu" \
    -e FREETOKEN_SM61_RUNNER_VERIFIED=1 \
    -v "$repo_root:$container_root" \
    -w "$container_root" \
    "$image" \
    bash -lc 'PYTHONPATH=python pytest -q tests/project/test_hardware_smoke.py -m "sm61 and not dual_p4"'
}

run_single_h2() {
  if [[ -z "${FREETOKEN_PASCAL_MODEL_PATH:-}" ]]; then
    echo "${level} level requires FREETOKEN_PASCAL_MODEL_PATH for the Qwen H2 path" >&2
    return 1
  fi
  if [[ ! -f "$FREETOKEN_PASCAL_MODEL_PATH" ]]; then
    echo "FREETOKEN_PASCAL_MODEL_PATH must name the first pinned GGUF shard" >&2
    return 1
  fi
  if [[ -z "${FREETOKEN_PASCAL_PLE_ARTIFACT:-}" ]]; then
    echo "${level} level requires FREETOKEN_PASCAL_PLE_ARTIFACT for the Qwen H2 path" >&2
    return 1
  fi
  if [[ ! -d "$FREETOKEN_PASCAL_PLE_ARTIFACT" ]]; then
    echo "FREETOKEN_PASCAL_PLE_ARTIFACT must name the dedicated PLE artifact directory" >&2
    return 1
  fi

  docker run --rm --gpus "device=$smoke_gpu" \
    -e FREETOKEN_SM61_RUNNER_VERIFIED=1 \
    -e FREETOKEN_DISABLE_KERNEL_CACHE=1 \
    -v "$repo_root:$container_root" \
    -w "$container_root" \
    "$image" \
    bash -lc 'PYTHONPATH=python pytest -q \
      tests/project/test_hardware_smoke.py \
      tests/project/test_gdn_pascal_hardware.py \
      -m "sm61 and not dual_p4"'

  # Engine startup intentionally requires CUDA to be uninitialized.  Run the real-model test
  # in a fresh container after the ordinary smoke tests have completed their CUDA allocation.
  local model_dir
  model_dir="$(dirname -- "$FREETOKEN_PASCAL_MODEL_PATH")"
  local -a qwen_env=(
    -e FREETOKEN_SM61_RUNNER_VERIFIED=1
    -e FREETOKEN_DISABLE_KERNEL_CACHE=1
    -e "FREETOKEN_PASCAL_MODEL_PATH=${FREETOKEN_PASCAL_MODEL_PATH}"
    -e "FREETOKEN_PASCAL_PLE_BACKEND=${FREETOKEN_PASCAL_PLE_BACKEND:-mmap}"
  )
  local -a qwen_mounts=(
    # Shard discovery needs every sibling in the pinned split, not only shard one.
    -v "$model_dir:$model_dir:ro"
  )
  qwen_env+=( -e "FREETOKEN_PASCAL_PLE_ARTIFACT=${FREETOKEN_PASCAL_PLE_ARTIFACT}" )
  qwen_mounts+=( -v "$FREETOKEN_PASCAL_PLE_ARTIFACT:$FREETOKEN_PASCAL_PLE_ARTIFACT:ro" )
  docker run --rm --gpus "device=$smoke_gpu" \
    "${qwen_env[@]}" \
    -v "$repo_root:$container_root" \
    "${qwen_mounts[@]}" \
    -w "$container_root" \
    "$image" \
    bash -lc 'PYTHONPATH=python pytest -q -s \
      tests/project/test_qwen38_gguf_cache_zero_hardware.py \
      -m "sm61 and not dual_p4" && \
      python scripts/validate_evidence.py \
      results/hardware/qwen38-gguf-cache-zero-h2.json'
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

run_dual_short() {
  if [[ -z "$profile_id" ]]; then
    echo "dual-p4-short requires FREETOKEN_PASCAL_PROFILE_ID (ecc-on or ecc-off)" >&2
    return 1
  fi
  docker run --rm --gpus all \
    -v "$repo_root:$container_root" \
    -w "$container_root" \
    "$image" \
    python scripts/run_dual_p4_short.py \
      --inventory "$inventory_path" \
      --output results/hardware/qwen38-dual-p4-device.json \
      --repository-commit "$repository_commit" \
      --expected-profile "$profile_id"
}

run_warm_p4() {
  if [[ -z "$profile_id" ]]; then
    echo "warm-p4 requires FREETOKEN_PASCAL_PROFILE_ID (ecc-on or ecc-off)" >&2
    return 1
  fi
  if [[ -z "${FREETOKEN_PASCAL_MODEL_PATH:-}" || ! -f "$FREETOKEN_PASCAL_MODEL_PATH" ]]; then
    echo "warm-p4 requires FREETOKEN_PASCAL_MODEL_PATH naming the first pinned shard" >&2
    return 1
  fi
  if [[ -z "${FREETOKEN_PASCAL_PLE_ARTIFACT:-}" || ! -d "$FREETOKEN_PASCAL_PLE_ARTIFACT" ]]; then
    echo "warm-p4 requires FREETOKEN_PASCAL_PLE_ARTIFACT" >&2
    return 1
  fi
  local full_h2="${FREETOKEN_PASCAL_FULL_H2_EVIDENCE:-results/hardware/qwen38-gguf-cache-zero-h2.json}"
  if [[ ! -f "$full_h2" ]]; then
    echo "warm-p4 requires canonical full-H2 evidence at $full_h2" >&2
    return 1
  fi
  local model_dir
  model_dir="$(dirname -- "$FREETOKEN_PASCAL_MODEL_PATH")"
  # The outer bound includes CPU/NVMe shard hashing. The producer arms an
  # independent 300-second process watchdog immediately before Engine/GPU
  # acquisition and leaves it armed through shutdown.
  docker run --rm --gpus "device=$smoke_gpu" \
    -e FREETOKEN_SM61_RUNNER_VERIFIED=1 \
    -e FREETOKEN_DISABLE_KERNEL_CACHE=1 \
    -v "$repo_root:$container_root" \
    -v "$model_dir:$model_dir:ro" \
    -v "$FREETOKEN_PASCAL_PLE_ARTIFACT:$FREETOKEN_PASCAL_PLE_ARTIFACT:ro" \
    -w "$container_root" \
    "$image" \
    env PYTHONPATH=python timeout --foreground --signal=TERM --kill-after=5s 900s \
      python scripts/run_qwen38_warm_h2.py \
      --full-h2 "$full_h2" \
      --inventory "$inventory_path" \
      --model "$FREETOKEN_PASCAL_MODEL_PATH" \
      --ple-artifact "$FREETOKEN_PASCAL_PLE_ARTIFACT" \
      --ple-backend "${FREETOKEN_PASCAL_PLE_BACKEND:-pread}" \
      --expected-profile "$profile_id" \
      --cpu-threads "${FREETOKEN_PASCAL_CPU_THREADS:-8}" \
      --repository-commit "$repository_commit" \
      --output results/hardware/qwen38-gguf-cache-zero-warm-h2.json
}

run_router_p4() {
  if [[ -z "$profile_id" ]]; then
    echo "router-p4 requires FREETOKEN_PASCAL_PROFILE_ID (ecc-on or ecc-off)" >&2
    return 1
  fi
  if [[ "$smoke_gpu" != "0" ]]; then
    echo "router-p4 currently requires FREETOKEN_SMOKE_GPU=0 for exact host/container identity" >&2
    return 1
  fi
  docker run --rm --gpus "device=$smoke_gpu" \
    -v "$repo_root:$container_root" \
    -w "$container_root" \
    "$image" \
    env PYTHONPATH=python python -m pytest -q tests/moe/test_fused_moe.py -k fused_topk
  docker run --rm --gpus "device=$smoke_gpu" \
    -v "$repo_root:$container_root" \
    -w "$container_root" \
    "$image" \
    env PYTHONPATH=python timeout --foreground --signal=TERM --kill-after=5s 300s \
      python scripts/run_router_pascal_h2.py \
      --inventory "$inventory_path" \
      --output "results/hardware/qwen38-router-h2-${profile_id}.json" \
      --gpu-index "$smoke_gpu" \
      --expected-profile "$profile_id" \
      --repository-commit "$repository_commit"
}

run_qsa_p4() {
  if [[ -z "$profile_id" ]]; then
    echo "qsa-p4 requires FREETOKEN_PASCAL_PROFILE_ID (ecc-on or ecc-off)" >&2
    return 1
  fi
  if [[ "$smoke_gpu" != "0" ]]; then
    echo "qsa-p4 currently requires FREETOKEN_SMOKE_GPU=0 for exact host/container identity" >&2
    return 1
  fi
  docker run --rm --gpus "device=$smoke_gpu" \
    -v "$repo_root:$container_root" \
    -w "$container_root" \
    "$image" \
    env PYTHONPATH=python python -m pytest -q tests/project/test_qsa_pascal_h2.py
  docker run --rm --gpus "device=$smoke_gpu" \
    -v "$repo_root:$container_root" \
    -w "$container_root" \
    "$image" \
    env PYTHONPATH=python timeout --foreground --signal=TERM --kill-after=5s 300s \
      python scripts/run_qsa_pascal_h2.py \
      --inventory "$inventory_path" \
      --output "results/hardware/qwen38-qsa-h2-${profile_id}.json" \
      --gpu-index "$smoke_gpu" \
      --expected-profile "$profile_id" \
      --repository-commit "$repository_commit"
}

case "$level" in
  smoke)
    run_single_smoke
    ;;
  single-p4)
    run_single_h2
    ;;
  dual-p4)
    run_single_h2
    run_dual_smoke
    ;;
  dual-p4-short)
    run_dual_short
    ;;
  warm-p4)
    run_warm_p4
    ;;
  router-p4)
    run_router_p4
    ;;
  qsa-p4)
    run_qsa_p4
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
