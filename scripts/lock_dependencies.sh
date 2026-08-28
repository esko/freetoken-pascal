#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

expected_uv="0.12.3"
actual_uv="$(uv --version | awk '{print $2}')"
if [[ "${actual_uv}" != "${expected_uv}" ]]; then
    echo "ERROR: uv ${expected_uv} required to regenerate locks; found ${actual_uv}" >&2
    exit 1
fi

uv pip compile requirements/cpu.in \
    --python-version 3.12 \
    --python-platform x86_64-manylinux_2_28 \
    --generate-hashes \
    --output-file requirements/cpu.lock
uv pip compile requirements/cuda126.in \
    --python-version 3.12 \
    --python-platform x86_64-manylinux_2_28 \
    --index-strategy unsafe-best-match \
    --generate-hashes \
    --output-file requirements/cuda126.lock
