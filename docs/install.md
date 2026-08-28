# Install

> **FreeToken-Pascal:** Tesla P4 requires the pinned CUDA 12.6 environment. CUDA 13 is rejected on
> Pascal. Build and validate the downstream environment as described in
> [development-environments.md](development-environments.md), and review the selected runtime paths
> in [pascal-compatibility.md](pascal-compatibility.md).

## Requirements

- Linux x86_64; Tesla P4 release target, with CUDA compute capability 6.0 as the minimum
- CUDA 12.6 toolkit and a driver compatible with CUDA 12.6
- Python 3.12 for the reproducible downstream environment, with
  [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

## Method 1: Install from PyPI

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
```

The generic PyPI package is not the reproducible Pascal release path. CUDA kernels are JIT-compiled
on first use and need the pinned CUDA 12.6 toolkit with `nvcc` on `PATH`.

## Method 2: Install from source

```bash
git clone https://github.com/esko/freetoken-pascal.git && cd freetoken-pascal
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

## Verify

```bash
source .venv/bin/activate
ft --version
ft serve --model ~/path/to/Qwen3.8-Flash-Next
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"hi"}]}'
```

Then head to [quickstart.md](quickstart.md).
