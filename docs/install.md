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

## Method 1: Install an audited Pascal wheel bundle

```bash
FREETOKEN_WHEEL=./freetoken_pascal-<version>-cp312-...whl \
FREETOKEN_KERNEL_CACHE_WHEEL=./freetoken_kernel_cache-<version>+cu126-py3-none-any.whl \
bash install.sh --yes
```

The default installer profile is `pascal`. It accepts only local wheels with matching embedded build
metadata proving CUDA 12.6 and `sm_61`, and it performs that audit before bootstrapping `uv`, creating
a venv, or modifying the install directories. It installs the core runtime through the CUDA 12.6
Torch index without `[accel]`, FlashInfer, SGLang, or any CUDA 13 index. Until such audited wheels
are published, use the source environment below; do not substitute a generic PyPI or upstream
CUDA 13 wheel.

The inherited modern installer is available only through the explicit
`FREETOKEN_INSTALL_PROFILE=upstream-cu130` profile. It is not supported on Tesla P4 and is not a
FreeToken-Pascal release path.

## Method 2: Install from source

```bash
git clone https://github.com/esko/freetoken-pascal.git && cd freetoken-pascal
uv venv && source .venv/bin/activate
uv pip install -e .
```

Use the pinned CUDA 12.6 environment from
[development-environments.md](development-environments.md) for reproducible development and H1
builds. Do not install the `accel` extra on Pascal.

## Verify

```bash
source .venv/bin/activate
ft --version
ft serve --model ~/path/to/Qwen3.8-Flash-Next
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"hi"}]}'
```

Then head to [quickstart.md](quickstart.md).
