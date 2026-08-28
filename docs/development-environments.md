# Reproducible development environments

FreeToken-Pascal has separate pinned H0 CPU and H1 CUDA build environments. Both target Linux
x86_64 and Python 3.12.11. The CUDA environment uses CUDA 12.6.3, Torch 2.11.0+cu126, and emits
shipping CUDA code explicitly for `sm_61`; CUDA 13 is unsupported.

The authoritative versions, image digests, and architecture policy are in
`manifests/toolchain.json`. Dependency inputs and hash-locked resolutions are under
`requirements/`. Do not hand-edit a lockfile.

## One-command setup

Build the hosted CPU environment:

```bash
make env-cpu
```

Build the GPU-free CUDA compile environment:

```bash
make env-cuda126
```

Open an editable shell with the checkout mounted into the matching environment:

```bash
make dev-cpu
# or
make dev-cuda126
```

No NVIDIA driver or GPU is needed to build either image or run the H1 compile gate. Validate the
CUDA compiler, Torch wheel, and generated `sm_61` cubin with:

```bash
docker run --rm freetoken-pascal:cuda126 scripts/ci/verify_cuda126.sh
```

The gate records Python, compiler, CUDA, Torch, Triton, and architecture selections in
`build/toolchain/inventory.json` and copies the authoritative manifest beside it. A mounted output
directory is required if the evidence must survive container removal; the CI workflow demonstrates
that invocation.

## Hosted tests

Run the same H0 checks CI uses:

```bash
docker run --rm freetoken-pascal:cpu \
  bash -lc 'python scripts/validate_docs.py && python scripts/check_upstream_manifest.py && \
  python scripts/check_toolchain.py && python -m compileall -q python tests scripts && \
  ruff check scripts && ruff format --check scripts && \
  PYTHONPATH=python pytest -q tests/project tests/daemon'
```

The CPU image installs `requirements/cpu.lock` with hash verification. It does not install Torch
or CUDA packages, so H0 checks cannot acquire a GPU dependency accidentally. Run the pre-commit
gate on the host, where Git is available, with `pre-commit run --all-files`.

## Regenerate locks

Install exactly uv 0.12.3, edit the corresponding `.in` file, and run:

```bash
scripts/lock_dependencies.sh
```

The script resolves for CPython 3.12 on x86_64 manylinux and refuses a different uv version. Review
the full lock diff and rerun H0/H1 before committing it.

## Cleanup

Remove the two local development image tags with:

```bash
make env-clean
```

This does not prune unrelated images, volumes, or the Docker build cache.
