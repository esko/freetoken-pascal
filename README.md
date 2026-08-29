# FreeToken-Pascal

[![CI](https://github.com/esko/freetoken-pascal/actions/workflows/ci.yml/badge.svg)](https://github.com/esko/freetoken-pascal/actions/workflows/ci.yml)

FreeToken-Pascal is a downstream engineering project to run large hybrid MoE models efficiently on NVIDIA Pascal GPUs, with the first complete target being **Qwen3.8-Flash-Next on a dual Tesla P4 (`sm_61`) server**.

The project is based on [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) and retains its source history. It is independently maintained, is not endorsed by FlashML, and keeps the upstream `freetoken` Python import package and `ft` command while publishing downstream metadata as FreeToken-Pascal.

## Status

The complete upstream runtime source tree was imported at the commit pinned in `manifests/upstreams.yaml`. Substantial H0/H1 work is already present for CUDA 12.6/`sm_61` compilation, the Qwen3.8 text architecture, heterogeneous GGUF ingestion, file-backed PLE/expert layouts, AVX2 low-bit CPU experts, topology/affinity controls, no-swap admission, and correctness-oriented Qwen GGUF bridges.

The issue #77 integration branch synchronizes exact upstream commit `58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6`, including merged Qwen3.8 PR #257. The upstream modular model, QSA, PLE, router, and state contracts are now authoritative; closed PR #232 remains historical provenance only for explicitly inventoried downstream GGUF, Pascal, CPU-expert, file-backed PLE, and reference-test deltas.

The remaining critical path is the dedicated PLE serving artifact, serving-ready host-expert integration, Pascal GPU kernels, post-prefill placement safety, QSA long-context workspace/synchronization work, one-/two-P4 ownership and cache policies, adaptive CPU/GPU execution, long-context serving, and hardware-qualified release evidence. See the [2026-08-29 status and evidence review](docs/reviews/2026-08-29-status-and-evidence-review.md).

The target P4 GPUs have not arrived yet. Hardware qualification issues [#9](https://github.com/esko/freetoken-pascal/issues/9) and [#29](https://github.com/esko/freetoken-pascal/issues/29) are explicitly blocked, while upstream reconciliation, hosted, CPU, converter, storage, cache-simulation, tiny-model and `sm_61` compile work proceeds.

## Product target

The v1 product is a text-serving runtime with:

- Qwen3.8-Flash-Next text inference;
- CUDA 12.6 and NVIDIA Pascal `sm_61` support;
- one- and two-P4 operation, with dual P4 as the release target;
- a dedicated NVMe-backed PLE/N-gram serving artifact with measured random-I/O behavior;
- a complete low-bit GGUF/K/I expert bank in DDR4;
- AVX2 Xeon expert execution;
- Pascal low-bit/DP4A expert kernels and a fused exact `topk=10` router;
- per-GPU hot-expert caches with cache-zero and static-hot controls;
- concurrent CPU and GPU expert execution with contention-aware scheduling;
- per-GPU placement planning through the first large prefill, a startup canary, automatic backoff and fail-readiness safety;
- bounded QSA score/top-k/gather workspaces and measured long-context scaling;
- named Q4 reference and Q3 throughput-candidate profiles;
- long-context GDN/QSA/PLE correctness;
- OpenAI-compatible serving, telemetry, reproducible benchmarks, and operational packaging;
- an optional exact context-derived n-gram coding profile that cannot block the core release.

## Source checkout

Clone the downstream repository with its complete history:

```bash
git clone https://github.com/esko/freetoken-pascal.git
cd freetoken-pascal
git remote add upstream https://github.com/FlashML-org/FreeToken.git
git fetch upstream
```

The imported upstream currently targets its own CUDA environment. The pinned CUDA 12.6/Python 3.12 downstream environment is delivered separately by issue #7; do not treat upstream's CUDA 13 defaults as the Pascal release environment.

See [the upstream installation guide](docs/install.md) for the imported runtime's prerequisites and [the downstream integration map](docs/upstream-map.md) for the exact sync procedure.

Validate or inspect pinned source provenance with:

```bash
python scripts/check_upstream_manifest.py
python scripts/report_upstream_changes.py
```

## Documentation

- [Documentation index](docs/README.md)
- [Product scope](docs/product-scope.md)
- [Architecture](docs/architecture.md)
- [Implementation plan](docs/implementation-plan.md)
- [2026-08-29 status and evidence review](docs/reviews/2026-08-29-status-and-evidence-review.md)
- [Orchestrator guide](docs/orchestrator-guide.md)
- [Live backlog map](docs/backlog.md)
- [Architecture decisions](docs/adr/README.md)
- [Testing strategy](docs/testing-strategy.md)
- [Release criteria](docs/release-criteria.md)
- [Upstream integration map](docs/upstream-map.md)

## Source projects

This work builds on and tracks:

- [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken), with merged Qwen3.8 support from PR #257 as the next downstream sync target
- [FreeToken Pascal support PR #19](https://github.com/FlashML-org/FreeToken/pull/19)
- [FreeToken CUDA 12.6 dependency PR #26](https://github.com/FlashML-org/FreeToken/pull/26)
- [FreeToken Qwen3.8/Qwen4 PR #232](https://github.com/FlashML-org/FreeToken/pull/232), retained only as historical provenance for downstream code that remains distinct after issue #77
- [FreeToken mmap PLE PR #279](https://github.com/FlashML-org/FreeToken/pull/279), tracked through an immutable pin as an open donor/reference for issue #13 rather than a merged dependency
- [FreeToken GGUF/K/I PR #131](https://github.com/FlashML-org/FreeToken/pull/131)
- [FreeToken Qwen MoE TP PR #104](https://github.com/FlashML-org/FreeToken/pull/104)
- [PXA/PXQ llama.cpp](https://github.com/poisonxa16/pxq_llama.cpp)
- [llama.cpp Qwen4-exp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742), merged into upstream llama.cpp and used as the external GGUF oracle
- [vLLM Qwen3.8 PR #53896](https://github.com/vllm-project/vllm/pull/53896)
- [vLLM PLE-offload PR #53899](https://github.com/vllm-project/vllm/pull/53899)
- [llama.cpp expert-residency experiment](https://github.com/timadinorth/llama.cpp/pull/1) as a static-hot placement reference
- [Qwen3.8 DGX Spark deployment and exact n-gram benchmark](https://github.com/sxuff/qwen38-flash-next-dgx-spark) as an external performance-test reference

Exact revisions must be recorded in `manifests/upstreams.yaml`; branch names and moving PR heads are never sufficient provenance.

## Working rules

1. Preserve a zero-cache or CPU-only fallback that remains correct.
2. Separate correctness changes from performance changes.
3. Never claim a speedup without same-model, same-quant, same-prompt A/B evidence.
4. Keep hardware-dependent work blocked until the P4s and self-hosted runner exist.
5. Make every experimental optimization independently switchable and observable.
6. Reconcile merged upstream functionality before extending duplicate downstream implementations.
7. Do not broaden the core v1 release to vision, GLM, native MTP, pruning, or general cloud serving before the release gates are met.
8. Optional performance profiles must fail or disable safely without weakening ordinary decode readiness.

## License

Apache-2.0. Upstream copyright and license notices are preserved. See `NOTICE` and `manifests/upstreams.yaml` for attribution and exact provenance.
