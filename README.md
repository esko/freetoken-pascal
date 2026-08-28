# FreeToken-Pascal

[![CI](https://github.com/esko/freetoken-pascal/actions/workflows/ci.yml/badge.svg)](https://github.com/esko/freetoken-pascal/actions/workflows/ci.yml)

FreeToken-Pascal is a downstream engineering project to run large hybrid MoE models efficiently on NVIDIA Pascal GPUs, with the first complete target being **Qwen3.8-Flash-Next on a dual Tesla P4 (`sm_61`) server**.

The project is based on [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) and retains its source history. It is independently maintained, is not endorsed by FlashML, and keeps the upstream `freetoken` Python import package and `ft` command while publishing downstream metadata as FreeToken-Pascal.

## Status

The complete upstream runtime source tree was imported at the commit pinned in `manifests/upstreams.yaml`. Pascal/CUDA 12.6 integration, Qwen3.8 support, low-bit host experts, and the remaining v1 work proceed through the dependency-ordered [epic #4](https://github.com/esko/freetoken-pascal/issues/4).

The target P4 GPUs have not arrived yet. Hardware qualification issues [#9](https://github.com/esko/freetoken-pascal/issues/9) and [#29](https://github.com/esko/freetoken-pascal/issues/29) are explicitly blocked, while hosted, CPU, converter, cache-simulation, tiny-model and `sm_61` compile work proceeds.

## Product target

The v1 product is a text-serving runtime with:

- Qwen3.8-Flash-Next text inference;
- CUDA 12.6 and NVIDIA Pascal `sm_61` support;
- one- and two-P4 operation, with dual P4 as the release target;
- low-bit GGUF/K/I expert banks in host RAM;
- PLE mmap/offload from the PCIe NVMe drive with page-cache controls;
- AVX2 Xeon expert execution;
- a per-GPU hot-expert cache;
- concurrent CPU and GPU expert execution with contention-aware scheduling;
- long-context GDN/QSA/PLE correctness;
- OpenAI-compatible serving, telemetry, reproducible benchmarks, and operational packaging.

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

## Documentation

- [Documentation index](docs/README.md)
- [Product scope](docs/product-scope.md)
- [Architecture](docs/architecture.md)
- [Implementation plan](docs/implementation-plan.md)
- [Orchestrator guide](docs/orchestrator-guide.md)
- [Live backlog map](docs/backlog.md)
- [Architecture decisions](docs/adr/README.md)
- [Testing strategy](docs/testing-strategy.md)
- [Release criteria](docs/release-criteria.md)
- [Upstream integration map](docs/upstream-map.md)

## Source projects

This work builds on and tracks:

- [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)
- [FreeToken Pascal support PR #19](https://github.com/FlashML-org/FreeToken/pull/19)
- [FreeToken CUDA 12.6 dependency PR #26](https://github.com/FlashML-org/FreeToken/pull/26)
- [FreeToken Qwen3.8/Qwen4 PR #232](https://github.com/FlashML-org/FreeToken/pull/232)
- [FreeToken GGUF/K/I PR #131](https://github.com/FlashML-org/FreeToken/pull/131)
- [FreeToken Qwen MoE TP PR #104](https://github.com/FlashML-org/FreeToken/pull/104)
- [PXA/PXQ llama.cpp](https://github.com/poisonxa16/pxq_llama.cpp)
- [llama.cpp Qwen4-exp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)
- [vLLM Qwen3.8 PR #53896](https://github.com/vllm-project/vllm/pull/53896)
- [vLLM PLE-offload PR #53899](https://github.com/vllm-project/vllm/pull/53899)

Exact revisions must be recorded in `manifests/upstreams.yaml`; branch names and moving PR heads are never sufficient provenance.

## Working rules

1. Preserve a zero-cache or CPU-only fallback that remains correct.
2. Separate correctness changes from performance changes.
3. Never claim a speedup without same-model, same-quant, same-prompt A/B evidence.
4. Keep hardware-dependent work blocked until the P4s and self-hosted runner exist.
5. Make every experimental optimization independently switchable and observable.
6. Do not broaden v1 to vision, GLM, MTP, or general cloud serving before the release gates are met.

## License

Apache-2.0. Upstream copyright and license notices are preserved. See `NOTICE` and `manifests/upstreams.yaml` for attribution and exact provenance.
