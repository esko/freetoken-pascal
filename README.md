# FreeToken-Pascal

FreeToken-Pascal is a downstream engineering project to run large hybrid MoE models efficiently on NVIDIA Pascal GPUs, with the first complete target being **Qwen3.8-Flash-Next on a dual Tesla P4 (`sm_61`) server**.

The project keeps FreeToken's dynamic expert paging and CPU/GPU hybrid execution model, adds a reproducible CUDA 12.6 Pascal toolchain, low-bit GGUF expert banks, AVX2 CPU expert kernels, Pascal-focused GPU kernels, dual-GPU ownership, and production-grade validation.

## Status

Planning and repository bootstrap. The target P4 GPUs have not arrived yet, so all hosted, CPU-only, static-analysis, converter, cache-simulation, and tiny-model work must be completed before hardware-gated tasks begin.

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

See [Product scope](docs/product-scope.md), [Architecture](docs/architecture.md), [Implementation plan](docs/implementation-plan.md), and [Orchestrator guide](docs/orchestrator-guide.md).

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

Apache-2.0. Upstream copyright and license notices must be preserved for copied or modified source.
