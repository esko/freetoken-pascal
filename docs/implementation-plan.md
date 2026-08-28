# Implementation plan

## Delivery strategy

The project is implemented as a sequence of gated vertical slices. Hardware-independent work comes first. Each phase ends with a usable fallback and a decision gate; no phase assumes the next one succeeds.

## Phase 0 — downstream foundation

### Outcomes

- upstream FreeToken history is merged into this repository;
- exact source pins and license provenance exist;
- CUDA 12.6 build environment is reproducible;
- hosted CI, H1 compilation and deferred H2/H3 workflows exist;
- tiny models, fixtures, benchmark schema and result tooling exist.

### Exit gate

A clean clone passes hosted CI and compiles the intended source set for `sm_61` without requiring a GPU.

## Phase 1 — Pascal and Qwen4 reference runtime

### Outcomes

- integrate FreeToken Pascal and CUDA 12.6 work;
- integrate Qwen3.8/Qwen4 text architecture;
- integrate GGUF K/I loading;
- support heterogeneous expert-bank types needed by target artifacts;
- implement PLE mmap/offload;
- produce a cache-disabled, correctness-first short-context path.

### Exit gate

A tiny Qwen4 model passes CPU/reference tests. When the P4 arrives, a single P4 generates deterministic short text with cache disabled.

## Phase 2 — AVX2 host expert backend

### Outcomes

- model-agnostic expert execution ABI;
- AVX2 Q4_K implementation first;
- additional Q2/Q3/IQ types selected from real model census;
- gate/up activation/down fused at the right boundary;
- bounded pinned-memory and NUMA-aware host bank;
- parity and microbenchmark suite.

The ABI slice precedes AVX2 kernels. It defines immutable heterogeneous expert-bank
descriptors, prepare/execute/group/cancel/telemetry contracts, caller-owned partial
accumulation, bounded workspace and explicit decoder/thread-pool/NUMA hooks. Its
microbenchmark interface records raw repeated timings for the supplied production
geometry and every requested miss width from 1 through top-k; it does not by itself
constitute a performance claim. The Issue #16 threaded route adapter is opt-in,
native-only, and census-gated per layer; it keeps serial execution for scalar,
unsupported, and mixed-reference configurations.

The standalone Qwen GGUF CPU bridge is decode-only and owns its mapped host weights for the
life of its heterogeneous CPU layout and Q4 executor. The CUDA engine rejects this GGUF
combination until the production layer ABI can consume per-projection mappings without a
homogeneous GPU cache; this is an integration blocker, not a hardware-performance claim.

The H0 `QwenGGUFCpuMoELayer` is an explicit CPU-only adapter around that bundle. It
supports routed decode with the existing full-softmax Torch reference when CPU router
logits are supplied, plus precomputed routes for direct parity tests. Its Qwen default
preserves the model's unrenormalized selected probabilities; callers may opt into the
existing renormalized mode explicitly. Calls require phase `decode` and group size one.
It does not attach to Qwen model construction or the CUDA Engine, and it does not provide
prefill, grouped, CUDA, TP>1 or performance evidence.

### Exit gate

For each shipping quant and shape, the CPU expert output passes error tolerances against dequantize-plus-reference matmul. End-to-end cache-zero output remains correct.

## Phase 3 — Pascal GPU expert cache

### Outcomes

- PXA-derived `sm_61` low-bit GPU kernels;
- fused Qwen3.8 `topk=10` router with a permanent Torch reference path;
- Qwen4 TP=2 and fixed layer ownership;
- fixed-address per-GPU expert slot pools;
- static cache and mixed CPU/GPU output merge;
- async LFRU fill and persisted heat;
- routing telemetry and offline cache simulator.

### Exit gate

Single and dual P4 static-cache runs are correct. Async future-token fill cannot delay the current token and improves a locality-positive trace.

## Phase 4 — adaptive hybrid execution

### Outcomes

- current-step miss partition;
- concurrent CPU and H2D/GPU execution;
- measured `q*` tables under contention;
- adaptive policy with pure-CPU and fetch-all alternatives;
- NUMA and per-rank worker tuning;
- decode prefetch and copy batching.

### Exit gate

The scheduler never chooses an unsupported path, exposes its decisions, and beats or safely falls back to the best pure policy on representative decode workloads.

## Phase 5 — prefill, long context and serving

### Outcomes

- prefill expert grouping, chunked streaming and double buffering;
- GDN/QSA/PLE state validation through long contexts;
- semantic state checkpoint/restore;
- hardened OpenAI-compatible serving;
- Docker/Compose operations, health and metrics.

### Exit gate

32K and 128K coding sessions run correctly with streaming, cancellation, restore and restart. Prefill does not thrash an undersized token-oriented cache.

## Phase 6 — hardware qualification and release

### Outcomes

- actual P4/NUMA topology profile;
- one- and two-P4 qualification;
- benchmark comparison with llama.cpp/PXQ;
- warm/cold PLE and cache measurements;
- soak and fault injection;
- v1 release artifact, image, docs and known limitations.

### Exit gate

Every required checkbox in `release-criteria.md` has linked evidence.

## Critical path

```text
upstream import
  → Pascal compile
  → Qwen4 + GGUF + PLE reference
  → AVX2 expert backend
  → P4 expert kernels
  → fused topk=10 Pascal router
  → TP2 ownership
  → static mixed execution
  → async cache
  → current-step q*
  → prefill/long context
  → serving/package
  → release qualification
```

## Parallel work

Before P4 arrival, independent workers can handle:

- source import/provenance;
- build container and H1 CI;
- Qwen4/GGUF loader integration;
- CPU expert ABI/kernels;
- fused router reference/dispatch and `sm_61` compile work after #14 H0 lands;
- cache trace simulator;
- benchmark result schema;
- server/config/metrics contracts;
- documentation and tests.

Avoid concurrent edits to the same model loader, quant registry or cache core. The orchestrator assigns ownership and serializes those merges.

## Stop/reassess conditions

Create or amend an ADR before continuing if:

- FreeToken upstream merges equivalent work with a conflicting design;
- Qwen3.8 artifact cannot fit the host operating envelope without a different quant;
- P4 lacks sufficient VRAM for the required always-active trunk;
- AVX2 CPU experts are so slow that current-step CPU work cannot help;
- no realistic cache size produces useful routing locality;
- TP=2 communication costs exceed one-P4 plus CPU performance;
- the full-QSA implementation is not correct on Pascal;
- a vLLM-based implementation becomes demonstrably smaller and faster to complete.

A failed optimization does not invalidate the project. Preserve the last correct fallback and update the plan from evidence.
