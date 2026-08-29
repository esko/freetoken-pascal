# Implementation plan

## Delivery strategy

The project is implemented as a sequence of gated vertical slices. Hardware-independent work comes first. Each phase ends with a usable fallback and a decision gate; no phase assumes the next one succeeds. Optional performance profiles are isolated from the core release.

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
- implement the dedicated, separately sharded PLE file as a core NVMe-backed path;
- provide mmap and positional-read PLE backends with random-access advice;
- provide direct and adaptive vectorized dedupe/order/coalesce lookup paths, bounded asynchronous prefetch and physical read-amplification telemetry;
- define an explicit PLE row-codec boundary with IQ4_NL as the initial reference;
- produce a cache-disabled, correctness-first short-context path.

### Exit gate

A tiny Qwen4 model passes CPU/reference tests. The dedicated PLE artifact is independently checksummed and both backends return identical rows and failure behavior. When a P4 arrives, one P4 generates deterministic short text with cache disabled inside a safe placement profile.

## Phase 2 — AVX2 host expert backend and model profiles

### Outcomes

- model-agnostic expert execution ABI;
- AVX2 Q4_K implementation first;
- additional Q2/Q3/IQ types selected from real model censuses;
- pinned `reference-q4` and named `throughput-q3` whole-model identities;
- gate/up activation/down fused at the right boundary;
- bounded pinned-memory and NUMA-aware host bank;
- keep the complete quantized expert bank available through a measured no-swap DDR4 serving representation without accidentally retaining a second complete anonymous copy alongside uncontrolled page cache;
- SSD expert reads are startup backing or an explicitly gated experiment, not the serving design;
- parity, quality and microbenchmark suite.

The ABI slice precedes AVX2 kernels. It defines immutable heterogeneous expert-bank descriptors, prepare/execute/group/cancel/telemetry contracts, caller-owned partial accumulation, bounded workspace and explicit decoder/thread-pool/NUMA hooks. Its microbenchmark interface records raw repeated timings for the supplied production geometry and every requested miss width from 1 through top-k; it does not by itself constitute a performance claim. The Issue #16 threaded route adapter is opt-in, native-only, and census-gated per layer; it keeps serial execution for scalar, unsupported, and mixed-reference configurations.

The standalone Qwen GGUF CPU bridge is decode-only and owns its mapped host weights for the life of its heterogeneous CPU layout and Q4 executor. The CUDA engine rejects this GGUF combination until the production layer ABI can consume per-projection mappings without a homogeneous GPU cache; this is an integration blocker, not a hardware-performance claim.

The H0 `QwenGGUFCpuMoELayer` is an explicit CPU-only adapter around that bundle. It supports routed decode with the existing full-softmax Torch reference when CPU router logits are supplied, plus precomputed routes for direct parity tests. Its Qwen default preserves the model's unrenormalized selected probabilities; callers may opt into the existing renormalized mode explicitly. Calls require phase `decode` and group size one. It does not attach to Qwen model construction or the CUDA Engine, and it does not provide prefill, grouped, CUDA, TP>1 or performance evidence.

The H0 model-graph bridge adds explicit `Qwen4ExpModel.attach_gguf_cpu_expert_bundle()` and matching detach methods, with a `ForCausalLM` delegate. Attachment is opt-in, validates every layer against the shared bundle before mutation, and preserves the exact resident expert state-dict surface. The bundle remains caller-owned and is never closed by the model. This construction and lifetime foundation does not make the CUDA-oriented trunk, router, shared expert, or LM head CPU-runnable, and it does not remove the Engine guard.

The next standalone bridge is `GGUFCpuEagerBridge` in `moe/gguf_transfer.py`. It is an explicit experimental H0/H1 wrapper around the CPU layer, with required `phase="decode"`, `group_size=1`, TP1 and `cache_size=0`. It rejects prefill, grouped requests, graph capture and caller workspaces before transfer. CPU inputs use the adapter directly; non-CPU inputs use an injected blocking transfer seam for hidden states and router logits/prepared routes, invoke the adapter once, and copy the independent routed result back to the original device and dtype. It makes no stream, pinned-memory, overlap or performance claim. Engine, CLI and default paths remain unchanged. Real CUDA transfer behavior is H2-unverified.

### Exit gate

For each shipping quant and shape, CPU expert output passes error tolerances against dequantize-plus-reference matmul. End-to-end cache-zero output remains correct. Q4 and the named Q3 profile have complete identities/censuses and the fixed routing/tool/long-context/coding quality corpus is ready.

## Phase 3 — placement safety and Pascal GPU expert cache

### Outcomes

- per-GPU placement planner covering resident tensors, shared experts, recurrent/QSA/KV state, CUDA context, workspaces, transfer buffers and expert slots;
- startup canary, explicit headroom, automatic cache/placement backoff and fail-readiness behavior under issue #73;
- Pascal DP4A low-bit GPU kernels and format-specific tuning, ahead of generic FP16/BF16/FP8 optimization;
- fused Qwen3.8 `topk=10` router with a permanent Torch reference path;
- fixed-address per-GPU expert slot pools;
- trace-derived static-hot profile as a mandatory cache comparator and warm-start source;
- measured comparison of layer-owned, disjoint expert-owned, replicated and trunk split/TP policies;
- static mixed CPU/GPU output merge;
- async LFRU fill and persisted heat;
- routing telemetry and offline cache simulator.

### Exit gate

Single- and dual-P4 static-cache runs are correct and pass the placement canary with documented reserve. The static-hot profile has measured hit/throughput evidence. Async future-token fill cannot delay the current token and improves a locality-positive trace relative to the appropriate static control or remains disabled.

## Phase 4 — adaptive hybrid execution

### Outcomes

- current-step miss partition;
- concurrent CPU and H2D/GPU execution;
- measured `q*` tables under contended memory/PCIe conditions;
- adaptive policy with pure-CPU and fetch-all alternatives;
- NUMA and per-rank worker tuning;
- decode prefetch and copy batching;
- safe interaction with the #73 reserve and ownership policy.

### Exit gate

The scheduler never chooses an unsupported/unsafe path, exposes its decisions, and beats or safely falls back to the best pure/static policy on representative decode workloads.

## Phase 5 — prefill, long context, serving and optional coding profile

### Core outcomes

- prefill expert grouping, chunked streaming and double buffering;
- GDN/QSA/PLE state validation through long contexts;
- semantic state checkpoint/restore;
- hardened OpenAI-compatible serving;
- Docker/Compose operations, health and metrics.

### Optional outcome

Issue #74 may add exact context-derived n-gram speculation after ordinary state and serving semantics are stable. It uses the same target model for verification, must preserve deterministic output, reports its PLE I/O and negative controls, and automatically disables on low-acceptance workloads. It cannot block or replace the core phase exit.

### Exit gate

32K and 128K coding sessions run correctly with streaming, cancellation, restore and restart. Prefill does not thrash an undersized token-oriented cache. Optional #74 is either independently qualified or omitted from core release claims.

## Phase 6 — hardware qualification and release

### Outcomes

- actual P4/NUMA topology profile;
- one- and two-P4 qualification;
- #73 placement-cliff sweep and release safety margin;
- whole-model `reference-q4` and `throughput-q3` comparison;
- component-level Q5/Q8 and shipping CPU/GPU format comparison;
- benchmark comparison with merged llama.cpp/PXQ;
- independently measured cold-cache, warm-cache, major-page-fault, physical-read/amplification and steady-state PLE behavior for both storage backends;
- cache-zero, static-hot, static, dynamic and current-step-hybrid controls;
- soak and fault injection;
- v1 release artifact, image, docs and known limitations;
- optional `coding-ngram` profile only when its separate evidence passes.

### Exit gate

Every required core checkbox in `release-criteria.md` has linked evidence. Optional-profile failure does not fail the core release; the profile is omitted or disabled.

## Critical path

```text
upstream import
  → Pascal compile
  → Qwen4 + GGUF + dedicated NVMe PLE reference
  → random-advised mmap/pread + adaptive PLE I/O/prefetch
  → AVX2 expert backend + named Q4/Q3 profiles
  → serving-ready host-expert integration
  → placement planner/canary/backoff
  → Pascal DP4A expert kernels
  → fused topk=10 Pascal router
  → static-hot cache and mixed merge
  → dual-P4 policy comparison
  → async cache
  → current-step q*
  → prefill/long context
  → serving/package
  → release qualification

optional after stable state/serving:
  → exact context-derived n-gram profile
```

## Parallel work

Before P4 arrival, independent workers can handle:

- source import/provenance;
- build container and H1 CI;
- Qwen4/GGUF loader integration;
- dedicated PLE file format, random advice, mmap/pread backends, adaptive planner, read-amplification metrics and asynchronous prefetch;
- CPU expert ABI/kernels;
- Q4/Q3 identity, census, quant conversion and mixed-precision correctness A/B tests;
- serving-ready host expert integration;
- placement planner/canary/backoff logic and result schemas;
- fused router reference/dispatch and `sm_61` compile work;
- cache trace/static-hot simulator;
- optional n-gram proposal/state fixtures, kept independent from core dependencies;
- server/config/metrics contracts;
- documentation and tests.

Avoid concurrent edits to the same model loader, quant registry or cache core. The orchestrator assigns ownership and serializes those merges.

## Stop/reassess conditions

Create or amend an ADR before continuing if:

- FreeToken upstream merges equivalent work with a conflicting design;
- neither Q4 nor the named Q3 profile fits the host operating envelope with safe headroom;
- P4 lacks sufficient VRAM for the required always-active trunk and state after placement backoff;
- placement canary evidence shows unavoidable spill/fallback cliffs below a useful configuration;
- the dedicated PLE shard cannot provide stable random-I/O behavior without unrelated-weight coupling or unacceptable read amplification;
- AVX2 CPU experts are so slow that current-step CPU work cannot help;
- no realistic cache size beats the static-hot/control path;
- every two-P4 policy loses to one P4 plus CPU because of communication/merge cost;
- full-QSA implementation is not correct on Pascal;
- a vLLM-based implementation becomes demonstrably smaller and faster to complete.

A failed optimization does not invalidate the project. Preserve the last correct fallback and update the plan from evidence.
