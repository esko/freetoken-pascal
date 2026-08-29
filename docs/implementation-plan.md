# Implementation plan

## Delivery strategy

The project is implemented as a sequence of gated vertical slices. Hardware-independent work comes first. Each phase ends with a usable fallback and a decision gate; no phase assumes the next one succeeds. Optional performance profiles are isolated from the core release.

## Phase 0 — downstream foundation and upstream reconciliation

### Outcomes

- upstream FreeToken history is merged into this repository;
- exact source pins and license provenance exist;
- CUDA 12.6 build environment is reproducible;
- hosted CI, H1 compilation and deferred H2/H3 workflows exist;
- tiny models, fixtures, benchmark schema and result tooling exist;
- issue #77 advances the downstream to exact upstream commit `58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6`, which contains merged Qwen3.8 PR #257;
- model/QSA/PLE/cache/state code adapted from closed PR #232 is compared file by file with merged upstream and retained only where downstream behavior remains materially distinct;
- open PLE mmap PR #279 is recorded through an immutable `planned` or `reference` pin before any code is mined.

### Exit gate

A clean clone passes hosted CI and compiles the reconciled source set for `sm_61` without requiring a GPU. The selected FreeToken upstream commit is an ancestor, provenance reflects PR #257 as the primary Qwen source, and there is one authoritative downstream runtime path per Qwen capability.

## Phase 1 — Pascal and Qwen4 reference runtime

### Outcomes

- preserve/reapply FreeToken Pascal and CUDA 12.6 work on the reconciled upstream base;
- validate merged upstream Qwen3.8/Qwen4 text architecture against downstream and independent references;
- integrate downstream GGUF K/I loading without duplicating upstream model semantics;
- support heterogeneous expert-bank types needed by target artifacts;
- implement the dedicated, separately sharded PLE file as a core NVMe-backed path;
- use pinned PR #279 and other sources only as issue-scoped donors/references;
- provide mmap and positional-read PLE backends with random-access advice;
- provide direct and adaptive vectorized dedupe/order/coalesce lookup paths, bounded asynchronous prefetch and physical read-amplification telemetry;
- define an explicit PLE row-codec boundary with IQ4_NL as the initial reference;
- produce a cache-disabled, correctness-first short-context path.

### Exit gate

A tiny Qwen4 model passes CPU/reference tests after the upstream reconciliation. The dedicated PLE artifact is independently checksummed and both backends return identical rows and failure behavior. When a P4 arrives, one P4 generates deterministic short text with cache disabled inside a safe placement profile.

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

The ABI slice precedes AVX2 kernels. It defines immutable heterogeneous expert-bank descriptors, prepare/execute/group/cancel/telemetry contracts, caller-owned partial accumulation, bounded workspace and explicit decoder/thread-pool/NUMA hooks. Its microbenchmark interface records raw repeated timings for the supplied production geometry and every requested miss width from 1 through top-k; it does not by itself constitute a performance claim. The issue #16 threaded route adapter is opt-in, native-only, and census-gated per layer; it keeps serial execution for scalar, unsupported, and mixed-reference configurations.

The standalone Qwen GGUF CPU bridge is decode-only and owns its mapped host weights for the life of its heterogeneous CPU layout and Q4 executor. The CUDA engine rejects this GGUF combination until the production layer ABI can consume per-projection mappings without a homogeneous GPU cache; this is an integration blocker, not a hardware-performance claim.

The H0 `QwenGGUFCpuMoELayer` is an explicit CPU-only adapter around that bundle. It supports routed decode with the exact full-softmax Torch reference when CPU router logits are supplied, plus precomputed routes for direct parity tests. Its Qwen default preserves the model's unrenormalized selected probabilities. Calls require phase `decode` and group size one. It does not provide prefill, grouped, CUDA, TP>1 or performance evidence.

The H0 model-graph bridge transactionally replaces routed experts only for construction, lifecycle and correctness tests. The bundle remains caller-owned. This foundation does not make the CUDA-oriented trunk, router, shared expert, or LM head CPU-runnable and does not remove the Engine guard.

The standalone `GGUFCpuEagerBridge` is an explicit experimental H0/H1 wrapper around the CPU layer. It rejects prefill, grouped requests, graph capture and caller workspaces before transfer. CPU inputs use the adapter directly; non-CPU inputs use an injected blocking transfer seam, execute the adapter once, and copy the independent routed result back. It makes no stream, pinned-memory, overlap or performance claim. Real CUDA transfer behavior is H2-unverified.

### Exit gate

For each shipping quant and shape, CPU expert output passes error tolerances against dequantize-plus-reference matmul. End-to-end cache-zero output remains correct. Q4 and the named Q3 profile have complete identities/censuses and the fixed routing/tool/long-context/coding quality corpus is ready.

## Phase 3 — placement safety and Pascal GPU expert cache

### Outcomes

- per-GPU placement planner covering resident tensors, shared experts, recurrent/QSA/KV state, persistent and transient QSA score/top-k/gather workspaces, CUDA context, generic workspaces, transfer buffers and expert slots;
- post-load and post-first-large-prefill canary, explicit headroom, automatic cache/context/batch/placement backoff and fail-readiness behavior under issue #73;
- Pascal DP4A low-bit GPU kernels and format-specific tuning, ahead of generic FP16/BF16/FP8 optimization;
- fused Qwen3.8 `topk=10` router with a permanent Torch reference path;
- fixed-address per-GPU expert slot pools;
- trace-derived static-hot profile as a mandatory cache comparator and warm-start source;
- measured comparison of layer-owned, disjoint expert-owned, replicated and trunk split/TP policies;
- static mixed CPU/GPU output merge;
- async LFRU fill and persisted heat;
- routing telemetry and offline cache simulator.

### Exit gate

Single- and dual-P4 static-cache runs are correct and pass the post-prefill placement canary with documented reserve. The static-hot profile has measured hit/throughput evidence. Async future-token fill cannot delay the current token and improves a locality-positive trace relative to the appropriate static control or remains disabled.

## Phase 4 — adaptive hybrid execution and QSA scaling

### Outcomes

- current-step miss partition;
- concurrent CPU and H2D/GPU execution;
- measured `q*` tables under contended memory/PCIe conditions;
- adaptive policy with pure-CPU and fetch-all alternatives;
- NUMA and per-rank worker tuning;
- decode prefetch and copy batching;
- safe interaction with the #73 reserve and ownership policy;
- issue #76 phase-level QSA telemetry for projection, compressed-index maintenance, scoring, top-k selection, row gather, sparse attention, state update, allocation and host synchronization;
- bounded/reusable QSA score/top-k/gather workspaces with deterministic capacity checks;
- safe context/chunk/backoff and controlled errors instead of process abort;
- context-scaling benchmark harness from short context through 128K, with a 262K qualification attempt.

### Exit gate

The scheduler never chooses an unsupported or unsafe path, exposes its decisions, and beats or safely falls back to the best pure/static policy on representative decode workloads. QSA workspaces are bounded, the first-large-prefill high-water is inside the #73 reserve, and 32K/128K context behavior has independent correctness and performance evidence.

## Phase 5 — prefill, long context, serving and optional coding profile

### Core outcomes

- prefill expert grouping, chunked streaming and double buffering;
- GDN/QSA/PLE state validation through long contexts after issue #76 workspace/context qualification;
- semantic state checkpoint/restore;
- hardened OpenAI-compatible serving;
- Docker/Compose operations, health and metrics.

### Optional outcome

Issue #74 may add exact context-derived n-gram speculation after ordinary state and serving semantics are stable. It uses the same target model for verification, must preserve deterministic output, reports PLE I/O and negative controls, and automatically disables on low-acceptance workloads. It cannot block or replace the core phase exit.

### Exit gate

32K and 128K coding sessions run correctly with streaming, cancellation, restore and restart. Prefill does not thrash an undersized token-oriented cache, QSA exhaustion is controlled rather than fatal, and long-context overhead is measured. Optional #74 is either independently qualified or omitted from core release claims.

## Phase 6 — hardware qualification and release

### Outcomes

- actual P4/NUMA topology profile;
- one- and two-P4 qualification;
- #73 placement-cliff sweep and release safety margin through the first large prefill;
- #76 QSA workspace, synchronization and context-scaling evidence;
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
initial upstream import and provenance
  → issue #77: sync merged FreeToken Qwen3.8 PR #257
  → reconcile/retire duplicate PR #232 downstream code
  → Pascal CUDA 12.6 compile and tiny Qwen reference
  → GGUF + dedicated NVMe PLE reference
  → random-advised mmap/pread + adaptive PLE I/O/prefetch
  → AVX2 expert backend + named Q4/Q3 profiles
  → serving-ready host-expert integration
  → placement planner/post-prefill canary/backoff
  → Pascal DP4A expert kernels
  → fused topk=10 Pascal router
  → QSA workspace/context-scaling qualification
  → static-hot cache and mixed merge
  → dual-P4 policy comparison
  → async cache
  → current-step q*
  → prefill/long-context state
  → serving/package
  → release qualification

optional after stable state/serving:
  → exact context-derived n-gram profile
```

## Parallel work

Issue #77 owns broad Qwen model/QSA/PLE/cache/state reconciliation and must be serialized against edits to those same files. Before P4 arrival, independent non-overlapping workers can handle:

- CUDA 12.6/H1 environment maintenance;
- GGUF loader and quant-format work outside conflicting Qwen model files;
- CPU expert ABI/kernels;
- Q4/Q3 identity, census and mixed-precision quality fixtures;
- placement planner and result-schema pure logic;
- QSA benchmark/telemetry design that does not duplicate the upstream sync diff;
- fused router reference/dispatch and `sm_61` compile work after its model-call boundary is stable;
- cache trace/static-hot simulation formats;
- optional n-gram proposal/state fixtures, kept independent from core dependencies;
- server/config/metrics contracts;
- documentation and tests.

After #77 merges, workers can safely proceed on:

- dedicated PLE file format, random advice, mmap/pread backends, adaptive planner, read-amplification metrics and asynchronous prefetch;
- serving-ready host expert integration;
- QSA workspace reuse/synchronization changes;
- Pascal GPU and cache integration.

Avoid concurrent edits to the same model loader, quant registry, QSA backend, PLE implementation or cache core. The orchestrator assigns ownership and serializes those merges.

## Stop/reassess conditions

Create or amend an ADR before continuing if:

- issue #77 shows that merged upstream #257 conflicts fundamentally with required Pascal/GGUF/storage contracts;
- neither Q4 nor the named Q3 profile fits the host operating envelope with safe headroom;
- P4 lacks sufficient VRAM for the required always-active trunk and state after post-prefill placement backoff;
- placement canary evidence shows unavoidable spill/fallback cliffs below a useful configuration;
- QSA selection/workspace/synchronization overhead makes required 32K or 128K operation unusable and cannot be bounded safely;
- the dedicated PLE shard cannot provide stable random-I/O behavior without unrelated-weight coupling or unacceptable read amplification;
- AVX2 CPU experts are so slow that current-step CPU work cannot help;
- no realistic cache size beats the static-hot/control path;
- every two-P4 policy loses to one P4 plus CPU because of communication/merge cost;
- full-QSA implementation is not correct on Pascal;
- a vLLM-based implementation becomes demonstrably smaller and faster to complete.

A failed optimization does not invalidate the project. Preserve the last correct fallback and update the plan from evidence.
