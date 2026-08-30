# Implementation plan

## Delivery strategy

The project is implemented as a sequence of gated vertical slices. Hardware-independent work comes first. Each phase ends with a usable fallback and a decision gate; no phase assumes the next one succeeds. Optional performance profiles are isolated from the core release.

The execution order treats completed issue #77/PR #81 as the authoritative upstream Qwen3.8 reconciliation prerequisite: exact upstream commit `58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6` is an ancestor, duplicate pre-sync paths are not extended, and all new work consumes the reconciled contracts. The ordinary-decode, cache-zero path remains the required base path throughout. Routed expert matrices and the PLE table may be compressed aggressively, while control/state tensors retain an explicit conservative precision decision, and every quality claim must include long trajectories rather than relying on “the model runs”, short prompts, throughput, or needle retrieval.

Modern-GPU and unified-memory measurements are feasibility evidence for selecting experiments only. They are not Pascal performance predictions; all P4 placement, precision, and throughput defaults remain H2/H3/H4 hardware-gated.

## Phase 0 — downstream foundation and upstream reconciliation

### Outcomes

- upstream FreeToken history is merged into this repository;
- exact source pins and license provenance exist;
- CUDA 12.6 build environment is reproducible;
- hosted CI, H1 compilation and deferred H2/H3 workflows exist;
- tiny models, fixtures, benchmark schema and result tooling exist;
- issue #77 was completed via PR #81: exact upstream commit `58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6` is an ancestor and PR #257 is the authoritative merged Qwen3.8 source;
- model/QSA/PLE/cache/state code adapted from closed PR #232 is compared file by file with merged upstream and retained only where downstream behavior remains materially distinct;
- open PLE mmap PR #279 is recorded through an immutable `planned` or `reference` pin before any code is mined.

### Exit gate

A clean clone passes hosted CI and compiles the reconciled source set for `sm_61` without requiring a GPU. PR #81's ancestor/provenance and file-by-file delta evidence are present, PR #257 is the primary Qwen source, and there is one authoritative downstream runtime path per Qwen capability.

## Phase 1 — Pascal and Qwen4 reference runtime

### Outcomes

- preserve/reapply FreeToken Pascal and CUDA 12.6 work on the reconciled upstream base;
- validate merged upstream Qwen3.8/Qwen4 text architecture against downstream and independent references;
- retain the merged issue #93 H0/H1 GDN backend decision, permanent reference/fallback contract, standalone CUDA 12.6 `sm_61` compile seam, ragged/concurrent isolation and donor provenance; `pascal-fp32` is explicit-only with visible fallback/rejection, and its H2 one-P4 parity and end-to-end qualification remains deferred;
- integrate downstream GGUF K/I loading without duplicating upstream model semantics;
- support heterogeneous expert-bank types needed by target artifacts;
- implement the dedicated, separately sharded PLE file as a core NVMe-backed path; NVMe PLE bytes remain a separate ownership boundary, independently observable and pageable, and are never generalized into routed-expert swap or generic expert execution;
- use pinned PR #279 and other sources only as issue-scoped donors/references;
- provide mmap and positional-read PLE backends with `MADV_RANDOM`/`POSIX_FADV_RANDOM` advice, explicit application/failure telemetry, and invalid range/manifest/codec negative cases;
- provide direct and adaptive vectorized dedupe/order/coalesce lookup paths, bounded cancellable asynchronous prefetch that cannot publish stale/partial rows, and physical read-amplification telemetry, with cold-cache and warm-cache evidence kept separate;
- define an explicit PLE row-codec boundary with IQ4_NL as the initial reference and qualify the ADR 0011 experiment matrix (BF16 mmap control, FP8 per-row, INT4 group-16, NVFP4-style group-16, near-lossless Q6/Q8);
- produce a cache-disabled, correctness-first short-context path.

### H0/H1 exit gate

A tiny Qwen4 model passes CPU/reference tests on the completed #77/#81 reconciliation, including merged issue #93 GDN recurrence/reference, ragged/concurrent isolation, reset/restore and visible fallback/rejection cases. The dedicated PLE artifact is independently checksummed; both backends return identical rows, failure telemetry, and no-stale prefetch behavior; and the H1 `sm_61` compile gate passes. These hosted gates do not claim P4 behavior.

### Deferred H2/H3 gate

After P4 installation, one-P4 GDN parity/end-to-end qualification, PLE I/O behavior, and safe cache-disabled decode must pass on hardware. The optimized `pascal-fp32` GDN path is explicit-only and remains disabled unless that evidence passes.

## Phase 2 — AVX2 host expert backend and model profiles

### Outcomes

- model-agnostic expert execution ABI;
- AVX2 Q4_K implementation first;
- additional Q2/Q3/IQ types selected from real model censuses;
- pinned `reference-q4` and named `throughput-q3` whole-model identities;
- explicitly tracked non-default `candidate-ap-q4` (`AP-Q4_K_XL`) and `candidate-ap-iq4` (`AP-IQ4_XS`) identities, each gated by immutable provenance, census, converter, quality, CPU-cost and P4 evidence;
- a machine-readable sensitive-tensor census for every profile, with exact tensor identity/class, dtype or quant format, scale representation, conversion provenance, and precision rationale;
- a sensitive-tensor precision island covering routers, `shared_expert_gate` and scales, GDN state-driving/control projections including reconciled `in_proj_a`/`in_proj_b` classes, residual/hyperconnection write gates, norms, and other continuously active controls identified by reference comparison;
- source/lossless precision as the baseline for sensitive classes, with Q8 or lower treated as an explicit per-class experiment rather than an inherited Q2/Q3/Q4 rule;
- independent scale/dequant parity tests for `shared_expert_gate` and equivalent router/GDN/control tensors;
- controlled broken fixtures that mis-scale a shared-expert gate and perturb/reduce a GDN state-control tensor, with the quality harness required to fail on both;
- gate/up activation/down fused at the right boundary;
- bounded pinned-memory and NUMA-aware host bank;
- load and pre-fault the complete quantized routed-expert bank into its DDR4 serving allocation, keep it as the sole steady-state source for CPU execution and P4 cache fills, and account separately for PLE Linux page-cache residency without creating an uncontrolled duplicate full-bank copy;
- SSD expert reads are startup backing or an explicitly gated experiment, not the serving design;
- parity, quality and microbenchmark suite, including long-horizon multi-turn coding, repeated tool calls/results, state-dependent reasoning, structured transformations, long generation, loop/token-ceiling detection, checkpoint/restore and suffix replay with intermediate router/gate/GDN state where feasible.

The ABI slice precedes AVX2 kernels. It defines immutable heterogeneous expert-bank descriptors, prepare/execute/group/cancel/telemetry contracts, caller-owned partial accumulation, bounded workspace and explicit decoder/thread-pool/NUMA hooks. Its microbenchmark interface records raw repeated timings for the supplied production geometry and every requested miss width from 1 through top-k; it does not by itself constitute a performance claim. The issue #16 threaded route adapter is opt-in, native-only, and census-gated per layer; it keeps serial execution for scalar, unsupported, and mixed-reference configurations.

The standalone Qwen GGUF CPU bridge is decode-only and owns its mapped host weights for the life of its heterogeneous CPU layout and Q4 executor. The CUDA engine rejects this GGUF combination until the production layer ABI can consume per-projection mappings without a homogeneous GPU cache; this is an integration blocker, not a hardware-performance claim.

The H0 `QwenGGUFCpuMoELayer` is an explicit CPU-only adapter around that bundle. It supports routed decode with the exact full-softmax Torch reference when CPU router logits are supplied, plus precomputed routes for direct parity tests. Its Qwen default preserves the model's unrenormalized selected probabilities. Calls require phase `decode` and group size one. It does not provide prefill, grouped, CUDA, TP>1 or performance evidence.

The H0 model-graph bridge transactionally replaces routed experts only for construction, lifecycle and correctness tests. The bundle remains caller-owned. This foundation does not make the CUDA-oriented trunk, router, shared expert, or LM head CPU-runnable and does not remove the Engine guard.

The standalone `GGUFCpuEagerBridge` is an explicit experimental H0/H1 wrapper around the CPU layer. It rejects prefill, grouped requests, graph capture and caller workspaces before transfer. CPU inputs use the adapter directly; non-CPU inputs use an injected blocking transfer seam, execute the adapter once, and copy the independent routed result back. It makes no stream, pinned-memory, overlap or performance claim. Real CUDA transfer behavior is H2-unverified.

### Exit gate

### H0 exit gate

For each shipping quant and shape, CPU expert output passes error tolerances against dequantize-plus-reference matmul. Every profile has a complete sensitive-tensor census and explicit precision/scale provenance, shared-gate/control scale/dequant parity passes, and deliberately mis-scaled shared-gate and degraded GDN-control fixtures fail the quality harness. The loaded, pre-faulted DDR4 expert bank is the sole steady-state CPU/cache-fill source and PLE page-cache accounting is recorded. End-to-end cache-zero ordinary decode remains correct, and Q4/Q3/AP identities and conversion evidence are complete; no Q3 or component precision choice is a release default yet.

### Deferred H2/H3 qualification

Actual P4 expert kernels, whole-model Q4/Q3/AP profile quality and throughput, and any Q8/lower sensitive-class promotion remain deferred to hardware. CPU or modern-GPU results cannot select the P4 default.

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
- H0 NUMA and per-rank worker policy hooks with physical-core affinity and no accidental oversubscription, followed after P4 installation by discovery of each `P4 -> PCIe root -> NUMA node -> CPU socket` relationship;
- hard pin/staging limits plus page-residency and cross-node counters, then deliberate local-node, remote-node, and interleaved host-bank/staging placement controls measured with end-to-end decode, not synthetic bandwidth alone; unexplained cross-socket traffic or page migration fails H3;
- decode prefetch and copy batching;
- safe interaction with the #73 reserve and ownership policy;
- issue #76 phase-level QSA telemetry for projection, compressed-index maintenance, scoring, top-k selection, row gather, sparse attention, state update, allocation and host synchronization;
- bounded/reusable QSA score/top-k/gather workspaces with deterministic capacity checks;
- safe context/chunk/backoff and controlled errors instead of process abort;
- context-scaling benchmark harness from short context through 128K, with a 262K qualification attempt.

### Exit gate

The scheduler never chooses an unsupported or unsafe path, exposes its decisions, and beats or safely falls back to the best pure/static policy on representative ordinary-decode workloads. QSA workspaces are bounded, the first-large-prefill high-water is inside the #73 reserve, and 32K/128K context behavior has independent correctness and performance evidence. After hardware installation, each P4's CPU expert workers, expert-bank pages, and staging buffers are assigned from its measured local NUMA node where possible, and local/remote/interleaved end-to-end results are retained before H3 policy selection.

## Phase 5 — prefill, long context, serving and optional coding profile

### Core outcomes

- prefill expert grouping, chunked streaming and double buffering;
- GDN/QSA/PLE state validation through long contexts after issue #76 workspace/context qualification;
- complete #26 state schema covering GDN, QSA/KV/index/filter, PLE rolling/hash, hyperconnection, positions and sampling metadata, with version/checksum/model-quant-precision compatibility rejection, atomic cleanup and preserved-prefix suffix replay;
- long-horizon qualification across multi-turn coding, repeated tool calls/results, state-dependent reasoning, structured transformations, long generation, and looping/token-ceiling failure modes, with router/gate/GDN intermediate comparisons where feasible;
- matched uninterrupted, chunked, checkpoint/restore, and suffix-replay trajectories so cumulative GDN/control-state error is visible even when short retrieval and simple questions pass;
- hardened OpenAI-compatible serving;
- Docker/Compose operations, health and metrics.

### Optional outcome

Issue #74 may add exact context-derived n-gram speculation after ordinary state and serving semantics are stable. It uses the same target model for verification, must preserve deterministic output, reports PLE I/O and negative controls, and automatically disables on low-acceptance workloads. It cannot block or replace the core phase exit.

### H3/H4 exit gate

32K and 128K coding sessions run correctly with streaming, cancellation, restore and restart, and the long-horizon corpus passes for multi-turn coding, repeated tools, state-dependent reasoning, structured transforms, and long generation without looping or token-ceiling failure. Intermediate state comparisons and suffix replay agree within declared tolerances, the versioned/checksummed state schema rejects incompatible model/quant/precision/runtime/TP checkpoints, atomic cleanup works, and a reduced-precision GDN control fixture fails even if its short-prompt output remains coherent. Prefill does not thrash an undersized token-oriented cache, QSA exhaustion is controlled rather than fatal, and long-context overhead is measured. Optional #74 is either independently qualified or omitted from core release claims.

## Phase 6 — hardware qualification and release

### Outcomes

- actual P4/NUMA topology profile and physical-core/no-oversubscription, pin/staging-limit, page-residency and cross-node-counter evidence;
- immutable mapping of every installed P4 to its PCIe root, NUMA node, and CPU socket, followed by local-node, remote-node, and interleaved end-to-end decode measurements for workers, expert-bank pages, and staging;
- one- and two-P4 qualification;
- #73 placement-cliff sweep and release safety margin through the first large prefill;
- #76 QSA workspace, synchronization and context-scaling evidence;
- whole-model `reference-q4` and `throughput-q3` comparison;
- component-level Q5/Q8 and shipping CPU/GPU format comparison, plus the ADR 0011 PLE codec experiment matrix;
- benchmark comparison with merged llama.cpp/PXQ;
- independently measured cold-cache, warm-cache, major-page-fault, physical-read/amplification and steady-state PLE behavior for both storage backends;
- cache-zero, static-hot, static, dynamic and current-step-hybrid controls;
- soak and fault injection;
- v1 release artifact, image, docs and known limitations;
- optional `coding-ngram` profile only when its separate evidence passes.
- explicit evidence that NVMe steady-state reads are dedicated to PLE; routed experts execute from the complete DDR4 bank or bounded P4 cache, never from generic SSD expert swap;
- explicit caveats separating modern-GPU feasibility measurements from actual-P4 performance evidence.

### Exit gate

Every required core checkbox in `release-criteria.md` has linked evidence. Optional-profile failure does not fail the core release; the profile is omitted or disabled.

## Critical path

```text
initial upstream import and provenance
  → completed issue #77 via PR #81: verify upstream ancestor and authoritative delta ledger
  → Pascal CUDA 12.6 compile and tiny Qwen reference
  → merged issue #93 H0/H1 GDN reference/fallback and `sm_61` compile seam (H2 deferred)
  → GGUF + dedicated NVMe PLE reference
  → random-advised mmap/pread + adaptive PLE I/O/prefetch
  → issue #14 reference corpus and sensitive-tensor scale/precision failure gates
  → AVX2 expert backend + issue #17 named Q4/Q3 profiles and sensitive census
  → issue #18 host-bank/NUMA policy hooks
  → serving-ready host-expert integration
  → placement planner/post-prefill canary/backoff
  → Pascal DP4A expert kernels
  → upstream arbitrary-K router adaptation and measured Pascal fallback decision
  → QSA selected-row path + workspace/context-scaling qualification
  → static-hot cache and mixed merge
  → dual-P4 policy comparison
  → measured P4→PCIe→NUMA topology and local/remote/interleaved end-to-end placement sweep
  → async cache
  → current-step q*
  → prefill/long-context state
  → serving/package
  → release qualification

optional after stable state/serving:
  → exact context-derived n-gram profile
```

## Parallel work

Issue #77's broad Qwen model/QSA/PLE/cache/state reconciliation is complete via PR #81 and is the authoritative prerequisite; no duplicate pre-sync path may be extended. Before P4 arrival, independent non-overlapping workers can handle:

- CUDA 12.6/H1 environment maintenance;
- GGUF loader and quant-format work outside conflicting Qwen model files;
- CPU expert ABI/kernels;
- Q4/Q3/AP candidate identity, sensitive-tensor census, converter-regression, scale/dequant parity and mixed-precision quality fixtures against the reconciled Qwen graph; no profile default is implied;
- merged issue #93 H0/H1 reference/fallback tests and `sm_61` compile work; H2 remains hardware-deferred;
- placement planner and result-schema pure logic;
- QSA benchmark/telemetry design that does not duplicate the upstream sync diff;
- merged-upstream arbitrary-K router adaptation/reference/dispatch and `sm_61` compile work after its model-call boundary is stable;
- cache trace/static-hot simulation formats;
- optional n-gram proposal/state fixtures, kept independent from core dependencies;
- server/config/metrics contracts;
- documentation and tests.

With #77/PR #81 complete, workers can proceed on:

- dedicated PLE file format, random advice, mmap/pread backends, adaptive planner, read-amplification metrics and asynchronous prefetch;
- serving-ready host expert integration;
- QSA workspace reuse/synchronization changes;
- Pascal GPU and cache integration.

Avoid concurrent edits to the same model loader, quant registry, QSA backend, PLE implementation or cache core. The orchestrator assigns ownership and serializes those merges.

## Stop/reassess conditions

Create or amend an ADR before continuing if:

- the completed #77 reconciliation shows that merged upstream #257 conflicts fundamentally with required Pascal/GGUF/storage contracts;
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

A failed optimization does not invalidate the project. Preserve the last correct fallback and update the plan from evidence. A quality harness that does not fail the intentionally broken shared-gate and GDN-control fixtures, or a placement result based only on synthetic bandwidth or GPU ordinal, is an unmet gate rather than evidence to waive.
