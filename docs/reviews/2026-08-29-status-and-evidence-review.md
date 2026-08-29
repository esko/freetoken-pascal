# FreeToken-Pascal Status and Evidence Review — 2026-08-29

## Review scope

This review checks the current `esko/freetoken-pascal` repository against the North Star:

> Provide a highly performant and reliable way to serve the full Qwen3.8-Flash-Next MoE model on the dual-Xeon Fujitsu server with two Tesla P4 GPUs.

The external review covered Hugging Face model artifacts and discussions, Reddit/LocalLLaMA operating reports, public X-linked primary repositories where direct X access was unreliable, and current work in FreeToken, llama.cpp, vLLM, PXA/PXQ, and related forks. Community measurements prioritize experiments; they are not performance predictions for Pascal.

## Executive assessment

The project is materially ahead of its original planning status. It is no longer a documentation-only fork and does not need a broad architectural reset.

The repository already contains substantial H0/H1 implementation for:

- a pinned FreeToken downstream source tree and provenance ledger;
- CUDA 12.6 and `sm_61` compile gates;
- the Qwen3.8/Qwen4 text architecture, QSA, GDN, hyperconnections, PLE state, and reference hooks;
- safe heterogeneous GGUF ingestion and exact tensor census;
- file-range-backed expert and PLE mappings;
- a model-agnostic CPU expert ABI;
- direct packed Q4_K, Q5_K, Q5_1, and Q8_0 AVX2 GEMV paths;
- threaded route execution, CPU topology discovery, worker-affinity verification, NUMA policy hooks, and no-swap admission;
- standalone Qwen GGUF CPU expert bundles, model attachment, and a blocking device bridge for correctness experiments;
- PLE warm/cold modes, page residency, major/minor fault, and process-I/O telemetry;
- a defined exact `topk=10` Pascal-router work item.

A new upstream fact changes the immediate action order: FreeToken PR #257 merged on August 28 and current upstream `main` contains a newer text-only Qwen3.8 implementation with full QSA, GDN, PLE, CUDA-graph/radix-state support, and hybrid MoE execution. The downstream tree still contains earlier adaptations based partly on closed PR #232. Issue #77 must therefore precede substantial new model/cache/storage work: sync exact upstream commit `58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6`, compare every Qwen path, eliminate duplicate implementations, and preserve only the downstream Pascal/GGUF/CPU/storage/placement/evidence delta that remains necessary.

After that reconciliation, the critical path is:

1. dedicated PLE serving artifact and optimized random-I/O implementation;
2. serving-ready GGUF/host-expert integration rather than the current explicit H0 bridge;
3. Pascal GPU expert kernels and the fused router;
4. safe one-P4 placement and post-prefill canary qualification;
5. QSA long-context workspace and synchronization optimization;
6. dual-P4 ownership/cache topology;
7. static then dynamic expert caching and CPU/GPU co-execution;
8. long-context serving, deployment, soak, and release evidence.

No real P4 execution result is present yet. That is the largest remaining uncertainty and must not be replaced by results from 3090/4090/5090, GB10, Apple unified memory, P40, or other architectures.

## Current implementation status

| Area | Status | Assessment |
|---|---|---|
| Upstream/provenance | Implemented but stale Qwen base | Strong ledger and history. Issue #77 must advance to merged upstream PR #257 and reconcile the downstream delta. |
| CUDA 12.6 / `sm_61` | H1 implemented | Compile gates exist; runtime confirmation waits for P4. |
| Qwen3.8 architecture | H0/H1 implemented | Text backbone and reference/state paths exist, but duplicate/upstream divergence must be resolved by #77 before further expansion. |
| GGUF/K/I ingestion | Implemented | Safe shard/offset/stride handling and heterogeneous census are a project strength. |
| PLE source mapping | Implemented reference path | Exact GGUF range mapping, IQ4_NL decode, warm modes, and telemetry exist. |
| Dedicated PLE serving artifact | Planned, not complete | Issue #13 makes it core v1; extraction, positional reads, random advice, vectorized planning, physical-I/O telemetry, and prefetch remain. |
| CPU expert ABI | Implemented | Model-neutral boundary and failure semantics are well designed. |
| AVX2 experts | Substantial H0 implementation | Q4_K/Q5_K/Q5_1/Q8_0 direct packed paths and threading exist; complete real-artifact and target-host evidence remains. |
| GGUF CPU serving integration | Partial | Bundle/layer/model/eager bridges prove contracts but are deliberately not the final serving path. |
| Pascal GPU experts | Planned | PXA/PXQ-derived `sm_61` work has not reached a validated P4 path. |
| `topk=10` router | Planned | Correct Torch fallback exists; fused Pascal path remains issue #38. |
| Placement safety | Planned | Issue #73 requires post-load/post-first-large-prefill high-water, canary, backoff, and fail-readiness. |
| QSA long-context path | Partial reference work | QSA exists, but context-scaling attribution, reusable workspace bounds, synchronization removal, and controlled OOM behavior remain issue #76. |
| Static hot expert cache | Planned | Issue #21 now makes it a mandatory comparator and warm-start source. |
| Dynamic cache / `q*` | Planned | Architecture is sound; no P4 locality/overlap evidence yet. |
| Dual GPU | Planned | Candidate policies must be measured rather than assuming TP or disjoint ownership. |
| Long-context state | Partial reference work | Required H3/H4 state/save/restore evidence remains. |
| Serving/operations | Upstream base plus downstream plans | Production cancellation, health, metrics, Compose, and fault qualification remain. |

## External model and quant review

### Q4 remains the quality reference

The Unsloth `UD-Q4_K_XL` artifact remains the appropriate high-quality baseline. It is large enough that exact host-memory accounting matters, but modern-GPU field reports show that full CPU-MoE placement can retain much of decode throughput while freeing VRAM for context and the active trunk.

The project should preserve Q4 as the reference profile even if it is not the eventual throughput default.

### A named Q3 profile is a first-class throughput candidate

A pinned `UD-Q3_K_XL` deployment on one GB10 reported an approximately 90 GB artifact, roughly 20 tok/s decode at a 32K prompt, and exact outputs versus its paired baseline in a small coding/tool corpus. Its separate exact n-gram sweep is also the strongest reproducible coding-specific speculation evidence found in this review.

This does not prove Q3 quality or speed on P4. It is strong enough to replace vague “some 3-bit comparison” language. Issue #17 now requires one immutable Q3 artifact, a complete census, imatrix/calibration provenance where available, and direct qualification beside Q4.

### Imatrix provenance matters

Community GGUFs with published calibration/imatrix provenance are preferable to opaque requants. Bartowski and AtomicChat artifacts are useful evidence sources because they document chat-template/tool-heavy calibration or architecture-specific importance matrices.

A third-party quant does not become trusted merely because it is popular. Release use still requires an immutable revision, checksums, tensor census, tokenizer/template verification, license review, and the project's routing/tool/long-context/coding gates.

### Q5 and Q8 are normally component formats

A full Q5 or Q8 model is unlikely to fit the 128 GB host envelope with safe operating headroom. The useful question is whether selected continuously active or sensitive tensors benefit from Q5/Q8 while routed experts use lower-bit CPU/P4 formats.

The format matrix must distinguish:

- whole-model Q4/Q3 fit candidates;
- expert-bank CPU formats;
- GPU-cache formats;
- router/control/shared/trunk formats;
- PLE row codecs.

### MTP exists but should not block v1

Several standalone and built-in Qwen3.8 MTP artifacts now exist, including low-bit heads. Their model cards and discussions expose incompatible runtime/head layouts. MTP remains post-v1 until one pinned runtime/head pair, state semantics, P4 verification performance, quality, and fallback behavior are established.

### Pruned expert variants remain secondary

REAP/MEP-style reduced-expert artifacts demonstrate storage reduction with modest benchmark loss, but they alter the model and do not automatically reduce configured top-k active work. They are not aligned with the current full-model North Star and remain post-v1 experiments.

## Runtime and community evidence

### FreeToken now has merged Qwen3.8 support

PR #257 merged into FreeToken on August 28. It is now the primary upstream Qwen3.8 engine source, not merely a reference. It includes the text architecture, full QSA, PLE, CUDA graph decode, hybrid radix state, and MoE offload/hybrid execution. Its 4090/5090 results are useful proof that the architecture works on modern GPUs, not evidence of P4 speed.

Issue #77 makes a clean upstream sync and semantic delta reconciliation the immediate priority. PR #232 becomes historical provenance only where downstream code remains distinct after that sync.

### FreeToken has a new mmap PLE branch

Open PR #279 adds `--ple-backend mmap` and reports fitting Qwen3.8 with a 250K FP16 context inside 128 GB total RAM plus VRAM on a 5090/96 GB host. This directly validates the project's NVMe-backed PLE direction.

The PR does not replace issue #13. The downstream requirement is broader: dedicated PLE storage independent of unrelated model weights, mmap and positional-read backends, random-access advice, adaptive batching/ordering, physical read-amplification telemetry, codec identity, and P4/Xeon validation. PR #279 should be pinned and mined as a donor/reference, not merged from a moving head.

### Upstream llama.cpp support is merged

Qwen3.8-Flash-Next support is no longer only an experimental llama.cpp PR. PR #27742 merged, including text, QSA, vision, conversion, and quantizer fixes. A separately pinned merged revision should be the principal external GGUF correctness and operational comparator.

### Static hot-expert residency is a serious baseline

A llama.cpp residency experiment splits each layer's routed experts into hot GPU and cold CPU branches and reports approximately 1.6× decode improvement at equal VRAM on Qwen3.8 Q4_K_XL on a 5090, with identical output and unchanged prefill.

The number does not transfer to P4. The experiment validates two planning choices:

- expert-level placement is more valuable than arbitrary complete expert layers;
- a trace-derived static hot profile should precede dynamic-cache claims.

### CPU-MoE is a credible baseline

Multiple Qwen3.8 llama.cpp reports show that leaving all routed experts in system RAM can cost surprisingly little decode throughput on modern GPUs while substantially reducing VRAM use. This supports the mandatory cache-zero/CPU-backed baseline and reserving P4 VRAM for the active trunk, state, and hottest experts.

It does not prove the Haswell AVX2 path will be fast enough. Target-host CPU and contention measurements remain decisive.

### Exact context-derived speculation matches coding-agent work

The pinned `sxuff/qwen38-flash-next-dgx-spark` paired sweep compared ordinary decode with `ngram-mod` using identical requests, model, sampling, and output hashes. It reported about 3.8× wall-clock improvement for copying Python, 2.6× for copying JSON, 1.9× for a structured transformation, approximately neutral behavior for novel code with no usable drafts, and about 2.49× aggregate wall-clock improvement with exact paired outputs.

This is one machine and one sweep, not a universal benchmark. It maps directly to coding-agent edits, patches, repeated tool output, JSON, and configuration transformations. ADR 0011 therefore permits it as optional v1 work without making it release-critical.

### PLE access must be treated as random I/O

Large-context community experiments have reported severe read amplification when sequential readahead is applied to sparse PLE access. Random-access advice reduced physical traffic substantially in one reported deployment.

The exact result must be reproduced on the server's NVMe, but the implementation implications are low risk:

- mmap requests random access where supported;
- positional reads use equivalent advice;
- physical block-device bytes are compared with logical packed PLE bytes;
- dedupe/sort/coalesce is adaptive rather than unconditional for tiny decode batches.

### VRAM placement has a post-prefill cliff

Hybrid reports show severe throughput collapse when a placement change pushes the runtime beyond practical VRAM capacity, even when the model starts and returns correct output. Other Qwen3.8 reports show post-load headroom shrinking after the first large prefill because selection/gather workspaces are allocated or retained, with nearby configurations aborting on top-k workspace exhaustion.

Issue #73 measures both post-load and post-first-large-prefill high-water and refuses readiness when the canary cannot preserve the configured reserve.

### QSA needs its own long-context workstream

Community measurements show Qwen3.8 decode falling materially as context grows even when the routed-expert policy is unchanged. Potential contributors include QSA score/top-k selection, compressed-index maintenance, gathered sparse attention, metadata transfer, host synchronization, workspace allocation, and graph/eager behavior.

This is not evidence that one specific component is responsible on P4. Issue #76 isolates the phases, bounds workspaces, removes avoidable synchronization, integrates post-prefill high-water into #73, and requires controlled failure instead of server abort.

## Architecture assessment

### What should remain unchanged

- FreeToken remains the correct primary base for expert-cache/co-execution work.
- Current upstream FreeToken contracts should be consumed before duplicate downstream extensions are added.
- Merged llama.cpp remains the external GGUF correctness/deployment comparator.
- The complete unpruned model and permanent CPU/cache-zero fallbacks remain mandatory.
- ADR 0010's three-tier architecture remains the center:
  - dedicated PLE on NVMe;
  - complete routed experts in DDR4;
  - dense/shared/state/hot experts in P4 VRAM.
- Correctness and performance changes remain separable.
- Real P4 evidence remains required for all defaults.

### Clarifications and modifications

1. **Sync before extending.** Issue #77 reconciles merged upstream PR #257 and removes duplicate Qwen code before major new work.
2. **Expert residency must not imply a second full anonymous copy.** Memory accounting prevents complete duplicate expert copies plus uncontrolled page cache from exhausting 128 GB.
3. **PLE gets a random-I/O contract.** Add random advice, read-amplification telemetry, adaptive vectorized planning, and an explicit codec boundary. Mine PR #279 only through an immutable pin.
4. **Placement gets a post-prefill safety gate.** Observe every category, reserve headroom, run a large-prefill/decode canary, and back off cache, context, batch, or placement.
5. **QSA gets an independent performance/workspace contract.** Profile each phase, bound/reuse scratch, eliminate avoidable host synchronization, and fail cleanly when context cannot fit.
6. **Q4 and Q3 become named profiles.** Q4 is the reference; the pinned Q3 is the whole-model throughput candidate. Q5/Q8 are normally component experiments.
7. **Static hot experts precede dynamic claims.** Cache-zero, static-hot, async dynamic, and current-step hybrid modes are compared independently.
8. **Dual-P4 policy stays evidence-driven.** Compare layer-owned, disjoint expert-owned, replicated, and trunk split/TP combinations.
9. **Exact n-gram speculation is optional v1 work.** It cannot delay or destabilize core release. MTP remains later.

## Recommended execution order

### Before the P4s arrive

1. Complete issue #77: sync upstream FreeToken `58f4b9ec...`, reconcile PR #257, retire duplicate #232 paths, and pin PR #279 as a planned/reference donor.
2. Complete issue #13's dedicated PLE artifact, mmap/positional reads, random advice, adaptive planner, read-amplification telemetry, and codec boundary.
3. Finish real Q4 and selected Q3 census/parity coverage through issue #17.
4. Finish the serving-ready host-expert integration that replaces the current correctness-only blocking bridge.
5. Implement H0/H1 portions of #73: placement planner, post-load/post-prefill canary, fallback state machine, and schemas.
6. Implement H0/H1 portions of #76: QSA telemetry, workspace accounting/reuse, synchronization audit, controlled-OOM tests, and context-sweep harness.
7. Prepare static hot-profile import/simulation and persisted-heat formats for #21/#22.
8. Keep optional #74 independent so it cannot block the core path.

### When the first P4 is installed

1. Qualify airflow, power, clocks, PCIe link, and NUMA locality before model work.
2. Run the smallest deterministic Qwen fixture and kernel parity suite.
3. Establish cache-zero Q4 and Q3 placement envelopes through the first large prefill with #73.
4. Measure QSA-only and end-to-end scaling across short, 32K, 64K and 128K contexts with #76.
5. Benchmark format-specific Pascal kernels and fused `topk=10` before dynamic cache scheduling.
6. Run static hot-expert placement and collect real routing locality.

### When both P4s are installed

1. Compare layer-owned, disjoint expert-owned, replicated, and trunk split/TP policies.
2. Enable asynchronous LFRU only after the static comparator is stable.
3. Add current-step `q*` only after contended CPU/PCIe/GPU timings are available.
4. Re-run #73/#76 under the selected two-card topology and qualify state, serving, cancellation, and deployment.
5. Run optional exact n-gram profiling only after ordinary state semantics are stable.

## Go/no-go criteria

Reassess the design if any of the following are observed on real hardware:

- the active trunk and required state cannot fit across the P4s with safe post-prefill reserve;
- QSA selection/workspace/synchronization overhead makes required 32K or 128K operation unusable and cannot be bounded safely;
- the target Xeons cannot execute enough CPU misses to help any hybrid split;
- realistic expert-cache sizes show negligible locality over the static baseline;
- PCIe/merge overhead makes disjoint ownership slower than layer ownership;
- PLE physical I/O remains critical after random advice and batching;
- the selected Q3 fails routing/tool/long-context quality gates;
- dynamic cache or speculation cannot avoid negative-control regressions.

A failed optimization does not invalidate the product. Preserve the last reliable fallback and change the default from measured evidence.
