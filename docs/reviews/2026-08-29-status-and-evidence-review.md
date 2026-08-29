# FreeToken-Pascal Status and Evidence Review — 2026-08-29

## Review scope

This review checks the current `esko/freetoken-pascal` repository against the North Star:

> Provide a highly performant and reliable way to serve the full Qwen3.8-Flash-Next MoE model on the dual-Xeon Fujitsu server with two Tesla P4 GPUs.

The public model/runtime review covered Hugging Face model artifacts and discussions, Reddit/LocalLLaMA reports, public X mirrors where direct access was unavailable, and active work in FreeToken, llama.cpp, vLLM, PXA/PXQ, and related forks. Community measurements are used to prioritize experiments, not as predictions for Pascal.

## Executive assessment

The project is materially ahead of its original planning status. It is no longer a documentation-only fork and no longer needs a broad architectural reset.

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
- PLE warm/cold modes, page residency, major/minor fault, and `/proc/self/io` telemetry;
- a defined exact `topk=10` Pascal-router work item.

The critical path is now integration and hardware qualification:

1. dedicated PLE serving artifact and optimized random-I/O implementation;
2. serving-ready GGUF/host-expert integration rather than the current explicit H0 bridge;
3. Pascal GPU expert kernels and the fused router;
4. safe one-P4 placement and canary qualification;
5. dual-P4 ownership/cache topology;
6. static then dynamic expert caching and CPU/GPU co-execution;
7. long-context, serving, deployment, soak, and release evidence.

No real P4 execution result is present yet. That is the largest remaining uncertainty and must not be replaced by results from 3090/4090/5090, GB10, Apple unified memory, or other architectures.

## Current implementation status

| Area | Status | Assessment |
|---|---|---|
| Upstream/provenance | Implemented | Strong. Exact SHAs, import method, licensing, and responsible issue/PR are tracked. |
| CUDA 12.6 / `sm_61` | H1 implemented | Compile gates exist; runtime confirmation waits for P4. |
| Qwen3.8 architecture | H0/H1 implemented | Text backbone and reference/state paths are present. H2 end-to-end validation remains. |
| GGUF/K/I ingestion | Implemented | Safe shard/offset/stride handling and heterogeneous census are a project strength. |
| PLE source mapping | Implemented reference path | Exact GGUF range mapping, IQ4_NL decode, warm modes, and telemetry exist. |
| Dedicated PLE serving artifact | Planned, not complete | ADR 0010 and issue #13 now make it core v1; extraction, `pread`, vectorized planning, and prefetch remain. |
| CPU expert ABI | Implemented | Model-neutral boundary and failure semantics are well designed. |
| AVX2 experts | Substantial H0 implementation | Q4_K/Q5_K/Q5_1/Q8_0 direct packed paths and threading exist; complete real-artifact and target-host evidence remains. |
| GGUF CPU serving integration | Partial | The explicit bundle/layer/model/eager bridges prove contracts but are deliberately not the final serving path. |
| Pascal GPU experts | Planned | PXA/PXQ-derived `sm_61` work has not reached a validated P4 path. |
| `topk=10` router | Planned | Correct Torch fallback exists; fused Pascal path remains issue #38. |
| Static hot expert cache | Planned | Data structures and acceptance contract exist in issue #21. |
| Dynamic cache / `q*` | Planned | Architecture is sound; no P4 locality/overlap evidence yet. |
| Dual GPU | Planned | ADR 0010 now requires ownership-policy comparison rather than assuming TP. |
| Long-context state | Partial reference work | Required H3/H4 state/save/restore evidence remains. |
| Serving/operations | Upstream base plus downstream plans | Production cancellation, health, metrics, Compose, and fault qualification remain. |

## External model and quant review

### Q4 remains the quality reference

The Unsloth `UD-Q4_K_XL` artifact remains the appropriate high-quality baseline. It is large enough that exact host-memory accounting matters, but field reports show that full CPU-MoE placement can retain most decode throughput on a strong GPU while freeing VRAM for context and the active trunk.

The project should preserve Q4 as the reference profile even if it is not the eventual throughput default.

### A named Q3 profile is now a first-class throughput candidate

A pinned `UD-Q3_K_XL` deployment on one GB10 reported an approximately 90 GB artifact, roughly 20 tok/s decode at a 32K prompt, and exact outputs versus its paired baseline in a small coding/tool quality corpus. Its separate exact n-gram sweep is also the best reproducible coding-specific speculation evidence currently available.

This does not prove Q3 quality or speed on P4, but it is strong enough to replace the vague “some 3-bit comparison” language. The repository should pin one Q3 artifact, produce its complete census, and qualify it beside Q4.

### Imatrix provenance matters

Community GGUFs with published calibration/imatrix provenance are preferable to opaque requants. AtomicChat's Qwen3.8 artifacts and associated discussion are particularly useful as an evidence source because they document architecture-specific importance matrices and a dedicated PLE/table shard approach.

A third-party quant does not become trusted merely because it is popular. The release process still requires immutable revision, checksums, tensor census, tokenizer/template verification, and the project's own routing/tool/long-context gates.

### Q5 and Q8 are component formats, not required whole-model profiles

A full Q5 or Q8 model is unlikely to fit the 128 GB host envelope with safe operating headroom. The useful question is whether selected continuously active or sensitive tensors benefit from Q5/Q8 while routed experts use a lower-bit CPU/P4 format.

The format matrix must distinguish:

- whole-model fit candidates such as Q4 and the selected Q3;
- expert-bank CPU formats;
- GPU-cache formats;
- router/control/shared/trunk tensor formats;
- PLE row codecs.

### MTP exists but should not block v1

Standalone and built-in Qwen3.8 MTP artifacts now exist, including low-bit heads. Current model cards and discussions also warn that several runtime/head layouts are incompatible and must not be mixed.

MTP is worth tracking after ordinary decode is correct, but it remains a post-v1 or optional experiment until:

- one pinned runtime/head pair is defined;
- state and verification semantics are proven;
- P4 verification kernels are measured;
- quality and fallback behavior are established.

### Pruned expert variants remain secondary

REAP/MEP-style reduced-expert artifacts demonstrate that storage can be reduced with modest benchmark loss, but they alter the model and do not reduce the configured top-k active experts automatically. They are not aligned with the current full-model North Star and remain post-v1 experiments.

## Runtime and community evidence

### Upstream llama.cpp support is now merged

Qwen3.8-Flash-Next support is no longer only an experimental llama.cpp PR. PR #27742 merged into llama.cpp, including text, QSA, vision, conversion, and required quantizer fixes. The merged implementation should now be the principal external GGUF correctness and operational comparator.

### FreeToken Qwen4 work is pinned but closed unmerged

FreeToken PR #232 closed without merging. Its head remains a valuable pinned source and produced substantial 3090/4090/5090 validation, including 256K context and tool/coding checks. The downstream fork is therefore responsible for maintaining and validating its adapted Qwen4 code rather than assuming upstream will absorb it.

### Static hot-expert residency is a serious baseline

A current llama.cpp residency PR splits each layer's routed experts into hot GPU and cold CPU branches and reports approximately 1.6× decode improvement at equal VRAM on Qwen3.8 Q4_K_XL on a 5090, with identical output and unchanged prefill.

The exact number does not transfer to P4, but the result validates two project choices:

- expert-level placement is more valuable than placing arbitrary complete expert layers;
- a trace-derived static hot profile should be measured before claiming value from a dynamic cache.

Static placement is both a comparator and a warm-start seed for LFRU.

### CPU-MoE is a credible baseline

Multiple Qwen3.8 llama.cpp reports show that leaving all routed experts in system RAM can cost surprisingly little decode throughput on modern GPUs while substantially reducing VRAM use. This supports the project's mandatory cache-zero/CPU-backed baseline and the decision to reserve P4 VRAM for the active trunk, state, and only the hottest routed experts.

It does not prove the Haswell AVX2 path will be fast enough. The target-host CPU and contention measurements remain decisive.

### Exact context-derived speculation matches coding-agent work

The pinned `sxuff/qwen38-flash-next-dgx-spark` paired sweep compared ordinary decode with `ngram-mod` using identical requests, model, sampling, and output hashes. It reported:

- about 3.8× wall-clock improvement for copying Python;
- about 2.6× for copying JSON;
- about 1.9× for a structured transformation;
- approximately neutral behavior for novel code with no usable drafts;
- about 2.49× aggregate wall-clock improvement, with exact paired outputs.

This is one machine and one sweep, not a universal benchmark. It nevertheless maps directly to coding-agent edits, patches, repeated tool output, JSON, and configuration transformations. ADR 0011 therefore permits it as an optional v1 profile without making it release-critical.

### PLE access must be treated as random I/O

Large-context community experiments have reported dramatic read amplification when the kernel applies sequential readahead to sparse PLE access. Random-access advice reduced physical disk traffic by an order of magnitude in one reported deployment.

The exact result must be reproduced on the server's NVMe, but the implementation implication is low-risk:

- mmap should request random access where supported;
- positional reads should use equivalent file advice;
- physical block-device bytes must be compared with logical packed PLE bytes;
- dedupe/sort/coalesce should be adaptive, not unconditional for tiny decode batches.

### VRAM placement has a cliff, not a smooth curve

Hybrid model reports show severe throughput collapse when a placement change pushes the runtime beyond practical VRAM capacity, even if the model still starts and returns correct output. The project currently has strong memory accounting but no release-critical startup canary/backoff contract.

Issue #73 closes this reliability gap.

## Architecture assessment

### What should remain unchanged

- FreeToken remains the correct primary base for the expert-cache/co-execution research target.
- Upstream llama.cpp remains the external GGUF correctness and deployment comparator.
- The complete unpruned model and permanent CPU/cache-zero fallbacks remain mandatory.
- The three-tier architecture in ADR 0010 is the correct center:
  - dedicated PLE on NVMe;
  - complete routed experts in DDR4;
  - dense/shared/state/hot experts in P4 VRAM.
- Correctness and performance changes remain separable.
- Real P4 evidence remains required for all defaults.

### Clarifications and modifications

1. **Expert residency must not imply a second full anonymous copy.** The complete expert bank may be an explicitly managed file-backed resident serving mapping or another measured representation. Memory accounting must prevent simultaneous duplicate expert copies plus uncontrolled page cache from exhausting 128 GB.
2. **PLE gets a random-I/O contract.** Add random advice, read-amplification telemetry, adaptive vectorized planning, and an explicit row-codec boundary.
3. **Placement gets a safety gate.** Plan and observe every VRAM category, reserve headroom, run a startup canary, and automatically back off optional placement/cache slots.
4. **Q4 and Q3 become named profiles.** Q4 is the reference; the pinned Q3 is the whole-model throughput candidate. Q5/Q8 are evaluated at component scope where appropriate.
5. **Static hot experts precede dynamic claims.** Cache-zero, static-hot, async dynamic, and current-step hybrid modes must be compared independently.
6. **Dual-P4 policy stays evidence-driven.** Compare layer-owned, disjoint expert-owned, replicated, and trunk split/TP combinations. Do not force every layer's expert partials through cross-GPU traffic without measuring it.
7. **Exact n-gram speculation is optional v1 work.** It cannot delay or destabilize the core release. MTP remains later.

## Recommended execution order

### Before the P4s arrive

1. Complete issue #13's dedicated PLE artifact, mmap/`pread`, random advice, adaptive dedupe/sort/coalesce, read-amplification telemetry, and codec boundary.
2. Finish real Q4 and selected Q3 census/parity coverage through issue #17.
3. Finish the serving-ready host-expert integration that replaces the current correctness-only blocking bridge.
4. Implement the H0/H1 portions of issue #73: memory planner, placement profiles, canary schema, fallback state machine, and tests.
5. Prepare static hot-profile import/simulation and persisted-heat formats for issues #21/#22.
6. Keep the optional issue #74 independent so it cannot block the core path.

### When the first P4 is installed

1. Qualify airflow, power, clocks, PCIe link, and NUMA locality before model work.
2. Run the smallest deterministic Qwen fixture and kernel parity suite.
3. Establish cache-zero Q4 and Q3 placement envelopes with the new cliff guard.
4. Benchmark format-specific Pascal kernels and fused `topk=10` before dynamic cache scheduling.
5. Run static hot-expert placement and collect real routing locality.

### When both P4s are installed

1. Compare layer-owned, disjoint expert-owned, replicated, and trunk split/TP policies.
2. Enable asynchronous LFRU only after the static comparator is stable.
3. Add current-step `q*` co-execution only after contended CPU/PCIe/GPU timings are available.
4. Qualify long context, checkpoints, cancellation, serving, and deployment.
5. Run optional exact n-gram profiling only after ordinary state semantics are stable.

## Go/no-go criteria

Reassess the design if any of the following are observed on real hardware:

- the active trunk and required state cannot fit across the P4s with a safe reserve;
- the target Xeons cannot execute enough CPU misses to help any hybrid split;
- realistic expert-cache sizes show negligible locality over the static baseline;
- PCIe/merge overhead makes disjoint ownership slower than simple layer ownership;
- PLE physical I/O remains a critical-path bottleneck after random advice and batching;
- the selected Q3 fails routing/tool/long-context quality gates;
- dynamic cache or speculation cannot avoid regressions on negative-control workloads.

A failed optimization does not invalidate the product. The correct response is to preserve the last reliable fallback and change the default from measured evidence.