# ADR 0011: Evidence-Gated Performance Profiles and Placement Safety

- Status: Accepted
- Date: 2026-08-29
- Amended: 2026-08-30

## Context

The three-tier storage decision in ADR 0010 establishes where PLE, routed experts, and hot computation belong. Recent Qwen3.8 field results add six constraints that are not fully captured by storage placement alone:

1. Sparse PLE reads can trigger severe kernel readahead and read amplification unless the runtime explicitly communicates random-access intent and measures physical I/O.
2. Hybrid GPU placement can have a sharp cliff: one additional layer, workspace, context allocation, or cache increment can silently trigger fallback or spill and collapse throughput while outputs remain correct.
3. QSA selection and top-k/gather workspaces can grow or become retained during the first large prefill, so post-load free VRAM does not establish a safe serving envelope. Context-dependent host synchronization and selection cost can also dominate long-context decode.
4. Whole-model bit width does not predict speed. Q4 is the quality reference, a specific Q3 artifact is a serious whole-model throughput candidate, and Q5/Q8 are primarily component-level candidates for sensitive or continuously active tensors on this 128 GB host.
5. Qwen3.8-Flash-Next is highly non-uniform in quantization sensitivity. Tiny router, shared-expert gating, recurrent-state and residual/control tensors can affect every token and every layer; a scaling or precision error in a few hundred KiB can catastrophically damage long reasoning while short prompts, retrieval probes and throughput still look healthy. Recurrent GDN control projections can also accumulate small numeric errors across many decode steps.
6. Exact context-derived n-gram speculation has produced large, output-identical gains on copy-heavy code and structured transformations while remaining neutral on a novel-code control. Native MTP also exists, but its artifacts and runtime contracts are still fragmented and its value on Pascal is unproven.

The North Star remains a reliable, highly performant, full-model Qwen3.8-Flash-Next server on the dual-Xeon, dual-P4 target. No optimization may hide a slower fallback, change model topology, or become a release dependency without evidence from the actual hardware.

## Decision

### Deployment profiles

FreeToken-Pascal defines independently reproducible profiles rather than one prematurely fixed quant recipe:

- `reference-q4`: the pinned `UD-Q4_K_XL` artifact, cache-zero and ordinary decode available, used as the principal quality and correctness reference.
- `throughput-q3`: one pinned, checksummed Q3 artifact selected from a real tensor census and benchmarked as a whole-model fit candidate. It may become the operational default only after mixed-precision quality and P4 throughput gates pass.
- `candidate-ap-q4`: Agention's `AP-Q4_K_XL` is an evidence candidate because its published artifact is smaller than the current Q4 reference while retaining standard llama.cpp tensor types. It does not replace `reference-q4` until its immutable identity, tensor census, tokenizer/template, converter provenance, quality, CPU cost, and P4 behavior pass downstream gates.
- `candidate-ap-iq4`: Agention's `AP-IQ4_XS` is an aggressive fit candidate under the same gates. Published perplexity alone is not a selection criterion.
- `coding-ngram`: an optional exact context-derived speculative profile layered on the winning base profile. It never blocks the core v1 release and automatically falls back to ordinary decode.

Q5, Q8, and other high-precision formats are benchmarked for the tensors or components where they are plausible. The project does not require an impossible full-model Q8 profile to fit in 128 GB merely to satisfy the format matrix.

### Sensitive-tensor precision island

Quantization policy is architecture-aware rather than a whole-model bitrate switch. Routed expert matrices and the PLE table may be compressed aggressively, but control/state tensors never inherit the routed-expert quant tier implicitly.

Every candidate profile carries a machine-readable sensitive-tensor census and explicit precision decision for at least:

- MoE router tensors and router-adjacent scaling;
- `shared_expert_gate` tensors and their quantization scales;
- Gated DeltaNet state-driving/control projections, including `in_proj_a` / `in_proj_b`-class tensors where present in the reconciled graph;
- residual/hyperconnection write gates and other continuously active control tensors;
- normalization and other small tensors shown by reference comparison to be numerically sensitive.

The correctness baseline preserves these tensors at their authoritative source precision or the nearest lossless runtime representation. A release candidate may reduce an individual class to Q8 or another format only after tensor-level scale/dequant parity and paired model-level quality evidence show that the change is safe. Sub-8-bit control/state tensors are experimental by default and may not be selected merely because the surrounding expert bank is Q2/Q3/Q4.

The loader must preserve and validate quantization scales for sensitive tensors independently from payload bytes. Sanity checks include finite/range checks and reference comparison of representative gate/control outputs so a missing or misapplied scale fails loudly rather than appearing as a valid fast model.

The quality gate is deliberately capable of detecting stateful failures that short retrieval can miss. Mixed-precision promotion therefore requires long-horizon reasoning/coding/tool trajectories and repeated-state tests in addition to first-token logits, perplexity-like metrics, needle retrieval, or short deterministic prompts. A deliberately degraded sensitive-tensor fixture must fail the gate, proving that the gate is sensitive to this failure class.

### PLE random-I/O contract

The dedicated PLE serving artifact remains pageable and independently addressed. The mmap backend uses random-access advice where supported (`MADV_RANDOM`); the positional-read backend uses the equivalent file advice where supported (`POSIX_FADV_RANDOM`). Advice selection and failures are observable.

The lookup planner may deduplicate, sort, and coalesce requests, but it must be adaptive. Tiny decode lookups may bypass sorting when planner overhead exceeds expected I/O benefit; wide prefill or speculative-verification batches use the vectorized path. The benchmark contract records logical bytes requested, unique packed bytes, application read bytes, block-device bytes, major faults, and read-amplification ratio.

IQ4_NL remains the initial reference codec. One lookup contract must support the public experiment matrix: BF16 mmap as the precision control, FP8 per-row, INT4 group-16, NVFP4-style group-16, and near-lossless Q6/Q8 candidates. These names describe codecs to qualify, not accepted artifacts or Pascal winners. No alternative becomes a default without immutable provenance, byte/layout validation, PLE reconstruction, row-decode and transfer measurements, and model-level quality gates.

### Placement-cliff and QSA workspace safety

Every release-capable configuration has a per-GPU placement plan and a measured startup canary. The planner accounts separately for resident tensors, shared experts, recurrent/QSA/KV state, persistent and transient QSA score/top-k/gather workspaces, CUDA context, generic workspaces, transfer buffers, expert-cache slots, and a configurable safety reserve.

Readiness requires observed allocation and selected-kernel telemetry to agree with the plan after both model load and a representative large-prefill canary. Unexpected fallback, managed-memory use, host spill, repeated allocation recovery, unbounded or unexplained retained workspace growth, or insufficient headroom causes automatic cache/context/batch/placement backoff and a repeated canary. The server fails readiness if no safe profile passes. Release defaults retain a measured margin below the post-prefill cliff rather than using the last configuration that merely loads or starts.

QSA selection, gather, sparse-attention and state-update costs are measured separately across context tiers. Workspace exhaustion produces controlled backoff or request/readiness failure, not process abort. Reusable workspaces are bounded and may be captured/reused only where dynamic PLE/QSA state semantics remain correct.

Merged FreeToken PR #257 is the first implementation donor for QSA scoring, exact block top-k, selected-row expansion/gather, sparse attention, and reusable scratch. Pascal qualification compares an upstream-derived Triton path, a small CUDA 12.6 path when needed, and the permanent Torch FP32 reference. QSA H0/H1 work and the first P4 context-depth sweep precede dynamic-cache performance qualification.

### Router implementation order

Merged FreeToken PR #257's arbitrary-`K` fused router is the first donor for Qwen's exact top-10-of-512 route. Issue #38 first adapts and validates that implementation for CUDA 12.6 and `sm_61`. A bespoke CUDA router is written only if Triton cannot support Pascal correctly or loses the controlled P4 benchmark. The permanent full-softmax Torch reference remains available and observable in every case.

### Expert-cache evidence sequence

Before adaptive admission is judged, the runtime measures a static hot-expert profile derived from a trace. This is both a baseline and a possible warm-start seed. Dynamic LFRU, asynchronous fills, and current-step CPU/GPU splitting must beat or safely fall back to cache-zero and static-hot controls.

Dual-P4 testing compares at least:

- layer-owned expert caches;
- disjoint expert ownership;
- replicated caches where capacity permits;
- trunk layer split or tensor parallelism combined with the expert-cache policies.

No policy is selected from topology intuition alone, and no routed expert may bounce between P4s in the release path.

### Speculation policy

Exact context-derived n-gram speculation is permitted as an optional v1 profile because it uses the same target model for verification and can preserve deterministic output. It receives its own correctness, state, PLE-I/O, low-acceptance, and paired-performance gates. Ordinary decoding remains the required fallback and the core release does not wait for this profile.

Native MTP, external draft models, DFlash, and lossy speculative prefill remain post-v1 experiments until artifact compatibility and Pascal measurements are stable. MTP artifacts must be tracked in an explicit compatibility matrix and may not be mixed across incompatible runtime layouts.

## Consequences

- Issue #13 gains random-access advice, read-amplification telemetry, adaptive vectorized lookup, and an explicit PLE codec boundary. SSD residency remains PLE-specific in the v1 steady-state path; routed-expert SSD execution remains an explicitly labeled experiment rather than a fallback.
- Issue #17 treats a named Q3 whole-model artifact as the throughput candidate while benchmarking Q5/Q8 at component scope where full-model fit is impossible, and now owns the sensitive-tensor precision census and promotion gates.
- Issue #17 adds AP-Q4_K_XL and AP-IQ4_XS as gated candidates. A reported static-IQ4_XS converter oracle remains excluded until its immutable source and conversion log are independently verified.
- Converter regression coverage includes centered hyperconnection `1 + weight`, GDN fused-projection segmentation/head ordering, QSA/indexer projection splitting, PLE row scale interpretation, first/middle/last expert addressing, tokenizer/chat-template identity, and independent scale handling for sensitive control/state tensors.
- Issue #14 must include long-horizon semantic and agentic probes plus a degraded-sensitive-tensor positive control; short retrieval success alone cannot establish correctness.
- Issue #18 must bind host expert-bank/staging placement to measured PCIe/NUMA topology and compare local-node versus cross-socket access before selecting the H3 policy.
- Issue #26 must stress recurrent state over long trajectories and checkpoint/restore boundaries so cumulative numerical/state corruption is observable.
- Issue #38 adapts the merged upstream arbitrary-`K` router first and creates a Pascal CUDA fallback only when measured evidence requires it.
- Issue #21 uses static hot-expert placement as a mandatory comparator and warm-start source before dynamic cache claims.
- Issue #73 makes placement planning, post-load/post-prefill high-water accounting, canary execution, backoff, and fail-readiness release-critical.
- Issue #76 makes QSA context scaling, workspace bounds, synchronization and controlled-OOM behavior explicit release work.
- Issue #76 is P3 critical-path work and precedes dynamic-cache performance claims.
- Issue #74 adds exact context-derived speculation as optional, output-preserving work that cannot block core v1.
- Issue #29 reports placement cliffs, QSA context scaling, PLE read amplification, Q4/Q3 whole-model profiles, sensitive-tensor precision decisions, component-format tests, long-horizon quality gates, and all relevant fallbacks.
- Field measurements from modern GPUs and unified-memory systems are evidence for what to test, not performance predictions for the P4 server.

## Alternatives considered

Use the smallest quant as the default: rejected because decoder kernels and dequantization cost can reverse the apparent bit-width ordering.

Apply one quant tier uniformly to every tensor: rejected because tiny continuously active control/state tensors can dominate semantic correctness while contributing negligible storage. Compression budget is spent on routed experts and PLE first; control/state precision is reduced only from evidence.

Fill VRAM until allocation fails: rejected because spill and fallback can occur before a visible OOM and may produce a healthy but unusably slow service.

Trust post-load free VRAM: rejected because first-large-prefill QSA/top-k/gather workspaces can create a later high-water mark or OOM.

Always sort PLE IDs: rejected because decode-width planner overhead can exceed its benefit; the policy is adaptive and measured.

Make n-gram speculation release-critical: rejected because the base server must remain complete for novel generation and because Pascal verification performance is not yet known.

Adopt MTP immediately: rejected because multiple incompatible heads and runtime patches exist, and no P4 evidence currently justifies coupling it to v1.
