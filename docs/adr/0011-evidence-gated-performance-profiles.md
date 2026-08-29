# ADR 0011: Evidence-Gated Performance Profiles and Placement Safety

- Status: Accepted
- Date: 2026-08-29

## Context

The three-tier storage decision in ADR 0010 establishes where PLE, routed experts, and hot computation belong. Recent Qwen3.8 field results add four constraints that are not fully captured by storage placement alone:

1. Sparse PLE reads can trigger severe kernel readahead and read amplification unless the runtime explicitly communicates random-access intent and measures physical I/O.
2. Hybrid GPU placement can have a sharp cliff: one additional layer, workspace, context allocation, or cache increment can silently trigger fallback or spill and collapse throughput while outputs remain correct.
3. Whole-model bit width does not predict speed. Q4 is the quality reference, a specific Q3 artifact is a serious whole-model throughput candidate, and Q5/Q8 are primarily component-level candidates for sensitive or continuously active tensors on this 128 GB host.
4. Exact context-derived n-gram speculation has produced large, output-identical gains on copy-heavy code and structured transformations while remaining neutral on a novel-code control. Native MTP also exists, but its artifacts and runtime contracts are still fragmented and its value on Pascal is unproven.

The North Star remains a reliable, highly performant, full-model Qwen3.8-Flash-Next server on the dual-Xeon, dual-P4 target. No optimization may hide a slower fallback, change model topology, or become a release dependency without evidence from the actual hardware.

## Decision

### Deployment profiles

FreeToken-Pascal defines independently reproducible profiles rather than one prematurely fixed quant recipe:

- `reference-q4`: the pinned `UD-Q4_K_XL` artifact, cache-zero and ordinary decode available, used as the principal quality and correctness reference.
- `throughput-q3`: one pinned, checksummed Q3 artifact selected from a real tensor census and benchmarked as a whole-model fit candidate. It may become the operational default only after mixed-precision quality and P4 throughput gates pass.
- `coding-ngram`: an optional exact context-derived speculative profile layered on the winning base profile. It never blocks the core v1 release and automatically falls back to ordinary decode.

Q5, Q8, and other high-precision formats are benchmarked for the tensors or components where they are plausible. The project does not require an impossible full-model Q8 profile to fit in 128 GB merely to satisfy the format matrix.

### PLE random-I/O contract

The dedicated PLE serving artifact remains pageable and independently addressed. The mmap backend uses random-access advice where supported (`MADV_RANDOM`); the positional-read backend uses the equivalent file advice where supported (`POSIX_FADV_RANDOM`). Advice selection and failures are observable.

The lookup planner may deduplicate, sort, and coalesce requests, but it must be adaptive. Tiny decode lookups may bypass sorting when planner overhead exceeds expected I/O benefit; wide prefill or speculative-verification batches use the vectorized path. The benchmark contract records logical bytes requested, unique packed bytes, application read bytes, block-device bytes, major faults, and read-amplification ratio.

IQ4_NL remains the initial reference codec. The PLE interface identifies its row codec explicitly so near-lossless Q6/Q8 or other row formats can be evaluated without changing lookup semantics. No alternative becomes a default without PLE reconstruction and model-level quality gates.

### Placement-cliff safety

Every release-capable configuration has a per-GPU placement plan and a measured startup canary. The planner accounts separately for resident tensors, shared experts, recurrent/QSA/KV state, CUDA context, workspaces, transfer buffers, expert-cache slots, and a configurable safety reserve.

Readiness requires observed allocation and selected-kernel telemetry to agree with the plan. Unexpected fallback, managed-memory use, host spill, repeated allocation recovery, or insufficient headroom causes automatic cache/placement backoff and a repeated canary. The server fails readiness if no safe profile passes. Release defaults retain a measured margin rather than using the last configuration that merely starts.

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

- Issue #13 gains random-access advice, read-amplification telemetry, adaptive vectorized lookup, and an explicit PLE codec boundary.
- Issue #17 treats a named Q3 whole-model artifact as the throughput candidate while benchmarking Q5/Q8 at component scope where full-model fit is impossible.
- Issue #21 uses static hot-expert placement as a mandatory comparator and warm-start source before dynamic cache claims.
- Issue #73 makes placement planning, canary execution, backoff, and fail-readiness release-critical.
- Issue #74 adds exact context-derived speculation as optional, output-preserving work that cannot block core v1.
- Issue #29 reports placement cliffs, PLE read amplification, Q4/Q3 whole-model profiles, component-format tests, and all relevant fallbacks.
- Field measurements from modern GPUs and unified-memory systems are evidence for what to test, not performance predictions for the P4 server.

## Alternatives considered

Use the smallest quant as the default: rejected because decoder kernels and dequantization cost can reverse the apparent bit-width ordering.

Fill VRAM until allocation fails: rejected because spill and fallback can occur before a visible OOM and may produce a healthy but unusably slow service.

Always sort PLE IDs: rejected because decode-width planner overhead can exceed its benefit; the policy is adaptive and measured.

Make n-gram speculation release-critical: rejected because the base server must remain complete for novel generation and because Pascal verification performance is not yet known.

Adopt MTP immediately: rejected because multiple incompatible heads and runtime patches exist, and no P4 evidence currently justifies coupling it to v1.