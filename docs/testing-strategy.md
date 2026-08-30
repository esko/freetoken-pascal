# Testing strategy

## Principles

- Test the smallest pure function before the complete model.
- Compare optimized kernels with an independent reference.
- Separate file-format correctness, model-graph correctness and performance.
- Treat coherent prose as insufficient evidence.
- Preserve failure-path and fallback tests.
- Hardware emulation does not replace P4 evidence.
- Treat a model that returns correct text through an unsafe spill/fallback configuration as a failed deployment test.
- Keep optional-profile failures independent from core ordinary-decode readiness.

## Gate levels

### H0 — hosted

Runs on GitHub-hosted Linux without a GPU:

- formatting and manifest validation;
- Python/unit tests;
- GGUF metadata and tensor-census tests;
- quant block-size/row-stride tests;
- pinned Qwen3.8 Q4/Q3 byte-range row references, with declared artifact hashes kept distinct from payload-verified evidence;
- bounded real-artifact expert probes that verify HTTP ranges and descriptor geometry, then retain raw scalar/native A/B timings and kernel telemetry plus separate comparisons of both outputs against the pinned gguf-py 0.19.0 dense FP32 SwiGLU oracle over the exact fetched bytes as H0 `range_evidence: measured/artifact-byte` (never P4 evidence);
- host simulators for CUDA kernel arithmetic where feasible;
- cache policy and q-star scheduler simulation;
- source-provenance checks;
- tiny model/config conversion;
- deterministic state serialization without CUDA;
- dedicated PLE shard format and separation from unrelated model tensors;
- mmap/positional-read parity, random-advice dispatch, direct/vectorized planner parity, adaptive planner selection, asynchronous prefetch and cancellation;
- deterministic page-cache, major-fault, block-I/O and read-amplification metric parsing with synthetic counters; `freetoken.ple_io_evidence` must reject process-only physical-I/O claims, counter regressions, device ambiguity/change, invalid phases and zero denominators;
- PLE row-codec identity and reconstruction tests;
- routing simulation, Q4/Q3 profile identity, quant conversion and mixed-precision quality A/B fixtures;
- placement-plan accounting, missing-category rejection, stale-profile invalidation, automatic-backoff state machine and canary result schemas;
- QSA workspace category accounting from concrete shapes, 64-bit overflow rejection, incomplete-group boundaries, and controlled pre-launch capacity errors;
- static-hot trace/profile generation, oracle hit-rate simulation and profile compatibility checks;
- optional exact n-gram proposal/index/state-machine tests that do not import or require the full model.

Issue #18 also provides a bounded host-resource lifecycle observation:

```text
make stress-host-resources STRESS_ARGS="--seed 18 --iterations 8 --threads 2 --allow-fallback"
```

The harness allocates one tiny pageable policy-owned bank set at a time and drives the production `Q4KExecutor` through an embedded deterministic Q4/mixed-GEMV primitive seam. It checks serial/thread parity, concurrent `Busy`, cancellation rollback/recovery, and close races. Its JSON records the seed, executor/primitive census, actual current-allocation policy accounting separately from cumulative allocations, accepted/busy/cancelled/recovered counts, and before/after resource observations. `claim_status` is always `observation_only`: RSS, swap, descriptor, and thread observations are descriptive, not pass/fail evidence, and this test makes no no-swap, NUMA, staging, performance, or P4 claim. The fallback flag is only for CPU-only H0 jobs; omit it on a Torch host to require the production HostBank path. Use an outer process timeout in CI or on a host. The RSS report names its per-iteration sample maximum `sampled_max_after_iteration`; it is not a live or process-lifetime peak.

### H1 — CUDA compilation

Runs in a CUDA 12.6 build container, GPU optional:

- compile every shipping CUDA translation unit for `sm_61`;
- reject accidental CUDA 13 or `sm_70+` instructions;
- inspect fatbins/architectures;
- compile external PXA extension;
- compile the fused `topk=10` router and every placement/canary telemetry path;
- run CPU host simulators against the same kernel tables;
- verify no required source is silently excluded;
- compile optional #74 verification kernels only when that feature is enabled, without making them a core dependency.

### H2 — single P4

- device kernel parity for every supported quant/shape;
- tiny model end-to-end generation;
- one-P4 Qwen3.8 short-context correctness for `reference-q4` and the named Q3 candidate where supported;
- independent PLE cold-cache, warm-cache, major-page-fault, physical-read/amplification and steady-state behavior for mmap and positional-read backends;
- PLE direct versus vectorized/adaptive planner behavior at decode, prefill and verification widths;
- file-backed PLE first/middle/last row parity, invalid index/range/hash/codec failures and page-fault/storage-read telemetry;
- cache size zero, static-hot and static-cache behavior;
- #73 placement sweep, canary, automatic backoff, hidden-fallback rejection and safe reserve;
- power, clock and thermal stability;
- invalid type/shape/bounds tests where safe;
- optional #74 paired ordinary/ngram tests only after core state correctness.

### H3 — dual P4

- layer-owned, disjoint expert-owned, replicated and trunk split/TP policy correctness/measurement where feasible;
- per-rank expert/source placement;
- no unintended cross-GPU expert transfer or double-counted replicated partials;
- NUMA policy comparison;
- static-hot, static, async-fill and hybrid scheduling;
- #73 per-rank placement/canary/backoff and failure recovery;
- 32K and 128K state correctness;
- cancellation and restart;
- deterministic fallback when one optimization is disabled;
- mixed-precision quality gates for routing, long-context retrieval, tool calls, structured output and coding;
- optional #74 exact-output, low-acceptance and PLE-I/O behavior across both cards.

### H4 — release

- model/profile/PLE checksum and quant census;
- benchmark suite against merged llama.cpp/PXQ reference;
- Q4/Q3 whole-model and component-format evidence;
- placement-cliff evidence and selected safety reserve;
- real fresh-code, edit, copy, long-context, tool and structured-output workloads;
- 8-hour soak;
- fault injection for OOM, placement overcommit, unexpected fallback, bad model/PLE metadata, failed fill and cancelled request;
- Docker/Compose clean deployment;
- reproducible build from pinned sources;
- optional #74 evidence only when that profile is published.

## Required reference comparisons

### Quant kernel

```text
packed kernel output
vs
dequantize to float + torch/reference matmul
```

Report maximum absolute error, relative RMS and cosine similarity. Use adversarial values as well as random tensors. The Issue #16 mixed direct GEMV tests exercise Q5_1 and Q8_0 down rows at input width 640 and Q5_K gate/up rows at input width 2560.

The CPU-only Qwen GGUF bridge tests cover host lifetime and idempotent close, exact mixed-layout and kernel-census preservation, explicit Torch/NumPy decode conversion, unsupported runtime modes, cache-size zero, and the engine's fail-closed homogeneous-cache guard. The standalone routed-layer adapter tests compare direct bundle execution with an independent full-softmax route reference across route widths, duplicate IDs, padded rows and mixed Qwen geometries, and cover shape, dtype, ID, TP and lifecycle failures. These tests remain CPU-only and do not imply full Engine or model-graph support.

The Qwen model attachment tests use fake layers and a borrowed bundle to prove complete prevalidation, all-layer adapter construction, middle-layer rollback, duplicate and closed/TP2 rejection, detach identity restoration, state-dict invariance, and caller-owned bundle lifetime. They do not claim full-model CPU execution or serving support.

The eager bridge tests use a fake transfer seam to prove direct CPU execution, ordered hidden/router and hidden/route copies, exactly one adapter invocation, output device and dtype restoration, output independence, observer propagation, pre-transfer rejection of prefill/group/graph/workspace modes, stale-telemetry clearing, and close-versus-in-flight admission. They are H0 seam tests only; blocking real-CUDA copies and serving integration remain H2-unverified. The eager model-attachment tests additionally prove explicit batch-derived execution context, ordinary expert call compatibility, shared-before-routed ordering, all-layer construction rollback, busy detach preflight, state-dict invariance, and closure of wrappers without closing the borrowed bundle. They also exercise a model forward/attach lock race, a direct bridge request arriving during multi-bridge admission freeze, close-failure rollback, and rejection of `load_state_dict` until detach. The resident originals are deliberately retained, so these tests do not claim memory savings or serving support.

### PLE storage operation

Compare on identical dedicated bytes and row sequences:

- source GGUF range versus extracted serving artifact;
- mmap versus positional read;
- direct versus vectorized planner;
- forced random advice versus unsupported/no-advice control where safe;
- cold versus warm versus steady page-cache state;
- IQ4_NL reference versus any candidate row codec.

Verify ordered outputs, duplicate rows, cross-shard requests, cancellation, short files, stale manifests and bounded scratch. Report logical packed bytes and actual block-device bytes; a planner test that observes only application reads cannot establish read-amplification behavior.

### MoE operation

Compare:

- all CPU;
- all eligible GPU;
- mixed CPU/GPU;
- cache size zero;
- static-hot cache;
- dynamic fill disabled/enabled;
- current-step fetch disabled/forced/automatic;
- same routed IDs and weights.

Before optimized CPU kernels are accepted, the model-agnostic expert ABI is tested with the pinned Qwen3.8 Q4 and Q3 census layouts, arbitrary and duplicate expert IDs, empty and maximum-width selections, padded rows, both router-weight placements, caller accumulation, cancellation rollback, malformed sources, workspace limits, and independent concurrent executors. Dense fixtures are compared with an independently written reference. Packed decoders must declare and stay within reusable scratch bounds; allocation-free claims apply only to paths measured by an allocation test.

CPU expert microbenchmarks retain every raw repeat and separately report the supplied matrix geometry, quant layout, token count, route/miss count, workspace size, and executor telemetry. Later performance comparisons must use identical inputs and may not infer an AVX2 speedup from the reference executor.

### Router operation

Adapt merged FreeToken PR #257's arbitrary-`K` fused router first, then compare it with the permanent FP32 Torch reference using the same 512 logits. Expert IDs are exact on no-tie inputs. Selected probabilities use the softmax denominator across all experts, with separate tests for renormalized weights, padded rows, ties and NaN/Inf behavior. H1 proves CUDA 12.6/`sm_61` build viability; H2 determines whether upstream Triton is usable and faster on P4. A bespoke CUDA implementation is required only if that evidence rejects Triton. Performance evidence records router-only time separately from complete layer and token time.

### Placement operation

Test the #73 planner and canary with synthetic and real categories:

- ordinary resident tensors;
- shared experts;
- GDN/KV recurrent state;
- explicit persistent/transient QSA state and workspaces;
- CUDA context and workspaces;
- transfer/partial buffers;
- static/dynamic cache slots;
- safety reserve.

Missing categories fail closed. Forced overcommit must trigger backoff or fail readiness, never a nominally healthy slow profile. Model output remains unchanged across safe backoff steps. Stale placement profiles are rejected on any model, quant, context, binary, driver or topology change.

The H0 placement fixture uses `freetoken.engine.placement_plan` with exact versioned categories, asymmetric one- and two-GPU capacities, cache-zero profiles, synthetic post-load and post-first-large-prefill observations, separate driver/allocator counters, allocator consistency checks, absolute live/peak agreement, reserve-bounded tolerance, and deterministic monotonic backoff.

### Model graph

Compare logits and selected internal state at:

- token 1;
- short prefill;
- recurrent continuation;
- state checkpoint/restore;
- dense-equivalent QSA region;
- sparse QSA region;
- context boundary transitions.

The comparison artifact is a deterministic ZIP containing a strict JSON identity manifest and non-pickle NumPy arrays. Both subject and independent reference identities must bind to the same model artifact hash, quant census, corpus hash, prompt hash, context-token count, quantization, and cache mode, including the pinned tokenizer repository and revision. The prompt hash covers the exact rendered chat-template bytes sent to the model. A different implementation label does not establish independence when revision and commit identities are identical. Non-finite JSON values, forged metric predicates, undeclared archive members, empty arrays, mutable workload substitutions, and malformed expectation contracts fail closed.

The H0 fixture suite validates this evidence protocol and the prompt materializer. It does not claim Qwen3.8 model parity. Issue #14 remains open until real-P4 H2 runs attach independent router, GDN, QSA, PLE, continuation-token, and selected-logit evidence for short, incremental, chunked, reset, checkpoint/restore, 32K, 128K, and 262K qualification cases.

Every mixed-precision candidate preserves exact routed expert IDs on the fixed routing corpus and meets recorded output tolerances for long-context retrieval, tool-call selection/arguments, structured JSON and coding. These gates compare against the declared reference profile and are reported separately from token-level numerical tolerances.

Converter fixtures independently check centered hyperconnection `1 + weight`, GDN fused-projection segmentation and value-head ordering, QSA/indexer projection splitting, PLE row scale interpretation, first/middle/last expert addressing, and tokenizer/chat-template identity. A community artifact that exposes a converter defect may serve as an oracle, but cannot become a release profile without immutable provenance and the complete artifact gates.

### Optional exact n-gram operation

When #74 is enabled, compare ordinary decode and speculation using identical request bytes, model/profile, sampling and placement. Verify:

- byte-identical deterministic outputs;
- no speculative state committed before verification;
- rejected/cancelled proposal state equals ordinary replay;
- context edit/truncation/checkpoint invalidation;
- bounded index memory;
- automatic disable on low acceptance;
- novel-code negative-control overhead;
- PLE vectorized-read semantics and physical I/O during multi-token verification.

This suite never gates core v1 when the optional profile is disabled.

## Test artifacts

Every hardware run uploads:

- inventory;
- exact command/environment;
- model/profile/PLE checksums and tensor census;
- placement plan/canary/backoff evidence;
- logs;
- metrics JSON;
- raw repeated-run samples;
- temperatures/clocks;
- failure artifacts.

Do not commit model weights, private prompts or secrets.

## Imported baseline at Issue #5

The initial FreeToken history import runs the 31 tests under `tests/daemon` as the CPU-safe, no-model-weight upstream baseline. They exercise daemon state, process management, accounting, logs, checkpoint metadata, and import safety without importing Torch or CUDA.

The remainder of the imported upstream suite is not deleted or reported as passing. It currently requires one or more of the upstream CUDA 13/Torch dependency set, GPU kernels, or external weights. Issue #7 owns the pinned CUDA 12.6/Python 3.12 environment, and issue #8 owns classification of the full suite into H0, H1, and deferred H2-H4 jobs plus a source-wide lint baseline. Tests requiring real P4 execution remain blocked under issues #9 and #29.
