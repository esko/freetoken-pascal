# Testing strategy

## Principles

- Test the smallest pure function before the complete model.
- Compare optimized kernels with an independent reference.
- Separate file-format correctness, model-graph correctness and performance.
- Treat coherent prose as insufficient evidence.
- Preserve failure-path and fallback tests.
- Hardware emulation does not replace P4 evidence.

## Gate levels

### H0 — hosted

Runs on GitHub-hosted Linux without a GPU:

- formatting and manifest validation;
- Python/unit tests;
- GGUF metadata and tensor-census tests;
- quant block-size/row-stride tests;
- pinned Qwen3.8 Q4/Q3 byte-range row references, with declared artifact hashes kept
  distinct from payload-verified evidence;
- bounded real-artifact expert probes that verify HTTP ranges and descriptor geometry,
  then retain raw scalar/forced-AVX2 A/B timings and kernel telemetry as H0
  `range_evidence: measured/artifact-byte` (never P4 evidence);
- host simulators for CUDA kernel arithmetic where feasible;
- cache policy and q-star scheduler simulation;
- source-provenance checks;
- tiny model/config conversion;
- deterministic state serialization without CUDA.

### H1 — CUDA compilation

Runs in a CUDA 12.6 build container, GPU optional:

- compile every shipping CUDA translation unit for `sm_61`;
- reject accidental CUDA 13 or `sm_70+` instructions;
- inspect fatbins/architectures;
- compile external PXA extension;
- run CPU host simulators against the same kernel tables;
- verify no required source is silently excluded.

### H2 — single P4

- device kernel parity for every supported quant/shape;
- tiny model end-to-end generation;
- one-P4 Qwen3.8 short-context correctness;
- PLE warm/cold behavior;
- file-backed PLE first/middle/last row parity, invalid index/range/hash failures and
  page-fault/storage-read telemetry;
- cache size zero and static-cache behavior;
- power, clock and thermal stability;
- invalid type/shape/bounds tests where safe.

### H3 — dual P4

- TP/layer ownership correctness;
- per-rank expert/source placement;
- no unintended cross-GPU expert transfer;
- NUMA policy comparison;
- static, async-fill and hybrid scheduling;
- 32K and 128K state correctness;
- cancellation and restart;
- deterministic fallback when one optimization is disabled.

### H4 — release

- model checksum and quant census;
- benchmark suite against llama.cpp/PXQ reference;
- real coding prompts and edit workloads;
- 8-hour soak;
- fault injection for OOM, bad model metadata, failed fill and cancelled request;
- Docker/Compose clean deployment;
- reproducible build from pinned sources.

## Required reference comparisons

### Quant kernel

```text
packed kernel output
vs
dequantize to float + torch/reference matmul
```

Report maximum absolute error, relative RMS and cosine similarity. Use adversarial values as well as random tensors.
The Issue #16 mixed direct GEMV tests exercise Q5_1 and Q8_0 down rows at input width 640 and Q5_K gate/up rows at input width 2560.

The CPU-only Qwen GGUF bridge tests cover host lifetime and idempotent close, exact mixed-layout and kernel-census preservation, explicit Torch/NumPy decode conversion, unsupported runtime modes, cache-size zero, and the engine's fail-closed homogeneous-cache guard.

### MoE operation

Compare:

- all CPU;
- all eligible GPU;
- mixed CPU/GPU;
- cache size zero;
- current-step fetch disabled;
- same routed IDs and weights.

Before optimized CPU kernels are accepted, the model-agnostic expert ABI is tested with
the pinned Qwen3.8 Q4 and 3-bit census layouts, arbitrary and duplicate expert IDs,
empty and maximum-width selections, padded rows, both router-weight placements, caller
accumulation, cancellation rollback, malformed sources, workspace limits, and
independent concurrent executors. Dense fixtures are compared with an independently
written reference. Packed decoders must declare and stay within reusable scratch bounds;
allocation-free claims apply only to paths measured by an allocation test.

CPU expert microbenchmarks retain every raw repeat and separately report the supplied
matrix geometry, quant layout, token count, route/miss count, workspace size, and
executor telemetry. Later performance comparisons must use identical inputs and may not
infer an AVX2 speedup from the reference executor.

### Router operation

Compare the fused Pascal router with the permanent FP32 Torch reference using the same
512 logits. Expert IDs are exact on no-tie inputs. Selected probabilities use the
softmax denominator across all experts, with separate tests for renormalized weights,
padded rows, ties and NaN/Inf behavior. Performance evidence records router-only time
separately from complete layer and token time.

### Model graph

Compare logits and selected internal state at:

- token 1;
- short prefill;
- recurrent continuation;
- state checkpoint/restore;
- dense-equivalent QSA region;
- sparse QSA region;
- context boundary transitions.

The comparison artifact is a deterministic ZIP containing a strict JSON identity manifest and
non-pickle NumPy arrays. Both subject and independent reference identities must bind to the same
model artifact hash, quant census, corpus hash, prompt hash, context-token count, quantization, and
cache mode, including the pinned tokenizer repository and revision. The prompt hash covers the exact
rendered chat-template bytes sent to the model. A different implementation label does not establish
independence when revision and commit identities are identical. Non-finite JSON values, forged
metric predicates, undeclared archive members, empty arrays, mutable workload substitutions, and
malformed expectation contracts fail closed.

The H0 fixture suite validates this evidence protocol and the prompt materializer. It does not claim
Qwen3.8 model parity. Issue #14 remains open until real-P4 H2 runs attach independent router, GDN,
QSA, PLE, continuation-token, and selected-logit evidence for short, incremental, chunked,
reset, checkpoint/restore, 32K, 128K, and 262K qualification cases.

## Test artifacts

Every hardware run uploads:

- inventory;
- exact command/environment;
- model checksum and tensor census;
- logs;
- metrics JSON;
- raw repeated-run samples;
- temperatures/clocks;
- failure artifacts.

Do not commit model weights, private prompts or secrets.

## Imported baseline at Issue #5

The initial FreeToken history import runs the 31 tests under `tests/daemon` as the CPU-safe, no-model-weight upstream baseline. They exercise daemon state, process management, accounting, logs, checkpoint metadata, and import safety without importing Torch or CUDA.

The remainder of the imported upstream suite is not deleted or reported as passing. It currently requires one or more of the upstream CUDA 13/Torch dependency set, GPU kernels, or external weights. Issue #7 owns the pinned CUDA 12.6/Python 3.12 environment, and issue #8 owns classification of the full suite into H0, H1, and deferred H2-H4 jobs plus a source-wide lint baseline. Tests requiring real P4 execution remain blocked under issues #9 and #29.
