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

### MoE operation

Compare:

- all CPU;
- all eligible GPU;
- mixed CPU/GPU;
- cache size zero;
- current-step fetch disabled;
- same routed IDs and weights.

### Model graph

Compare logits and selected internal state at:

- token 1;
- short prefill;
- recurrent continuation;
- state checkpoint/restore;
- dense-equivalent QSA region;
- sparse QSA region;
- context boundary transitions.

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
