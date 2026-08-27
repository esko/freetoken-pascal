# Backlog work breakdown

GitHub issues are the execution source of truth. This document defines the complete work breakdown and dependency intent so the backlog can be audited for omissions.

## P0 — foundation

- Import upstream FreeToken history and establish the downstream sync model.
- Pin every upstream source and preserve licensing.
- Create CUDA 12.6/Python 3.12 development and build containers.
- Add tiny-model, host-simulator and benchmark fixtures.
- Complete hosted CI and H1 `sm_61` compile gates.
- Prepare, but do not run, H2/H3 self-hosted workflows.
- Define result schemas, quant census and provenance manifests.

## P1 — Pascal Qwen4 reference

- Integrate Pascal runtime fallbacks and CUDA 12.6 dependency selection.
- Integrate Qwen3.8/Qwen4 text architecture.
- Integrate generic GGUF K/I types.
- Support heterogeneous per-layer expert-bank quant types.
- Implement PLE mmap/offload and warm/cold controls.
- Establish short-context and long-context reference correctness.

## P2 — host experts

- Define a model-agnostic low-bit expert backend.
- Implement AVX2 Q4_K expert execution.
- Add the additional Q2/Q3/IQ types required by selected artifacts.
- Add parity, malformed-input and microbenchmark tests.
- Implement bounded pinning/staging and NUMA-local host banks.

## P3 — P4 cache and dual GPU

- Port/adapt PXA low-bit expert kernels to a FreeToken extension for `sm_61`.
- Add Qwen4 TP=2.
- Define deterministic layer/GPU/NUMA ownership.
- Implement fixed-address expert slot arenas.
- Implement static mixed CPU/GPU execution.
- Implement asynchronous LFRU fill and persisted heat.
- Add routing telemetry, cache simulation and oracle hit analysis.

## P4 — adaptive scheduler and prefill

- Implement current-step CPU/GPU miss splitting.
- Benchmark real concurrent CPU, H2D and GPU costs.
- Implement `q*` autotuning and runtime EMA adaptation.
- Add safe pure-CPU/fetch-all fallback.
- Add decode prefetch and batched copy submission.
- Implement prefill expert deduplication, grouping, chunked streaming and double buffering.
- Expose all decisions and timings through metrics.

## P5 — state, server and operations

- Validate GDN/QSA/PLE state at 32K, 128K and qualifying 262K.
- Add semantic state checkpoint/restore.
- Harden OpenAI-compatible streaming, cancellation and limits.
- Add Docker/Compose packaging.
- Add health, metrics, failure recovery and security controls.
- Complete user and operator documentation.

## P6 — hardware and release

- Install P4s, capture topology, power and thermal evidence.
- Configure self-hosted runner.
- Qualify one P4.
- Qualify dual P4 and NUMA policies.
- Benchmark against cache-zero FreeToken, llama.cpp and PXQ.
- Run real coding/edit workloads.
- Run soak and fault injection.
- Freeze pins, publish image/artifacts, tag v1 and close the epic.

## Completeness audit

The v1 backlog is incomplete if it lacks an item for any of:

- source provenance;
- loader/converter correctness;
- unsupported quant failure;
- CPU fallback;
- GPU kernel parity;
- one-GPU bring-up;
- two-GPU ownership;
- cache-zero path;
- current-step hybrid merge;
- prefill wider than cache;
- long-context state;
- cancellation;
- observability;
- topology/NUMA;
- thermal/power evidence;
- clean deployment;
- release reproducibility.

Post-v1 items such as vision, GLM and speculation must not be used to block release.
