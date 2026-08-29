# System architecture

## Design objective

Maximize delivered token throughput for Qwen3.8-Flash-Next on two 8 GB Pascal GPUs without pruning experts or forcing the complete model into VRAM.

The architecture treats the server as a memory hierarchy:

```text
NVMe
  └─ GGUF and PLE backing files
        │ mmap/page faults
Host RAM, NUMA-owned
  ├─ complete low-bit routed expert bank
  ├─ PLE page cache
  ├─ bounded pinned staging or pinned expert regions
  └─ AVX2 CPU expert executor
        │
        ├──────────── PCIe H2D ────────────┐
        │                                  │
P4 #0 VRAM                            P4 #1 VRAM
  ├─ owned trunk layers                 ├─ owned trunk layers
  ├─ recurrent/QSA state                ├─ recurrent/QSA state
  ├─ fixed expert slots                 ├─ fixed expert slots
  └─ CUDA workspaces                    └─ CUDA workspaces
```

## Model partition

### CPU/NVMe tier

- complete routed-expert weights;
- PLE table;
- source-of-truth low-bit weights for cache fills;
- CPU reference and AVX2 expert path;
- cache heat and telemetry state.

### GPU tier

- embeddings/output when placement is beneficial;
- QSA, GDN, hyperconnections, routers and shared experts;
- recurrent, QSA and KV state;
- fixed-address cache slots for hot routed experts;
- transfer staging and partial-output buffers.

The exact ordinary-tensor placement is measured. The architecture requires the routed expert bank to remain complete on the host even when a subset is cached.

## GGUF ingestion

The GGUF reader resolves a complete immutable shard set before exposing a tensor. It
keeps the native per-tensor type and row stride, including heterogeneous expert
projections, and records the source shard and absolute payload offset. Uniform fused
projections may share one packed buffer; mixed types keep independent buffers. The full
contract, selected artifact identities and census commands are in
[`gguf-qwen38.md`](gguf-qwen38.md).

Host expert descriptors group only identical `(projection, quant type, shape, row stride)`
geometries into a slot pool. Source and slot offsets are bounds-checked independently.
The backing mappings are private and exposed read-only, remain unpinned, and retain the
original artifact as their source of truth.

The Issue #18 host-bank policy provides an explicit pre-load gate for FTW checkpoints.
Its omitted CLI/config value remains `None` and preserves legacy loader behavior, while an explicit `pageable` strategy leaves every source mapping in page-backed host memory and cannot consume the CUDA pin quota.
The explicit `pinned` strategy registers only its selected layers after fill and rejects a page-rounded source size above `max_pinned_bytes` before FTW reads or allocation.
The explicit `bounded-staging` strategy keeps all sources pageable and allocates a fixed `HostStagingRing` only within `max_staging_bytes`.
Plans and startup accounting expose source bytes, requested and applied pinned/staging bytes, per-layer residency, and the selected NUMA intent.
The NUMA intent is a bounded policy hook in this slice and does not claim physical placement until target-host validation.
The CLI and `EngineConfig` construct and validate the policy directly.
Engine serving currently accepts only explicit `pinned` FTW policy; `pageable` and `bounded-staging` remain preflight-only until their serving transfer paths are wired, and unsupported custom-provider or dummy paths reject an explicit policy instead of silently dropping it.

The CPU expert boundary is model-agnostic. An immutable layout supplies layer and
projection identities, expert count and top-k, matrix geometry, native quant type,
row/expert strides, and the bounded mapped source span. Callers prepare a fixed-size
workspace before execution, then submit routed IDs and weights either as one request or
as an explicitly ordered group. The executor returns a partial hidden-state contribution
that may be accumulated by the caller. Invalid IDs, unsupported layouts, cancellation,
and workspace overflow fail closed and publish telemetry without committing partial
output.

`Busy` is a rejection before request ownership, not a failed execution. It carries
telemetry but does not mutate the supplied output buffer because that buffer may alias
the request already executing. Callers must treat output as unreadable unless execution
returns successfully.

The dense/dequantize executor is the permanent CPU correctness oracle. Its packed
decoder may use separately declared bounded scratch; production backends must not
allocate per token or route. Thread-pool and NUMA objects are policy hooks rather than
implicit global state. The Issue #15 reference executor is serial, so later parallel
backends must preserve request isolation, cancellation, accumulation, and telemetry
semantics when those hooks become active.

The compiled `CpuMoeExecutor` now consumes this policy boundary directly. It
validates the requested count against the discovered process mask before building
host-bank pointer tables or the native pool; `resolve_threads_and_affinity()` remains
available for benchmark compatibility and is not the serving executor's resolver.
The native pool validates each requested singleton mask and reads it back at startup,
including the optional flag-sync coordinator. Python reports `planned-unverified`
until that report says `verified`, and reports `fallback` on a native error. This is
CPU affinity telemetry only: it makes no NUMA placement or performance claim.

The H0 CPU topology foundation is the Torch-free `freetoken.moe.cpu_topology` module.
`discover_cpu_topology()` reads the process affinity mask and injectable Linux sysfs
topology data, groups only allowed SMT siblings, and reports immutable physical-core
representatives with `full`, `partial`, or `logical-only` confidence, source, and
fallback reason. `WorkerAffinityPolicy` produces an immutable `WorkerPlan` with exact
worker CPU IDs, requested/effective thread counts, optional coordinator reservation,
and an observable `planned-unverified` affinity status. Its default `mask` partition
uses the discovered process mask as authoritative; `contiguous` and `numa` rank
partitions require explicit selection. No affinity is changed by this module, and a
NUMA plan is an intent only until target-host enforcement and placement validation.

On 2026-08-29, the target-host Gorilla observation at source commit
`e6a28d06a5ec6d76745730254b89a5b697fa5e53` reported two Xeon E5-2673 v3 sockets, 48
online logical CPUs, SMT pairs `0/24` through `23/47`, and a process/cgroup mask of
`0-47`. The descriptive sysfs result was 24 representatives (`0-23`), with node-0
representatives `0-11` and node-1 representatives `12-23`; `numactl --hardware`
reported local/remote distances of 10/21. This is H0 host-topology evidence only and
does not claim worker affinity enforcement, memory placement, P4 availability, or
performance.

Issue #16 adds a Torch/CUDA-independent Q4_K adapter at this boundary. It decodes the
GGML 144-byte, 256-value block layout and uses direct packed-row GEMV for compatible
gate, up, or down descriptors, with a scalar implementation retained as the reference
and runtime fallback. The optional native helper is loaded from
`FREETOKEN_Q4K_NATIVE_LIB` or the package extension after AVX2 and FMA capability
checks; it is split into
baseline dispatch, baseline scalar, and `-mavx2 -mfma` translation units so the shipping
baseline does not require newer instructions. A descriptor is eligible only when its
quant type and quant name both identify Q4_K and its input width and packed row stride
match complete 256-value blocks; inconsistent or partial packed descriptors fail closed.
The Q4 artifact remains heterogeneous: layer 2 gate/up uses Q5_K and the promoted
down banks use Q5_1 or Q8_0. The separate `mixed_gemv` primitive now provides direct
packed-row GEMV for those three formats, with format-tagged AVX2 dispatch and the
same scalar reference fallback. Its optional helper is loaded from
`FREETOKEN_MIXED_GEMV_NATIVE_LIB` or the package extension. The H0 adapter
dispatches each packed projection to its format-specific primitive and retains
the dense/dequantize ABI oracle. An opt-in `num_threads > 1` runner is created
only when the selected layer has native AVX2 coverage for all three projections
and valid census geometry. It partitions route columns into private worker
requests, reduces partials in partition order, and commits once on the owner
thread. Scalar helpers, unavailable native libraries, unsupported geometries,
and partial layers remain serial. The runner owns a persistent pool, prepared
private buffers, and a bounded public-result ring; omitted caller output is
returned from that prepared ring only after success, so results never alias
worker scratch.

The standalone `QwenGGUFCpuExpertBundle` owns a `QwenGGUFHostWeights` mapping, builds the exact heterogeneous `CpuExpertLayout`, and owns a `Q4KExecutor` for decode-only CPU use.
Its Torch adapter accepts only CPU tensors, copies through explicit NumPy float32/int32 buffers, and returns a CPU tensor in the hidden-state dtype.
Its bridge-local thread policy resolves an omitted request or `num_threads=0` to one
serial worker, and rejects a positive request above the physical-core count visible
through the process affinity.  The bundle reports the requested and effective counts,
the actual participating partitions from the last decode, the selected kernel census,
and any threading fallback reason; this policy does not reinterpret the existing
`EngineConfig.moe_cpu_threads` value.  The physical-core check is an admission guard
only; it does not pin workers or claim NUMA placement.
It rejects GPU, hybrid, offload, nonzero-cache, prefill, grouped, and closed-mapping requests before execution.
The CUDA `Engine` registration seam fails closed for Qwen GGUF rather than constructing the homogeneous `OffloadMoeCache`.
The standalone `QwenGGUFCpuMoELayer` adapts one layer's bundle to the existing routed-expert
interface for H0 CPU decode probes. It accepts explicit CPU router logits or a precomputed
CPU route, preserves full-softmax and observer semantics (Qwen's default is no selected-
route renormalization), and returns the bundle's CPU result without creating a cache.
The adapter requires an explicit decode phase and a single request group. It is an explicit test/reference seam only: it is not
attached implicitly during Qwen model construction, does not transfer CUDA tensors, and
does not enable the serving Engine. An explicit
`Qwen4ExpModel.attach_gguf_cpu_expert_bundle()` (and the matching
`Qwen4ExpForCausalLM` delegate) can transactionally replace every layer's routed expert
object for H0 construction and lifecycle tests. It validates all layer IDs, dimensions,
expert counts, top-k, activation, router-weight placement, and TP1 before mutation;
detach restores the exact objects and never closes the caller-owned bundle. The
model's CPU adapter does not make its CUDA-oriented trunk, router, shared expert, or
LM head CPU-runnable, and the Engine registration guard remains in place.

The standalone `GGUFCpuEagerBridge` is an explicit, model-neutral H0/H1 seam for a
caller that already owns a Qwen GGUF CPU layer. Every call must name `phase="decode"`
and uses `group_size=1`, TP1 and `cache_size=0`; prefill, grouped work, caller
workspaces and CUDA graph capture fail before any transfer. CPU tensors call the
adapter directly. Device tensors use an injected blocking transfer seam (or the
blocking `Tensor.to` default) to copy hidden states and either router logits or prepared
routes to CPU, execute the adapter exactly once, then copy the independent routed result
back to the original device and dtype. The seam has no nonblocking, pinned-memory,
stream, overlap or performance contract. Request-scoped telemetry reports the transfer
path, copied fields/bytes, one CPU execution, adapter telemetry and errors; a failed
request clears the prior success. Bridge close only rejects new work and never closes
the borrowed layer or bundle. This is an experimental correctness boundary: real
CUDA-transfer and serving evidence remain H2-unverified. An explicit
`Qwen4ExpModel.attach_gguf_cpu_eager_bridge()` (and its `ForCausalLM` delegate) can
transactionally install one bridge per routed-expert layer. The attachment derives
phase and request group from the active batch, rejects prefill, grouped work and graph
capture before expert execution, and exposes per-layer bridge telemetry. Detach restores
the resident expert identities and closes only attachment-created wrappers; bridge
admission is frozen across the whole detach transaction and both freezes and closes are
rolled back if a bridge is busy or close fails. Model forward mode selection and
`load_state_dict` are serialized with that lifecycle lock; loading while an attachment is
active fails with a detach-before-load error. The borrowed bundle and transfer seam remain
caller-owned. The resident originals are retained for state-dict fidelity, so this mode is
not memory-saving or serving-ready. Engine, CLI and default paths remain unchanged.

## MoE decode operation

For each layer and decode step:

1. Router produces selected expert IDs and weights.
2. Cache map partitions selections into:
   - resident GPU hits;
   - current-step GPU fetch candidates;
   - CPU execution candidates.
3. GPU hits run on the owning P4.
4. Selected misses are copied and run on the P4.
5. Remaining misses run through the AVX2 executor.
6. GPU and CPU branches execute concurrently where dependencies allow.
7. Partial outputs are accumulated consistently with the reference semantics.
8. Cache policy updates recency/frequency and admits completed fills.
9. Metrics record the decision and measured costs.

The Qwen3.8 router selects 10 of 512 experts on every MoE layer. Its permanent
`torch-reference` path computes the full FP32 softmax across all 512 logits before
returning the selected weights. A `pascal-fused` path may select by logits, but it must
preserve that complete denominator, both renormalization modes, padded-row behavior and
the documented tie/NaN/Inf policy. Runtime logs expose the forced or `auto` path and
fallback reason. The fused path remains opt-in until same-workload P4 measurements show
an end-to-end improvement outside run-to-run noise.

For `m` misses and `q` current GPU fetches, the autotuner minimizes the measured critical path:

```text
T(q) = max(
  T_cpu(m - q),
  T_h2d_and_gpu(q)
) + T_merge
```

The scheduler uses tables measured under concurrent contention, not theoretical DRAM or PCIe bandwidth.

Router weights follow the upstream MoE contract exactly. With
`apply_router_weight_on_input=false`, the route weight scales the down-projection output.
With it enabled, the weight scales both gate and up projection outputs before the
activation. Because the activation is nonlinear, these modes are deliberately distinct
and reference tests cover both.

## Cache design

- fixed-size arenas; tensor addresses do not change;
- per-GPU layer ownership;
- no cross-GPU cache coherence in v1;
- a persistent expert-to-slot mapping;
- LFRU initial policy;
- minimum capacity or fairness per owned layer;
- nonblocking future-token fill as the safe first optimization;
- optional current-step fill after the autotuner is validated;
- persisted heat may seed startup, but runtime policy remains authoritative;
- cache size zero is a supported reference mode.

## Prefill design

Large prefill activates many experts and needs a different policy:

- group tokens by expert;
- deduplicate each expert per layer/chunk;
- stream bounded groups of expert weights;
- double-buffer transfer and GPU computation when VRAM permits;
- use CPU execution when transfer cannot be hidden;
- tune `batch` and `ubatch` separately;
- never force token-oriented LRU behavior on a forward wider than the cache.

## Dual-GPU design

Initial release topology:

```text
P4 #0 owns a contiguous first layer range
P4 #1 owns the remaining layer range
```

Each rank owns:

- its trunk layers;
- cache slots only for those layers;
- a CPU worker pool and host pages local to its NUMA node where possible;
- its transfer streams and metrics.

TP and layer ownership must not require an expert to bounce between P4s. Graph/layer split alternatives can be benchmarked, but v1 chooses one reproducible default.

## PLE design

- PLE remains a host operation and is never placed in P4 VRAM by default.
- The backing range is independently identifiable and warmable.
- Normal decode, speculative-style batched lookup, warm cache and cold cache are separate benchmarks.
- The implementation must not pin the full PLE table.
- Page-cache state is reported where practical.
- Selected IQ4_NL rows dequantize on the host and only the bounded result transfers to the
  execution device.
- Cold, OS-readahead, targeted-row and explicit full-model warm modes are distinct and
  observable.

## State and serving

A request state includes:

- GDN recurrent state;
- QSA KV/index/filter state;
- PLE rolling/hash state;
- hyperconnection streams;
- token positions and sampling state.

Checkpoints are created at semantic boundaries for coding-agent workflows. Restore must be deterministic against replay from the same boundary.

## Fallback hierarchy

1. optimized dual-P4 hybrid;
2. dual-P4 cache hits plus CPU misses;
3. one-P4 cache plus CPU misses;
4. CPU expert bank with GPU trunk;
5. CPU reference/tiny-model validation.

Unsupported quant, shape or topology must fail before generation or select an explicit slower path. Silent use of uninitialized output or an unreported fallback is forbidden.
