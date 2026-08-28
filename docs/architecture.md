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

The CPU expert boundary is model-agnostic. An immutable layout supplies layer and
projection identities, expert count and top-k, matrix geometry, native quant type,
row/expert strides, and the bounded mapped source span. Callers prepare a fixed-size
workspace before execution, then submit routed IDs and weights either as one request or
as an explicitly ordered group. The executor returns a partial hidden-state contribution
that may be accumulated by the caller. Invalid IDs, unsupported layouts, cancellation,
and workspace overflow fail closed and publish telemetry without committing partial
output.

The dense/dequantize executor is the permanent CPU correctness oracle. Its packed
decoder may use separately declared bounded scratch; production backends must not
allocate per token or route. Thread-pool and NUMA objects are policy hooks rather than
implicit global state. The Issue #15 reference executor is serial, so later parallel
backends must preserve request isolation, cancellation, accumulation, and telemetry
semantics when those hooks become active.

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
