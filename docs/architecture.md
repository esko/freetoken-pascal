# System architecture

## Design objective

Maximize delivered token throughput for Qwen3.8-Flash-Next on two 8 GB Pascal GPUs without pruning experts or forcing the complete model into VRAM.

The architecture treats the server as three explicit storage and execution tiers:

```text
NVMe
  ├─ dedicated contiguous PLE/N-gram file or shard set
  └─ immutable GGUF model and expert source files
        │ mmap faults or sorted positional reads
DDR4, NUMA-owned
  ├─ complete quantized routed-expert bank
  ├─ Linux page cache for frequently accessed PLE rows
  ├─ bounded staging only where explicitly configured
  └─ AVX2 CPU expert executor
        │
        ├──────────── PCIe H2D ────────────┐
        │                                  │
P4 #0 VRAM                            P4 #1 VRAM
  ├─ dense latency-critical tensors      ├─ dense latency-critical tensors
  ├─ shared experts and runtime state    ├─ shared experts and runtime state
  ├─ adaptive hot routed-expert cache    ├─ adaptive hot routed-expert cache
  └─ CUDA workspaces                     └─ CUDA workspaces
```

PLE is a normal v1 NVMe-backed input, not emergency backing, while the complete expert bank is a DDR4-resident source for CPU execution and cache fills.
The P4s receive only dense latency-critical tensors, shared experts, runtime state, and bounded hot routed-expert cache entries.
N-gram/PLE in this document names the required table and lookup substrate; it does not include speculative decoding, which remains outside v1.

## Model partition

### NVMe tier

- dedicated contiguous PLE/N-gram file or shard set;
- immutable GGUF model and expert source files;
- independently identifiable PLE offsets and manifest identity.
- one immutable row-codec descriptor and registry identity/version; IQ4_NL v1 is the only
  accepted codec, while future row formats must retain the same storage/lookup contract.

### DDR4 tier

- complete quantized routed-expert bank;
- Linux page cache for frequently accessed PLE rows;
- CPU reference and AVX2 expert execution;
- bounded staging buffers and cache heat/telemetry state.

The complete expert bank is loaded and pre-faulted into DDR4 for v1, is covered by the release no-swap policy, and supplies all steady-state CPU expert execution and GPU cache fills. SSD-backed expert reads are startup backing or an explicitly labeled experiment only.
PLE mappings remain pageable by default so the OS can consume spare RAM dynamically, and the full PLE table is never permanently pinned.

### GPU tier

- dense latency-critical tensors, including embeddings/output when placement is beneficial;
- QSA, GDN, hyperconnections, routers, and shared experts;
- recurrent, QSA, KV, and other runtime state;
- adaptive fixed-address cache slots for hot routed experts;
- transfer staging and partial-output buffers.

Pascal DP4A integer kernels and format-specific tuning are prioritized over FP16, BF16, and FP8 paths until actual P4 measurements justify otherwise.

### QSA workspace contract

`freetoken.attention.qsa_workspace` is the Torch-free H0 accounting boundary for the merged QSA implementation from FreeToken commit `bd8f3d519a48777bf22ee5c7c8f58f4f3ff31b40` and its current tip `58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6`.
`QSAWorkspaceInputs` accepts the concrete context, ragged token-row, request `batch_size`, raw
page-table token-slot width, page, ring, head, dtype, and layer dimensions.
The planner derives the page count and compressed score columns from the raw page-table width and
uses the runtime's 128 MiB FP32 score-tile cap; it never treats token slots as already-compressed
columns.
Capture plans separately accept `capture_max_batch_size` because every graph buffer is allocated
for that maximum, while eager metadata uses the active request batch.
`calculate_qsa_workspace()` inventories the score, top-k, expand-gather, attention, and state categories from those dimensions and returns component shapes, byte totals, and a checked aggregate without importing CUDA or allocating memory.
The score category includes the index query, FP32 logits, and visible-block vectors.
The top-k category includes block IDs and the candidate scratch required by the upstream split path.
The expand-gather category includes the selected logical-index rows, including the incomplete causal-group tail.
The attention category includes the output and, when split attention is selected, FP32 partial output and log-sum-exp buffers.
The state category includes the persistent compressed slab, pending ring, and index RoPE table plus
the per-forward pooled rows, first-position rows, and `QSASparseMetadata.cmp_rows`/
`ring_rows` scatter plans. The latter are `[token_rows]` int32 buffers and remain live from index
compression through selected-row attention, including graph capture.
Eager high-water accounting overlaps actual batch metadata with each live phase: pooled/index rows,
selection (`q_index`, retained indices, and one score/top-k chunk), and selected-row attention.
The selected-row phases also retain both per-forward scatter plans.
Capture high-water accounting includes every `_graph` buffer simultaneously and the active capture
attention allocations, including the active capture batch's scatter plans.
The eager Torch top-k fallback is accounted separately for its Python-visible column arange,
visibility mask, values, chosen indices, validity mask, int32 cast, and `where` output. PyTorch
allocator fragmentation and opaque kernel-internal top-k workspace cannot be observed by this
Torch-free planner, so the resulting Torch estimate is conservative but not an exact guarantee.
CUDA graph capture requires the Triton top-k backend; a Torch top-k request is rejected as an
eager-only plan rather than being reported as capture-safe.
`QSAWorkspacePlan.validate_capacity()` is a pure preflight accounting primitive for a future placement/launch owner and reports structured `ready` or `insufficient-capacity` telemetry.
Negative, zero-invalid, incomplete-category, shape-inconsistent, and checked 64-bit arithmetic inputs fail closed with a controlled `ValueError` subtype.
This H0 contract makes no kernel, throughput, or Tesla P4 claim and does not change QSA selection, token budget, or dispatch defaults.

The Torch-free `freetoken.engine.placement_plan` module is the H0 owner for the per-GPU placement schema and startup-canary decision contract.
It keys each rank by stable GPU UUID plus rank and keeps dense/resident weights, shared experts, GDN/KV recurrent state, every persistent and transient QSA phase category, CUDA context, generic workspaces, transfer buffers, static and dynamic cache slots, and safety reserve explicit.
Placement inputs require explicit non-placeholder UUIDs, reject duplicate cards, and permit zero observed free bytes while requiring positive physical capacity.
QSA state is represented only by the explicit `qsa_persistent_state` and `qsa_transient_state` categories; it is not folded into the GDN/KV recurrent-state bucket.
Its required high-water is non-QSA demand plus QSA persistent bytes plus QSA transient high-water bytes plus reserve, so lifetime phases are not double-counted.
The planner and canary evaluator are immutable, versioned, allocation-free accounting only; they do not claim runtime placement or hardware qualification.
Canary observations compare allocator allocated bytes with the live non-QSA plus persistent-QSA demand and allocator high-water with the live plus transient-QSA peak, while keeping driver total/free and allocator reserved bytes separate.
They reject unavailable or inconsistent samples before readiness.
The bounded backoff remains pending until both post-load and post-first-large-prefill pass for the same profile; a failure resets checkpoint evidence before advancing, and readiness is never inferred from a boolean.

The exact ordinary-tensor placement is measured. The architecture requires the routed expert bank to remain complete in DDR4 even when a subset is cached in VRAM.
The PLE file or shard set has a separate ownership and accounting boundary so model-weight mappings cannot be accidentally paged, prefetched, or pinned with it.

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
The NUMA intent is a bounded policy hook; physical placement remains disabled by default and is only attempted for the explicit mapping-scoped enforcement option described below.
The CLI and `EngineConfig` construct and validate the policy directly.
Engine serving currently accepts only explicit `pinned` FTW policy; `pageable` and `bounded-staging` remain preflight-only until their serving transfer paths are wired, and unsupported custom-provider or dummy paths reject an explicit policy instead of silently dropping it.
The opt-in `require_no_swap` guard reads `/proc/self/status`, `/proc/meminfo`, and `/proc/swaps` before policy-owned allocation or FTW index reads. It reports process `VmSwap`, system totals, active devices, probe source/errors, raw `swap_status`, and `no_swap_observed` (`true`, `false`, or `null` when unavailable). Active swap, non-zero process swap, or an unavailable/ambiguous probe fails closed; the guard never runs `swapoff`, changes sysctls, or treats `mmap`/`mlock` behavior as proof of no swap. This is a point-in-time admission check, not a full-model residency guarantee.

NUMA placement is a separate opt-in H0 slice. `enforce_numa_placement` resolves online nodes intersected with the process `Mems_allowed_list` and applies Linux x86_64 `mbind` only to policy-owned FTW `MAP_PRIVATE|MAP_ANONYMOUS` writable mappings, before first touch. `preferred` with no node has no explicit target and records a fallback without issuing `mbind`; preferred/interleave also record a fallback when the placement syscall is unavailable or fails. `bind` requires a selected node, fails closed, and rolls back the allocation prefix. The implementation does not call `set_mempolicy`, use migration flags, change system settings, bind worker threads, or claim complete model residency. An optional bounded self-only `move_pages` sample is reported as `verified`, `partial`, or `unavailable` with counts and unknown/error details.

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
Worker startup timeout is terminal: the native pool reports `timed-out`, rejects
later workload submission before task/barrier entry, and the Python serving wrapper
fails construction. Affinity errors that complete startup remain usable via the
unpinned host-function fallback. Runtime task exception handling is unchanged by
this slice and is not an affinity guarantee. The native five-second wait bounds
startup-readiness reporting only; teardown still joins native threads, with the
H1 process timeout providing the outer protection. Teardown leaves completion
flags unchanged so incomplete output is never advertised as ready.

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
Target-host swap state remains an operational observation rather than a hardware
claim: if a Gorilla inventory records any active swap device or non-zero `VmSwap`,
`require_no_swap` is expected to reject the explicit host-bank load. No swap device
state is inferred from the topology fixture or from allocation behavior.

Issue #16 adds a Torch/CUDA-independent Q4_K adapter at this boundary. It decodes the
GGML 144-byte, 256-value block layout and uses direct packed-row GEMV for compatible
gate, up, or down descriptors, with a scalar implementation retained as the reference
and runtime fallback. The optional native helper is loaded from
`FREETOKEN_Q4K_NATIVE_LIB` or the package extension after AVX2 and FMA capability
checks; it is split into
baseline dispatch, baseline scalar, and an explicitly AVX-512/AMX-disabled
`-mavx2 -mfma` translation unit so the shipping baseline does not require newer
instructions. A descriptor is eligible only when its
quant type and quant name both identify Q4_K and its input width and packed row stride
match complete 256-value blocks; inconsistent or partial packed descriptors fail closed.
The Q4 artifact remains heterogeneous: layer 2 gate/up uses Q5_K and the promoted
down banks use Q5_1 or Q8_0. The separate `mixed_gemv` primitive now provides direct
packed-row GEMV for those three formats, with format-tagged AVX2 dispatch and the
same scalar reference fallback. Its optional helper is loaded from
`FREETOKEN_MIXED_GEMV_NATIVE_LIB` or the package extension. The H0 adapter
dispatches each packed projection to its format-specific primitive and retains
the dense/dequantize ABI oracle. Q4_K, Q5_1 and Q5_K AVX2 kernels expand packed nibbles
and high bits directly into vector lanes, while Q8_0 sign-extends packed int8 codes
directly into AVX2 lanes. Q8_0 and Q5_K GEMV inline their dot bodies to avoid a call
boundary for each packed block. Every packed GEMV still reduces and accumulates one
quant block at a time in the reference order. An opt-in `num_threads > 1` runner is created
only when the selected layer has native AVX2 coverage for all three projections
and valid census geometry. It partitions route columns into private worker
requests, reduces partials in partition order, and commits once on the owner
thread. Scalar helpers, unavailable native libraries, unsupported geometries,
and partial layers remain serial. The runner owns a persistent pool, prepared
private buffers, and a bounded public-result ring; omitted caller output is
returned from that prepared ring only after success, so results never alias
worker scratch. When the standalone GGUF bridge receives a positive
`num_threads`, it resolves the shared cpuset-aware `WorkerPlan` and passes it
to this runner. Only an internally owned pool gets worker-local singleton
affinity requests; each worker reads its mask back before the report can say
`verified`. A pin or readback error drains that request and reruns the serial
reference path with `fallback` affinity telemetry. Direct `Q4KExecutor` callers
without a plan retain the existing pool behavior, and caller-supplied pools
cannot be combined with an explicit plan. This is H0 affinity verification
only: it changes no owner mask and makes no NUMA or performance claim.

The standalone `QwenGGUFCpuExpertBundle` owns a `QwenGGUFHostWeights` mapping, builds the exact heterogeneous `CpuExpertLayout`, and owns a `Q4KExecutor` for decode-only CPU use.
Its Torch adapter accepts only CPU tensors, copies through explicit NumPy float32/int32 buffers, and returns a CPU tensor in the hidden-state dtype.
Its bridge-local thread policy resolves an omitted request or `num_threads=0` to one
serial worker, and rejects a positive request above the physical-core count visible
through the process affinity.  The bundle reports the requested and effective counts,
the actual participating partitions from the last decode, the selected kernel census,
and any threading fallback reason; this policy does not reinterpret the existing
`EngineConfig.moe_cpu_threads` value.  The physical-core check is an admission guard
only. For a positive request, the Q4 runner also reports requested and observed
worker CPUs, per-worker affinity errors, and `planned-unverified`, `verified`,
`not-applicable`, or `fallback` status. Only its internally owned pool is pinned;
the bridge never changes the owner mask or claims NUMA placement.
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

Hardware qualification first tests disjoint expert ownership across the two P4s, including correctness, transfer, load balance, and recovery behavior.
Conventional tensor parallelism is evaluated only after that policy has evidence and remains disabled until its communication cost is measured on the actual PCIe topology.
No dual-P4 release default exists until H3 compares the candidates and records one reproducible choice. The ordering and selection gate supersede ADR 0007's initial contiguous-layer default. The selected policy must not require an expert to bounce between P4s.

## PLE design

- PLE is a core v1 host operation whose normal source is a dedicated contiguous NVMe file or shard set.
- The PLE manifest records file identity, contiguous logical row geometry, shard boundaries, and byte ranges independently from GGUF model weights.
- The loader rejects overlapping or ambiguous PLE ranges and never maps, pins, or warms unrelated model-weight bytes as part of a PLE operation.
- The lookup API supports both an mmap/page-fault backend and a positional-read (`pread`) backend with the same ordered result contract.
- Each batch is deduplicated by row ID, sorted by physical offset for I/O, and restored to caller order after reads complete.
- Asynchronous prefetch is bounded, cancelable, and observable; it may warm Linux page cache but does not permanently pin the full table.
- Selected IQ4_NL rows dequantize in DDR4 and only the bounded result transfers to the execution device.
- Cold-cache, warm-cache, major-page-fault, and steady-state decode behavior are separate benchmark dimensions with independent counters and reporting.
- Cold, OS-readahead, targeted-row, and explicit full-model warm modes are distinct and observable; full-model warm is a measurement mode, not a residency requirement.

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

No final quantization recipe is claimed before Q4, Q5, Q8, and CPU-format candidates are benchmarked on the actual P4 hardware.
