# ADR 0010: Three-Tier PLE, Expert Bank, and P4 Residency

- Status: Accepted
- Date: 2026-08-29

## Context

The target server has substantially more NVMe and DDR4 capacity than either Tesla P4's 8 GB of VRAM.
The PLE/N-gram table is large, sparse in access, and latency-sensitive when it is consulted during decoding.
The complete quantized routed-expert bank must remain available for CPU execution and cache fills, while only a hot subset can fit in VRAM.
Treating all files as one pageable or pinned pool would allow unrelated model weights to be faulted or pinned with PLE and would make cache and benchmark results ambiguous.
Pascal hardware also favors integer DP4A kernels and format-specific tuning, but the P4s are not yet installed.

## Decision

FreeToken-Pascal v1 uses three explicit storage and execution tiers:

1. NVMe is the normal home of the separately sharded N-gram/PLE table and its immutable source files.
2. DDR4 holds the complete quantized expert bank and provides the Linux page cache for frequently accessed PLE rows.
3. The two P4 VRAM spaces hold dense latency-critical tensors, shared experts, runtime state, and bounded adaptive caches of hot routed experts.

Here, N-gram/PLE names the model's required lookup table and its storage substrate. It does not add the n-gram speculative-decoding algorithm, which remains outside v1.

The PLE table uses a dedicated contiguous file or shard set with an independently identifiable byte range and manifest entry.
PLE mappings and reads must not include unrelated model-weight ranges, and PLE pinning must never implicitly pin those ranges.
The v1 PLE path supports both mmap/page-fault and positional-read backends behind one lookup contract.
Lookup scheduling batches requests, removes duplicate row IDs, sorts physical reads, and supports asynchronous prefetch without changing result order.
PLE buffers remain pageable by default, and the operating system may use spare DDR4 for its page cache; v1 does not permanently pin the full PLE table.

PLE benchmarks report cold-cache, warm-cache, major-page-fault, and steady-state behavior independently.
The complete expert bank is loaded and pre-faulted into a DDR4 serving allocation, remains covered by the no-swap runtime policy, and is the only steady-state source for CPU execution and GPU cache fills. File-backed expert bytes may be used during startup, while SSD-backed expert execution is limited to explicitly labeled experiments.
Pascal DP4A integer kernels and format-specific Q4, Q5, and Q8 tuning take priority over unmeasured FP16, BF16, or FP8 paths.
Before P4 availability, quantization work may convert and compare candidates, but no final quantization recipe is selected until Q4, Q5, Q8, and CPU-format candidates are benchmarked on the actual P4s.
Dual-P4 qualification first tests disjoint expert ownership and its correctness/transfer behavior before conventional tensor parallelism is considered. H3 evidence must compare the candidates and record one reproducible release default; no default is selected before that gate. This policy-selection order supersedes ADR 0007's initial contiguous-layer default.

## Consequences

The loader, manifest, cache, and benchmark harness must keep PLE identity, offsets, residency, and cache state separate from GGUF model weights.
The PLE API must expose backend choice, batch deduplication, sorting, prefetch activity, page-fault counters, and cache temperature in logs and metrics.
The host expert bank consumes DDR4 and requires startup/runtime residency and no-swap evidence, but it is not permanently pinned and is not a reason to reserve all remaining RAM for pinned pages.
VRAM allocation must make dense tensors, shared experts, runtime state, and adaptive routed-expert slots observable independently.
CPU and host simulation work can proceed at H0/H1, while DP4A tuning, disjoint ownership policy selection, and conventional tensor-parallel comparisons remain H2/H3 work.
The decision adds file-format and benchmark work to the v1 critical path, but it prevents storage aliasing and produces evidence that can distinguish I/O, page-cache, PCIe, and compute costs.

## Alternatives considered

Treat PLE as optional emergency backing: rejected because PLE is a normal v1 decode dependency and its latency must be measured as a first-class path.
Pin the full PLE table: rejected because it consumes scarce host pin budget and couples PLE residency to unrelated model files.
Execute experts directly from SSD: rejected as a v1 default because it adds unpredictable I/O to every routed operation; it remains startup backing or an experiment.
Use conventional tensor parallelism first: rejected because the two P4s lack NVLink and disjoint expert ownership offers a lower-traffic, independently testable first policy.
Choose a final quantization format before P4 testing: rejected because host CPU behavior and Pascal DP4A behavior can rank Q4, Q5, Q8, and CPU formats differently.
