# Product scope

## Product statement

FreeToken-Pascal v1 is a downstream FreeToken runtime optimized for **text-only Qwen3.8-Flash-Next inference on a dual NVIDIA Tesla P4 server**. It is designed as a three-tier system where NVMe holds the dedicated PLE/N-gram shards, DDR4 holds the complete quantized expert bank and the Linux page cache for hot PLE rows, and the GPUs accelerate dense latency-critical work plus a dynamically selected hot subset of routed experts.

The product is not a generic inference platform. Its v1 purpose is to maximize reliable throughput on the owner's actual server without changing the model topology or sacrificing experts.

## Reference hardware

- Fujitsu PRIMERGY RX2530 M1
- 2 × Intel Xeon E5-2673 v3, 24 physical cores / 48 threads total
- approximately 128 GB ECC DDR4, 8 memory channels across two NUMA nodes
- 2 × Tesla P4, 8 GB each, Pascal `sm_61` — **not installed yet**
- 1 TB PCIe M.2 NVMe for immutable model files and the dedicated PLE/N-gram file or shard set
- Ubuntu Server 26.04 LTS
- CUDA 12.6-class toolchain required for Pascal
- no NVLink; PCIe/NUMA topology must be measured after installation

## In-scope v1 capabilities

### Model

- Qwen3.8-Flash-Next text backbone
- Gated DeltaNet, QSA, hyperconnections, PLE and routed/shared experts
- native tokenizer and chat template
- deterministic state reset and restore
- 32K required validation context
- 128K release validation context
- 262K native-context qualification when memory and current upstream correctness permit; failure to qualify must be documented, not hidden

### Model storage and quantization

- GGUF/K/I weight loading
- pinned Unsloth `UD-Q4_K_XL` as the `reference-q4` quality and correctness profile
- one pinned, checksummed `UD-Q3_K_XL` or equivalent census-verified Q3 artifact as the `throughput-q3` whole-model candidate
- heterogeneous quant types across ordinary tensors and expert layers
- expert gate/up and down banks may use different quant types
- Q5, Q8, and other high-precision formats benchmarked at component scope for sensitive or continuously active tensors; a full-model Q8 fit is not required on the 128 GB host
- dedicated contiguous PLE/N-gram shards as a core v1 path, with explicit mmap and positional-read backends
- random-access file advice, batched/adaptive deduplication and offset ordering, bounded asynchronous prefetch, and read-amplification telemetry
- Linux page-cache residency for frequently accessed PLE rows without permanently pinning the full table
- an explicit PLE row-codec identity, with IQ4_NL as the initial reference and alternative codecs gated by reconstruction and model-quality evidence
- exact tensor census and model checksums recorded for every benchmark
- actual-P4 whole-model and component-format benchmarks before any final release recipe is selected

### Execution

- CPU-only or cache-disabled reference mode
- AVX2 low-bit CPU routed-expert execution
- one-P4 bring-up mode
- dual-P4 release mode
- measured comparison of layer-owned, disjoint expert-owned, replicated, and trunk split/TP policies
- fixed-address per-GPU expert cache
- trace-derived static hot-expert profile as a mandatory comparator and possible warm-start seed
- LFRU or measured successor policy
- asynchronous cache fill
- concurrent CPU and GPU execution for one MoE operation
- contention-aware split of current misses
- separate decode and prefill policies
- NUMA-aware host placement and worker pools
- per-GPU placement planning, safety headroom, deterministic startup canary, automatic backoff, and fail-readiness behavior for unsafe or cliff configurations
- automatic fallback when hybrid execution is slower or unsafe
- optional exact context-derived `coding-ngram` speculative profile that preserves target-model verification semantics and never blocks the core v1 release

### Serving and observability

- OpenAI-compatible text API
- streaming, cancellation and bounded request resources
- one primary interactive request; limited concurrency may be supported but is not the primary optimization target
- metrics for cache hits, misses, evictions, CPU/GPU expert counts, transfer bytes/times, scheduler decisions, NUMA placement, PP/TG and memory
- independent PLE metrics for cold cache, warm cache, major page faults, steady-state latency, backend, logical/unique rows, physical bytes, read amplification, batching, sorting and prefetch
- explicit storage-tier and residency metrics for NVMe PLE, DDR4 expert bank/page cache, and P4 dense/shared/runtime/hot-cache allocations
- placement-plan versus observed-allocation telemetry and startup-canary evidence
- optional n-gram metrics for proposed/accepted tokens, span lengths, verification cost, PLE access and low-acceptance fallback
- reproducible benchmark commands and machine-readable results
- Docker/Compose deployment compatible with the server's standard stack
- crash-safe logs and actionable unsupported-configuration errors

## Out of scope for the core v1 release

- vision tower and image serving
- Windows support
- GLM-5.3 or other model families
- native MTP, DFlash, external draft models, lossy speculative prefill, and mandatory speculative decoding
- distributed multi-node serving
- expert parallelism across more than two local P4s
- training, fine-tuning, REAP pruning or topology changes
- a custom quantization research program beyond formats required for the target profiles and component matrix
- public SaaS multi-tenancy
- replacing upstream FreeToken's general architecture

These may be added only after v1 release or by an accepted ADR that changes scope and backlog. The optional `coding-ngram` profile is permitted by ADR 0011 but cannot delay, weaken, or replace the ordinary-decode core release.

The required N-gram/PLE table and lookup substrate described above are part of the base Qwen3.8 model path. Exact context-derived speculation is a separate optional feature and remains disabled unless its own evidence gates pass.

## Release success

The project is successful when the release criteria in `release-criteria.md` pass on the target server and the full system can be rebuilt from a clean checkout using pinned sources and documented commands. Optional profiles are published only when their additional criteria pass; an unfinished optional profile does not block v1.0.
