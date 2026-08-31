# Hardware target and deferred validation

## Known host

| Component | Target |
|---|---|
| Server | Fujitsu PRIMERGY RX2530 M1 |
| CPU | 2 × Xeon E5-2673 v3 |
| ISA | AVX2, no AVX-512/AMX |
| NUMA | two CPU sockets |
| Memory | about 128 GB ECC DDR4 |
| Storage | 1 TB PCIe M.2 NVMe for models |
| GPU | 2 × Tesla P4, 8 GB, `sm_61` |
| Interconnect | PCIe, no NVLink |
| OS | Ubuntu Server 26.04 LTS |

Both P4 cards and the PCIe NVMe device are installed on Gorilla.
Passive inventory and bounded one-allocation CUDA arithmetic have passed on both cards through Torch 2.11.0 with CUDA 12.6.
ECC is currently enabled, yielding about 7,599 MiB of Torch-visible memory per card from the nominal 7,680 MiB.
The intended performance profile may disable ECC, but usable memory and correctness must be recaptured after the required reset rather than combined with the ECC-on evidence.
The cards are intentionally throttled while airflow optimization is incomplete, so sustained-load, thermal, link-under-load, performance, H2 completion and H3 completion remain unqualified.
The self-hosted runner is not yet configured.

## Tier contract

NVMe is the normal v1 home of the dedicated contiguous PLE/N-gram file or shard set, not emergency backing.
DDR4 holds the complete quantized expert bank and the Linux page cache for frequently accessed PLE rows.
Dual-P4 VRAM is reserved for dense latency-critical tensors, shared experts, runtime state, and adaptive hot routed-expert caches.
The full PLE table must remain pageable and must not be permanently pinned; spare DDR4 may be consumed dynamically by the OS page cache.
PLE reads must support mmap and positional-read backends, with batched, deduplicated, offset-sorted requests and bounded asynchronous prefetch.
PLE measurement separates cold-cache, warm-cache, major-page-fault, and steady-state behavior.
SSD expert execution is startup backing or an explicitly labeled experiment, while the complete expert bank remains in DDR4 for v1.

## Completed arrival intake

The pre-arrival H0/H1 work included:

- repository import and upstream pinning;
- CUDA 12.6 container and `sm_61` compilation;
- host simulators and CPU tests;
- Qwen4 and GGUF parsers/loaders;
- tensor-census tools;
- dedicated PLE file-format, mmap, positional-read, batching, deduplication, sorting, prefetch, and page-cache tests;
- AVX2 expert kernels and parity tests;
- cache policy simulation from captured or synthetic traces;
- CPU expert execution and Q4/Q5/Q8/CPU-format conversion/correctness comparisons;
- API/config/metrics contracts;
- CI and benchmark harnesses;
- tiny-model tests on available non-Pascal hardware where architecture-independent.

## Arrival checklist

Current status:

1. [x] Install both cards; airflow qualification remains pending.
2. [x] Record the passive inventory:
   - `nvidia-smi -q`
   - `nvidia-smi topo -m`
   - `lspci -tv`
   - `numactl -H`
   - PCIe link width/speed
   - GPU power and application-clock capabilities
   - passive thermal and clock state (sustained-load qualification remains pending)
3. [x] Verify both GPUs report compute capability 6.1 and 7,680 MiB usable VRAM.
4. [x] Map GPU 0 to NUMA 0 and GPU 1 to NUMA 1; the NVMe is on NUMA 0.
5. [x] Verify driver 580.173.02 with the CUDA 12.6 project container and install NVIDIA Container Toolkit 1.20.0.
6. [ ] Register the self-hosted GitHub runner with labels:
   `self-hosted`, `linux`, `x64`, `cuda`, `sm61`.
7. [x] Run bounded identity/allocation/arithmetic on each card independently.
   A seconds-long FP32 characterization also exercised both links at Gen3 ×16 without exceeding 36 °C, but is explicitly non-qualifying while clocks and airflow are provisional.
8. [ ] Correct airflow and qualify one card under bounded sustained load before dual-card load.
9. [ ] Attach the complete inventory and thermal artifacts to the hardware qualification issue.

## Memory rules

- Do not add RAM solely for model capacity before measuring.
- Never permit swap during a performance run.
- Keep ZFS ARC bounded during inference qualification.
- PLE is backed by its dedicated NVMe file or shard set, and cold-cache, warm-cache, major-page-fault, and steady-state behavior must be measured separately.
- Do not permanently pin the full PLE table; use Linux page cache and bounded staging where appropriate.
- Excessive pinned memory can destabilize the host; use bounded staging or a measured pinned expert bank.
- Memory pages for a GPU-owned layer should prefer the GPU-local NUMA node unless A/B evidence favors interleaving.

## Deferred P4 decisions

Prioritize Pascal DP4A integer kernels and format-specific tuning over FP16, BF16, and FP8 paths.
Before conventional tensor parallelism, test disjoint expert ownership across both P4s for correctness, transfer cost, balance, and recovery.
Do not claim a final quantization recipe until Q4, Q5, Q8, and CPU-format candidates have been benchmarked on the actual P4s.

## Thermal rules

Tesla P4 is passively cooled. A hardware result is invalid if clocks throttle because of airflow or power limits. Record temperature, clocks and power during every release benchmark.
