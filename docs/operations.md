# Operations guide

This document describes the intended release operation. Commands become authoritative only after the packaging issue is complete.

## Deployment model

- Ubuntu Server host
- NVIDIA driver compatible with CUDA 12.6
- Docker Engine + Compose
- model files on the 1 TB PCIe NVMe
- one FreeToken-Pascal service with both P4s visible
- OpenAI-compatible API bound to the private network or Tailscale
- no public unauthenticated exposure

## Filesystem layout

```text
/srv/freetoken-pascal/
  compose.yaml
  config/
  models/
  ple/
  cache/
  results/
  logs/
```

Model and dedicated PLE shards stay on NVMe, but PLE shards live in their own directory and contain no unrelated model tensors. The complete quantized expert bank is loaded into DDR4 for serving. The full PLE is not permanently pinned; Linux may use spare DDR4 as page cache. Benchmark results and cache heat must not be stored in the Git repository.

## Startup sequence

1. Validate driver, GPU count and compute capability.
2. Validate model checksum and tensor census.
3. Validate available RAM, VRAM and pinned-memory budget, including capacity for the complete quantized expert bank in DDR4 without pinning the full PLE. For an explicit host-bank policy, add `--host-bank-require-no-swap` to fail closed on active swap, process `VmSwap`, or an unavailable/ambiguous procfs probe; this is a point-in-time admission check and performs no host swap mutation. For NUMA placement, opt in separately with `--host-bank-enforce-numa-placement`; inspect `numa_status`, target/allowed nodes, applied mappings, and any fallback reason in startup accounting. This is mapping placement telemetry, not a complete residency or affinity guarantee.
4. Read the selected PLE mmap or positional-read backend and warm policy, then confirm the logged dedicated shard identity, row geometry and absence of unrelated tensors.
5. Run startup microbench/autotune or load a hardware-profile cache tied to exact checksums.
6. Log selected CPU/GPU kernels, layer ownership, cache slots and NUMA policy.
7. Start the API health endpoint.
8. Optionally request bounded targeted PLE readahead before readiness; do not lock or eagerly retain the entire PLE in RAM.
9. Serve traffic only after a short deterministic self-test.

## Health

Readiness requires:

- model loaded;
- deterministic probe completed;
- all expected ranks active;
- no unsupported fallback;
- temperature and VRAM within limits.

Liveness must not claim healthy while a CUDA worker is wedged.

## Shutdown

- stop accepting requests;
- cancel or drain in-flight work;
- persist validated expert heat atomically;
- flush metrics and logs;
- release pinned memory and CUDA resources;
- exit so the container supervisor can restart cleanly.

## Resource controls

The release must document:

- maximum context;
- maximum concurrent requests;
- model/pinned/page-cache RAM expectations;
- ZFS ARC recommendation;
- Docker memory and shared-memory settings;
- request body and output limits;
- timeout and cancellation behavior.

Operational dashboards report PLE cold/warm phase, major faults, block-device reads, logical versus unique rows, coalesced ranges and prefetch effectiveness separately from expert-bank residency. Startup backing for expert files does not authorize steady-state SSD expert execution unless the experimental mode is explicitly selected and logged.

For H0 host-resource lifecycle diagnostics, run the bounded observation with an outer
supervisor timeout, for example `timeout 60s make stress-host-resources`. The default
requires Torch and the production HostBank path; CPU-only H0 jobs must opt in to the
explicit test-only `--allow-fallback` argument. Keep the JSON artifact with the commit,
seed, iteration/thread settings, fake kernel census, policy accounting, and before/after
FD/thread/live-buffer observations. Its RSS field `sampled_max_after_iteration` is the
maximum of samples taken after each iteration, not a live or process-lifetime peak. This
is an `observation_only` cleanup and parity check; its RSS and swap fields do not establish
capacity, no-swap safety, NUMA placement,
staging behavior, throughput, or P4 support.

## Upgrade

Upgrades use immutable image tags and pinned model checksums. A new image must pass the deterministic startup probe and can roll back without converting model state.
