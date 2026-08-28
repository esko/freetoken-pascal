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
  cache/
  results/
  logs/
```

PLE and model shards stay on NVMe. Benchmark results and cache heat must not be stored in the Git repository.

## Startup sequence

1. Validate driver, GPU count and compute capability.
2. Validate model checksum and tensor census.
3. Validate available RAM, VRAM, pinned-memory budget and no swap pressure.
4. Read `--ple-warm-mode` and confirm the logged PLE source, mapping and warm policy.
5. Run startup microbench/autotune or load a hardware-profile cache tied to exact checksums.
6. Log selected CPU/GPU kernels, layer ownership, cache slots and NUMA policy.
7. Start the API health endpoint.
8. Optionally request targeted, PLE readahead or explicit full-model warming before readiness.
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

## Upgrade

Upgrades use immutable image tags and pinned model checksums. A new image must pass the deterministic startup probe and can roll back without converting model state.
