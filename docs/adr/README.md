# Architecture decision records

ADRs are immutable after acceptance except for status and explicit supersession notes. New decisions use the next four-digit number.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-freetoken-primary-base.md) | Use FreeToken as the primary runtime base | Accepted |
| [0002](0002-downstream-fork-and-upstream-sync.md) | Maintain a downstream fork with pinned upstream merges | Accepted |
| [0003](0003-cuda126-sm61-toolchain.md) | Use CUDA 12.6 and explicit `sm_61` builds | Accepted |
| [0004](0004-gguf-low-bit-host-experts.md) | Use GGUF low-bit host expert banks | Accepted |
| [0005](0005-hybrid-expert-cache.md) | Use fixed-address per-GPU caches and adaptive hybrid execution | Accepted |
| [0006](0006-avx2-cpu-expert-backend.md) | Build an AVX2 CPU expert backend | Accepted |
| [0007](0007-dual-p4-layer-ownership.md) | Use deterministic layer ownership for dual P4 | Superseded by 0010 |
| [0008](0008-hardware-gated-ci.md) | Separate hosted, compile and self-hosted hardware gates | Accepted |
| [0009](0009-text-only-qwen38-v1.md) | Limit v1 to text-only Qwen3.8-Flash-Next | Accepted |
| [0010](0010-three-tier-ple-and-expert-residency.md) | Separate NVMe PLE, DDR4 experts/page cache, and dual-P4 hot compute | Accepted |
