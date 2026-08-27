# ADR 0003: Cuda126 Sm61 Toolchain

- Status: Accepted
- Date: 2026-08-28

## Context

CUDA 13 and its Torch builds no longer carry Pascal support. Tesla P4 requires compute capability 6.1 and several post-Pascal CUDA/Triton features need fallbacks.

## Decision

Use a pinned CUDA 12.6 build toolchain and a Torch build that contains Pascal-compatible cubins. Compile shipping CUDA explicitly for `sm_61`. Keep CUDA 13 unsupported for the Pascal product.

## Consequences

CI needs a distinct H1 compile gate. Dependencies must not silently upgrade to CUDA 13. Some modern third-party kernels are unavailable and require native/PXA/Torch fallbacks.

## Alternatives considered

CUDA 12.1/12.2: demonstrated by some forks but 12.6 aligns with current pre-Turing support. CUDA 13: impossible for Pascal. CPU-only: fails product goal.
