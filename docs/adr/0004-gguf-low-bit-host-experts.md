# ADR 0004: Gguf Low Bit Host Experts

- Status: Accepted
- Date: 2026-08-28

## Context

The full FP8/BF16 model does not fit the target host memory comfortably. Existing Q4 and 3-bit GGUF artifacts fit and have mature CPU representations. The host expert bank is the source of truth for CPU execution and GPU cache fills.

## Decision

Support GGUF K/I quantized expert banks, beginning with the exact types in Q4_K_XL and a selected 3-bit artifact. Support heterogeneous tensor and expert-layer types. Record a complete tensor census and fail closed for unsupported types.

## Consequences

The loader and cache must understand row stride and type per pool/layer. CPU and GPU kernels may differ by format. GGUF correctness must be independently checked against llama.cpp.

The explicit host-bank policy may opt into a read-only no-swap admission guard. Before policy-owned allocation or FTW metadata reads it parses process `VmSwap`, system `SwapTotal`/`SwapFree`, and active `/proc/swaps` rows. Active swap, process swap, or unavailable/ambiguous procfs data rejects the policy; no swapoff, sysctl, privilege escalation, or mmap/mlock inference is permitted. Accounting records raw status, `no_swap_observed`, and source/errors; without the opt-in the policy does not probe procfs. This remains a point-in-time admission check rather than a guarantee about the complete model's later residency.

## Alternatives considered

Safetensors FP8/BF16: too large. Convert everything to one custom PXQ format: poor CPU fallback and unnecessary quality risk. Prune experts: changes model capability and does not directly reduce active compute.
