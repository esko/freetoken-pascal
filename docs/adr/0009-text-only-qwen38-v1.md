# ADR 0009: Text Only Qwen38 V1

- Status: Accepted
- Date: 2026-08-28

## Context

Qwen3.8-Flash-Next includes vision and MTP, and other models are attractive, but the user's immediate workload is coding-agent text inference. Broadening increases model, cache and kernel combinations before the core runtime is proven.

## Decision

v1 supports text-only Qwen3.8-Flash-Next. Vision, GLM, MTP, DFlash and n-gram speculation are explicitly post-v1 unless a scope-changing ADR is accepted.

## Consequences

The release can finish with a bounded test matrix. Architecture interfaces should remain reusable, but no speculative feature may delay v1.

## Alternatives considered

Full multimodal day one: unnecessary scope. GLM simultaneously: triples active-path and quant work. MTP first: obscures baseline and complicates state validation.
