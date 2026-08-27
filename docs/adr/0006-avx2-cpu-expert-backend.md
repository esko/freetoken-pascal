# ADR 0006: Avx2 Cpu Expert Backend

- Status: Accepted
- Date: 2026-08-28

## Context

The target dual Xeon E5-v3 host has AVX2 but no AVX-512 or AMX. Current FreeToken CPU work cannot assume modern server instructions, while CPU execution is essential for cache misses.

## Decision

Create a model-agnostic AVX2 expert backend and adapt proven low-bit primitives from llama.cpp/ik/PXA. Implement only quant types required by target artifacts, with dequantize-plus-reference parity tests.

## Consequences

Some newest vLLM CPU kernels are unusable. Kernel work must respect NUMA and bounded pinning. The backend becomes reusable for later MoEs without coupling to Qwen class names.

## Alternatives considered

Use generic Torch CPU: correctness reference but likely too slow. Require AVX-512/AMX: incompatible. Execute every miss on GPU: transfer bottleneck and no q-star balance.
