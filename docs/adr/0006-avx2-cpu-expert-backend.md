# ADR 0006: Avx2 Cpu Expert Backend

- Status: Accepted
- Date: 2026-08-28

## Context

The target dual Xeon E5-v3 host has AVX2 but no AVX-512 or AMX. Current FreeToken CPU work cannot assume modern server instructions, while CPU execution is essential for cache misses.

## Decision

Create a model-agnostic AVX2 expert backend and adapt proven low-bit primitives from llama.cpp/ik/PXA. Implement only quant types required by target artifacts, with dequantize-plus-reference parity tests.

The stable boundary is established before the optimized kernels: immutable
per-projection descriptors retain native quant identifiers, geometry, strides and
bounded source addresses; execution uses a prepared workspace, explicit partial-output
accumulation, cancellation and per-request telemetry. The dense/dequantize path remains
the correctness oracle. Packed decoders and optimized executors must declare bounded
scratch and avoid per-token or per-route heap allocation. Thread-pool and NUMA policy
are injected hooks, not process-global behavior.

The compiled executor's affinity hook is backed by the process mask and
sysfs-derived physical-core plan. Positive worker counts are rejected when they
exceed visible physical-core capacity before native construction; native startup
must read back exact singleton masks before reporting verification. A flag-sync
coordinator is optional and uses a spare planned core only when capacity allows.
The standalone Q4 GGUF bridge reuses the same immutable plan for positive
`num_threads` requests. Its internally owned Python pool applies and reads back
one singleton mask per participating worker; a failure drains the request and
reruns the serial reference executor. Direct Q4 callers without a plan and
caller-supplied pools retain their existing behavior, while explicit plans with
external pools are rejected. The report distinguishes planned, verified, and
fallback affinity state and never changes the owner process mask.

## Consequences

Some newest vLLM CPU kernels are unusable. Kernel work must respect NUMA and bounded pinning. The backend becomes reusable for later MoEs without coupling to Qwen class names.

Grouped execution is ordered and serial in the initial reference implementation; it is
not an atomic multi-request transaction. Optimized parallel implementations may use the
injected hooks but must retain request isolation, fail-closed output commit, and the same
observable telemetry contract.
An executor that rejects a concurrent request as busy attaches telemetry but leaves its
unowned caller buffer untouched; only an accepted request may commit or clear its output.

## Alternatives considered

Use generic Torch CPU: correctness reference but likely too slow. Require AVX-512/AMX: incompatible. Execute every miss on GPU: transfer bottleneck and no q-star balance.
