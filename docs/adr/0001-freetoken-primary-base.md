# ADR 0001: Freetoken Primary Base

- Status: Accepted
- Date: 2026-08-28

## Context

The desired optimization is dynamic expert paging and concurrent CPU/GPU execution. FreeToken already centers its architecture on this problem, while adding the same behavior to llama.cpp/PXQ or vLLM would require substantial new scheduler work.

## Decision

Use FreeToken as the primary implementation base. Use llama.cpp/PXQ as the GGUF, AVX2 and Pascal-kernel donor/reference; use vLLM as a Qwen4, TP, PLE and serving oracle; use Colibrì as a cache-policy reference.

## Consequences

The project inherits a smaller upstream ecosystem and several moving PRs. It must maintain disciplined source pins and independent correctness comparisons. The core optimization is native rather than bolted on.

## Alternatives considered

PXQ/llama as primary: easier initial Q4 baseline but lacks the target cache/scheduler. vLLM as primary: stronger serving ecosystem but the critical concurrent hybrid execution is unfinished. Colibrì: useful design, larger Qwen4/Pascal port.
