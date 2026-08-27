# ADR 0005: Hybrid Expert Cache

- Status: Accepted
- Date: 2026-08-28

## Context

Static CPU-MoE leaves GPU capacity unused; static GPU layers waste slots on cold experts. FreeToken's dynamic scheduling can exploit short-term expert locality and balance CPU compute against PCIe transfer.

## Decision

Use fixed-address per-GPU expert caches, LFRU initially, asynchronous future-token fill, and a measured current-step `q*` split. CPU and GPU partials execute concurrently. Cache zero and pure policies remain supported.

## Consequences

The merge path and scheduler are correctness-critical. Transfer and CPU work contend for host memory, so autotuning must use concurrent measurements. Every decision requires telemetry and safe fallback.

## Alternatives considered

Static `-cmoe`: simple baseline, lower ceiling. Fetch every miss: stalls on PCIe. CPU every miss: leaves hot reuse unaccelerated. Pruning: model-changing and not throughput-focused.
