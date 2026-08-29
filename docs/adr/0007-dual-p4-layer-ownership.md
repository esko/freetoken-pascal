# ADR 0007: Dual P4 Layer Ownership

- Status: Superseded by ADR 0010
- Date: 2026-08-28

## Context

The P4s have no NVLink and only 8 GB each. A global coherent cache or cross-GPU expert movement would add complexity and PCIe traffic.

## Decision

Assign contiguous model-layer ranges to each P4 initially. Each GPU caches only experts for owned layers and uses host pages/workers associated with its local NUMA node where measured beneficial. No cross-GPU cache coherence in v1.

ADR 0010 supersedes the ordering and default-selection portion of this decision. Disjoint routed-expert ownership is now qualified before conventional tensor parallelism, and no final dual-P4 default is selected until H3 evidence compares the candidates. The no-cross-GPU-cache-coherence constraint remains in force.

## Consequences

Layer balance and TP communication must be measured. Graph split may be benchmarked, but the release has one documented default. A per-rank cache simplifies correctness and recovery.

The host-bank NUMA extension is not a layer-ownership default: its H0 placement
hook is explicit, mapping-scoped, and reports fallback or sampled observations.
It does not yet establish per-rank NUMA locality or a performance result.

## Alternatives considered

Shared global cache: better theoretical utilization but requires coherence/P2P. Row-split experts: frequent inter-GPU reduction. One GPU only: valid bring-up, not release target.
