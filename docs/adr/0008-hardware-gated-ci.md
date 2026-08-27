# ADR 0008: Hardware Gated Ci

- Status: Accepted
- Date: 2026-08-28

## Context

GitHub-hosted runners do not provide Tesla P4 hardware. The cards are not yet available, but development must proceed without pretending emulation proves runtime correctness.

## Decision

Use H0 hosted checks, H1 CUDA `sm_61` compile checks, and manual/self-hosted H2-H4 hardware workflows. Hardware artifacts include topology, commands, raw results, clocks and temperatures.

## Consequences

Some issues remain open until hardware arrives. CI remains useful immediately. Release claims have an auditable evidence trail.

## Alternatives considered

Disable CI until hardware: wastes available validation. Run only on modern GPU: misleading for Pascal. Require hardware on every PR: blocks independent work.
