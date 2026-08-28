# Agent operating contract

This file is the authoritative execution guide for coding agents working in this repository. Read it together with `docs/orchestrator-guide.md`, `docs/product-scope.md`, the applicable ADRs, and the GitHub issue being implemented.

## Mission

Deliver FreeToken-Pascal v1: a correct, reproducible, text-only Qwen3.8-Flash-Next serving runtime for one or two Tesla P4 GPUs, with dual P4 as the release target. The runtime must combine Pascal GPU acceleration, low-bit GGUF expert banks in host memory, AVX2 CPU expert execution, a hot-expert GPU cache, and adaptive concurrent CPU/GPU expert execution.

## Non-negotiable rules

1. **Correctness precedes optimization.** Every optimized path needs a correct reference path and an A/B validation.
2. **Never hide fallback.** The logs and metrics must show which kernels, quant types, cache policy, split policy, and execution devices were actually selected.
3. **Keep experiments gated.** New optimizations default off until correctness and performance gates pass.
4. **No unverifiable performance claims.** Record exact commit, model checksum, quant census, flags, prompt, context, temperature, hardware, clocks, and run statistics.
5. **Preserve upstream licenses and attribution.** Copy license headers with source and record provenance in `NOTICE` and `manifests/upstreams.yaml`.
6. **Do not broaden v1.** Vision, Windows, GLM, general cloud serving, and MTP are outside v1 unless an ADR changes scope.
7. **Do not block on unavailable P4 hardware.** Complete CPU, converter, parser, tiny-model, cache-simulation, and CI work first. Hardware issues stay blocked until the cards and self-hosted runner exist.
8. **One issue, one coherent change.** Do not mix architecture ports, correctness fixes, and kernel tuning in one unreviewable patch.
9. **Safe fallback must remain usable.** Cache size zero and hybrid split disabled must produce a stable CPU-backed path.
10. **Update documentation with code.** Architecture, ADR status, manifest pins, benchmark commands, and operational instructions must not drift.
11. **Parallelize bounded independent work.** Use `gpt-5.6-luna` subagents with `xhigh` reasoning whenever a subtask can run safely without conflicting edits; the main agent retains integration and verification ownership.

## Required workflow

1. Read the issue, its blockers, related ADRs, and upstream references.
2. Reproduce or create the failing test before implementing the change.
3. Record source commit SHAs before copying or adapting code.
4. Implement the smallest complete vertical slice.
5. Run hosted tests locally.
6. Run hardware tests only when the issue is hardware-ready.
7. Add benchmark evidence when the issue claims performance.
8. Update docs and manifests.
9. Open a focused PR using the repository template.
10. Do not close an issue until every acceptance checkbox is evidenced in the PR.

## Branch and commit policy

- Branch: `issue-<number>-<short-name>`.
- Use conventional commits: `feat:`, `fix:`, `perf:`, `test:`, `docs:`, `build:`, `ci:`, `refactor:`.
- Prefer several reviewable commits over one generated dump.
- Never force-push `main`.
- Squash merge unless preserving a deliberate upstream merge commit.

## Validation classes

- **H0 hosted:** formatting, manifests, static analysis, unit tests, host simulators, converter tests.
- **H1 CUDA compile:** CUDA 12.6 compile for `sm_61`; no GPU required.
- **H2 single-P4:** correctness and kernel parity on one P4.
- **H3 dual-P4:** TP, ownership, NUMA, cache, and end-to-end serving.
- **H4 release:** long-context, coding workloads, soak, fault recovery, and reproducibility.

Every issue must declare the highest class it requires. H2/H3 issues must not be marked complete using only emulation.

## Definition of done

A change is done only when:

- tests cover success and failure paths;
- unsupported hardware/configurations fail clearly;
- fallback behavior is tested;
- metrics expose the selected behavior;
- user-facing flags are documented;
- source attribution is recorded;
- CI passes;
- hardware evidence is attached when required;
- no unrelated scope is bundled.
