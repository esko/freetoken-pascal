# Backlog work breakdown

GitHub issues are the execution source of truth. [Epic #4](https://github.com/esko/freetoken-pascal/issues/4) contains the live completion checklist. This document records the phase map and completeness audit.

## Issue map

| Phase | Issue | Outcome | Gate |
|---|---:|---|---|
| P0 | [#5](https://github.com/esko/freetoken-pascal/issues/5) | Import upstream FreeToken history and downstream sync | H0 |
| P0 | [#6](https://github.com/esko/freetoken-pascal/issues/6) | Pin provenance and licensing | H0 |
| P0 | [#7](https://github.com/esko/freetoken-pascal/issues/7) | CUDA 12.6/Python 3.12 environments | H0/H1 |
| P0 | [#8](https://github.com/esko/freetoken-pascal/issues/8) | Fixtures, schemas, hosted and `sm_61` compile CI | H0/H1 |
| P0 | [#9](https://github.com/esko/freetoken-pascal/issues/9) | Install/qualify P4s and self-hosted runner | H2/H3, hardware blocked |
| P1 | [#10](https://github.com/esko/freetoken-pascal/issues/10) | FreeToken Pascal/CUDA 12.6 support | H1/H2 |
| P1 | [#11](https://github.com/esko/freetoken-pascal/issues/11) | Qwen3.8/Qwen4 text architecture | H0-H2 |
| P1 | [#12](https://github.com/esko/freetoken-pascal/issues/12) | Safe GGUF K/I loader and tensor census | H0-H2 |
| P1 | [#13](https://github.com/esko/freetoken-pascal/issues/13) | Heterogeneous expert pools and PLE mmap | H0-H2 |
| P1 | [#14](https://github.com/esko/freetoken-pascal/issues/14) | Independent short/long-context reference | H0/H2 |
| P2 | [#15](https://github.com/esko/freetoken-pascal/issues/15) | Model-agnostic CPU expert ABI | H0 |
| P2 | [#16](https://github.com/esko/freetoken-pascal/issues/16) | AVX2 Q4_K expert kernels | H0/target CPU |
| P2 | [#17](https://github.com/esko/freetoken-pascal/issues/17) | Census-required Q2/Q3/IQ CPU formats | H0/target CPU |
| P2 | [#18](https://github.com/esko/freetoken-pascal/issues/18) | NUMA-aware host banks and bounded pinning | H0/H3 |
| P3 | [#19](https://github.com/esko/freetoken-pascal/issues/19) | PXA/PXQ `sm_61` GPU expert backend | H1/H2 |
| P3 | [#20](https://github.com/esko/freetoken-pascal/issues/20) | Qwen4 TP=2 and dual-P4 ownership | H0/H3 |
| P3 | [#21](https://github.com/esko/freetoken-pascal/issues/21) | Fixed cache and correct mixed partial merge | H0-H3 |
| P3 | [#22](https://github.com/esko/freetoken-pascal/issues/22) | Async LFRU, persisted heat and telemetry | H0-H3 |
| P4 | [#23](https://github.com/esko/freetoken-pascal/issues/23) | Concurrent current-step CPU/GPU misses | H0-H3 |
| P4 | [#24](https://github.com/esko/freetoken-pascal/issues/24) | Contention-aware `q*` and fallback | H0-H3 |
| P4 | [#25](https://github.com/esko/freetoken-pascal/issues/25) | Decode prefetch and prefill streaming | H0-H3 |
| P5 | [#26](https://github.com/esko/freetoken-pascal/issues/26) | Long-context state and semantic checkpoints | H0/H3/H4 |
| P5 | [#27](https://github.com/esko/freetoken-pascal/issues/27) | OpenAI serving, cancellation and observability | H0/H3/H4 |
| P5 | [#28](https://github.com/esko/freetoken-pascal/issues/28) | Docker/Compose and production operations | H0/H3/H4 |
| P6 | [#29](https://github.com/esko/freetoken-pascal/issues/29) | Hardware qualification, benchmark, soak and v1.0 | H2-H4, hardware blocked |

## Critical dependency chain

```text
#5 → #7 → #10 → #11 → #12 → #13 → #14
                    └→ #15 → #16 → #18
                         #12 → #19
#9 + #11 → #20
#16 + #19 + #20 → #21 → #22 → #23 → #24 → #25
#14 + #20 + #24 + #25 → #26 → #27 → #28 → #29
```

[#6](https://github.com/esko/freetoken-pascal/issues/6) and [#8](https://github.com/esko/freetoken-pascal/issues/8) support all phases. [#17](https://github.com/esko/freetoken-pascal/issues/17) is required for the selected 3-bit profile and any additional release-artifact bank types.

## Work possible before P4 arrival

The orchestrator should prioritize H0/H1 portions of #5–#8 and #10–#19, plus the pure logic/tests in #20–#25. Issues #9 and #29 are explicitly blocked. Other issues remain open until their H2/H3 evidence is attached, even when their hosted implementation is ready.

## Completeness audit

The v1 backlog includes explicit work and acceptance evidence for:

- source import, provenance and licensing;
- reproducible CUDA/Python toolchains;
- hosted, compile and hardware CI;
- loader/converter and malformed-file correctness;
- Qwen4 GDN/QSA/hyperconnection/PLE semantics;
- heterogeneous expert bank types and PLE NVMe/page cache;
- AVX2 CPU fallback and required low-bit formats;
- Pascal GPU kernel parity;
- one-GPU bring-up and two-GPU ownership;
- cache-zero, static cache, async fill and current-step hybrid merge;
- contention-aware scheduling and safe pure fallbacks;
- prefill wider than cache capacity;
- long-context state and semantic restore;
- streaming, cancellation, health, metrics and security limits;
- topology, NUMA, thermal and power evidence;
- clean Docker/Compose deployment and rollback;
- benchmarks, soak, fault injection and release reproducibility.

Post-v1 items such as vision, GLM, MTP, DFlash, n-gram speculation and pruning must not be used to block release.