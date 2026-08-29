# Backlog work breakdown

GitHub issues are the execution source of truth. [Epic #4](https://github.com/esko/freetoken-pascal/issues/4) contains the live completion checklist. This document records the phase map and completeness audit.

## Issue map

| Phase | Issue | Outcome | Gate |
|---|---:|---|---|
| P0 | [#5](https://github.com/esko/freetoken-pascal/issues/5) | Import upstream FreeToken history and downstream sync model | H0 |
| P0 | [#6](https://github.com/esko/freetoken-pascal/issues/6) | Pin provenance and licensing | H0 |
| P0 | [#7](https://github.com/esko/freetoken-pascal/issues/7) | CUDA 12.6/Python 3.12 environments | H0/H1 |
| P0 | [#8](https://github.com/esko/freetoken-pascal/issues/8) | Fixtures, schemas, hosted and `sm_61` compile CI | H0/H1 |
| P0 | [#77](https://github.com/esko/freetoken-pascal/issues/77) | Sync merged FreeToken Qwen3.8 support and reconcile downstream delta | H0/H1 |
| P0 | [#9](https://github.com/esko/freetoken-pascal/issues/9) | Install/qualify P4s and self-hosted runner | H2/H3, hardware blocked |
| P1 | [#10](https://github.com/esko/freetoken-pascal/issues/10) | FreeToken Pascal/CUDA 12.6 support | H1/H2 |
| P1 | [#11](https://github.com/esko/freetoken-pascal/issues/11) | Qwen3.8/Qwen4 text architecture | H0-H2 |
| P1 | [#12](https://github.com/esko/freetoken-pascal/issues/12) | Safe GGUF K/I loader and tensor census | H0-H2 |
| P1 | [#13](https://github.com/esko/freetoken-pascal/issues/13) | Dedicated NVMe PLE format, random-I/O backends, adaptive batching/prefetch and heterogeneous expert pools | H0-H2 |
| P1 | [#14](https://github.com/esko/freetoken-pascal/issues/14) | Independent short/long-context reference | H0/H2 |
| P2 | [#15](https://github.com/esko/freetoken-pascal/issues/15) | Model-agnostic CPU expert ABI | H0 |
| P2 | [#16](https://github.com/esko/freetoken-pascal/issues/16) | AVX2 Q4_K expert kernels | H0/target CPU |
| P2 | [#17](https://github.com/esko/freetoken-pascal/issues/17) | Named Q3 profile and census-required Q2/Q3/IQ CPU formats | H0/target CPU |
| P2 | [#18](https://github.com/esko/freetoken-pascal/issues/18) | NUMA-aware host banks and bounded pinning | H0/H3 |
| P3 | [#19](https://github.com/esko/freetoken-pascal/issues/19) | Pascal DP4A and format-tuned GPU expert backend | H1/H2 |
| P3 | [#73](https://github.com/esko/freetoken-pascal/issues/73) | VRAM placement-cliff guard and startup canary | H0-H3 |
| P3 | [#20](https://github.com/esko/freetoken-pascal/issues/20) | Compare dual-P4 expert ownership and trunk policies | H0/H3 |
| P3 | [#38](https://github.com/esko/freetoken-pascal/issues/38) | Fused Qwen3.8 `topk=10` Pascal router | H0-H2 |
| P3 | [#76](https://github.com/esko/freetoken-pascal/issues/76) | QSA long-context overhead and workspace safety | H0-H3 |
| P3 | [#21](https://github.com/esko/freetoken-pascal/issues/21) | Fixed cache, static-hot comparator and correct mixed partial merge | H0-H3 |
| P3 | [#22](https://github.com/esko/freetoken-pascal/issues/22) | Async LFRU, persisted heat and telemetry | H0-H3 |
| P4 | [#23](https://github.com/esko/freetoken-pascal/issues/23) | Concurrent current-step CPU/GPU misses | H0-H3 |
| P4 | [#24](https://github.com/esko/freetoken-pascal/issues/24) | Contention-aware `q*` and fallback | H0-H3 |
| P4 | [#25](https://github.com/esko/freetoken-pascal/issues/25) | Decode prefetch and prefill streaming | H0-H3 |
| P5 | [#26](https://github.com/esko/freetoken-pascal/issues/26) | Long-context state and semantic checkpoints | H0/H3/H4 |
| P5 | [#27](https://github.com/esko/freetoken-pascal/issues/27) | OpenAI serving, cancellation and observability | H0/H3/H4 |
| P5 | [#28](https://github.com/esko/freetoken-pascal/issues/28) | Docker/Compose and production operations | H0/H3/H4 |
| P5 | [#74](https://github.com/esko/freetoken-pascal/issues/74) | Optional exact context-derived n-gram profile | H0/H2/H3, non-blocking |
| P6 | [#29](https://github.com/esko/freetoken-pascal/issues/29) | Hardware qualification, benchmark, soak and v1.0 | H2-H4, hardware blocked |

## Critical dependency chain

```text
#5 → #6 → #77
          ├→ #13 → #14
          ├→ #17
          ├→ #19
          ├→ #38
          ├→ #73
          └→ #76
#7 → #10
#11 + #12 + #77 → #13 → #14
#12 + #15 → #16 → #17/#18
#77 + #12 + #16 → #17
#12 + #19 + #77 → #73
#9 + #11 + #13 + #73 → #20
#16 + #19 + #20 + #73 + #76 → #21
#14 + #38 → #22 → #23 → #24 → #25
#11 + #14 + #73 + #77 → #76
#25 + #76 → long-context portions of #26 → #27 → #28 → #29
#13 → #25 + #26 + #28 + #29
#13 + #14 + #25 + #26 + #27 → #74  (optional; does not block #29)
```

[#8](https://github.com/esko/freetoken-pascal/issues/8) supports all phases. [#17](https://github.com/esko/freetoken-pascal/issues/17) is required for the selected Q3 profile and any additional release-artifact bank types.

## Work possible before P4 arrival

The orchestrator should execute #77 first, then prioritize H0/H1 portions of #13–#19, #38, #73 and #76, plus the pure logic/tests in #20–#25 and #74. This includes upstream Qwen delta reconciliation, the PLE file format, random-access advice, mmap/`pread`, adaptive batching, read-amplification telemetry, CPU expert execution, Q4/Q3 census and quality fixtures, routing simulation, placement planning/canary logic, QSA workspace accounting and host-synchronization profiling, quant conversion, metrics and correctness A/B tests. Issues #9 and #29 are explicitly blocked. DP4A tuning, placement-cliff measurement, context-scaling measurements, dual-P4 policy selection and final quant/profile selection remain H2/H3 work, and other issues remain open until their required hardware evidence is attached.

## Completeness audit

The core v1 backlog includes explicit work and acceptance evidence for:

- source import, merged-upstream Qwen reconciliation, provenance and licensing;
- reproducible CUDA/Python toolchains;
- hosted, compile and hardware CI;
- loader/converter and malformed-file correctness;
- one authoritative Qwen4 GDN/QSA/hyperconnection/PLE implementation after #77;
- heterogeneous expert bank types resident in DDR4 and a dedicated NVMe PLE layout served through the Linux page cache;
- mmap and positional-read PLE backends with random advice, adaptive deduplication/ordering, asynchronous prefetch and physical read-amplification evidence;
- AVX2 CPU fallback and required low-bit formats;
- named Q4 reference and Q3 whole-model throughput profiles, gated AP-Q4/AP-IQ4 candidates, converter oracles, and component-level higher-precision tests;
- Pascal GPU kernel parity;
- upstream-first exact full-softmax `topk=10` router adaptation, parity, and measured Pascal fallback decision;
- placement planning, post-load/post-prefill high-water accounting, headroom, canary, automatic backoff and fail-readiness behavior;
- bounded QSA score/top-k/gather workspaces, context-scaling telemetry, controlled OOM behavior and long-context performance evidence;
- one-GPU bring-up and measured two-GPU ownership/trunk policies;
- cache-zero, static-hot, static cache, async fill and current-step hybrid merge;
- contention-aware scheduling and safe pure fallbacks;
- prefill wider than cache capacity;
- long-context state and semantic restore;
- streaming, cancellation, health, metrics and security limits;
- topology, NUMA, thermal and power evidence;
- clean Docker/Compose deployment and rollback;
- benchmarks, soak, fault injection and release reproducibility.

The optional v1 backlog may additionally provide exact context-derived n-gram speculation, but it cannot block core v1. Post-v1 items such as vision, GLM, native MTP, DFlash, external draft models, lossy speculative prefill and pruning must not be used to block release.
