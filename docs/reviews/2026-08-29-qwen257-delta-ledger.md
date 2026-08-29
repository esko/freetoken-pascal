# FreeToken PR #257 reconciliation ledger

- Downstream issue: #77
- Initial downstream base: `9ef3651309fe4058672f2cc92069238dea06be1b`
- Upstream PR #257 merge: `bd8f3d519a48777bf22ee5c7c8f58f4f3ff31b40`
- Exact synchronized upstream tip: `58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6`
- Validation class: H0/H1; no P4 evidence

The exact upstream tip is a parent of the downstream reconciliation merge. The merged PR #257 implementation is authoritative for Qwen configuration, modular GDN/hyperconnection/MoE/PLE/model code, QSA cache/backend contracts, the arbitrary-`K` router, and associated tests. Closed PR #232 remains historical provenance only for downstream code that still differs for Pascal, GGUF, file-backed PLE, or reference testing.

| Capability | Authoritative path after sync | Retained downstream delta |
|---|---|---|
| Model/config | `models/qwen4_exp/{config,attention,gdn,hc,moe,ple,model,weight}.py` | Text-only rejection, GGUF parser/factory, CPU-expert attachment lifecycle, debug/evidence hooks |
| QSA runtime | `attention/qsa_sparse.py`, `kernel/triton/qsa/`, `kvcache/qsa_pool.py` | Pascal-safe dispatch remains gated; the old Torch gathered-row path is reference-only and is not registered |
| PLE runtime | upstream `PLELayer` and state-slot contract | Dedicated artifact mapping, mmap/`pread`, flattened mapped-table adapter, bounded warming and telemetry |
| Router | upstream arbitrary-`K` fused router | Permanent full-softmax Torch oracle and issue #38 Pascal qualification policy |
| Expert execution | upstream MoE/offload interfaces | Low-bit GGUF host banks, AVX2 execution, eager cache-zero bridge and transfer telemetry |
| State/cache | upstream QSA and PLE slot-state contracts | Downstream capacity, placement, no-swap and failure policy |
| Weight loading | upstream modular loader | GGUF conversion, heterogeneous quant census, PLE hash constants and fail-closed shape checks |

## Retained PR #232-derived paths

- `models/qwen4_exp/gguf.py`, `gguf_attach.py`, and GGUF portions of `config.py` remain because merged PR #257 does not implement the downstream low-bit GGUF/CPU serving contract.
- `attention/qsa.py` plus `kernel/triton/qsa_legacy.py` are unregistered reference-only paths used for CPU/Pascal parity fixtures until issue #76 migrates every comparison to the shipping selected-row backend.
- `tests/models/qwen4_exp/legacy_downstream.py` contains pre-sync PLE reference helpers for regression tests only. It is outside the installed runtime and cannot be selected by serving.
- `reference.py` remains an independent device-neutral semantic oracle for GDN, hyperconnection and routed/shared expert comparisons.

## Reconciliation invariants

- The dedicated GGUF PLE artifact remains independently mapped and never causes unrelated weights to be warmed or pinned.
- GGUF construction converts current upstream modules before state loading, restores PLE hash constants, and rejects incompatible row geometry.
- Cache-zero and the CPU expert bridge remain available; upstream GPU/cache paths do not silently replace them.
- Vision and MTP remain outside v1.
- Triton QSA/router sources are present for H1 compile and future H2 qualification, but no P4 support or performance is claimed from hosted tests.
