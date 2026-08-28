# Qwen3.8 Flash Next text architecture

FreeToken-Pascal v1 loads the Qwen3.8 Flash Next language backbone registered as
`Qwen4ExpForConditionalGeneration`. The implementation combines alternating Gated DeltaNet and
Qwen Sparse Attention layers, four-stream hyperconnections, routed and shared experts, and the PLE
hash/convolution reference interface.

The correctness path is deliberately text-only. Vision weights and MTP weights are skipped by the
loader. Direct image encoding and image-token execution fail with a clear v1-scope message. The
imported upstream CUDA 13 image, Windows transport changes, vision tower,
and RTX 3090 benchmark artifacts are not part of this downstream slice.

## State and attention

GDN uses the existing per-request linear state pool. Fresh requests zero their recurrent state;
chunked requests continue from their assigned slot. PLE owns a dilated-convolution state keyed by
the same request table slot and replaces it whenever `cached_len == 0`, preventing state leakage
when a slot is reused. The model forces the naive cache and disables CUDA graphs while the PLE
reference path performs host work. Issue #13 owns the release PLE mmap/offload representation.

QSA retains full-resolution K/V and stores one compressed index key per configured token group.
Before the token budget is reached, selection is dense-equivalent. Beyond it, the indexer selects
compressed groups, expands them to original logical rows, includes the visible incomplete tail,
and runs exact sparse GQA over the original K/V values.

## Correctness controls

`freetoken.models.qwen4_exp.reference` contains device-neutral equations for GDN recurrence,
hyperconnection mixing, and routed/shared expert semantics. They are not used by production
forwards. The model's debug hook is disabled by default and can capture selected logits plus cloned
PLE state for A/B tests.

The pinned H1 CUDA environment runs the Qwen config/model helpers, CPU reference equations, QSA
selection/output, QSA cache geometry, the exact non-power-of-two Torch router fallback, and
text-only failures. Issue #38 owns a separately gated CUDA 12.6 / `sm_61` fused `topk=10`
router; the Torch path remains its permanent oracle and unsupported-shape fallback. H1
also verifies that the existing CUDA translation-unit census remains compilable for `sm_61`; QSA's
runtime-generated Triton kernels cannot be compiled without a GPU. Real runtime generation, fused
GDN and QSA parity, and independent selected-logit comparison remain H2 requirements and cannot be
satisfied until Issue #9 provides a Tesla P4.
