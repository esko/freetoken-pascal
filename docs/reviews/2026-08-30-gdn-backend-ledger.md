# Qwen4 GDN backend and donor-audit ledger

- Downstream issue: #93
- H1 branch/base: `issue-93-pascal-gdn-h1` at `71b6b46ebe` (merged PR #95)
- H2 branch/base: `issue-93-p4-gdn-h2` from `eaa8acaa97` (merged PR #112)
- Validation class: H0 plus H1 source/compile census and bounded H2 kernel parity

Issue #93 establishes the backend decision boundary for Qwen3.8-Flash-Next GatedDeltaNet.
The immutable, Torch-free contract lives in
`python/freetoken/models/qwen4_exp/gdn_contract.py`. It receives a capability tuple, an
activation dtype, and explicit package probes. `auto` selects the eligible in-tree FLA path only
for supported modern-GPU inputs; `sm_61`, unsupported dtypes, and unavailable FLA select the
observable `torch-reference` fallback. The `triton-candidate` path is explicit-only and remains
gated until its donor audit and H1/H2 parity evidence are complete. The standalone `pascal-fp32`
path is explicit-only, restricted to `sm_61`, and requires a positive qualification gate;
it is never selected by `auto`. The H2 model boundary stages BF16/FP16 activations to FP32 for
the recurrence only and retains reference projection, convolution, gating, normalization and
output projection.
The mode and package
availability probes are frozen when the GDN object is constructed; a process-environment mutation
cannot switch a live request between backends.

| Area | Exact source/pin | Use in this issue | Status |
|---|---|---|---|
| In-tree FLA GDN kernels | FreeToken-Pascal history, initial vendoring commit `3af9d90ee5` | Existing eligible modern-GPU implementation | Preserved; imported only after the decision; nonzero-state H2 parity pending |
| Pascal/GDN recurrence donor | `poisonxa16/pxq_llama.cpp`, release commit `d34d74e93b95761e67a17a649cf2faf039e7888e` | Audit DeltaNet recurrence and Pascal constraints | Adapted standalone recurrence; graph/convolution fusion excluded |
| Qwen4 model/GDN oracle | `ggml-org/llama.cpp`, PR #27742 head `eaf93765572e794b8e3754fe45adbe12d381e997` | Compare Qwen4 GGUF/model semantics | Reference only; no source copied |

The audited PXA blobs at `d34d74e93b95761e67a17a649cf2faf039e7888e` are recorded in
`manifests/upstreams.yaml` under `pxq-llama.audit_blobs`:

- `ggml/src/ggml-cuda/delta-net.cu` —
  `4f8e131c754a54423ab3c349f6bc072ac9973364`
- `ggml/src/ggml-cuda/delta-net.cuh` —
  `36c4c89e3f79b04ac84af04e69f6bb946ccd9c61`
- `ggml/src/ggml-cuda/pxa-deltanet-fuse.cuh` —
  `f6d1d2bdd8773c919c5b8c8b4c882b48f2f47af9`
- `ggml/src/ggml-cuda/ssm-conv.cu` —
  `d48f4435b0738ff34c8cd6784ff36cc66803d49b`
- `ggml/src/ggml-cuda/ssm-conv.cuh` —
  `8e6c1f00bfa03daf521694b24143f5d47c4019ae`
- `src/graphs/build_qwen4exp.cpp` —
  `d35931a86ad4f2a1369e976493e87aad3bc5a814`
- `src/llama-delta-net.cpp` —
  `dbb2d2383192577b297fd73b1c1093dadab499e0`
- `src/llama-delta-net.h` —
  `b6de906074302d72a2dd8944da52fc655f370470`

The standalone source `python/freetoken/kernel/csrc/jit/gdn_pascal.cu` adapts only the FP32
recurrence arithmetic from the audited `delta-net.cu` blob. It uses explicit D64/D128 device
instantiations, FreeToken TVM-FFI tensor validation, ragged `cu_seqlens` request addressing,
unique pool slots, and the current `[slots, value_heads, K, V]` state layout. It consumes beta
already transformed by sigmoid and does not include donor graph or convolution fusion. The
`gdn_pascal` adapter validates these invariants before JIT loading. The source is in the CUDA
manifest so standalone nvcc census must inspect its `sm_61` device image.

The arithmetic is intentionally not byte-for-byte donor behavior. FreeToken normalizes Q and K,
applies the Q head-dimension scale, consumes beta after sigmoid, omits the donor's decay and state
clamps, preserves `[K,V]` state storage, and uses one FreeToken-owned block per request/value head.
These divergences match the permanent reference contract and require real-P4 output/state A/B
evidence before the adapter can be registered with model dispatch.

No donor Triton or graph-fusion code is imported by this slice. The candidate remains disabled from
`auto`, and forcing it without an affirmative availability probe fails closed. The FLA kernel's
recurrent state is indexed as [V,K] while the current pool contract is [K,V]; equal head
dimensions make the tensors shape-compatible but do not establish axis-order equivalence. The
direct snapshot path therefore remains an explicit limitation pending canonical FLA/Pascal
axis-order comparison. Cross-backend switching and checkpoint parity are not claimed. Hosted tests cover decision
immutability, visible fallback reasons, invalid/forced modes, observer delivery, constructor
snapshotting, reference state behavior, the standalone Pascal source/adapter contract, and the
static guarantee that the Qwen4 GDN boundary resolves before any FLA call.

The permanent Torch/reference seam additionally defines the H0 state semantics that every future
backend must preserve. Deterministic CPU tests compare chunk evaluation with tokenwise decode from
nonzero state, restore both convolution and recurrent state before replaying a suffix, reset and
replay the complete sequence, exercise ragged requests mapped to noncontiguous slots, and verify
that concurrent request updates leave unaddressed slots byte-for-byte unchanged. These reference
contracts alone do not establish FLA/Pascal checkpoint interchangeability, kernel
registration, or device concurrency; those remain issue #93 work.

## Bounded P4 parity

The standalone recurrence was source-JIT compiled with CUDA 12.6 and
`FREETOKEN_DISABLE_KERNEL_CACHE=1`, then launched independently on each installed Tesla P4.
The H2 fixture covers D64/D128, GQA ratios one and two, nonzero initial state, ragged disjoint
slots, untouched-slot isolation, chunk-versus-tokenwise decode, and checkpoint/restore suffix
replay against `gdn_reference.recurrent_gated_delta_rule` at `rtol=3e-5, atol=3e-5`.
Both physical cards passed the bounded suite, including the artifact's D128/HK16/HV48 geometry.
The ECC-on runs peaked at 43 degrees C and approximately
24 W on GPU 0 and 42 degrees C and approximately 24 W on GPU 1; clocks remained intentionally
constrained and these observations are not performance or sustained thermal qualification.

PR #114 added bounded kernel-only observations and PR #115 repaired the BF16 reference fallback.
The subsequent ECC-off model-boundary run passed ragged prefill, carried-state decode, output and
recurrent/conv-state parity, telemetry, and untouched-slot isolation at the real Qwen geometry.
The model seam remains explicit, eager-only and unavailable from the factory. Automatic dispatch,
FLA/Pascal state interchange, tracking/checkpoint snapshots, end-to-end model A/B measurement and
any fused projection/convolution/recurrence path remain open under issue #93.
