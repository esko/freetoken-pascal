# Qwen4 GDN backend and donor-audit ledger

- Downstream issue: #93
- Branch: `issue-93-gdn-backend-contract`
- Base: `018296a7b75f15b3db66a1e4ff40d32b3d47fc41`
- Validation class: H0 only; no CUDA, sm_61 compile, or P4 evidence

Issue #93 establishes the backend decision boundary for Qwen3.8-Flash-Next GatedDeltaNet.
The immutable, Torch-free contract lives in
`python/freetoken/models/qwen4_exp/gdn_contract.py`. It receives a capability tuple, an
activation dtype, and explicit package probes. `auto` selects the eligible in-tree FLA path only
for supported modern-GPU inputs; `sm_61`, unsupported dtypes, and unavailable FLA select the
observable `torch-reference` fallback. The `triton-candidate` path is explicit-only and remains
gated until its donor audit and H1/H2 parity evidence are complete. The mode and package
availability probes are frozen when the GDN object is constructed; a process-environment mutation
cannot switch a live request between backends.

| Area | Exact source/pin | Use in this issue | Status |
|---|---|---|---|
| In-tree FLA GDN kernels | FreeToken-Pascal history, initial vendoring commit `3af9d90ee5` | Existing eligible modern-GPU implementation | Preserved; imported only after the decision; nonzero-state H2 parity pending |
| Pascal/GDN fusion candidate | `poisonxa16/pxq_llama.cpp`, release commit `d34d74e93b95761e67a17a649cf2faf039e7888e` | Audit `ssm_scan`/DeltaNet/GDN fusion and Pascal constraints | Planned donor; no source copied |
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

No donor CUDA or Triton code is imported by this slice. The candidate remains disabled from
`auto`, and forcing it without an affirmative availability probe fails closed. The FLA kernel's
recurrent state is indexed as [V,K] while the current pool contract is [K,V]; equal head
dimensions make the tensors shape-compatible but do not establish axis-order equivalence. The
direct snapshot path therefore remains an explicit H0 limitation pending canonical nonzero-state
H2 parity. Backend switching and checkpoint parity are not claimed. Hosted tests cover decision
immutability, visible fallback reasons, invalid/forced modes, observer delivery, constructor
snapshotting, reference state behavior, and the static guarantee that the Qwen4 GDN boundary
resolves before any FLA call.
