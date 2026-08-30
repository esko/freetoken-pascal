# Qwen4 GDN backend and donor-audit ledger

- Downstream issue: #93
- Branch: `issue-93-gdn-backend-contract`
- Base: `018296a7b75f15b3db66a1e4ff40d32b3d47fc41`
- Validation class: H0 only; no CUDA, sm_61 compile, or P4 evidence

Issue #93 establishes the backend decision boundary for Qwen3.8-Flash-Next GatedDeltaNet.
The immutable, Torch-free contract lives in
`python/freetoken/models/qwen4_exp/gdn_contract.py`. It receives a capability tuple, an
activation dtype, and explicit package probes. `auto` selects the qualified in-tree FLA path only
for supported modern-GPU inputs; `sm_61`, unsupported dtypes, and unavailable FLA select the
observable `torch-reference` fallback. The `triton-candidate` path is explicit-only and remains
gated until its donor audit and H1/H2 parity evidence are complete.

| Area | Exact source/pin | Use in this issue | Status |
|---|---|---|---|
| In-tree FLA GDN kernels | FreeToken-Pascal history, initial vendoring commit `3af9d90ee5` | Existing qualified modern-GPU implementation | Preserved; imported only after the decision |
| FLA donor audit | `sgl-project/sglang`, main observed at `a6e402136872653a1eed5efc133fe37382c09e85` on 2026-08-30 | Compare indexed state-pool and fused sigmoid-gating semantics | Reference only; no source copied |
| Pascal/GDN fusion candidate | `poisonxa16/pxq_llama.cpp`, pin `066a37e9540a1ca21375fdeb377836fe69ecb729` | Audit `ssm_scan`/DeltaNet/GDN fusion and Pascal constraints | Planned donor; no source copied |
| Qwen4 model/GDN oracle | `ggml-org/llama.cpp`, PR #27742 head `eaf93765572e794b8e3754fe45adbe12d381e997` | Compare Qwen4 GGUF/model semantics | Reference only; no source copied |

No donor CUDA or Triton code is imported by this slice. The candidate remains disabled from
`auto`, and forcing it without an affirmative availability probe fails closed. Hosted tests cover
decision immutability, visible fallback reasons, invalid/forced modes, observer delivery, and the
static guarantee that the Qwen4 GDN boundary resolves before any FLA call.
