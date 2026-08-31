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
when a slot is reused. Linear-attention models default to the hybrid-radix cache; CUDA graphs stay
disabled while the PLE reference path performs host work. Issue #13 owns the release PLE
mmap/offload representation.

The Qwen4 GDN boundary resolves `FREETOKEN_GDN_MODE` (`auto`, `reference`, `triton-candidate`, or
`pascal-fp32`) at construction, before importing a fused kernel. The mode and package
availability probes are frozen for the lifetime of the GDN object. `auto` selects the eligible
in-tree FLA implementation only on supported modern-GPU inputs; `sm_61`, unsupported dtypes, or
unavailable packages select the stateful `torch-reference` fallback and expose the reason through
the debug observer. `triton-candidate` is explicit-only and fails closed until its donor and H1/H2
parity gates are complete. `pascal-fp32` is an explicit-only `sm_61` recurrence seam; it is never
selected by `auto` and requires a positive qualification gate. Its H2 model boundary stages
BF16/FP16 recurrence inputs to FP32 while retaining the reference projection, convolution, gate,
normalization and output-projection stages. It remains eager-only and rejects graph capture,
non-FP32 recurrent pools and unsupported tracking/checkpoint snapshots. The immutable decision
contract is Torch-free so this policy is testable in H0. The standalone adapter preserves the
pool's `[slots, value_heads, K, V]` layout, consumes pre-sigmoided beta, and addresses ragged
requests with unique slot IDs and `cu_seqlens`. The FLA recurrent-state view is [V,K] while the
pool view is [K,V]; equal dimensions only make these views shape-compatible, so axis-order and
nonzero-state parity remain H2 work. Backend switching and checkpoint parity are not claimed by
H0, and factory activation remains disabled.

QSA retains full-resolution K/V and stores one compressed index key per configured token group.
Before the token budget is reached, selection is dense-equivalent. Beyond it, the indexer selects
compressed groups, expands them to original logical rows, includes the visible incomplete tail,
and runs exact sparse GQA over the original K/V values.
The Torch-free `freetoken.attention.qsa_workspace` planner is an H0 accounting primitive for the
concrete score, top-k, expand-gather, attention, and state allocations.
It derives compressed score geometry from the raw page-table token-slot width, keeps eager ragged
batch metadata separate from graph maximum-batch buffers, and accounts for the runtime's bounded
128 MiB score tile.
It does not allocate CUDA memory or alter QSA dispatch, and a later placement owner may consume its
capacity telemetry as a preflight input.
The `freetoken.engine.qsa_placement` adapter binds its derived persistent and transient categories
to a placement plan or canonical profile and rejects arbitrary QSA byte overrides.
It partitions the active eager or capture high-water exactly across #73's ten buckets, while dual-GPU ownership remains deferred until rank-local geometry is explicit.

## Correctness controls

`freetoken.models.qwen4_exp.reference` contains device-neutral equations for GDN recurrence,
hyperconnection mixing, and routed/shared expert semantics. They are not used by production
forwards. The model's debug hook is disabled by default. When explicitly enabled for a correctness
run, it captures cloned semantic observations at stable boundaries: global router IDs and weights,
GDN recurrent state, logical QSA selections/state, PLE contribution/state, and selected logits.
Capture allocations therefore cannot affect production runs, and captured route IDs precede any
cache-slot remapping. Observation payloads use active request UIDs, logical positions, and sequence
boundaries; padded rows and physical allocator/page-table identities are excluded from compared
state.

`tests/fixtures/qwen38-reference-corpus.json` defines the synthetic, deterministic prompt matrix and
pins its tokenizer revision. Every case must be rendered with the exact serving chat template, and
the evidence hashes and token-counts that rendered byte string rather than the intermediate message
objects. Long-context templates must prove their exact 32K, 128K, or 262K token count and their
single retrieval-needle position; declaring a target in fixture metadata is not evidence.
`scripts/write_qwen38_observations.py` writes non-pickle semantic array bundles, and
`scripts/compare_qwen38_observations.py` binds both implementations to the same artifact, quant
census, corpus, prompt, context length, cache mode, and quantization before comparing them. Exact
routing/state observations and documented numeric tolerances are both required. Synthetic bundles
exercise the H0 contract only and cannot satisfy the H2 model gate.

The pinned H1 CUDA environment runs the Qwen config/model helpers, CPU reference equations, QSA
selection/output, QSA cache geometry, the exact non-power-of-two Torch router fallback, and
text-only failures. Issue #38 owns a separately gated CUDA 12.6 / `sm_61` fused `topk=10`
router; the Torch path remains its permanent oracle and unsupported-shape fallback. H1
also verifies that the existing CUDA translation-unit census remains compilable for `sm_61`; QSA's
runtime-generated Triton kernels cannot be compiled without a GPU. Real runtime generation, fused
GDN and QSA parity, and independent selected-logit comparison remain H2 requirements and cannot be
satisfied until Issue #9 provides a Tesla P4.
