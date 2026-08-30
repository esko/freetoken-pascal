# Release criteria

FreeToken-Pascal core v1 is complete only when every required item is evidenced. Optional profiles have separate gates and cannot block the ordinary-decode release.

## Build and provenance

- [ ] Clean clone builds from documented commands.
- [ ] CUDA 12.6 build contains `sm_61` code and no mandatory newer-architecture path.
- [ ] Upstream commits, copied files and local modifications are pinned.
- [ ] Apache notices and source headers are preserved.
- [ ] Container image and source revision are identifiable.
- [ ] Q4/Q3 model, dedicated PLE, placement and optional-profile identities are immutable and checksummed.

## Model correctness

- [ ] Qwen3.8-Flash-Next text model loads from the pinned `reference-q4` artifact.
- [ ] The named `throughput-q3` artifact loads and has an independent complete census when it is included in release comparison.
- [ ] Tensor censuses and checksums match the recorded release manifests.
- [ ] Every supported profile has a machine-readable sensitive-tensor census recording exact tensor identity/class, dtype or quant format, scale representation, conversion provenance, selected precision, and rationale.
- [ ] The sensitive-tensor precision island explicitly covers MoE routers, `shared_expert_gate` and scales, GDN state-driving/control projections including reconciled `in_proj_a`/`in_proj_b` classes, residual/hyperconnection write gates, norms, and other continuously active control tensors identified by reference comparison.
- [ ] Sensitive tensors begin at source/lossless precision; any Q8 or lower promotion is per-class and evidence-gated, and no broad Q2/Q3/Q4 rule can silently demote them.
- [ ] `shared_expert_gate` and equivalent router/GDN/control tensors pass independent scale/dequant parity and finite/range checks against the authoritative reference.
- [ ] GDN, QSA, hyperconnections, router, shared expert and PLE pass reference checks.
- [ ] QSA selected blocks/rows, sparse-attention output and recurrent/index state pass independent dense-equivalent, sparse-region and context-boundary checks.
- [ ] 32K and 128K continuations are coherent and state-correct.
- [ ] 262K qualification is either passed or explicitly documented as unsupported with measured reason.
- [ ] Checkpoint/restore matches replay.
- [ ] Tool/JSON output probes pass.
- [ ] Q4/Q3 and component-format routing, long-context retrieval, tool-call, structured-output and coding quality gates pass within their declared budgets.
- [ ] The quality harness deterministically fails a deliberately mis-scaled shared-expert-gate fixture and a perturbed/reduced-precision GDN state-control fixture, even when short prompts or retrieval remain superficially coherent.
- [ ] Long-horizon qualification covers multi-turn coding, repeated tool calls/results, state-dependent reasoning, structured transformations, long generation, looping/token-ceiling failure, checkpoint/restore and suffix replay, with intermediate router/gate/GDN state compared where feasible.
- [ ] Bad quant IDs, codecs, offsets, strides, shapes and profiles fail closed.

## Execution

- [ ] Cache-zero CPU-backed mode works.
- [ ] AVX2 CPU expert backend passes parity.
- [ ] The complete quantized expert bank is available through the documented DDR4 serving representation, remains resident under the required no-swap policy, and does not create an uncontrolled duplicate full-bank copy.
- [ ] Pascal DP4A/format-specific GPU expert backends pass parity.
- [ ] Issue #93's GDN backend/reference fallback and CUDA 12.6 `sm_61` compile contract pass; any optimized GDN default remains disabled until real-P4 parity and end-to-end evidence pass.
- [ ] The exact full-softmax `topk=10` router and permanent reference fallback pass parity.
- [ ] The PLE uses dedicated NVMe shard files that contain no unrelated model tensors.
- [ ] mmap and positional-read PLE backends pass identical row and failure-path tests.
- [ ] Supported PLE backends request random-access advice and report whether it was applied.
- [ ] PLE planning is adaptive: direct and vectorized dedupe/order/coalesce paths are correct and observable.
- [ ] PLE prefetch is bounded and asynchronous where enabled.
- [ ] The full PLE is not permanently pinned; spare DDR4 remains available to the Linux page cache.
- [ ] NVMe steady-state residency is PLE-only: routed experts execute from the complete DDR4 bank or bounded P4 cache, and generic SSD-backed expert swap/execution is absent from the v1 path.
- [ ] One P4 works.
- [ ] #73 placement planning, observed post-load and post-first-large-prefill allocation, startup canary, automatic backoff and fail-readiness gates pass with a documented reserve.
- [ ] #76 QSA score/top-k/gather workspaces are bounded/reused, capacity-checked before launch and cannot grow across repeated requests.
- [ ] QSA/top-k workspace exhaustion produces controlled backoff/error rather than process abort or corrupt output.
- [ ] Two P4s work under the selected measured ownership/trunk policy; layer-owned, disjoint, replicated and TP/split candidates are compared where feasible.
- [ ] After installation, release evidence captures each `P4 -> PCIe root -> NUMA node -> CPU socket` mapping; worker pools, expert-bank pages and staging buffers use the measured local node where possible, without inferring locality from GPU ordinal.
- [ ] Local-node, deliberately remote-node and interleaved placement controls are compared using end-to-end decode results before H3 policy selection.
- [ ] Static-hot cache works and is reported beside cache-zero.
- [ ] Async fill works or remains disabled with evidence.
- [ ] Concurrent CPU/GPU partial execution works or safely falls back.
- [ ] q-star autotuner selects a safe policy under contention or a deterministic pure fallback.
- [ ] Prefill policy handles forwards wider than the cache.
- [ ] Expert-bank swap is absent and any steady-state PLE NVMe reads are measured and explained.

## Performance

- [ ] Raw data and commands are published as release artifacts.
- [ ] The best core policy improves end-to-end coding throughput over cache-zero FreeToken by at least 5% outside noise.
- [ ] The auto policy avoids more than 5% regressions on core workloads.
- [ ] Results separate load, prefill and decode.
- [ ] QSA results separate projection, compressed-index maintenance, score/top-k selection, gather, sparse attention, state update, allocation and host synchronization.
- [ ] QSA-only and end-to-end PP/TG are reported at short, 2K, 8K, 16K, 32K, 64K, 128K and attempted 262K context tiers where feasible.
- [ ] Post-load, first-small-prefill, first-large-prefill and steady-decode VRAM high-water are reported separately.
- [ ] PLE results separately report cold-cache, warm-cache, major-page-fault, physical bytes/read amplification and steady-state phases for both backends.
- [ ] Whole-model Q4 and named Q3 profiles are benchmarked on actual P4s before the operational default is recorded.
- [ ] Q5/Q8 and other higher-precision results are labeled at component scope unless a complete profile actually fits the host envelope.
- [ ] Modern-GPU, unified-memory and other non-P4 throughput figures are labeled as feasibility evidence only and are never used as Pascal performance predictions or release defaults.
- [ ] Cache-zero, static-hot, static-cache, async and current-step-hybrid modes are distinguishable in the evidence.
- [ ] NUMA placement sweeps report end-to-end decode impact for local, remote and interleaved controls; synthetic bandwidth alone is not accepted as locality evidence.
- [ ] Placement sweeps record the first unsafe/cliff point and the lower safe release setting after the large-prefill canary.
- [ ] Merged upstream llama.cpp/PXQ comparisons are included with caveats about differing formats/runtimes.
- [ ] Temperatures and clocks show no throttling.

## Optional exact n-gram profile

These items are required only if `coding-ngram` is published with v1:

- [ ] Deterministic treatment outputs are byte-identical to ordinary-decode baselines on the fixed corpus.
- [ ] GDN/QSA/PLE/checkpoint state after accepted, rejected, cancelled and edited-context paths matches ordinary replay.
- [ ] Copy/edit/structured-transform and novel-code negative controls are reported separately.
- [ ] Proposed/accepted tokens, span distribution, verification cost, PLE I/O/read amplification and automatic-disable behavior are published.
- [ ] Low-acceptance workloads have no material regression or automatically disable the profile.

An unfinished or unsuccessful optional profile is omitted from release claims; it does not fail core v1.

## Serving and operations

- [ ] OpenAI-compatible streaming works.
- [ ] Cancellation releases work, QSA scratch/state and memory.
- [ ] Request limits and timeouts are enforced.
- [ ] Health/readiness detects failed workers, unsafe placement, retained workspace growth, unexpected fallback and failed canaries.
- [ ] Docker/Compose deployment works from a clean host.
- [ ] Logs expose selected kernels, quant/profile, placement, QSA workspace/context policy, storage backend and scheduler.
- [ ] Prometheus-compatible or machine-readable metrics exist.
- [ ] 8-hour soak passes.
- [ ] OOM, first-large-prefill growth, QSA/top-k workspace exhaustion, placement overcommit, malformed model/PLE, failed fill and worker restart tests pass.

## Documentation

- [ ] README and operations guide are current.
- [ ] Every accepted ADR matches implementation.
- [ ] Required issue backlog, including #73 and #76, is closed or deferred through an explicit post-v1 decision.
- [ ] Optional #74 status is explicit and does not block core v1.
- [ ] Known limitations and safe defaults are documented.
- [ ] A release tag and changelog exist.
