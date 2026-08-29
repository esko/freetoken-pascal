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
- [ ] GDN, QSA, hyperconnections, router, shared expert and PLE pass reference checks.
- [ ] QSA selected blocks/rows, sparse-attention output and recurrent/index state pass independent dense-equivalent, sparse-region and context-boundary checks.
- [ ] 32K and 128K continuations are coherent and state-correct.
- [ ] 262K qualification is either passed or explicitly documented as unsupported with measured reason.
- [ ] Checkpoint/restore matches replay.
- [ ] Tool/JSON output probes pass.
- [ ] Q4/Q3 and component-format routing, long-context retrieval, tool-call, structured-output and coding quality gates pass within their declared budgets.
- [ ] Bad quant IDs, codecs, offsets, strides, shapes and profiles fail closed.

## Execution

- [ ] Cache-zero CPU-backed mode works.
- [ ] AVX2 CPU expert backend passes parity.
- [ ] The complete quantized expert bank is available through the documented DDR4 serving representation, remains resident under the required no-swap policy, and does not create an uncontrolled duplicate full-bank copy.
- [ ] Pascal DP4A/format-specific GPU expert backends pass parity.
- [ ] The exact full-softmax `topk=10` router and permanent reference fallback pass parity.
- [ ] The PLE uses dedicated NVMe shard files that contain no unrelated model tensors.
- [ ] mmap and positional-read PLE backends pass identical row and failure-path tests.
- [ ] Supported PLE backends request random-access advice and report whether it was applied.
- [ ] PLE planning is adaptive: direct and vectorized dedupe/order/coalesce paths are correct and observable.
- [ ] PLE prefetch is bounded and asynchronous where enabled.
- [ ] The full PLE is not permanently pinned; spare DDR4 remains available to the Linux page cache.
- [ ] One P4 works.
- [ ] #73 placement planning, observed post-load and post-first-large-prefill allocation, startup canary, automatic backoff and fail-readiness gates pass with a documented reserve.
- [ ] #76 QSA score/top-k/gather workspaces are bounded/reused, capacity-checked before launch and cannot grow across repeated requests.
- [ ] QSA/top-k workspace exhaustion produces controlled backoff/error rather than process abort or corrupt output.
- [ ] Two P4s work under the selected measured ownership/trunk policy; layer-owned, disjoint, replicated and TP/split candidates are compared where feasible.
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
- [ ] Cache-zero, static-hot, static-cache, async and current-step-hybrid modes are distinguishable in the evidence.
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
