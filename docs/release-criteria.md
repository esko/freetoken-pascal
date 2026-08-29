# Release criteria

FreeToken-Pascal v1 is complete only when every required item is evidenced.

## Build and provenance

- [ ] Clean clone builds from documented commands.
- [ ] CUDA 12.6 build contains `sm_61` code and no mandatory newer-architecture path.
- [ ] Upstream commits, copied files and local modifications are pinned.
- [ ] Apache notices and source headers are preserved.
- [ ] Container image and source revision are identifiable.

## Model correctness

- [ ] Qwen3.8-Flash-Next text model loads from the selected Q4_K_XL artifact.
- [ ] Tensor census and checksums match the recorded release manifest.
- [ ] GDN, QSA, hyperconnections, router, shared expert and PLE pass reference checks.
- [ ] 32K and 128K continuations are coherent and state-correct.
- [ ] 262K qualification is either passed or explicitly documented as unsupported with measured reason.
- [ ] Checkpoint/restore matches replay.
- [ ] Tool/JSON output probes pass.
- [ ] Mixed-precision routing, long-context retrieval, tool-call and structured-output quality gates pass.
- [ ] Bad quant IDs, offsets, strides and shapes fail closed.

## Execution

- [ ] Cache-zero CPU-backed mode works.
- [ ] AVX2 CPU expert backend passes parity.
- [ ] The complete quantized expert bank is loaded and pre-faulted into DDR4, remains resident under the required no-swap policy, and is the only steady-state source for expert execution/cache fills.
- [ ] Pascal DP4A/format-specific GPU expert backends pass parity.
- [ ] The PLE uses dedicated NVMe shard files that contain no unrelated model tensors.
- [ ] mmap and positional-read PLE backends pass identical row and failure-path tests.
- [ ] PLE reads are batched, deduplicated and sorted, with bounded asynchronous prefetch.
- [ ] The full PLE is not permanently pinned; spare DDR4 remains available to the Linux page cache.
- [ ] One P4 works.
- [ ] Two P4s work with measured disjoint expert ownership and a conventional TP comparison.
- [ ] Static cache works.
- [ ] Async fill works.
- [ ] Concurrent CPU/GPU partial execution works.
- [ ] q-star autotuner selects a safe policy under contention.
- [ ] Prefill policy handles forwards wider than the cache.
- [ ] Expert-bank swap is absent and any steady-state PLE NVMe reads are measured and explained.

## Performance

- [ ] Raw data and commands are published as release artifacts.
- [ ] The best policy improves end-to-end coding throughput over cache-zero FreeToken by at least 5% outside noise.
- [ ] The auto policy avoids more than 5% regressions on core workloads.
- [ ] Results separate load, prefill and decode.
- [ ] PLE results separately report cold-cache, warm-cache, major-page-fault and steady-state phases for both backends.
- [ ] Q4, Q5, Q8 and shipping CPU formats are benchmarked on actual P4s before the final quant recipe is recorded.
- [ ] llama.cpp/PXQ comparison is included with caveats about differing formats.
- [ ] Temperatures and clocks show no throttling.

## Serving and operations

- [ ] OpenAI-compatible streaming works.
- [ ] Cancellation releases work and memory.
- [ ] Request limits and timeouts are enforced.
- [ ] Health checks detect failed workers.
- [ ] Docker/Compose deployment works from a clean host.
- [ ] Logs expose selected kernels, placement and scheduler.
- [ ] Prometheus-compatible or machine-readable metrics exist.
- [ ] 8-hour soak passes.
- [ ] OOM, malformed model, failed fill and worker restart tests pass.

## Documentation

- [ ] README and operations guide are current.
- [ ] Every accepted ADR matches implementation.
- [ ] Issue backlog is closed or deferred through an explicit post-v1 decision.
- [ ] Known limitations and safe defaults are documented.
- [ ] A release tag and changelog exist.
