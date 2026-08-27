# Benchmarking

## Objective

Measure throughput improvements that survive complete serving, not just isolated kernels.

## Mandatory metadata

Each result record contains:

- runtime and commit SHA;
- upstream manifest hash;
- build container/toolchain;
- driver and CUDA versions;
- CPU, memory population, NUMA topology;
- GPU names, bus IDs, clocks, power limits and temperatures;
- model filename, SHA-256, GGUF metadata and tensor-type census;
- context, batch, ubatch, concurrency and sampling;
- placement, cache size/policy, pinned-memory policy and q-star mode;
- prompt ID and token counts;
- raw repetitions, median and dispersion.

## Core metrics

- load and warm-up time;
- PP512, PP4096 and PP8192;
- TG128 and TG512;
- 8K coding prompt to 512 output;
- 32K coding prompt to 512 output;
- 128K long-context continuation;
- PLE cold versus warm;
- cache hit/miss/oracle-hit rate;
- experts per token on CPU, cached GPU and current-step GPU;
- H2D bytes and duration;
- CPU MoE, GPU MoE and merge critical-path duration;
- remote NUMA traffic;
- VRAM, RAM, page faults and NVMe reads;
- output validity and task result.

## Benchmark modes

1. CPU-backed reference, cache zero.
2. Static hot expert cache.
3. Async future-token fill.
4. Current-step adaptive split.
5. Best measured dual-P4 policy.
6. llama.cpp Qwen4 Q4_K_XL reference.
7. PXQ/llama reference when Qwen4 support is available.

Use the exact same model bytes where the runtime permits. If formats differ, state that the comparison is not a codec-controlled A/B.

## Statistics

- at least one untimed warm-up;
- at least five measured repetitions for release claims;
- alternate A/B order to control thermal and cache drift;
- report median, min/max and coefficient of variation;
- discard only runs with a documented external failure;
- retain raw samples.

## Performance gates

An optimization ships enabled only if:

- correctness gates pass;
- its target workload improves by at least 5% outside run-to-run noise;
- no core workload regresses by more than 5% unless the policy auto-disables there;
- memory use remains within the documented operating envelope;
- the selected policy is visible in logs and metrics.

The final product must demonstrate a measurable end-to-end gain over its cache-zero FreeToken fallback on the target coding workload. The project does not promise an absolute TPS before hardware qualification.
