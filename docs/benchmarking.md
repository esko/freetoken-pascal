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
- placement, PLE backend/shard identity, page-cache state, cache size/policy, pinned-memory policy and q-star mode;
- prompt ID and token counts;
- raw repetitions, median and dispersion.

## Core metrics

- load and warm-up time;
- PP512, PP4096 and PP8192;
- TG128 and TG512;
- 8K coding prompt to 512 output;
- 32K coding prompt to 512 output;
- 128K long-context continuation;
- PLE cold-cache, warm-cache, major-page-fault and steady-state phases reported independently;
- PLE logical rows requested, unique rows read, coalesced ranges, prefetch hits, bytes requested and bytes read;
- cache hit/miss/oracle-hit rate;
- experts per token on CPU, cached GPU and current-step GPU;
- H2D bytes and duration;
- CPU MoE, GPU MoE and merge critical-path duration;
- router-only duration and complete layer/token duration;
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
8. Explicitly experimental SSD expert execution, if implemented, reported separately from every release mode.

Run the PLE suite separately for mmap and positional-read backends against the same dedicated shard bytes and row sequence. Cold-cache runs must begin from a recorded cache state and must never be mixed statistically with warm or steady-state samples. Warm-cache runs rely on ordinary Linux page-cache behavior rather than permanently pinning the full PLE. Report major faults and block-device reads for each phase; do not infer physical I/O solely from application-level read calls.

The release quant comparison includes Q4, Q5, Q8 and the shipping CPU formats on actual P4 hardware. Pre-P4 conversion and correctness results may narrow candidates, but cannot establish the final recipe.

For the H0 mixed CPU slice, collect Gorilla-relevant normal and promoted raw
route/thread sweeps with:

```bash
PYTHONPATH=python python scripts/bench_q4_k_threaded.py --profile normal --output normal.json
PYTHONPATH=python python scripts/bench_q4_k_threaded.py --profile promoted --output promoted.json
```

The harness uses the production 2560-hidden/640-intermediate geometry by
default, retains route widths 1/2/4/8/10 and every requested thread count, and
records the selected quant kernels and fallback. Its `synthetic` and
`observation_only` fields are deliberate: raw samples are evidence for later
target-host analysis and make no performance claim.

The focused real-artifact target-CPU benchmark uses one selected expert, one token and
one route from the Qwen3.8 Q4 artifact. Run it once for the normal layer-0 geometry and
once for the promoted layer-2 Q5_K/Q8_0 geometry, using the bounded range cache:

```bash
python scripts/build_target_cpu_native.py \
  --output-dir .cache/freetoken/target-cpu-native

export FREETOKEN_Q4K_NATIVE_LIB="$PWD/.cache/freetoken/target-cpu-native/q4_k_native.so"
export FREETOKEN_MIXED_GEMV_NATIVE_LIB="$PWD/.cache/freetoken/target-cpu-native/mixed_gemv_native.so"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export GOTO_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHONPATH=python python benchmarks/bench_qwen38_real_expert.py \
  --layer 0 --offline \
  --native-build-metadata .cache/freetoken/target-cpu-native/build.json \
  --output results/qwen38-target-cpu-layer0.json
PYTHONPATH=python python benchmarks/bench_qwen38_real_expert.py \
  --layer 2 --offline \
  --native-build-metadata .cache/freetoken/target-cpu-native/build.json \
  --output results/qwen38-target-cpu-layer2.json
python scripts/validate_evidence.py results/qwen38-target-cpu-layer0.json
python scripts/validate_evidence.py results/qwen38-target-cpu-layer2.json
```

This is preliminary H0 evidence, not a full-engine or P4 result. The report retains every
warmup and raw sample, exact commit and command, CPU/ISA, BLAS environment and process
affinity, manifest revision, selected range hashes, kernel/fallback telemetry and a
per-sample correctness comparison. It reports two independent descriptive comparisons:
dense-resident (dequantization once outside timing) and a cold full-reference procedure.
The latter includes source/layout validation, packed byte/view setup, hashing, dequantization,
and dense execution inside every reference sample. Their medians and ratios are never merged.
The benchmark requires at least five warmups, fails if forced AVX2 is not actually selected,
and fails rather than reporting statistics when either reference comparison mismatches.

Router qualification additionally alternates forced `torch-reference`, forced
`pascal-fused` and `auto` modes with cache, scheduler, model, quant, prompt and sampling
held fixed. A router microbenchmark alone cannot enable the fused default; the
same-workload end-to-end result must improve outside run-to-run noise.

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

Mixed-precision candidates also pass fixed routing-decision, long-context retrieval, tool-call and structured-output quality probes before they may become defaults.

The final product must demonstrate a measurable end-to-end gain over its cache-zero FreeToken fallback on the target coding workload. The project does not promise an absolute TPS before hardware qualification.
