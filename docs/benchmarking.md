# Benchmarking

## Objective

Measure throughput improvements that survive complete serving, not just isolated kernels, while detecting unsafe placement, hidden fallback, I/O amplification and workload-specific regressions.

## Mandatory metadata

Each result record contains:

- runtime and commit SHA;
- upstream manifest hash;
- build container/toolchain;
- driver and CUDA versions;
- CPU, memory population, NUMA topology;
- GPU names, bus IDs, clocks, power limits and temperatures;
- model profile, filename, SHA-256, GGUF metadata and tensor-type census;
- canonical placement-profile identity and digest, including the bound plan/backoff schema version;
- dedicated PLE artifact checksum, row codec, geometry and shard identity;
- context, batch, ubatch, concurrency and sampling;
- planned and observed placement, safety reserve, PLE backend/advice, page-cache state, cache size/policy, pinned-memory policy and q-star mode;
- selected CPU/GPU/router kernels and all fallback reasons;
- startup-canary result and any automatic backoff steps;
- prompt ID/hash and token counts;
- raw repetitions, arm order, median and dispersion.

## Core metrics

- load, extraction/conversion and warm-up time;
- startup-canary PP/TG and placement state;
- PP512, PP4096 and PP8192;
- TG128 and TG512;
- 8K coding prompt to 512 output;
- 32K coding prompt to 512 output;
- 128K long-context continuation;
- PLE cold-cache, warm-cache, major-page-fault and steady-state phases reported independently;
- PLE logical rows, unique rows, packed bytes requested, application bytes read, block-device bytes read, coalesced ranges, planner time, prefetch hits and read-amplification ratio;
- cache hit/miss/oracle-hit rate;
- experts per token on CPU, cached GPU and current-step GPU;
- H2D bytes and duration;
- CPU MoE, GPU MoE and merge critical-path duration;
- router-only duration and complete layer/token duration;
- remote NUMA and cross-GPU traffic;
- planned/allocated/reserved/free VRAM by category;
- RAM, expert residency, page cache, page faults, swap and NVMe reads;
- output validity and task result.

## Whole-model profiles

1. `reference-q4`: pinned `UD-Q4_K_XL`, ordinary decode, cache-zero available.
2. `throughput-q3`: one pinned, census-verified Q3 artifact selected by issue #17.

Q4 and Q3 use the same prompts, sampling, context, cache/placement mode and server settings where their geometry permits. A different imatrix, tokenizer/template, tensor promotion policy or model revision is recorded as a confounder rather than hidden.

Q5, Q8 and other higher-precision formats are normally component-level experiments for continuously active or sensitive tensors. They are not reported as whole-model profiles unless a complete artifact actually fits the 128 GB operating envelope with release headroom.

## Core benchmark modes

1. CPU-backed reference, cache zero.
2. Static hot expert cache from a reproducible trace.
3. Static cache under each dual-P4 ownership candidate.
4. Async future-token fill.
5. Current-step forced CPU/GPU splits.
6. Current-step automatic split.
7. Best measured one-P4 and dual-P4 policy inside the #73 safe placement envelope.
8. Merged upstream llama.cpp Qwen3.8 reference.
9. PXQ/llama reference when Qwen3.8 support is available.
10. Explicitly experimental SSD expert execution, if implemented, reported separately from every release mode.

Dynamic-cache and q-star claims always include cache-zero and static-hot controls. A speedup relative only to a deliberately cold dynamic cache is not sufficient.

## PLE benchmark contract

Run the PLE suite separately for mmap and positional-read backends against the same dedicated shard bytes and row sequence.

- mmap random-access mode reports whether `MADV_RANDOM` was accepted;
- positional-read random-access mode reports whether `POSIX_FADV_RANDOM` or its equivalent was accepted;
- direct small-decode and vectorized dedupe/order/coalesce planner modes are benchmarked independently;
- automatic planner selection is compared with the best forced mode at decode, prefill and multi-token verification widths;
- cold-cache runs begin from a recorded cache state and are never mixed statistically with warm or steady-state samples;
- warm-cache runs rely on ordinary Linux page-cache behavior rather than permanently pinning the full PLE;
- physical block-device bytes are reported alongside logical packed-row bytes and application reads;
- major faults and read amplification are release metrics, not troubleshooting notes.

The vectorized planner remains the default, and adaptive threshold selection carries no performance claim until a reproducible benchmark provides evidence.

A large unexplained amplification or unrelated model-weight I/O invalidates the serving configuration.

### Injected PLE I/O evidence

The H0 evidence primitive in `freetoken.ple_io_evidence` consumes two cumulative
counter snapshots for one explicitly labeled `cold`, `warm`, or `steady` phase.
`PLEIOEvidenceRecorder` accepts an injected counter provider so synthetic tests and
the eventual NVMe probe share the same delta and validation logic.  Each snapshot
keeps major faults, logical packed-row bytes, application bytes, and application
read calls separate from physical block-device bytes.

Physical bytes are valid only when both snapshots include an independently sampled,
non-decreasing block-device counter, a stable unambiguous device identity, and an
exact allowlisted source kind: `block-device-stat`, `sysfs-block-stat`, `iostat`,
`blktrace`, or `nvme-cli`. Any command/path detail is carried separately and must
remain stable across the phase. `/proc/self/io`, rusage, or any other process-only
read counter is never promoted to physical I/O. Counter regressions,
identity/source changes, invalid phases, missing physical provenance, and a zero
logical-byte denominator fail closed. The resulting read-amplification ratio is
`physical_block_device_bytes / logical_packed_bytes` for that phase.

`PLEIOEvidenceRecorder` is transactional at both phase boundaries: it samples and
validates the begin snapshot before activating a phase, and consumes the active
phase only after end sampling and delta measurement succeed. A provider or
validation failure therefore leaves the phase active (or inactive for a failed
begin) so the caller can retry with a fresh snapshot; a successful phase is
recorded exactly once.

This H0 primitive does not drop caches, issue privileged mutations, or claim that a
host process counter measures device traffic. The later NVMe harness owns collection
of the external block-device snapshots and must attach its device identity and raw
counter source to every phase result.

On Linux, `LinuxPLEBlockCounterProbe` is the reference external-counter adapter for
an explicit dedicated PLE payload file. It opens the payload without following symlinks,
pins its identity with `fstat`, resolves its `st_dev`
`/sys/dev/block/<major>:<minor>`, treats a partition as terminal, follows a single
sysfs `slaves` entry for non-partition mappings, and rejects missing, cyclic,
virtual, or multi-device mappings. It reads field 3 of the terminal `/sys/.../stat`
line (the documented cumulative `sectors read` field, one-based) and converts each
sector using the Linux ABI's fixed 512-byte unit, independently of
`queue/logical_block_size`. The returned `PLEIOCounters` carries
`physical_source=sysfs-block-stat`, a terminal identity in `major:minor/name` form,
and stable `field=3(sectors-read):sector-size=512` detail. A process counter
provider may be injected for the other snapshot fields, but `/proc/self/io` is
never consulted or promoted by this probe. For example:

```python
probe = LinuxPLEBlockCounterProbe("/srv/freetoken-pascal/ple/table.bin")
recorder = PLEIOEvidenceRecorder(probe.sample)
```

The payload and terminal sysfs attributes remain pinned or identity-checked through each
sample so a path, device, or counter replacement fails instead of changing provenance.
The probe is read-only and does not drop caches. An overlay/tmpfs payload, a
non-regular payload, an unavailable sysfs mapping, malformed stat or partition
marker, and any ambiguous backing device fail closed before evidence is emitted.

## Placement-cliff benchmark contract

Issue #73 sweeps context/state allocation, batch/ubatch, ordinary tensor placement and cache slots one controlled step at a time.

For every step record:

- planned and observed VRAM categories;
- free/reserved headroom;
- allocation retry/failure, managed-memory, spill and fallback telemetry;
- deterministic startup-canary result;
- PP/TG and end-to-end request time;
- clocks, power and temperature.

The evidence bundle includes the last safe point, the first unsafe/cliff point and the lower release setting with its retained margin. The release default is not the maximum configuration that merely starts.

## Dual-P4 policy matrix

Compare, where feasible:

- layer-owned expert caches;
- disjoint expert ownership;
- replicated hot caches;
- trunk layer split;
- conventional TP=2;
- trunk split/TP combined with each expert-cache policy.

Use identical safe aggregate expert-cache bytes and context/state requirements. Report per-GPU cache capacity/hit rate, PCIe/NCCL traffic, CPU partial work, merge time, remote NUMA traffic and failure behavior. If candidates are within noise, prefer the simpler policy with the clearer fallback and recovery story.

## Optional exact n-gram profile

If issue #74 is implemented, run a separate paired suite with ordinary decode and `coding-ngram` using identical request bytes, model/profile, seed, sampling and server placement.

The corpus includes:

- exact Python/code copying;
- edit/patch-like output with large unchanged spans;
- exact JSON copying;
- structured tool/config transformation;
- novel-code generation with no expected reusable span;
- low-acceptance long-running negative control.

Record exact output hashes, wall time, server decode time, proposed/accepted/rejected tokens, span distribution, verification batches, PLE logical/physical bytes, major faults, read amplification and automatic-disable decisions. A result on copy-heavy prompts cannot be generalized to novel generation.

This profile is omitted from release claims when it is unfinished, output-divergent or unable to avoid negative-control regressions.

## Existing H0 CPU evidence

For the H0 mixed CPU slice, collect Gorilla-relevant normal and promoted raw route/thread sweeps with:

```bash
PYTHONPATH=python python scripts/bench_q4_k_threaded.py --profile normal --output normal.json
PYTHONPATH=python python scripts/bench_q4_k_threaded.py --profile promoted --output promoted.json
```

The harness uses the production 2560-hidden/640-intermediate geometry by default, retains route widths 1/2/4/8/10 and every requested thread count, and records the selected quant kernels and fallback. Its `synthetic` and `observation_only` fields are deliberate: raw samples are evidence for later target-host analysis and make no performance claim.

The focused real-artifact target-CPU benchmark uses one selected expert, one token and one route from the Qwen3.8 Q4 artifact. Run it once for the normal layer-0 geometry and once for the promoted layer-2 Q5_K/Q8_0 geometry, using the bounded range cache:

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

This is preliminary H0 evidence, not a full-engine or P4 result. The report retains every warmup and raw sample, exact commit and command, CPU/ISA, BLAS environment and process affinity, manifest revision, selected range hashes, kernel/fallback telemetry and a per-sample correctness comparison. It reports two independent descriptive comparisons: dense-resident (dequantization once outside timing) and a cold full-reference procedure. The latter includes source/layout validation, packed byte/view setup, hashing, dequantization and dense execution inside every reference sample. Their medians and ratios are never merged.

The benchmark requires at least five warmups, fails if forced AVX2 is not actually selected, and fails rather than reporting statistics when either reference comparison mismatches.

Router qualification additionally alternates forced `torch-reference`, forced `pascal-fused` and `auto` modes with cache, scheduler, model, quant, prompt and sampling held fixed. A router microbenchmark alone cannot enable the fused default; the same-workload end-to-end result must improve outside run-to-run noise.

Use the exact same model bytes where the runtime permits. If formats differ, state that the comparison is not a codec-controlled A/B.

## Statistics

- at least one untimed warm-up;
- at least five measured repetitions for release claims;
- alternate A/B order to control thermal and cache drift;
- report median, min/max and coefficient of variation;
- discard only runs with a documented external failure;
- retain raw samples;
- retain negative controls and unsuccessful policies.

## Performance gates

An optimization ships enabled only if:

- correctness gates pass;
- its target workload improves by at least 5% outside run-to-run noise;
- no core workload regresses by more than 5% unless the policy auto-disables there;
- memory use remains within the documented operating envelope and #73 reserve;
- no hidden fallback, spill or unexplained read amplification is active;
- the selected policy is visible in logs and metrics.

Mixed-precision candidates also pass fixed routing-decision, long-context retrieval, tool-call, structured-output and coding probes before they may become defaults.

The final product must demonstrate a measurable end-to-end gain over its cache-zero FreeToken fallback on the target coding workload. The project does not promise an absolute TPS before hardware qualification.
