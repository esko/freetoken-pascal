# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

**`bench_ple_io.py`** — H0 dedicated-artifact PLE evidence. It runs identical row
batches through mmap and positional-read backends and records independent cold, warm
and steady phases, including physical block-device provenance when `--linux-probe` is
explicitly supplied. It is not a throughput or P4 benchmark.

```bash
PYTHONPATH=python python benchmarks/bench_ple_io.py \
    --artifact /srv/freetoken-pascal/ple/table --linux-probe \
    --output results/ple-io.json
```

**`bench_gdn_pascal.py`** — bounded, kernel-only H2 timing for the explicit Pascal FP32
GDN recurrence against the independent Torch reference. The defaults use the Qwen3.8
`D=128/HK=16/HV=48` geometry and hard-limit the warmup and sample counts. It does not
measure full-model GDN or end-to-end decode, and its raw observational JSON is not release
evidence accepted by `scripts/validate_evidence.py`. Capture `nvidia-smi` temperature,
power, clocks, throttle reasons, ECC mode, and the repository hardware inventory beside it.

The CUDA image intentionally has no Git executable. Obtain the commit from the exact host
checkout mounted into the container and inject it explicitly:

```bash
commit=$(git rev-parse HEAD)
nvidia-smi --query-gpu=index,uuid,ecc.mode.current,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,clocks_throttle_reasons.active --format=csv
docker run --rm --gpus device=0 \
    -e FREETOKEN_BENCHMARK_COMMIT="$commit" \
    -e FREETOKEN_DISABLE_KERNEL_CACHE=1 \
    -v "$PWD:/workspace/freetoken-pascal" \
    -w /workspace/freetoken-pascal freetoken-pascal:cuda126 \
    bash -lc 'PYTHONPATH=python python benchmarks/bench_gdn_pascal.py \
      --tokens 1 --warmups 2 --repeats 5 --output results/hardware/gdn-pascal-t1.json'
```

**`bench_gdn_model_pascal.py`** — bounded single-layer model-boundary H2 A/B timing for the same
explicit Pascal and Torch-reference implementations. It covers the real Qwen4ExpGatedDeltaNet BF16
projection, causal convolution, gate, recurrence, gated norm, and output projection for both
prefill and one-token decode, with nonzero carried state and correctness checked first. The
fixed Qwen3.8 geometry is `hidden=2560/D=128/HK=16/HV=48`; it is explicit-only, thermally constrained,
and not a complete-model or release benchmark. The host wall clock includes Python dispatch and
synchronous Pascal metadata validation when the cold fallback is selected; scheduler-issued
metadata proofs avoid that repeated device-to-host validation. Device-scoped CUDA events include
device work plus any stream idle caused by synchronous dispatch. Paired sample order alternates.
The report's `selected_behavior.metadata_validation` identifies the selected path. The
`metadata_proof_timings` block separately labels allocator-cold proof construction (after
emptying PyTorch's caching allocator, without resetting the CUDA runtime), allocator-warm proof
reissues, and warm semantic proof validation
for both prefill and decode. Keep the
run short, and capture `nvidia-smi` telemetry separately:

Every timed sample also contains phase-level CUDA-event and host-wall intervals plus aggregate
`phase_statistics`. The required phases are layer total, projection, convolution, qkv preparation,
gate, recurrence device work, norm, and output projection. Pascal samples additionally report
metadata validation and combined adapter/launch host overhead; the latter's CUDA interval overlaps
the recurrence interval and must not be added as separate device work.

```bash
commit=$(git rev-parse HEAD)
nvidia-smi --query-gpu=index,uuid,ecc.mode.current,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,clocks_throttle_reasons.active --format=csv
docker run --rm --gpus device=0 \
    -e FREETOKEN_BENCHMARK_COMMIT="$commit" \
    -e FREETOKEN_DISABLE_KERNEL_CACHE=1 \
    -v "$PWD:/workspace/freetoken-pascal" \
    -w /workspace/freetoken-pascal freetoken-pascal:cuda126 \
    bash -lc 'PYTHONPATH=python python benchmarks/bench_gdn_model_pascal.py \
      --prefill-tokens 8 --warmups 2 --repeats 5 \
      --output results/hardware/gdn-model-pascal-t8.json'
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.
