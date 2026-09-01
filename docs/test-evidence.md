# Test fixtures and evidence

Issue work must leave machine-checkable evidence without allowing synthetic data or hosted CI to
masquerade as P4 qualification. The fixture and schema infrastructure under `tests/fixtures/` and
`schemas/` is the shared contract for later correctness, kernel, cache, and benchmark work.

## Fixture policy

All repository fixtures are synthetic and Apache-2.0 licensed. The tiny Qwen4 configuration has
four experts and no vision or MTP fields. Issue #11 uses it for CPU reference equations and H1
configuration coverage; real runtime logits and generation remain H2-gated.

Regenerate the binary GGUF fixtures with:

```bash
PYTHONPATH=python python scripts/generate_test_fixtures.py
```

The manifest pins every resulting byte stream by SHA-256. Valid inputs cover heterogeneous F32,
Q4_0, and Q6_K tensor layouts. Negative inputs cover bad magic, truncated metadata, unknown and
known-but-unsupported quant identifiers, invalid quant block dimensions, misalignment, and
out-of-range offsets. The reader must reject all of them before tensor bytes reach a kernel.

Routing traces use `(layer, expert)` as the cache identity. `locality-positive.json` must outperform
the cyclic adversarial trace under the host LRU simulator, while capacity zero remains the stable
all-miss control. Later LFRU and q-star changes must retain these controls and add their own traces.

## Evidence schemas

Validate every example or result bundle with:

```bash
python scripts/validate_evidence.py [result.json ...]
```

The versioned schemas cover benchmark results, hardware inventory, quantization census, and
correctness comparisons. Benchmark evidence records the exact commit, model checksum, quant census,
flags, prompt hash, context, temperature, clocks, raw repeated runs, and the kernels/cache/split/device
behavior actually selected. Correctness evidence requires an independent reference and numeric or
exact comparison; fluent text alone is not evidence.

Every example carries `evidence_status: synthetic`. Real runs must use `measured` and replace every
placeholder checksum and hardware field with captured values.

The QSA H2 context-sweep contract is `qwen38-qsa-h2-evidence.schema.json`.
Its synthetic fixture covers the registered `qsa_sparse` backend at contexts 128, 512, and 2048, with prefill and decode raw samples, composite phase sums, workspace plans, and allocator checkpoints.
The measured producer is intentionally limited to one tiny QSA layer and one request per context/phase, and records startup-canary, cancellation/restore, chunking, graph-capture, sustained-load, thermal, performance, and full-model limitations explicitly.

## Hardware gates and retention

The ordinary hosted workflow is H0/H1 only. Both Tesla P4 cards are installed on Gorilla, while
cooling qualification and any self-hosted runner remain pending. The local hardware gate first records inventory and calls
`scripts/check_hardware_inventory.py`; zero GPUs, a non-6.1 device, or fewer than two devices for a
dual-P4 level is a hard failure before pytest starts. Verified environment flags then unlock the
`sm61` and `dual_p4` markers. Otherwise pytest reports an explicit deferred skip reason.
Measured inventory may additionally bind the immutable `ecc-on` or `ecc-off` profile. When a
profile is requested, every GPU's current and pending ECC state must agree with it; mixed modes or
a profile mismatch fail before device tests start. The current Gorilla operating profile is
`ecc-off`. Earlier ECC-on observations remain separate historical evidence.

Artifact retention is:

- 14 days for H0/H1 compile and fixture evidence;
- 30 days for H2/H3 hardware investigations;
- 90 days for H4 release candidates and final qualification bundles.

The current hardware workflow uses 30 days. Issue #29 owns the separate H4 release retention and
permanent release attachments.

## Current bounded P4 evidence

On 2026-09-01 Gorilla completed the first real Qwen3.8 GGUF cache-zero vertical H2 run. The
selected model revision was `c8b5954a88c2775c546b92593eda40ea041d3176`; all four manifest shard
sizes matched. This gate did not stream-hash the model shard bytes, so the revision and size
match are recorded without claiming a cryptographic content check. The dedicated IQ4_NL PLE artifact had SHA-256
`dd55c28902f38cd88134b2a569c51282c5ffce30080487e1a645740115c56cc3`.

- `mmap`: evidence captured at superseded stacked commit `0b5cadb0a7`; its exact patch is
  reproduced by clean-branch commit `d924ba4b9a` (diff-equivalent): two output tokens, 11.192 seconds for the complete
  five-token prompt plus generation call, 96 logical/unique PLE rows, 8,640 packed bytes,
  `MADV_RANDOM`, 35 minor faults, zero observed major faults, and no block-device reads because the
  PLE was already warm in Linux page cache.
- `pread`: evidence captured at superseded stacked commit `3b52df561c`; its exact patch is
  reproduced by clean-branch commit `c34648dbbe` (diff-equivalent): the identical two output token IDs, 11.120 seconds
  for the complete prompt plus generation call, the same 96 rows and 8,640 packed bytes through 96
  positional reads, `POSIX_FADV_RANDOM`, 23 minor faults, and zero observed major faults or
  block-device reads on the warm cache.
- Clean-tip `pread` qualification: commit `3c9dc0422a` passed the complete single-P4 gate on
  Gorilla. The gate stream-hashed all four model shards and the dedicated PLE payload, validated
  the generated document against `qwen38-gguf-cache-zero-h2-evidence.schema.json`, produced the
  identical token IDs `[201519, 8691]`, and retained the raw document at
  `results/hardware/qwen38-gguf-cache-zero-h2.json` on Gorilla for the 30-day H2 retention window.
  Startup took 130.748 seconds and the complete five-token prompt plus two-token generation call
  took 114.010 seconds. Because whole-artifact verification displaced useful expert pages before
  inference, this is cold-after-verification correctness evidence and not a steady-state TPS result.
  PLE telemetry recorded 96 requested/unique/sorted rows, 96 positional reads, 8,640 packed bytes,
  20 minor faults, zero major faults, and zero observed physical storage-read bytes from the warm
  page cache.
- The clean-tip gate asserted direct AVX2 kernels for every one of the 48 routed layers, eight actual
  worker threads with verified affinity to CPUs 0-7, CPU execution, and a file-backed GGUF expert
  source. No SSD or GPU expert execution occurred.
- Runtime allocation reached 5.86 GiB on GPU 0 with about 2.20 GiB free after initialization.
  Peak sampled temperature was 54 C at roughly 25 W under the intentional 75 W power limit.
- The inventory at clean commit `0cb4bb3500` (patch-equivalent to superseded stacked commit
  `7594bce5b4`) recorded two ECC-disabled Tesla P4 cards on separate PCI roots and NUMA nodes,
  plus the PCIe Gen3 x4 NVMe on node 0. Seven short single-card tests and one dual-card discovery
  test passed. Active-load PCIe link qualification and cooling remain unqualified.
- The identity-hardened bounded `dual-p4-short` producer at `cd8ce1dc44` ran on both ECC-off cards
  without loading a model. One isolated 1 MiB addition per device completed in 0.394 seconds
  (0.483 seconds total), with a 35 C peak and 23.58 W peak under the 75 W limit. The evidence binds
  the dedicated inventory SHA-256
  `d64d287fa25d429dcedcd097ace0c3a99fd5f2d6a5a62301165655848c986d17`, both unique
  UUID/PCI-root/NUMA identities, and explicit non-serving/no-TPS/no-thermal-qualification claims.
  Raw evidence is retained on Gorilla as `results/hardware/qwen38-dual-p4-device.json` with
  SHA-256 `308583cfdcdac106727e25351f930039aee9d1a026cfd3b51c6a806f880f52c9` for the H2/H3
  investigation window.
- The bounded warm-cache producer at `fe6e7ca91c` reused canonical full-H2 artifact
  SHA-256 `740e0c1ab79acf5f5473c70751f645bd6f1e91235cc0b826cb14e18529f16b7e`
  instead of rehashing the four model shards; normal Engine startup still performed the dedicated
  PLE integrity hash. Startup took 151.029 seconds. One deterministic two-token warmup took
  114.488 seconds and included first-use GGUF CUDA extension compilation; the immediately repeated
  identical five-token prompt plus two-token request took 5.244 seconds and returned the same IDs
  `[201519, 8691]`. The measured request read 96 unique PLE rows/8,640 packed bytes through
  `pread`, with zero major faults and zero observed physical storage-read bytes. Peak sampled GPU
  state across the two requests was 52 C and 29.93 W under the 75 W limit. This is a single bounded
  warm-call observation, not decode-only or steady-state TPS. Raw evidence is retained on Gorilla
  as `results/hardware/qwen38-gguf-cache-zero-warm-h2.json` with SHA-256
  `f4d0ad2189e62615554c7f15136921a60af11a8251a1b7e01871bf6b66e06385`; its dedicated
  ECC-off inventory has SHA-256
  `51d6e6654444b5b0983b83f3da5f0525888eccb5c1de894c1037bcafc6745a98`.
  This retained observation predates the strengthened producer contract and is historical only:
  same-size model content drift was not excluded cryptographically during that short run.
- The identity-hardened warm producer at `cd8ce1dc44` then rehashed all four current model shards
  against the canonical full-H2 identities before GPU acquisition. It retained the mandatory PLE
  integrity pass and completed under the separate-process 300-second GPU-ownership watchdog.
  Startup took 149.908 seconds, the deterministic warmup took 114.457 seconds, and the identical
  measured request took 5.067 seconds; both returned `[201519, 8691]`. The measured request again
  read 96 unique PLE rows/8,640 packed bytes through `pread`, with zero major faults and zero
  observed physical storage-read bytes. Peak sampled state was 52 C and 30.31 W. Raw evidence has
  SHA-256 `386afae5680e4f7ff8f2717f2e95e79137fe5c3f05fac572001fa1dc0747795b`; its dedicated
  inventory has SHA-256 `ca6fc4b81d48b2a38e456f2ae187844f863dc4c2009414e786caab7f4f159156`.
- The bounded router-only H2 gate at `ce5b08f8b7` first passed 19 selected real-CUDA
  router edge-case tests, including finite ties, non-finite/all-invalid rows, padded rows,
  CUDA-graph capture and the unchanged `auto` fallback. It then completed the exact 108-case
  matrix spanning BF16/FP16/FP32, 1/2/4/8/32/128 rows, padding/random/exceptional inputs and
  both renormalization modes. All candidate cases launched, all expert IDs matched exactly,
  every weight comparison passed, maximum absolute error was
  `1.1920928955078125e-7`, and maximum relative RMS error was
  `1.4996938091371703e-7`. Across the deliberately heterogeneous matrix, the descriptive
  median synchronized observation was 687,104 ns for the Torch reference and 165,529 ns for
  the forced Triton candidate; this is not a model-level or dispatch-enabling performance
  claim. The candidate's first call may include JIT and reached 3.130 seconds in the first
  recorded case. The 40.001-second matrix ended at 45 C and 24.05 W under the provisional
  75 W cap. Raw ECC-off evidence is retained on Gorilla as
  `results/hardware/qwen38-router-h2-ecc-off.json` with SHA-256
  `2d92de17a785270be0b5815e81d22b255af6290a9ff7b739cd91d5ed9ad672ab`; its dedicated
  inventory has SHA-256 `9dd75ee45bf091658a38040ae8d051cb74c48a35b7fe3226d2f73ea4a0aceb27`.
  `auto` remains the Torch reference pending same-model end-to-end A/B and issue #14's broader
  correctness corpus; no thermal, TPS or dual-P4 policy qualification is implied.

The complete 77.0 GiB expert bank was mapped/file-backed for this slice. No evidence was collected
that all expert pages were prefaulted into DDR4 or protected from swap, so this run must not be
described as proving resident/no-swap expert-bank behavior. The H2 run also did not provide cold-cache
fault/read-amplification evidence. The clean-tip pread rerun did provide model-shard cryptographic
hash evidence; the earlier mmap/pread observations did not.

The earlier clean-branch commit mappings establish source equivalence for the warm mmap/pread
observations. Commit `3c9dc0422a` supplies the independent clean-tip cryptographic and execution
rerun for the pread path; a corresponding clean-tip mmap rerun remains optional follow-up evidence.

These were bounded correctness/integration runs, not repeated steady-state decode benchmarks.
The reported approximately 0.18 output tokens per second divides two output tokens by the complete
prompt-plus-generation call and must not be presented as steady-state decode TPS. Cold-cache PLE
fault/read-amplification, ECC-on comparison, long-context, and sustained thermal evidence remain
separate required profiles.

The first registered-backend QSA context sweep ran on Gorilla at commit `3a8f31738b` under the
ECC-off profile. The 15.442-second bounded run used one tiny Qwen4-Exp QSA layer and one request,
with one prefill and one immediately following decode at 128, 512, and 2,048 tokens. All six
forwards completed through the visible `torch-fp32-reference` path. The observed total CUDA-event
times were 3,324.188/14.756 ms, 1,435.772/7.133 ms, and 9,863.488/12.749 ms for prefill/decode at
the three respective contexts. The 2,048-token prefill attributed 7,704.363 ms to composite
selection, 1,754.887 ms to selected-row attention, and reached a cumulative allocator high-water
of 339.16 MiB. These one-sample tiny-geometry observations locate work for further profiling; the
128-token prefill includes first-use overhead, and none of these numbers is a throughput or stable
performance claim. The run ended at 39 C and 25.03 W under the provisional 75 W cap.

Raw evidence is retained on Gorilla as `results/hardware/qwen38-qsa-h2-ecc-off.json` with SHA-256
`ccb49c293009cb7c44dd19042e8004b3a844b548d3d21062a3c0f5d757ba38e2`; its dedicated inventory
has SHA-256 `a282cc180496e7d0ffbc2d93c89aba8d0ab0f17771b570f0ba0cc126ce88a9c5`. The schema and semantic
validator passed after independently checking that reported driver-free memory never exceeded
driver-total memory. Full-model serving, repeated/warmed samples, chunked prefill, 8K-262K
contexts, optimized QSA, cancellation/restore, placement backoff, sustained thermals, and H3
remain unqualified.

The short-evidence profiles deliberately keep these claims separate:

- a warm-cache single-P4 request must stream-hash the current model split against the canonical
  full-H2 identities before GPU acquisition, and Engine startup still performs its mandatory
  dedicated-PLE integrity validation; a separate-process 300-second watchdog remains armed through
  shutdown and the measured document is atomically published only after successful cleanup; the
  outer 900-second process limit separately allows for CPU/NVMe hashing before GPU acquisition;
- a direct dual-P4 probe measures only two-device identity, bounded allocation/arithmetic, topology,
  and instantaneous telemetry, and must identify itself as non-serving;
- neither profile establishes steady-state TPS, a selected dual-P4 policy, or thermal qualification.

Each short profile binds the exact hardware inventory by SHA-256, including ECC profile, UUIDs,
PCI roots, and NUMA nodes. The full H2 artifact remains authoritative for model, PLE, repository,
and cache-zero execution identity.
