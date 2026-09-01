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

## Hardware gates and retention

The ordinary hosted workflow is H0/H1 only. Both Tesla P4 cards are installed on Gorilla, while
cooling qualification and any self-hosted runner remain pending. The local hardware gate first records inventory and calls
`scripts/check_hardware_inventory.py`; zero GPUs, a non-6.1 device, or fewer than two devices for a
dual-P4 level is a hard failure before pytest starts. Verified environment flags then unlock the
`sm61` and `dual_p4` markers. Otherwise pytest reports an explicit deferred skip reason.

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
- Run telemetry reported every routed layer as `mixed_avx2`, eight actual worker threads with
  verified affinity, CPU execution, and a file-backed GGUF expert source. This is an observation
  from the retained run record; an H2 gate assertion for each layer and worker/affinity field is
  still required before treating it as a qualification claim. No SSD or GPU expert execution
  occurred.
- Runtime allocation reached 5.86 GiB on GPU 0 with about 2.20 GiB free after initialization.
  Peak sampled temperature was 54 C at roughly 25 W under the intentional 75 W power limit.
- The inventory at clean commit `0cb4bb3500` (patch-equivalent to superseded stacked commit
  `7594bce5b4`) recorded two ECC-disabled Tesla P4 cards on separate PCI roots and NUMA nodes,
  plus the PCIe Gen3 x4 NVMe on node 0. Seven short single-card tests and one dual-card discovery
  test passed. Active-load PCIe link qualification and cooling remain unqualified.

The complete 77.0 GiB expert bank was mapped/file-backed for this slice. No evidence was collected
that all expert pages were prefaulted into DDR4 or protected from swap, so this run must not be
described as proving resident/no-swap expert-bank behavior. The H2 run also did not provide cold-cache
fault/read-amplification or model-shard cryptographic-hash evidence.

The clean-branch commit mappings above establish source equivalence but do not replace a rerun at
the clean branch tip; that rerun is still required for strict reproducibility.

These were bounded correctness/integration runs, not repeated steady-state decode benchmarks.
The reported approximately 0.18 output tokens per second divides two output tokens by the complete
prompt-plus-generation call and must not be presented as steady-state decode TPS. Cold-cache PLE
fault/read-amplification, ECC-on comparison, long-context, and sustained thermal evidence remain
separate required profiles.
