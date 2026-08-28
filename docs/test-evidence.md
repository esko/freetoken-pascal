# Test fixtures and evidence

Issue work must leave machine-checkable evidence without allowing synthetic data or hosted CI to
masquerade as P4 qualification. The fixture and schema infrastructure under `tests/fixtures/` and
`schemas/` is the shared contract for later correctness, kernel, cache, and benchmark work.

## Fixture policy

All repository fixtures are synthetic and Apache-2.0 licensed. The tiny Qwen4 configuration has
four experts and no vision or MTP fields. It is a shape/configuration contract, not a substitute for
the executable tiny model and reference logits owned by Issue #11.

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

The ordinary hosted workflow is H0/H1 only. H2 and H3 are deferred because the Tesla P4 cards and
self-hosted runner are not installed. The manual hardware workflow first records inventory and calls
`scripts/check_hardware_inventory.py`; zero GPUs, a non-6.1 device, or fewer than two devices for a
dual-P4 level is a hard failure before pytest starts. Verified environment flags then unlock the
`sm61` and `dual_p4` markers. Otherwise pytest reports an explicit deferred skip reason.

Artifact retention is:

- 14 days for H0/H1 compile and fixture evidence;
- 30 days for H2/H3 hardware investigations;
- 90 days for H4 release candidates and final qualification bundles.

The current hardware workflow uses 30 days. Issue #29 owns the separate H4 release retention and
permanent release attachments.
