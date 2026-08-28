# Qwen3.8 GGUF contract and census

Issue #12 selects two artifacts from `unsloth/Qwen3.8-Flash-Next-GGUF` at immutable
revision `c8b5954a88c2775c546b92593eda40ea041d3176`:

- `UD-Q4_K_XL`, four shards, 1,224 tensors;
- `UD-Q3_K_XL`, three shards, 1,224 tensors.

Exact shard sizes and LFS SHA-256 values are in `manifests/qwen38-gguf.json`. The first
shard of each variant contains the complete model and tokenizer metadata and zero tensors.
This is valid and must not be rejected or used as evidence that the model has no weights.

## Census

Run a payload-verified census after downloading every complete shard:

```bash
PYTHONPATH=python python scripts/gguf_census.py \
  /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --artifact-manifest manifests/qwen38-gguf.json \
  --variant UD-Q4_K_XL \
  --output results/qwen38-q4-census.json
```

For review of sparse logical files made from real headers, add
`--trust-declared-sha256`. That output is labeled `artifact-metadata`, and each shard hash
is labeled `declared`; it is never reported as measured evidence. Release evidence must
omit that flag and hash every payload byte.

The selected Q4 tensor counts are BF16 24, F32 557, IQ4_NL 1, Q4_K 94, Q5_1 43,
Q5_K 2 and Q8_0 503. The selected Q3 counts are BF16 24, F32 557, IQ3_XXS 94,
IQ4_NL 44, IQ4_XS 2, Q6_K 1 and Q8_0 502. The census records every tensor's name,
torch-order shape, type, shard, absolute offset, bytes, rows and packed row stride. It
also records every expert bank and compatible slot pool separately, because gate, up and
down projections may use different formats and the down format varies by layer.

The Q4 artifact resolves to six slot geometries. Gate/up use Q4_K except layer 2, which
uses Q5_K. Down uses Q5_1 except layers 2, 4, 30, 46 and 47, which use Q8_0. The Q3
artifact also resolves to six geometries: gate/up use IQ3_XXS except layer 2 IQ4_XS;
down uses IQ4_NL with the same five Q8_0 promoted layers. Pool IDs, exact packed row
bytes, bytes per slot and layer membership are recorded in `expert_slot_pools`.

File-backed host accounting is also part of census schema version 3. Q4 contains
77,017,907,200 expert bytes, 28,800,138,240 PLE bytes and 5,505,584,640 ordinary tensor
bytes. Q3 contains 55,823,564,800 expert bytes, the same PLE size and 5,351,626,240
ordinary tensor bytes. Both full mapped tensor sets remain below the 128 GiB operating
envelope; mappings are reclaimable file-backed pages and report zero anonymous or pinned
host-source bytes.

## Loader rules

- Split filenames are one-based; `split.no` is zero-based. Missing shards, count
  disagreements, duplicate tensor names, invalid dimensions, misalignment, overlap,
  overflow and out-of-bounds payloads fail before loading.
- Unknown formats are distinct from known formats unavailable to a selected execution
  mode. Neither may silently fall back.
- Uniform fused projections use one packed buffer. Mixed projections retain one buffer
  per part and concatenate their computed outputs, preserving each part's row stride.
- llama.cpp tiles Qwen linear-attention V heads. The loader reverses row transforms and
  presents grouped activations in tiled order for the packed `ssm_out` columns, avoiding
  a dense weight copy.
- GGUF stores `ssm_a` as `A = -exp(A_log)`; the loader validates negativity and restores
  `A_log = log(-A)`.
- Qwen centered RMSNorm tensors are converted back from llama.cpp's effective scale by
  subtracting one.
- `per_layer_token_embd.weight` and routed expert tensors are read-only file-range mappings.
  Each expert descriptor retains its shard, absolute offset, quant type, row stride and
  bytes per expert. No complete expert or PLE copy is allocated or pinned.

Inspect a complete local artifact without touching its tensor payload pages:

```bash
PYTHONPATH=python python scripts/inspect_gguf_host.py \
  /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
```

## PLE mmap controls

The GGUF `per_layer_token_embd.weight` tensor has 320,001,536 IQ4_NL rows of 160
elements. Each selected 90-byte packed row is copied and dequantized independently,
then the 16 n-gram heads are assembled into the 2,560-element PLE input. Row bounds are
checked before any lookup. The exact artifact multipliers, head offsets and prime vocab
sizes come from GGUF metadata; they are not regenerated or guessed.

`--ple-warm-mode` exposes four distinct modes:

- `cold` is the default and issues `MADV_DONTNEED` for the PLE range;
- `page-cache-warm` requests OS readahead for the complete PLE range;
- `targeted` synchronously touches only rows selected by each lookup;
- `full-model-warm` explicitly touches every page in every model shard.

Telemetry reports the selected mode, mapped/resident bytes where `mincore` is available,
lookup rows and packed bytes, output bytes, targeted rows, process minor/major page faults
and `/proc/self/io` storage-read bytes. Full-model warm is never selected implicitly.

## Validation status

Hosted tests cover the complete type tables, CUDA-switch agreement, sharded zero-tensor
metadata layouts, failure paths, schema invariants, mixed buffers and real byte-range rows
from both pinned Q4 and Q3 artifacts. Hosted PLE tests compare IQ4_NL selected rows against
gguf-py and exercise invalid IDs, short ranges, hash mismatches and every warm mode. H1
compiles the same GGUF CUDA translation unit for `sm_61`. CUDA expert parity and short
generation remain H2 and require a real Tesla P4; they are not waived by host references.

## Real expert byte probe (H0)

The bounded Issue #16 probe downloads one selected expert's gate, up and down packed ranges
from the immutable Hugging Face revision, using HTTP `Range` and the pinned manifest/census
metadata. It verifies the response `Content-Range`, body length, shard bounds, quant type and
packed row stride before constructing a one-expert `CpuExpertLayout`.

```bash
PYTHONPATH=python python scripts/probe_qwen38_expert.py \
  --layer 0 --expert 0 \
  --cache-dir .cache/freetoken/qwen38-range \
  --output results/qwen38-expert-layer0.json
```

Use `--layer 2` to probe the promoted Q5_K/Q8_0 projection family, and `--offline` to require
the selected ranges to already exist in the cache. Cache files contain only the three selected
ranges; no complete shard is downloaded or committed. The JSON report keeps raw timing samples
and kernel telemetry for an internal forced-scalar versus native executor A/B comparison, and separately compares
both executor outputs against the pinned independent `gguf-py==0.19.0` `dequantize` plus dense FP32 SwiGLU oracle
over the same three fetched byte ranges. Oracle identity, packed and dense output hashes, comparison errors and
tolerances are recorded separately from raw timing, and missing or mismatched gguf versions fail a real probe
clearly. The report labels the partial payload as `range_evidence: measured/artifact-byte`, remains H0/no-P4
evidence, and makes no cache, hybrid-split or full-engine performance claim.

For a target-CPU timing comparison over those same ranges, use
`benchmarks/bench_qwen38_real_expert.py` with `--offline` after populating the probe cache.
Run `--layer 0` and `--layer 2` separately. Each run is fixed to one token, one route and
one selected expert. The report keeps the exact commit/command, CPU ISA, BLAS environment,
process affinity, manifest revision, census and declared model identities, range hashes,
native-library hashes/build metadata, warmups, raw samples, selected kernels
and fallbacks. It separately compares native packed execution with a dense-resident
reference (dequantization once outside timing) and a cold full-reference procedure whose
timing includes source validation, byte/view setup, hashing, dequantization and dense execution.
Build the native libraries first with `make target-cpu-native`, pass its `build.json` through
`--native-build-metadata`, and explicitly set every documented BLAS thread variable to one.
It requires five or more warmups and fails
closed on scalar fallback or any output mismatch; this is preliminary H0 target-CPU evidence,
not a P4 or full-engine performance claim.

The H0 routed-layer adapter is available to CPU correctness probes through
`freetoken.moe.QwenGGUFCpuMoELayer`. Construct it with an already-open
`QwenGGUFCpuExpertBundle` and the exact layer geometry, then use `routed_forward` for a
precomputed route or `forward` with CPU router logits. The adapter preserves the complete
softmax denominator, defaults to the Qwen unrenormalized selected probabilities, accepts
route widths up to the configured Qwen top-k, and forwards padding and bundle telemetry.
The optional `phase` and `group_size` keywords must be `decode` and `1`. Its caller owns
the shared bundle and must close it after all layer adapters finish. It rejects CUDA,
prefill, grouped, nonzero-cache and TP>1 execution; it is not wired into Qwen model
construction or the serving Engine.
