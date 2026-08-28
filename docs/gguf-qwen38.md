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
also records every expert pool separately, because gate, up and down projections may use
different formats and the down format varies by layer.

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
- `per_layer_token_embd.weight` and routed expert tensors remain mmap-backed host sources;
  Issue #13 owns those banks. Their types and layouts are still fully included in this
  issue's census.

## Validation status

Hosted tests cover the complete type tables, CUDA-switch agreement, sharded zero-tensor
metadata layouts, failure paths, schema invariants, mixed buffers and real byte-range rows
from both pinned Q4 and Q3 artifacts. H1 compiles the same GGUF CUDA translation unit for
`sm_61`. CUDA dequant/MMVQ/MMQ parity and tiny/full generation remain H2 and require a real
Tesla P4; they are not waived by the host references.
