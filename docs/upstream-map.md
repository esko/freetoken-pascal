# Upstream integration map

This project is a downstream integration, not an independent reimplementation. Every imported change must use a pinned commit SHA.

## Primary upstreams

| Capability | Primary source | Secondary oracle |
|---|---|---|
| Hybrid MoE engine and q-star scheduling | FreeToken | FreeToken paper / Colibrì |
| Pascal compile/runtime fallbacks | FreeToken PR #19 + #26 | uaysk/ampir vLLM Pascal work |
| Qwen3.8/Qwen4, QSA, PLE | FreeToken PR #232 | vLLM #53896/#53899; llama.cpp #27742 |
| GGUF K/I types and Qwen MoE loader | FreeToken PR #131 | llama.cpp; humanjesse/vllm-v100 |
| Qwen MoE TP patterns | FreeToken PR #104 | vLLM Qwen3.8 TP |
| low-bit Pascal GPU kernels | PXA/PXQ llama | llama.cpp CPU reference |
| AVX2 low-bit CPU kernels | llama.cpp / ik_llama / PXA | dequantize + dense reference |
| expert cache policy concepts | FreeToken / flashlib | vLLM #37190; Colibrì |
| transfer/prefetch design | FreeToken | vLLM #29941/#51710 |

## Integration method

The first implementation issue merges upstream FreeToken history into this repository once, using the existing project-bootstrap commit as the other parent. From that point:

- `upstream/freetoken` tracks FreeToken main;
- feature PRs are imported or semantically replayed as focused commits;
- local changes live in reviewable downstream commits;
- no moving PR head is referenced without recording its SHA;
- each copied file retains license headers;
- `manifests/upstreams.yaml` records source path, destination path, source SHA and local differences.

## Conflict policy

When sources disagree:

1. Transformers/model-author implementation defines high-level model semantics.
2. Current upstream FreeToken defines engine contracts.
3. llama.cpp/vLLM serve as independent correctness oracles.
4. PXA defines its quant format and Pascal kernel arithmetic.
5. Downstream performance changes may alter reduction order only with quantified tolerance and model-level validation.

## Upstream sync cadence

- check FreeToken and selected PRs before starting each phase;
- pin a new baseline only in a dedicated sync PR;
- rerun hosted correctness after every sync;
- rerun H2/H3 gates for changes touching kernels, model graphs, cache maps, quant loaders or TP;
- do not mix a broad upstream sync with a downstream optimization.
