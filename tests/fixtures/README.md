# Synthetic test fixtures

All files in this directory are original synthetic fixtures distributed under the repository's
Apache-2.0 license. They contain no model weights, private prompts, user data, benchmark
measurements, or hardware claims. The Qwen3.8 reference corpus contains only short synthetic
prompts and deterministic long-context templates.

- `qwen4-tiny/` is a deliberately small text-only configuration and deterministic reference
  contract. Real runtime generation remains gated on the H2 Tesla P4 job.
- `gguf/` is generated deterministically by `scripts/generate_test_fixtures.py`. The valid file
  contains only 282 bytes of artificial tensor payload; malformed variants exercise metadata,
  quant identifier, block stride, and offset failures.
- `routing/` contains a locality-positive trace and a cyclic adversarial trace for host cache
  simulation.
- `qwen38-reference-corpus.json` pins the tokenizer revision and deterministic prompt cases used
  for independent correctness evidence. The reference harness hashes and tokenizer-counts the exact
  rendered chat-template prompt, and verifies each long-context needle; metadata alone is never
  treated as measured evidence.
- `kernels/` contains arithmetic values checked against simple independent scalar formulas.
- `sensitive/` contains deterministic tensor-level positive controls for a mis-scaled
  shared-expert gate and a perturbed GDN control. These prove H0 rejection sensitivity;
  they are not long-horizon model-quality evidence.
- `results/` contains schema-valid synthetic evidence. The `evidence_status` field prevents these
  examples from being confused with measured release results.
