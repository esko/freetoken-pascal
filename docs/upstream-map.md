# Upstream integration map

This project is a downstream integration, not an independent reimplementation. Every imported change must use a pinned commit SHA.

The machine-readable contract is `manifests/upstreams.schema.json`. Validate the ledger, imported destination paths, and NOTICE membership with:

```bash
python scripts/check_upstream_manifest.py
```

Each source has one usage state:

- `imported`: source is present in this tree, has at least one path ledger entry, and is listed in NOTICE when required;
- `planned`: pinned donor code intended for a named implementation issue but not copied yet;
- `reference`: a pinned correctness oracle whose code is not copied.

Only `imported` sources declare destination paths. Changing a source to `imported` requires the responsible issue/PR, exact source and destination paths, import method, local differences, license, and header policy in the same commit.

Issue #16 pins FreeToken commit
`6f6c8640145eeca9df013e383ff51bf6bbff22f9` as the Q4_K W4A16 arithmetic oracle.
The pinned llama.cpp `eaf93765572e794b8e3754fe45adbe12d381e997` and PXQ
`066a37e9540a1ca21375fdeb377836fe69ecb729` implementations confirm the packed
Q4_K, Q5_K, Q5_1 and Q8_0 layouts. Their AVX2 vector-dot paths require Q8_K
activations and are not copied into this backend. The downstream AVX2 W4A16
implementation, mixed-format direct GEMV primitive and Torch-free mixed-format
reference decoders are original code; these sources remain reference-only and
therefore add no NOTICE entry.

## Primary upstreams

| Capability | Primary source | Secondary oracle |
|---|---|---|
| Hybrid MoE engine and q-star scheduling | FreeToken | FreeToken paper / Colibrì |
| Pascal compile/runtime fallbacks | FreeToken PR #19 + #26 | uaysk/ampir vLLM Pascal work |
| Qwen3.8/Qwen4, QSA, PLE | FreeToken PR #232 | vLLM #53896/#53899; llama.cpp #27742 |
| GGUF K/I types and Qwen MoE loader | FreeToken PR #131 | llama.cpp; humanjesse/vllm-v100 |
| Qwen MoE TP patterns | FreeToken PR #104 | vLLM Qwen3.8 TP |
| low-bit Pascal GPU kernels | PXA/PXQ llama | llama.cpp CPU reference |
| AVX2 low-bit CPU kernels | downstream implementation using FreeToken `6f6c8640` W4A16 semantics | llama.cpp/PXQ Q4_K layout and dequantize + dense reference |
| expert cache policy concepts | FreeToken / flashlib | vLLM #37190; Colibrì |
| transfer/prefetch design | FreeToken | vLLM #29941/#51710 |

## Integration method

FreeToken commit `9ef3651309fe4058672f2cc92069238dea06be1b` is the initial downstream base. It was merged once with `--allow-unrelated-histories`, using the project-bootstrap history as the other parent. Both histories are therefore ancestors of the integration merge.

From that point:

- the canonical remote is named `upstream` and its fetched mainline is `upstream/main`;
- a temporary local `upstream/freetoken` branch may be used to inspect a pinned source commit, but downstream commits never land on it;
- feature PRs are imported or semantically replayed as focused commits;
- local changes live in reviewable downstream commits;
- no moving PR head is referenced without recording its SHA;
- each copied file retains license headers;
- `manifests/upstreams.yaml` records source path, destination path, source SHA and local differences.

## Initial import commands

The one-time import used these commands from the project bootstrap `main` branch:

```bash
git remote add upstream https://github.com/FlashML-org/FreeToken.git
git fetch upstream main
git switch -c issue-5-upstream-import
git merge --allow-unrelated-histories --no-commit \
  9ef3651309fe4058672f2cc92069238dea06be1b
# Resolve the five add/add conflicts deliberately, stage the complete tree, then:
git commit
```

The conflict decisions were:

- keep the downstream README and contribution contract, updated to describe the imported runtime;
- retain the upstream runtime build metadata while changing distribution identity and URLs to FreeToken-Pascal;
- retain both FreeToken Authors and FreeToken-Pascal contributor copyright notices;
- combine upstream ignore rules with downstream model, CUDA, trace, and private-result rules;
- preserve all downstream `.github`, ADR, manifest, testing, release, and governance files.

## Future FreeToken sync

Start every broad sync from a clean, current downstream `main`. Never use an unpinned moving branch as the merge operand:

```bash
git remote get-url upstream
# If the remote is absent in a fresh clone:
git remote add upstream https://github.com/FlashML-org/FreeToken.git
git fetch --prune upstream main
git log --oneline --left-right main...upstream/main
git diff --stat 9ef3651309fe4058672f2cc92069238dea06be1b..upstream/main
git rev-parse upstream/main
```

Review the commits and select an exact 40-character SHA, recorded below as `<new-sha>`. Then create a dedicated sync branch and merge only that commit:

```bash
git switch main
git pull --ff-only origin main
git switch -c sync/freetoken-YYYYMMDD
git branch --force upstream/freetoken <new-sha>
git merge --no-commit <new-sha>
```

Resolve conflicts by preserving downstream product scope and identity while retaining upstream engine contracts. Inspect every conflict and never use a repository-wide `ours` or `theirs` resolution. Before committing:

```bash
git diff --name-only --diff-filter=U
git diff --check
python scripts/check_upstream_manifest.py
python scripts/validate_docs.py
```

Update the `freetoken` manifest `ref`, import ledger, and any changed notices in the same sync commit. Complete the merge, validate it, and prove reachability:

```bash
git add <resolved-paths> manifests/upstreams.yaml docs/upstream-map.md NOTICE
git commit
git merge-base HEAD <new-sha>
git merge-base --is-ancestor <new-sha> HEAD
git merge-base --is-ancestor origin/main HEAD
make check
```

If the selected upstream range touches kernels, model graphs, cache maps, quant loaders, or tensor parallelism, the sync PR remains open until the applicable H2/H3 gates can run. Do not claim hardware validation from H0 or H1 results.

## Focused upstream PR imports

Fetch and pin the PR head without merging a moving GitHub ref directly:

```bash
git fetch upstream refs/pull/<number>/head
git rev-parse FETCH_HEAD
git switch -c import/freetoken-pr-<number>-<short-name> main
git cherry-pick --no-commit <pinned-pr-sha>
```

Record the exact PR SHA and source/destination paths in `manifests/upstreams.yaml`. Adapt only the issue-scoped files, preserve source headers, and document semantic reimplementations as local modifications.

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

## Upstream change report

Inspect all pins without network access:

```bash
python scripts/report_upstream_changes.py --offline --json
```

Compare every recorded moving locator (`upstream_ref`) with its immutable pin:

```bash
python scripts/report_upstream_changes.py
python scripts/report_upstream_changes.py --json > results/upstream-changes.json
```

The report labels an unchanged locator `current`, a different remote head `changed`, and a failed lookup `unavailable`. Changed rows include a GitHub compare URL. A changed PR head does not update the manifest automatically: inspect the diff, verify the exact commit, and update the pin only in the issue that consumes it or in a focused provenance/sync PR.
