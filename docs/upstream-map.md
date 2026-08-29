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

Issue #16 pins FreeToken commit `6f6c8640145eeca9df013e383ff51bf6bbff22f9` as the Q4_K W4A16 arithmetic oracle. The pinned llama.cpp `eaf93765572e794b8e3754fe45adbe12d381e997` and PXQ `066a37e9540a1ca21375fdeb377836fe69ecb729` implementations confirm the packed Q4_K, Q5_K, Q5_1, Q8_0, IQ3_XXS, IQ4_NL and IQ4_XS layouts. Their AVX2 vector-dot paths require Q8_K activations and are not copied into this backend.

The IQ3_XXS grid and IQ4_NL codebook constants are adapted byte-for-byte from the MIT-licensed llama.cpp files and are recorded as an imported source in the manifest and NOTICE. The scalar decoder control flow, bounded workspace bridge, and downstream AVX2 W4A16/mixed GEMV paths are local implementations; the remaining llama.cpp entries stay reference-only and add no NOTICE entry.

## Current source status

- FreeToken PR #257 merged on 2026-08-28 at merge commit `bd8f3d519a48777bf22ee5c7c8f58f4f3ff31b40`. Current upstream `main` was observed at immutable commit `58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6`, which contains that Qwen3.8 implementation.
- The independent FP8 row-wise `_scaled_mm` safety fix at `58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6` is imported as the first conflict-free issue #77 slice. Downstream additionally gates the load-time probe on native-FP8 capability so Pascal stays on W8A16 without allocating probe tensors; this is not P4 performance evidence.
- The issue #77 sync branch merges exact upstream commit `58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6` and reconciles the older adaptations sourced partly from closed, unmerged PR #232. PR #232 remains historical provenance only for materially distinct downstream GGUF, Pascal, CPU-expert, file-backed PLE, and reference-test paths.
- FreeToken PR #279 is open and adds an mmap PLE backend. Its observed head `feaeaa31c0cea385a1c9ee107d4b1053f83b35db` is pinned in the manifest as an immutable `reference` for issue #13. It is not imported by issue #77, and a moving PR head must never be merged implicitly.
- llama.cpp PR #27742 merged into upstream llama.cpp on 2026-08-27. The existing pinned PR-head SHA remains a valid byte-level oracle for imported/reference work, while deployment comparisons use a separately pinned current merged llama.cpp revision.
- vLLM Qwen3.8 and PLE-offload PRs remain model/TP/MTP/offload references rather than Pascal runtime dependencies.
- PXA/PXQ has no downstream-consumed Qwen3.8 implementation at the current pin; issue #19 uses it as a Pascal kernel donor/reference, not as a ready runtime.

## Primary upstreams

| Capability | Primary source | Secondary oracle |
|---|---|---|
| Hybrid MoE engine and q-star scheduling | current FreeToken after issue #77 sync | FreeToken paper / Colibrì |
| Pascal compile/runtime fallbacks | FreeToken PR #19 + #26 | uaysk/ampir vLLM Pascal work |
| Qwen3.8/Qwen4, QSA, PLE and state | merged FreeToken PR #257 through issue #77 | merged llama.cpp Qwen3.8; vLLM Qwen3.8/PLE work |
| Historical downstream Qwen delta | closed PR #232 only where still distinct after issue #77 | pre-sync downstream tests and provenance ledger |
| PLE mmap donor/reference | open FreeToken PR #279 after immutable pin | downstream issue #13; merged llama.cpp PLE behavior |
| GGUF K/I types and Qwen MoE loader | FreeToken PR #131 plus downstream GGUF work | merged llama.cpp; humanjesse/vllm-v100 |
| Qwen MoE TP patterns | FreeToken PR #104 | merged PR #257 / vLLM Qwen3.8 TP |
| low-bit Pascal GPU kernels | PXA/PXQ llama | llama.cpp CPU reference |
| AVX2 low-bit CPU kernels | downstream implementation using FreeToken `6f6c8640` W4A16 semantics | llama.cpp/PXQ Q4_K layout and dequantize + dense reference |
| expert cache policy concepts | FreeToken / flashlib | vLLM expert cache; Colibrì |
| static hot-expert comparator | downstream issue #21 | `timadinorth/llama.cpp` PR #1 |
| transfer/prefetch design | FreeToken | vLLM weight-prefetch/offload work |
| exact context-derived speculation evidence | downstream optional issue #74 | `sxuff/qwen38-flash-next-dgx-spark` paired llama.cpp benchmark |
| placement-cliff safety | downstream issue #73 | community hybrid-placement reports, reproduced only on target hardware |
| QSA context/workspace safety | downstream issue #76 | merged FreeToken PR #257, merged llama.cpp Qwen3.8, current vLLM work |

The static-residency and n-gram repositories above are reference-only unless exact code is later imported. Any copied file or adapted implementation requires a new pinned manifest entry, license review, responsible issue/PR, and NOTICE update where required.

FreeToken PR #257 was previously recorded as a reference at head `e3b6d7bdc74da8c9f88d182415ec4e8dfbb967c3` and merge `bd8f3d519a48777bf22ee5c7c8f58f4f3ff31b40`. After issue #77 it becomes part of the primary upstream base rather than only a PLE-loader oracle. The sync must reconcile its Qwen model, QSA backend, PLE implementation, CUDA-graph/radix state, hybrid MoE and server contracts with downstream changes.

The file-backed downstream loader currently semantically adopts fail-closed PLE checks without copying donor code or changing the pageable GGUF mmap contract. Issue #77 must preserve those downstream-only safety/storage contracts while eliminating duplicate architecture code.

## Integration method

FreeToken commit `9ef3651309fe4058672f2cc92069238dea06be1b` is the initial downstream base. It was merged once with `--allow-unrelated-histories`, using the project-bootstrap history as the other parent. Both histories are therefore ancestors of the integration merge.

From that point:

- the canonical remote is named `upstream` and its fetched mainline is `upstream/main`;
- a temporary local `upstream/freetoken` branch may be used to inspect a pinned source commit, but downstream commits never land on it;
- focused feature PRs are imported or semantically replayed only when absent from the selected upstream base;
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

## Immediate issue #77 sync

Use the exact upstream commit selected by issue #77 rather than a moving `main` ref:

```bash
git remote get-url upstream || \
  git remote add upstream https://github.com/FlashML-org/FreeToken.git
git fetch --prune upstream main
git cat-file -e 58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6^{commit}
git switch main
git pull --ff-only origin main
git switch -c sync/freetoken-qwen38-257
git branch --force upstream/freetoken \
  58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6
git merge --no-commit \
  58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6
```

The sync PR must contain a file-by-file Qwen delta ledger. Do not resolve model/QSA/PLE conflicts by choosing one repository side wholesale. Preserve upstream engine contracts where valid and reapply downstream-only Pascal, GGUF, CPU-expert, storage-tier, placement and evidence requirements as focused commits.

Before committing:

```bash
git diff --name-only --diff-filter=U
git diff --check
python scripts/check_upstream_manifest.py
python scripts/validate_docs.py
```

Complete the sync and prove ancestry:

```bash
git add <resolved-paths> manifests/upstreams.yaml docs/upstream-map.md NOTICE
git commit
git merge-base --is-ancestor \
  58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6 HEAD
make check
```

## Future FreeToken sync

Start every broad sync from a clean, current downstream `main`. Never use an unpinned moving branch as the merge operand:

```bash
git remote get-url upstream
# If the remote is absent in a fresh clone:
git remote add upstream https://github.com/FlashML-org/FreeToken.git
git fetch --prune upstream main
git log --oneline --left-right main...upstream/main
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

Resolve conflicts by preserving downstream product scope and identity while retaining upstream engine contracts. Inspect every conflict and never use a repository-wide `ours` or `theirs` resolution. Update the `freetoken` manifest `ref`, import ledger, and any changed notices in the same sync commit.

If the selected upstream range touches kernels, model graphs, cache maps, quant loaders, or tensor parallelism, the sync PR remains open until the applicable H0/H1 gates pass and all hardware-dependent validation is explicitly deferred. Do not claim P4 validation from H0 or H1 results.

## Focused upstream PR imports

Fetch and pin the PR head without merging a moving GitHub ref directly:

```bash
git fetch upstream refs/pull/<number>/head
git rev-parse FETCH_HEAD
git switch -c import/freetoken-pr-<number>-<short-name> main
git cherry-pick --no-commit <pinned-pr-sha>
```

Before importing, verify that the selected upstream base does not already contain equivalent work. Record the exact PR SHA and source/destination paths in `manifests/upstreams.yaml`. Adapt only the issue-scoped files, preserve source headers, and document semantic reimplementations as local modifications.

## Conflict policy

When sources disagree:

1. Transformers/model-author implementation defines high-level model semantics.
2. Current pinned upstream FreeToken defines engine contracts.
3. Merged llama.cpp and current vLLM serve as independent correctness/operational oracles.
4. PXA defines its quant format and Pascal kernel arithmetic.
5. Downstream performance changes may alter reduction order only with quantified tolerance and model-level validation.
6. Community measurements define experiments, not defaults or target-hardware claims.

## Upstream sync cadence

- check FreeToken and selected PRs before starting each phase;
- pin a new baseline only in a dedicated sync PR;
- rerun hosted correctness and H1 `sm_61` compilation after every sync;
- rerun H2/H3 gates for changes touching kernels, model graphs, cache maps, quant loaders or TP once P4 hardware exists;
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
