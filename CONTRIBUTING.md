# Contributing to FreeToken-Pascal

FreeToken-Pascal is a hardware-focused downstream of [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken). Changes must remain reviewable, reproducible, and attributable. This repository is independently maintained and does not imply upstream endorsement.

## Start here

Read:

1. `AGENTS.md`
2. `docs/product-scope.md`
3. `docs/architecture.md`
4. `docs/testing-strategy.md`
5. the ADRs relevant to the issue

Select an unblocked GitHub issue. Do not start a downstream optimization before its correctness dependency is complete. Problems that reproduce on an unmodified upstream checkout should also be reported to upstream according to its contribution policy.

## Development environment

The supported downstream development environment is Linux x86-64, Python 3.12, and CUDA 12.6 for CUDA compilation. Hosted tests must run without a GPU. Pascal runtime tests require a self-hosted runner labelled `sm61`.

The imported upstream revision still declares its own CUDA 13 dependency set. Until issue #7 lands the downstream environment, use the repository-level H0 checks without interpreting upstream CUDA requirements as Pascal support:

```bash
make check
```

## Branches and pull requests

- Downstream issue branches use `issue-<number>-<short-name>`.
- The remote-tracking branch `upstream/main` is the fetched FreeToken mainline; documentation may call this upstream line `upstream/freetoken` conceptually, but no local work is committed directly on it.
- Focused upstream PR imports use `import/freetoken-pr-<number>-<short-name>` and pin the source head SHA before adaptation.
- Broad upstream synchronization uses `sync/freetoken-<YYYYMMDD>` and must not include downstream performance work.

A PR must:

- reference exactly one primary issue;
- explain the reference and optimized paths;
- list copied/adapted source commits;
- include test commands and results;
- include benchmark methodology for performance changes;
- state whether P4 hardware was used;
- update docs and upstream manifests;
- preserve a safe fallback.

Large changes should be split into architecture/correctness, kernel implementation, and tuning PRs.

## Performance changes

A performance PR must compare identical model bytes, prompts, context, sampling, and placement. Report medians and dispersion across repeated runs. Cold start, prefill, and decode must be separated. Withdraw or correct results that cannot be reproduced.

## Licensing

The project is Apache-2.0. Preserve upstream copyright headers. Add copied or substantially adapted components to `NOTICE` and pin their source SHA in `manifests/upstreams.yaml`.
