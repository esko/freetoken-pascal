# Contributing

FreeToken-Pascal is a hardware-focused downstream project. Changes must remain reviewable, reproducible, and attributable.

## Start here

Read:

1. `AGENTS.md`
2. `docs/product-scope.md`
3. `docs/architecture.md`
4. `docs/testing-strategy.md`
5. the ADRs relevant to the issue

Select an unblocked GitHub issue. Do not start a downstream optimization before its correctness dependency is complete.

## Development environment

The supported development environment is Linux x86-64, Python 3.12, and CUDA 12.6 for CUDA compilation. Hosted tests must run without a GPU. Pascal runtime tests require a self-hosted runner labelled `sm61`.

The exact environment will be created by the bootstrap backlog. Until then, the repository-level checks are:

```bash
make check
```

## Pull requests

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
