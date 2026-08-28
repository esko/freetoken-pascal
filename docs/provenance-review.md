# Provenance and license review

## Review scope

This review covers the source tree after Issue #5 and the provenance schema introduced by Issue #6. The authoritative machine-readable ledger is `manifests/upstreams.yaml`; this document records the human review and reproducible verification procedure without duplicating every field.

## Current imported source

Only `freetoken` has `usage: imported`. FreeToken commit `9ef3651309fe4058672f2cc92069238dea06be1b` entered through PR #30 as a complete unrelated-history merge. Its original commit graph, source paths, root Apache-2.0 license, and file contents remain reachable.

The full-tree `.` to `.` ledger entry is intentional: every current kernel, model, cache, server, test, benchmark, and asset imported in PR #30 comes from that one exact source tree. The root conflict decisions are enumerated in the manifest and `docs/upstream-map.md`. No later donor code has been copied or adapted yet.

## NOTICE reconciliation

FreeToken did not contain an upstream `NOTICE` file at the pinned commit. Its Apache-2.0 `LICENSE` and copyright line are retained in the downstream root license. The downstream NOTICE lists only the imported `freetoken` source ID.

The llama.cpp, PXQ, and vLLM entries are currently planned donors or reference-only correctness oracles. Their code is not present as a separate downstream import, so listing them as included products would be inaccurate. The validator rejects either a missing required NOTICE ID or a NOTICE ID whose manifest entry is not imported and notice-required.

## Source headers

The initial import is a history merge rather than a file copy, so all source-file headers are preserved exactly. FreeToken does not apply an SPDX line to every source file; Issue #6 does not manufacture headers that were absent upstream. Future `copied` or `adapted` ledger entries must state whether the original header was preserved or an SPDX header was added, and reviewers must compare that entry with the pinned source before merge.

## Pin reachability verification

On 2026-08-28, every manifest SHA was resolved through the GitHub commit API for its declared repository. The obsolete Qwen4 PR #232 value was not reachable and was replaced with the verified head `ad752c9970e0dc3f1b09aeec38235332149336ed`. The previously unresolved llama.cpp Qwen4 and PXQ entries were pinned to `eaf93765572e794b8e3754fe45adbe12d381e997` and `066a37e9540a1ca21375fdeb377836fe69ecb729` respectively.

Reproduce commit reachability for any entry with:

```bash
npx -y gh-axi api /repos/<owner>/<repository>/commits/<sha> --jq .sha
```

Check whether recorded branch or PR locators have moved without changing the pins:

```bash
python scripts/report_upstream_changes.py
```

A moved locator is a review signal, not proof that the pin is invalid. Release-critical pins change only in a focused issue/PR after the new commit and license implications are reviewed.
