# ADR 0002: Downstream Fork And Upstream Sync

- Status: Accepted
- Date: 2026-08-28

## Context

The GitHub repository was created independently and begins with project documentation. Deep model, cache and kernel changes are too invasive for a patch-only plugin, but losing upstream history would make sync and attribution difficult.

## Decision

Merge upstream FreeToken history into this repository once with an explicit unrelated-history merge. Thereafter maintain normal upstream remotes and focused downstream commits. Pin imported PR heads by SHA and keep a source-to-destination ledger.

## Consequences

The first history merge is unusual but subsequent syncs have a common ancestor. Broad syncs must remain separate from optimization changes. The repository can build the complete product directly.

## Alternatives considered

Submodule plus patch series: clean separation but cumbersome for deep changes. Reimplement as a plugin: insufficient access to model/cache internals. Force-push upstream history over project docs: would discard bootstrap history.
