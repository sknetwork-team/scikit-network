# Rustdoc Style Guide

This repository uses a contract-first rustdoc style for consistent, idiomatic
documentation.

## Core Rules

- Use `//!` for crate and module docs.
- Use `///` for item docs (structs, enums, traits, functions, methods).
- Start with a one-line summary describing behavior.
- Prefer user-facing contract details over implementation internals.

## Required Sections (public, fallible APIs)

For public APIs, include sections when relevant:

- `# Arguments` for non-trivial parameters.
- `# Returns` for non-obvious return semantics.
- `# Errors` for `Result`-returning APIs, listing typed error variants.
- `# Panics` only when panic behavior is intentional and reachable.
- `# Examples` with small, deterministic snippets.

## Internal/Private Item Policy

- Document complex internal algorithms and helpers with invariants and shape or
  numerical assumptions.
- Skip boilerplate docs for trivial private helpers.

## Wording Conventions

- Use present tense and active voice.
- Keep prose concise and specific.
- Prefer stable terminology used in this crate (adjacency, bipartite,
  not-fitted, damping factor, etc.).

## Example Conventions

- Keep examples compile-friendly and deterministic.
- Keep examples short; avoid loading large fixtures.
- Use realistic estimator workflows (`new`/`fit`/`predict`/`transform`) where
  applicable.

## PR Documentation Checklist

- New public API items include rustdoc.
- New fallible APIs include `# Errors`.
- New public estimators include at least one usage example in module/item docs,
  unless explicitly deferred.
- Complex internal algorithms touched by the PR include invariant notes if not
  already documented.

## Agent-user documentation

Agents (AI coding assistants) consume this crate through structured navigation
docs in addition to rustdoc:

- **`AGENTS.md`** — module map, Python parity paths, call patterns, error
  contracts, and known divergences. Update when adding public modules or
  changing estimator signatures.
- **Crate-level `//!` docs in `lib.rs`** — module index table and pointers to
  `AGENTS.md` / `PORTING_MEMO.md`.
- **Stable terminology** — use the same terms in rustdoc and `AGENTS.md`
  (adjacency, bipartite, not-fitted, damping factor) so agents can cross-reference.

Internal-only surfaces (`bench`, `#[cfg(test)]` modules) must be marked
`#[doc(hidden)]` and documented in `AGENTS.md` as non-public.

## Publish gate

Before `cargo publish`, all public items must pass:

```bash
RUSTFLAGS='-D missing_docs' cargo doc --no-deps
```

See [`PUBLISHING.md`](PUBLISHING.md) for the full checklist.
