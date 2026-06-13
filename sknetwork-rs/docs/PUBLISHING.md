# Publishing Checklist (sknetwork-rs)

Gate document for publishing `sknetwork-rs` to [crates.io](https://crates.io).
Both **human** and **agent** consumers must be able to discover, understand, and
use the crate without reading source code.

## Human documentation (required)

| Item | Status | Notes |
|------|--------|-------|
| `README.md` at crate root | ✅ | crates.io landing page; quick start + module table |
| Crate-level rustdoc (`src/lib.rs`) | ✅ | Module index, examples, parity pointer |
| Module-level rustdoc (`//!` on every `pub mod`) | ✅ | All top-level modules have `//!` + example |
| Item-level rustdoc (`///` on every `pub` item) | ✅ | Completed 2026-06-13; `missing_docs` gate passes |
| `# Errors` on fallible APIs | ✅ | Added across fallible public APIs |
| `# Examples` on estimators | ✅ | Module-level examples on all top-level modules |
| `docs/rustdoc_style.md` | ✅ | Authoring contract |
| `cargo doc --no-deps` builds cleanly | ✅ | No doc-test failures |

### Rustdoc completeness gate

Enable before publish:

```bash
RUSTFLAGS='-D missing_docs' cargo doc --no-deps
```

Or run the bundled script:

```bash
./scripts/check_publish_docs.sh
```

**Publish is blocked** until `missing_docs` passes with zero errors. ✅ Gate passes as of 2026-06-13.

## Agent documentation (required)

| Item | Status | Notes |
|------|--------|-------|
| `AGENTS.md` at crate root | ✅ | Module map, conventions, call patterns, divergences |
| Structured module → Python path table | ✅ | In `AGENTS.md` |
| Error-contract guidance | ✅ | In `AGENTS.md` + rustdoc `# Errors` |
| Known divergences linked | ✅ | Points to `PORTING_MEMO.md` |
| Non-public surfaces documented | ✅ | `bench`, `test_graphs` called out |

Agents should be able to answer these without opening source:

1. What graph type does the crate use?
2. Where is the Python equivalent of algorithm X?
3. What error will calling `predict` before `fit` return?
4. Which modules are internal-only?

## Cargo.toml metadata (required)

| Field | Required | Current |
|-------|----------|---------|
| `description` | yes | set |
| `readme` | yes | `README.md` |
| `license` | yes | `BSD-3-Clause` |
| `repository` | yes | scikit-network monorepo URL |
| `homepage` | recommended | scikit-network GitHub |
| `documentation` | recommended | docs.rs URL |
| `keywords` | recommended | graph, network, sparse, clustering, pagerank (max 5) |
| `categories` | recommended | `science`, `algorithms` |
| `authors` | recommended | Scikit-network team |
| `rust-version` | recommended | `1.88` (edition 2024) |
| `include` | recommended | `/src/` only (root-anchored; `readme` + `license-file` auto-included) |
| `LICENSE` file | yes | BSD-3-Clause text at crate root |
| `rust-toolchain.toml` | recommended | pins MSRV for CI/contributors |

## API surface hygiene (required)

| Item | Status | Action |
|------|--------|--------|
| `bench` module hidden from public docs | ✅ | `#[doc(hidden)]` + `feature = "bench"` |
| Benchmark binaries gated behind `bench` feature | ✅ | `required-features = ["bench"]` on `[[bin]]` |
| No dev-only artifacts in publish tarball | ✅ | `include` whitelist in `Cargo.toml` |
| `cargo package --list` reviewed | ✅ | `./scripts/check_publish_package.sh` |

## Pre-publish commands

```bash
# 1. Full test suite
cargo test

# 2. Documentation gate (must pass)
./scripts/check_publish_docs.sh

# 3. Package contents gate (must pass)
./scripts/check_publish_package.sh

# 4. Dry-run upload
cargo publish --dry-run
```

## Post-publish

- Confirm docs.rs build succeeded: <https://docs.rs/sknetwork-rs>
- Update root `README.rst` with a link to the Rust crate (optional)
- Tag release in git

## Remediation priority for missing rustdoc

✅ **Completed 2026-06-13.** All public items documented across ~80 source files.
Re-check after adding new public API:

```bash
RUSTFLAGS='-D missing_docs' cargo doc --no-deps 2>&1 | rg -c "missing documentation"
```

Target: **0**.
