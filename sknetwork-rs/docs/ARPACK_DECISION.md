# Decision memo: ARPACK binding vs pure-Rust Lanczos

**Status:** decided — **defer ARPACK** for now.  
**Date:** 2026-06-12  
**Scope:** `sknetwork-rs` partial SVD (`LanczosSVD` / `svds_arpack`) and long-term portable library distribution.  
**Related:** [`benchmarking/linalg/LANCZOS_PERF_MEMO.md`](../benchmarking/linalg/LANCZOS_PERF_MEMO.md) (performance rounds, continue-Krylov attempts).

---

## Context

The Rust Lanczos path (`symmetric_eigsh` + `svds_arpack`) is a matrix-free reimplementation of SciPy `svds` / ARPACK IRLM:

- **Quality:** default `q₀` restart passes all tier **S+M+L** σₖ gates (rel ≤ 1e-3 vs Python).
- **Speed:** citeseer tier M ~**59 ms** Rust `eigsh` vs ~**21 ms** Python ARPACK (Py÷Rs ~0.3); several dense M graphs already favor Rust.
- **Blocked work:** ARPACK-style **continued Krylov** after implicit restart (fewer restarts, target ~40–55 ms on citeseer) fails σₖ parity despite correct-looking locked λ values — Ritz **vectors** desync from compressed `(V, T)`. Rounds 9–11 documented in the perf memo.

The project goal remains a **portable library** (Rust crate and, eventually, redistributable wheels), not only an in-house benchmark binary.

---

## What binding ARPACK would mean

ARPACK-NG (`dsaupd` / `dseupd` for real symmetric problems) would replace the Rust IRLM loop while keeping:

| Layer | Change |
|-------|--------|
| Gram matvec | **Unchanged** — `SparseMatvecCache::gram_matvec_nd`, same CSR paths |
| Eigen solve | **Replaced** — call ARPACK with `y ← A x` callback |
| SVD packing | **Unchanged** — `scipy_arpack_svd_pack` (Rayleigh–Ritz after `eigsh`) |

Conceptual stack:

```
svds_arpack  →  ARPACK dsaupd/dseupd  →  gram_matvec (existing)
                    ↓
              scipy_arpack_svd_pack (existing)
```

Rust binding options considered:

| Option | Notes |
|--------|--------|
| [`arpack`](https://docs.rs/arpack/) + [`arpack-sys`](https://docs.rs/arpack-sys/) | Safe wrapper over ARPACK-NG ≥ 3.8 C API (`dsaupd_c` / `dseupd_c`); matvec closure API |
| `arpack-sys` only | Full `iparam` / `v0` control for strict SciPy parity |
| System `libarpack2-dev` | Fine for local spikes; **not** sufficient for portable wheels |

Expected benefits:

- Correct continued Krylov / implicit-restart semantics (the current pain point).
- Likely near-parity speed with Python on citation graphs (algorithm match, not just matvec tuning).

Open integration items (even for a spike):

- Map `EigWhich::Lm` → `Which::LargestMagnitude`, `tol²`, `choose_ncv`, `maxiter`.
- **`v0` / seed-42 parity** — safe `arpack::Options` today exposes `tol`, `max_iter`, `ncv` only; fixed `v0` for benchmarks may need lower-level `dsaupd` or wrapper extension.
- **Thread safety** — `arpack` crate serializes calls (Fortran `SAVE` state); OK for `benchmark_ipc`, relevant if Lanczos runs concurrently later.

---

## Portability: the deciding constraint

“Portable library” implies artifacts that work **without** the user installing system Fortran libraries.

| Distribution | System ARPACK (`apt install libarpack2-dev`) | Vendored / static ARPACK |
|--------------|--------------------------------------------|---------------------------|
| Rust crate (crates.io), optional feature | Acceptable for advanced users | Better default for “it just works” |
| Python wheel (`pip install …`) | **Unacceptable** — missing `.so` at runtime | **Required** (SciPy model) |
| Conda / distro packages | Declare `libarpack` dependency | Optional |

A production ARPACK path for this project is therefore **not** `cargo add arpack` + pkg-config on the end user. It is:

1. **Vendor** ARPACK-NG (submodule or tarball),
2. **Static-link** (+ Fortran runtime) into release binaries / PyO3 extensions,
3. **auditwheel** / delocate / equivalent for manylinux/macOS wheels.

That is **build and packaging** work, distinct from the numerical spike, but mandatory before claiming portability.

---

## Effort (revised, honest)

| Goal | Effort |
|------|--------|
| Spike: system ARPACK on citeseer, σₖ + timing check | **0.5–1 day** |
| Feature-flag backend, tier M green with ARPACK | **1–2 days** |
| `v0` parity via `arpack-sys` | **+1–3 days** (optional) |
| Vendored ARPACK + static wheels (manylinux, etc.) | **+1–2 weeks** (only if required for release) |

Earlier “several weeks” estimates conflated **prove ARPACK helps** with **ship portable wheels everywhere**. The spike is short; the long tail is packaging.

---

## Pure Rust vs ARPACK (for a portable library)

| | Pure Rust IRLM (current) | ARPACK binding |
|--|--------------------------|----------------|
| σₖ on default path | **Green** (S+M+L) | Reference (expected green) |
| Continued Krylov / speed | Blocked on numerics | Reference behavior |
| citeseer vs Python | ~3× slower | Likely ~1× |
| Portable wheels | **Easy** — Rust-only artifact | **Harder** — Fortran + vendored build |
| Long-term maintenance | Own every restart bug | Own integration + packaging |
| License / binary size | Clean | BSD ARPACK; larger binaries |

**Pure Rust wins distribution.** **ARPACK wins algorithm correctness and speed vs SciPy** with less research risk.

A common hybrid (not chosen now):

- Release wheels: ARPACK backend (vendored).
- Dev / fallback: pure-Rust `eigsh` (`features = ["pure-lanczos"]`).

---

## Decision

**Do not bind ARPACK for now.** Focus on other porting and parity work (see [`PORTING_MEMO.md`](../PORTING_MEMO.md)).

Rationale:

1. **Immediate priority** is broader sknetwork-rs porting, not Lanczos packaging or Fortran CI.
2. **Default Rust Lanczos is already quality-green**; the speed gap is acceptable to defer while other modules advance.
3. **Portable ARPACK** commits the project to vendored Fortran builds — a deliberate release-engineering bet, not a side effect of a one-day spike.
4. Continued-Krylov speedups in pure Rust have **diminishing returns** vs porting `saitr`/`sgets` correctly; if Lanczos speed becomes P0 again, **revisit ARPACK with a vendoring plan** rather than more ad-hoc restart patches.

---

## If we revisit ARPACK later

Checklist:

1. Run system-ARPACK spike (no vendoring) — confirm Py÷Rs on citeseer and σₖ gates.
2. Decide wheel strategy: static vendored ARPACK-NG vs optional system library for Rust-only consumers.
3. Extend or fork `arpack` wrapper for `v0` / benchmark parity.
4. Keep `symmetric_eigsh` as fallback feature for environments without Fortran toolchain.
5. Document license (ARPACK-NG BSD) and binary size impact.

Until then, Lanczos performance notes live in [`LANCZOS_PERF_MEMO.md`](../benchmarking/linalg/LANCZOS_PERF_MEMO.md); **do not enable** `SKNETWORK_EIGSH_CONTINUE=1` in quality gates.

---

## References

- SciPy `svds` ARPACK path: `scipy/sparse/linalg/_eigen/_svds.py` (Gram operator + `eigsh` + `svd(Av)` post-processing).
- Rust crates: [arpack](https://docs.rs/arpack/latest/arpack/), [arpack-sys](https://docs.rs/arpack-sys/latest/arpack_sys/).
- ARPACK-NG: [https://github.com/opencollab/arpack-ng](https://github.com/opencollab/arpack-ng)
