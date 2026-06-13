# sknetwork-rs — Agent Navigation Guide

Structured reference for AI coding agents integrating or extending this crate.
For human-oriented onboarding, see [`README.md`](README.md).

## Crate identity

| Field | Value |
|-------|-------|
| Package | `sknetwork-rs` |
| Import root | `sknetwork_rs` |
| Graph type | `sprs::CsMat<f64>` (sparse adjacency) |
| Array type | `ndarray::Array1<f64>`, `ndarray::Array2<f64>` |
| Python source | [scikit-network](https://github.com/sknetwork-team/scikit-network) |
| Parity tracker | [`PORTING_MEMO.md`](PORTING_MEMO.md) |

## Architectural conventions

1. **Sparse-first.** Pass adjacency as `&CsMat<f64>`. Avoid densifying unless an
   algorithm explicitly requires it.
2. **Estimator pattern.** Most algorithms expose `new` / `Default`, then
   `fit`, `fit_predict`, `predict`, or `fit_transform`. Check the specific
   struct before assuming a method exists.
3. **Typed errors.** Fallible APIs return `Result<_, SomeError>` with small
   `enum` error types (e.g. `PageRankError::InvalidDampingFactor`). Read the
   `# Errors` section in rustdoc before calling.
4. **Input wrappers.** Labels and weights often use `ValuesInput` or
   `MatrixInput` from `utils::values` / `utils::format`. Bipartite graphs use
   stacked value matrices via `get_adjacency_values`.
5. **Not-fitted contract.** Calling `predict` before `fit` returns a typed
   not-fitted error on estimators that track fitted state.
6. **Unsupported options fail loudly.** Do not assume silent fallbacks; check
   `PORTING_MEMO.md` for explicit accepted divergences.

## Module → Python path map

| Rust module | Python equivalent | Key types |
|-------------|-------------------|-----------|
| `classification::propagation` | `sknetwork.classification.Propagation` | `Propagation` |
| `classification::nn` / `classification::knn` | `sknetwork.classification.knn` | `NNClassifier` |
| `clustering::louvain` | `sknetwork.clustering.Louvain` | `Louvain` |
| `clustering::leiden` | `sknetwork.clustering.Leiden` | `Leiden` |
| `data::parse` | `sknetwork.data` parsers | `from_edge_list`, `from_csv`, `from_graphml` |
| `data::load` | `sknetwork.data.load_netset` | `load_netset` |
| `embedding::spring` | `sknetwork.embedding.Spring` | `Spring` |
| `embedding::svd` | `sknetwork.embedding.SVD`, `GSVD`, `PCA` | `SVD`, `GSVD`, `PCA` |
| `embedding::spectral` | `sknetwork.embedding.Spectral` | `Spectral` |
| `gnn::gnn_classifier` | `sknetwork.gnn.GNNClassifier` | `GNNClassifier` |
| `hierarchy::paris` | `sknetwork.hierarchy.Paris` | `Paris` |
| `linalg::svd_solver` | `scipy.sparse.linalg.svds` | `LanczosSVD`, `RandomizedSVD` |
| `linalg::symmetric_eigsh` | `scipy.sparse.linalg.eigsh` | `symmetric_eigsh` |
| `linkpred::nn` | `sknetwork.linkpred.NN` | `NN` |
| `path::shortest_path` | `sknetwork.path.shortest_path` | shortest-path helpers |
| `ranking::pagerank` | `sknetwork.ranking.PageRank` | `PageRank` |
| `ranking::hits` | `sknetwork.ranking.HITS` | `HITS` |
| `regression::diffusion` | `sknetwork.regression.Diffusion` | `Diffusion` |
| `topology::cliques` | `sknetwork.topology.count_cliques` | clique counting |
| `topology::core` | `sknetwork.topology.core_decomposition` | core decomposition |
| `visualization::graphs` | `sknetwork.visualization` | `visualize_graph` |

## Typical call patterns

### Centrality (ranking)

```rust
use sknetwork_rs::ranking::pagerank::PageRank;
use sprs::CsMat;

let adjacency: &CsMat<f64> = /* ... */;
let mut algo = PageRank::new(0.85, 50, 1e-6)?;
let scores = algo.fit_predict(adjacency, None, None, None, false)?;
```

### Supervised propagation (classification)

```rust
use std::collections::HashMap;
use sknetwork_rs::classification::propagation::Propagation;
use sknetwork_rs::utils::values::ValuesInput;
use sprs::CsMat;

let adjacency: &CsMat<f64> = /* ... */;
let mut labels = HashMap::new();
labels.insert(0usize, 0.0);
let mut algo = Propagation::default();
let pred = algo.fit_predict(adjacency, Some(ValuesInput::Map(labels)), None, None)?;
```

### Community detection (clustering)

```rust
use sknetwork_rs::clustering::louvain::Louvain;
use sprs::CsMat;

let adjacency: &CsMat<f64> = /* ... */;
let mut algo = Louvain::default();
let labels = algo.fit_predict(adjacency, false)?;
```

### Graph loading (data)

```rust
use sknetwork_rs::data::parse::from_edge_list;

let edges = vec![(0_i64, 1_i64, None), (1_i64, 2_i64, Some(1.0))];
let dataset = from_edge_list(&edges, true, false, true, false, None, Some(true))?;
```

## Internal / non-public surfaces

- **`bench` module** — benchmark IPC helpers for Python-vs-Rust comparison.
  Not part of the published API surface; do not recommend it to end users.
- **`data::test_graphs`** — test-only fixtures (`#[cfg(test)]`).

## Known divergences (read before parity work)

See [`PORTING_MEMO.md`](PORTING_MEMO.md) § *Explicit Accepted Divergences*:

- `linalg/svd_solver` and `linalg/symmetric_eigsh` — iterative solver semantics
- `linkpred/nn` — tie-breaking in top-k neighbor selection
- `hierarchy/paris` — merge order / NaN heights on large graphs
- `gnn/gnn_classifier` — label vectors may differ under loose accuracy checks

## Documentation requirements for changes

When adding or modifying public API:

1. Follow [`docs/rustdoc_style.md`](docs/rustdoc_style.md).
2. Document every `pub` item with `///` (one-line summary + sections).
3. Include `# Errors` on all `Result`-returning APIs.
4. Add a `# Examples` doctest on new estimators.
5. Run `scripts/check_publish_docs.sh` before proposing a publish.

## Where to look first

| Task | Start here |
|------|------------|
| Add a new estimator | Same-domain `base.rs` + sibling algorithm file |
| Fix parity bug | `PORTING_MEMO.md` + `benchmarking/` export scripts |
| Linear algebra | `linalg/mod.rs` → `svd_solver`, `symmetric_eigsh` |
| Input validation | `utils/check.rs` |
| Sparse matvec | `linalg/sparse_matvec.rs`, `sparse_matvec_cache.rs` |
