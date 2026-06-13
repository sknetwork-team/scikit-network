# sknetwork-rs

Rust port of core graph-learning and network-analysis primitives from
[scikit-network](https://github.com/sknetwork-team/scikit-network).

The crate targets **sparse, matrix-first** workflows: adjacency graphs are
represented as `sprs::CsMat<f64>`, and estimators follow scikit-learn-style
`fit` / `predict` / `transform` patterns where applicable.

## Quick start

Add to `Cargo.toml`:

```toml
[dependencies]
sknetwork-rs = "0.1"
sprs = "0.11"
ndarray = "0.16"
```

PageRank on a 4-node identity graph:

```rust
use sknetwork_rs::ranking::pagerank::PageRank;
use sprs::CsMat;

let adjacency = CsMat::<f64>::eye(4);
let mut algo = PageRank::default();
let scores = algo.fit_predict(&adjacency, None, None, None, false).unwrap();
```

## Module map

| Module | Purpose | Primary entry points |
|--------|---------|----------------------|
| [`classification`](https://docs.rs/sknetwork-rs/latest/sknetwork_rs/classification/) | Supervised label propagation, diffusion, NN classifiers | `propagation::Propagation`, `nn::NNClassifier` |
| [`clustering`](https://docs.rs/sknetwork-rs/latest/sknetwork_rs/clustering/) | Community detection | `louvain::Louvain`, `leiden::Leiden` |
| [`data`](https://docs.rs/sknetwork-rs/latest/sknetwork_rs/data/) | Graph I/O and parsing | `parse::from_edge_list`, `load::load_netset` |
| [`embedding`](https://docs.rs/sknetwork-rs/latest/sknetwork_rs/embedding/) | Layout and low-rank embeddings | `spring::Spring`, `svd::SVD`, `spectral::Spectral` |
| [`gnn`](https://docs.rs/sknetwork-rs/latest/sknetwork_rs/gnn/) | Graph neural networks | `gnn_classifier::GNNClassifier` |
| [`hierarchy`](https://docs.rs/sknetwork-rs/latest/sknetwork_rs/hierarchy/) | Hierarchical clustering | `paris::Paris`, `louvain_hierarchy::LouvainHierarchy` |
| [`linalg`](https://docs.rs/sknetwork-rs/latest/sknetwork_rs/linalg/) | Sparse linear algebra | `svd_solver`, `symmetric_eigsh`, `polynome` |
| [`linkpred`](https://docs.rs/sknetwork-rs/latest/sknetwork_rs/linkpred/) | Link prediction | `nn::NN` |
| [`path`](https://docs.rs/sknetwork-rs/latest/sknetwork_rs/path/) | Shortest paths and search | `shortest_path`, `distances`, `search` |
| [`ranking`](https://docs.rs/sknetwork-rs/latest/sknetwork_rs/ranking/) | Centrality and ranking | `pagerank::PageRank`, `hits::HITS`, `katz::Katz` |
| [`regression`](https://docs.rs/sknetwork-rs/latest/sknetwork_rs/regression/) | Diffusion regression | `diffusion::Diffusion` |
| [`topology`](https://docs.rs/sknetwork-rs/latest/sknetwork_rs/topology/) | Structural graph analysis | `cliques`, `core`, `cycles`, `triangles` |
| [`utils`](https://docs.rs/sknetwork-rs/latest/sknetwork_rs/utils/) | Shared input helpers | `check`, `format`, `values` |
| [`visualization`](https://docs.rs/sknetwork-rs/latest/sknetwork_rs/visualization/) | SVG graph and dendrogram rendering | `graphs`, `dendrograms` |

## Documentation

- **API reference (rustdoc):** <https://docs.rs/sknetwork-rs>
- **Repository (monorepo):** <https://github.com/sknetwork-team/scikit-network/tree/master/sknetwork-rs>
- **Python parity notes:** <https://github.com/sknetwork-team/scikit-network/blob/master/sknetwork-rs/PORTING_MEMO.md>

## Python parity

This crate is a port of the Python `scikit-network` library. Module paths are
kept close to their Python counterparts (e.g. `sknetwork.ranking.PageRank` →
`sknetwork_rs::ranking::pagerank::PageRank`). Known behavioral divergences and
solver semantics are tracked in the repository
[`PORTING_MEMO.md`](https://github.com/sknetwork-team/scikit-network/blob/master/sknetwork-rs/PORTING_MEMO.md).

## License

BSD-3-Clause — same as scikit-network.
