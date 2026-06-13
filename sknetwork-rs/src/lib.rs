//! Rust port of core graph-learning and network-analysis primitives from
//! [`scikit-network`](https://github.com/sknetwork-team/scikit-network).
//!
//! Graphs are represented as sparse adjacency matrices (`sprs::CsMat<f64>`).
//! Estimators follow scikit-learn-style `fit` / `predict` / `transform`
//! workflows where applicable.
//!
//! # Module index
//!
//! | Module | Algorithms |
//! |--------|------------|
//! | [`classification`] | Label propagation, diffusion, NN classifiers |
//! | [`clustering`] | Louvain, Leiden, propagation clustering |
//! | [`data`] | Edge-list / CSV / GraphML parsers, Netset loader |
//! | [`embedding`] | Spring, ForceAtlas, spectral, SVD, random projection |
//! | [`gnn`] | GNN layers, activations, classifiers |
//! | [`hierarchy`] | Paris, Louvain hierarchy |
//! | [`linalg`] | Sparse eig/SVD solvers, normalizers, operators |
//! | [`linkpred`] | Nearest-neighbor link prediction |
//! | [`path`] | Shortest paths, BFS, distances |
//! | [`ranking`] | PageRank, HITS, Katz, closeness, betweenness |
//! | [`regression`] | Diffusion regression |
//! | [`topology`] | Cliques, cores, cycles, triangles, structure |
//! | [`utils`] | Input checking, formatting, value conversion |
//! | [`visualization`] | SVG graph and dendrogram rendering |
//!
//! # Agent and parity documentation
//!
//! - Human README: repository `README.md`
//! - AI agent guide: repository `AGENTS.md`
//! - Python parity tracker: repository `PORTING_MEMO.md`
//! - Pre-publish checklist: repository `sknetwork-rs/docs/PUBLISHING.md`
//!
//! # Examples
//!
//! ```rust,no_run
//! use sknetwork_rs::ranking::pagerank::PageRank;
//! use sprs::CsMat;
//!
//! let adjacency = CsMat::<f64>::eye(4);
//! let mut algo = PageRank::default();
//! let _ = algo.fit_predict(&adjacency, None, None, None, false);
//! ```

#![warn(missing_docs)]

#[cfg(feature = "bench")]
#[doc(hidden)]
pub mod bench;
pub mod classification;
pub mod clustering;
pub mod data;
pub mod embedding;
pub mod gnn;
pub mod hierarchy;
pub mod linalg;
pub mod linkpred;
pub mod path;
pub mod ranking;
pub mod regression;
pub mod topology;
pub mod utils;
pub mod visualization;

#[cfg(test)]
mod contract_matrix;
