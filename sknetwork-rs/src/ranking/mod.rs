//! Ranking and centrality estimators for sparse graphs.
//!
//! Includes PageRank, HITS, Katz, closeness, and betweenness variants with
//! shared post-processing helpers.
//!
//! # Examples
//!
//! ```rust,no_run
//! use sknetwork_rs::ranking::pagerank::PageRank;
//! use sprs::CsMat;
//!
//! let adjacency = CsMat::<f64>::eye(4);
//! let mut algo = PageRank::default();
//! let _scores = algo.fit_predict(&adjacency, None, None, None, false).unwrap();
//! ```

/// Shared ranking base trait and state contracts.
pub mod base;
/// Betweenness centrality estimator.
pub mod betweenness;
/// Closeness centrality estimator.
pub mod closeness;
/// HITS hub/authority ranking on bipartite graphs.
pub mod hits;
/// Katz centrality via damped path counts.
pub mod katz;
/// PageRank centrality with bipartite support.
pub mod pagerank;
/// Post-processing helpers for ranking scores.
pub mod postprocess;

