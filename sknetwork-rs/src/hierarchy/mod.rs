//! Hierarchical clustering and dendrogram post-processing.
//!
//! Provides Paris and Louvain-iteration hierarchies plus utilities for
//! dendrogram validation, reordering, and bipartite splitting.
//!
//! # Examples
//!
//! ```rust,no_run
//! use sknetwork_rs::hierarchy::paris::Paris;
//! use sprs::CsMat;
//!
//! let adjacency = CsMat::<f64>::eye(4);
//! let mut algo = Paris::default();
//! let _dendrogram = algo.fit_predict(&adjacency, false).unwrap();
//! ```

/// Shared hierarchy traits and error types.
pub mod base;
/// Louvain Hierarchy submodule.
pub mod louvain_hierarchy;
/// Metrics submodule.
pub mod metrics;
/// Paris submodule.
pub mod paris;
/// Postprocess submodule.
pub mod postprocess;

