//! Path and reachability algorithms.
//!
//! Provides shortest-path, DAG traversal, and graph-search utilities.
//!
//! # Examples
//!
//! ```rust,no_run
//! use sknetwork_rs::path::search::breadth_first_search;
//! use sprs::CsMat;
//!
//! let adjacency = CsMat::<f64>::eye(4);
//! let _order = breadth_first_search(&adjacency, 0).unwrap();
//! ```

/// DAG extraction from distance or node order.
pub mod dag;
/// Breadth-first and multi-source distance routines.
pub mod distances;
/// Graph search primitives.
pub mod search;
/// Shortest-path DAG construction.
pub mod shortest_path;

