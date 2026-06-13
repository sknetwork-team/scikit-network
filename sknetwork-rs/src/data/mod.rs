//! Data loading and parsing utilities.
//!
//! Provides local/remote dataset loaders and graph parsers for edge lists,
//! adjacency lists, CSV-like inputs, and GraphML-like sources.
//!
//! # Examples
//!
//! ```rust,no_run
//! use sknetwork_rs::data::parse::from_edge_list;
//!
//! let edges = vec![(0_i64, 1_i64, None), (1_i64, 2_i64, Some(1.0))];
//! let _dataset = from_edge_list(&edges, true, false, true, false, None, Some(true)).unwrap();
//! ```

/// Dataset container types and attribute storage.
pub mod base;
/// Netset loading, saving, and data-home management.
pub mod load;
/// Edge-list, CSV, and GraphML graph parsers.
pub mod parse;
#[cfg(test)]
pub mod test_graphs;

