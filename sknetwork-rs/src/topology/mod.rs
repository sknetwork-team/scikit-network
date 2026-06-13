//! Topological analysis utilities for sparse graphs.
//!
//! Includes cycle/core/clique/triangle/structure routines and
//! Weisfeiler-Lehman relabeling helpers.
//!
//! # Examples
//!
//! ```rust,no_run
//! use sknetwork_rs::topology::core::get_core_decomposition;
//! use sprs::CsMat;
//!
//! let adjacency = CsMat::<f64>::eye(4);
//! let _core = get_core_decomposition(&adjacency);
//! ```

/// Clique counting via core-ordered DAG listing.
pub mod cliques;
/// Core submodule.
pub mod core;
/// Cycles submodule.
pub mod cycles;
/// Structure submodule.
pub mod structure;
/// Triangles submodule.
pub mod triangles;
/// Weisfeiler Lehman submodule.
pub mod weisfeiler_lehman;

