//! Visualization helpers for graph and dendrogram rendering.
//!
//! # Examples
//!
//! ```rust,no_run
//! use sknetwork_rs::visualization::graphs::visualize_graph;
//! use sprs::CsMat;
//!
//! let adjacency = CsMat::<f64>::eye(4);
//! let _svg = visualize_graph(
//!     &adjacency,
//!     None,
//!     None,
//!     300.0,
//!     300.0,
//!     None,
//!     None,
//!     None,
//!     false,
//!     true,
//!     1.0,
//!     "gray",
//!     "red",
//! );
//! ```

/// Named color palettes for SVG rendering.
pub mod colors;
/// Dendrograms submodule.
pub mod dendrograms;
/// Graphs submodule.
pub mod graphs;

