//! Shared utility helpers.
//!
//! Includes input checking/formatting, sparse-neighbor helpers, value
//! conversion, membership transforms, and spatial indexing.
//!
//! # Examples
//!
//! ```rust,no_run
//! use sknetwork_rs::utils::check::check_square;
//!
//! let _ = check_square((4, 4)).unwrap();
//! ```

/// Input validation helpers for adjacency matrices and algorithm parameters.
pub mod check;
/// Format submodule.
pub mod format;
/// Membership submodule.
pub mod membership;
/// Neighbors submodule.
pub mod neighbors;
/// Spatial Index submodule.
pub mod spatial_index;
/// Tfidf submodule.
pub mod tfidf;
/// Values submodule.
pub mod values;

