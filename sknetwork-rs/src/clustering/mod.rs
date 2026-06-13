//! Graph clustering and community-detection algorithms.
//!
//! Includes modularity-based methods (`Louvain`, `Leiden`) and utility
//! routines for post-processing cluster labels and aggregates.
//!
//! # Examples
//!
//! ```rust,no_run
//! use sknetwork_rs::clustering::louvain::Louvain;
//! use sprs::CsMat;
//!
//! let adjacency = CsMat::<f64>::eye(4);
//! let mut algo = Louvain::default();
//! let _labels = algo.fit_predict(&adjacency, false).unwrap();
//! ```

pub mod base;
pub mod kcenters;
pub mod leiden;
pub mod louvain;
pub mod metrics;
pub mod postprocess;
pub mod propagation_clustering;

