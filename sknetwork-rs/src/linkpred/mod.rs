//! Link-prediction models and shared link-prediction contracts.
//!
//! # Examples
//!
//! ```rust,no_run
//! use sknetwork_rs::linkpred::nn::NNLinker;
//! use sprs::CsMat;
//!
//! let adjacency = CsMat::<f64>::eye(4);
//! let mut linker = NNLinker::default();
//! let _ = linker.fit_predict(&adjacency, None);
//! ```

/// Shared link-prediction state and contracts.
pub mod base;
/// Nearest-neighbor link prediction from embeddings.
pub mod nn;

