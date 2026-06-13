//! Classification algorithms and evaluation helpers.
//!
//! This module groups supervised label-propagation style estimators and
//! classification metrics on sparse graph inputs.
//!
//! # Examples
//!
//! ```rust,no_run
//! use std::collections::HashMap;
//!
//! use sknetwork_rs::classification::propagation::Propagation;
//! use sknetwork_rs::utils::values::ValuesInput;
//! use sprs::CsMat;
//!
//! let adjacency = CsMat::<f64>::eye(4);
//! let mut labels = HashMap::new();
//! labels.insert(0usize, 0.0);
//! labels.insert(1usize, 1.0);
//!
//! let mut algo = Propagation::default();
//! let _pred = algo.fit_predict(&adjacency, Some(ValuesInput::Map(labels)), None, None).unwrap();
//! ```

pub mod base;
pub mod diffusion;
pub mod metrics;
pub mod nn;
pub mod pagerank;
pub mod propagation;

// Python uses `classification.knn` for the NN classifier; keep a Rust alias
// module for path-level parity.
/// Alias module exposing [`nn::NNClassifier`] at the `classification::knn` path.
pub mod knn {
    pub use crate::classification::nn::{EmbeddingMethod, NNClassifier, NNClassifierError};
}

