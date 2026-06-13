//! Graph neural network building blocks and estimators.
//!
//! Includes activation/loss/optimizer primitives, convolutional layers, and
//! classifier orchestration with typed training and inference errors.
//!
//! # Examples
//!
//! ```rust,no_run
//! use ndarray::{Array1, Array2};
//! use sknetwork_rs::gnn::base::BaseGNN;
//! use sknetwork_rs::gnn::gnn_classifier::GNNClassifier;
//! use sprs::CsMat;
//!
//! let adjacency = CsMat::<f64>::eye(4);
//! let features = Array2::<f64>::ones((4, 3));
//! let labels = Array1::from_vec(vec![0, 1, 0, 1]);
//! let mut clf = GNNClassifier::new(vec![3, 4, 2], "cross_entropy", "adam", 1e-2, 2).unwrap();
//! let _ = clf.fit_predict(&adjacency, &features, &labels);
//! ```

/// Activation functions for GNN layers.
pub mod activation;
/// Base submodule.
pub mod base;
/// Base Activation submodule.
pub mod base_activation;
/// Base Layer submodule.
pub mod base_layer;
/// Gnn Classifier submodule.
pub mod gnn_classifier;
/// Layer submodule.
pub mod layer;
/// Loss submodule.
pub mod loss;
/// Neighbor Sampler submodule.
pub mod neighbor_sampler;
/// Optimizer submodule.
pub mod optimizer;
/// Utils submodule.
pub mod utils;

