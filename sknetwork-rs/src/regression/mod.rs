//! Graph-based regression estimators.
//!
//! Contains diffusion-style regression methods and base regression contracts.
//!
//! # Examples
//!
//! ```rust,no_run
//! use sknetwork_rs::regression::diffusion::Diffusion;
//! use sprs::CsMat;
//!
//! let adjacency = CsMat::<f64>::eye(4);
//! let mut algo = Diffusion::default();
//! let _values = algo.fit_predict(&adjacency, None, None, None, None, false).unwrap();
//! ```

/// Shared regression state and prediction contracts.
pub mod base;
/// Diffusion and Dirichlet graph regression estimators.
pub mod diffusion;

