//! Sparse linear-algebra primitives used by higher-level algorithms.
//!
//! Includes iterative eig/SVD routines, polynomial operators, normalizers, and
//! structured sparse-lowrank operator wrappers.
//!
//! # Examples
//!
//! ```rust,no_run
//! use ndarray::Array1;
//! use sknetwork_rs::linalg::polynome::Polynome;
//! use sprs::CsMat;
//!
//! let adjacency = CsMat::<f64>::eye(4);
//! let coefs = Array1::from_vec(vec![0.0, 1.0]);
//! let poly = Polynome::new(&adjacency, &coefs).unwrap();
//! let _ = poly.dot_vec(&Array1::ones(4));
//! ```

/// Safe sparse matrix multiplication helpers.
pub mod basics;
/// Dense linear algebra utilities.
pub mod dense_linalg;
/// Eig Solver submodule.
pub mod eig_solver;
/// Laplacian submodule.
pub mod laplacian;
/// Normalizer submodule.
pub mod normalizer;
/// Operators submodule.
pub mod operators;
/// Polynome submodule.
pub mod polynome;
/// Ppr Solver submodule.
pub mod ppr_solver;
/// Random number generation helpers for iterative solvers.
pub mod rng;
/// Sparse Lowrank submodule.
pub mod sparse_lowrank;
/// Sparse matrix-vector products.
pub mod sparse_matvec;
/// Cached sparse matrix-vector products.
pub mod sparse_matvec_cache;
/// Partial SVD solvers (Lanczos and randomized).
pub mod svd_solver;
/// Implicitly restarted Lanczos symmetric eigendecomposition.
pub mod symmetric_eigsh;

