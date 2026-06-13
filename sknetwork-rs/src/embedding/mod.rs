//! Graph embedding and layout algorithms.
//!
//! Contains spectral and random-projection embeddings, SVD-style methods, and
//! force-based layout engines.
//!
//! # Examples
//!
//! ```rust,no_run
//! use sknetwork_rs::embedding::spring::Spring;
//! use sprs::CsMat;
//!
//! let adjacency = CsMat::<f64>::eye(4);
//! let mut algo = Spring::default();
//! let _embedding = algo.fit_transform(&adjacency, None, None).unwrap();
//! ```

/// ForceAtlas2 force-directed layout.
pub mod force_atlas;
/// Louvain-cluster membership embedding.
pub mod louvain_embedding;
/// Sparse random-projection graph embedding.
pub mod random_projection;
/// Spectral graph embedding (random-walk and Laplacian).
pub mod spectral;
/// Spring-electrical force layout.
pub mod spring;
/// SVD, GSVD, and PCA bipartite embeddings.
pub mod svd;

