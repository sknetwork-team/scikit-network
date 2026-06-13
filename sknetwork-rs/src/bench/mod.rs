//! Shared logic for benchmark binaries (`benchmark_ipc`, `benchmark_ranking`, …).
//! Extend with other domains (e.g. `clustering`) as more modules gain Python-vs-Rust benches.

pub mod classification;
pub mod clustering;
pub mod data;
pub mod embedding;
pub mod gnn;
pub mod hierarchy;
pub mod linkpred;
pub mod linalg;
pub mod path;
pub mod ranking;
pub mod regression;
pub mod topology;
