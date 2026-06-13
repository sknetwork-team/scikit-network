//! Ranking helpers for benchmark processes (no I/O).

use sprs::CsMat;

use crate::ranking::katz::Katz;
use crate::ranking::pagerank::PageRank;

pub fn pagerank_scores(
    adjacency: &CsMat<f64>,
    damping_factor: f64,
    n_iter: usize,
    tol: f64,
) -> Result<Vec<f64>, String> {
    let mut model = PageRank::new(damping_factor, n_iter, tol).map_err(|e| format!("{e:?}"))?;
    model
        .fit_predict(adjacency, None, None, None, false)
        .map_err(|e| format!("{e:?}"))
}

pub fn katz_scores(
    adjacency: &CsMat<f64>,
    damping_factor: f64,
    path_length: usize,
) -> Result<Vec<f64>, String> {
    let mut model = Katz::new(damping_factor, path_length);
    model.fit_predict(adjacency).map_err(|e| format!("{e:?}"))
}
