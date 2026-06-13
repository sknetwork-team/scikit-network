use sprs::CsMat;

use crate::utils::format::{MatrixInput, get_adjacency_values};
use crate::utils::values::ValuesInput;

/// Errors raised while computing PageRank scores.
#[derive(Debug, Clone, PartialEq)]
pub enum PageRankError {
    /// The damping factor is outside `(0, 1)`.
    InvalidDampingFactor,
    /// Adjacency or value formatting checks failed.
    InvalidInput,
}

/// PageRank centrality estimator on sparse adjacency matrices.
///
/// The implementation supports square and bipartite inputs (via value stacking
/// in `get_adjacency_values`) and stores row/column score splits for bipartite
/// graphs.
#[derive(Debug, Clone)]
pub struct PageRank {
    /// Teleportation complement in `(0, 1)`.
    pub damping_factor: f64,
    /// Maximum number of power-iteration steps.
    pub n_iter: usize,
    /// L1 convergence threshold between successive iterates.
    pub tol: f64,
    /// Whether the last fit used a bipartite adjacency layout.
    pub bipartite: bool,
    /// Fitted row scores (or full scores for square graphs).
    pub scores: Vec<f64>,
    /// Row-node scores when the input is bipartite.
    pub scores_row: Option<Vec<f64>>,
    /// Column-node scores when the input is bipartite.
    pub scores_col: Option<Vec<f64>>,
}

impl Default for PageRank {
    fn default() -> Self {
        Self::new(0.85, 50, 1e-6).unwrap()
    }
}

impl PageRank {
    /// Creates a PageRank estimator with explicit convergence settings.
    ///
    /// # Arguments
    /// - `damping_factor`: Teleportation complement in `(0, 1)`.
    /// - `n_iter`: Maximum number of power-iteration steps.
    /// - `tol`: L1 convergence threshold between successive iterates.
    ///
    /// # Errors
    /// Returns [`PageRankError::InvalidDampingFactor`] when `damping_factor`
    /// is outside `(0, 1)`.
    pub fn new(damping_factor: f64, n_iter: usize, tol: f64) -> Result<Self, PageRankError> {
        if !(0.0..1.0).contains(&damping_factor) {
            return Err(PageRankError::InvalidDampingFactor);
        }
        Ok(Self {
            damping_factor,
            n_iter,
            tol,
            bipartite: false,
            scores: Vec::new(),
            scores_row: None,
            scores_col: None,
        })
    }

    fn pagerank_power_iteration(
        adjacency: &CsMat<f64>,
        restart: &[f64],
        damping_factor: f64,
        n_iter: usize,
        tol: f64,
    ) -> Vec<f64> {
        let n = adjacency.rows();
        let mut scores = vec![1.0 / n as f64; n];
        let mut out_weight = vec![0.0; n];
        for i in 0..n {
            out_weight[i] = adjacency
                .outer_view(i)
                .map(|row| row.data().iter().sum())
                .unwrap_or(0.0);
        }

        for _ in 0..n_iter {
            let mut next = vec![1.0 - damping_factor; n];
            for i in 0..n {
                next[i] *= restart[i];
            }

            let dangling_mass: f64 = (0..n)
                .filter(|&i| out_weight[i] == 0.0)
                .map(|i| scores[i])
                .sum();

            for i in 0..n {
                if let Some(row) = adjacency.outer_view(i) {
                    if out_weight[i] > 0.0 {
                        let coeff = damping_factor * scores[i] / out_weight[i];
                        for (j, v) in row.iter() {
                            next[j] += coeff * v;
                        }
                    }
                }
            }

            let dangling_coeff = damping_factor * dangling_mass;
            for i in 0..n {
                next[i] += dangling_coeff * restart[i];
            }

            let diff: f64 = next
                .iter()
                .zip(scores.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            scores = next;
            if diff <= tol {
                break;
            }
        }
        let s: f64 = scores.iter().sum();
        if s > 0.0 {
            for x in &mut scores {
                *x /= s;
            }
        }
        scores
    }

    /// Fits the estimator and stores node scores.
    ///
    /// # Errors
    /// Returns [`PageRankError::InvalidInput`] when matrix/value formatting
    /// checks fail.
    pub fn fit(
        &mut self,
        input_matrix: &CsMat<f64>,
        weights: Option<ValuesInput>,
        weights_row: Option<ValuesInput>,
        weights_col: Option<ValuesInput>,
        force_bipartite: bool,
    ) -> Result<(), PageRankError> {
        let (adjacency, values, bipartite) = get_adjacency_values(
            MatrixInput::Sparse(input_matrix.to_owned()),
            true,
            force_bipartite,
            false,
            weights,
            weights_row,
            weights_col,
            0.0,
            Some("probs"),
        )
        .map_err(|_| PageRankError::InvalidInput)?;

        self.bipartite = bipartite;
        self.scores = Self::pagerank_power_iteration(
            &adjacency,
            &values,
            self.damping_factor,
            self.n_iter,
            self.tol,
        );
        if self.bipartite {
            let n_row = input_matrix.rows();
            self.scores_row = Some(self.scores[..n_row].to_vec());
            self.scores_col = Some(self.scores[n_row..].to_vec());
            self.scores = self.scores_row.clone().unwrap_or_default();
        }
        Ok(())
    }

    /// Fits the estimator and returns row scores.
    ///
    /// # Errors
    /// Returns the same errors as [`PageRank::fit`].
    pub fn fit_predict(
        &mut self,
        input_matrix: &CsMat<f64>,
        weights: Option<ValuesInput>,
        weights_row: Option<ValuesInput>,
        weights_col: Option<ValuesInput>,
        force_bipartite: bool,
    ) -> Result<Vec<f64>, PageRankError> {
        self.fit(
            input_matrix,
            weights,
            weights_row,
            weights_col,
            force_bipartite,
        )?;
        Ok(self.scores.clone())
    }

    /// Returns fitted scores for rows or columns.
    ///
    /// For non-bipartite usage, `columns = true` returns an empty vector.
    pub fn predict(&self, columns: bool) -> Vec<f64> {
        if columns {
            self.scores_col.clone().unwrap_or_default()
        } else {
            self.scores.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use sprs::TriMat;

    use super::*;
    use crate::data::test_graphs::test_bigraph;

    fn cyclic_digraph(n: usize) -> CsMat<f64> {
        let mut tri = TriMat::<f64>::new((n, n));
        for i in 0..n {
            tri.add_triplet(i, (i + 1) % n, 1.0);
        }
        tri.to_csr::<usize>()
    }

    #[test]
    fn test_params() {
        assert!(PageRank::new(1789.0, 10, 1e-6).is_err());
    }

    #[test]
    fn test_cycle() {
        let adjacency = cyclic_digraph(5);
        let truth = vec![0.2; 5];
        let mut pr = PageRank::new(0.85, 100, 1e-10).unwrap();
        let scores = pr.fit_predict(&adjacency, None, None, None, false).unwrap();
        let err: f64 = scores
            .iter()
            .zip(truth.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(err < 1e-6);
    }

    #[test]
    fn test_seeding() {
        let adjacency = cyclic_digraph(5);
        let mut pr = PageRank::default();
        let mut arr = vec![0.0; 5];
        arr[0] = 1.0;
        let s1 = pr
            .fit_predict(
                &adjacency,
                Some(ValuesInput::Vector(arr)),
                None,
                None,
                false,
            )
            .unwrap();
        let mut m = HashMap::new();
        m.insert(0usize, 1.0f64);
        let s2 = pr
            .fit_predict(&adjacency, Some(ValuesInput::Map(m)), None, None, false)
            .unwrap();
        let err: f64 = s1.iter().zip(s2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(err < 1e-8);
    }

    #[test]
    fn test_bigraph_columns() {
        let biadjacency = test_bigraph();
        let mut pr = PageRank::default();
        let mut col_weights = HashMap::new();
        col_weights.insert(0usize, 1.0);
        pr.fit(
            &biadjacency,
            None,
            None,
            Some(ValuesInput::Map(col_weights)),
            true,
        )
        .unwrap();
        let col = pr.predict(true);
        let scores_col = pr.scores_col.clone().unwrap_or_default();
        let err: f64 = col
            .iter()
            .zip(scores_col.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(err < 1e-12);
    }
}
