use sprs::CsMat;

use crate::utils::format::{MatrixInput, get_adjacency};

/// Errors raised by [`RandomProjection`] embedding.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingError {
    /// Adjacency extraction or input formatting failed.
    InvalidInput,
}

/// Sparse random-projection graph embedding estimator.
#[derive(Debug, Clone)]
pub struct RandomProjection {
    /// Output embedding dimensionality.
    pub n_components: usize,
    /// Damping factor applied after each projection step.
    pub alpha: f64,
    /// Number of power-iteration projection rounds.
    pub n_iter: usize,
    /// When true, normalize rows by out-degree (random-walk scaling).
    pub random_walk: bool,
    /// Diagonal regularization strength (`< 0` disables).
    pub regularization: f64,
    /// Row-normalize the final embedding when true.
    pub normalized: bool,
    /// Fitted node embedding (row nodes for bipartite inputs).
    pub embedding: Vec<Vec<f64>>,
    /// Row-node embedding when the input is bipartite.
    pub embedding_row: Option<Vec<Vec<f64>>>,
    /// Column-node embedding when the input is bipartite.
    pub embedding_col: Option<Vec<Vec<f64>>>,
    /// Whether the last fit used a bipartite layout.
    pub bipartite: bool,
    /// Whether regularization was active during the last fit.
    pub regularized: bool,
}

impl Default for RandomProjection {
    fn default() -> Self {
        Self {
            n_components: 2,
            alpha: 0.5,
            n_iter: 3,
            random_walk: false,
            regularization: -1.0,
            normalized: true,
            embedding: Vec::new(),
            embedding_row: None,
            embedding_col: None,
            bipartite: false,
            regularized: false,
        }
    }
}

impl RandomProjection {
    /// Creates a random-projection estimator with explicit hyperparameters.
    ///
    /// # Arguments
    /// - `n_components`: Output embedding dimensionality.
    /// - `alpha`: Damping factor applied after each projection step.
    /// - `n_iter`: Number of power-iteration projection rounds.
    /// - `random_walk`: Normalize rows by out-degree when true.
    /// - `regularization`: Diagonal regularization strength (`< 0` disables).
    /// - `normalized`: Row-normalize the final embedding when true.
    pub fn new(
        n_components: usize,
        alpha: f64,
        n_iter: usize,
        random_walk: bool,
        regularization: f64,
        normalized: bool,
    ) -> Self {
        Self {
            n_components,
            alpha,
            n_iter,
            random_walk,
            regularization,
            normalized,
            ..Self::default()
        }
    }

    fn deterministic_random_matrix(n: usize, k: usize) -> Vec<Vec<f64>> {
        let mut out = vec![vec![0.0; k]; n];
        for i in 0..n {
            for j in 0..k {
                // deterministic pseudo-random values in [-1, 1]
                let v = (((i as f64 + 1.37) * (j as f64 + 3.11)).sin() * 104729.0).sin();
                out[i][j] = v;
            }
        }
        out
    }

    fn row_l2_normalize(x: &mut [Vec<f64>]) {
        for row in x {
            let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for v in row {
                    *v /= norm;
                }
            }
        }
    }

    fn sparse_dense_mul(
        adjacency: &CsMat<f64>,
        x: &[Vec<f64>],
        random_walk: bool,
        regularization: f64,
    ) -> Vec<Vec<f64>> {
        let n = adjacency.rows();
        let k = if x.is_empty() { 0 } else { x[0].len() };
        let mut out = vec![vec![0.0; k]; n];
        let mut row_sum = vec![0.0; n];
        if random_walk {
            for i in 0..n {
                row_sum[i] = adjacency
                    .outer_view(i)
                    .map(|row| row.data().iter().sum())
                    .unwrap_or(0.0);
            }
        }
        for i in 0..n {
            if let Some(row) = adjacency.outer_view(i) {
                let denom = if random_walk && row_sum[i] > 0.0 {
                    row_sum[i]
                } else {
                    1.0
                };
                for (j, v) in row.iter() {
                    let w = *v / denom;
                    for c in 0..k {
                        out[i][c] += w * x[j][c];
                    }
                }
                if regularization > 0.0 {
                    for c in 0..k {
                        out[i][c] += regularization * x[i][c] / denom.max(1.0);
                    }
                }
            }
        }
        out
    }

    /// Fits the estimator on an adjacency or biadjacency matrix.
    ///
    /// # Arguments
    /// - `input_matrix`: Sparse adjacency or biadjacency input.
    /// - `force_bipartite`: Treat the input as bipartite even when square.
    ///
    /// # Errors
    /// Returns [`EmbeddingError::InvalidInput`] when adjacency extraction fails.
    pub fn fit(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<(), EmbeddingError> {
        let (adjacency, bipartite) = get_adjacency(
            MatrixInput::Sparse(input_matrix.to_owned()),
            true,
            force_bipartite,
            false,
            false,
        )
        .map_err(|_| EmbeddingError::InvalidInput)?;
        self.bipartite = bipartite;
        self.regularized = self.regularization > 0.0;

        let n = adjacency.rows();
        let mut factor = Self::deterministic_random_matrix(n, self.n_components);
        let mut embedding = factor.clone();

        for _ in 0..self.n_iter {
            let mut next = Self::sparse_dense_mul(
                &adjacency,
                &factor,
                self.random_walk,
                self.regularization.max(0.0),
            );
            for row in &mut next {
                for v in row {
                    *v *= self.alpha;
                }
            }
            for i in 0..n {
                for c in 0..self.n_components {
                    embedding[i][c] += next[i][c];
                }
            }
            factor = next;
        }

        if self.normalized {
            Self::row_l2_normalize(&mut embedding);
        }

        self.embedding = embedding;
        if self.bipartite {
            let n_row = input_matrix.rows();
            self.embedding_row = Some(self.embedding[..n_row].to_vec());
            self.embedding_col = Some(self.embedding[n_row..].to_vec());
            self.embedding = self.embedding_row.clone().unwrap_or_default();
        } else {
            self.embedding_row = None;
            self.embedding_col = None;
        }
        Ok(())
    }

    /// Fits the estimator and returns the node embedding.
    ///
    /// # Errors
    /// Returns [`EmbeddingError::InvalidInput`] when [`RandomProjection::fit`] fails.
    pub fn fit_transform(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<Vec<Vec<f64>>, EmbeddingError> {
        self.fit(input_matrix, force_bipartite)?;
        Ok(self.embedding.clone())
    }

    /// Returns the fitted embedding without refitting.
    pub fn transform(&self) -> Vec<Vec<f64>> {
        self.embedding.clone()
    }

    /// Returns row or column embeddings for bipartite graphs.
    ///
    /// # Arguments
    /// - `columns`: Return column-node embeddings when true.
    pub fn predict(&self, columns: bool) -> Vec<Vec<f64>> {
        if columns {
            self.embedding_col.clone().unwrap_or_default()
        } else {
            self.embedding.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::{
        test_bigraph, test_digraph, test_disconnected_graph, test_graph,
    };

    #[test]
    fn test_random_projection() {
        for mut algo in [
            RandomProjection::default(),
            RandomProjection::new(2, 0.5, 3, true, -1.0, true),
        ] {
            let adjacency = test_graph();
            let embedding = algo.fit_transform(&adjacency, false).unwrap();
            assert_eq!(embedding[0].len(), 2);

            let embedding = algo.fit_transform(&adjacency, true).unwrap();
            assert_eq!(embedding[0].len(), 2);

            let adjacency = test_digraph();
            let embedding = algo.fit_transform(&adjacency, false).unwrap();
            assert_eq!(embedding[0].len(), 2);

            let adjacency = test_disconnected_graph();
            let embedding = algo.fit_transform(&adjacency, false).unwrap();
            assert_eq!(embedding[0].len(), 2);

            let biadjacency = test_bigraph();
            let embedding = algo.fit_transform(&biadjacency, false).unwrap();
            assert_eq!(embedding[0].len(), 2);
            assert_eq!(algo.embedding_col.clone().unwrap_or_default()[0].len(), 2);
        }
    }
}
