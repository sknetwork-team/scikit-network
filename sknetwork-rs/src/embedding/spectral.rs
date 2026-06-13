use sprs::CsMat;

use crate::topology::structure::is_connected;
use crate::utils::format::{MatrixInput, get_adjacency};

/// Errors raised by [`Spectral`] embedding.
#[derive(Debug, Clone, PartialEq)]
pub enum SpectralError {
    /// The input matrix is too small or adjacency extraction failed.
    InvalidInput,
    /// `decomposition` is not `rw` or `laplacian`.
    UnknownDecomposition,
}

/// Spectral graph embedding via random-walk or Laplacian eigenvectors.
#[derive(Debug, Clone)]
pub struct Spectral {
    /// Number of embedding dimensions to retain.
    pub n_components: usize,
    /// Decomposition mode: `rw` (random walk) or `laplacian`.
    pub decomposition: String,
    /// Diagonal regularization strength (`< 0` auto-selects from connectivity).
    pub regularization: f64,
    /// Row-normalize the final embedding when true.
    pub normalized: bool,
    /// Whether the last fit used a bipartite layout.
    pub bipartite: bool,
    /// Whether regularization was active during the last fit.
    pub regularized: bool,
    /// Fitted node embedding (row nodes for bipartite inputs).
    pub embedding: Vec<Vec<f64>>,
    /// Row-node embedding when the input is bipartite.
    pub embedding_row: Option<Vec<Vec<f64>>>,
    /// Column-node embedding when the input is bipartite.
    pub embedding_col: Option<Vec<Vec<f64>>>,
    /// Selected eigenvalues from the last fit.
    pub eigenvalues: Vec<f64>,
    /// Selected eigenvectors from the last fit.
    pub eigenvectors: Vec<Vec<f64>>,
}

impl Default for Spectral {
    fn default() -> Self {
        Self::new(2, "rw", -1.0, true)
    }
}

impl Spectral {
    /// Creates a spectral embedding estimator.
    ///
    /// # Arguments
    /// - `n_components`: Number of embedding dimensions to retain.
    /// - `decomposition`: `rw` for random-walk Laplacian or `laplacian` for combinatorial.
    /// - `regularization`: Diagonal regularization (`< 0` auto-selects from connectivity).
    /// - `normalized`: Row-normalize the final embedding when true.
    pub fn new(
        n_components: usize,
        decomposition: &str,
        regularization: f64,
        normalized: bool,
    ) -> Self {
        Self {
            n_components,
            decomposition: decomposition.to_lowercase(),
            regularization,
            normalized,
            bipartite: false,
            regularized: false,
            embedding: Vec::new(),
            embedding_row: None,
            embedding_col: None,
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
        }
    }

    /// Resolves the effective regularization from connectivity and user input.
    ///
    /// # Arguments
    /// - `regularization`: Requested regularization (`< 0` triggers auto-selection).
    /// - `adjacency`: Graph adjacency used for connectivity checks.
    pub fn get_regularization(&self, regularization: f64, adjacency: &CsMat<f64>) -> f64 {
        if regularization < 0.0 {
            match is_connected(adjacency, "strong", false) {
                Ok(true) => 0.0,
                _ => regularization.abs(),
            }
        } else {
            regularization
        }
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

    fn normalize_columns(mut x: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        if x.is_empty() {
            return x;
        }
        let n = x.len();
        let k = x[0].len();
        for c in 0..k {
            // Modified Gram-Schmidt against previous columns
            for p in 0..c {
                let mut dot = 0.0;
                for i in 0..n {
                    dot += x[i][c] * x[i][p];
                }
                for i in 0..n {
                    x[i][c] -= dot * x[i][p];
                }
            }
            let norm = (0..n).map(|i| x[i][c] * x[i][c]).sum::<f64>().sqrt();
            if norm > 0.0 {
                for i in 0..n {
                    x[i][c] /= norm;
                }
            }
        }
        x
    }

    fn apply_s(adjacency: &CsMat<f64>, degrees: &[f64], x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = adjacency.rows();
        let k = x.first().map(|r| r.len()).unwrap_or(0);
        let mut out = vec![vec![0.0; k]; n];
        for i in 0..n {
            if let Some(row) = adjacency.outer_view(i) {
                let inv_i = 1.0 / degrees[i].sqrt();
                for (j, v) in row.iter() {
                    let coeff = v * inv_i * (1.0 / degrees[j].sqrt());
                    for c in 0..k {
                        out[i][c] += coeff * x[j][c];
                    }
                }
            }
        }
        out
    }

    fn apply_l(adjacency: &CsMat<f64>, degrees: &[f64], x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = adjacency.rows();
        let k = x.first().map(|r| r.len()).unwrap_or(0);
        let mut out = vec![vec![0.0; k]; n];
        for i in 0..n {
            for c in 0..k {
                out[i][c] += degrees[i] * x[i][c];
            }
            if let Some(row) = adjacency.outer_view(i) {
                for (j, v) in row.iter() {
                    for c in 0..k {
                        out[i][c] -= v * x[j][c];
                    }
                }
            }
        }
        out
    }

    fn sort_indices_by_values(values: &[f64], ascending: bool) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..values.len()).collect();
        if ascending {
            idx.sort_by(|&a, &b| {
                values[a]
                    .partial_cmp(&values[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            idx.sort_by(|&a, &b| {
                values[b]
                    .partial_cmp(&values[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        idx
    }

    fn orthogonal_iteration(
        adjacency: &CsMat<f64>,
        degrees: &[f64],
        m: usize,
        mode_rw: bool,
        n_iter: usize,
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = adjacency.rows();
        let mut q = vec![vec![0.0; m]; n];
        for i in 0..n {
            for c in 0..m {
                q[i][c] = ((i as f64 + 1.0) * (c as f64 + 2.0)).sin();
            }
        }
        q = Self::normalize_columns(q);

        for _ in 0..n_iter {
            let y = if mode_rw {
                Self::apply_s(adjacency, degrees, &q)
            } else {
                // For Laplacian smallest eigs, iterate on -L to get largest then flip ordering later.
                let mut ly = Self::apply_l(adjacency, degrees, &q);
                for i in 0..n {
                    for c in 0..m {
                        ly[i][c] = -ly[i][c];
                    }
                }
                ly
            };
            q = Self::normalize_columns(y);
        }

        let aq = if mode_rw {
            Self::apply_s(adjacency, degrees, &q)
        } else {
            Self::apply_l(adjacency, degrees, &q)
        };
        let mut evals = vec![0.0; m];
        for c in 0..m {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..n {
                num += q[i][c] * aq[i][c];
                den += q[i][c] * q[i][c];
            }
            if den > 0.0 {
                evals[c] = num / den;
            }
        }
        (evals, q)
    }

    /// Fits spectral eigenvectors and stores the embedding.
    ///
    /// # Arguments
    /// - `input_matrix`: Sparse adjacency or biadjacency input.
    /// - `force_bipartite`: Treat the input as bipartite even when square.
    ///
    /// # Errors
    /// Returns:
    /// - [`SpectralError::InvalidInput`] for empty or invalid inputs.
    /// - [`SpectralError::UnknownDecomposition`] for unsupported decomposition names.
    pub fn fit(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<(), SpectralError> {
        let (adjacency, bipartite) = get_adjacency(
            MatrixInput::Sparse(input_matrix.to_owned()),
            false,
            force_bipartite,
            false,
            false,
        )
        .map_err(|_| SpectralError::InvalidInput)?;
        self.bipartite = bipartite;

        let n = adjacency.rows();
        if n < 2 {
            return Err(SpectralError::InvalidInput);
        }
        let k = self.n_components.min(n.saturating_sub(2)).max(1);
        let reg = self.get_regularization(self.regularization, &adjacency);
        self.regularized = reg > 0.0;

        let use_rw = match self.decomposition.as_str() {
            "rw" => true,
            "laplacian" => false,
            _ => return Err(SpectralError::UnknownDecomposition),
        };
        let mut degrees = vec![0.0; n];
        for i in 0..n {
            let mut d = 0.0;
            if let Some(row) = adjacency.outer_view(i) {
                d = row.data().iter().sum();
            }
            d += reg;
            if d <= 0.0 {
                d = 1.0;
            }
            degrees[i] = d;
        }

        let (eigs, evecs, embedding) = if use_rw {
            let (vals, vecs_all) =
                Self::orthogonal_iteration(&adjacency, &degrees, k + 1, true, 40);
            let sorted = Self::sort_indices_by_values(&vals, false);
            let selected: Vec<usize> = sorted.into_iter().skip(1).take(k).collect();
            let mut eigvecs = vec![vec![0.0; k]; n];
            for (c_new, &c_old) in selected.iter().enumerate() {
                for i in 0..n {
                    eigvecs[i][c_new] = vecs_all[i][c_old] / degrees[i].sqrt();
                }
            }
            let eigvals: Vec<f64> = selected.iter().map(|&idx| vals[idx]).collect();
            (eigvals, eigvecs.clone(), eigvecs)
        } else {
            let (vals, vecs_all) =
                Self::orthogonal_iteration(&adjacency, &degrees, k + 1, false, 60);
            let sorted = Self::sort_indices_by_values(&vals, true);
            let selected: Vec<usize> = sorted.into_iter().skip(1).take(k).collect();
            let mut eigvecs = vec![vec![0.0; k]; n];
            for (c_new, &c_old) in selected.iter().enumerate() {
                for i in 0..n {
                    eigvecs[i][c_new] = vecs_all[i][c_old];
                }
            }
            let eigvals: Vec<f64> = selected.iter().map(|&idx| vals[idx]).collect();
            (eigvals, eigvecs.clone(), eigvecs)
        };

        let mut embedding = embedding;
        if self.normalized && !embedding.is_empty() {
            Self::row_l2_normalize(&mut embedding);
        }

        self.eigenvectors = evecs;
        self.eigenvalues = eigs;
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
    /// Returns the same errors as [`Spectral::fit`].
    pub fn fit_transform(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<Vec<Vec<f64>>, SpectralError> {
        self.fit(input_matrix, force_bipartite)?;
        Ok(self.embedding.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::{
        test_bigraph, test_digraph, test_disconnected_graph, test_graph,
    };

    #[test]
    fn test_shapes_and_normalization() {
        for adjacency in [test_graph(), test_disconnected_graph()] {
            let mut spectral = Spectral::new(3, "rw", -1.0, true);
            let embedding = spectral.fit_transform(&adjacency, false).unwrap();
            assert_eq!(embedding[0].len(), 3);
            for row in &embedding {
                let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
                assert!((norm - 1.0).abs() < 1e-8 || norm == 0.0);
            }
        }
    }

    #[test]
    fn test_directed_and_bipartite() {
        let mut spectral = Spectral::new(3, "laplacian", -1.0, false);
        let embedding = spectral.fit_transform(&test_digraph(), false).unwrap();
        assert_eq!(embedding.len(), 10);

        let bi = test_bigraph();
        let (n_row, n_col) = bi.shape();
        let mut spectral = Spectral::new(3, "rw", -1.0, false);
        spectral.fit(&bi, false).unwrap();
        assert_eq!(
            spectral.embedding_row.clone().unwrap_or_default().len(),
            n_row
        );
        assert_eq!(
            spectral.embedding_col.clone().unwrap_or_default().len(),
            n_col
        );
    }

    #[test]
    fn test_regularization_helper() {
        let adjacency = test_graph();
        let method = Spectral::default();
        assert_eq!(method.get_regularization(-1.0, &adjacency), 0.0);
    }

    #[test]
    fn test_invalid_decomposition_rejected() {
        let adjacency = test_graph();
        let mut spectral = Spectral::new(3, "bad-decomp", -1.0, true);
        assert_eq!(
            spectral.fit(&adjacency, false),
            Err(SpectralError::UnknownDecomposition)
        );
    }

    #[test]
    fn test_decomposition_case_insensitive() {
        let adjacency = test_graph();
        let mut spectral = Spectral::new(3, "LaPlAcIaN", -1.0, false);
        let embedding = spectral.fit_transform(&adjacency, false).unwrap();
        assert_eq!(embedding.len(), adjacency.rows());
    }
}
