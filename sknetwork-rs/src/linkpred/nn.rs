use sprs::{CsMat, TriMat};

use crate::embedding::spectral::Spectral;
use crate::utils::check::check_n_neighbors;
use crate::utils::format::{MatrixInput, get_adjacency};

/// Embedding backend used by [`NNLinker`].
#[derive(Debug, Clone)]
pub enum EmbeddingMethod {
    /// Spectral embedding via [`Spectral`].
    Spectral(Spectral),
}

#[derive(Debug, Clone)]
enum EmbeddingData {
    Dense(Vec<Vec<f64>>),
    Sparse(CsMat<f64>),
}

/// Errors raised by [`NNLinker`].
#[derive(Debug, Clone, PartialEq)]
pub enum NNError {
    /// Adjacency extraction or embedding fit failed.
    InvalidInput,
    /// `predict` or `fit_predict` called before `fit`.
    NotFitted,
}

/// Nearest-neighbor link-prediction estimator from node embeddings.
#[derive(Debug, Clone)]
pub struct NNLinker {
    /// Number of neighbors per row (`None` uses all columns).
    pub n_neighbors: Option<usize>,
    /// Minimum cosine similarity to keep a predicted link.
    pub threshold: f64,
    /// Optional embedding method (`None` uses normalized adjacency rows).
    pub embedding_method: Option<EmbeddingMethod>,
    /// Whether the last fit used a bipartite layout.
    pub bipartite: bool,
    /// Fitted link-score matrix.
    pub links: Option<CsMat<f64>>,
}

impl Default for NNLinker {
    fn default() -> Self {
        Self {
            n_neighbors: Some(10),
            threshold: 0.0,
            embedding_method: None,
            bipartite: false,
            links: None,
        }
    }
}

impl NNLinker {
    /// Creates a nearest-neighbor linker with explicit settings.
    ///
    /// # Arguments
    /// - `n_neighbors`: Neighbors per row (`None` uses all columns).
    /// - `threshold`: Minimum similarity to keep a predicted link.
    /// - `embedding_method`: Optional embedding backend.
    pub fn new(
        n_neighbors: Option<usize>,
        threshold: f64,
        embedding_method: Option<EmbeddingMethod>,
    ) -> Self {
        Self {
            n_neighbors,
            threshold,
            embedding_method,
            ..Self::default()
        }
    }

    fn normalize_rows_dense(x: &mut [Vec<f64>]) {
        for row in x {
            let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for v in row {
                    *v /= norm;
                }
            }
        }
    }

    fn normalize_rows_sparse(x: &CsMat<f64>) -> CsMat<f64> {
        let (n, m) = x.shape();
        let mut tri = TriMat::<f64>::new((n, m));
        for (i, row) in x.outer_iterator().enumerate() {
            let norm = row.data().iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for (j, v) in row.iter() {
                    tri.add_triplet(i, j, v / norm);
                }
            } else {
                for (j, v) in row.iter() {
                    tri.add_triplet(i, j, *v);
                }
            }
        }
        tri.to_csr::<usize>()
    }

    fn top_k_indices(sims: &[f64], k: usize) -> Vec<usize> {
        let kk = k.min(sims.len());
        if kk == 0 {
            return Vec::new();
        }
        let mut idx: Vec<usize> = (0..sims.len()).collect();
        if kk < idx.len() {
            idx.select_nth_unstable_by(kk - 1, |&a, &b| {
                sims[b]
                    .partial_cmp(&sims[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            idx.truncate(kk);
        }
        idx
    }

    fn fit_core(&self, embedding: &EmbeddingData, mask: &[bool], n_row: usize) -> CsMat<f64> {
        let n = match embedding {
            EmbeddingData::Dense(x) => x.len(),
            EmbeddingData::Sparse(x) => x.rows(),
        };
        let (index_col, n_col) = if n_row < n {
            ((n_row..n).collect::<Vec<_>>(), n - n_row)
        } else {
            ((0..n).collect::<Vec<_>>(), n)
        };
        let mut global_to_local = vec![usize::MAX; n];
        for (pos, &j) in index_col.iter().enumerate() {
            global_to_local[j] = pos;
        }
        let requested = self.n_neighbors.unwrap_or(n_col);
        let n_neighbors = if n_row < n {
            // Bipartite row->col predictions do not require self-exclusion.
            requested.min(n_col)
        } else {
            check_n_neighbors(requested, n_col)
        };

        let mut tri = TriMat::<f64>::new((n_row, n_col));

        match embedding {
            EmbeddingData::Dense(x) => {
                for i in 0..n_row {
                    if !mask[i] {
                        continue;
                    }
                    let vector = &x[i];
                    let mut sims = vec![0.0; n_col];
                    for (pos, &j) in index_col.iter().enumerate() {
                        let mut s = 0.0;
                        for c in 0..vector.len() {
                            s += x[j][c] * vector[c];
                        }
                        sims[pos] = s;
                    }
                    let nn = Self::top_k_indices(&sims, n_neighbors);
                    for &jpos in &nn {
                        if sims[jpos] >= self.threshold {
                            tri.add_triplet(i, jpos, sims[jpos]);
                        }
                    }
                }
            }
            EmbeddingData::Sparse(x) => {
                let csc = x.to_csc();
                for i in 0..n_row {
                    if !mask[i] {
                        continue;
                    }
                    let mut sims = vec![0.0; n_col];
                    if let Some(row_i) = x.outer_view(i) {
                        // Accumulate similarities through shared nonzero features.
                        for (&feature, &val_i) in row_i.indices().iter().zip(row_i.data().iter()) {
                            if let Some(col_view) = csc.outer_view(feature) {
                                for (&j, &val_j) in col_view
                                    .indices()
                                    .iter()
                                    .zip(col_view.data().iter())
                                {
                                    let pos = global_to_local[j];
                                    if pos != usize::MAX {
                                        sims[pos] += val_i * val_j;
                                    }
                                }
                            }
                        }
                    }
                    let mut nn = Self::top_k_indices(&sims, n_neighbors);
                    // Keep Python-like behavior when threshold <= 0: allow filling top-k with zeros.
                    if self.threshold <= 0.0 && nn.len() < n_neighbors {
                        for pos in 0..n_col {
                            if nn.len() >= n_neighbors {
                                break;
                            }
                            if !nn.contains(&pos) && sims[pos] == 0.0 {
                                nn.push(pos);
                            }
                        }
                    }
                    for &jpos in &nn {
                        if sims[jpos] >= self.threshold {
                            tri.add_triplet(i, jpos, sims[jpos]);
                        }
                    }
                }
            }
        }
        tri.to_csr::<usize>()
    }

    /// Fits link scores from embeddings or normalized adjacency.
    ///
    /// # Arguments
    /// - `input_matrix`: Sparse adjacency or biadjacency input.
    /// - `index`: Optional row indices to score (`None` scores all rows).
    ///
    /// # Errors
    /// Returns [`NNError::InvalidInput`] when adjacency or embedding fit fails.
    pub fn fit(
        &mut self,
        input_matrix: &CsMat<f64>,
        index: Option<&[usize]>,
    ) -> Result<(), NNError> {
        let n_row = input_matrix.rows();
        let (adjacency, bip) = get_adjacency(
            MatrixInput::Sparse(input_matrix.to_owned()),
            true,
            false,
            false,
            false,
        )
        .map_err(|_| NNError::InvalidInput)?;
        self.bipartite = bip;
        let mut mask = vec![false; n_row];
        if let Some(ix) = index {
            for &i in ix {
                if i < n_row {
                    mask[i] = true;
                }
            }
        } else {
            for m in &mut mask {
                *m = true;
            }
        }

        let embedding = match &mut self.embedding_method {
            None => EmbeddingData::Sparse(Self::normalize_rows_sparse(&adjacency)),
            Some(EmbeddingMethod::Spectral(method)) => {
                let mut emb = method
                    .fit_transform(&adjacency, false)
                    .map_err(|_| NNError::InvalidInput)?;
                Self::normalize_rows_dense(&mut emb);
                EmbeddingData::Dense(emb)
            }
        };
        let links = self.fit_core(&embedding, &mask, n_row);
        self.links = Some(links);
        Ok(())
    }

    /// Fits link scores and returns the predicted link matrix.
    ///
    /// # Errors
    /// Returns:
    /// - [`NNError::InvalidInput`] when [`NNLinker::fit`] fails.
    /// - [`NNError::NotFitted`] if links were not stored after fit.
    pub fn fit_predict(
        &mut self,
        input_matrix: &CsMat<f64>,
        index: Option<&[usize]>,
    ) -> Result<CsMat<f64>, NNError> {
        self.fit(input_matrix, index)?;
        self.links.clone().ok_or(NNError::NotFitted)
    }

    /// Returns the fitted link-score matrix.
    ///
    /// # Errors
    /// Returns [`NNError::NotFitted`] when called before `fit`.
    pub fn predict(&self) -> Result<CsMat<f64>, NNError> {
        self.links.clone().ok_or(NNError::NotFitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::{test_digraph, test_graph};
    use crate::utils::neighbors::get_degrees;

    fn test_bigraph() -> CsMat<f64> {
        let mut tri = TriMat::<f64>::new((6, 8));
        for i in 0..6 {
            tri.add_triplet(i, i % 8, 1.0);
            tri.add_triplet(i, (i + 2) % 8, 1.0);
        }
        tri.to_csr::<usize>()
    }

    #[test]
    fn test_link_prediction() {
        for input_matrix in [test_graph(), test_digraph(), test_bigraph()] {
            let n_neighbors = 3usize;
            let threshold = 0.5;
            let mut algo = NNLinker::new(Some(n_neighbors), threshold, None);
            let links = algo.fit_predict(&input_matrix, None).unwrap();
            assert_eq!(links.shape(), input_matrix.shape());
            assert!(get_degrees(&links, false).iter().all(|&d| d <= n_neighbors));
            assert!(links.data().iter().all(|&x| x >= threshold));

            let spectral = Spectral::new(5, "rw", -1.0, true);
            let mut algo = NNLinker::new(
                Some(n_neighbors),
                0.0,
                Some(EmbeddingMethod::Spectral(spectral)),
            );
            let links = algo.fit_predict(&input_matrix, None).unwrap();
            assert_eq!(links.shape(), input_matrix.shape());
        }
    }

    #[test]
    fn test_predict_not_fitted() {
        let algo = NNLinker::default();
        assert!(matches!(algo.predict(), Err(NNError::NotFitted)));
    }
}
