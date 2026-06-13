//! k-nearest-neighbor graph classifier with optional embeddings.

use ndarray::Array1;
use sprs::{CsMat, TriMat};

use crate::classification::base::{BaseClassifierError, BaseClassifierState};
use crate::embedding::louvain_embedding::{LouvainEmbedding, LouvainEmbeddingError};
use crate::embedding::spectral::{Spectral, SpectralError};
use crate::utils::check::{CheckError, check_n_neighbors};
use crate::utils::format::{MatrixInput, get_adjacency_values};
use crate::utils::spatial_index::{KDTree, brute_knn};
use crate::utils::values::ValuesInput;

/// Error type for [`NNClassifier`] fitting and prediction.
#[derive(Debug, Clone, PartialEq)]
pub enum NNClassifierError {
    /// Input matrix, seed, or neighbor-count validation failed.
    Check(CheckError),
    /// Wrapped shared classifier state error.
    Base(BaseClassifierError),
    /// Spectral embedding computation failed.
    Spectral(SpectralError),
    /// Louvain embedding computation failed.
    Louvain(LouvainEmbeddingError),
    /// No seed labels with non-negative values were provided.
    NoSeedLabels,
}

impl From<CheckError> for NNClassifierError {
    fn from(value: CheckError) -> Self {
        Self::Check(value)
    }
}
impl From<BaseClassifierError> for NNClassifierError {
    fn from(value: BaseClassifierError) -> Self {
        Self::Base(value)
    }
}
impl From<SpectralError> for NNClassifierError {
    fn from(value: SpectralError) -> Self {
        Self::Spectral(value)
    }
}
impl From<LouvainEmbeddingError> for NNClassifierError {
    fn from(value: LouvainEmbeddingError) -> Self {
        Self::Louvain(value)
    }
}

/// Embedding backend used by [`NNClassifier`].
#[derive(Debug, Clone)]
pub enum EmbeddingMethod {
    /// Spectral embedding coordinates.
    Spectral(Spectral),
    /// Louvain embedding coordinates.
    Louvain(LouvainEmbedding),
}

/// k-nearest-neighbor classifier on graph adjacency or embeddings.
#[derive(Debug, Clone)]
pub struct NNClassifier {
    /// Number of neighbors used for majority voting.
    pub n_neighbors: usize,
    /// Optional embedding method; uses sparse adjacency rows when `None`.
    pub embedding_method: Option<EmbeddingMethod>,
    /// Whether to L2-normalize embedding rows before distance queries.
    pub normalize: bool,
    /// Shared fitted-state container.
    pub state: BaseClassifierState,
}

impl Default for NNClassifier {
    fn default() -> Self {
        Self {
            n_neighbors: 3,
            embedding_method: None,
            normalize: true,
            state: BaseClassifierState::default(),
        }
    }
}

impl NNClassifier {
    /// Creates a k-NN classifier with explicit hyperparameters.
    pub fn new(
        n_neighbors: usize,
        embedding_method: Option<EmbeddingMethod>,
        normalize: bool,
    ) -> Self {
        Self {
            n_neighbors,
            embedding_method,
            normalize,
            state: BaseClassifierState::default(),
        }
    }

    fn instantiate_vars(labels: &[i32]) -> (Vec<usize>, Vec<usize>) {
        let mut train = Vec::new();
        let mut test = Vec::new();
        for (i, &label) in labels.iter().enumerate() {
            if label >= 0 {
                train.push(i);
            } else {
                test.push(i);
            }
        }
        (train, test)
    }

    fn normalize_rows(embedding: &mut [Vec<f64>]) {
        for row in embedding {
            let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for v in row {
                    *v /= norm;
                }
            }
        }
    }

    fn fit_core(
        &self,
        embedding: &[Vec<f64>],
        labels: &[i32],
        labels_contiguous: &[usize],
        labels_original: &[i32],
        index_train: &[usize],
        index_test: &[usize],
    ) -> Result<(CsMat<f64>, Vec<i32>), NNClassifierError> {
        if index_train.is_empty() {
            return Err(NNClassifierError::NoSeedLabels);
        }
        let k = check_n_neighbors(self.n_neighbors.max(1), index_train.len());
        let n_labels = labels_original.len();
        let mut tri = TriMat::<f64>::new((labels.len(), n_labels));
        let train_points: Vec<Vec<f64>> = index_train.iter().map(|&i| embedding[i].clone()).collect();
        let use_tree = !train_points.is_empty() && train_points[0].len() <= 24;
        let tree = if use_tree {
            KDTree::build(train_points.clone())
        } else {
            None
        };

        for &i in index_test {
            let vector = &embedding[i];
            let neigh_local = if let Some(t) = &tree {
                t.knn_query(vector, k)
            } else {
                brute_knn(&train_points, vector, k)
            };
            for local in neigh_local {
                let j = index_train[local];
                tri.add_triplet(i, labels_contiguous[j], 1.0);
            }
        }

        for &i in index_train {
            tri.add_triplet(i, labels_contiguous[i], 1.0);
        }

        let probs_raw = tri.to_csr::<usize>();
        let mut tri_norm = TriMat::<f64>::new(probs_raw.shape());
        for (i, row) in probs_raw.outer_iterator().enumerate() {
            let s: f64 = row.data().iter().sum();
            if s > 0.0 {
                for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                    tri_norm.add_triplet(i, j, v / s);
                }
            }
        }
        let probs = tri_norm.to_csr::<usize>();

        let mut labels_pred = vec![0i32; probs.rows()];
        for (i, slot) in labels_pred.iter_mut().enumerate().take(probs.rows()) {
            if let Some(row) = probs.outer_view(i) {
                let mut best_label = 0usize;
                let mut best_score = f64::NEG_INFINITY;
                for (&label, &score) in row.indices().iter().zip(row.data().iter()) {
                    if score > best_score {
                        best_score = score;
                        best_label = label;
                    }
                }
                *slot = labels_original[best_label];
            }
        }
        Ok((probs, labels_pred))
    }

    fn sparse_row_distance_sq(adjacency: &CsMat<f64>, i: usize, j: usize, inv_norms: &[f64]) -> f64 {
        let row_i = adjacency.outer_view(i);
        let row_j = adjacency.outer_view(j);
        match (row_i, row_j) {
            (Some(a), Some(b)) => {
                let ia = a.indices();
                let ib = b.indices();
                let va = a.data();
                let vb = b.data();
                let sa = inv_norms[i];
                let sb = inv_norms[j];
                let mut pa = 0usize;
                let mut pb = 0usize;
                let mut dist2 = 0.0;
                while pa < ia.len() && pb < ib.len() {
                    if ia[pa] == ib[pb] {
                        let d = va[pa] * sa - vb[pb] * sb;
                        dist2 += d * d;
                        pa += 1;
                        pb += 1;
                    } else if ia[pa] < ib[pb] {
                        let d = va[pa] * sa;
                        dist2 += d * d;
                        pa += 1;
                    } else {
                        let d = vb[pb] * sb;
                        dist2 += d * d;
                        pb += 1;
                    }
                }
                while pa < ia.len() {
                    let d = va[pa] * sa;
                    dist2 += d * d;
                    pa += 1;
                }
                while pb < ib.len() {
                    let d = vb[pb] * sb;
                    dist2 += d * d;
                    pb += 1;
                }
                dist2
            }
            (Some(a), None) => {
                let s = inv_norms[i];
                a.data().iter().map(|v| (v * s) * (v * s)).sum()
            }
            (None, Some(b)) => {
                let s = inv_norms[j];
                b.data().iter().map(|v| (v * s) * (v * s)).sum()
            }
            (None, None) => 0.0,
        }
    }

    fn fit_core_sparse(
        &self,
        adjacency: &CsMat<f64>,
        labels: &[i32],
        labels_contiguous: &[usize],
        labels_original: &[i32],
        index_train: &[usize],
        index_test: &[usize],
    ) -> Result<(CsMat<f64>, Vec<i32>), NNClassifierError> {
        if index_train.is_empty() {
            return Err(NNClassifierError::NoSeedLabels);
        }
        let k = check_n_neighbors(self.n_neighbors.max(1), index_train.len());
        let n_labels = labels_original.len();
        let mut tri = TriMat::<f64>::new((labels.len(), n_labels));

        let inv_norms: Vec<f64> = if self.normalize {
            (0..adjacency.rows())
                .map(|i| {
                    let norm = adjacency
                        .outer_view(i)
                        .map(|row| row.data().iter().map(|v| v * v).sum::<f64>().sqrt())
                        .unwrap_or(0.0);
                    if norm > 0.0 { 1.0 / norm } else { 1.0 }
                })
                .collect()
        } else {
            vec![1.0; adjacency.rows()]
        };

        for &i in index_test {
            let mut dist_local: Vec<(usize, f64)> = index_train
                .iter()
                .enumerate()
                .map(|(local, &j)| {
                    (
                        local,
                        Self::sparse_row_distance_sq(adjacency, i, j, &inv_norms),
                    )
                })
                .collect();
            dist_local.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (local, _) in dist_local.into_iter().take(k) {
                let j = index_train[local];
                tri.add_triplet(i, labels_contiguous[j], 1.0);
            }
        }

        for &i in index_train {
            tri.add_triplet(i, labels_contiguous[i], 1.0);
        }

        let probs_raw = tri.to_csr::<usize>();
        let mut tri_norm = TriMat::<f64>::new(probs_raw.shape());
        for (i, row) in probs_raw.outer_iterator().enumerate() {
            let s: f64 = row.data().iter().sum();
            if s > 0.0 {
                for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                    tri_norm.add_triplet(i, j, v / s);
                }
            }
        }
        let probs = tri_norm.to_csr::<usize>();

        let mut labels_pred = vec![0i32; probs.rows()];
        for (i, slot) in labels_pred.iter_mut().enumerate().take(probs.rows()) {
            if let Some(row) = probs.outer_view(i) {
                let mut best_label = 0usize;
                let mut best_score = f64::NEG_INFINITY;
                for (&label, &score) in row.indices().iter().zip(row.data().iter()) {
                    if score > best_score {
                        best_score = score;
                        best_label = label;
                    }
                }
                *slot = labels_original[best_label];
            }
        }
        Ok((probs, labels_pred))
    }

    /// Fits the classifier from sparse seed labels.
    ///
    /// # Errors
    /// Returns [`NNClassifierError::Check`] for formatting failures,
    /// [`NNClassifierError::NoSeedLabels`] when no seeds are provided,
    /// embedding errors from [`NNClassifierError::Spectral`] or
    /// [`NNClassifierError::Louvain`], and [`NNClassifierError::Base`] for
    /// bipartite split errors.
    pub fn fit(
        &mut self,
        input_matrix: &CsMat<f64>,
        labels: Option<ValuesInput>,
        labels_row: Option<ValuesInput>,
        labels_col: Option<ValuesInput>,
    ) -> Result<(), NNClassifierError> {
        let (adjacency, labels, bipartite) = get_adjacency_values(
            MatrixInput::Sparse(input_matrix.clone()),
            true,
            false,
            false,
            labels,
            labels_row,
            labels_col,
            -1.0,
            None,
        )?;
        let labels_i32: Vec<i32> = labels.iter().map(|x| *x as i32).collect();
        let mut labels_unique: Vec<i32> = labels_i32.iter().copied().filter(|x| *x >= 0).collect();
        labels_unique.sort_unstable();
        labels_unique.dedup();
        let label_to_idx: std::collections::HashMap<i32, usize> = labels_unique
            .iter()
            .enumerate()
            .map(|(i, &l)| (l, i))
            .collect();
        let labels_contiguous: Vec<usize> = labels_i32
            .iter()
            .map(|&l| label_to_idx.get(&l).copied().unwrap_or(0))
            .collect();
        let (index_train, index_test) = Self::instantiate_vars(&labels_i32);

        let mut dense_embedding: Option<Vec<Vec<f64>>> = None;
        match &mut self.embedding_method {
            None => {}
            Some(EmbeddingMethod::Spectral(method)) => {
                let mut emb = method.fit_transform(&adjacency, false)?;
                if self.normalize {
                    Self::normalize_rows(&mut emb);
                }
                dense_embedding = Some(emb);
            }
            Some(EmbeddingMethod::Louvain(method)) => {
                let mut emb = method.fit_transform(&adjacency, false)?;
                if self.normalize {
                    Self::normalize_rows(&mut emb);
                }
                dense_embedding = Some(emb);
            }
        }

        let (probs, labels_pred) = if let Some(e) = dense_embedding {
            self.fit_core(
                &e,
                &labels_i32,
                &labels_contiguous,
                &labels_unique,
                &index_train,
                &index_test,
            )?
        } else {
            self.fit_core_sparse(
                &adjacency,
                &labels_i32,
                &labels_contiguous,
                &labels_unique,
                &index_train,
                &index_test,
            )?
        };

        self.state.bipartite = Some(bipartite);
        self.state.labels = Some(Array1::from_vec(labels_pred));
        self.state.probs = Some(probs);
        if bipartite {
            self.state.split_vars(input_matrix.shape())?;
        } else {
            self.state.labels_row = self.state.labels.clone();
            self.state.labels_col = self.state.labels.clone();
            self.state.probs_row = self.state.probs.clone();
            self.state.probs_col = self.state.probs.clone();
        }
        Ok(())
    }

    /// Fits the classifier and returns predicted labels.
    ///
    /// # Errors
    /// Propagates all errors from [`Self::fit`] and [`Self::predict`].
    pub fn fit_predict(
        &mut self,
        input_matrix: &CsMat<f64>,
        labels: Option<ValuesInput>,
        labels_row: Option<ValuesInput>,
        labels_col: Option<ValuesInput>,
    ) -> Result<Array1<i32>, NNClassifierError> {
        self.fit(input_matrix, labels, labels_row, labels_col)?;
        self.predict(false)
    }

    /// Fits the classifier and returns class probabilities.
    ///
    /// # Errors
    /// Propagates all errors from [`Self::fit`] and [`Self::predict_proba`].
    pub fn fit_predict_proba(
        &mut self,
        input_matrix: &CsMat<f64>,
        labels: Option<ValuesInput>,
        labels_row: Option<ValuesInput>,
        labels_col: Option<ValuesInput>,
    ) -> Result<CsMat<f64>, NNClassifierError> {
        self.fit(input_matrix, labels, labels_row, labels_col)?;
        self.predict_proba(false)
    }

    /// Fits the classifier and returns class probabilities.
    ///
    /// # Errors
    /// Propagates all errors from [`Self::fit`] and [`Self::transform`].
    pub fn fit_transform(
        &mut self,
        input_matrix: &CsMat<f64>,
        labels: Option<ValuesInput>,
        labels_row: Option<ValuesInput>,
        labels_col: Option<ValuesInput>,
    ) -> Result<CsMat<f64>, NNClassifierError> {
        self.fit(input_matrix, labels, labels_row, labels_col)?;
        self.transform(false)
    }

    /// Returns fitted labels for rows or columns.
    ///
    /// # Errors
    /// Returns [`NNClassifierError::Base`] when the estimator is not fitted.
    pub fn predict(&self, columns: bool) -> Result<Array1<i32>, NNClassifierError> {
        Ok(self.state.predict(columns)?)
    }
    /// Returns fitted class probabilities for rows or columns.
    ///
    /// # Errors
    /// Returns [`NNClassifierError::Base`] when probabilities are unavailable.
    pub fn predict_proba(&self, columns: bool) -> Result<CsMat<f64>, NNClassifierError> {
        Ok(self.state.predict_proba(columns)?)
    }
    /// Returns fitted class probabilities (alias for [`Self::predict_proba`]).
    ///
    /// # Errors
    /// Returns [`NNClassifierError::Base`] when probabilities are unavailable.
    pub fn transform(&self, columns: bool) -> Result<CsMat<f64>, NNClassifierError> {
        Ok(self.state.transform(columns)?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;
    use crate::data::test_graphs::{test_bigraph, test_digraph, test_graph};

    #[test]
    fn test_classification() {
        for adjacency in [test_graph(), test_digraph(), test_bigraph()] {
            let mut labels = HashMap::new();
            labels.insert(0usize, 0.0);
            labels.insert(1usize, 1.0);

            let mut algo = NNClassifier::new(1, None, true);
            let labels_pred = algo
                .fit_predict(
                    &adjacency,
                    Some(ValuesInput::Map(labels.clone())),
                    None,
                    None,
                )
                .unwrap();
            let n_unique = labels_pred.iter().copied().collect::<HashSet<_>>().len();
            assert_eq!(n_unique, 2);

            let spectral = Spectral::new(2, "rw", -1.0, true);
            let mut algo = NNClassifier::new(1, Some(EmbeddingMethod::Spectral(spectral)), false);
            let labels_pred = algo
                .fit_predict(
                    &adjacency,
                    Some(ValuesInput::Map(labels.clone())),
                    None,
                    None,
                )
                .unwrap();
            let n_unique = labels_pred.iter().copied().collect::<HashSet<_>>().len();
            assert_eq!(n_unique, 2);
        }
    }
}
