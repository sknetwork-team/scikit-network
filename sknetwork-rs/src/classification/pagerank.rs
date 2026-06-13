//! PageRank-based semi-supervised graph classifier.

use ndarray::Array1;
use sprs::{CsMat, TriMat};

use crate::classification::base::{BaseClassifierError, BaseClassifierState};
use crate::ranking::pagerank::{PageRank, PageRankError};
use crate::utils::check::CheckError;
use crate::utils::format::{MatrixInput, get_adjacency_values};
use crate::utils::values::ValuesInput;

/// Error type for [`PageRankClassifier`] fitting and prediction.
#[derive(Debug, Clone, PartialEq)]
pub enum PageRankClassifierError {
    /// Input matrix or seed formatting failed.
    Check(CheckError),
    /// Underlying PageRank computation failed.
    PageRank(PageRankError),
    /// Wrapped shared classifier state error.
    Base(BaseClassifierError),
    /// No seed labels with non-negative values were provided.
    EmptyLabels,
}

impl From<CheckError> for PageRankClassifierError {
    fn from(value: CheckError) -> Self {
        Self::Check(value)
    }
}

impl From<PageRankError> for PageRankClassifierError {
    fn from(value: PageRankError) -> Self {
        Self::PageRank(value)
    }
}

impl From<BaseClassifierError> for PageRankClassifierError {
    fn from(value: BaseClassifierError) -> Self {
        Self::Base(value)
    }
}

/// PageRank classifier using one restart vector per class.
#[derive(Debug, Clone)]
pub struct PageRankClassifier {
    /// Underlying PageRank estimator configuration.
    pub algorithm: PageRank,
    /// Shared fitted-state container.
    pub state: BaseClassifierState,
}

impl Default for PageRankClassifier {
    fn default() -> Self {
        Self::new(0.85, 10, 0.0).unwrap_or_else(|_| Self {
            algorithm: PageRank::default(),
            state: BaseClassifierState::default(),
        })
    }
}

impl PageRankClassifier {
    /// Creates a PageRank classifier with explicit convergence settings.
    ///
    /// # Errors
    /// Returns [`PageRankClassifierError::PageRank`] when PageRank parameters
    /// are invalid.
    pub fn new(
        damping_factor: f64,
        n_iter: usize,
        tol: f64,
    ) -> Result<Self, PageRankClassifierError> {
        Ok(Self {
            algorithm: PageRank::new(damping_factor, n_iter, tol)?,
            state: BaseClassifierState::default(),
        })
    }

    fn normalize_rows(
        scores_by_label: &[Vec<f64>],
        labels_unique: &[i32],
    ) -> (Array1<i32>, CsMat<f64>) {
        let n = scores_by_label.first().map(|v| v.len()).unwrap_or(0);
        let k = labels_unique
            .iter()
            .max()
            .map(|x| (*x as usize) + 1)
            .unwrap_or(0);
        let mut labels = vec![-1; n];
        let mut tri = TriMat::<f64>::new((n, k));

        for i in 0..n {
            let mut row_sum = 0.0;
            for scores in scores_by_label {
                row_sum += scores[i];
            }
            let mut best_idx = 0usize;
            let mut best_val = f64::NEG_INFINITY;
            for (j, scores) in scores_by_label.iter().enumerate() {
                let v = if row_sum > 0.0 {
                    scores[i] / row_sum
                } else {
                    0.0
                };
                if v > 0.0 {
                    tri.add_triplet(i, labels_unique[j] as usize, v);
                }
                if v > best_val {
                    best_val = v;
                    best_idx = j;
                }
            }
            if !labels_unique.is_empty() {
                labels[i] = labels_unique[best_idx];
            }
        }
        (Array1::from_vec(labels), tri.to_csr::<usize>())
    }

    /// Fits the classifier from sparse seed labels.
    ///
    /// # Errors
    /// Returns [`PageRankClassifierError::Check`] for formatting failures,
    /// [`PageRankClassifierError::EmptyLabels`] when no seeds are provided,
    /// [`PageRankClassifierError::PageRank`] for ranking failures, and
    /// [`PageRankClassifierError::Base`] for bipartite split errors.
    pub fn fit(
        &mut self,
        input_matrix: &CsMat<f64>,
        labels: Option<ValuesInput>,
        labels_row: Option<ValuesInput>,
        labels_col: Option<ValuesInput>,
    ) -> Result<(), PageRankClassifierError> {
        let (adjacency, seeds, bipartite) = get_adjacency_values(
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
        let seeds_i32: Vec<i32> = seeds.into_iter().map(|x| x as i32).collect();
        let mut labels_unique: Vec<i32> = seeds_i32.iter().copied().filter(|x| *x >= 0).collect();
        labels_unique.sort_unstable();
        labels_unique.dedup();
        if labels_unique.is_empty() {
            return Err(PageRankClassifierError::EmptyLabels);
        }

        let mut scores_by_label = Vec::<Vec<f64>>::new();
        for label in &labels_unique {
            let binary: Vec<f64> = seeds_i32
                .iter()
                .map(|x| if x == label { 1.0 } else { 0.0 })
                .collect();
            let mut algo = self.algorithm.clone();
            let scores = algo.fit_predict(
                &adjacency,
                Some(ValuesInput::Vector(binary)),
                None,
                None,
                false,
            )?;
            scores_by_label.push(scores);
        }

        let (labels_pred, probs) = Self::normalize_rows(&scores_by_label, &labels_unique);
        self.state.bipartite = Some(bipartite);
        self.state.labels = Some(labels_pred);
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
    ) -> Result<Array1<i32>, PageRankClassifierError> {
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
    ) -> Result<CsMat<f64>, PageRankClassifierError> {
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
    ) -> Result<CsMat<f64>, PageRankClassifierError> {
        self.fit(input_matrix, labels, labels_row, labels_col)?;
        self.transform(false)
    }

    /// Returns fitted labels for rows or columns.
    ///
    /// # Errors
    /// Returns [`PageRankClassifierError::Base`] when the estimator is not fitted.
    pub fn predict(&self, columns: bool) -> Result<Array1<i32>, PageRankClassifierError> {
        Ok(self.state.predict(columns)?)
    }

    /// Returns fitted class probabilities for rows or columns.
    ///
    /// # Errors
    /// Returns [`PageRankClassifierError::Base`] when probabilities are unavailable.
    pub fn predict_proba(&self, columns: bool) -> Result<CsMat<f64>, PageRankClassifierError> {
        Ok(self.state.predict_proba(columns)?)
    }

    /// Returns fitted class probabilities (alias for [`Self::predict_proba`]).
    ///
    /// # Errors
    /// Returns [`PageRankClassifierError::Base`] when probabilities are unavailable.
    pub fn transform(&self, columns: bool) -> Result<CsMat<f64>, PageRankClassifierError> {
        Ok(self.state.transform(columns)?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::data::test_graphs::test_graph;

    #[test]
    fn test_classifier_runs() {
        let adjacency = test_graph();
        let mut labels = HashMap::new();
        labels.insert(0usize, 0.0);
        labels.insert(1usize, 1.0);
        let mut clf = PageRankClassifier::default();
        let pred = clf
            .fit_predict(&adjacency, Some(ValuesInput::Map(labels)), None, None)
            .unwrap();
        assert_eq!(pred.len(), adjacency.rows());
    }
}
