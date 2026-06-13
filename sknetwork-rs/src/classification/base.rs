//! Shared classifier state and prediction helpers.
//!
//! Provides [`BaseClassifierState`] for storing labels and class probabilities
//! across supervised graph classifiers.

use ndarray::Array1;
use sprs::CsMat;

/// Error type for shared classifier state operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaseClassifierError {
    /// The estimator has not been fitted yet.
    NotFitted,
    /// Label or probability dimensions do not match the expected shape.
    InvalidShape,
}

/// Fitted-state container shared by graph classifiers.
#[derive(Debug, Clone, Default)]
pub struct BaseClassifierState {
    /// Whether the last fit used a bipartite input.
    pub bipartite: Option<bool>,
    /// Predicted labels for the primary node set.
    pub labels: Option<Array1<i32>>,
    /// Row-normalized class probabilities for the primary node set.
    pub probs: Option<CsMat<f64>>,
    /// Predicted labels for bipartite row nodes.
    pub labels_row: Option<Array1<i32>>,
    /// Predicted labels for bipartite column nodes.
    pub labels_col: Option<Array1<i32>>,
    /// Class probabilities for bipartite row nodes.
    pub probs_row: Option<CsMat<f64>>,
    /// Class probabilities for bipartite column nodes.
    pub probs_col: Option<CsMat<f64>>,
}

impl BaseClassifierState {
    /// Returns fitted labels for rows or columns.
    ///
    /// # Errors
    /// Returns [`BaseClassifierError::NotFitted`] when labels are unavailable.
    pub fn predict(&self, columns: bool) -> Result<Array1<i32>, BaseClassifierError> {
        if columns {
            self.labels_col
                .clone()
                .ok_or(BaseClassifierError::NotFitted)
        } else {
            self.labels.clone().ok_or(BaseClassifierError::NotFitted)
        }
    }

    /// Returns fitted class probabilities for rows or columns.
    ///
    /// # Errors
    /// Returns [`BaseClassifierError::NotFitted`] when probabilities are unavailable.
    pub fn predict_proba(&self, columns: bool) -> Result<CsMat<f64>, BaseClassifierError> {
        if columns {
            self.probs_col.clone().ok_or(BaseClassifierError::NotFitted)
        } else {
            self.probs.clone().ok_or(BaseClassifierError::NotFitted)
        }
    }

    /// Returns fitted class probabilities (alias for [`Self::predict_proba`]).
    ///
    /// # Errors
    /// Returns [`BaseClassifierError::NotFitted`] when probabilities are unavailable.
    pub fn transform(&self, columns: bool) -> Result<CsMat<f64>, BaseClassifierError> {
        self.predict_proba(columns)
    }

    /// Splits stacked bipartite labels and probabilities into row and column views.
    ///
    /// # Arguments
    /// - `shape`: `(n_row, n_col)` dimensions of the original bipartite matrix.
    ///
    /// # Errors
    /// Returns [`BaseClassifierError::NotFitted`] when state is missing, or
    /// [`BaseClassifierError::InvalidShape`] when lengths mismatch `shape`.
    pub fn split_vars(&mut self, shape: (usize, usize)) -> Result<(), BaseClassifierError> {
        let labels = self.labels.clone().ok_or(BaseClassifierError::NotFitted)?;
        let probs = self.probs.clone().ok_or(BaseClassifierError::NotFitted)?;
        let n_row = shape.0;
        let n_col = shape.1;
        if labels.len() != n_row + n_col || probs.rows() != n_row + n_col {
            return Err(BaseClassifierError::InvalidShape);
        }
        self.labels_row = Some(labels.slice(ndarray::s![..n_row]).to_owned());
        self.labels_col = Some(labels.slice(ndarray::s![n_row..]).to_owned());
        self.probs_row = Some(probs.slice_outer(0..n_row).to_owned());
        self.probs_col = Some(probs.slice_outer(n_row..(n_row + n_col)).to_owned());
        self.labels = self.labels_row.clone();
        self.probs = self.probs_row.clone();
        Ok(())
    }
}
