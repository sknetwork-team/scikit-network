//! Shared clustering state and prediction helpers.
//!
//! Provides [`BaseClusteringState`] for storing labels, membership
//! probabilities, and bipartite split outputs across clustering estimators.

use ndarray::Array1;
use sprs::{CsMat, TriMat};

use crate::clustering::postprocess::aggregate_graph;
use crate::utils::membership::{MembershipError, get_membership};

/// Error type for shared clustering state operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaseClusteringError {
    /// The estimator has not been fitted yet.
    NotFitted,
    /// Bipartite column labels are missing from the fitted state.
    MissingBipartiteLabels,
    /// Label or matrix dimensions do not match the expected shape.
    InvalidShape,
    /// Membership matrix construction failed.
    Membership(MembershipError),
}

/// Fitted-state container shared by clustering estimators.
#[derive(Debug, Clone, Default)]
pub struct BaseClusteringState {
    /// Cluster labels for the primary node set (rows in unipartite graphs).
    pub labels: Option<Array1<i32>>,
    /// Cluster labels for bipartite row nodes.
    pub labels_row: Option<Array1<i32>>,
    /// Cluster labels for bipartite column nodes.
    pub labels_col: Option<Array1<i32>>,
    /// Row-normalized membership probabilities for the primary node set.
    pub probs: Option<CsMat<f64>>,
    /// Membership probabilities for bipartite row nodes.
    pub probs_row: Option<CsMat<f64>>,
    /// Membership probabilities for bipartite column nodes.
    pub probs_col: Option<CsMat<f64>>,
    /// Aggregated adjacency between clusters.
    pub aggregate: Option<CsMat<f64>>,
    /// Whether the last fit used a bipartite input.
    pub bipartite: Option<bool>,
}

fn sparse_matmul(a: &CsMat<f64>, b: &CsMat<f64>) -> CsMat<f64> {
    let (ar, ac) = a.shape();
    let (br, bc) = b.shape();
    if ac != br {
        return CsMat::zero((ar, bc));
    }

    let mut tri = TriMat::<f64>::new((ar, bc));
    for i in 0..ar {
        let Some(a_row) = a.outer_view(i) else {
            continue;
        };
        let mut acc = std::collections::HashMap::<usize, f64>::new();
        for (k, a_ik) in a_row.iter() {
            let Some(b_row) = b.outer_view(k) else {
                continue;
            };
            for (j, b_kj) in b_row.iter() {
                *acc.entry(j).or_insert(0.0) += a_ik * b_kj;
            }
        }
        for (j, v) in acc {
            if v != 0.0 {
                tri.add_triplet(i, j, v);
            }
        }
    }
    tri.to_csr::<usize>()
}

fn row_normalize(mat: &CsMat<f64>) -> CsMat<f64> {
    let (r, c) = mat.shape();
    let mut tri = TriMat::<f64>::new((r, c));
    for i in 0..r {
        if let Some(row) = mat.outer_view(i) {
            let s: f64 = row.data().iter().sum();
            if s > 0.0 {
                for (j, v) in row.iter() {
                    tri.add_triplet(i, j, v / s);
                }
            }
        }
    }
    tri.to_csr::<usize>()
}

impl BaseClusteringState {
    /// Resets all fitted attributes to their default empty state.
    pub fn init_vars(&mut self) {
        *self = Self::default();
    }

    /// Returns fitted cluster labels for rows or columns.
    ///
    /// # Errors
    /// Returns [`BaseClusteringError::NotFitted`] when labels are unavailable.
    pub fn predict(&self, columns: bool) -> Result<Array1<i32>, BaseClusteringError> {
        if columns {
            self.labels_col
                .clone()
                .ok_or(BaseClusteringError::NotFitted)
        } else {
            self.labels.clone().ok_or(BaseClusteringError::NotFitted)
        }
    }

    /// Returns fitted membership probabilities for rows or columns.
    ///
    /// # Errors
    /// Returns [`BaseClusteringError::NotFitted`] when probabilities are unavailable.
    pub fn predict_proba(&self, columns: bool) -> Result<CsMat<f64>, BaseClusteringError> {
        if columns {
            self.probs_col.clone().ok_or(BaseClusteringError::NotFitted)
        } else {
            self.probs.clone().ok_or(BaseClusteringError::NotFitted)
        }
    }

    /// Returns fitted membership probabilities (alias for [`Self::predict_proba`]).
    ///
    /// # Errors
    /// Returns [`BaseClusteringError::NotFitted`] when probabilities are unavailable.
    pub fn transform(&self, columns: bool) -> Result<CsMat<f64>, BaseClusteringError> {
        self.predict_proba(columns)
    }

    /// Splits stacked bipartite labels into row and column views.
    ///
    /// # Arguments
    /// - `shape`: `(n_row, n_col)` dimensions of the original bipartite matrix.
    ///
    /// # Errors
    /// Returns [`BaseClusteringError::NotFitted`] when labels are missing, or
    /// [`BaseClusteringError::InvalidShape`] when label length mismatches `shape`.
    pub fn split_vars(&mut self, shape: (usize, usize)) -> Result<(), BaseClusteringError> {
        let labels = self.labels.clone().ok_or(BaseClusteringError::NotFitted)?;
        let n_row = shape.0;
        let n_col = shape.1;
        if labels.len() != n_row + n_col {
            return Err(BaseClusteringError::InvalidShape);
        }
        self.labels_row = Some(labels.slice(ndarray::s![..n_row]).to_owned());
        self.labels_col = Some(labels.slice(ndarray::s![n_row..]).to_owned());
        self.labels = self.labels_row.clone();
        Ok(())
    }

    /// Computes optional membership probabilities and aggregate graph outputs.
    ///
    /// # Arguments
    /// - `input_matrix`: Original adjacency or biadjacency matrix passed to `fit`.
    /// - `return_probs`: Whether to populate membership probability matrices.
    /// - `return_aggregate`: Whether to populate the cluster aggregate matrix.
    ///
    /// # Errors
    /// Returns [`BaseClusteringError::NotFitted`] when labels are missing,
    /// [`BaseClusteringError::MissingBipartiteLabels`] for incomplete bipartite
    /// state, [`BaseClusteringError::InvalidShape`] on aggregate failures, or
    /// [`BaseClusteringError::Membership`] when membership construction fails.
    pub fn secondary_outputs(
        &mut self,
        input_matrix: &CsMat<f64>,
        return_probs: bool,
        return_aggregate: bool,
    ) -> Result<(), BaseClusteringError> {
        if !(return_probs || return_aggregate) {
            return Ok(());
        }
        let bipartite = self.bipartite.unwrap_or(false);

        if !bipartite {
            let labels = self.labels.clone().ok_or(BaseClusteringError::NotFitted)?;
            let probs = get_membership(&labels, None).map_err(BaseClusteringError::Membership)?;
            if return_probs {
                let p = sparse_matmul(input_matrix, &probs);
                self.probs = Some(row_normalize(&p));
            }
            if return_aggregate {
                let agg = aggregate_graph(input_matrix, Some(&labels), None, None)
                    .map_err(|_| BaseClusteringError::InvalidShape)?;
                self.aggregate = Some(agg);
            }
            return Ok(());
        }

        let labels_row = self.labels_row.clone().or_else(|| self.labels.clone());
        let labels_row = labels_row.ok_or(BaseClusteringError::NotFitted)?;
        let labels_col = self
            .labels_col
            .clone()
            .ok_or(BaseClusteringError::MissingBipartiteLabels)?;

        let n_labels = labels_row
            .iter()
            .chain(labels_col.iter())
            .copied()
            .filter(|x| *x >= 0)
            .max()
            .map(|x| x as usize + 1)
            .unwrap_or(0);
        let probs_row =
            get_membership(&labels_row, Some(n_labels)).map_err(BaseClusteringError::Membership)?;
        let probs_col =
            get_membership(&labels_col, Some(n_labels)).map_err(BaseClusteringError::Membership)?;

        if return_probs {
            let p_row = sparse_matmul(input_matrix, &probs_col);
            let p_col = sparse_matmul(&input_matrix.transpose_view().to_csr(), &probs_row);
            self.probs_row = Some(row_normalize(&p_row));
            self.probs_col = Some(row_normalize(&p_col));
            self.probs = self.probs_row.clone();
        }
        if return_aggregate {
            let left = sparse_matmul(&probs_row.transpose_view().to_csr(), input_matrix);
            let aggregate = sparse_matmul(&left, &probs_col);
            self.aggregate = Some(aggregate);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use sprs::TriMat;

    use super::*;
    use crate::data::test_graphs::test_graph;

    fn star_wars() -> CsMat<f64> {
        let mut tri = TriMat::<f64>::new((4, 3));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(0, 2, 1.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(2, 0, 1.0);
        tri.add_triplet(2, 1, 1.0);
        tri.add_triplet(2, 2, 1.0);
        tri.add_triplet(3, 1, 1.0);
        tri.add_triplet(3, 2, 1.0);
        tri.to_csr::<usize>()
    }

    #[test]
    fn test_split_and_predict() {
        let mut state = BaseClusteringState {
            labels: Some(Array1::from_vec(vec![0, 0, 1, 1, 0, 1, 0])),
            ..Default::default()
        };
        state.split_vars((4, 3)).unwrap();
        assert_eq!(state.predict(false).unwrap().to_vec(), vec![0, 0, 1, 1]);
        assert_eq!(state.predict(true).unwrap().to_vec(), vec![0, 1, 0]);
    }

    #[test]
    fn test_secondary_outputs_graph() {
        let adjacency = test_graph();
        let mut state = BaseClusteringState {
            labels: Some(Array1::from_vec(vec![0, 0, 1, 1, 0, 1, 0, 1, 1, 0])),
            bipartite: Some(false),
            ..Default::default()
        };
        state.secondary_outputs(&adjacency, true, true).unwrap();
        let probs = state.probs.clone().unwrap_or_else(|| CsMat::zero((0, 0)));
        let agg = state
            .aggregate
            .clone()
            .unwrap_or_else(|| CsMat::zero((0, 0)));
        assert_eq!(probs.rows(), adjacency.rows());
        assert_eq!(agg.shape(), (2, 2));
    }

    #[test]
    fn test_secondary_outputs_bipartite() {
        let biadjacency = star_wars();
        let mut state = BaseClusteringState {
            labels_row: Some(Array1::from_vec(vec![0, 0, 1, 1])),
            labels_col: Some(Array1::from_vec(vec![0, 1, 0])),
            labels: Some(Array1::from_vec(vec![0, 0, 1, 1])),
            bipartite: Some(true),
            ..Default::default()
        };
        state.secondary_outputs(&biadjacency, true, true).unwrap();
        assert_eq!(
            state
                .probs_row
                .clone()
                .unwrap_or_else(|| CsMat::zero((0, 0)))
                .rows(),
            4
        );
        assert_eq!(
            state
                .probs_col
                .clone()
                .unwrap_or_else(|| CsMat::zero((0, 0)))
                .rows(),
            3
        );
        assert_eq!(
            state
                .aggregate
                .clone()
                .unwrap_or_else(|| CsMat::zero((0, 0)))
                .shape(),
            (2, 2)
        );
    }
}
