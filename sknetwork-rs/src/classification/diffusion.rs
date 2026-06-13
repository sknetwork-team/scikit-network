//! Diffusion-based semi-supervised graph classifier.

use ndarray::Array1;
use sprs::{CsMat, TriMat};

use crate::classification::base::{BaseClassifierError, BaseClassifierState};
use crate::path::distances::get_distances_multi;
use crate::utils::check::CheckError;
use crate::utils::format::{MatrixInput, get_adjacency_values};
use crate::utils::membership::{MembershipError, get_membership};
use crate::utils::values::ValuesInput;

/// Error type for [`DiffusionClassifier`] fitting and prediction.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffusionClassifierError {
    /// Input matrix or seed formatting failed.
    Check(CheckError),
    /// Membership matrix construction failed.
    Membership(MembershipError),
    /// Wrapped shared classifier state error.
    Base(BaseClassifierError),
    /// `n_iter` must be strictly positive.
    InvalidNIter,
    /// No seed labels with non-negative values were provided.
    NoSeedLabels,
    /// Shortest-path distance computation failed.
    DistanceComputationFailed,
}

impl From<CheckError> for DiffusionClassifierError {
    fn from(value: CheckError) -> Self {
        Self::Check(value)
    }
}

impl From<MembershipError> for DiffusionClassifierError {
    fn from(value: MembershipError) -> Self {
        Self::Membership(value)
    }
}

impl From<BaseClassifierError> for DiffusionClassifierError {
    fn from(value: BaseClassifierError) -> Self {
        Self::Base(value)
    }
}

/// Diffusion classifier using heat-kernel style label propagation.
#[derive(Debug, Clone)]
pub struct DiffusionClassifier {
    /// Number of diffusion iterations.
    pub n_iter: usize,
    /// Whether to center class scores before softmax normalization.
    pub centering: bool,
    /// Temperature scale used when `centering` is enabled.
    pub scale: f64,
    /// Shared fitted-state container.
    pub state: BaseClassifierState,
}

impl Default for DiffusionClassifier {
    fn default() -> Self {
        Self::new(10, true, 5.0).unwrap_or_else(|_| Self {
            n_iter: 10,
            centering: true,
            scale: 5.0,
            state: BaseClassifierState::default(),
        })
    }
}

impl DiffusionClassifier {
    /// Creates a diffusion classifier with explicit hyperparameters.
    ///
    /// # Errors
    /// Returns [`DiffusionClassifierError::InvalidNIter`] when `n_iter <= 0`.
    pub fn new(
        n_iter: isize,
        centering: bool,
        scale: f64,
    ) -> Result<Self, DiffusionClassifierError> {
        if n_iter <= 0 {
            return Err(DiffusionClassifierError::InvalidNIter);
        }
        Ok(Self {
            n_iter: n_iter as usize,
            centering,
            scale,
            state: BaseClassifierState::default(),
        })
    }

    fn row_normalized(adjacency: &CsMat<f64>) -> CsMat<f64> {
        let mut tri = TriMat::<f64>::new(adjacency.shape());
        for (i, row) in adjacency.outer_iterator().enumerate() {
            let sum: f64 = row.data().iter().sum();
            if sum > 0.0 {
                for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                    tri.add_triplet(i, j, v / sum);
                }
            }
        }
        tri.to_csr::<usize>()
    }

    fn sparse_dot_dense(adjacency: &CsMat<f64>, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = adjacency.rows();
        let k = x.first().map(|r| r.len()).unwrap_or(0);
        let mut out = vec![vec![0.0; k]; n];
        for (i, row) in adjacency.outer_iterator().enumerate() {
            for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                for c in 0..k {
                    out[i][c] += v * x[j][c];
                }
            }
        }
        out
    }

    fn row_softmax_scale(x: &mut [Vec<f64>], scale: f64) {
        for row in x.iter_mut() {
            let mut max_v = f64::NEG_INFINITY;
            for &v in row.iter() {
                if v > max_v {
                    max_v = v;
                }
            }
            let mut sum = 0.0;
            for v in row.iter_mut() {
                *v = ((*v - max_v) * scale).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }
    }

    /// Fits the classifier from sparse seed labels.
    ///
    /// # Errors
    /// Returns [`DiffusionClassifierError::Check`] for formatting failures,
    /// [`DiffusionClassifierError::NoSeedLabels`] when no seeds are provided,
    /// [`DiffusionClassifierError::Membership`] for membership errors,
    /// [`DiffusionClassifierError::DistanceComputationFailed`] when distances fail,
    /// and [`DiffusionClassifierError::Base`] for bipartite split errors.
    pub fn fit(
        &mut self,
        input_matrix: &CsMat<f64>,
        labels: Option<ValuesInput>,
        labels_row: Option<ValuesInput>,
        labels_col: Option<ValuesInput>,
        force_bipartite: bool,
    ) -> Result<(), DiffusionClassifierError> {
        let (adjacency, seeds, bipartite) = get_adjacency_values(
            MatrixInput::Sparse(input_matrix.clone()),
            true,
            force_bipartite,
            false,
            labels,
            labels_row,
            labels_col,
            -1.0,
            None,
        )?;
        let labels_raw: Vec<i32> = seeds.iter().map(|x| *x as i32).collect();
        let mut labels_unique: Vec<i32> = labels_raw.iter().copied().filter(|x| *x >= 0).collect();
        labels_unique.sort_unstable();
        labels_unique.dedup();
        if labels_unique.is_empty() {
            return Err(DiffusionClassifierError::NoSeedLabels);
        }

        let mut remap = std::collections::HashMap::<i32, i32>::new();
        for (i, label) in labels_unique.iter().enumerate() {
            remap.insert(*label, i as i32);
        }
        let labels_reindexed = Array1::from_vec(
            labels_raw
                .iter()
                .map(|l| {
                    if *l >= 0 {
                        *remap.get(l).unwrap_or(&-1)
                    } else {
                        -1
                    }
                })
                .collect(),
        );

        let membership = get_membership(&labels_reindexed, None)?;
        let k = membership.cols();
        let n = adjacency.rows();
        let seed_mask: Vec<bool> = labels_raw.iter().map(|x| *x >= 0).collect();
        let seed_rows: Vec<usize> = seed_mask
            .iter()
            .enumerate()
            .filter_map(|(i, b)| if *b { Some(i) } else { None })
            .collect();

        let mut temperatures = vec![vec![0.5; k]; n];
        for i in 0..n {
            if seed_mask[i] {
                temperatures[i] = vec![0.0; k];
                if let Some(row) = membership.outer_view(i) {
                    for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                        temperatures[i][j] = v;
                    }
                }
            }
        }
        let temperatures_seeds: Vec<Vec<f64>> =
            seed_rows.iter().map(|&i| temperatures[i].clone()).collect();

        let diffusion = Self::row_normalized(&adjacency);
        for _ in 0..self.n_iter {
            temperatures = Self::sparse_dot_dense(&diffusion, &temperatures);
            for (m, &i) in seed_rows.iter().enumerate() {
                temperatures[i] = temperatures_seeds[m].clone();
            }
        }

        if self.centering {
            for c in 0..k {
                let mean = temperatures.iter().map(|r| r[c]).sum::<f64>() / n as f64;
                for row in &mut temperatures {
                    row[c] -= mean;
                }
            }
        }

        let distances = get_distances_multi(&adjacency, &seed_rows)
            .map_err(|_| DiffusionClassifierError::DistanceComputationFailed)?;
        for i in 0..n {
            if distances[i] < 0 {
                temperatures[i].fill(0.0);
            }
        }

        let mut labels_pred = vec![-1; n];
        for i in 0..n {
            if distances[i] >= 0 {
                let (argmax, _) = temperatures[i]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, &0.0));
                labels_pred[i] = labels_unique[argmax];
            }
        }

        if self.centering {
            Self::row_softmax_scale(&mut temperatures, self.scale);
        } else {
            for row in &mut temperatures {
                let s: f64 = row.iter().sum();
                if s > 0.0 {
                    for v in row {
                        *v /= s;
                    }
                }
            }
        }

        let mut tri = TriMat::<f64>::new((
            n,
            labels_unique
                .iter()
                .max()
                .map(|x| (*x as usize) + 1)
                .unwrap_or(0),
        ));
        for (i, row) in temperatures.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v > 0.0 {
                    tri.add_triplet(i, labels_unique[j] as usize, v);
                }
            }
        }

        self.state.bipartite = Some(bipartite);
        self.state.labels = Some(Array1::from_vec(labels_pred));
        self.state.probs = Some(tri.to_csr::<usize>());
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
        force_bipartite: bool,
    ) -> Result<Array1<i32>, DiffusionClassifierError> {
        self.fit(
            input_matrix,
            labels,
            labels_row,
            labels_col,
            force_bipartite,
        )?;
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
        force_bipartite: bool,
    ) -> Result<CsMat<f64>, DiffusionClassifierError> {
        self.fit(
            input_matrix,
            labels,
            labels_row,
            labels_col,
            force_bipartite,
        )?;
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
        force_bipartite: bool,
    ) -> Result<CsMat<f64>, DiffusionClassifierError> {
        self.fit(
            input_matrix,
            labels,
            labels_row,
            labels_col,
            force_bipartite,
        )?;
        self.transform(false)
    }

    /// Returns fitted labels for rows or columns.
    ///
    /// # Errors
    /// Returns [`DiffusionClassifierError::Base`] when the estimator is not fitted.
    pub fn predict(&self, columns: bool) -> Result<Array1<i32>, DiffusionClassifierError> {
        Ok(self.state.predict(columns)?)
    }

    /// Returns fitted class probabilities for rows or columns.
    ///
    /// # Errors
    /// Returns [`DiffusionClassifierError::Base`] when probabilities are unavailable.
    pub fn predict_proba(&self, columns: bool) -> Result<CsMat<f64>, DiffusionClassifierError> {
        Ok(self.state.predict_proba(columns)?)
    }

    /// Returns fitted class probabilities (alias for [`Self::predict_proba`]).
    ///
    /// # Errors
    /// Returns [`DiffusionClassifierError::Base`] when probabilities are unavailable.
    pub fn transform(&self, columns: bool) -> Result<CsMat<f64>, DiffusionClassifierError> {
        Ok(self.state.transform(columns)?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::data::test_graphs::{test_bigraph, test_digraph, test_graph};

    #[test]
    fn test_diffusion_graph() {
        let adjacency = test_graph();
        let n_nodes = adjacency.rows();
        let mut labels = HashMap::new();
        labels.insert(0usize, 0.0);
        labels.insert(1usize, 1.0);

        let mut algo = DiffusionClassifier::default();
        algo.fit(
            &adjacency,
            Some(ValuesInput::Map(labels.clone())),
            None,
            None,
            false,
        )
        .unwrap();
        assert_eq!(algo.state.labels.as_ref().unwrap().len(), n_nodes);

        let adjacency = test_digraph();
        let mut algo = DiffusionClassifier::new(10, false, 5.0).unwrap();
        algo.fit(
            &adjacency,
            Some(ValuesInput::Map(labels.clone())),
            None,
            None,
            false,
        )
        .unwrap();
        assert_eq!(algo.state.labels.as_ref().unwrap().len(), n_nodes);

        assert!(matches!(
            DiffusionClassifier::new(0, true, 5.0),
            Err(DiffusionClassifierError::InvalidNIter)
        ));
    }

    #[test]
    fn test_bipartite_predict_api() {
        let biadjacency = test_bigraph();
        let (n_row, n_col) = biadjacency.shape();
        let mut labels_row = HashMap::new();
        labels_row.insert(0usize, 0.0);
        labels_row.insert(1usize, 1.0);

        let mut algo = DiffusionClassifier::default();
        let labels_pred = algo
            .fit_predict(
                &biadjacency,
                None,
                Some(ValuesInput::Map(labels_row.clone())),
                None,
                false,
            )
            .unwrap();
        assert_eq!(labels_pred.len(), n_row);
        assert_eq!(algo.predict(true).unwrap().len(), n_col);
        assert_eq!(
            algo.fit_predict_proba(
                &biadjacency,
                None,
                Some(ValuesInput::Map(labels_row)),
                None,
                false
            )
            .unwrap()
            .rows(),
            n_row
        );
    }
}
