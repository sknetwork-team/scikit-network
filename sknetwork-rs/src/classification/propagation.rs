//! Label-propagation semi-supervised graph classifier.

use ndarray::Array1;
use sprs::{CsMat, TriMat};

use crate::classification::base::{BaseClassifierError, BaseClassifierState};
use crate::utils::check::CheckError;
use crate::utils::format::{MatrixInput, get_adjacency_values};
use crate::utils::membership::{MembershipError, get_membership};
use crate::utils::values::ValuesInput;

/// Error type for [`Propagation`] fitting and prediction.
#[derive(Debug, Clone, PartialEq)]
pub enum PropagationError {
    /// Input matrix or seed formatting failed.
    Check(CheckError),
    /// Membership matrix construction failed.
    Membership(MembershipError),
    /// Wrapped shared classifier state error.
    Base(BaseClassifierError),
    /// Node-order option is not supported.
    UnknownNodeOrder,
}

impl From<CheckError> for PropagationError {
    fn from(value: CheckError) -> Self {
        Self::Check(value)
    }
}

impl From<MembershipError> for PropagationError {
    fn from(value: MembershipError) -> Self {
        Self::Membership(value)
    }
}

impl From<BaseClassifierError> for PropagationError {
    fn from(value: BaseClassifierError) -> Self {
        Self::Base(value)
    }
}

/// Node processing order for label-propagation updates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeOrder {
    /// Process nodes in index order.
    Index,
    /// Process nodes in a deterministic pseudo-random order.
    Random,
    /// Process low in-degree nodes first.
    Increasing,
    /// Process high in-degree nodes first.
    Decreasing,
    /// Placeholder for unsupported order strings.
    Unknown,
}

/// Label-propagation classifier for sparse seed labels.
#[derive(Debug, Clone)]
pub struct Propagation {
    /// Maximum number of propagation passes.
    pub n_iter: usize,
    /// Node processing order policy.
    pub node_order: NodeOrder,
    /// Whether neighbor votes use edge weights.
    pub weighted: bool,
    /// Shared fitted-state container.
    pub state: BaseClassifierState,
}

impl Default for Propagation {
    fn default() -> Self {
        Self::new(-1, None, true)
    }
}

impl Propagation {
    /// Creates a label-propagation classifier.
    ///
    /// # Arguments
    /// - `n_iter`: Number of update passes (`<0` means iterate until stable).
    /// - `node_order`: Optional node-order policy (`index`, `random`,
    ///   `increasing`, `decreasing`).
    /// - `weighted`: Whether to use edge weights in label voting.
    pub fn new(n_iter: i32, node_order: Option<&str>, weighted: bool) -> Self {
        let n_iter = if n_iter < 0 {
            usize::MAX
        } else {
            n_iter as usize
        };
        let node_order = match node_order.map(str::to_lowercase).as_deref() {
            Some("random") => NodeOrder::Random,
            Some("increasing") => NodeOrder::Increasing,
            Some("decreasing") => NodeOrder::Decreasing,
            Some("index") | None => NodeOrder::Index,
            Some(_) => NodeOrder::Unknown,
        };
        Self {
            n_iter,
            node_order,
            weighted,
            state: BaseClassifierState::default(),
        }
    }

    fn instantiate_vars(labels: &[i32]) -> (Vec<usize>, Vec<usize>, Vec<i32>) {
        let mut index_seed = Vec::new();
        let mut index_remain = Vec::new();
        let mut labels_seed = Vec::new();
        for (i, &label) in labels.iter().enumerate() {
            if label >= 0 {
                index_seed.push(i);
                labels_seed.push(label);
            } else {
                index_remain.push(i);
            }
        }
        (index_seed, index_remain, labels_seed)
    }

    fn in_weights(adjacency: &CsMat<f64>) -> Vec<f64> {
        let mut weights = vec![0.0; adjacency.cols()];
        for row in adjacency.outer_iterator() {
            for (&col, &w) in row.indices().iter().zip(row.data().iter()) {
                weights[col] += w;
            }
        }
        weights
    }

    fn deterministic_shuffle(indices: &mut [usize]) {
        // Deterministic Fisher-Yates to preserve reproducibility without external RNG dependency.
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in (1..indices.len()).rev() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (seed as usize) % (i + 1);
            indices.swap(i, j);
        }
    }

    fn update_labels(
        adjacency: &CsMat<f64>,
        labels: &mut [i32],
        index_remain: &[usize],
        weighted: bool,
    ) {
        for &i in index_remain {
            let Some(row) = adjacency.outer_view(i) else {
                continue;
            };
            let mut scores = std::collections::HashMap::<i32, f64>::new();
            for (&j, &w) in row.indices().iter().zip(row.data().iter()) {
                let label = labels[j];
                if label >= 0 {
                    let inc = if weighted { w } else { 1.0 };
                    *scores.entry(label).or_insert(0.0) += inc;
                }
            }
            if let Some((&best_label, _)) = scores.iter().max_by(|a, b| {
                a.1.partial_cmp(b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(b.0).reverse())
            }) {
                labels[i] = best_label;
            }
        }
    }

    fn normalize_rows(mat: &CsMat<f64>) -> CsMat<f64> {
        let mut tri = TriMat::<f64>::new((mat.rows(), mat.cols()));
        for (i, row) in mat.outer_iterator().enumerate() {
            let sum: f64 = row.data().iter().sum();
            if sum > 0.0 {
                for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                    tri.add_triplet(i, j, v / sum);
                }
            }
        }
        tri.to_csr::<usize>()
    }

    /// Fits the classifier from sparse seeds.
    ///
    /// # Errors
    /// Returns typed errors for formatting issues, membership errors, unknown
    /// node-order options, and not-fitted state splits.
    pub fn fit(
        &mut self,
        input_matrix: &CsMat<f64>,
        labels: Option<ValuesInput>,
        labels_row: Option<ValuesInput>,
        labels_col: Option<ValuesInput>,
    ) -> Result<(), PropagationError> {
        let (adjacency, seeds, bipartite) = get_adjacency_values(
            MatrixInput::Sparse(input_matrix.clone()),
            true,
            false,
            false,
            labels,
            labels_row,
            labels_col,
            -1.0,
            Some("labels"),
        )?;
        self.state.bipartite = Some(bipartite);
        let seeds_i32: Vec<i32> = seeds.iter().map(|x| *x as i32).collect();
        let n = adjacency.rows();
        let (index_seed, mut index_remain, labels_seed) = Self::instantiate_vars(&seeds_i32);

        match self.node_order {
            NodeOrder::Random => Self::deterministic_shuffle(&mut index_remain),
            NodeOrder::Increasing => {
                let mut idx: Vec<usize> = (0..n).collect();
                let in_w = Self::in_weights(&adjacency);
                idx.sort_by(|&a, &b| {
                    in_w[a]
                        .partial_cmp(&in_w[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let rank: std::collections::HashMap<usize, usize> =
                    idx.iter().enumerate().map(|(r, &v)| (v, r)).collect();
                index_remain.sort_by_key(|i| *rank.get(i).unwrap_or(&usize::MAX));
            }
            NodeOrder::Decreasing => {
                let mut idx: Vec<usize> = (0..n).collect();
                let in_w = Self::in_weights(&adjacency);
                idx.sort_by(|&a, &b| {
                    in_w[b]
                        .partial_cmp(&in_w[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let rank: std::collections::HashMap<usize, usize> =
                    idx.iter().enumerate().map(|(r, &v)| (v, r)).collect();
                index_remain.sort_by_key(|i| *rank.get(i).unwrap_or(&usize::MAX));
            }
            NodeOrder::Index => {}
            NodeOrder::Unknown => return Err(PropagationError::UnknownNodeOrder),
        }

        let mut labels_all = vec![-1; n];
        for (&i, &label) in index_seed.iter().zip(labels_seed.iter()) {
            labels_all[i] = label;
        }
        let mut prev_remain = vec![i32::MIN; index_remain.len()];

        let mut t = 0usize;
        while t < self.n_iter {
            let current: Vec<i32> = index_remain.iter().map(|&i| labels_all[i]).collect();
            if current == prev_remain {
                break;
            }
            prev_remain = current;
            Self::update_labels(&adjacency, &mut labels_all, &index_remain, self.weighted);
            t += 1;
        }

        let labels_arr = Array1::from_vec(labels_all.clone());
        let membership = get_membership(&labels_arr, None).map_err(PropagationError::Membership)?;
        let probs = Self::normalize_rows(&(&adjacency * &membership));

        self.state.labels = Some(labels_arr);
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
    ) -> Result<Array1<i32>, PropagationError> {
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
    ) -> Result<CsMat<f64>, PropagationError> {
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
    ) -> Result<CsMat<f64>, PropagationError> {
        self.fit(input_matrix, labels, labels_row, labels_col)?;
        self.transform(false)
    }

    /// Predicts labels after fitting.
    ///
    /// # Errors
    /// Returns a wrapped [`BaseClassifierError::NotFitted`] when called before
    /// `fit`.
    pub fn predict(&self, columns: bool) -> Result<Array1<i32>, PropagationError> {
        Ok(self.state.predict(columns)?)
    }

    /// Returns fitted class probabilities for rows or columns.
    ///
    /// # Errors
    /// Returns [`PropagationError::Base`] when probabilities are unavailable.
    pub fn predict_proba(&self, columns: bool) -> Result<CsMat<f64>, PropagationError> {
        Ok(self.state.predict_proba(columns)?)
    }

    /// Returns class-membership probabilities after fitting.
    ///
    /// # Errors
    /// Returns a wrapped [`BaseClassifierError::NotFitted`] when called before
    /// `fit`.
    pub fn transform(&self, columns: bool) -> Result<CsMat<f64>, PropagationError> {
        Ok(self.state.transform(columns)?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::data::test_graphs::{test_bigraph, test_digraph, test_graph};

    #[test]
    fn test_algo() {
        for adjacency in [test_graph(), test_digraph(), test_bigraph()] {
            let n = adjacency.rows();
            let mut labels = HashMap::new();
            labels.insert(0usize, 0.0);
            labels.insert(1usize, 1.0);
            let mut propagation = Propagation::new(3, None, false);
            let labels_pred = propagation
                .fit_predict(
                    &adjacency,
                    Some(ValuesInput::Map(labels.clone())),
                    None,
                    None,
                )
                .unwrap();
            assert_eq!(labels_pred.len(), n);

            for order in ["random", "decreasing", "increasing"] {
                let mut propagation = Propagation::new(-1, Some(order), true);
                let labels_pred = propagation
                    .fit_predict(
                        &adjacency,
                        Some(ValuesInput::Map(labels.clone())),
                        None,
                        None,
                    )
                    .unwrap();
                assert_eq!(labels_pred.len(), n);
            }
        }
    }

    #[test]
    fn test_unknown_node_order_rejected() {
        let adjacency = test_graph();
        let mut labels = HashMap::new();
        labels.insert(0usize, 0.0);
        labels.insert(1usize, 1.0);
        let mut propagation = Propagation::new(3, Some("bad-order"), true);
        assert_eq!(
            propagation.fit(&adjacency, Some(ValuesInput::Map(labels)), None, None),
            Err(PropagationError::UnknownNodeOrder)
        );
    }

    #[test]
    fn test_node_order_case_insensitive() {
        let adjacency = test_graph();
        let mut labels = HashMap::new();
        labels.insert(0usize, 0.0);
        labels.insert(1usize, 1.0);
        let mut propagation = Propagation::new(3, Some("RaNdOm"), true);
        let labels_pred = propagation
            .fit_predict(&adjacency, Some(ValuesInput::Map(labels)), None, None)
            .unwrap();
        assert_eq!(labels_pred.len(), adjacency.rows());
    }
}
