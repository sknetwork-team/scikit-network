//! Label-propagation clustering for unipartite and bipartite graphs.

use ndarray::Array1;
use sprs::CsMat;

use crate::clustering::base::{BaseClusteringError, BaseClusteringState};
use crate::utils::format::{MatrixInput, get_adjacency};

/// Error type for [`PropagationClustering`] fitting and prediction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropagationClusteringError {
    /// Input matrix formatting failed.
    InvalidInput,
    /// `node_order` is not a supported option.
    UnknownNodeOrder,
    /// Wrapped shared clustering state error.
    Base(BaseClusteringError),
}

/// Label-propagation clustering estimator.
#[derive(Debug, Clone)]
pub struct PropagationClustering {
    /// Maximum propagation passes (`<0` uses a large default cap).
    pub n_iter: isize,
    /// Node processing order (`increasing`, `decreasing`, `random`, or `index`).
    pub node_order: String,
    /// Whether neighbor votes use edge weights.
    pub weighted: bool,
    /// Whether to reindex clusters by first-seen order after fitting.
    pub sort_clusters: bool,
    /// Whether to compute membership probabilities in `fit`.
    pub return_probs: bool,
    /// Whether to compute the cluster aggregate matrix in `fit`.
    pub return_aggregate: bool,
    /// Shared fitted-state container.
    pub state: BaseClusteringState,
}

impl Default for PropagationClustering {
    fn default() -> Self {
        Self::new(5, "decreasing", true, true, true, true)
    }
}

impl PropagationClustering {
    /// Creates a propagation-clustering estimator.
    ///
    /// # Arguments
    /// - `node_order`: One of `increasing`, `decreasing`, `random`, or `index`.
    pub fn new(
        n_iter: isize,
        node_order: &str,
        weighted: bool,
        sort_clusters: bool,
        return_probs: bool,
        return_aggregate: bool,
    ) -> Self {
        Self {
            n_iter,
            node_order: node_order.to_lowercase(),
            weighted,
            sort_clusters,
            return_probs,
            return_aggregate,
            state: BaseClusteringState::default(),
        }
    }

    fn normalize_labels(labels: &[i32]) -> Array1<i32> {
        let mut map = std::collections::HashMap::<i32, i32>::new();
        let mut next = 0i32;
        let mut out = vec![0i32; labels.len()];
        for (i, &lab) in labels.iter().enumerate() {
            let idx = *map.entry(lab).or_insert_with(|| {
                let x = next;
                next += 1;
                x
            });
            out[i] = idx;
        }
        Array1::from_vec(out)
    }

    fn node_order(&self, adjacency: &CsMat<f64>) -> Result<Vec<usize>, PropagationClusteringError> {
        let n = adjacency.rows();
        let mut order: Vec<usize> = (0..n).collect();
        let mut degree = vec![0.0; n];
        for (i, row) in adjacency.outer_iterator().enumerate() {
            degree[i] = row.data().iter().sum();
        }
        match self.node_order.as_str() {
            "increasing" => {
                order.sort_by(|&a, &b| {
                    degree[a]
                        .partial_cmp(&degree[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            "decreasing" => {
                order.sort_by(|&a, &b| {
                    degree[b]
                        .partial_cmp(&degree[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            "random" => {
                let mut seed: u64 = 0xD1B54A32D192ED03;
                for i in (1..order.len()).rev() {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let j = (seed as usize) % (i + 1);
                    order.swap(i, j);
                }
            }
            "index" => {}
            _ => return Err(PropagationClusteringError::UnknownNodeOrder),
        }
        Ok(order)
    }

    fn propagate_labels(
        &self,
        adjacency: &CsMat<f64>,
    ) -> Result<Array1<i32>, PropagationClusteringError> {
        let n = adjacency.rows();
        let mut labels: Vec<i32> = (0..n as i32).collect();
        let order = self.node_order(adjacency)?;
        let max_iter = if self.n_iter < 0 {
            1000
        } else {
            self.n_iter as usize
        };

        for _ in 0..max_iter {
            let mut changed = false;
            for &i in &order {
                let Some(row) = adjacency.outer_view(i) else {
                    continue;
                };
                let mut votes = std::collections::HashMap::<i32, f64>::new();
                for (j, v) in row.iter() {
                    let w = if self.weighted { *v } else { 1.0 };
                    *votes.entry(labels[j]).or_insert(0.0) += w;
                }
                if votes.is_empty() {
                    continue;
                }
                let mut best_label = labels[i];
                let mut best_vote = f64::NEG_INFINITY;
                for (lab, score) in votes {
                    if score > best_vote + 1e-15
                        || ((score - best_vote).abs() <= 1e-15 && lab < best_label)
                    {
                        best_vote = score;
                        best_label = lab;
                    }
                }
                if best_label != labels[i] {
                    labels[i] = best_label;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut out = Self::normalize_labels(&labels);
        if self.sort_clusters {
            // normalize_labels keeps first-seen order, which is enough for stable parity-first behavior.
            out = Self::normalize_labels(&out.to_vec());
        }
        Ok(out)
    }

    /// Fits the estimator on an adjacency matrix.
    ///
    /// # Errors
    /// Returns [`PropagationClusteringError::InvalidInput`] for formatting
    /// failures, [`PropagationClusteringError::UnknownNodeOrder`] for unsupported
    /// order strings, and [`PropagationClusteringError::Base`] for bipartite split
    /// or secondary-output errors.
    pub fn fit(&mut self, input_matrix: &CsMat<f64>) -> Result<(), PropagationClusteringError> {
        let (adjacency, bipartite) = get_adjacency(
            MatrixInput::Sparse(input_matrix.to_owned()),
            true,
            false,
            false,
            false,
        )
        .map_err(|_| PropagationClusteringError::InvalidInput)?;

        self.state.init_vars();
        self.state.bipartite = Some(bipartite);
        self.state.labels = Some(self.propagate_labels(&adjacency)?);
        if bipartite {
            self.state
                .split_vars(input_matrix.shape())
                .map_err(PropagationClusteringError::Base)?;
        }
        self.state
            .secondary_outputs(input_matrix, self.return_probs, self.return_aggregate)
            .map_err(PropagationClusteringError::Base)?;
        Ok(())
    }

    /// Fits the estimator and returns cluster labels.
    ///
    /// # Errors
    /// Propagates all errors from [`Self::fit`] and [`Self::predict`].
    pub fn fit_predict(
        &mut self,
        input_matrix: &CsMat<f64>,
    ) -> Result<Array1<i32>, PropagationClusteringError> {
        self.fit(input_matrix)?;
        self.predict(false)
    }

    /// Returns fitted cluster labels for rows or columns.
    ///
    /// # Errors
    /// Returns [`PropagationClusteringError::Base`] when the estimator is not fitted.
    pub fn predict(&self, columns: bool) -> Result<Array1<i32>, PropagationClusteringError> {
        self.state
            .predict(columns)
            .map_err(PropagationClusteringError::Base)
    }

    /// Fits the estimator and returns membership probabilities.
    ///
    /// # Errors
    /// Propagates all errors from [`Self::fit`] and [`Self::predict_proba`].
    pub fn fit_predict_proba(
        &mut self,
        input_matrix: &CsMat<f64>,
    ) -> Result<CsMat<f64>, PropagationClusteringError> {
        self.fit(input_matrix)?;
        self.predict_proba(false)
    }

    /// Returns fitted membership probabilities for rows or columns.
    ///
    /// # Errors
    /// Returns [`PropagationClusteringError::Base`] when probabilities are unavailable.
    pub fn predict_proba(&self, columns: bool) -> Result<CsMat<f64>, PropagationClusteringError> {
        self.state
            .predict_proba(columns)
            .map_err(PropagationClusteringError::Base)
    }

    /// Fits the estimator and returns membership probabilities.
    ///
    /// # Errors
    /// Propagates all errors from [`Self::fit`] and [`Self::transform`].
    pub fn fit_transform(
        &mut self,
        input_matrix: &CsMat<f64>,
    ) -> Result<CsMat<f64>, PropagationClusteringError> {
        self.fit(input_matrix)?;
        self.transform(false)
    }

    /// Returns fitted membership probabilities (alias for [`Self::predict_proba`]).
    ///
    /// # Errors
    /// Returns [`PropagationClusteringError::Base`] when probabilities are unavailable.
    pub fn transform(&self, columns: bool) -> Result<CsMat<f64>, PropagationClusteringError> {
        self.state
            .transform(columns)
            .map_err(PropagationClusteringError::Base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::{
        test_bigraph, test_digraph, test_disconnected_graph, test_graph,
    };

    #[test]
    fn test_regular_shapes() {
        let mut algo = PropagationClustering::default();
        for adjacency in [test_graph(), test_digraph(), test_disconnected_graph()] {
            let n = adjacency.rows();
            let labels = algo.fit_predict(&adjacency).unwrap();
            assert_eq!(labels.len(), n);
            let membership = algo.fit_transform(&adjacency).unwrap();
            assert_eq!(membership.rows(), n);
            assert!(membership.cols() >= 1);
        }
    }

    #[test]
    fn test_bipartite_shapes() {
        let biadjacency = test_bigraph();
        let (n_row, n_col) = biadjacency.shape();
        let mut algo = PropagationClustering::default();
        algo.fit(&biadjacency).unwrap();
        assert_eq!(algo.predict(false).unwrap().len(), n_row);
        assert_eq!(algo.predict(true).unwrap().len(), n_col);
        assert_eq!(algo.predict_proba(false).unwrap().rows(), n_row);
        assert_eq!(algo.predict_proba(true).unwrap().rows(), n_col);
    }

    #[test]
    fn test_node_order_options() {
        let adjacency = test_graph();
        for node_order in ["random", "increasing", "decreasing", "index"] {
            let mut algo = PropagationClustering::new(5, node_order, true, true, true, true);
            let labels = algo.fit_predict(&adjacency).unwrap();
            assert_eq!(labels.len(), adjacency.rows());
        }
    }

    #[test]
    fn test_node_order_unknown_rejected() {
        let adjacency = test_graph();
        let mut algo = PropagationClustering::new(5, "bad-order", true, true, true, true);
        assert_eq!(
            algo.fit(&adjacency),
            Err(PropagationClusteringError::UnknownNodeOrder)
        );
    }

    #[test]
    fn test_node_order_case_insensitive() {
        let adjacency = test_graph();
        let mut algo = PropagationClustering::new(5, "InCrEaSiNg", true, true, true, true);
        let labels = algo.fit_predict(&adjacency).unwrap();
        assert_eq!(labels.len(), adjacency.rows());
    }
}
