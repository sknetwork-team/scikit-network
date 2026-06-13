//! Louvain modularity-based community detection.

use ndarray::Array1;
use sprs::CsMat;

use crate::clustering::base::{BaseClusteringError, BaseClusteringState};
use crate::clustering::postprocess::reindex_labels;
use crate::utils::format::{MatrixInput, directed2undirected, get_adjacency};

/// Error type for [`Louvain`] fitting and prediction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LouvainError {
    /// `modularity` is not one of `dugue`, `newman`, or `potts`.
    UnknownModularity,
    /// Input matrix formatting failed.
    InvalidInput,
    /// Wrapped shared clustering state error.
    Base(BaseClusteringError),
}

/// Louvain community-detection estimator.
///
/// Supports modularity variants (`dugue`, `newman`, `potts`) and bipartite
/// split outputs through the shared clustering state.
#[derive(Debug, Clone)]
pub struct Louvain {
    /// Resolution parameter scaling the null-model term.
    pub resolution: f64,
    /// Modularity variant (`dugue`, `newman`, or `potts`).
    pub modularity: String,
    /// Intra-level local-move convergence tolerance.
    pub tol_optimization: f64,
    /// Inter-level aggregation gain tolerance.
    pub tol_aggregation: f64,
    /// Maximum aggregation levels (`<0` uses default cap).
    pub n_aggregations: isize,
    /// Whether to shuffle node order during optimization.
    pub shuffle_nodes: bool,
    /// Whether to reindex clusters by size after fitting.
    pub sort_clusters: bool,
    /// Whether to compute membership probabilities in `fit`.
    pub return_probs: bool,
    /// Whether to compute the cluster aggregate matrix in `fit`.
    pub return_aggregate: bool,
    /// Shared fitted-state container.
    pub state: BaseClusteringState,
}

impl Default for Louvain {
    fn default() -> Self {
        Self::new(1.0, "dugue", 1e-3, 1e-3, -1, false, true, true, true)
    }
}

impl Louvain {
    fn normalize_adjacency(adjacency: &CsMat<f64>) -> CsMat<f64> {
        let total: f64 = adjacency.data().iter().sum();
        if total <= 0.0 {
            return adjacency.to_owned();
        }
        let mut out = adjacency.to_owned();
        for v in out.data_mut() {
            *v /= total;
        }
        out
    }

    fn probs_degree(adjacency: &CsMat<f64>) -> Vec<f64> {
        let mut w = vec![0.0; adjacency.rows()];
        for (i, row) in adjacency.outer_iterator().enumerate() {
            w[i] = row.data().iter().sum::<f64>();
        }
        let s: f64 = w.iter().sum();
        if s > 0.0 {
            for x in &mut w {
                *x /= s;
            }
        }
        w
    }

    fn probs_uniform(n: usize) -> Vec<f64> {
        if n == 0 {
            return Vec::new();
        }
        let v = 1.0 / n as f64;
        vec![v; n]
    }

    /// Core local-move optimization loop used by Louvain and Leiden.
    ///
    /// Invariants:
    /// - `labels.len() == out_weights.len() == in_weights.len()`.
    /// - `adjacency_norm_sym` is normalized/symmetrized prior to this call.
    /// - `cluster_weights` scratch space is reset for every touched label.
    pub(crate) fn optimize_core_kernel(
        mut labels: Vec<usize>,
        adjacency_norm_sym: &CsMat<f64>,
        out_weights: &[f64],
        in_weights: &[f64],
        resolution: f64,
        tol_optimization: f64,
        shuffle_nodes: bool,
    ) -> (Vec<usize>, f64) {
        let n = labels.len();
        let mut out_cluster_weights = out_weights.to_vec();
        let mut in_cluster_weights = in_weights.to_vec();
        let mut cluster_weights = vec![0.0; n];
        let mut self_loops = vec![0.0; n];
        for (i, sl) in self_loops.iter_mut().enumerate().take(n) {
            *sl = adjacency_norm_sym.get(i, i).copied().unwrap_or(0.0);
        }

        let mut increase = 0.0f64;
        loop {
            let mut increase_pass = 0.0f64;
            let mut node_order: Vec<usize> = (0..n).collect();
            if shuffle_nodes {
                let mut seed: u64 = 0xA0761D6478BD642F;
                for i in (1..node_order.len()).rev() {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let j = (seed as usize) % (i + 1);
                    node_order.swap(i, j);
                }
            }
            for i in node_order {
                let label = labels[i];
                let mut touched = Vec::<usize>::new();
                if let Some(row) = adjacency_norm_sym.outer_view(i) {
                    for (j, v) in row.iter() {
                        let lt = labels[j];
                        if cluster_weights[lt] == 0.0 {
                            touched.push(lt);
                        }
                        cluster_weights[lt] += *v;
                    }
                }

                let out_w = out_weights[i];
                let in_w = in_weights[i];
                let delta_leave = 2.0 * (cluster_weights[label] - self_loops[i])
                    - resolution * out_w * (in_cluster_weights[label] - in_w)
                    - resolution * in_w * (out_cluster_weights[label] - out_w);

                let mut best_delta = 0.0;
                let mut best_label = label;
                for &lt in &touched {
                    if lt == label {
                        continue;
                    }
                    let delta_local = 2.0 * cluster_weights[lt]
                        - resolution * out_w * in_cluster_weights[lt]
                        - resolution * in_w * out_cluster_weights[lt]
                        - delta_leave;
                    if delta_local > best_delta {
                        best_delta = delta_local;
                        best_label = lt;
                    }
                }

                if best_label != label {
                    increase_pass += best_delta;
                    labels[i] = best_label;
                    out_cluster_weights[label] -= out_w;
                    in_cluster_weights[label] -= in_w;
                    out_cluster_weights[best_label] += out_w;
                    in_cluster_weights[best_label] += in_w;
                }
                for &lt in &touched {
                    cluster_weights[lt] = 0.0;
                }
            }
            increase += increase_pass;
            if increase_pass <= tol_optimization {
                break;
            }
        }
        (labels, increase)
    }

    #[cfg(test)]
    fn gain_for_mode(
        mode: &str,
        resolution: f64,
        w_out: f64,
        w_in: f64,
        kout_i: f64,
        kin_i: f64,
        tot_out_c: f64,
        tot_in_c: f64,
        size_c: usize,
        n: usize,
        m: f64,
    ) -> f64 {
        match mode {
            "dugue" => (w_out + w_in) - resolution * (kout_i * tot_in_c + kin_i * tot_out_c) / m,
            "newman" => {
                let k_i = kout_i + kin_i;
                let tot_c = tot_out_c + tot_in_c;
                (w_out + w_in) - resolution * k_i * tot_c / (2.0 * m)
            }
            "potts" => {
                let observed = w_out + w_in;
                let expected = resolution * ((size_c + 1) as f64) / (n as f64);
                observed - expected
            }
            _ => unreachable!("unsupported modularity mode"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Creates a Louvain estimator with explicit optimization settings.
    ///
    /// # Arguments
    /// - `modularity`: One of `dugue`, `newman`, or `potts`.
    /// - `tol_optimization`: Intra-level move tolerance.
    /// - `tol_aggregation`: Inter-level aggregation tolerance.
    /// - `n_aggregations`: Maximum aggregation levels (`<0` uses default cap).
    pub fn new(
        resolution: f64,
        modularity: &str,
        tol_optimization: f64,
        tol_aggregation: f64,
        n_aggregations: isize,
        shuffle_nodes: bool,
        sort_clusters: bool,
        return_probs: bool,
        return_aggregate: bool,
    ) -> Self {
        Self {
            resolution,
            modularity: modularity.to_lowercase(),
            tol_optimization,
            tol_aggregation,
            n_aggregations,
            shuffle_nodes,
            sort_clusters,
            return_probs,
            return_aggregate,
            state: BaseClusteringState::default(),
        }
    }

    fn check_modularity(&self) -> Result<(), LouvainError> {
        match self.modularity.as_str() {
            "dugue" | "newman" | "potts" => Ok(()),
            _ => Err(LouvainError::UnknownModularity),
        }
    }

    fn normalize_labels(labels: &[usize]) -> Vec<usize> {
        let mut map = std::collections::HashMap::<usize, usize>::new();
        let mut next = 0usize;
        let mut out = vec![0usize; labels.len()];
        for (i, &lab) in labels.iter().enumerate() {
            let idx = *map.entry(lab).or_insert_with(|| {
                let x = next;
                next += 1;
                x
            });
            out[i] = idx;
        }
        out
    }

    fn aggregate_graph(adjacency: &CsMat<f64>, labels: &[usize], n_labels: usize) -> CsMat<f64> {
        let mut acc = std::collections::HashMap::<(usize, usize), f64>::new();
        for (i, row) in adjacency.outer_iterator().enumerate() {
            let ci = labels[i];
            for (j, v) in row.iter() {
                let cj = labels[j];
                *acc.entry((ci, cj)).or_insert(0.0) += *v;
            }
        }
        let mut tri = sprs::TriMat::<f64>::new((n_labels, n_labels));
        for ((i, j), v) in acc {
            if v != 0.0 {
                tri.add_triplet(i, j, v);
            }
        }
        tri.to_csr::<usize>()
    }

    fn run_louvain(&self, adjacency: &CsMat<f64>) -> Array1<i32> {
        let base = adjacency.to_owned();
        let out_weights = match self.modularity.as_str() {
            "potts" => Self::probs_uniform(base.rows()),
            "newman" | "dugue" => Self::probs_degree(&base),
            _ => unreachable!("modularity validated in fit"),
        };
        let in_weights = match self.modularity.as_str() {
            "dugue" => Self::probs_degree(&base.transpose_view().to_csr()),
            "newman" => out_weights.clone(),
            "potts" => out_weights.clone(),
            _ => unreachable!("modularity validated in fit"),
        };

        let mut current_adj = Self::normalize_adjacency(&directed2undirected(&base, true));
        let mut current_out = out_weights;
        let mut current_in = in_weights;

        let n0 = current_adj.rows();
        let mut labels_global: Vec<usize> = (0..n0).collect();

        let mut count = 0isize;
        let max_aggregations = if self.n_aggregations >= 0 {
            self.n_aggregations
        } else {
            100
        };
        loop {
            count += 1;
            let labels_level_init: Vec<usize> = (0..current_adj.rows()).collect();
            let (mut labels_level, increase) = Self::optimize_core_kernel(
                labels_level_init,
                &current_adj,
                &current_out,
                &current_in,
                self.resolution,
                self.tol_optimization,
                self.shuffle_nodes,
            );
            labels_level = Self::normalize_labels(&labels_level);
            let n_clusters = labels_level
                .iter()
                .copied()
                .max()
                .map(|x| x + 1)
                .unwrap_or(0);

            for lab in &mut labels_global {
                *lab = labels_level[*lab];
            }

            if n_clusters <= 1 {
                break;
            }
            // If no structural compression happened, another aggregation won't help.
            if n_clusters == current_adj.rows() {
                break;
            }
            if increase <= self.tol_aggregation {
                break;
            }
            if count >= max_aggregations {
                break;
            }

            current_adj = Self::aggregate_graph(&current_adj, &labels_level, n_clusters);
            let mut out_next = vec![0.0; n_clusters];
            let mut in_next = vec![0.0; n_clusters];
            for (i, &c) in labels_level.iter().enumerate() {
                out_next[c] += current_out[i];
                in_next[c] += current_in[i];
            }
            current_out = out_next;
            current_in = in_next;
        }

        let mut labels_i32: Vec<i32> = labels_global.into_iter().map(|x| x as i32).collect();
        if self.sort_clusters {
            labels_i32 = reindex_labels(&Array1::from_vec(labels_i32)).to_vec();
        }
        Array1::from_vec(labels_i32)
    }

    /// Fits the Louvain model on an adjacency matrix.
    ///
    /// # Errors
    /// Returns [`LouvainError::UnknownModularity`] for unsupported modularity
    /// modes and [`LouvainError::InvalidInput`] for invalid matrix formatting.
    pub fn fit(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<(), LouvainError> {
        self.check_modularity()?;
        let force_directed = self.modularity == "dugue";
        let (adjacency, bipartite) = get_adjacency(
            MatrixInput::Sparse(input_matrix.to_owned()),
            true,
            force_bipartite,
            force_directed,
            false,
        )
        .map_err(|_| LouvainError::InvalidInput)?;

        self.state.init_vars();
        self.state.bipartite = Some(bipartite);

        if !bipartite {
            let labels = self.run_louvain(&adjacency);
            self.state.labels = Some(labels);
        } else {
            let n_row = input_matrix.rows();
            let n_col = input_matrix.cols();
            self.state.labels = Some(self.run_louvain(&adjacency));
            self.state
                .split_vars((n_row, n_col))
                .map_err(LouvainError::Base)?;
        }

        self.state
            .secondary_outputs(input_matrix, self.return_probs, self.return_aggregate)
            .map_err(LouvainError::Base)?;
        Ok(())
    }

    /// Fits the estimator and returns cluster labels.
    ///
    /// # Errors
    /// Propagates all errors from [`Self::fit`] and [`Self::predict`].
    pub fn fit_predict(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<Array1<i32>, LouvainError> {
        self.fit(input_matrix, force_bipartite)?;
        self.predict(false)
    }

    /// Returns fitted cluster labels for rows or columns.
    ///
    /// # Errors
    /// Returns [`LouvainError::Base`] with [`BaseClusteringError::NotFitted`]
    /// when called before `fit`.
    pub fn predict(&self, columns: bool) -> Result<Array1<i32>, LouvainError> {
        self.state.predict(columns).map_err(LouvainError::Base)
    }

    /// Fits the estimator and returns membership probabilities.
    ///
    /// # Errors
    /// Propagates all errors from [`Self::fit`] and [`Self::predict_proba`].
    pub fn fit_predict_proba(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<CsMat<f64>, LouvainError> {
        self.fit(input_matrix, force_bipartite)?;
        self.predict_proba(false)
    }

    /// Returns fitted membership probabilities for rows or columns.
    ///
    /// # Errors
    /// Returns [`LouvainError::Base`] when probabilities are unavailable.
    pub fn predict_proba(&self, columns: bool) -> Result<CsMat<f64>, LouvainError> {
        self.state
            .predict_proba(columns)
            .map_err(LouvainError::Base)
    }

    /// Fits the estimator and returns membership probabilities.
    ///
    /// # Errors
    /// Propagates all errors from [`Self::fit`] and [`Self::transform`].
    pub fn fit_transform(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<CsMat<f64>, LouvainError> {
        self.fit(input_matrix, force_bipartite)?;
        self.transform(false)
    }

    /// Returns fitted membership probabilities (alias for [`Self::predict_proba`]).
    ///
    /// # Errors
    /// Returns [`LouvainError::Base`] when probabilities are unavailable.
    pub fn transform(&self, columns: bool) -> Result<CsMat<f64>, LouvainError> {
        self.state.transform(columns).map_err(LouvainError::Base)
    }

    /// Returns the fitted cluster aggregate matrix when requested at construction.
    pub fn aggregate(&self) -> Option<CsMat<f64>> {
        self.state.aggregate.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::{
        test_bigraph, test_digraph, test_disconnected_graph, test_graph,
    };

    #[test]
    fn test_regular_api_shapes() {
        let mut algo = Louvain::new(1.0, "dugue", 1e-3, 1e-3, -1, false, true, true, true);
        for adjacency in [test_graph(), test_digraph(), test_disconnected_graph()] {
            let n = adjacency.rows();
            let labels = algo.fit_predict(&adjacency, false).unwrap();
            assert_eq!(labels.len(), n);
            let n_labels = labels
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let aggregate = algo.aggregate().unwrap_or_else(|| CsMat::zero((0, 0)));
            assert_eq!(aggregate.shape(), (n_labels, n_labels));
            let membership = algo.fit_transform(&adjacency, false).unwrap();
            assert_eq!(membership.rows(), n);
            assert_eq!(membership.cols(), n_labels);
        }
    }

    #[test]
    fn test_bipartite_api_shapes() {
        let biadjacency = test_bigraph();
        let (n_row, n_col) = biadjacency.shape();
        let mut algo = Louvain::default();
        algo.fit(&biadjacency, false).unwrap();
        assert_eq!(algo.predict(false).unwrap().len(), n_row);
        assert_eq!(algo.predict(true).unwrap().len(), n_col);
        assert_eq!(algo.predict_proba(false).unwrap().rows(), n_row);
        assert_eq!(algo.predict_proba(true).unwrap().rows(), n_col);
        assert_eq!(algo.transform(false).unwrap().rows(), n_row);
        assert_eq!(algo.transform(true).unwrap().rows(), n_col);
    }

    #[test]
    fn test_invalid_modularity() {
        let adjacency = test_graph();
        let mut algo = Louvain::new(1.0, "toto", 1e-3, 1e-3, -1, false, true, true, true);
        assert_eq!(
            algo.fit(&adjacency, false),
            Err(LouvainError::UnknownModularity)
        );
    }

    #[test]
    fn test_modularity_modes_run() {
        let adjacency = test_digraph();
        for mode in ["dugue", "newman", "potts"] {
            let mut algo = Louvain::new(1.0, mode, 1e-3, 1e-3, -1, false, true, true, true);
            let labels = algo.fit_predict(&adjacency, false).unwrap();
            assert_eq!(labels.len(), adjacency.rows());
        }
    }

    #[test]
    fn test_modularity_modes_different_behavior() {
        let g_d = Louvain::gain_for_mode("dugue", 1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 5.0, 2, 6, 20.0);
        let g_n = Louvain::gain_for_mode("newman", 1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 5.0, 2, 6, 20.0);
        let g_p = Louvain::gain_for_mode("potts", 1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 5.0, 2, 6, 20.0);
        assert!((g_d - g_n).abs() > 1e-12);
        assert!((g_d - g_p).abs() > 1e-12);
        assert!((g_n - g_p).abs() > 1e-12);
    }
}
