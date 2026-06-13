//! Leiden community-detection with refinement and aggregation.

use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sprs::{CsMat, TriMat};

use crate::clustering::base::BaseClusteringError;
use crate::clustering::louvain::{Louvain, LouvainError};
use crate::clustering::postprocess::reindex_labels;
use crate::utils::format::{MatrixInput, get_adjacency};

/// Error type for [`Leiden`] fitting and prediction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeidenError {
    /// Wrapped Louvain backend or prediction error.
    Louvain(LouvainError),
    /// Internal state is inconsistent after an optimization step.
    InvalidState,
    /// Wrapped shared clustering state error.
    Base(BaseClusteringError),
}

/// Leiden community-detection estimator with refinement passes.
#[derive(Debug, Clone)]
pub struct Leiden {
    /// Louvain optimization backend and fitted state.
    pub inner: Louvain,
    /// Optional RNG seed for reproducible refinement moves.
    pub random_state: Option<u64>,
}

impl Default for Leiden {
    fn default() -> Self {
        Self::new(1.0, "dugue", 1e-3, 1e-3, -1, false, true, true, true)
    }
}

impl Leiden {
    fn normalize_labels(labels: &[i32]) -> Vec<usize> {
        let mut map = std::collections::HashMap::<i32, usize>::new();
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
        let mut tri = TriMat::<f64>::new((n_labels, n_labels));
        for ((i, j), v) in acc {
            if v != 0.0 {
                tri.add_triplet(i, j, v);
            }
        }
        tri.to_csr::<usize>()
    }

    fn partition_score(adjacency: &CsMat<f64>, labels: &[usize]) -> f64 {
        let total_weight: f64 = adjacency.data().iter().sum();
        if total_weight <= 0.0 {
            return 0.0;
        }
        let mut same = 0.0;
        for (i, row) in adjacency.outer_iterator().enumerate() {
            for (j, v) in row.iter() {
                if labels[i] == labels[j] {
                    same += *v;
                }
            }
        }
        same / total_weight
    }

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
        vec![1.0 / n as f64; n]
    }

    fn optimize_refine_kernel(
        labels: &[usize],
        adjacency_norm_sym: &CsMat<f64>,
        out_weights: &[f64],
        in_weights: &[f64],
        resolution: f64,
        rng: &mut StdRng,
    ) -> Vec<usize> {
        let n = labels.len();
        let mut labels_refined: Vec<usize> = (0..n).collect();
        let mut out_cluster_weights = out_weights.to_vec();
        let mut in_cluster_weights = in_weights.to_vec();
        let mut cluster_weights = vec![0.0; n];
        let mut self_loops = vec![0.0; n];
        for (i, sl) in self_loops.iter_mut().enumerate().take(n) {
            *sl = adjacency_norm_sym.get(i, i).copied().unwrap_or(0.0);
        }

        loop {
            let mut changed = false;
            for i in 0..n {
                let coarse = labels[i];
                let lab_ref = labels_refined[i];
                let mut touched = Vec::<usize>::new();
                if let Some(row) = adjacency_norm_sym.outer_view(i) {
                    for (j, v) in row.iter() {
                        if labels[j] == coarse {
                            let lt = labels_refined[j];
                            if cluster_weights[lt] == 0.0 {
                                touched.push(lt);
                            }
                            cluster_weights[lt] += *v;
                        }
                    }
                }
                let out_w = out_weights[i];
                let in_w = in_weights[i];
                let delta_leave = 2.0 * (cluster_weights[lab_ref] - self_loops[i])
                    - resolution * out_w * (in_cluster_weights[lab_ref] - in_w)
                    - resolution * in_w * (out_cluster_weights[lab_ref] - out_w);

                let mut candidates = Vec::<usize>::new();
                for &lt in &touched {
                    if lt == lab_ref {
                        continue;
                    }
                    let delta_local = 2.0 * cluster_weights[lt]
                        - resolution * out_w * in_cluster_weights[lt]
                        - resolution * in_w * out_cluster_weights[lt]
                        - delta_leave;
                    if delta_local > 0.0 {
                        candidates.push(lt);
                    }
                }
                if !candidates.is_empty() {
                    let idx = rng.random_range(0..candidates.len());
                    let target = candidates[idx];
                    changed = true;
                    labels_refined[i] = target;
                    out_cluster_weights[lab_ref] -= out_w;
                    in_cluster_weights[lab_ref] -= in_w;
                    out_cluster_weights[target] += out_w;
                    in_cluster_weights[target] += in_w;
                }
                for &lt in &touched {
                    cluster_weights[lt] = 0.0;
                }
            }
            if !changed {
                break;
            }
        }
        labels_refined
    }

    #[allow(clippy::too_many_arguments)]
    /// Creates a Leiden estimator with explicit optimization settings.
    ///
    /// # Arguments
    /// - `modularity`: One of `dugue`, `newman`, or `potts`.
    /// - `tol_optimization`: Intra-level move tolerance passed to the backend.
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
        // Parity-first: same optimization backend as Louvain,
        // while keeping a distinct Leiden API surface.
        Self {
            inner: Louvain::new(
                resolution,
                modularity,
                tol_optimization,
                tol_aggregation,
                n_aggregations,
                shuffle_nodes,
                sort_clusters,
                return_probs,
                return_aggregate,
            ),
            random_state: None,
        }
    }

    /// Fits the Leiden model on an adjacency matrix.
    ///
    /// # Errors
    /// Returns [`LeidenError::Louvain`] for unsupported modularity modes or
    /// backend failures, [`LeidenError::InvalidState`] for formatting issues,
    /// and [`LeidenError::Base`] for bipartite split or secondary-output errors.
    pub fn fit(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<(), LeidenError> {
        if !matches!(self.inner.modularity.as_str(), "dugue" | "newman" | "potts") {
            return Err(LeidenError::Louvain(LouvainError::UnknownModularity));
        }
        let (adjacency, bipartite) = get_adjacency(
            MatrixInput::Sparse(input_matrix.to_owned()),
            true,
            force_bipartite,
            false,
            false,
        )
        .map_err(|_| LeidenError::InvalidState)?;

        // Iterative Leiden structure with kernel-level optimize/refine/aggregate.
        let base = adjacency.clone();
        let out_weights = match self.inner.modularity.as_str() {
            "potts" => Self::probs_uniform(base.rows()),
            "newman" | "dugue" => Self::probs_degree(&base),
            _ => unreachable!("modularity validated at fit start"),
        };
        let in_weights = match self.inner.modularity.as_str() {
            "dugue" => Self::probs_degree(&base.transpose_view().to_csr()),
            "newman" => out_weights.clone(),
            "potts" => out_weights.clone(),
            _ => unreachable!("modularity validated at fit start"),
        };

        let mut current =
            Self::normalize_adjacency(&crate::utils::format::directed2undirected(&base, true));
        let mut current_out = out_weights;
        let mut current_in = in_weights;
        let n_total = adjacency.rows();
        let mut node_map: Vec<usize> = (0..n_total).collect();
        let mut level = 0isize;
        let mut prev_level_score: Option<f64> = None;
        let seed = self.random_state.unwrap_or_else(rand::random::<u64>);
        let mut rng = StdRng::seed_from_u64(seed);
        let max_levels = if self.inner.n_aggregations >= 0 {
            self.inner.n_aggregations
        } else {
            100
        };

        loop {
            level += 1;
            let mut level_algo = self.inner.clone();
            level_algo
                .fit(&current, false)
                .map_err(LeidenError::Louvain)?;
            let labels_level_i32 = level_algo
                .state
                .labels
                .clone()
                .ok_or(LeidenError::InvalidState)?;
            let coarse_norm = Self::normalize_labels(&labels_level_i32.to_vec());
            let optimize_score = Self::partition_score(&current, &coarse_norm);
            let refined_level = Self::optimize_refine_kernel(
                &coarse_norm,
                &current,
                &current_out,
                &current_in,
                self.inner.resolution,
                &mut rng,
            );
            let refined_norm = Self::normalize_labels(
                &refined_level.iter().map(|&x| x as i32).collect::<Vec<_>>(),
            );
            let n_clusters = refined_norm
                .iter()
                .copied()
                .max()
                .map(|x| x + 1)
                .unwrap_or(0);
            let refine_score = Self::partition_score(&current, &refined_norm);

            for entry in &mut node_map {
                *entry = refined_norm[*entry];
            }

            let base_score = prev_level_score.unwrap_or(0.0);
            let optimization_gain = optimize_score - base_score;
            let refinement_gain = refine_score - optimize_score;
            let low_opt_gain = optimization_gain <= self.inner.tol_aggregation;
            let low_refine_gain = refinement_gain <= self.inner.tol_aggregation;
            if n_clusters <= 1
                || n_clusters == current.rows()
                || level >= max_levels
                || (low_opt_gain && low_refine_gain)
            {
                break;
            }
            prev_level_score = Some(refine_score);
            current = Self::aggregate_graph(&current, &refined_norm, n_clusters);
            let mut out_next = vec![0.0; n_clusters];
            let mut in_next = vec![0.0; n_clusters];
            for (i, &c) in refined_norm.iter().enumerate() {
                out_next[c] += current_out[i];
                in_next[c] += current_in[i];
            }
            current_out = out_next;
            current_in = in_next;
        }

        let mut final_labels = Array1::from_vec(node_map.into_iter().map(|x| x as i32).collect());
        if self.inner.sort_clusters {
            final_labels = reindex_labels(&final_labels);
        }
        self.inner.state.labels = Some(final_labels);
        self.inner.state.bipartite = Some(bipartite);
        if bipartite {
            self.inner
                .state
                .split_vars(input_matrix.shape())
                .map_err(LeidenError::Base)?;
        }
        self.inner
            .state
            .secondary_outputs(
                input_matrix,
                self.inner.return_probs,
                self.inner.return_aggregate,
            )
            .map_err(LeidenError::Base)?;
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
    ) -> Result<Array1<i32>, LeidenError> {
        self.fit(input_matrix, force_bipartite)?;
        self.predict(false)
    }

    /// Returns fitted cluster labels for rows or columns.
    ///
    /// # Errors
    /// Returns [`LeidenError::Louvain`] when the estimator is not fitted.
    pub fn predict(&self, columns: bool) -> Result<Array1<i32>, LeidenError> {
        self.inner.predict(columns).map_err(LeidenError::Louvain)
    }

    /// Fits the estimator and returns membership probabilities.
    ///
    /// # Errors
    /// Propagates all errors from [`Self::fit`] and [`Self::predict_proba`].
    pub fn fit_predict_proba(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<CsMat<f64>, LeidenError> {
        self.fit(input_matrix, force_bipartite)?;
        self.predict_proba(false)
    }

    /// Returns fitted membership probabilities for rows or columns.
    ///
    /// # Errors
    /// Returns [`LeidenError::Louvain`] when probabilities are unavailable.
    pub fn predict_proba(&self, columns: bool) -> Result<CsMat<f64>, LeidenError> {
        self.inner
            .predict_proba(columns)
            .map_err(LeidenError::Louvain)
    }

    /// Fits the estimator and returns membership probabilities.
    ///
    /// # Errors
    /// Propagates all errors from [`Self::fit`] and [`Self::transform`].
    pub fn fit_transform(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<CsMat<f64>, LeidenError> {
        self.fit(input_matrix, force_bipartite)?;
        self.transform(false)
    }

    /// Returns fitted membership probabilities (alias for [`Self::predict_proba`]).
    ///
    /// # Errors
    /// Returns [`LeidenError::Louvain`] when probabilities are unavailable.
    pub fn transform(&self, columns: bool) -> Result<CsMat<f64>, LeidenError> {
        self.inner.transform(columns).map_err(LeidenError::Louvain)
    }

    /// Returns the fitted cluster aggregate matrix when requested at construction.
    pub fn aggregate(&self) -> Option<CsMat<f64>> {
        self.inner.aggregate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::{test_bigraph, test_disconnected_graph, test_graph};
    use crate::utils::format::bipartite2undirected;

    fn labels_are_component_connected(adjacency: &CsMat<f64>, labels: &Array1<i32>) -> bool {
        let n = adjacency.rows();
        let mut groups = std::collections::BTreeSet::<i32>::new();
        for &x in labels {
            groups.insert(x);
        }
        for g in groups {
            let nodes: Vec<usize> = (0..n).filter(|&i| labels[i] == g).collect();
            if nodes.len() <= 1 {
                continue;
            }
            let mut seen = std::collections::BTreeSet::<usize>::new();
            let mut stack = vec![nodes[0]];
            seen.insert(nodes[0]);
            while let Some(u) = stack.pop() {
                if let Some(row) = adjacency.outer_view(u) {
                    for (v, _) in row.iter() {
                        if labels[v] == g && !seen.contains(&v) {
                            seen.insert(v);
                            stack.push(v);
                        }
                    }
                }
            }
            if seen.len() != nodes.len() {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_disconnected() {
        let adjacency = test_disconnected_graph();
        let n = adjacency.rows();
        let labels = Leiden::default().fit_predict(&adjacency, false).unwrap();
        assert_eq!(labels.len(), n);
    }

    #[test]
    fn test_modularity() {
        let adjacency = test_graph();
        let mut leiden_d = Leiden::new(1.0, "dugue", 1e-3, 1e-3, -1, false, true, true, true);
        let mut leiden_n = Leiden::new(1.0, "newman", 1e-3, 1e-3, -1, false, true, true, true);
        let labels_d = leiden_d.fit_predict(&adjacency, false).unwrap();
        let labels_n = leiden_n.fit_predict(&adjacency, false).unwrap();
        assert_eq!(labels_d.to_vec(), labels_n.to_vec());
    }

    #[test]
    fn test_bipartite() {
        let biadjacency = test_bigraph();
        let adjacency = bipartite2undirected(&biadjacency);
        let mut leiden = Leiden::new(1.0, "newman", 1e-3, 1e-3, -1, false, true, true, true);
        let labels1 = leiden.fit_predict(&adjacency, false).unwrap();
        leiden.fit(&biadjacency, false).unwrap();
        let mut labels2 = leiden.predict(false).unwrap().to_vec();
        labels2.extend(leiden.predict(true).unwrap().to_vec());
        assert_eq!(labels1.to_vec(), labels2);
    }

    #[test]
    fn test_refined_clusters_are_connected() {
        let adjacency = test_graph();
        let mut leiden = Leiden::default();
        let labels = leiden.fit_predict(&adjacency, false).unwrap();
        assert!(labels_are_component_connected(&adjacency, &labels));
    }

    #[test]
    fn test_refine_seed_reproducible() {
        let adjacency = test_graph();
        let mut a = Leiden::default();
        a.random_state = Some(42);
        let mut b = Leiden::default();
        b.random_state = Some(42);
        let la = a.fit_predict(&adjacency, false).unwrap();
        let lb = b.fit_predict(&adjacency, false).unwrap();
        assert_eq!(la.to_vec(), lb.to_vec());
    }

    #[test]
    fn test_refine_seed_api_accepts_variation() {
        let adjacency = test_graph();
        let mut a = Leiden::default();
        a.random_state = Some(1);
        let mut b = Leiden::default();
        b.random_state = Some(2);
        let la = a.fit_predict(&adjacency, false).unwrap();
        let lb = b.fit_predict(&adjacency, false).unwrap();
        assert_eq!(la.len(), lb.len());
    }

    #[test]
    fn test_invalid_modularity_rejected() {
        let adjacency = test_graph();
        let mut leiden = Leiden::new(1.0, "bad-mod", 1e-3, 1e-3, -1, false, true, true, true);
        assert_eq!(
            leiden.fit(&adjacency, false),
            Err(LeidenError::Louvain(LouvainError::UnknownModularity))
        );
    }
}
