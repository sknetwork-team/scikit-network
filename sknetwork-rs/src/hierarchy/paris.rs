use sprs::CsMat;

use crate::hierarchy::base::BaseHierarchyState;
use crate::hierarchy::louvain_hierarchy::HierarchyAlgoError;
use crate::hierarchy::postprocess::{Dendrogram, reorder_dendrogram};
use crate::utils::check::{WeightsInput, get_probs, is_symmetric};
use crate::utils::format::{MatrixInput, directed2undirected, get_adjacency};

#[derive(Debug, Clone)]
/// Paris value.
pub struct Paris {
    /// Weights value.
    pub weights: String,
    /// Reorder value.
    pub reorder: bool,
    /// State value.
    pub state: BaseHierarchyState,
    /// Bipartite value.
    pub bipartite: bool,
}

impl Default for Paris {
    fn default() -> Self {
        Self::new("degree", true)
    }
}

impl Paris {
    /// Creates a new instance.
    pub fn new(weights: &str, reorder: bool) -> Self {
        Self {
            weights: weights.to_lowercase(),
            reorder,
            state: BaseHierarchyState::default(),
            bipartite: false,
        }
    }

    fn similarity(
        neighbors: &std::collections::HashMap<usize, std::collections::HashMap<usize, f64>>,
        out_w: &std::collections::HashMap<usize, f64>,
        in_w: &std::collections::HashMap<usize, f64>,
        node1: usize,
        node2: usize,
    ) -> f64 {
        let a = out_w.get(&node1).copied().unwrap_or(0.0) * in_w.get(&node2).copied().unwrap_or(0.0);
        let b = out_w.get(&node2).copied().unwrap_or(0.0) * in_w.get(&node1).copied().unwrap_or(0.0);
        let den = a + b;
        if den <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let e12 = neighbors
            .get(&node1)
            .and_then(|m| m.get(&node2))
            .copied()
            .unwrap_or(0.0);
        2.0 * e12 / den
    }

    fn merge_clusters(
        neighbors: &mut std::collections::HashMap<usize, std::collections::HashMap<usize, f64>>,
        cluster_sizes: &mut std::collections::HashMap<usize, usize>,
        out_w: &mut std::collections::HashMap<usize, f64>,
        in_w: &mut std::collections::HashMap<usize, f64>,
        next_cluster: &mut usize,
        node1: usize,
        node2: usize,
    ) {
        let new_node = *next_cluster;
        let map1 = neighbors.remove(&node1).unwrap_or_default();
        let map2 = neighbors.remove(&node2).unwrap_or_default();

        let mut merged = std::collections::HashMap::<usize, f64>::new();
        let self_loop = map1.get(&node1).copied().unwrap_or(0.0)
            + map1.get(&node2).copied().unwrap_or(0.0)
            + map2.get(&node1).copied().unwrap_or(0.0)
            + map2.get(&node2).copied().unwrap_or(0.0);
        merged.insert(new_node, self_loop);

        for (nbr, w) in map1 {
            if nbr != node1 && nbr != node2 {
                *merged.entry(nbr).or_insert(0.0) += w;
            }
        }
        for (nbr, w) in map2 {
            if nbr != node1 && nbr != node2 {
                *merged.entry(nbr).or_insert(0.0) += w;
            }
        }

        for &nbr in merged.keys() {
            if nbr == new_node {
                continue;
            }
            if let Some(map) = neighbors.get_mut(&nbr) {
                let w1 = map.remove(&node1).unwrap_or(0.0);
                let w2 = map.remove(&node2).unwrap_or(0.0);
                map.insert(new_node, w1 + w2 + map.get(&new_node).copied().unwrap_or(0.0));
            }
        }
        neighbors.insert(new_node, merged);

        let size = cluster_sizes.remove(&node1).unwrap_or(0) + cluster_sizes.remove(&node2).unwrap_or(0);
        cluster_sizes.insert(new_node, size);
        let out_sum = out_w.remove(&node1).unwrap_or(0.0) + out_w.remove(&node2).unwrap_or(0.0);
        out_w.insert(new_node, out_sum);
        let in_sum = in_w.remove(&node1).unwrap_or(0.0) + in_w.remove(&node2).unwrap_or(0.0);
        in_w.insert(new_node, in_sum);
        *next_cluster += 1;
    }

    /// Runs the fit step.
    ///
    /// # Errors
    ///
    /// Returns [`HierarchyAlgoError`] on failure.
    pub fn fit(&mut self, input_matrix: &CsMat<f64>, force_bipartite: bool) -> Result<(), HierarchyAlgoError> {
        self.state.init_vars();
        let weights = self.weights.to_lowercase();
        let (mut adjacency, bip) = get_adjacency(
            MatrixInput::Sparse(input_matrix.to_owned()),
            true,
            force_bipartite,
            false,
            false,
        )
        .map_err(|_| HierarchyAlgoError::InvalidInput)?;
        self.bipartite = bip;
        match weights.as_str() {
            "degree" => {}
            "uniform" => {
                let (r, c) = adjacency.shape();
                let mut tri = sprs::TriMat::<f64>::new((r, c));
                for (i, row) in adjacency.outer_iterator().enumerate() {
                    for (j, v) in row.iter() {
                        if *v != 0.0 {
                            tri.add_triplet(i, j, 1.0);
                        }
                    }
                }
                adjacency = tri.to_csr::<usize>();
            }
            _ => return Err(HierarchyAlgoError::UnknownWeights),
        }
        if !is_symmetric(&adjacency) {
            adjacency = directed2undirected(&adjacency, true);
        }
        if adjacency.rows() <= 1 {
            return Err(HierarchyAlgoError::InvalidInput);
        }

        let mut out_probs = get_probs(WeightsInput::Distribution(weights.clone()), &adjacency, false)
            .map_err(|_| HierarchyAlgoError::InvalidInput)?
            .to_vec();
        let mut in_probs = get_probs(
            WeightsInput::Distribution(weights),
            &adjacency.transpose_view().to_csr(),
            false,
        )
        .map_err(|_| HierarchyAlgoError::InvalidInput)?
        .to_vec();

        let mut null_nodes = Vec::<usize>::new();
        for i in 0..out_probs.len() {
            if out_probs[i] + in_probs[i] == 0.0 {
                null_nodes.push(i);
            }
        }
        if !null_nodes.is_empty() {
            let mut tri = sprs::TriMat::<f64>::new(adjacency.shape());
            for (i, row) in adjacency.outer_iterator().enumerate() {
                for (j, v) in row.iter() {
                    tri.add_triplet(i, j, *v);
                }
            }
            for i in &null_nodes {
                tri.add_triplet(*i, *i, 1.0);
            }
            adjacency = tri.to_csr::<usize>();
            out_probs = get_probs(WeightsInput::Distribution("degree".to_string()), &adjacency, false)
                .map_err(|_| HierarchyAlgoError::InvalidInput)?
                .to_vec();
            in_probs = get_probs(
                WeightsInput::Distribution("degree".to_string()),
                &adjacency.transpose_view().to_csr(),
                false,
            )
            .map_err(|_| HierarchyAlgoError::InvalidInput)?
            .to_vec();
        }

        let total_weight: f64 = adjacency.data().iter().sum();
        if total_weight <= 0.0 {
            return Err(HierarchyAlgoError::InvalidInput);
        }
        let n = adjacency.rows();
        let mut neighbors = std::collections::HashMap::<usize, std::collections::HashMap<usize, f64>>::new();
        let mut cluster_sizes = std::collections::HashMap::<usize, usize>::new();
        let mut cluster_out = std::collections::HashMap::<usize, f64>::new();
        let mut cluster_in = std::collections::HashMap::<usize, f64>::new();
        for i in 0..n {
            let mut map = std::collections::HashMap::<usize, f64>::new();
            if let Some(row) = adjacency.outer_view(i) {
                for (j, v) in row.iter() {
                    map.insert(j, v / total_weight);
                }
            }
            neighbors.insert(i, map);
            cluster_sizes.insert(i, 1);
            cluster_out.insert(i, out_probs[i]);
            cluster_in.insert(i, in_probs[i]);
        }

        let mut next_cluster = n;
        let mut dendrogram: Dendrogram = Vec::new();
        let mut connected_components = Vec::<(usize, usize)>::new();

        while !cluster_sizes.is_empty() {
            let node = *cluster_sizes.keys().next().ok_or(HierarchyAlgoError::ClusteringFailed)?;
            let mut chain = vec![node];
            while let Some(curr) = chain.pop() {
                let neigh_keys: Vec<usize> = neighbors
                    .get(&curr)
                    .map(|m| m.keys().copied().filter(|&x| x != curr).collect())
                    .unwrap_or_else(Vec::new);
                if neigh_keys.is_empty() {
                    let size = cluster_sizes.remove(&curr).unwrap_or(1);
                    connected_components.push((curr, size));
                    continue;
                }
                let mut max_sim = f64::NEG_INFINITY;
                let mut nearest = neigh_keys[0];
                for nb in neigh_keys {
                    let sim = Self::similarity(&neighbors, &cluster_out, &cluster_in, curr, nb);
                    if sim > max_sim || (sim == max_sim && nb < nearest) {
                        max_sim = sim;
                        nearest = nb;
                    }
                }
                if let Some(last) = chain.pop() {
                    if last == nearest {
                        let size = cluster_sizes.get(&curr).copied().unwrap_or(1)
                            + cluster_sizes.get(&nearest).copied().unwrap_or(1);
                        let height = if max_sim > 0.0 {
                            1.0 / max_sim
                        } else {
                            f64::INFINITY
                        };
                        dendrogram.push([curr as f64, nearest as f64, height, size as f64]);
                        Self::merge_clusters(
                            &mut neighbors,
                            &mut cluster_sizes,
                            &mut cluster_out,
                            &mut cluster_in,
                            &mut next_cluster,
                            curr,
                            nearest,
                        );
                    } else {
                        chain.push(last);
                        chain.push(curr);
                        chain.push(nearest);
                    }
                } else {
                    chain.push(curr);
                    chain.push(nearest);
                }
            }
        }

        if connected_components.is_empty() {
            return Err(HierarchyAlgoError::ClusteringFailed);
        }
        let (mut node, mut cluster_size) = connected_components
            .pop()
            .ok_or(HierarchyAlgoError::ClusteringFailed)?;
        for (next_node, next_size) in connected_components {
            cluster_size += next_size;
            dendrogram.push([node as f64, next_node as f64, f64::INFINITY, cluster_size as f64]);
            node = next_cluster;
            next_cluster += 1;
        }

        if self.reorder {
            let mut d = reorder_dendrogram(&dendrogram);
            for t in 1..d.len() {
                if d[t][2] < d[t - 1][2] {
                    d[t][2] = d[t - 1][2];
                }
            }
            self.state.dendrogram = Some(d);
        } else {
            self.state.dendrogram = Some(dendrogram);
        }
        if self.bipartite {
            self.state.split_vars(input_matrix.shape());
        }
        Ok(())
    }

    /// Runs the fit-predict step.
    pub fn fit_predict(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<Dendrogram, HierarchyAlgoError> {
        self.fit(input_matrix, force_bipartite)?;
        self.state.predict(false).map_err(HierarchyAlgoError::from)
    }

    /// Runs the fit-transform step.
    pub fn fit_transform(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<Dendrogram, HierarchyAlgoError> {
        self.fit_predict(input_matrix, force_bipartite)
    }

    /// Runs the predict step.
    ///
    /// # Errors
    ///
    /// Returns [`HierarchyAlgoError`] on failure.
    pub fn predict(&self, columns: bool) -> Result<Dendrogram, HierarchyAlgoError> {
        self.state.predict(columns).map_err(HierarchyAlgoError::from)
    }
}

#[cfg(test)]
mod tests {
    use sprs::TriMat;

    use super::*;
    use crate::data::test_graphs::{test_bigraph, test_graph};

    #[test]
    fn test_paris_reorder_option() {
        let input = test_graph();
        let mut p_true = Paris::new("degree", true);
        let mut p_false = Paris::new("degree", false);
        let d_true = p_true.fit_predict(&input, false).expect("paris true");
        let d_false = p_false.fit_predict(&input, false).expect("paris false");
        assert_eq!(d_true.len(), d_false.len());
        if !d_true.is_empty() {
            for t in 1..d_true.len() {
                assert!(d_true[t - 1][2] <= d_true[t][2]);
            }
        }
    }

    #[test]
    fn test_paris_weights_option() {
        let input = test_graph();
        let mut p_degree = Paris::new("degree", true);
        let mut p_uniform = Paris::new("uniform", true);
        let d_degree = p_degree.fit_predict(&input, false).expect("degree");
        let d_uniform = p_uniform.fit_predict(&input, false).expect("uniform");
        assert_eq!(d_degree.len(), d_uniform.len());

        let mut p_bad = Paris::new("bad-weight", true);
        assert!(matches!(
            p_bad.fit_predict(&input, false),
            Err(HierarchyAlgoError::UnknownWeights)
        ));
    }

    #[test]
    fn test_paris_weights_case_insensitive() {
        let input = test_graph();
        let mut p = Paris::new("UnIfOrM", true);
        let d = p.fit_predict(&input, false).expect("uniform case-insensitive");
        assert_eq!(d.len(), input.rows() - 1);
    }

    #[test]
    fn test_paris_disconnected_components_have_infinite_merge() {
        let mut tri = TriMat::<f64>::new((4, 4));
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(2, 3, 1.0);
        tri.add_triplet(3, 2, 1.0);
        let adjacency = tri.to_csr::<usize>();

        let mut p = Paris::new("degree", false);
        let d = p.fit_predict(&adjacency, false).expect("paris disconnected");
        assert_eq!(d.len(), adjacency.rows() - 1);
        assert!(d.iter().any(|row| row[2].is_infinite()));
    }

    #[test]
    fn test_paris_with_isolated_node() {
        let mut tri = TriMat::<f64>::new((4, 4));
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(1, 2, 1.0);
        tri.add_triplet(2, 1, 1.0);
        // node 3 is isolated and triggers null-weight handling
        let adjacency = tri.to_csr::<usize>();

        let mut p = Paris::new("degree", true);
        let d = p.fit_predict(&adjacency, false).expect("paris isolate");
        assert_eq!(d.len(), adjacency.rows() - 1);
        for row in &d {
            assert!(row[2].is_finite() || row[2].is_infinite());
            assert!(row[3] >= 2.0);
        }
    }

    #[test]
    fn test_paris_tie_break_deterministic() {
        // 4-cycle has symmetric tie situations
        let mut tri = TriMat::<f64>::new((4, 4));
        for (u, v) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            tri.add_triplet(u, v, 1.0);
            tri.add_triplet(v, u, 1.0);
        }
        let adjacency = tri.to_csr::<usize>();

        let mut p1 = Paris::new("uniform", false);
        let mut p2 = Paris::new("uniform", false);
        let d1 = p1.fit_predict(&adjacency, false).expect("paris run1");
        let d2 = p2.fit_predict(&adjacency, false).expect("paris run2");
        assert_eq!(d1.len(), d2.len());
        for (r1, r2) in d1.iter().zip(d2.iter()) {
            assert!((r1[2] - r2[2]).abs() < 1e-12);
            assert!((r1[3] - r2[3]).abs() < 1e-12);
            let p1 = (r1[0].min(r1[1]), r1[0].max(r1[1]));
            let p2 = (r2[0].min(r2[1]), r2[0].max(r2[1]));
            assert_eq!(p1, p2);
        }
    }

    #[test]
    fn test_paris_bipartite_split_consistency() {
        let biadjacency = test_bigraph();
        let mut p = Paris::new("degree", true);
        let row_d = p.fit_predict(&biadjacency, true).expect("paris bipartite row");
        let col_d = p.predict(true).expect("paris bipartite col");
        assert_eq!(row_d.len(), biadjacency.rows().saturating_sub(1));
        assert_eq!(col_d.len(), biadjacency.cols().saturating_sub(1));
        assert!(p.state.dendrogram_full.is_some());
    }
}
