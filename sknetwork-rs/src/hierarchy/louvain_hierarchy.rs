use sprs::CsMat;

use crate::clustering::louvain::Louvain;
use crate::hierarchy::base::{BaseHierarchyError, BaseHierarchyState};
use crate::hierarchy::postprocess::{
    Dendrogram, Tree, get_dendrogram_with_reorder, reorder_dendrogram,
};
use crate::utils::format::{MatrixInput, get_adjacency};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by hierarchy algo error operations.
pub enum HierarchyAlgoError {
    /// Indicates invalid input.
    InvalidInput,
    /// Indicates clustering failed.
    ClusteringFailed,
    /// Indicates unknown weights.
    UnknownWeights,
    /// Indicates not fitted.
    NotFitted,
}

impl From<BaseHierarchyError> for HierarchyAlgoError {
    fn from(_: BaseHierarchyError) -> Self {
        Self::NotFitted
    }
}

#[derive(Debug, Clone)]
/// LouvainIteration value.
pub struct LouvainIteration {
    /// Depth value.
    pub depth: isize,
    /// State value.
    pub state: BaseHierarchyState,
    /// Bipartite value.
    pub bipartite: bool,
    clustering: Louvain,
}

impl Default for LouvainIteration {
    fn default() -> Self {
        Self::new(3, 1.0, 1e-3, 1e-3, -1, false)
    }
}

impl LouvainIteration {
    /// Creates a new instance.
    pub fn new(
        depth: isize,
        resolution: f64,
        tol_optimization: f64,
        tol_aggregation: f64,
        n_aggregations: isize,
        shuffle_nodes: bool,
    ) -> Self {
        Self {
            depth,
            state: BaseHierarchyState::default(),
            bipartite: false,
            clustering: Louvain::new(
                resolution,
                "dugue",
                tol_optimization,
                tol_aggregation,
                n_aggregations,
                shuffle_nodes,
                true,
                true,
                true,
            ),
        }
    }

    fn induced_subgraph(adjacency: &CsMat<f64>, nodes: &[usize]) -> CsMat<f64> {
        let n = nodes.len();
        let map: std::collections::HashMap<usize, usize> =
            nodes.iter().enumerate().map(|(i, &u)| (u, i)).collect();
        let mut tri = sprs::TriMat::<f64>::new((n, n));
        for (new_i, &old_i) in nodes.iter().enumerate() {
            if let Some(row) = adjacency.outer_view(old_i) {
                for (old_j, v) in row.iter() {
                    if let Some(&new_j) = map.get(&old_j) {
                        tri.add_triplet(new_i, new_j, *v);
                    }
                }
            }
        }
        tri.to_csr::<usize>()
    }

    fn recursive_louvain(
        &mut self,
        adjacency: &CsMat<f64>,
        depth: isize,
        nodes: Vec<usize>,
    ) -> Result<Tree, HierarchyAlgoError> {
        let n = adjacency.rows();
        if n == 0 {
            return Ok(Tree::Node(Vec::new()));
        }
        if n == 1 {
            return Ok(Tree::Leaf(nodes[0]));
        }

        let labels = if adjacency.nnz() > 0 && depth != 0 {
            self.clustering
                .fit_predict(adjacency, false)
                .map_err(|_| HierarchyAlgoError::ClusteringFailed)?
                .to_vec()
        } else {
            vec![0i32; n]
        };
        let uniq: std::collections::BTreeSet<i32> = labels.iter().copied().collect();
        if uniq.len() == 1 {
            if n > 1 {
                return Ok(Tree::Node(nodes.into_iter().map(Tree::Leaf).collect()));
            }
            return Ok(Tree::Leaf(nodes[0]));
        }

        let mut groups = std::collections::BTreeMap::<i32, Vec<usize>>::new();
        for (i, &lab) in labels.iter().enumerate() {
            groups.entry(lab).or_default().push(i);
        }

        let mut children = Vec::<Tree>::new();
        for local_idx in groups.into_values() {
            let sub_nodes: Vec<usize> = local_idx.iter().map(|&i| nodes[i]).collect();
            let sub_adj = Self::induced_subgraph(adjacency, &local_idx);
            let child = self.recursive_louvain(&sub_adj, depth - 1, sub_nodes)?;
            children.push(child);
        }
        Ok(Tree::Node(children))
    }

    /// Runs the fit step.
    ///
    /// # Errors
    ///
    /// Returns [`HierarchyAlgoError`] on failure.
    pub fn fit(&mut self, input_matrix: &CsMat<f64>, force_bipartite: bool) -> Result<(), HierarchyAlgoError> {
        self.state.init_vars();
        let (adjacency, bip) = get_adjacency(
            MatrixInput::Sparse(input_matrix.to_owned()),
            true,
            force_bipartite,
            false,
            false,
        )
        .map_err(|_| HierarchyAlgoError::InvalidInput)?;
        self.bipartite = bip;
        let nodes: Vec<usize> = (0..adjacency.rows()).collect();
        let tree = self.recursive_louvain(&adjacency, self.depth, nodes)?;
        let mut dendrogram = get_dendrogram_with_reorder(&tree, false);
        if !dendrogram.is_empty() {
            let min_h = dendrogram.iter().map(|r| r[2]).fold(f64::INFINITY, f64::min);
            for row in &mut dendrogram {
                row[2] += 1.0 - min_h;
            }
        }
        self.state.dendrogram = Some(reorder_dendrogram(&dendrogram));
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

#[derive(Debug, Clone)]
/// LouvainHierarchy value.
pub struct LouvainHierarchy {
    /// State value.
    pub state: BaseHierarchyState,
    /// Bipartite value.
    pub bipartite: bool,
    clustering: Louvain,
}

impl Default for LouvainHierarchy {
    fn default() -> Self {
        Self::new(1.0, 1e-3, 1e-3, false)
    }
}

impl LouvainHierarchy {
    /// Creates a new instance.
    pub fn new(
        resolution: f64,
        tol_optimization: f64,
        tol_aggregation: f64,
        shuffle_nodes: bool,
    ) -> Self {
        Self {
            state: BaseHierarchyState::default(),
            bipartite: false,
            clustering: Louvain::new(
                resolution,
                "dugue",
                tol_optimization,
                tol_aggregation,
                1,
                shuffle_nodes,
                true,
                true,
                true,
            ),
        }
    }

    fn compress_tree_level(tree: Vec<Tree>, labels: &[i32]) -> Vec<Tree> {
        let mut buckets = std::collections::BTreeMap::<i32, Vec<Tree>>::new();
        for (idx, node) in tree.into_iter().enumerate() {
            let label = labels.get(idx).copied().unwrap_or(0);
            buckets.entry(label).or_default().push(node);
        }
        let mut out = Vec::with_capacity(buckets.len());
        for mut children in buckets.into_values() {
            if children.len() == 1 {
                out.push(children.remove(0));
            } else {
                out.push(Tree::Node(children));
            }
        }
        out
    }

    /// Runs the fit step.
    ///
    /// # Errors
    ///
    /// Returns [`HierarchyAlgoError`] on failure.
    pub fn fit(&mut self, input_matrix: &CsMat<f64>, force_bipartite: bool) -> Result<(), HierarchyAlgoError> {
        self.state.init_vars();
        let (adjacency, bip) = get_adjacency(
            MatrixInput::Sparse(input_matrix.to_owned()),
            true,
            force_bipartite,
            false,
            false,
        )
        .map_err(|_| HierarchyAlgoError::InvalidInput)?;
        self.bipartite = bip;
        let mut tree: Vec<Tree> = (0..adjacency.rows()).map(Tree::Leaf).collect();
        let mut current = adjacency.to_owned();
        loop {
            let labels = self
                .clustering
                .fit_predict(&current, false)
                .map_err(|_| HierarchyAlgoError::ClusteringFailed)?;
            tree = Self::compress_tree_level(tree, labels.as_slice().unwrap_or(&[]));
            let n_labels = labels
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let aggregate = self
                .clustering
                .aggregate()
                .ok_or(HierarchyAlgoError::ClusteringFailed)?;
            if n_labels <= 1 || aggregate.rows() == current.rows() {
                break;
            }
            current = aggregate;
        }
        let tree = if tree.len() == 1 {
            tree.remove(0)
        } else {
            Tree::Node(tree)
        };
        let mut dendrogram = get_dendrogram_with_reorder(&tree, false);
        if !dendrogram.is_empty() {
            let min_h = dendrogram.iter().map(|r| r[2]).fold(f64::INFINITY, f64::min);
            for row in &mut dendrogram {
                row[2] += 1.0 - min_h;
            }
        }
        self.state.dendrogram = Some(reorder_dendrogram(&dendrogram));
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
    use super::*;
    use crate::hierarchy::paris::Paris;
    use crate::data::test_graphs::{test_bigraph, test_digraph, test_graph};

    #[test]
    fn test_louvain_hierarchy_shapes() {
        for input in [test_graph(), test_digraph(), test_bigraph()] {
            let mut li = LouvainIteration::default();
            let d = li.fit_predict(&input, false).expect("li");
            assert_eq!(d.len(), input.rows().saturating_sub(1));

            let mut lh = LouvainHierarchy::default();
            let d = lh.fit_predict(&input, false).expect("lh");
            assert_eq!(d.len(), input.rows().saturating_sub(1));

            let mut p = Paris::default();
            let d = p.fit_predict(&input, false).expect("paris");
            assert_eq!(d.len(), input.rows().saturating_sub(1));
        }
    }

}
