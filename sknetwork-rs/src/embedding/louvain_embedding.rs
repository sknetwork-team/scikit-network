use std::collections::HashMap;

use sprs::CsMat;

use crate::clustering::louvain::Louvain;
use crate::utils::format::{MatrixInput, get_adjacency};

/// Errors raised by [`LouvainEmbedding`].
#[derive(Debug, Clone, PartialEq)]
pub enum LouvainEmbeddingError {
    /// The input matrix is empty or Louvain clustering failed.
    InvalidInput,
    /// `isolated_nodes` is not `remove`, `merge`, or `keep`.
    UnknownIsolatedNodes,
}

fn reindex_labels(
    labels: &[i32],
    labels_secondary: Option<&[i32]>,
    which: &str,
) -> Result<(Vec<i32>, Option<Vec<i32>>), LouvainEmbeddingError> {
    if !matches!(which, "remove" | "merge" | "keep") {
        return Err(LouvainEmbeddingError::UnknownIsolatedNodes);
    }
    let mut counts = HashMap::<i32, usize>::new();
    for &x in labels {
        *counts.entry(x).or_insert(0) += 1;
    }
    let mut keep: Vec<i32> = counts
        .iter()
        .filter_map(|(&k, &v)| if v > 1 { Some(k) } else { None })
        .collect();
    keep.sort_unstable();
    let mut keep_index = HashMap::<i32, i32>::new();
    for (i, &k) in keep.iter().enumerate() {
        keep_index.insert(k, i as i32);
    }
    let default_merge = keep.len() as i32;

    let mut mapped = vec![-1; labels.len()];
    for (i, &x) in labels.iter().enumerate() {
        mapped[i] = match which {
            "remove" => *keep_index.get(&x).unwrap_or(&-1),
            "merge" => *keep_index.get(&x).unwrap_or(&default_merge),
            "keep" => x.max(0),
            _ => unreachable!("isolated_nodes validated before mapping"),
        };
    }

    let mapped_secondary = labels_secondary.map(|sec| {
        sec.iter()
            .map(|&x| *keep_index.get(&x).unwrap_or(&-1))
            .collect::<Vec<_>>()
    });
    Ok((mapped, mapped_secondary))
}

fn row_normalize(adjacency: &CsMat<f64>) -> Vec<Vec<f64>> {
    let (n_row, n_col) = adjacency.shape();
    let mut out = vec![vec![0.0; n_col]; n_row];
    for i in 0..n_row {
        if let Some(row) = adjacency.outer_view(i) {
            let s: f64 = row.data().iter().sum();
            if s > 0.0 {
                for (j, v) in row.iter() {
                    out[i][j] = v / s;
                }
            }
        }
    }
    out
}

fn membership(labels: &[i32]) -> Vec<Vec<f64>> {
    let valid: Vec<i32> = labels.iter().copied().filter(|x| *x >= 0).collect();
    let n_labels = valid
        .iter()
        .copied()
        .max()
        .map(|x| x as usize + 1)
        .unwrap_or(0);
    let n = labels.len();
    let mut out = vec![vec![0.0; n_labels]; n];
    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            out[i][label as usize] = 1.0;
        }
    }
    out
}

fn dense_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }
    let k = b.first().map(|r| r.len()).unwrap_or(0);
    let mut out = vec![vec![0.0; k]; n];
    for i in 0..n {
        for j in 0..a[i].len() {
            let v = a[i][j];
            if v == 0.0 {
                continue;
            }
            for c in 0..k {
                out[i][c] += v * b[j][c];
            }
        }
    }
    out
}

/// Louvain-cluster membership embedding estimator.
#[derive(Debug, Clone)]
pub struct LouvainEmbedding {
    /// Policy for singleton clusters: `remove`, `merge`, or `keep`.
    pub isolated_nodes: String,
    /// Fitted Louvain cluster labels per node.
    pub labels: Vec<i32>,
    /// Fitted node embedding (row nodes for bipartite inputs).
    pub embedding: Vec<Vec<f64>>,
    /// Row-node embedding when the input is bipartite.
    pub embedding_row: Option<Vec<Vec<f64>>>,
    /// Column-node embedding when the input is bipartite.
    pub embedding_col: Option<Vec<Vec<f64>>>,
}

impl Default for LouvainEmbedding {
    fn default() -> Self {
        Self {
            isolated_nodes: "remove".to_string(),
            labels: Vec::new(),
            embedding: Vec::new(),
            embedding_row: None,
            embedding_col: None,
        }
    }
}

impl LouvainEmbedding {
    /// Creates an estimator with the given isolated-node policy.
    ///
    /// # Arguments
    /// - `isolated_nodes`: One of `remove`, `merge`, or `keep` (case-insensitive).
    pub fn new(isolated_nodes: &str) -> Self {
        Self {
            isolated_nodes: isolated_nodes.to_lowercase(),
            ..Self::default()
        }
    }

    /// Fits Louvain labels and builds membership embeddings.
    ///
    /// # Arguments
    /// - `input_matrix`: Sparse adjacency or biadjacency input.
    /// - `force_bipartite`: Treat the input as bipartite even when square.
    ///
    /// # Errors
    /// Returns:
    /// - [`LouvainEmbeddingError::InvalidInput`] for empty inputs or clustering failure.
    /// - [`LouvainEmbeddingError::UnknownIsolatedNodes`] for invalid `isolated_nodes`.
    pub fn fit(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<(), LouvainEmbeddingError> {
        let n_row = input_matrix.rows();
        let n_col = input_matrix.cols();
        if n_row == 0 || n_col == 0 {
            return Err(LouvainEmbeddingError::InvalidInput);
        }

        let treat_bip = force_bipartite || n_row != n_col;
        if !treat_bip {
            let mut louvain = Louvain::default();
            let labels = louvain
                .fit_predict(input_matrix, false)
                .map_err(|_| LouvainEmbeddingError::InvalidInput)?
                .to_vec();
            let (labels, _) = reindex_labels(&labels, None, &self.isolated_nodes)?;
            self.labels = labels.clone();
            let probs = row_normalize(input_matrix);
            let mem = membership(&labels);
            self.embedding = dense_mul(&probs, &mem);
            self.embedding_row = None;
            self.embedding_col = None;
        } else {
            let (adjacency, _) = get_adjacency(
                MatrixInput::Sparse(input_matrix.to_owned()),
                true,
                true,
                false,
                false,
            )
            .map_err(|_| LouvainEmbeddingError::InvalidInput)?;
            let mut louvain = Louvain::default();
            let labels_all = louvain
                .fit_predict(&adjacency, false)
                .map_err(|_| LouvainEmbeddingError::InvalidInput)?
                .to_vec();
            let labels_row = labels_all[..n_row].to_vec();
            let labels_col = labels_all[n_row..].to_vec();
            let (labels, labels_secondary) =
                reindex_labels(&labels_col, Some(&labels_row), &self.isolated_nodes)?;
            self.labels = labels.clone();

            let probs = row_normalize(input_matrix);
            let mem = membership(&labels);
            let embedding_row = dense_mul(&probs, &mem);

            let probs_t = row_normalize(&input_matrix.transpose_view().to_csr());
            let mem_row = membership(
                labels_secondary
                    .as_ref()
                    .map(|x| x.as_slice())
                    .unwrap_or(&labels_row),
            );
            let embedding_col = dense_mul(&probs_t, &mem_row);

            self.embedding = embedding_row.clone();
            self.embedding_row = Some(embedding_row);
            self.embedding_col = Some(embedding_col);
        }
        Ok(())
    }

    /// Fits the estimator and returns the node embedding.
    ///
    /// # Errors
    /// Returns the same errors as [`LouvainEmbedding::fit`].
    pub fn fit_transform(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<Vec<Vec<f64>>, LouvainEmbeddingError> {
        self.fit(input_matrix, force_bipartite)?;
        Ok(self.embedding.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::{test_bigraph, test_graph};

    #[test]
    fn test_predict_contract() {
        let adjacency = test_graph();
        let mut louvain = LouvainEmbedding::default();
        louvain.fit(&adjacency, false).unwrap();
        assert_eq!(louvain.embedding.len(), 10);
        louvain.fit(&adjacency, true).unwrap();
        assert_eq!(louvain.embedding.len(), 10);

        let biadjacency = test_bigraph();
        louvain.fit(&biadjacency, false).unwrap();
        assert_eq!(louvain.embedding_row.clone().unwrap_or_default().len(), 6);
        assert_eq!(louvain.embedding_col.clone().unwrap_or_default().len(), 8);

        for method in ["remove", "merge", "keep"] {
            let mut louvain = LouvainEmbedding::new(method);
            let embedding = louvain.fit_transform(&adjacency, false).unwrap();
            assert_eq!(embedding.len(), adjacency.rows());
        }
    }

    #[test]
    fn test_unknown_isolated_nodes_rejected() {
        let adjacency = test_graph();
        let mut louvain = LouvainEmbedding::new("bad-option");
        assert_eq!(
            louvain.fit(&adjacency, false),
            Err(LouvainEmbeddingError::UnknownIsolatedNodes)
        );
    }

    #[test]
    fn test_isolated_nodes_case_insensitive() {
        let adjacency = test_graph();
        let mut louvain = LouvainEmbedding::new("MeRgE");
        let embedding = louvain.fit_transform(&adjacency, false).unwrap();
        assert_eq!(embedding.len(), adjacency.rows());
    }
}
