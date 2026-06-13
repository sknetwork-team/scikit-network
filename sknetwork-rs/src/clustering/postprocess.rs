//! Post-processing utilities for cluster labels and aggregate graphs.

use std::collections::HashMap;

use ndarray::Array1;
use sprs::{CsMat, TriMat};

/// Error type for clustering post-processing routines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusteringError {
    /// Required label vectors are missing.
    MissingLabels,
}

/// Reindexes cluster labels by descending cluster size.
///
/// Ties preserve the first occurrence order of each original label.
pub fn reindex_labels(labels: &Array1<i32>) -> Array1<i32> {
    let mut counts: HashMap<i32, usize> = HashMap::new();
    let mut first_pos: HashMap<i32, usize> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        *counts.entry(label).or_insert(0) += 1;
        first_pos.entry(label).or_insert(idx);
    }

    let mut order: Vec<(i32, usize, usize)> = counts
        .iter()
        .map(|(&label, &count)| (label, count, *first_pos.get(&label).unwrap_or(&usize::MAX)))
        .collect();
    // np.argsort(-counts) is stable; preserve first occurrence for ties.
    order.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));

    let mut new_index = HashMap::new();
    for (new_label, (old_label, _, _)) in order.iter().enumerate() {
        new_index.insert(*old_label, new_label as i32);
    }

    Array1::from_vec(
        labels
            .iter()
            .map(|label| *new_index.get(label).unwrap_or(&-1))
            .collect(),
    )
}

fn n_clusters(labels: &Array1<i32>) -> usize {
    labels
        .iter()
        .filter(|&&x| x >= 0)
        .map(|&x| x as usize)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
}

/// Aggregates an adjacency or biadjacency matrix by cluster labels.
///
/// # Arguments
/// - `labels`: Unipartite labels when `labels_row` is `None`.
/// - `labels_row`: Optional row labels for bipartite aggregation.
/// - `labels_col`: Optional column labels; defaults to `labels_row`.
///
/// # Errors
/// Returns [`ClusteringError::MissingLabels`] when no label vector is provided.
pub fn aggregate_graph(
    input_matrix: &CsMat<f64>,
    labels: Option<&Array1<i32>>,
    labels_row: Option<&Array1<i32>>,
    labels_col: Option<&Array1<i32>>,
) -> Result<CsMat<f64>, ClusteringError> {
    let row_labels = match labels_row {
        Some(l) => l,
        None => labels.ok_or(ClusteringError::MissingLabels)?,
    };

    let col_labels = labels_col.unwrap_or(row_labels);
    let n_row_clusters = n_clusters(row_labels);
    let n_col_clusters = n_clusters(col_labels);

    let mut acc: HashMap<(usize, usize), f64> = HashMap::new();
    for (i, row) in input_matrix.outer_iterator().enumerate() {
        let Some(&ri_raw) = row_labels.get(i) else {
            continue;
        };
        if ri_raw < 0 {
            continue;
        }
        let ri = ri_raw as usize;
        for (j, v) in row.iter() {
            let Some(&cj_raw) = col_labels.get(j) else {
                continue;
            };
            if cj_raw < 0 {
                continue;
            }
            let cj = cj_raw as usize;
            *acc.entry((ri, cj)).or_insert(0.0) += *v;
        }
    }

    let mut tri = TriMat::<f64>::new((n_row_clusters, n_col_clusters));
    for ((i, j), v) in acc {
        if v != 0.0 {
            tri.add_triplet(i, j, v);
        }
    }
    Ok(tri.to_csr::<usize>())
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use sprs::{CsMat, TriMat};

    use super::*;

    fn house() -> CsMat<f64> {
        let mut tri = TriMat::<f64>::new((5, 5));
        let edges = [(0usize, 1usize), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)];
        for (u, v) in edges {
            tri.add_triplet(u, v, 1.0);
            tri.add_triplet(v, u, 1.0);
        }
        tri.to_csr::<usize>()
    }

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
    fn test_reindex_clusters() {
        let truth = array![1, 1, 2, 0, 0, 0];

        let labels = array![0, 0, 1, 2, 2, 2];
        let output = reindex_labels(&labels);
        assert_eq!(output.to_vec(), truth.to_vec());

        let labels = array![0, 0, 5, 2, 2, 2];
        let output = reindex_labels(&labels);
        assert_eq!(output.to_vec(), truth.to_vec());
    }

    #[test]
    fn test_aggregate_graph() {
        let adjacency = house();
        let labels = array![0, 0, 1, 1, 2];
        let aggregate = aggregate_graph(&adjacency, Some(&labels), None, None).unwrap();
        assert_eq!(aggregate.shape(), (3, 3));

        let biadjacency = star_wars();
        let labels = array![0, 0, 1, 2];
        let labels_row = array![0, 1, 3, -1];
        let labels_col = array![0, 0, 1];
        let aggregate =
            aggregate_graph(&biadjacency, Some(&labels), None, Some(&labels_col)).unwrap();
        assert_eq!(aggregate.shape(), (3, 2));

        let aggregate =
            aggregate_graph(&biadjacency, None, Some(&labels_row), Some(&labels_col)).unwrap();
        assert_eq!(aggregate.shape(), (4, 2));
    }
}
