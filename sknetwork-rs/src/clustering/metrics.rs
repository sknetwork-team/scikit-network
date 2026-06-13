//! Clustering evaluation metrics.

use std::collections::HashMap;

use ndarray::Array1;
use sprs::CsMat;

use crate::utils::check::{CheckError, WeightsInput, get_probs};
use crate::utils::format::{MatrixInput, get_adjacency};

/// Error type for clustering metric computations.
#[derive(Debug, Clone, PartialEq)]
pub enum ClusteringMetricError {
    /// Input matrix or weight distribution is invalid.
    InvalidInput(CheckError),
    /// Bipartite column labels are required but missing.
    MissingLabelsCol,
    /// Label vector length does not match the adjacency dimension.
    DimensionMismatch,
}

/// Decomposed modularity score with fit and diversity terms.
#[derive(Debug, Clone, PartialEq)]
pub struct ModularityResult {
    /// Modularity score (`fit - resolution * diversity`).
    pub modularity: f64,
    /// Fraction of edge weight inside clusters.
    pub fit: f64,
    /// Expected within-cluster edge weight under the null model.
    pub diversity: f64,
}

/// Computes the modularity score for a partition.
///
/// # Arguments
/// - `labels_col`: Column labels required for bipartite inputs.
/// - `weights`: Node weight distribution (`degree` or `uniform`).
/// - `resolution`: Resolution parameter scaling the null-model term.
///
/// # Errors
/// Returns [`ClusteringMetricError::InvalidInput`] for bad matrix formatting or
/// unknown weight distributions, [`ClusteringMetricError::MissingLabelsCol`] for
/// bipartite inputs without column labels, and
/// [`ClusteringMetricError::DimensionMismatch`] when label lengths mismatch.
pub fn get_modularity(
    input_matrix: &CsMat<f64>,
    labels: &Array1<i32>,
    labels_col: Option<&Array1<i32>>,
    weights: &str,
    resolution: f64,
) -> Result<f64, ClusteringMetricError> {
    Ok(get_modularity_all(input_matrix, labels, labels_col, weights, resolution)?.modularity)
}

/// Computes modularity with decomposed fit and diversity terms.
///
/// # Arguments
/// - `labels_col`: Column labels required for bipartite inputs.
/// - `weights`: Node weight distribution (`degree` or `uniform`).
/// - `resolution`: Resolution parameter scaling the null-model term.
///
/// # Errors
/// Returns [`ClusteringMetricError::InvalidInput`] for bad matrix formatting or
/// unknown weight distributions, [`ClusteringMetricError::MissingLabelsCol`] for
/// bipartite inputs without column labels, and
/// [`ClusteringMetricError::DimensionMismatch`] when label lengths mismatch.
pub fn get_modularity_all(
    input_matrix: &CsMat<f64>,
    labels: &Array1<i32>,
    labels_col: Option<&Array1<i32>>,
    weights: &str,
    resolution: f64,
) -> Result<ModularityResult, ClusteringMetricError> {
    let (adjacency, bipartite) = get_adjacency(
        MatrixInput::Sparse(input_matrix.to_owned()),
        true,
        false,
        false,
        false,
    )
    .map_err(ClusteringMetricError::InvalidInput)?;

    let labels_all = if bipartite {
        let col = labels_col.ok_or(ClusteringMetricError::MissingLabelsCol)?;
        let mut v = labels.to_vec();
        v.extend(col.iter().copied());
        Array1::from_vec(v)
    } else {
        labels.to_owned()
    };

    if labels_all.len() != adjacency.rows() {
        return Err(ClusteringMetricError::DimensionMismatch);
    }

    let probs_row = get_probs(
        WeightsInput::Distribution(weights.to_string()),
        &adjacency,
        false,
    )
    .map_err(ClusteringMetricError::InvalidInput)?;
    let probs_col = get_probs(
        WeightsInput::Distribution(weights.to_string()),
        &adjacency.transpose_view().to_csr(),
        false,
    )
    .map_err(ClusteringMetricError::InvalidInput)?;

    let total_weight: f64 = adjacency.data().iter().sum();
    let mut same_cluster_weight = 0.0;
    for (i, row) in adjacency.outer_iterator().enumerate() {
        if labels_all[i] < 0 {
            continue;
        }
        for (j, v) in row.iter() {
            if labels_all[j] >= 0 && labels_all[i] == labels_all[j] {
                same_cluster_weight += *v;
            }
        }
    }
    let fit = if total_weight > 0.0 {
        same_cluster_weight / total_weight
    } else {
        0.0
    };

    let mut mass_row: HashMap<i32, f64> = HashMap::new();
    let mut mass_col: HashMap<i32, f64> = HashMap::new();
    for i in 0..labels_all.len() {
        let lbl = labels_all[i];
        if lbl < 0 {
            continue;
        }
        *mass_row.entry(lbl).or_insert(0.0) += probs_row[i];
        *mass_col.entry(lbl).or_insert(0.0) += probs_col[i];
    }

    let mut diversity = 0.0;
    for (label, pr) in &mass_row {
        diversity += pr * mass_col.get(label).copied().unwrap_or(0.0);
    }

    Ok(ModularityResult {
        modularity: fit - resolution * diversity,
        fit,
        diversity,
    })
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use sprs::{CsMat, TriMat};

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
    fn test_api_contract() {
        let adjacency = test_graph();
        let n = adjacency.rows();
        let mut labels = Array1::<i32>::zeros(n);
        labels[0] = 1;
        let unique_cluster = Array1::<i32>::zeros(n);

        let all = get_modularity_all(&adjacency, &labels, None, "degree", 1.0).unwrap();
        let mod_only = get_modularity(&adjacency, &labels, None, "degree", 1.0).unwrap();
        assert!((all.fit - all.diversity - mod_only).abs() < 1e-12);
        assert!(
            get_modularity(&adjacency, &unique_cluster, None, "degree", 1.0)
                .unwrap()
                .abs()
                < 1e-12
        );

        assert!(matches!(
            get_modularity(
                &adjacency,
                &labels.slice(ndarray::s![..3]).to_owned(),
                None,
                "degree",
                1.0
            ),
            Err(ClusteringMetricError::DimensionMismatch)
        ));
    }

    #[test]
    fn test_bimodularity() {
        let biadjacency = star_wars();
        let labels_row = array![0, 0, 1, 1];
        let labels_col = array![0, 1, 0];
        let score =
            get_modularity(&biadjacency, &labels_row, Some(&labels_col), "degree", 1.0).unwrap();
        assert!((score - 0.12).abs() < 0.02);

        assert!(matches!(
            get_modularity(&biadjacency, &labels_row, None, "degree", 1.0),
            Err(ClusteringMetricError::MissingLabelsCol)
        ));
        assert!(matches!(
            get_modularity(
                &biadjacency,
                &labels_row.slice(ndarray::s![..2]).to_owned(),
                Some(&labels_col),
                "degree",
                1.0
            ),
            Err(ClusteringMetricError::DimensionMismatch)
        ));
        assert!(matches!(
            get_modularity(
                &biadjacency,
                &labels_row,
                Some(&labels_col.slice(ndarray::s![..2]).to_owned()),
                "degree",
                1.0
            ),
            Err(ClusteringMetricError::DimensionMismatch)
        ));
    }

    #[test]
    fn test_weights_and_resolution() {
        let adjacency = test_graph();
        let n = adjacency.rows();
        let labels = Array1::from_vec((0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect());

        let q_degree = get_modularity(&adjacency, &labels, None, "degree", 1.0).unwrap();
        let q_uniform = get_modularity(&adjacency, &labels, None, "uniform", 1.0).unwrap();
        assert!((q_degree - q_uniform).abs() > 1e-12);

        let q_r0 = get_modularity(&adjacency, &labels, None, "degree", 0.0).unwrap();
        let all = get_modularity_all(&adjacency, &labels, None, "degree", 1.0).unwrap();
        assert!((q_r0 - all.fit).abs() < 1e-12);
    }

    #[test]
    fn test_unknown_weights_error() {
        let adjacency = test_graph();
        let labels = Array1::zeros(adjacency.rows());
        assert!(matches!(
            get_modularity(&adjacency, &labels, None, "unknown", 1.0),
            Err(ClusteringMetricError::InvalidInput(
                CheckError::UnknownDistribution
            ))
        ));
    }
}
