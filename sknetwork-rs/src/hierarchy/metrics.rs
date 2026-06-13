use std::collections::HashMap;

use sprs::CsMat;

use crate::hierarchy::postprocess::Dendrogram;
use crate::utils::check::{
    CheckError, WeightsInput, check_min_nnz, check_min_size, check_square, get_probs,
};
use crate::utils::format::{MatrixInput, check_format, directed2undirected};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by hierarchy metrics error operations.
pub enum HierarchyMetricsError {
    /// Indicates invalid input.
    InvalidInput,
    /// Indicates unknown weights.
    UnknownWeights,
}

#[derive(Debug, Clone)]
struct AggregateGraphState {
    neighbors: Vec<HashMap<usize, f64>>,
    cluster_out_weights: Vec<f64>,
    cluster_in_weights: Vec<f64>,
    alive: Vec<bool>,
}

fn instantiate_vars(
    adjacency: &CsMat<f64>,
    weights: &str,
) -> Result<(AggregateGraphState, Vec<f64>, Vec<f64>), HierarchyMetricsError> {
    let w = match weights {
        "uniform" | "degree" => weights,
        _ => return Err(HierarchyMetricsError::UnknownWeights),
    };
    let wr = get_probs(
        WeightsInput::Distribution(w.to_string()),
        adjacency,
        false,
    )
    .map_err(|_| HierarchyMetricsError::InvalidInput)?
    .to_vec();
    let wc = get_probs(
        WeightsInput::Distribution(w.to_string()),
        &adjacency.transpose_view().to_owned(),
        false,
    )
    .map_err(|_| HierarchyMetricsError::InvalidInput)?
    .to_vec();
    let sym = directed2undirected(adjacency, true);
    let n = adjacency.rows();
    let mut neighbors = vec![HashMap::<usize, f64>::new(); n];
    for (i, row) in sym.outer_iterator().enumerate() {
        for (j, v) in row.iter() {
            neighbors[i].insert(j, *v);
        }
    }
    Ok((
        AggregateGraphState {
            neighbors,
            cluster_out_weights: wr.clone(),
            cluster_in_weights: wc.clone(),
            alive: vec![true; n],
        },
        wr,
        wc,
    ))
}

impl AggregateGraphState {
    fn merge(&mut self, i: usize, j: usize) {
        let new_id = self.neighbors.len();
        let mut map = HashMap::<usize, f64>::new();
        for (&k, &v) in &self.neighbors[i] {
            if self.alive.get(k).copied().unwrap_or(false) && k != i && k != j {
                *map.entry(k).or_insert(0.0) += v;
            }
        }
        for (&k, &v) in &self.neighbors[j] {
            if self.alive.get(k).copied().unwrap_or(false) && k != i && k != j {
                *map.entry(k).or_insert(0.0) += v;
            }
        }

        self.neighbors.push(HashMap::new());
        for (&k, &v) in &map {
            self.neighbors[new_id].insert(k, v);
            self.neighbors[k].insert(new_id, v);
            self.neighbors[k].remove(&i);
            self.neighbors[k].remove(&j);
        }
        self.neighbors[i].clear();
        self.neighbors[j].clear();
        self.alive.push(true);
        self.alive[i] = false;
        self.alive[j] = false;

        let out = self.cluster_out_weights[i] + self.cluster_out_weights[j];
        let inn = self.cluster_in_weights[i] + self.cluster_in_weights[j];
        self.cluster_out_weights.push(out);
        self.cluster_in_weights.push(inn);
    }
}

/// Returns sampling distributions.
pub fn get_sampling_distributions(
    adjacency: &CsMat<f64>,
    dendrogram: &Dendrogram,
    weights: &str,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), HierarchyMetricsError> {
    let n = adjacency.rows();
    if dendrogram.len() != n.saturating_sub(1) {
        return Err(HierarchyMetricsError::InvalidInput);
    }
    let (mut aggregate_graph, _, _) = instantiate_vars(adjacency, weights)?;
    let mut cluster_weight = vec![0.0; n - 1];
    let mut edge_sampling = vec![0.0; n - 1];
    let mut node_sampling = vec![0.0; n - 1];
    for t in 0..(n - 1) {
        let i = dendrogram[t][0] as usize;
        let j = dendrogram[t][1] as usize;
        if i >= aggregate_graph.neighbors.len() || j >= aggregate_graph.neighbors.len() {
            return Err(HierarchyMetricsError::InvalidInput);
        }
        if !aggregate_graph.alive.get(i).copied().unwrap_or(false)
            || !aggregate_graph.alive.get(j).copied().unwrap_or(false)
        {
            return Err(HierarchyMetricsError::InvalidInput);
        }
        if let Some(v) = aggregate_graph.neighbors[i].get(&j) {
            edge_sampling[t] += 2.0 * *v;
        }
        node_sampling[t] += aggregate_graph.cluster_out_weights[i] * aggregate_graph.cluster_in_weights[j]
            + aggregate_graph.cluster_out_weights[j] * aggregate_graph.cluster_in_weights[i];
        cluster_weight[t] = aggregate_graph.cluster_out_weights[i]
            + aggregate_graph.cluster_out_weights[j]
            + aggregate_graph.cluster_in_weights[i]
            + aggregate_graph.cluster_in_weights[j];
        for node in [i, j] {
            if node < n {
                node_sampling[t] +=
                    aggregate_graph.cluster_out_weights[node] * aggregate_graph.cluster_in_weights[node];
                if let Some(v) = aggregate_graph.neighbors[node].get(&node) {
                    edge_sampling[t] += *v;
                }
            }
        }
        aggregate_graph.merge(i, j);
    }
    Ok((
        edge_sampling,
        node_sampling,
        cluster_weight.into_iter().map(|x| x / 2.0).collect(),
    ))
}

/// Computes dasgupta cost.
pub fn dasgupta_cost(
    adjacency: &CsMat<f64>,
    dendrogram: &Dendrogram,
    weights: &str,
    normalized: bool,
) -> Result<f64, HierarchyMetricsError> {
    let adjacency = check_format(MatrixInput::Sparse(adjacency.to_owned()), true)
        .map_err(map_check_error)?;
    check_square(adjacency.shape()).map_err(map_check_error)?;
    check_min_size(adjacency.shape().0, 2).map_err(map_check_error)?;
    let n = adjacency.shape().0;
    let (edge_sampling, _, cluster_weight) = get_sampling_distributions(&adjacency, dendrogram, weights)?;
    let mut cost: f64 = edge_sampling
        .iter()
        .zip(cluster_weight.iter())
        .map(|(a, b)| a * b)
        .sum();
    if !normalized {
        if weights == "degree" {
            cost *= adjacency.data().iter().sum::<f64>();
        } else {
            cost *= n as f64;
        }
    }
    Ok(cost)
}

/// Computes dasgupta score.
pub fn dasgupta_score(
    adjacency: &CsMat<f64>,
    dendrogram: &Dendrogram,
    weights: &str,
) -> Result<f64, HierarchyMetricsError> {
    Ok(1.0 - dasgupta_cost(adjacency, dendrogram, weights, true)?)
}

/// Computes tree sampling divergence.
pub fn tree_sampling_divergence(
    adjacency: &CsMat<f64>,
    dendrogram: &Dendrogram,
    weights: &str,
    normalized: bool,
) -> Result<f64, HierarchyMetricsError> {
    let mut adjacency = check_format(MatrixInput::Sparse(adjacency.to_owned()), true)
        .map_err(map_check_error)?;
    check_square(adjacency.shape()).map_err(map_check_error)?;
    check_min_nnz(adjacency.nnz(), 1).map_err(map_check_error)?;
    check_min_size(adjacency.shape().0, 2).map_err(map_check_error)?;

    let total = adjacency.data().iter().sum::<f64>();
    if total > 0.0 {
        let (r, c) = adjacency.shape();
        let mut tri = sprs::TriMat::<f64>::new((r, c));
        for (i, row) in adjacency.outer_iterator().enumerate() {
            for (j, v) in row.iter() {
                tri.add_triplet(i, j, *v / total);
            }
        }
        adjacency = tri.to_csr::<usize>();
    }

    let (edge_sampling, node_sampling, _) = get_sampling_distributions(&adjacency, dendrogram, weights)?;
    let mut score = 0.0;
    for (e, n) in edge_sampling.iter().zip(node_sampling.iter()) {
        if *e > 0.0 && *n > 0.0 {
            score += *e * (*e / *n).ln();
        }
    }
    if normalized {
        let wr = get_probs(
            WeightsInput::Distribution(weights.to_string()),
            &adjacency,
            false,
        )
        .map_err(map_check_error)?
        .to_vec();
        let wc = get_probs(
            WeightsInput::Distribution(weights.to_string()),
            &adjacency.transpose_view().to_owned(),
            false,
        )
        .map_err(map_check_error)?
        .to_vec();
        let mut mi = 0.0;
        for (i, row) in adjacency.outer_iterator().enumerate() {
            for (j, v) in row.iter() {
                let denom = wr[i] * wc[j];
                if *v > 0.0 && denom > 0.0 {
                    mi += *v * (*v / denom).ln();
                }
            }
        }
        if mi > 0.0 {
            score /= mi;
        }
    }
    Ok(score)
}

fn map_check_error(_: CheckError) -> HierarchyMetricsError {
    HierarchyMetricsError::InvalidInput
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::{test_digraph, test_graph};

    fn sample_dendrogram(n: usize) -> Dendrogram {
        let mut d = Vec::new();
        let mut a = 0usize;
        let mut b = 1usize;
        let mut next = n;
        for _ in 0..(n - 1) {
            d.push([a as f64, b as f64, (next - n) as f64, 2.0]);
            a = next;
            b += 1;
            next += 1;
            if b >= n {
                break;
            }
        }
        while d.len() < n - 1 {
            let t = d.len();
            d.push([n as f64 + t as f64 - 1.0, (t + 1) as f64, t as f64, 2.0]);
        }
        d
    }

    #[test]
    fn test_metrics_run() {
        for adjacency in [test_graph(), test_digraph()] {
            let d = sample_dendrogram(adjacency.rows());
            let c = dasgupta_cost(&adjacency, &d, "uniform", false).expect("cost");
            let s = dasgupta_score(&adjacency, &d, "uniform").expect("score");
            let t = tree_sampling_divergence(&adjacency, &d, "degree", true).expect("tsd");
            assert!(c >= 0.0);
            assert!(s.is_finite());
            assert!(t.is_finite());
        }
    }
}
