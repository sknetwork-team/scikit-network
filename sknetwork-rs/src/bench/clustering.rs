//! Clustering helpers for benchmark IPC (estimator ``fit_predict`` label vectors).

use serde_json::{Value, json};
use sprs::CsMat;

use crate::clustering::kcenters::KCenters;
use crate::clustering::leiden::Leiden;
use crate::clustering::louvain::Louvain;
use crate::clustering::metrics::get_modularity;
use crate::clustering::propagation_clustering::PropagationClustering;

fn labels_json(labels: &ndarray::Array1<i32>) -> Value {
    json!(labels.iter().map(|&x| x as i64).collect::<Vec<_>>())
}

fn pack_labels_only(labels: &ndarray::Array1<i32>) -> Value {
    json!({"labels": labels_json(labels)})
}

fn pack_labels_with_modularity(
    adj: &CsMat<f64>,
    labels: &ndarray::Array1<i32>,
    resolution: f64,
) -> Result<Value, String> {
    let q = get_modularity(adj, labels, None, "degree", resolution).map_err(|e| format!("{e:?}"))?;
    Ok(json!({
        "labels": labels_json(labels),
        "modularity": q,
    }))
}

/// Run a clustering estimator; returns a JSON array of per-node cluster labels.
pub fn clustering_dispatch(
    algorithm: &str,
    adj: &CsMat<f64>,
    params: &Value,
) -> Result<Value, String> {
    match algorithm {
        "louvain" => {
            let resolution = params
                .get("resolution")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            let modularity = params
                .get("modularity")
                .and_then(|v| v.as_str())
                .unwrap_or("dugue");
            let tol_optimization = params
                .get("tol_optimization")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-3);
            let tol_aggregation = params
                .get("tol_aggregation")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-3);
            let n_aggregations = params
                .get("n_aggregations")
                .and_then(|v| v.as_i64())
                .unwrap_or(-1) as isize;
            let shuffle_nodes = params
                .get("shuffle_nodes")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let sort_clusters = params
                .get("sort_clusters")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let return_probs = params
                .get("return_probs")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let return_aggregate = params
                .get("return_aggregate")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let mut model = Louvain::new(
                resolution,
                modularity,
                tol_optimization,
                tol_aggregation,
                n_aggregations,
                shuffle_nodes,
                sort_clusters,
                return_probs,
                return_aggregate,
            );
            let labels = model
                .fit_predict(adj, false)
                .map_err(|e| format!("{e:?}"))?;
            pack_labels_with_modularity(adj, &labels, resolution)
        }
        "leiden" => {
            let resolution = params
                .get("resolution")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            let modularity = params
                .get("modularity")
                .and_then(|v| v.as_str())
                .unwrap_or("dugue");
            let tol_optimization = params
                .get("tol_optimization")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-3);
            let tol_aggregation = params
                .get("tol_aggregation")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-3);
            let n_aggregations = params
                .get("n_aggregations")
                .and_then(|v| v.as_i64())
                .unwrap_or(-1) as isize;
            let shuffle_nodes = params
                .get("shuffle_nodes")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let sort_clusters = params
                .get("sort_clusters")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let return_probs = params
                .get("return_probs")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let return_aggregate = params
                .get("return_aggregate")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let mut model = Leiden::new(
                resolution,
                modularity,
                tol_optimization,
                tol_aggregation,
                n_aggregations,
                shuffle_nodes,
                sort_clusters,
                return_probs,
                return_aggregate,
            );
            let labels = model
                .fit_predict(adj, false)
                .map_err(|e| format!("{e:?}"))?;
            pack_labels_with_modularity(adj, &labels, resolution)
        }
        "propagation_clustering" => {
            let n_iter = params
                .get("n_iter")
                .and_then(|v| v.as_i64())
                .unwrap_or(5) as isize;
            let node_order = params
                .get("node_order")
                .and_then(|v| v.as_str())
                .unwrap_or("decreasing");
            let weighted = params
                .get("weighted")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let sort_clusters = params
                .get("sort_clusters")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let return_probs = params
                .get("return_probs")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let return_aggregate = params
                .get("return_aggregate")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let mut model = PropagationClustering::new(
                n_iter,
                node_order,
                weighted,
                sort_clusters,
                return_probs,
                return_aggregate,
            );
            let labels = model.fit_predict(adj).map_err(|e| format!("{e:?}"))?;
            Ok(pack_labels_only(&labels))
        }
        "kcenters" => {
            let n_clusters = params
                .get("n_clusters")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| "kcenters: n_clusters required".to_string())? as usize;
            let directed = params
                .get("directed")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let center_position = params
                .get("center_position")
                .and_then(|v| v.as_str())
                .unwrap_or("row");
            let n_init = params
                .get("n_init")
                .and_then(|v| v.as_u64())
                .unwrap_or(5) as usize;
            let max_iter = params
                .get("max_iter")
                .and_then(|v| v.as_u64())
                .unwrap_or(20) as usize;
            let mut model = KCenters::new(n_clusters, directed, center_position, n_init, max_iter);
            let labels = model
                .fit_predict(adj, false)
                .map_err(|e| format!("{e:?}"))?;
            Ok(pack_labels_only(&labels))
        }
        other => Err(format!("unknown clustering algorithm '{other}'")),
    }
}
