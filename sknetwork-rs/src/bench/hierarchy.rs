//! Hierarchy helpers for benchmark IPC (dendrogram outputs).

use serde_json::{Value, json};
use sprs::CsMat;

use crate::hierarchy::louvain_hierarchy::LouvainIteration;
use crate::hierarchy::paris::Paris;
use crate::hierarchy::metrics::{dasgupta_score, HierarchyMetricsError};
use crate::hierarchy::postprocess::Dendrogram;

fn dendrogram_json(dendrogram: &Dendrogram) -> Value {
    json!(
        dendrogram
            .iter()
            .map(|row| [row[0], row[1], row[2], row[3]])
            .collect::<Vec<_>>()
    )
}

fn return_dendrogram_requested(params: &Value) -> bool {
    params
        .get("return_dendrogram")
        .and_then(|v| v.as_bool())
        .unwrap_or(true)
}

fn model_params(params: &Value) -> Value {
    let mut out = params.clone();
    if let Some(obj) = out.as_object_mut() {
        obj.remove("return_dendrogram");
        obj.remove("force_bipartite");
    }
    out
}

fn force_bipartite(params: &Value) -> bool {
    params
        .get("force_bipartite")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn pack_dendrogram_response(
    dendrogram: &Dendrogram,
    return_dendrogram: bool,
    dasgupta: Option<f64>,
) -> Value {
    let mut out = json!({
        "n_merges": dendrogram.len(),
    });
    if return_dendrogram {
        out["dendrogram"] = dendrogram_json(dendrogram);
    }
    if let Some(score) = dasgupta {
        out["dasgupta_score"] = json!(score);
    }
    out
}

fn dasgupta_for_tree(adj: &CsMat<f64>, dendrogram: &Dendrogram, weights: &str) -> Result<f64, String> {
    dasgupta_score(adj, dendrogram, weights).map_err(|e| match e {
        HierarchyMetricsError::InvalidInput => "dasgupta: invalid input".to_string(),
        HierarchyMetricsError::UnknownWeights => "dasgupta: unknown weights".to_string(),
    })
}

/// Run a hierarchy estimator; returns dendrogram summary (and optional merges).
pub fn hierarchy_dispatch(
    algorithm: &str,
    adj: &CsMat<f64>,
    params: &Value,
) -> Result<Value, String> {
    let return_dendrogram = return_dendrogram_requested(params);
    let force_bip = force_bipartite(params);
    let params = model_params(params);
    match algorithm {
        "paris" => {
            let weights = params
                .get("weights")
                .and_then(|v| v.as_str())
                .unwrap_or("degree");
            let reorder = params
                .get("reorder")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let mut model = Paris::new(weights, reorder);
            let dendrogram = model
                .fit_predict(adj, force_bip)
                .map_err(|e| format!("{e:?}"))?;
            let score = dasgupta_for_tree(adj, &dendrogram, weights)?;
            Ok(pack_dendrogram_response(
                &dendrogram,
                return_dendrogram,
                Some(score),
            ))
        }
        "louvain_iteration" => {
            let depth = params
                .get("depth")
                .and_then(|v| v.as_i64())
                .unwrap_or(3) as isize;
            let resolution = params
                .get("resolution")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
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
            let mut model = LouvainIteration::new(
                depth,
                resolution,
                tol_optimization,
                tol_aggregation,
                n_aggregations,
                shuffle_nodes,
            );
            let dendrogram = model
                .fit_predict(adj, force_bip)
                .map_err(|e| format!("{e:?}"))?;
            let score = dasgupta_for_tree(adj, &dendrogram, "degree")?;
            Ok(pack_dendrogram_response(
                &dendrogram,
                return_dendrogram,
                Some(score),
            ))
        }
        other => Err(format!("unknown hierarchy algorithm '{other}'")),
    }
}
