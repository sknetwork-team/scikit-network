//! Embedding helpers for benchmark IPC (estimator ``fit_transform`` matrices).

use serde_json::{Value, json};
use sprs::CsMat;

use crate::embedding::louvain_embedding::LouvainEmbedding;
use crate::embedding::random_projection::RandomProjection;
use crate::embedding::spectral::Spectral;

fn embedding_json(embedding: &[Vec<f64>]) -> Value {
    json!(embedding)
}

fn return_embedding_requested(params: &Value) -> bool {
    params
        .get("return_embedding")
        .and_then(|v| v.as_bool())
        .unwrap_or(true)
}

fn model_params(params: &Value) -> Value {
    let mut out = params.clone();
    if let Some(obj) = out.as_object_mut() {
        obj.remove("return_embedding");
    }
    out
}

fn pack_embedding_response(embedding: &[Vec<f64>], return_embedding: bool) -> Value {
    if return_embedding {
        embedding_json(embedding)
    } else {
        json!({
            "n_rows": embedding.len(),
            "n_cols": embedding.first().map(|row| row.len()).unwrap_or(0),
        })
    }
}

/// Run an embedding estimator; returns a JSON matrix or shape-only payload.
pub fn embedding_dispatch(
    algorithm: &str,
    adj: &CsMat<f64>,
    params: &Value,
) -> Result<Value, String> {
    let return_embedding = return_embedding_requested(params);
    let params = model_params(params);
    match algorithm {
        "spectral" => {
            let n_components = params
                .get("n_components")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize;
            let decomposition = params
                .get("decomposition")
                .and_then(|v| v.as_str())
                .unwrap_or("rw");
            let regularization = params
                .get("regularization")
                .and_then(|v| v.as_f64())
                .unwrap_or(-1.0);
            let normalized = params
                .get("normalized")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let force_bipartite = params
                .get("force_bipartite")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let mut model = Spectral::new(n_components, decomposition, regularization, normalized);
            let embedding = model
                .fit_transform(adj, force_bipartite)
                .map_err(|e| format!("{e:?}"))?;
            Ok(pack_embedding_response(&embedding, return_embedding))
        }
        "random_projection" => {
            let n_components = params
                .get("n_components")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize;
            let alpha = params.get("alpha").and_then(|v| v.as_f64()).unwrap_or(0.5);
            let n_iter = params
                .get("n_iter")
                .and_then(|v| v.as_u64())
                .unwrap_or(3) as usize;
            let random_walk = params
                .get("random_walk")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let regularization = params
                .get("regularization")
                .and_then(|v| v.as_f64())
                .unwrap_or(-1.0);
            let normalized = params
                .get("normalized")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let force_bipartite = params
                .get("force_bipartite")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let mut model = RandomProjection::new(
                n_components,
                alpha,
                n_iter,
                random_walk,
                regularization,
                normalized,
            );
            let embedding = model
                .fit_transform(adj, force_bipartite)
                .map_err(|e| format!("{e:?}"))?;
            Ok(pack_embedding_response(&embedding, return_embedding))
        }
        "louvain_embedding" => {
            let isolated_nodes = params
                .get("isolated_nodes")
                .and_then(|v| v.as_str())
                .unwrap_or("remove");
            let force_bipartite = params
                .get("force_bipartite")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let mut model = LouvainEmbedding::new(isolated_nodes);
            let embedding = model
                .fit_transform(adj, force_bipartite)
                .map_err(|e| format!("{e:?}"))?;
            Ok(pack_embedding_response(&embedding, return_embedding))
        }
        other => Err(format!("unknown embedding algorithm '{other}'")),
    }
}
