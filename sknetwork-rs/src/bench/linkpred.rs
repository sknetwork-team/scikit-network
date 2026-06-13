//! Link-prediction helpers for benchmark IPC (``fit_predict`` link matrices).

use serde_json::{Value, json};
use sprs::CsMat;

use crate::embedding::spectral::Spectral;
use crate::linkpred::nn::{EmbeddingMethod, NNLinker};

fn mat_sorted_edges(mat: &CsMat<f64>) -> Vec<[f64; 3]> {
    let mut edges: Vec<[f64; 3]> = Vec::with_capacity(mat.nnz());
    for (r, row) in mat.outer_iterator().enumerate() {
        for (c, v) in row.iter() {
            edges.push([r as f64, c as f64, *v]);
        }
    }
    edges.sort_by(|a, b| {
        a[0]
            .partial_cmp(&b[0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
    });
    edges
}

fn return_links_requested(params: &Value) -> bool {
    params
        .get("return_links")
        .and_then(|v| v.as_bool())
        .unwrap_or(true)
}

fn model_params(params: &Value) -> Value {
    let mut out = params.clone();
    if let Some(obj) = out.as_object_mut() {
        obj.remove("return_links");
        obj.remove("index");
    }
    out
}

fn index_from_params(params: &Value) -> Result<Option<Vec<usize>>, String> {
    let Some(idx) = params.get("index") else {
        return Ok(None);
    };
    let arr = idx
        .as_array()
        .ok_or_else(|| "index must be a JSON array".to_string())?;
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        out.push(
            v.as_u64()
                .ok_or_else(|| "index entries must be non-negative integers".to_string())?
                as usize,
        );
    }
    Ok(Some(out))
}

fn parse_n_neighbors(params: &Value) -> Result<Option<usize>, String> {
    match params.get("n_neighbors") {
        None => Ok(Some(10)),
        Some(v) if v.is_null() => Ok(None),
        Some(v) => v
            .as_u64()
            .map(|n| Some(n as usize))
            .ok_or_else(|| "n_neighbors must be a non-negative integer or null".to_string()),
    }
}

fn parse_embedding(params: &Value) -> Result<Option<EmbeddingMethod>, String> {
    let Some(spec) = params.get("embedding") else {
        return Ok(None);
    };
    if spec.is_null() {
        return Ok(None);
    }
    let obj = spec
        .as_object()
        .ok_or_else(|| "embedding must be null or an object".to_string())?;
    let method = obj
        .get("method")
        .and_then(|v| v.as_str())
        .unwrap_or("spectral");
    match method {
        "spectral" => {
            let n_components = obj
                .get("n_components")
                .and_then(|v| v.as_u64())
                .unwrap_or(5) as usize;
            let decomposition = obj
                .get("decomposition")
                .and_then(|v| v.as_str())
                .unwrap_or("rw");
            let regularization = obj
                .get("regularization")
                .and_then(|v| v.as_f64())
                .unwrap_or(-1.0);
            let normalized = obj
                .get("normalized")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            Ok(Some(EmbeddingMethod::Spectral(Spectral::new(
                n_components,
                decomposition,
                regularization,
                normalized,
            ))))
        }
        other => Err(format!("unknown linkpred embedding method '{other}'")),
    }
}

fn pack_links_response(links: &CsMat<f64>, return_links: bool) -> Value {
    let mut out = json!({
        "n_rows": links.rows(),
        "n_cols": links.cols(),
        "nnz": links.nnz(),
    });
    if return_links {
        out["edges"] = json!(mat_sorted_edges(links));
    }
    out
}

/// Run a link-prediction estimator; returns link-matrix summary (and optional edges).
pub fn linkpred_dispatch(
    algorithm: &str,
    adj: &CsMat<f64>,
    params: &Value,
) -> Result<Value, String> {
    let return_links = return_links_requested(params);
    let index = index_from_params(params)?;
    let params = model_params(params);
    match algorithm {
        "nn" => {
            let n_neighbors = parse_n_neighbors(&params)?;
            let threshold = params
                .get("threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let embedding = parse_embedding(&params)?;
            let mut model = NNLinker::new(n_neighbors, threshold, embedding);
            let links = model
                .fit_predict(adj, index.as_deref())
                .map_err(|e| format!("{e:?}"))?;
            Ok(pack_links_response(&links, return_links))
        }
        other => Err(format!("unknown linkpred algorithm '{other}'")),
    }
}
