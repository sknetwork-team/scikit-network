//! Classification helpers for benchmark IPC (``fit_predict`` label vectors).

use std::collections::HashMap;

use ndarray::Array1;
use serde_json::{Value, json};
use sprs::CsMat;

use crate::classification::diffusion::DiffusionClassifier;
use crate::classification::pagerank::PageRankClassifier;
use crate::classification::propagation::Propagation;
use crate::utils::values::ValuesInput;

fn labels_json(labels: &Array1<i32>) -> Value {
    json!(labels.iter().map(|&x| x as i64).collect::<Vec<_>>())
}

fn seeds_from_params(params: &Value) -> Result<Option<ValuesInput>, String> {
    let Some(seeds) = params.get("seeds") else {
        return Ok(None);
    };
    if let Some(obj) = seeds.as_object() {
        let mut map = HashMap::<usize, f64>::new();
        for (key, value) in obj {
            let node = key
                .parse::<usize>()
                .map_err(|_| format!("invalid seed node index '{key}'"))?;
            let label = value
                .as_f64()
                .ok_or_else(|| format!("invalid seed label for node {node}"))?;
            map.insert(node, label);
        }
        return Ok(Some(ValuesInput::Map(map)));
    }
    if let Some(arr) = seeds.as_array() {
        let mut vec = Vec::with_capacity(arr.len());
        for value in arr {
            vec.push(
                value
                    .as_f64()
                    .ok_or_else(|| "seed vector entries must be numeric".to_string())?,
            );
        }
        return Ok(Some(ValuesInput::Vector(vec)));
    }
    Err("seeds must be a JSON object or array".to_string())
}

fn model_params(params: &Value) -> Value {
    let mut out = params.clone();
    if let Some(obj) = out.as_object_mut() {
        obj.remove("seeds");
    }
    out
}

/// Run a classification estimator; returns a JSON array of per-node labels.
pub fn classification_dispatch(
    algorithm: &str,
    adj: &CsMat<f64>,
    params: &Value,
) -> Result<Value, String> {
    let seeds = seeds_from_params(params)?;
    let params = model_params(params);
    match algorithm {
        "propagation" => {
            let n_iter = params
                .get("n_iter")
                .and_then(|v| v.as_i64())
                .unwrap_or(-1) as i32;
            let node_order = params.get("node_order").and_then(|v| v.as_str());
            let weighted = params
                .get("weighted")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let mut model = Propagation::new(n_iter, node_order, weighted);
            let labels = model
                .fit_predict(adj, seeds, None, None)
                .map_err(|e| format!("{e:?}"))?;
            Ok(labels_json(&labels))
        }
        "pagerank" => {
            let damping_factor = params
                .get("damping_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.85);
            let n_iter = params
                .get("n_iter")
                .and_then(|v| v.as_u64())
                .unwrap_or(10) as usize;
            let tol = params.get("tol").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let mut model = PageRankClassifier::new(damping_factor, n_iter, tol)
                .map_err(|e| format!("{e:?}"))?;
            let labels = model
                .fit_predict(adj, seeds, None, None)
                .map_err(|e| format!("{e:?}"))?;
            Ok(labels_json(&labels))
        }
        "diffusion" => {
            let n_iter = params
                .get("n_iter")
                .and_then(|v| v.as_i64())
                .unwrap_or(10) as isize;
            let centering = params
                .get("centering")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let scale = params.get("scale").and_then(|v| v.as_f64()).unwrap_or(5.0);
            let mut model = DiffusionClassifier::new(n_iter, centering, scale)
                .map_err(|e| format!("{e:?}"))?;
            let labels = model
                .fit_predict(adj, seeds, None, None, false)
                .map_err(|e| format!("{e:?}"))?;
            Ok(labels_json(&labels))
        }
        other => Err(format!("unknown classification algorithm '{other}'")),
    }
}
