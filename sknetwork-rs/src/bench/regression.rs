//! Regression helpers for benchmark IPC (``fit_predict`` value vectors).

use std::collections::HashMap;

use serde_json::{Value, json};
use sprs::CsMat;

use crate::regression::diffusion::Diffusion;
use crate::utils::values::ValuesInput;

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
            let v = value
                .as_f64()
                .ok_or_else(|| format!("invalid seed value for node {node}"))?;
            map.insert(node, v);
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

/// Run a regression estimator; returns a JSON array of per-node values.
pub fn regression_dispatch(
    algorithm: &str,
    adj: &CsMat<f64>,
    params: &Value,
) -> Result<Value, String> {
    let seeds = seeds_from_params(params)?;
    let params = model_params(params);
    match algorithm {
        "diffusion" => {
            let n_iter = params
                .get("n_iter")
                .and_then(|v| v.as_i64())
                .unwrap_or(3) as isize;
            let damping_factor = params
                .get("damping_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);
            let mut model = Diffusion::new(n_iter, damping_factor).map_err(|e| format!("{e:?}"))?;
            let values = model
                .fit_predict(adj, seeds, None, None, None, false)
                .map_err(|e| format!("{e:?}"))?;
            Ok(json!(values))
        }
        other => Err(format!("unknown regression algorithm '{other}'")),
    }
}
