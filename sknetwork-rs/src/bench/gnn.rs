//! GNN helpers for benchmark IPC (semi-supervised node classification).

use ndarray::{Array1, Array2};
use serde_json::{Value, json};
use sprs::CsMat;

use crate::gnn::base::BaseGNN;
use crate::gnn::gnn_classifier::GNNClassifier;

fn return_labels_requested(params: &Value) -> bool {
    params
        .get("return_labels")
        .and_then(|v| v.as_bool())
        .unwrap_or(true)
}

fn model_params(params: &Value) -> Value {
    let mut out = params.clone();
    if let Some(obj) = out.as_object_mut() {
        obj.remove("return_labels");
    }
    out
}

fn array2_from_json(value: &Value) -> Result<Array2<f64>, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| "features must be a JSON array of rows".to_string())?;
    if rows.is_empty() {
        return Ok(Array2::<f64>::zeros((0, 0)));
    }
    let n_rows = rows.len();
    let n_cols = rows[0]
        .as_array()
        .ok_or_else(|| "features rows must be numeric arrays".to_string())?
        .len();
    let mut out = Array2::<f64>::zeros((n_rows, n_cols));
    for (i, row) in rows.iter().enumerate() {
        let arr = row
            .as_array()
            .ok_or_else(|| format!("features row {i} must be an array"))?;
        if arr.len() != n_cols {
            return Err(format!(
                "features row {i} has {} cols, expected {n_cols}",
                arr.len()
            ));
        }
        for (j, cell) in arr.iter().enumerate() {
            out[[i, j]] = cell
                .as_f64()
                .ok_or_else(|| format!("features[{i},{j}] must be numeric"))?;
        }
    }
    Ok(out)
}

fn labels_from_json(value: &Value, n_nodes: usize) -> Result<Array1<i32>, String> {
    if let Some(arr) = value.as_array() {
        if arr.len() != n_nodes {
            return Err(format!(
                "labels vector length {} != n_nodes {n_nodes}",
                arr.len()
            ));
        }
        let mut out = Array1::<i32>::zeros(n_nodes);
        for (i, cell) in arr.iter().enumerate() {
            let v = cell
                .as_i64()
                .ok_or_else(|| format!("labels[{i}] must be integer"))?;
            out[i] = v as i32;
        }
        return Ok(out);
    }
    if let Some(obj) = value.as_object() {
        let mut out = Array1::<i32>::from_elem(n_nodes, -1);
        for (key, cell) in obj {
            let i = key
                .parse::<usize>()
                .map_err(|_| format!("invalid label node index '{key}'"))?;
            if i >= n_nodes {
                return Err(format!("label node index {i} out of range (n={n_nodes})"));
            }
            let v = cell
                .as_i64()
                .ok_or_else(|| format!("invalid label for node {i}"))?;
            out[i] = v as i32;
        }
        return Ok(out);
    }
    Err("labels must be a JSON array or object".to_string())
}

fn labels_json(labels: &Array1<i32>) -> Value {
    json!(labels.iter().map(|&x| x as i64).collect::<Vec<_>>())
}

/// Run a GNN estimator; returns predicted labels and training summary.
pub fn gnn_dispatch(
    algorithm: &str,
    adj: &CsMat<f64>,
    params: &Value,
) -> Result<Value, String> {
    let return_labels = return_labels_requested(params);
    let features_val = params
        .get("features")
        .ok_or_else(|| "missing features matrix".to_string())?;
    let labels_val = params
        .get("labels")
        .ok_or_else(|| "missing labels".to_string())?;
    let params = model_params(params);
    match algorithm {
        "gnn_classifier" => {
            let features = array2_from_json(features_val)?;
            let labels = labels_from_json(labels_val, adj.rows())?;
            if features.nrows() != adj.rows() {
                return Err(format!(
                    "features rows {} != adjacency rows {}",
                    features.nrows(),
                    adj.rows()
                ));
            }

            let dims: Vec<usize> = params
                .get("dims")
                .and_then(|v| v.as_array())
                .ok_or_else(|| "dims must be a JSON array".to_string())?
                .iter()
                .map(|x| {
                    x.as_u64()
                        .ok_or_else(|| "dims entries must be unsigned integers".to_string())
                        .map(|n| n as usize)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let layer_name = params
                .get("layer_name")
                .or_else(|| params.get("layer_types"))
                .and_then(|v| v.as_str())
                .unwrap_or("gcn");
            let loss = params
                .get("loss")
                .and_then(|v| v.as_str())
                .unwrap_or("cross_entropy");
            let optimizer = params
                .get("optimizer")
                .and_then(|v| v.as_str())
                .unwrap_or("adam");
            let learning_rate = params
                .get("learning_rate")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.01);
            let n_epochs = params
                .get("n_epochs")
                .and_then(|v| v.as_u64())
                .unwrap_or(3) as usize;
            let sample_size = params
                .get("sample_size")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize);
            let sample_sizes = params.get("sample_sizes").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .map(|x| {
                        x.as_u64()
                            .ok_or_else(|| "sample_sizes entries must be unsigned integers".to_string())
                            .map(|n| n as usize)
                    })
                    .collect::<Result<Vec<_>, _>>()
            });
            let sample_sizes = match sample_sizes {
                Some(Ok(v)) => Some(v),
                Some(Err(e)) => return Err(e),
                None => None,
            };

            let lname = layer_name.to_lowercase();
            let mut model = if lname.contains("sage") {
                if let Some(sizes) = sample_sizes {
                    GNNClassifier::new_with_layer_sizes(
                        dims,
                        "sage",
                        Some(sizes),
                        sample_size,
                        loss,
                        optimizer,
                        learning_rate,
                        n_epochs,
                    )
                } else {
                    GNNClassifier::new_with_layer(
                        dims,
                        "sage",
                        sample_size,
                        loss,
                        optimizer,
                        learning_rate,
                        n_epochs,
                    )
                }
            } else {
                GNNClassifier::new(dims, loss, optimizer, learning_rate, n_epochs)
            }
            .map_err(|e| format!("{e:?}"))?;

            if let Some(seed) = params.get("random_state").and_then(|v| v.as_u64()) {
                model.random_state = Some(seed);
            }

            let pred = model
                .fit_predict(adj, &features, &labels)
                .map_err(|e| format!("{e:?}"))?;
            let final_loss = model.state().history.last().copied();
            let mut out = json!({
                "n_labels": pred.len(),
                "n_epochs_run": n_epochs,
            });
            if let Some(loss) = final_loss {
                out["final_loss"] = json!(loss);
            }
            if return_labels {
                out["labels"] = labels_json(&pred);
            }
            Ok(out)
        }
        other => Err(format!("unknown gnn algorithm '{other}'")),
    }
}
