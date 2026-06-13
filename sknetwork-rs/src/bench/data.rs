//! Data I/O benchmarks: parsers and bundle / netset loaders.

use std::path::Path;

use serde_json::{json, Value};
use sprs::CsMat;

use crate::data::load::{load, load_dataset_folder, load_netset_with_options, LoadOptions};
use crate::data::parse::{from_csv, from_graphml, GraphDataset, ParseResult};

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

fn load_options_from_params(params: &Value) -> LoadOptions {
    LoadOptions {
        adjacency_only: params
            .get("adjacency_only")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        materialize_csr: params
            .get("materialize_csr")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    }
}

fn matrix_only_from_params(params: &Value) -> bool {
    params
        .get("matrix_only")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn dataset_summary(ds: &GraphDataset, matrix_only: bool) -> Result<Value, String> {
    let mat = ds
        .adjacency
        .as_ref()
        .or(ds.biadjacency.as_ref())
        .ok_or_else(|| "dataset has no adjacency or biadjacency".to_string())?;
    let mut out = json!({
        "n_rows": mat.rows(),
        "n_cols": mat.cols(),
        "nnz": mat.nnz(),
        "layout": if ds.adjacency.is_some() { "adjacency" } else { "biadjacency" },
    });
    if !matrix_only {
        out["edges"] = json!(mat_sorted_edges(mat));
        if let Some(labels) = &ds.names {
            out["names"] = json!(labels);
        }
        if let Some(labels) = &ds.names_str {
            out["names"] = json!(labels);
        }
        if let Some(labels) = &ds.names_row {
            out["names_row"] = json!(labels);
        }
        if let Some(labels) = &ds.names_col {
            out["names_col"] = json!(labels);
        }
    }
    Ok(out)
}

fn parse_result_summary(result: ParseResult, matrix_only: bool) -> Result<Value, String> {
    match result {
        ParseResult::Matrix(m) => {
            let ds = GraphDataset {
                adjacency: Some(m),
                biadjacency: None,
                names: None,
                names_str: None,
                names_row: None,
                names_col: None,
                node_attribute: None,
                edge_attribute: None,
                meta: None,
            };
            dataset_summary(&ds, matrix_only)
        }
        ParseResult::Dataset(ds) => dataset_summary(&ds, matrix_only),
    }
}

pub fn data_dispatch(op: &str, params: &Value) -> Result<Value, String> {
    let matrix_only = matrix_only_from_params(params);
    match op {
        "from_csv" => {
            let path = params
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "from_csv: missing path".to_string())?;
            let directed = params.get("directed").and_then(|v| v.as_bool()).unwrap_or(false);
            let bipartite = params.get("bipartite").and_then(|v| v.as_bool()).unwrap_or(false);
            let weighted = params.get("weighted").and_then(|v| v.as_bool()).unwrap_or(true);
            let reindex = params.get("reindex").and_then(|v| v.as_bool()).unwrap_or(false);
            let matrix_only_kw = params.get("matrix_only").and_then(|v| v.as_bool());
            let parsed = from_csv(
                path,
                None,
                None,
                "#%",
                None,
                directed,
                bipartite,
                weighted,
                reindex,
                None,
                matrix_only_kw,
            )
            .map_err(|e| format!("from_csv failed: {e:?}"))?;
            parse_result_summary(parsed, matrix_only)
        }
        "from_graphml" => {
            let path = params
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "from_graphml: missing path".to_string())?;
            let weight_key = params
                .get("weight_key")
                .and_then(|v| v.as_str())
                .unwrap_or("weight");
            let ds = from_graphml(path, weight_key).map_err(|e| format!("from_graphml failed: {e:?}"))?;
            dataset_summary(&ds, matrix_only)
        }
        "load_netset_bundle" => {
            let path = params
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "load_netset_bundle: missing path".to_string())?;
            let folder = Path::new(path);
            let opts = load_options_from_params(params);
            let ds = load_dataset_folder(folder, &opts)
                .map_err(|e| format!("load_netset_bundle failed: {e:?}"))?;
            dataset_summary(&ds, matrix_only)
        }
        "load_csr_bundle" => {
            let path = params
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "load_csr_bundle: missing path".to_string())?;
            let ds = load(Path::new(path)).map_err(|e| format!("load_csr_bundle failed: {e:?}"))?;
            dataset_summary(&ds, matrix_only)
        }
        "load_csr_folder" => {
            let path = params
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "load_csr_folder: missing path".to_string())?;
            let opts = load_options_from_params(params);
            let ds = load_dataset_folder(Path::new(path), &opts)
                .map_err(|e| format!("load_csr_folder failed: {e:?}"))?;
            dataset_summary(&ds, matrix_only)
        }
        "materialize_csr" => {
            let path = params
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "materialize_csr: missing path".to_string())?;
            let opts = LoadOptions {
                adjacency_only: false,
                materialize_csr: true,
            };
            let ds = load_dataset_folder(Path::new(path), &opts)
                .map_err(|e| format!("materialize_csr failed: {e:?}"))?;
            dataset_summary(&ds, true)
        }
        "load_netset" => {
            let dataset = params
                .get("dataset")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "load_netset: missing dataset".to_string())?;
            let opts = load_options_from_params(params);
            let ds = if let Some(home) = params.get("data_home").and_then(|v| v.as_str()) {
                load_netset_with_options(Some(dataset), Some(Path::new(home)), &opts)
            } else {
                load_netset_with_options(Some(dataset), None, &opts)
            }
            .map_err(|e| format!("load_netset failed: {e:?}"))?;
            dataset_summary(&ds, matrix_only)
        }
        other => Err(format!("unknown data op '{other}'")),
    }
}
