//! Path helpers for benchmark IPC (dispatches by operation name).
//!
//! **Not covered:** Python `breadth_first_search` is defined as `numpy.argsort(get_distances(...))`
//! with NumPy’s tie-breaking, which does not match a stable `(distance, node_id)` sort and is not
//! worth replicating here; benchmark `get_distances` instead.

use serde_json::{Value, json};
use sprs::CsMat;

use crate::path::distances::get_distances_multi;
use crate::path::shortest_path::get_shortest_path;

fn csr_nonzero_edges_sorted(mat: &CsMat<f64>) -> Vec<Vec<usize>> {
    let mut pairs: Vec<[usize; 2]> = Vec::new();
    for r in 0..mat.rows() {
        if let Some(row) = mat.outer_view(r) {
            for (c, val) in row.iter() {
                if *val != 0.0 {
                    pairs.push([r, c]);
                }
            }
        }
    }
    pairs.sort_by(|a, b| a[0].cmp(&b[0]).then(a[1].cmp(&b[1])));
    pairs.into_iter().map(|e| vec![e[0], e[1]]).collect()
}

/// Run a path primitive on an in-memory adjacency. Returns JSON ``Value`` for IPC.
pub fn path_dispatch(op: &str, adj: &CsMat<f64>, params: &Value) -> Result<Value, String> {
    match op {
        "get_distances" => {
            let sources: Vec<usize> = params
                .get("sources")
                .and_then(|v| v.as_array())
                .and_then(|arr| {
                    arr.iter()
                        .map(|x| x.as_u64().map(|u| u as usize))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| "get_distances: sources array of integers required".to_string())?;
            if sources.is_empty() {
                return Err("get_distances: sources must be non-empty".into());
            }
            let d = get_distances_multi(adj, &sources).map_err(|e| format!("{e:?}"))?;
            Ok(json!(d))
        }
        "get_shortest_path" => {
            let sources: Vec<usize> = params
                .get("sources")
                .and_then(|v| v.as_array())
                .and_then(|arr| {
                    arr.iter()
                        .map(|x| x.as_u64().map(|u| u as usize))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| "get_shortest_path: sources array required".to_string())?;
            if sources.is_empty() {
                return Err("get_shortest_path: sources must be non-empty".into());
            }
            let dag = get_shortest_path(adj, &sources).map_err(|e| format!("{e:?}"))?;
            Ok(json!({ "edges": csr_nonzero_edges_sorted(&dag) }))
        }
        other => Err(format!("unknown path op '{other}'")),
    }
}
