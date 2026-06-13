//! Topology helpers for benchmark IPC (dispatches by operation name).

use serde_json::{Value, json};
use sprs::CsMat;

use crate::topology::core::get_core_decomposition;
use crate::topology::cycles::is_acyclic;
use crate::topology::structure::{
    get_connected_components, is_bipartite, is_connected, BipartiteResult,
};
use crate::topology::triangles::{count_triangles, get_clustering_coefficient};
use crate::utils::check::is_symmetric;

/// Run a topology primitive on an in-memory adjacency. Returns JSON ``Value`` suitable for IPC
/// (integer, float, bool, or array of integers).
pub fn topology_dispatch(op: &str, adj: &CsMat<f64>, params: &Value) -> Result<Value, String> {
    match op {
        "count_triangles" => {
            let parallelize = params
                .get("parallelize")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let n = count_triangles(adj, parallelize).map_err(|e| format!("{e:?}"))?;
            Ok(json!(n))
        }
        "get_clustering_coefficient" => {
            let parallelize = params
                .get("parallelize")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let c = get_clustering_coefficient(adj, parallelize).map_err(|e| format!("{e:?}"))?;
            Ok(json!(c))
        }
        "count_cliques" => {
            let k = params
                .get("clique_size")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| "count_cliques: clique_size required".to_string())?;
            if k < 2 {
                return Err("count_cliques: clique_size must be >= 2".into());
            }
            let core_order = params.get("core_order").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .map(|x| {
                        x.as_i64()
                            .ok_or_else(|| "core_order entries must be integers".to_string())
                            .map(|n| n as i32)
                    })
                    .collect::<Result<Vec<_>, _>>()
            });
            let core_order = match core_order {
                Some(Ok(v)) => Some(v),
                Some(Err(e)) => return Err(e),
                None => None,
            };
            let n = crate::topology::cliques::count_cliques_with_core_order(
                adj,
                k as usize,
                core_order.as_deref(),
            )
            .map_err(|e| format!("{e:?}"))?;
            Ok(json!(n))
        }
        "is_connected" => {
            let connection = params
                .get("connection")
                .and_then(|v| v.as_str())
                .unwrap_or("weak");
            let force_bipartite = params
                .get("force_bipartite")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let b = is_connected(adj, connection, force_bipartite).map_err(|e| format!("{e:?}"))?;
            Ok(json!(b))
        }
        "is_symmetric" => Ok(json!(is_symmetric(adj))),
        "is_bipartite" => {
            let return_bio = params
                .get("return_biadjacency")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if return_bio {
                return Err(
                    "is_bipartite: return_biadjacency=true is not supported over benchmark_ipc"
                        .into(),
                );
            }
            match is_bipartite(adj, false).map_err(|e| format!("{e:?}"))? {
                BipartiteResult::Bool(b) => Ok(json!(b)),
                BipartiteResult::Full(b, _, _, _) => Ok(json!(b)),
            }
        }
        "get_connected_components" => {
            let connection = params
                .get("connection")
                .and_then(|v| v.as_str())
                .unwrap_or("weak");
            let force_bipartite = params
                .get("force_bipartite")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let labels =
                get_connected_components(adj, connection, force_bipartite).map_err(|e| format!("{e:?}"))?;
            Ok(json!(labels))
        }
        "is_acyclic" => {
            let directed = match params.get("directed") {
                None => None,
                Some(v) if v.is_null() => None,
                Some(v) => Some(
                    v.as_bool()
                        .ok_or_else(|| "is_acyclic: directed must be bool or null".to_string())?,
                ),
            };
            let b = is_acyclic(adj, directed).map_err(|e| format!("{e:?}"))?;
            Ok(json!(b))
        }
        "get_core_decomposition" => {
            let labels = get_core_decomposition(adj).map_err(|e| format!("{e:?}"))?;
            Ok(json!(labels))
        }
        other => Err(format!("unknown topology op '{other}'")),
    }
}
