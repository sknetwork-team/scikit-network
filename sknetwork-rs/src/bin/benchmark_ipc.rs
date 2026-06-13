//! Long-lived benchmark process: load data once, serve JSON-line requests on stdin/stdout.
//! Used by Python `benchmarking.lib.rust_ipc` for apples-to-apples timing (algorithm only, graph resident).

use std::io::{BufRead, BufReader, Write};

use serde::Deserialize;
use serde_json::json;
use sprs::CsMat;

use sknetwork_rs::bench::classification::classification_dispatch;
use sknetwork_rs::bench::clustering::clustering_dispatch;
use sknetwork_rs::bench::data::data_dispatch;
use sknetwork_rs::bench::embedding::embedding_dispatch;
use sknetwork_rs::bench::gnn::gnn_dispatch;
use sknetwork_rs::bench::hierarchy::hierarchy_dispatch;
use sknetwork_rs::bench::linkpred::linkpred_dispatch;
use sknetwork_rs::bench::linalg::linalg_dispatch;
use sknetwork_rs::bench::path::path_dispatch;
use sknetwork_rs::bench::ranking::{katz_scores, pagerank_scores};
use sknetwork_rs::bench::regression::regression_dispatch;
use sknetwork_rs::bench::topology::topology_dispatch;
use sknetwork_rs::data::load::{load_netset_with_options, LoadOptions};

const IPC_LOAD_OPTS: LoadOptions = LoadOptions {
    adjacency_only: true,
    materialize_csr: true,
};

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum Request {
    LoadNetset { dataset: String },
    RankingRun {
        algorithm: String,
        params: serde_json::Value,
    },
    /// Topology primitives: ``op`` is e.g. ``count_triangles``, ``is_connected``, …
    TopologyRun {
        op: String,
        #[serde(default)]
        params: serde_json::Value,
    },
    /// Path primitives: ``op`` is e.g. ``get_distances``, ``get_shortest_path``, …
    PathRun {
        op: String,
        #[serde(default)]
        params: serde_json::Value,
    },
    ClusteringRun {
        algorithm: String,
        #[serde(default)]
        params: serde_json::Value,
    },
    EmbeddingRun {
        algorithm: String,
        #[serde(default)]
        params: serde_json::Value,
    },
    ClassificationRun {
        algorithm: String,
        #[serde(default)]
        params: serde_json::Value,
    },
    RegressionRun {
        algorithm: String,
        #[serde(default)]
        params: serde_json::Value,
    },
    LinkpredRun {
        algorithm: String,
        #[serde(default)]
        params: serde_json::Value,
    },
    HierarchyRun {
        algorithm: String,
        #[serde(default)]
        params: serde_json::Value,
    },
    GnnRun {
        algorithm: String,
        #[serde(default)]
        params: serde_json::Value,
    },
    /// Data I/O: ``op`` is e.g. ``from_csv``, ``load_csr_bundle``, ``load_netset_bundle``.
    DataRun {
        op: String,
        #[serde(default)]
        params: serde_json::Value,
    },
    /// Partial SVD: ``algorithm`` is ``lanczos`` or ``halko``.
    LinalgRun {
        algorithm: String,
        #[serde(default)]
        params: serde_json::Value,
    },
    Quit,
}

struct State {
    adjacency: Option<CsMat<f64>>,
}

fn write_response<W: Write>(w: &mut W, v: serde_json::Value) -> Result<(), String> {
    serde_json::to_writer(&mut *w, &v).map_err(|e| e.to_string())?;
    writeln!(w).map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn run() -> Result<(), String> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut line = String::new();
    let mut state = State { adjacency: None };

    loop {
        line.clear();
        let n = reader.read_line(&mut line).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: Request = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                write_response(
                    &mut stdout,
                    json!({"ok": false, "error": format!("invalid request JSON: {e}")}),
                )?;
                continue;
            }
        };

        match req {
            Request::Quit => {
                write_response(&mut stdout, json!({"ok": true}))?;
                break;
            }
            Request::LoadNetset { dataset } => {
                match load_netset_with_options(Some(&dataset), None, &IPC_LOAD_OPTS) {
                Ok(ds) => {
                    let adj = if let Some(a) = ds.adjacency {
                        a
                    } else if let Some(b) = ds.biadjacency {
                        b
                    } else {
                        write_response(
                            &mut stdout,
                            json!({"ok": false, "error": "dataset has neither adjacency nor biadjacency"}),
                        )?;
                        continue;
                    };
                    state.adjacency = Some(adj);
                    write_response(&mut stdout, json!({"ok": true}))?;
                }
                Err(e) => {
                    write_response(
                        &mut stdout,
                        json!({"ok": false, "error": format!("load_netset failed: {e:?}")}),
                    )?;
                }
                }
            }
            Request::TopologyRun { op, params } => {
                let Some(adj) = state.adjacency.as_ref() else {
                    write_response(
                        &mut stdout,
                        json!({"ok": false, "error": "no graph loaded; send load_netset first"}),
                    )?;
                    continue;
                };
                match topology_dispatch(&op, adj, &params) {
                    Ok(result) => {
                        write_response(&mut stdout, json!({"ok": true, "result": result}))?;
                    }
                    Err(e) => {
                        write_response(&mut stdout, json!({"ok": false, "error": e}))?;
                    }
                }
            }
            Request::PathRun { op, params } => {
                let Some(adj) = state.adjacency.as_ref() else {
                    write_response(
                        &mut stdout,
                        json!({"ok": false, "error": "no graph loaded; send load_netset first"}),
                    )?;
                    continue;
                };
                match path_dispatch(&op, adj, &params) {
                    Ok(result) => {
                        write_response(&mut stdout, json!({"ok": true, "result": result}))?;
                    }
                    Err(e) => {
                        write_response(&mut stdout, json!({"ok": false, "error": e}))?;
                    }
                }
            }
            Request::ClusteringRun { algorithm, params } => {
                let Some(adj) = state.adjacency.as_ref() else {
                    write_response(
                        &mut stdout,
                        json!({"ok": false, "error": "no graph loaded; send load_netset first"}),
                    )?;
                    continue;
                };
                match clustering_dispatch(&algorithm, adj, &params) {
                    Ok(result) => {
                        let mut resp = json!({"ok": true});
                        if let Some(obj) = result.as_object() {
                            if let Some(labels) = obj.get("labels") {
                                resp["labels"] = labels.clone();
                            }
                            if let Some(modularity) = obj.get("modularity") {
                                resp["modularity"] = modularity.clone();
                            }
                        } else {
                            resp["labels"] = result;
                        }
                        write_response(&mut stdout, resp)?;
                    }
                    Err(e) => {
                        write_response(&mut stdout, json!({"ok": false, "error": e}))?;
                    }
                }
            }
            Request::EmbeddingRun { algorithm, params } => {
                let Some(adj) = state.adjacency.as_ref() else {
                    write_response(
                        &mut stdout,
                        json!({"ok": false, "error": "no graph loaded; send load_netset first"}),
                    )?;
                    continue;
                };
                match embedding_dispatch(&algorithm, adj, &params) {
                    Ok(result) => {
                        let mut resp = json!({"ok": true});
                        if let Some(obj) = result.as_object() {
                            if let Some(n_rows) = obj.get("n_rows") {
                                resp["n_rows"] = n_rows.clone();
                            }
                            if let Some(n_cols) = obj.get("n_cols") {
                                resp["n_cols"] = n_cols.clone();
                            }
                        } else {
                            resp["embedding"] = result;
                        }
                        write_response(&mut stdout, resp)?;
                    }
                    Err(e) => {
                        write_response(&mut stdout, json!({"ok": false, "error": e}))?;
                    }
                }
            }
            Request::ClassificationRun { algorithm, params } => {
                let Some(adj) = state.adjacency.as_ref() else {
                    write_response(
                        &mut stdout,
                        json!({"ok": false, "error": "no graph loaded; send load_netset first"}),
                    )?;
                    continue;
                };
                match classification_dispatch(&algorithm, adj, &params) {
                    Ok(labels) => {
                        write_response(&mut stdout, json!({"ok": true, "labels": labels}))?;
                    }
                    Err(e) => {
                        write_response(&mut stdout, json!({"ok": false, "error": e}))?;
                    }
                }
            }
            Request::RegressionRun { algorithm, params } => {
                let Some(adj) = state.adjacency.as_ref() else {
                    write_response(
                        &mut stdout,
                        json!({"ok": false, "error": "no graph loaded; send load_netset first"}),
                    )?;
                    continue;
                };
                match regression_dispatch(&algorithm, adj, &params) {
                    Ok(values) => {
                        write_response(&mut stdout, json!({"ok": true, "values": values}))?;
                    }
                    Err(e) => {
                        write_response(&mut stdout, json!({"ok": false, "error": e}))?;
                    }
                }
            }
            Request::LinkpredRun { algorithm, params } => {
                let Some(adj) = state.adjacency.as_ref() else {
                    write_response(
                        &mut stdout,
                        json!({"ok": false, "error": "no graph loaded; send load_netset first"}),
                    )?;
                    continue;
                };
                match linkpred_dispatch(&algorithm, adj, &params) {
                    Ok(links) => {
                        write_response(&mut stdout, json!({"ok": true, "links": links}))?;
                    }
                    Err(e) => {
                        write_response(&mut stdout, json!({"ok": false, "error": e}))?;
                    }
                }
            }
            Request::HierarchyRun { algorithm, params } => {
                let Some(adj) = state.adjacency.as_ref() else {
                    write_response(
                        &mut stdout,
                        json!({"ok": false, "error": "no graph loaded; send load_netset first"}),
                    )?;
                    continue;
                };
                match hierarchy_dispatch(&algorithm, adj, &params) {
                    Ok(result) => {
                        write_response(&mut stdout, json!({"ok": true, "result": result}))?;
                    }
                    Err(e) => {
                        write_response(&mut stdout, json!({"ok": false, "error": e}))?;
                    }
                }
            }
            Request::GnnRun { algorithm, params } => {
                let Some(adj) = state.adjacency.as_ref() else {
                    write_response(
                        &mut stdout,
                        json!({"ok": false, "error": "no graph loaded; send load_netset first"}),
                    )?;
                    continue;
                };
                match gnn_dispatch(&algorithm, adj, &params) {
                    Ok(result) => {
                        write_response(&mut stdout, json!({"ok": true, "result": result}))?;
                    }
                    Err(e) => {
                        write_response(&mut stdout, json!({"ok": false, "error": e}))?;
                    }
                }
            }
            Request::DataRun { op, params } => match data_dispatch(&op, &params) {
                Ok(summary) => {
                    write_response(&mut stdout, json!({"ok": true, "summary": summary}))?;
                }
                Err(e) => {
                    write_response(&mut stdout, json!({"ok": false, "error": e}))?;
                }
            },
            Request::LinalgRun { algorithm, params } => {
                let Some(adj) = state.adjacency.as_ref() else {
                    write_response(
                        &mut stdout,
                        json!({"ok": false, "error": "no graph loaded; send load_netset first"}),
                    )?;
                    continue;
                };
                match linalg_dispatch(adj, &algorithm, &params) {
                    Ok(result) => {
                        let mut resp = json!({"ok": true});
                        if let Some(obj) = result.as_object() {
                            for (k, v) in obj {
                                resp[k] = v.clone();
                            }
                        }
                        write_response(&mut stdout, resp)?;
                    }
                    Err(e) => {
                        write_response(&mut stdout, json!({"ok": false, "error": e}))?;
                    }
                }
            }
            Request::RankingRun { algorithm, params } => {
                let Some(adj) = state.adjacency.as_ref() else {
                    write_response(
                        &mut stdout,
                        json!({"ok": false, "error": "no graph loaded; send load_netset first"}),
                    )?;
                    continue;
                };

                let scores_res: Result<Vec<f64>, String> = match algorithm.as_str() {
                    "pagerank" => {
                        let Some(damping) = params.get("damping_factor").and_then(|x| x.as_f64())
                        else {
                            write_response(
                                &mut stdout,
                                json!({"ok": false, "error": "pagerank: missing damping_factor"}),
                            )?;
                            continue;
                        };
                        let Some(n_iter) = params.get("n_iter").and_then(|x| x.as_u64()) else {
                            write_response(
                                &mut stdout,
                                json!({"ok": false, "error": "pagerank: missing n_iter"}),
                            )?;
                            continue;
                        };
                        let Some(tol) = params.get("tol").and_then(|x| x.as_f64()) else {
                            write_response(
                                &mut stdout,
                                json!({"ok": false, "error": "pagerank: missing tol"}),
                            )?;
                            continue;
                        };
                        pagerank_scores(adj, damping, n_iter as usize, tol)
                    }
                    "katz" => {
                        let Some(damping) = params.get("damping_factor").and_then(|x| x.as_f64())
                        else {
                            write_response(
                                &mut stdout,
                                json!({"ok": false, "error": "katz: missing damping_factor"}),
                            )?;
                            continue;
                        };
                        let Some(path_length) = params.get("path_length").and_then(|x| x.as_u64())
                        else {
                            write_response(
                                &mut stdout,
                                json!({"ok": false, "error": "katz: missing path_length"}),
                            )?;
                            continue;
                        };
                        katz_scores(adj, damping, path_length as usize)
                    }
                    other => {
                        write_response(
                            &mut stdout,
                            json!({"ok": false, "error": format!("unknown ranking algorithm '{other}'")}),
                        )?;
                        continue;
                    }
                };

                match scores_res {
                    Ok(scores) => {
                        write_response(&mut stdout, json!({"ok": true, "scores": scores}))?;
                    }
                    Err(e) => {
                        write_response(&mut stdout, json!({"ok": false, "error": e}))?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        let mut stdout = std::io::stdout();
        let _ = write_response(&mut stdout, json!({"ok": false, "error": e}));
        std::process::exit(1);
    }
}
