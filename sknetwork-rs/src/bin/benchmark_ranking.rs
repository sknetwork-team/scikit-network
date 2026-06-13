//! One-shot CLI for manual smoke tests (loads NetSet on each invocation).
//! For fair Python-vs-Rust timing use `benchmark_ipc` + `benchmarking.lib.rust_ipc`.

use std::env;

use sknetwork_rs::bench::ranking::{katz_scores, pagerank_scores};
use sknetwork_rs::data::load::{load_netset_with_options, LoadOptions};

fn print_scores(scores: &[f64]) {
    for (idx, s) in scores.iter().enumerate() {
        if idx > 0 {
            print!(",");
        }
        print!("{:.16}", s);
    }
    println!();
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        return Err(
            "usage: benchmark_ranking <algorithm> <dataset_name> <params...>".to_string(),
        );
    }
    let algorithm = &args[1];
    let dataset_name = &args[2];
    let opts = LoadOptions {
        adjacency_only: true,
        materialize_csr: true,
    };
    let dataset = load_netset_with_options(Some(dataset_name), None, &opts)
        .map_err(|e| format!("failed to load dataset via load_netset: {e:?}"))?;
    let adjacency = if let Some(a) = dataset.adjacency {
        a
    } else if let Some(b) = dataset.biadjacency {
        b
    } else {
        return Err("dataset has neither adjacency nor biadjacency".to_string());
    };

    match algorithm.as_str() {
        "pagerank" => {
            if args.len() != 6 {
                return Err(
                    "pagerank params: <damping_factor> <n_iter> <tol> (after dataset name)"
                        .to_string(),
                );
            }
            let damping_factor: f64 = args[3]
                .parse()
                .map_err(|_| "invalid pagerank damping_factor".to_string())?;
            let n_iter: usize = args[4]
                .parse()
                .map_err(|_| "invalid pagerank n_iter".to_string())?;
            let tol: f64 = args[5]
                .parse()
                .map_err(|_| "invalid pagerank tol".to_string())?;

            let scores = pagerank_scores(&adjacency, damping_factor, n_iter, tol)?;
            print_scores(&scores);
            Ok(())
        }
        "katz" => {
            if args.len() != 5 {
                return Err(
                    "katz params: <damping_factor> <path_length> (after dataset name)"
                        .to_string(),
                );
            }
            let damping_factor: f64 = args[3]
                .parse()
                .map_err(|_| "invalid katz damping_factor".to_string())?;
            let path_length: usize = args[4]
                .parse()
                .map_err(|_| "invalid katz path_length".to_string())?;

            let scores = katz_scores(&adjacency, damping_factor, path_length)?;
            print_scores(&scores);
            Ok(())
        }
        _ => Err(format!("unsupported algorithm '{algorithm}'")),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
