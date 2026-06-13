//! Partial SVD benchmarks (Lanczos / Halko) for ``benchmark_ipc``.

use ndarray::Array1;
use serde_json::{json, Value};
use sprs::CsMat;

use crate::linalg::rng::svds_default_v0;
use crate::linalg::svd_solver::{fit_partial_svd, LanczosSVD, PartialSvdResult, SvdSolverKind, SVDInput};

fn parse_v0(params: &Value, n: usize) -> Option<Array1<f64>> {
    let arr = params.get("v0")?.as_array()?;
    if arr.len() != n {
        return None;
    }
    Some(Array1::from_vec(
        arr.iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect::<Vec<_>>(),
    ))
}

fn v0_from_params(params: &Value, n_row: usize, n_col: usize) -> Array1<f64> {
    let n = n_row.min(n_col);
    if let Some(v0) = parse_v0(params, n) {
        return v0;
    }
    let seed = params
        .get("v0_seed")
        .or_else(|| params.get("random_state"))
        .and_then(|v| v.as_u64());
    svds_default_v0(n, seed)
}

fn frobenius_norm_sq(a: &CsMat<f64>) -> f64 {
    a.data()
        .iter()
        .map(|v| {
            let x = *v;
            x * x
        })
        .sum()
}

/// ``‖A - UΣVᵀ‖_F / ‖A‖_F`` using the column-wise residual identity.
pub fn frobenius_rel_residual(a: &CsMat<f64>, result: &PartialSvdResult) -> f64 {
    let norm_a_sq = frobenius_norm_sq(a);
    if norm_a_sq <= 0.0 {
        return 0.0;
    }
    let input = SVDInput::Sparse(a.clone());
    let k = result.s.len();
    let mut err_sq = 0.0;
    for c in 0..k {
        let vc = result.v.column(c).to_owned();
        let av = match &input {
            SVDInput::Sparse(mat) => {
                let mut out = Array1::<f64>::zeros(mat.rows());
                for (i, row) in mat.outer_iterator().enumerate() {
                    let mut s = 0.0;
                    for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                        s += v * vc[j];
                    }
                    out[i] = s;
                }
                out
            }
            SVDInput::SparseLR(_) => unreachable!("benchmark uses sparse CSR only"),
        };
        let sigma = result.s[c];
        for i in 0..av.len() {
            let d = av[i] - result.u[[i, c]] * sigma;
            err_sq += d * d;
        }
    }
    err_sq.sqrt() / norm_a_sq.sqrt()
}

fn fit_lanczos_with_v0(
    adj: &CsMat<f64>,
    k: usize,
    v0: &Array1<f64>,
    random_state: Option<u64>,
) -> Result<PartialSvdResult, String> {
    let mut solver = LanczosSVD::default().with_random_state(random_state);
    solver
        .fit(SVDInput::Sparse(adj.clone()), k, Some(v0.clone()))
        .map_err(|e| format!("lanczos fit failed: {e:?}"))?;
    Ok(PartialSvdResult {
        u: solver
            .singular_vectors_left
            .ok_or_else(|| "missing U".to_string())?,
        s: solver
            .singular_values
            .ok_or_else(|| "missing singular values".to_string())?,
        v: solver
            .singular_vectors_right
            .ok_or_else(|| "missing V".to_string())?,
    })
}

/// Run partial SVD on the loaded adjacency/biadjacency matrix.
pub fn linalg_dispatch(adj: &CsMat<f64>, algorithm: &str, params: &Value) -> Result<Value, String> {
    let k = params
        .get("n_components")
        .and_then(|v| v.as_u64())
        .ok_or("missing n_components")? as usize;
    let timing_only = params
        .get("timing_only")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let (n_row, n_col) = adj.shape();
    let v0 = v0_from_params(params, n_row, n_col);
    let random_state = params
        .get("random_state")
        .and_then(|v| v.as_u64());

    let result = match algorithm {
        "lanczos" => fit_lanczos_with_v0(adj, k, &v0, random_state)?,
        "halko" => fit_partial_svd(
            SvdSolverKind::Halko,
            SVDInput::Sparse(adj.clone()),
            k,
            random_state,
        )
        .map_err(|e| format!("halko fit failed: {e:?}"))?,
        other => return Err(format!("unknown linalg algorithm '{other}'")),
    };

    if timing_only {
        return Ok(json!({"timing_only": true}));
    }

    let residual = frobenius_rel_residual(adj, &result);
    let singular_values: Vec<f64> = result.s.iter().copied().collect();
    Ok(json!({
        "singular_values": singular_values,
        "residual_frobenius_rel": residual,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;

    #[test]
    fn test_linalg_dispatch_lanczos() {
        let mut tri = TriMat::<f64>::new((4, 4));
        tri.add_triplet(0, 0, 2.0);
        tri.add_triplet(1, 1, 1.0);
        let a = tri.to_csr::<usize>();
        let v0 = svds_default_v0(4, Some(7));
        let out = linalg_dispatch(
            &a,
            "lanczos",
            &json!({"n_components": 2, "v0": v0.iter().copied().collect::<Vec<_>>()}),
        )
        .unwrap();
        assert!(out.get("singular_values").unwrap().as_array().unwrap().len() == 2);
    }
}
