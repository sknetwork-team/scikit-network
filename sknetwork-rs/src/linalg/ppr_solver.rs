use ndarray::Array1;
use sprs::CsMat;
use std::collections::VecDeque;

use crate::linalg::normalizer::normalize_sparse;
use crate::linalg::polynome::{Polynome, PolynomeError};

#[derive(Debug, Clone, PartialEq)]
/// Errors raised by pprerror operations.
pub enum PPRError {
    /// Indicates unknown solver.
    UnknownSolver,
    /// Indicates not implemented backend.
    NotImplementedBackend(String),
    /// Indicates invalid seeds.
    InvalidSeeds,
    /// Indicates polynome.
    Polynome(PolynomeError),
}

impl From<PolynomeError> for PPRError {
    fn from(value: PolynomeError) -> Self {
        Self::Polynome(value)
    }
}

fn normalize_prob(mut x: Vec<f64>) -> Result<Vec<f64>, PPRError> {
    if x.iter().any(|v| *v < 0.0) {
        return Err(PPRError::InvalidSeeds);
    }
    let s: f64 = x.iter().sum();
    if s <= 0.0 {
        return Err(PPRError::InvalidSeeds);
    }
    for v in &mut x {
        *v /= s;
    }
    Ok(x)
}

fn normalize_scores(mut x: Vec<f64>) -> Result<Vec<f64>, PPRError> {
    for v in &mut x {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    let s: f64 = x.iter().sum();
    if s <= 0.0 {
        return Err(PPRError::InvalidSeeds);
    }
    for v in &mut x {
        *v /= s;
    }
    Ok(x)
}

fn power_iteration(
    adjacency: &CsMat<f64>,
    seeds: &[f64],
    damping_factor: f64,
    n_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = adjacency.rows();
    let mut scores = vec![1.0 / n as f64; n];
    let mut out_weight = vec![0.0; n];
    for (i, row) in adjacency.outer_iterator().enumerate() {
        out_weight[i] = row.data().iter().sum();
    }
    for _ in 0..n_iter {
        let mut next = vec![0.0; n];
        for i in 0..n {
            next[i] = (1.0 - damping_factor) * seeds[i];
        }
        let dangling_mass: f64 = (0..n)
            .filter(|&i| out_weight[i] == 0.0)
            .map(|i| scores[i])
            .sum();

        for i in 0..n {
            if out_weight[i] > 0.0
                && let Some(row) = adjacency.outer_view(i)
            {
                let coeff = damping_factor * scores[i] / out_weight[i];
                for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                    next[j] += coeff * v;
                }
            }
        }
        let dangling_coeff = damping_factor * dangling_mass;
        for i in 0..n {
            next[i] += dangling_coeff * seeds[i];
        }
        let diff: f64 = next
            .iter()
            .zip(scores.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        scores = next;
        if diff < tol {
            break;
        }
    }
    let sum: f64 = scores.iter().sum();
    if sum > 0.0 {
        for s in &mut scores {
            *s /= sum;
        }
    }
    scores
}

fn random_surfer_apply(
    adjacency: &CsMat<f64>,
    out_weight: &[f64],
    seeds: &[f64],
    damping_factor: f64,
    x: &[f64],
) -> Vec<f64> {
    let n = adjacency.rows();
    let mut out = vec![0.0; n];
    let sum_x: f64 = x.iter().sum();
    for i in 0..n {
        out[i] =
            (1.0 - damping_factor * if out_weight[i] > 0.0 { 1.0 } else { 0.0 }) * seeds[i] * sum_x;
    }
    for i in 0..n {
        if out_weight[i] > 0.0
            && let Some(row) = adjacency.outer_view(i)
        {
            let coeff = damping_factor * x[i] / out_weight[i];
            for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                out[j] += coeff * v;
            }
        }
    }
    out
}

fn bicgstab_solver(
    adjacency: &CsMat<f64>,
    seeds: &[f64],
    damping_factor: f64,
    n_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = adjacency.rows();
    let mut out_weight = vec![0.0; n];
    for (i, row) in adjacency.outer_iterator().enumerate() {
        out_weight[i] = row.data().iter().sum();
    }
    // Solve (I - A)x = b where A is the random-surfer linear part and b = (1-alpha) seeds.
    let b: Vec<f64> = seeds.iter().map(|s| (1.0 - damping_factor) * s).collect();
    let mut x = vec![1.0 / n as f64; n];
    let mut r = {
        let ax = random_surfer_apply(adjacency, &out_weight, seeds, damping_factor, &x);
        b.iter()
            .zip(ax.iter())
            .map(|(bi, ai)| bi - ai)
            .collect::<Vec<f64>>()
    };
    let r_hat = r.clone();
    let mut p = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut rho_old = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;

    for _ in 0..n_iter.max(1) {
        let rho_new: f64 = r_hat.iter().zip(r.iter()).map(|(a, b)| a * b).sum();
        if rho_new.abs() < 1e-15 {
            break;
        }
        let beta = (rho_new / rho_old) * (alpha / omega);
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }
        v = random_surfer_apply(adjacency, &out_weight, seeds, damping_factor, &p);
        let denom: f64 = r_hat.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        if denom.abs() < 1e-15 {
            break;
        }
        alpha = rho_new / denom;
        let mut s = vec![0.0; n];
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }
        let s_norm = s.iter().map(|x| x * x).sum::<f64>().sqrt();
        if s_norm < tol {
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            break;
        }
        let t = random_surfer_apply(adjacency, &out_weight, seeds, damping_factor, &s);
        let tt: f64 = t.iter().map(|x| x * x).sum();
        if tt.abs() < 1e-15 {
            break;
        }
        omega = t.iter().zip(s.iter()).map(|(ti, si)| ti * si).sum::<f64>() / tt;
        for i in 0..n {
            x[i] += alpha * p[i] + omega * s[i];
        }
        for i in 0..n {
            r[i] = s[i] - omega * t[i];
        }
        let r_norm = r.iter().map(|x| x * x).sum::<f64>().sqrt();
        if r_norm < tol {
            break;
        }
        rho_old = rho_new;
    }
    let sum: f64 = x.iter().sum();
    if sum > 0.0 {
        for xi in &mut x {
            *xi /= sum;
        }
    }
    x
}

fn lanczos_eigen(
    adjacency: &CsMat<f64>,
    seeds: &[f64],
    damping_factor: f64,
    n_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = adjacency.rows();
    let mut out_weight = vec![0.0; n];
    for (i, row) in adjacency.outer_iterator().enumerate() {
        out_weight[i] = row.data().iter().sum();
    }
    let mut x = vec![1.0 / n as f64; n];
    let mut prev = x.clone();
    for _ in 0..n_iter.max(1) {
        let mut y = random_surfer_apply(adjacency, &out_weight, seeds, damping_factor, &x);
        let s: f64 = y.iter().sum();
        if s > 0.0 {
            for yi in &mut y {
                *yi /= s;
            }
        }
        let diff: f64 = y.iter().zip(prev.iter()).map(|(a, b)| (a - b).abs()).sum();
        prev = y.clone();
        x = y;
        if diff < tol {
            break;
        }
    }
    x
}

fn d_iteration(
    adjacency: &CsMat<f64>,
    seeds: &[f64],
    damping_factor: f64,
    n_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = adjacency.rows();
    let mut out_weight = vec![0.0; n];
    for (i, row) in adjacency.outer_iterator().enumerate() {
        out_weight[i] = row.data().iter().sum();
    }

    let mut scores = vec![0.0; n];
    let mut fluid: Vec<f64> = seeds.iter().map(|s| (1.0 - damping_factor) * s).collect();
    for _ in 0..n_iter {
        let mut max_fluid = 0.0;
        for i in 0..n {
            if fluid[i] > max_fluid {
                max_fluid = fluid[i];
            }
            let f = fluid[i];
            if f <= 0.0 {
                continue;
            }
            scores[i] += f;
            fluid[i] = 0.0;

            if out_weight[i] > 0.0 {
                if let Some(row) = adjacency.outer_view(i) {
                    let coeff = damping_factor * f / out_weight[i];
                    for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                        fluid[j] += coeff * v;
                    }
                }
            } else {
                // Dangling mass restarts according to seeds.
                let add = damping_factor * f;
                for k in 0..n {
                    fluid[k] += add * seeds[k];
                }
            }
        }
        if max_fluid < tol {
            break;
        }
    }
    let sum: f64 = scores.iter().sum();
    if sum > 0.0 {
        for s in &mut scores {
            *s /= sum;
        }
    }
    scores
}

fn push_pagerank(adjacency: &CsMat<f64>, seeds: &[f64], damping_factor: f64, tol: f64) -> Vec<f64> {
    let n = adjacency.rows();
    let mut out_weight = vec![0.0; n];
    for (i, row) in adjacency.outer_iterator().enumerate() {
        out_weight[i] = row.data().iter().sum();
    }

    let mut p = vec![0.0; n];
    let mut r = seeds.to_vec();
    let mut in_queue = vec![false; n];
    let mut q = VecDeque::<usize>::new();

    let is_active = |i: usize, r: &[f64], out_weight: &[f64]| -> bool {
        if out_weight[i] > 0.0 {
            r[i] / out_weight[i] > tol
        } else {
            r[i] > tol
        }
    };

    for i in 0..n {
        if is_active(i, &r, &out_weight) {
            q.push_back(i);
            in_queue[i] = true;
        }
    }

    while let Some(u) = q.pop_front() {
        in_queue[u] = false;
        let ru = r[u];
        if ru <= 0.0 {
            continue;
        }
        if !is_active(u, &r, &out_weight) {
            continue;
        }

        // Keep exact PR decomposition invariant.
        p[u] += (1.0 - damping_factor) * ru;
        r[u] = 0.0;

        if out_weight[u] > 0.0 {
            if let Some(row) = adjacency.outer_view(u) {
                let coeff = damping_factor * ru / out_weight[u];
                for (&v, &w) in row.indices().iter().zip(row.data().iter()) {
                    r[v] += coeff * w;
                    if !in_queue[v] && is_active(v, &r, &out_weight) {
                        q.push_back(v);
                        in_queue[v] = true;
                    }
                }
            }
        } else {
            // Dangling node: restart according to personalization.
            let mass = damping_factor * ru;
            for v in 0..n {
                r[v] += mass * seeds[v];
                if !in_queue[v] && is_active(v, &r, &out_weight) {
                    q.push_back(v);
                    in_queue[v] = true;
                }
            }
        }
    }

    // Remaining residual contributes to estimate.
    for i in 0..n {
        p[i] += (1.0 - damping_factor) * r[i];
    }
    let sum: f64 = p.iter().sum();
    if sum > 0.0 {
        for v in &mut p {
            *v /= sum;
        }
    }
    p
}

/// Returns pagerank.
pub fn get_pagerank(
    adjacency: &CsMat<f64>,
    seeds: &Array1<f64>,
    damping_factor: f64,
    n_iter: usize,
    tol: f64,
    solver: &str,
) -> Result<Array1<f64>, PPRError> {
    let seeds = normalize_prob(seeds.to_vec())?;
    let scores = match solver {
        "piteration" => power_iteration(adjacency, &seeds, damping_factor, n_iter, tol),
        "diteration" => d_iteration(adjacency, &seeds, damping_factor, n_iter, tol),
        "push" => push_pagerank(adjacency, &seeds, damping_factor, tol),
        "bicgstab" => bicgstab_solver(adjacency, &seeds, damping_factor, n_iter, tol),
        "lanczos" => lanczos_eigen(adjacency, &seeds, damping_factor, n_iter, tol),
        "RH" => {
            let a = normalize_sparse(adjacency, 1).map_err(|_| PPRError::InvalidSeeds)?;
            let at = &a.transpose_view().to_csr() * damping_factor;
            let coeffs = Array1::<f64>::from_vec(vec![1.0 - damping_factor; n_iter + 1]);
            let poly = Polynome::new(&at, &coeffs)?;
            poly.dot_vec(&Array1::from_vec(seeds))
                .iter()
                .copied()
                .collect()
        }
        _ => return Err(PPRError::UnknownSolver),
    };
    let scores = normalize_scores(scores)?;
    Ok(Array1::from_vec(scores))
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*;
    use crate::data::test_graphs::{test_digraph, test_graph};

    fn is_proba_array(x: &Array1<f64>) -> bool {
        let s = x.sum();
        (s - 1.0).abs() < 1e-8 && x.iter().all(|v| *v >= 0.0)
    }

    #[test]
    fn test_solvers_return_probabilities() {
        for adjacency in [test_graph(), test_digraph()] {
            let n = adjacency.rows();
            let seeds = Array1::from_vec(vec![1.0 / n as f64; n]);
            for solver in ["piteration", "diteration", "push", "RH"] {
                let pr = get_pagerank(&adjacency, &seeds, 0.85, 100, 1e-2, solver).unwrap();
                assert!(is_proba_array(&pr), "solver={solver}");
            }
        }
    }

    #[test]
    fn test_additional_backends_return_probabilities() {
        let adjacency = test_graph();
        let n = adjacency.rows();
        let seeds = Array1::from_vec(vec![1.0 / n as f64; n]);
        for solver in ["bicgstab", "lanczos"] {
            let pr = get_pagerank(&adjacency, &seeds, 0.85, 50, 1e-4, solver).unwrap();
            assert!(is_proba_array(&pr), "solver={solver}");
        }
    }

    #[test]
    fn test_unknown_solver() {
        let adjacency = test_graph();
        let n = adjacency.rows();
        let seeds = Array1::from_vec(vec![1.0 / n as f64; n]);
        assert!(matches!(
            get_pagerank(&adjacency, &seeds, 0.85, 10, 1e-2, "toto"),
            Err(PPRError::UnknownSolver)
        ));
    }
}
