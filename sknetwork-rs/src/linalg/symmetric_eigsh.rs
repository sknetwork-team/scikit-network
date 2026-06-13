//! Symmetric sparse eigenvalue solver via Implicitly Restarted Lanczos (IRLM).
//!
//! Matrix-free implementation aligned with ``scipy.sparse.linalg.eigsh`` /
//! ARPACK ``*saupd``: Lanczos tridiagonalization, Simon/Parlett selective
//! reorthogonalization, Rayleigh–Ritz extraction, true residual tests,
//! vector locking, and implicit restart with sorted shifts. Scales to large
//! ``n`` through matvec-only access (no dense operator materialization).

use ndarray::{Array1, Array2};
use std::time::Instant;

use super::dense_linalg::{dense_matmul, l2_norm, qr_orthonormalize, symmetric_eigh};
use super::rng::{arpack_machine_tol, choose_ncv, svds_default_v0};

/// Cumulative phase timers for ``SKNETWORK_EIGSH_PROFILE=1`` (stderr summary at end of ``eigsh``).
#[derive(Debug, Default, Clone)]
pub struct EigshProfile {
    /// Restarts value.
    pub restarts: u32,
    /// Matvec Calls value.
    pub matvec_calls: u32,
    /// Matvec Us value.
    pub matvec_us: u64,
    /// Extend Us value.
    pub extend_us: u64,
    /// Tridiag Us value.
    pub tridiag_us: u64,
    /// Ritz Us value.
    pub ritz_us: u64,
    /// Implicit Restart Us value.
    pub implicit_restart_us: u64,
    /// Output Qr Us value.
    pub output_qr_us: u64,
}

fn eigsh_profile_enabled() -> bool {
    std::env::var_os("SKNETWORK_EIGSH_PROFILE").is_some()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// EigWhich enum.
pub enum EigWhich {
    /// Indicates lm.
    Lm,
    /// Indicates sm.
    Sm,
    /// Indicates la.
    La,
    /// Indicates sa.
    Sa,
}

impl EigWhich {
    /// Computes parse.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "LM" => Some(Self::Lm),
            "SM" => Some(Self::Sm),
            "LA" => Some(Self::La),
            "SA" => Some(Self::Sa),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
/// EigshOptions value.
pub struct EigshOptions {
    /// Ncv value.
    pub ncv: Option<usize>,
    /// Maxiter value.
    pub maxiter: Option<usize>,
    /// Tol value.
    pub tol: f64,
    /// V0 value.
    pub v0: Option<Array1<f64>>,
    /// Seed for the default ``standard_normal`` start vector (SciPy ``svds`` path).
    pub rng_seed: Option<u64>,
    /// Final MGS on returned Ritz vectors (``svds`` applies its own QR).
    pub orthonormalize: bool,
}

impl Default for EigshOptions {
    fn default() -> Self {
        Self {
            ncv: None,
            maxiter: None,
            tol: 0.0,
            v0: None,
            rng_seed: None,
            orthonormalize: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by eigsh error operations.
pub enum EigshError {
    /// Indicates invalid components.
    InvalidComponents,
    /// Indicates empty problem.
    EmptyProblem,
    /// Indicates unknown which.
    UnknownWhich,
    /// Indicates no convergence.
    NoConvergence,
}

fn cmp_which(which: EigWhich, a: f64, b: f64) -> std::cmp::Ordering {
    match which {
        EigWhich::Lm => b
            .abs()
            .partial_cmp(&a.abs())
            .unwrap_or(std::cmp::Ordering::Equal),
        EigWhich::Sm => a
            .abs()
            .partial_cmp(&b.abs())
            .unwrap_or(std::cmp::Ordering::Equal),
        EigWhich::La => b
            .partial_cmp(&a)
            .unwrap_or(std::cmp::Ordering::Equal),
        EigWhich::Sa => a
            .partial_cmp(&b)
            .unwrap_or(std::cmp::Ordering::Equal),
    }
}

fn reorthogonalize_slice(w: &mut [f64], basis: &[Array1<f64>]) {
    for b in basis {
        let dot = dot_slices(w, b.as_slice().unwrap());
        axpy_sub_slice(w, dot, b.as_slice().unwrap());
    }
}

#[inline]
fn dot_slices(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline]
fn axpy_sub_slice(w: &mut [f64], coeff: f64, v: &[f64]) {
    if coeff == 0.0 {
        return;
    }
    for i in 0..w.len() {
        w[i] -= coeff * v[i];
    }
}

#[inline]
fn dot_basis_col(w: &[f64], v: &Array2<f64>, col: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..w.len() {
        s += w[i] * v[[i, col]];
    }
    s
}

#[inline]
fn axpy_sub_basis_col(w: &mut [f64], coeff: f64, v: &Array2<f64>, col: usize) {
    if coeff == 0.0 {
        return;
    }
    for i in 0..w.len() {
        w[i] -= coeff * v[[i, col]];
    }
}

/// Gershgorin bound on ``‖T‖`` from the current partial Lanczos tridiagonal.
fn lanczos_anorm(alpha: &[f64], beta: &[f64]) -> f64 {
    let m = alpha.len();
    if m == 0 {
        return 1.0;
    }
    let mut bound = 0.0_f64;
    for i in 0..m {
        let mut r = alpha[i].abs();
        if i > 0 {
            r += beta[i - 1].abs();
        }
        if i < beta.len() {
            r += beta[i].abs();
        }
        bound = bound.max(r);
    }
    bound.max(1.0)
}

/// ``√ε · ‖T‖`` selective reorthogonalization threshold (Simon / ARPACK).
fn selective_reorthogonalize_threshold(anorm: f64) -> f64 {
    f64::EPSILON.sqrt() * anorm
}

/// Full modified Gram–Schmidt against the current Lanczos basis.
fn lanczos_full_reorthogonalize(
    w: &mut [f64],
    v: &Array2<f64>,
    j: usize,
    locked: &[Array1<f64>],
) {
    reorthogonalize_slice(w, locked);
    for col in 0..=j {
        let dot = dot_basis_col(w, v, col);
        axpy_sub_basis_col(w, dot, v, col);
    }
    reorthogonalize_slice(w, locked);
}

/// Simon selective reorthogonalization: orthonormalize against locked vectors, scan
/// ``|qᵢᵀw|``, then one full MGS pass over ``0..=j`` when any violation exceeds ``η``.
fn lanczos_selective_reorthogonalize(
    w: &mut [f64],
    v: &Array2<f64>,
    j: usize,
    locked: &[Array1<f64>],
    eta: f64,
) {
    reorthogonalize_slice(w, locked);
    let mut triggered = false;
    for col in 0..=j {
        if dot_basis_col(w, v, col).abs() > eta {
            triggered = true;
            break;
        }
    }
    if triggered {
        for col in 0..=j {
            let dot = dot_basis_col(w, v, col);
            axpy_sub_basis_col(w, dot, v, col);
        }
    }
    reorthogonalize_slice(w, locked);
}

/// ARPACK bound ``‖A y - θ y‖ / max(|θ|, 1) ≤ β_{m-1} |s_m| / max(|θ|, 1)`` for a Ritz
/// pair ``(θ, s)`` of the current Lanczos tridiagonal (``q_{m+1}`` unit).
fn tridiagonal_ritz_residual_bound(
    beta: &[f64],
    m: usize,
    ritz_vecs: &Array2<f64>,
    ridx: usize,
    theta: f64,
) -> f64 {
    if m == 0 {
        return f64::INFINITY;
    }
    let s_last = ritz_vecs[[m - 1, ridx]].abs();
    let b_tail = if m >= 2 {
        beta.get(m - 2).copied().unwrap_or(0.0)
    } else {
        0.0
    };
    b_tail * s_last / theta.abs().max(1.0)
}

/// Symmetric eigendecomposition of a Lanczos tridiagonal (once per restart).
fn tridiagonal_eigh(alpha: &[f64], beta: &[f64]) -> (Array1<f64>, Array2<f64>) {
    symmetric_eigh(tridiagonal_to_dense(alpha, beta))
}

fn normalize_start(mut v: Array1<f64>, n: usize, seed: Option<u64>) -> Array1<f64> {
    let nv = l2_norm(&v);
    if nv > 0.0 {
        v /= nv;
        return v;
    }
    let mut v = svds_default_v0(n, seed);
    let nv = l2_norm(&v);
    if nv > 0.0 {
        v /= nv;
    }
    v
}

fn tridiagonal_to_dense(alpha: &[f64], beta: &[f64]) -> Array2<f64> {
    let m = alpha.len();
    let mut t = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        t[[i, i]] = alpha[i];
        if i + 1 < m {
            t[[i, i + 1]] = beta[i];
            t[[i + 1, i]] = beta[i];
        }
    }
    t
}

fn eigenvalue_near(a: f64, b: f64, rtol: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() <= rtol * scale
}

/// ``k`` Ritz indices targeting distinct eigenvalues (clusters count as one slot).
fn distinct_wanted_ritz_indices(
    ritz_vals: &[f64],
    m: usize,
    k: usize,
    which: EigWhich,
    near_rtol: f64,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| cmp_which(which, ritz_vals[a], ritz_vals[b]));
    let mut out = Vec::with_capacity(k);
    for ridx in order {
        if out.len() >= k {
            break;
        }
        if out
            .iter()
            .any(|&w| eigenvalue_near(ritz_vals[w], ritz_vals[ridx], near_rtol))
        {
            continue;
        }
        out.push(ridx);
    }
    out
}

fn relative_residual_inplace<F>(
    matvec: &mut F,
    vec: &Array1<f64>,
    theta: f64,
    av: &mut Array1<f64>,
) -> f64
where
    F: FnMut(&Array1<f64>, &mut Array1<f64>),
{
    matvec(vec, av);
    let mut err = 0.0;
    for (a, &vi) in av.iter().zip(vec.iter()) {
        let d = a - theta * vi;
        err += d * d;
    }
    err.sqrt() / theta.abs().max(1.0)
}

/// ``out = V[:, :m] @ coeffs`` (Lanczos basis linear combination).
fn lanczos_basis_combo(v: &Array2<f64>, coeffs: &[f64], m: usize, out: &mut Array1<f64>) {
    out.fill(0.0);
    let n = out.len();
    for j in 0..m {
        let coeff = coeffs[j];
        if coeff == 0.0 {
            continue;
        }
        for i in 0..n {
            out[i] += coeff * v[[i, j]];
        }
    }
}

fn build_ritz_vector_into(
    v: &Array2<f64>,
    ritz_vecs_small: &Array2<f64>,
    ridx: usize,
    m: usize,
    out: &mut Array1<f64>,
) {
    out.fill(0.0);
    let n = out.len();
    for col in 0..m {
        let coeff = ritz_vecs_small[[col, ridx]];
        if coeff == 0.0 {
            continue;
        }
        for i in 0..n {
            out[i] += coeff * v[[i, col]];
        }
    }
    let nv = l2_norm(out);
    if nv > 0.0 {
        let inv = 1.0 / nv;
        for x in out.iter_mut() {
            *x *= inv;
        }
    }
}

/// Implicit QR step with shift `mu` on symmetric tridiagonal `(alpha, beta)`.
fn implicit_qr_step(alpha: &mut [f64], beta: &mut [f64], mu: f64) -> Array2<f64> {
    let m = alpha.len();
    let mut q = Array2::<f64>::eye(m);
    if m == 0 {
        return q;
    }

    let x0 = alpha[0] - mu;
    let x1 = if m > 1 { beta[0] } else { 0.0 };
    let r = (x0 * x0 + x1 * x1).sqrt();
    if r == 0.0 {
        return q;
    }
    let c = x0 / r;
    let s = x1 / r;

    let a00 = alpha[0];
    let a01 = if m > 1 { beta[0] } else { 0.0 };
    let a11 = if m > 1 { alpha[1] } else { 0.0 };
    let mut a12 = if m > 2 { beta[1] } else { 0.0 };

    alpha[0] = c * c * a00 + 2.0 * s * c * a01 + s * s * a11;
    if m > 1 {
        beta[0] = s * (c * a11 - s * a00) + c * (c * a01 - s * a11);
        alpha[1] = s * s * a00 - 2.0 * s * c * a01 + c * c * a11;
    }

    for i in 0..m {
        let qi0 = q[[i, 0]];
        let qi1 = if m > 1 { q[[i, 1]] } else { 0.0 };
        q[[i, 0]] = c * qi0 + s * qi1;
        if m > 1 {
            q[[i, 1]] = -s * qi0 + c * qi1;
        }
    }

    if m <= 1 {
        return q;
    }

    let mut k = 0;
    while k < m - 2 {
        let bulge = a12;
        if bulge.abs() < 1e-15 {
            break;
        }
        let p = alpha[k + 1];
        let qv = beta[k + 1];
        let rnorm = (p * p + bulge * bulge).sqrt();
        let c2 = p / rnorm;
        let s2 = bulge / rnorm;

        alpha[k + 1] = rnorm;
        beta[k + 1] = c2 * qv + s2 * (if k + 2 < m { alpha[k + 2] } else { 0.0 });
        if k + 2 < m {
            alpha[k + 2] = -s2 * qv + c2 * alpha[k + 2];
        }
        if k + 2 < m - 1 {
            a12 = s2 * beta[k + 2];
            beta[k + 2] *= c2;
        } else {
            a12 = 0.0;
        }

        for i in 0..m {
            let a = q[[i, k + 1]];
            let b = q[[i, k + 2]];
            q[[i, k + 1]] = c2 * a + s2 * b;
            q[[i, k + 2]] = -s2 * a + c2 * b;
        }
        k += 1;
    }
    if m >= 3 && a12.abs() > 1e-15 {
        let p = alpha[m - 2];
        let bulge = a12;
        let rnorm = (p * p + bulge * bulge).sqrt();
        let c2 = p / rnorm;
        let s2 = bulge / rnorm;
        alpha[m - 2] = rnorm;
        beta[m - 2] = c2 * beta[m - 2];
        for i in 0..m {
            let a = q[[i, m - 2]];
            let b = q[[i, m - 1]];
            q[[i, m - 2]] = c2 * a + s2 * b;
            q[[i, m - 1]] = -s2 * a + c2 * b;
        }
    }
    q
}

struct LanczosWorkspace {
    w: Array1<f64>,
    q_prev: Array1<f64>,
    q_curr: Array1<f64>,
    basis: Array2<f64>,
    basis_scratch: Array2<f64>,
    ritz_vec: Array1<f64>,
    matvec_out: Array1<f64>,
    last_ritz_vals: Array1<f64>,
    last_ritz_vecs: Array2<f64>,
    last_m: usize,
}

impl LanczosWorkspace {
    fn new(n: usize, ncv: usize) -> Self {
        Self {
            w: Array1::<f64>::zeros(n),
            q_prev: Array1::<f64>::zeros(n),
            q_curr: Array1::<f64>::zeros(n),
            basis: Array2::<f64>::zeros((n, ncv)),
            basis_scratch: Array2::<f64>::zeros((n, ncv)),
            ritz_vec: Array1::<f64>::zeros(n),
            matvec_out: Array1::<f64>::zeros(n),
            last_ritz_vals: Array1::<f64>::zeros(ncv),
            last_ritz_vecs: Array2::<f64>::zeros((ncv, ncv)),
            last_m: 0,
        }
    }

    fn store_ritz_decomposition(&mut self, m: usize, vals: &Array1<f64>, vecs: &Array2<f64>) {
        self.last_m = m;
        for i in 0..m {
            self.last_ritz_vals[i] = vals[i];
            for j in 0..m {
                self.last_ritz_vecs[[j, i]] = vecs[[j, i]];
            }
        }
    }
}

/// One Lanczos extension step (step index ``j``).
fn lanczos_step<F>(
    matvec: &mut F,
    j: usize,
    n: usize,
    ncv: usize,
    locked: &[Array1<f64>],
    workspace: &mut LanczosWorkspace,
    alpha: &mut Vec<f64>,
    beta: &mut Vec<f64>,
    freeze_alpha: bool,
) -> bool
where
    F: FnMut(&Array1<f64>, &mut Array1<f64>),
{
    matvec(&workspace.q_curr, &mut workspace.w);
    if j > 0 && j - 1 < beta.len() {
        let b_prev = beta[j - 1];
        for i in 0..workspace.w.len() {
            workspace.w[i] -= b_prev * workspace.q_prev[i];
        }
    }
    let a_j = if freeze_alpha && j < alpha.len() {
        alpha[j]
    } else {
        workspace.q_curr.dot(&workspace.w)
    };
    if j < alpha.len() {
        alpha[j] = a_j;
    } else {
        alpha.push(a_j);
    }
    for i in 0..workspace.w.len() {
        workspace.w[i] -= a_j * workspace.q_curr[i];
    }
    let v = &workspace.basis;
    if ncv < 20 || (j < 3 && n < 1024) {
        lanczos_full_reorthogonalize(workspace.w.as_slice_mut().unwrap(), v, j, locked);
    } else {
        let eta = selective_reorthogonalize_threshold(lanczos_anorm(alpha, beta));
        lanczos_selective_reorthogonalize(
            workspace.w.as_slice_mut().unwrap(),
            v,
            j,
            locked,
            eta,
        );
    }
    let b_j = l2_norm(&workspace.w);
    if j + 1 < ncv {
        if j < beta.len() {
            beta[j] = b_j;
        } else {
            beta.push(b_j);
        }
        if b_j > 1e-15 {
            std::mem::swap(&mut workspace.q_prev, &mut workspace.q_curr);
            for i in 0..workspace.w.len() {
                workspace.q_curr[i] = workspace.w[i] / b_j;
            }
            workspace.basis.column_mut(j + 1).assign(&workspace.q_curr);
            true
        } else {
            false
        }
    } else {
        true
    }
}

/// ``V[:, :keep] ← V[:, :m] @ Q[:, :keep]`` after implicit restart (ARPACK compression).
fn rotate_lanczos_basis(
    workspace: &mut LanczosWorkspace,
    q_acc: &Array2<f64>,
    m: usize,
    keep: usize,
    n: usize,
) {
    let v = &workspace.basis;
    let scratch = &mut workspace.basis_scratch;
    for c in 0..keep {
        for i in 0..n {
            let mut s = 0.0;
            for r in 0..m {
                s += v[[i, r]] * q_acc[[r, c]];
            }
            scratch[[i, c]] = s;
        }
    }
    for c in 0..keep {
        workspace.basis.column_mut(c).assign(&scratch.column(c));
    }
}

/// Build or extend a length-``ncv`` Lanczos factorization.
///
/// When ``keep == 0``, start from ``q0``. Otherwise continue from a compressed
/// factorization (rotated basis + truncated ``alpha`` / ``beta`` from implicit QR).
fn lanczos_factorization<F>(
    matvec: &mut F,
    n: usize,
    ncv: usize,
    keep: usize,
    mut q0: Option<Array1<f64>>,
    mut alpha: Vec<f64>,
    mut beta: Vec<f64>,
    locked: &[Array1<f64>],
    workspace: &mut LanczosWorkspace,
) -> (usize, Vec<f64>, Vec<f64>)
where
    F: FnMut(&Array1<f64>, &mut Array1<f64>),
{
    let j_start = if keep == 0 {
        let mut q = q0.take().unwrap_or_else(|| Array1::<f64>::zeros(n));
        reorthogonalize_slice(q.as_slice_mut().unwrap(), locked);
        let n0 = l2_norm(&q);
        if n0 > 0.0 {
            q /= n0;
        }
        workspace.q_curr.assign(&q);
        workspace.basis.column_mut(0).assign(&workspace.q_curr);
        alpha.clear();
        beta.clear();
        0
    } else if keep >= ncv {
        return (alpha.len(), alpha, beta);
    } else {
        workspace.q_curr.assign(&workspace.basis.column(keep - 1));
        if keep >= 2 {
            workspace.q_prev.assign(&workspace.basis.column(keep - 2));
        } else {
            workspace.q_prev.fill(0.0);
        }
        keep - 1
    };
    let freeze_upto = if keep > 0 { keep.saturating_sub(1) } else { 0 };
    for j in j_start..ncv {
        let freeze_alpha = keep > 0 && j < freeze_upto;
        if !lanczos_step(
            matvec,
            j,
            n,
            ncv,
            locked,
            workspace,
            &mut alpha,
            &mut beta,
            freeze_alpha,
        ) {
            break;
        }
    }
    (alpha.len(), alpha, beta)
}

#[derive(Clone)]
struct RitzCandidate {
    index: usize,
    value: f64,
    residual: f64,
}

/// Implicitly restarted Lanczos eigensolver (ARPACK ``*seupd`` analogue).
pub fn eigsh<F>(
    mut matvec: F,
    n: usize,
    k: usize,
    which: EigWhich,
    options: EigshOptions,
) -> Result<(Array1<f64>, Array2<f64>), EigshError>
where
    F: FnMut(&Array1<f64>, &mut Array1<f64>),
{
    if n == 0 {
        return Err(EigshError::EmptyProblem);
    }
    if k == 0 || k >= n {
        return Err(EigshError::InvalidComponents);
    }

    let ncv = options.ncv.unwrap_or_else(|| choose_ncv(k, n, false));
    if ncv > n || (ncv <= k + 1 && ncv < n) {
        return Err(EigshError::InvalidComponents);
    }
    let maxiter = options.maxiter.unwrap_or(n.saturating_mul(10)).max(1);
    let tol = if options.tol > 0.0 {
        options.tol
    } else {
        arpack_machine_tol()
    };

    let seed = options.rng_seed;
    let mut q0_init = Some(match options.v0 {
        Some(v) => {
            if v.len() != n {
                return Err(EigshError::InvalidComponents);
            }
            normalize_start(v, n, seed)
        }
        None => normalize_start(svds_default_v0(n, seed), n, seed),
    });

    let mut locked_vals: Vec<f64> = Vec::with_capacity(k);
    let mut locked_vecs: Vec<Array1<f64>> = Vec::with_capacity(k);
    let mut last_candidates: Vec<RitzCandidate> = Vec::new();
    let mut workspace = LanczosWorkspace::new(n, ncv);
    let eigsh_debug = std::env::var_os("SKNETWORK_EIGSH_DEBUG").is_some();
    let profile = eigsh_profile_enabled();
    let mut prof = EigshProfile::default();
    let mut restart_count = 0usize;
    let mut compressed_keep = 0usize;
    let mut factor_alpha: Vec<f64> = Vec::new();
    let mut factor_beta: Vec<f64> = Vec::new();
    let continue_krylov = std::env::var_os("SKNETWORK_EIGSH_CONTINUE").is_some();

    for _iter in 0..maxiter {
        if eigsh_debug || profile {
            restart_count += 1;
            if profile {
                prof.restarts += 1;
            }
        }
        if locked_vals.len() >= k {
            break;
        }

        let mut profiled_matvec = |x: &Array1<f64>, out: &mut Array1<f64>| {
            if profile {
                let t0 = Instant::now();
                matvec(x, out);
                prof.matvec_us += t0.elapsed().as_micros() as u64;
                prof.matvec_calls += 1;
            } else {
                matvec(x, out);
            }
        };
        let extend_t0 = profile.then(Instant::now);
        let q_start = if compressed_keep == 0 {
            q0_init.take()
        } else {
            None
        };
        let (m, mut alpha, mut beta) = lanczos_factorization(
            &mut profiled_matvec,
            n,
            ncv,
            compressed_keep,
            q_start,
            factor_alpha,
            factor_beta,
            &locked_vecs,
            &mut workspace,
        );
        factor_alpha = alpha.clone();
        factor_beta = beta.clone();
        if let Some(t0) = extend_t0 {
            prof.extend_us += t0.elapsed().as_micros() as u64;
        }
        if m == 0 {
            compressed_keep = 0;
            break;
        }

        let tridiag_t0 = profile.then(Instant::now);
        let (ritz_vals, ritz_vecs_small) = tridiagonal_eigh(&alpha, &beta);
        if let Some(t0) = tridiag_t0 {
            prof.tridiag_us += t0.elapsed().as_micros() as u64;
        }
        workspace.store_ritz_decomposition(m, &ritz_vals, &ritz_vecs_small);
        let v = &workspace.basis;

        let ritz_t0 = profile.then(Instant::now);
        let mut ritz_order: Vec<usize> = (0..m).collect();
        ritz_order.sort_by(|&a, &b| cmp_which(which, ritz_vals[a], ritz_vals[b]));
        ritz_order.truncate((3 * k).min(m));
        let n_want = k.saturating_sub(locked_vals.len());
        let target_ritz = distinct_wanted_ritz_indices(
            ritz_vals.as_slice().unwrap(),
            m,
            n_want,
            which,
            tol.sqrt(),
        );
        let target_set: std::collections::HashSet<usize> = target_ritz.iter().copied().collect();
        let candidates: Vec<RitzCandidate> = ritz_order
            .into_iter()
            .map(|ridx| {
                let theta = ritz_vals[ridx];
                let bound =
                    tridiagonal_ritz_residual_bound(&beta, m, &ritz_vecs_small, ridx, theta);
                let residual = if target_set.contains(&ridx) && bound > tol {
                    build_ritz_vector_into(v, &ritz_vecs_small, ridx, m, &mut workspace.ritz_vec);
                    relative_residual_inplace(
                        &mut profiled_matvec,
                        &workspace.ritz_vec,
                        theta,
                        &mut workspace.matvec_out,
                    )
                } else {
                    bound
                };
                RitzCandidate {
                    index: ridx,
                    value: theta,
                    residual,
                }
            })
            .collect();
        if let Some(t0) = ritz_t0 {
            prof.ritz_us += t0.elapsed().as_micros() as u64;
        }

        for &ridx in &target_ritz {
            if locked_vals.len() >= k {
                break;
            }
            let theta = ritz_vals[ridx];
            let residual = candidates
                .iter()
                .find(|c| c.index == ridx)
                .map(|c| c.residual)
                .unwrap_or(f64::INFINITY);
            if residual > tol {
                continue;
            }
            build_ritz_vector_into(v, &ritz_vecs_small, ridx, m, &mut workspace.ritz_vec);
            if locked_vals
                .iter()
                .any(|&lv| eigenvalue_near(lv, theta, tol.sqrt()))
            {
                continue;
            }
            locked_vals.push(theta);
            locked_vecs.push(workspace.ritz_vec.clone());
        }

        if locked_vals.len() >= k {
            break;
        }

        // Full-space pass: when ``ncv >= n`` and Lanczos completed ``m == n``, lock Ritz pairs.
        if ncv >= n {
            if m == n {
                let full_target = distinct_wanted_ritz_indices(
                    ritz_vals.as_slice().unwrap(),
                    m,
                    k.saturating_sub(locked_vals.len()),
                    which,
                    tol.sqrt(),
                );
                for ridx in full_target {
                    if locked_vals.len() >= k {
                        break;
                    }
                    let theta = ritz_vals[ridx];
                    let residual = candidates
                        .iter()
                        .find(|c| c.index == ridx)
                        .map(|c| c.residual)
                        .unwrap_or(f64::INFINITY);
                    if residual > tol {
                        continue;
                    }
                    build_ritz_vector_into(v, &ritz_vecs_small, ridx, m, &mut workspace.ritz_vec);
                    if locked_vals
                        .iter()
                        .any(|&lv| eigenvalue_near(lv, theta, tol.sqrt()))
                    {
                        continue;
                    }
                    locked_vals.push(theta);
                    locked_vecs.push(workspace.ritz_vec.clone());
                }
            }
            if locked_vals.len() >= k {
                break;
            }
            // Breakdown (``m < n``) or missed modes: restart with a fresh start vector.
            let retry_seed = seed
                .map(|s| s.wrapping_add(restart_count as u64))
                .unwrap_or_else(rand::random);
            q0_init = Some(normalize_start(
                svds_default_v0(n, Some(retry_seed)),
                n,
                seed,
            ));
            factor_alpha.clear();
            factor_beta.clear();
            compressed_keep = 0;
            continue;
        }

        let restart_t0 = profile.then(Instant::now);
        // Implicit restart: apply shifts for unwanted Ritz values (sorted, ARPACK order).
        let mut wanted_restart: Vec<usize> = (0..m).collect();
        wanted_restart.sort_by(|&a, &b| cmp_which(which, ritz_vals[a], ritz_vals[b]));
        wanted_restart.truncate(k.saturating_sub(locked_vals.len()));
        let wanted_idx = wanted_restart;
        let mut unwanted: Vec<usize> = (0..m).collect();
        for &w in &wanted_idx {
            unwanted.retain(|&x| x != w);
        }
        unwanted.sort_by(|&a, &b| {
            ritz_vals[a]
                .partial_cmp(&ritz_vals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let np = unwanted.len();
        let keep = m.saturating_sub(np).max(1).min(m);
        let mut q_acc = Array2::<f64>::eye(m);
        let mut a = alpha.clone();
        let mut b = beta.clone();
        for &u_idx in &unwanted {
            let mu = ritz_vals[u_idx];
            let q_step = implicit_qr_step(&mut a, &mut b, mu);
            q_acc = dense_matmul(&q_acc, &q_step);
        }
        if continue_krylov {
            a.truncate(keep);
            b.truncate(keep.saturating_sub(1));
            rotate_lanczos_basis(&mut workspace, &q_acc, m, keep, n);
            factor_alpha = a;
            factor_beta = b;
            compressed_keep = keep;
        } else {
            let q0_coeffs: Vec<f64> = (0..m).map(|i| q_acc[[i, 0]]).collect();
            let mut new_q0 = Array1::<f64>::zeros(n);
            lanczos_basis_combo(v, &q0_coeffs, m, &mut new_q0);
            reorthogonalize_slice(new_q0.as_slice_mut().unwrap(), &locked_vecs);
            q0_init = Some(normalize_start(new_q0, n, seed));
            factor_alpha.clear();
            factor_beta.clear();
            compressed_keep = 0;
        }
        last_candidates = candidates;
        if let Some(t0) = restart_t0 {
            prof.implicit_restart_us += t0.elapsed().as_micros() as u64;
        }
    }

    if locked_vals.len() < k {
        let m_last = workspace.last_m;
        let last_vals: Vec<f64> = (0..m_last).map(|i| workspace.last_ritz_vals[i]).collect();
        let target_last = distinct_wanted_ritz_indices(
            &last_vals,
            m_last,
            k.saturating_sub(locked_vals.len()),
            which,
            tol.sqrt(),
        );
        for ridx in target_last {
            if locked_vals.len() >= k {
                break;
            }
            let theta = last_vals[ridx];
            let residual = last_candidates
                .iter()
                .find(|c| c.index == ridx)
                .map(|c| c.residual)
                .unwrap_or(f64::INFINITY);
            if residual > tol {
                continue;
            }
            build_ritz_vector_into(
                &workspace.basis,
                &workspace.last_ritz_vecs,
                ridx,
                m_last,
                &mut workspace.ritz_vec,
            );
            if locked_vals
                .iter()
                .any(|&lv| eigenvalue_near(lv, theta, tol.sqrt()))
            {
                continue;
            }
            locked_vals.push(theta);
            locked_vecs.push(workspace.ritz_vec.clone());
        }
    }

    if eigsh_debug {
        eprintln!(
            "eigsh: n={n} k={k} ncv={ncv} restarts={restart_count} locked={}",
            locked_vals.len()
        );
        if !locked_vals.is_empty() {
            let preview: Vec<String> = locked_vals
                .iter()
                .take(5)
                .map(|v| format!("{v:.6}"))
                .collect();
            eprintln!("eigsh: locked_vals[:5]=[{preview}]", preview = preview.join(", "));
        }
    }
    if profile {
        let ro_us = prof.extend_us.saturating_sub(prof.matvec_us);
        eprintln!(
            "eigsh_profile: restarts={} matvec_calls={} matvec_us={} extend_us={} ro_us={} tridiag_us={} ritz_us={} implicit_restart_us={}",
            prof.restarts,
            prof.matvec_calls,
            prof.matvec_us,
            prof.extend_us,
            ro_us,
            prof.tridiag_us,
            prof.ritz_us,
            prof.implicit_restart_us,
        );
    }

    if locked_vals.is_empty() {
        return Err(EigshError::NoConvergence);
    }

    // Partial convergence: return the best locked pairs found (ARPACK may return ``iparam[4] < k``).
    let mut order: Vec<usize> = (0..locked_vals.len()).collect();
    order.sort_by(|&a, &b| cmp_which(which, locked_vals[a], locked_vals[b]));
    order.truncate(k);

    let out_k = order.len();
    let mut out_vals = Array1::<f64>::zeros(out_k);
    let mut out_vecs = Array2::<f64>::zeros((n, out_k));
    for (out_i, &idx) in order.iter().enumerate() {
        out_vals[out_i] = locked_vals[idx];
        for r in 0..n {
            out_vecs[[r, out_i]] = locked_vecs[idx][r];
        }
    }
    let out_vecs = if options.orthonormalize {
        let qr_t0 = profile.then(Instant::now);
        let q = qr_orthonormalize(out_vecs);
        if let Some(t0) = qr_t0 {
            prof.output_qr_us += t0.elapsed().as_micros() as u64;
            eprintln!("eigsh_profile: output_qr_us={}", prof.output_qr_us);
        }
        q
    } else {
        out_vecs
    };
    Ok((out_vals, out_vecs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn diag_matvec(d: &[f64]) -> impl Fn(&Array1<f64>, &mut Array1<f64>) {
        let d = d.to_vec();
        move |x: &Array1<f64>, out: &mut Array1<f64>| {
            for i in 0..d.len() {
                out[i] = d[i] * x[i];
            }
        }
    }

    fn diag_matvec_out(d: &[f64], x: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(x.len());
        diag_matvec(d)(x, &mut out);
        out
    }

    /// Reference path for unit tests only (not used in production ``eigsh``).
    fn eigsh_dense_reference<F>(
        matvec: &F,
        n: usize,
        k: usize,
        which: EigWhich,
    ) -> (Array1<f64>, Array2<f64>)
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let mut dense = Array2::<f64>::zeros((n, n));
        for j in 0..n {
            let mut ej = Array1::<f64>::zeros(n);
            ej[j] = 1.0;
            let col = matvec(&ej);
            for i in 0..n {
                dense[[i, j]] = col[i];
            }
        }
        let (evals, vecs) = symmetric_eigh(dense);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| cmp_which(which, evals[a], evals[b]));
        order.truncate(k);
        let mut out_vals = Array1::<f64>::zeros(k);
        let mut out_vecs = Array2::<f64>::zeros((n, k));
        for (out_i, &idx) in order.iter().enumerate() {
            out_vals[out_i] = evals[idx];
            for r in 0..n {
                out_vecs[[r, out_i]] = vecs[[r, idx]];
            }
        }
        (out_vals, out_vecs)
    }

    fn gram_10x8_matvec() -> impl Fn(&Array1<f64>, &mut Array1<f64>) {
        let mut a_rows: Vec<Vec<usize>> = vec![Vec::new(); 10];
        for i in 0..10 {
            for j in 0..8 {
                if (i + j) % 3 == 0 {
                    a_rows[i].push(j);
                }
            }
        }
        move |x: &Array1<f64>, out: &mut Array1<f64>| {
            let mut ax = [0.0_f64; 10];
            for (i, cols) in a_rows.iter().enumerate() {
                for &j in cols {
                    ax[i] += x[j];
                }
            }
            out.fill(0.0);
            for (i, cols) in a_rows.iter().enumerate() {
                let ai = ax[i];
                if ai == 0.0 {
                    continue;
                }
                for &j in cols {
                    out[j] += ai;
                }
            }
        }
    }

    #[test]
    fn test_eigsh_full_space_finds_k_modes_after_breakdown() {
        // Gram matrix diag(9, 4, 1, 0, 0): some start vectors trigger early Lanczos
        // breakdown (``m < n``). ``ncv == n`` must still return ``k`` largest modes.
        let d = vec![9.0, 4.0, 1.0, 0.0, 0.0];
        for seed in 0..32u64 {
            let (vals, _) = eigsh(
                diag_matvec(&d),
                5,
                3,
                EigWhich::Lm,
                EigshOptions {
                    ncv: Some(5),
                    rng_seed: Some(seed),
                    orthonormalize: false,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(vals.len(), 3, "seed={seed}");
            assert!((vals[0] - 9.0).abs() < 1e-6, "seed={seed}");
            assert!((vals[1] - 4.0).abs() < 1e-6, "seed={seed}");
            assert!((vals[2] - 1.0).abs() < 1e-6, "seed={seed}");
        }
    }

    #[test]
    fn test_eigsh_largest() {
        let d = vec![1.0, 5.0, 3.0, 2.0];
        let (vals, vecs) = eigsh(
            diag_matvec(&d),
            4,
            2,
            EigWhich::Lm,
            EigshOptions::default(),
        )
        .unwrap();
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 5.0).abs() < 1e-6);
        assert!((vals[1] - 3.0).abs() < 1e-6);
        let v0 = vecs.column(0).to_owned();
        let av = diag_matvec_out(&d, &v0);
        let diff = &av - &(&v0 * vals[0]);
        assert!(l2_norm(&diff) < 1e-6);
    }

    #[test]
    fn test_eigsh_gram_10x8() {
        let gram = gram_10x8_matvec();
        let v0 = Array1::from_vec(vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8]);
        let (vals, _) = eigsh(
            gram,
            8,
            3,
            EigWhich::Lm,
            EigshOptions {
                v0: Some(v0),
                ncv: Some(8),
                ..Default::default()
            },
        )
        .unwrap();
        assert!((vals[0] - 12.0).abs() < 1e-5);
        assert!((vals[1] - 9.0).abs() < 1e-5);
        assert!((vals[2] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_eigsh_matches_dense_reference_on_gram() {
        let gram = gram_10x8_matvec();
        let gram_out = |x: &Array1<f64>| {
            let mut out = Array1::<f64>::zeros(8);
            gram(x, &mut out);
            out
        };
        let (dense_vals, _) = eigsh_dense_reference(&gram_out, 8, 3, EigWhich::Lm);
        let (vals, _) = eigsh(
            gram,
            8,
            3,
            EigWhich::Lm,
            EigshOptions {
                ncv: Some(8),
                rng_seed: Some(42),
                ..Default::default()
            },
        )
        .unwrap();
        for i in 0..3 {
            assert!((vals[i] - dense_vals[i]).abs() < 1e-4);
        }
    }

    /// Matrix-free IRLM on ``n=600`` (matvec-only, no dense operator).
    #[test]
    #[ignore = "stress test; run: cargo test --release test_eigsh_large_diagonal -- --ignored"]
    fn test_eigsh_large_diagonal() {
        let n = 600;
        let d: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let (vals, vecs) = eigsh(
            diag_matvec(&d),
            n,
            4,
            EigWhich::Lm,
            EigshOptions {
                ncv: Some(choose_ncv(4, n, false)),
                maxiter: Some(n),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - n as f64).abs() < 1.0);
        assert!((vals[1] - (n - 1) as f64).abs() < 1.0);
        assert!((vals[2] - (n - 2) as f64).abs() < 1.0);
        assert!((vals[3] - (n - 3) as f64).abs() < 1.0);
        let v0 = vecs.column(0).to_owned();
        let mut av = Array1::<f64>::zeros(v0.len());
        assert!(
            relative_residual_inplace(&mut diag_matvec(&d), &v0, vals[0], &mut av) < 1e-8
        );
    }

    /// Ill-conditioned diagonal: clustered eigenvalues stress selective reorthogonalization.
    #[test]
    fn test_eigsh_clustered_spectrum() {
        let d: Vec<f64> = vec![100.0, 100.0 + 1e-8, 50.0, 25.0, 10.0];
        let (vals, _) = eigsh(
            diag_matvec(&d),
            5,
            3,
            EigWhich::Lm,
            EigshOptions {
                v0: Some(Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5])),
                ncv: Some(5),
                ..Default::default()
            },
        )
        .unwrap();
        assert!((vals[0] - 100.0).abs() < 1e-4);
        assert!((vals[1] - 50.0).abs() < 1e-3);
        assert!((vals[2] - 25.0).abs() < 1e-3);
    }

    #[test]
    fn test_eigsh_smallest_algebraic() {
        let d = vec![4.0, 1.0, 3.0, 2.0];
        let (vals, _) = eigsh(
            diag_matvec(&d),
            4,
            2,
            EigWhich::Sa,
            EigshOptions::default(),
        )
        .unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);
    }
}
