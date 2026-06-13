//! Sparse partial SVD solvers.
//!
//! - ``LanczosSVD`` follows the ARPACK path of ``scipy.sparse.linalg.svds``.
//! - ``RandomizedSVD`` follows ``sklearn.utils.extmath.randomized_svd`` (Halko et al.).

use ndarray::{s, Array1, Array2, ArrayView2};

use crate::linalg::dense_linalg::{
    dense_matmul, qr_economic, thin_svd, transpose_matrix,
};
use crate::linalg::rng::{choose_ncv, standard_normal_matrix};
use crate::linalg::sparse_lowrank::SparseLR;
use crate::linalg::sparse_matvec::{csc_matvec_into, csr_matvec_into};
use crate::linalg::sparse_matvec_cache::{CachedSparseApplicator, SparseMatvecCache};
use crate::linalg::symmetric_eigsh::{EigWhich, EigshOptions, eigsh};
use sprs::CsMat;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by svderror operations.
pub enum SVDError {
    /// Indicates invalid components.
    InvalidComponents,
    /// Indicates empty matrix.
    EmptyMatrix,
}

#[derive(Debug, Clone)]
/// SVDInput enum.
pub enum SVDInput {
    /// Indicates sparse.
    Sparse(CsMat<f64>),
    /// Indicates sparse lr.
    SparseLR(SparseLR),
}

fn sparse_matvec_into_nd(
    a: &CsMat<f64>,
    x: &Array1<f64>,
    out: &mut Array1<f64>,
    ranges: Option<&[(usize, usize)]>,
) {
    csr_matvec_into(
        a,
        x.as_slice().unwrap(),
        out.as_slice_mut().unwrap(),
        ranges,
    );
}

fn sparse_t_matvec_into_nd(
    a: &CsMat<f64>,
    x: &Array1<f64>,
    out: &mut Array1<f64>,
    a_t_csc: Option<&CsMat<f64>>,
    t_ranges: Option<&[(usize, usize)]>,
) {
    if let Some(at) = a_t_csc {
        csc_matvec_into(
            at,
            x.as_slice().unwrap(),
            out.as_slice_mut().unwrap(),
            t_ranges,
        );
        return;
    }
    out.fill(0.0);
    for (i, row) in a.outer_iterator().enumerate() {
        let xi = x[i];
        if xi == 0.0 {
            continue;
        }
        for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
            out[j] += v * xi;
        }
    }
}

/// Computes sparse matvec.
fn sparse_matvec(a: &CsMat<f64>, x: &Array1<f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(a.rows());
    sparse_matvec_into_nd(a, x, &mut out, None);
    out
}

/// Computes sparse t matvec.
fn sparse_t_matvec(a: &CsMat<f64>, x: &Array1<f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(a.cols());
    sparse_t_matvec_into_nd(a, x, &mut out, None, None);
    out
}

fn apply_a_into(
    input: &SVDInput,
    x: &Array1<f64>,
    out: &mut Array1<f64>,
    ranges: Option<&[(usize, usize)]>,
) {
    match input {
        SVDInput::Sparse(a) => sparse_matvec_into_nd(a, x, out, ranges),
        SVDInput::SparseLR(a) => {
            let y = a.dot_vec(x);
            out.assign(&y);
        }
    }
}

fn apply_at_into(input: &SVDInput, x: &Array1<f64>, out: &mut Array1<f64>) {
    apply_at_into_cached(input, x, out, None, None);
}

fn apply_at_into_cached(
    input: &SVDInput,
    x: &Array1<f64>,
    out: &mut Array1<f64>,
    a_t_csc: Option<&CsMat<f64>>,
    t_ranges: Option<&[(usize, usize)]>,
) {
    match input {
        SVDInput::Sparse(a) => sparse_t_matvec_into_nd(a, x, out, a_t_csc, t_ranges),
        SVDInput::SparseLR(a) => {
            let y = a.transpose().dot_vec(x);
            out.assign(&y);
        }
    }
}

fn apply_a(input: &SVDInput, x: &Array1<f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(rows(input));
    apply_a_into(input, x, &mut out, None);
    out
}

fn apply_at(input: &SVDInput, x: &Array1<f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(cols(input));
    apply_at_into(input, x, &mut out);
    out
}

/// Low-rank / generic Gram matvec (no CSR cache).
struct GramMatvec {
    scratch: Array1<f64>,
    transpose: bool,
}

impl GramMatvec {
    fn new(n_row: usize, n_col: usize, transpose: bool) -> Self {
        Self {
            scratch: Array1::<f64>::zeros(if transpose { n_col } else { n_row }),
            transpose,
        }
    }

    fn apply_into(&mut self, input: &SVDInput, x: &Array1<f64>, out: &mut Array1<f64>) {
        if self.transpose {
            apply_at_into(input, x, &mut self.scratch);
            apply_a_into(input, &self.scratch, out, None);
        } else {
            apply_a_into(input, x, &mut self.scratch, None);
            apply_at_into(input, &self.scratch, out);
        }
    }
}

fn apply_a_mat(input: &SVDInput, b: &Array2<f64>) -> Array2<f64> {
    let k = b.ncols();
    let mut out = Array2::<f64>::zeros((rows(input), k));
    for c in 0..k {
        let col = b.column(c).to_owned();
        let av = apply_a(input, &col);
        for i in 0..av.len() {
            out[[i, c]] = av[i];
        }
    }
    out
}

fn apply_at_mat(input: &SVDInput, b: &Array2<f64>) -> Array2<f64> {
    let k = b.ncols();
    let mut out = Array2::<f64>::zeros((cols(input), k));
    for c in 0..k {
        let col = b.column(c).to_owned();
        let atv = apply_at(input, &col);
        for j in 0..atv.len() {
            out[[j, c]] = atv[j];
        }
    }
    out
}

/// ``Qᵀ @ A`` when ``Q`` has shape ``(n_row, n_random)``.
fn qt_times_a(input: &SVDInput, q: &Array2<f64>, transposed: bool) -> Array2<f64> {
    let n_random = q.ncols();
    let out_cols = if transposed { rows(input) } else { cols(input) };
    let mut b = Array2::<f64>::zeros((n_random, out_cols));
    for c in 0..n_random {
        let q_col = q.column(c).to_owned();
        let row = if transposed {
            apply_a(input, &q_col)
        } else {
            apply_at(input, &q_col)
        };
        for j in 0..out_cols {
            b[[c, j]] = row[j];
        }
    }
    b
}

fn store_svd_triplets(
    singular_vectors_left: &mut Option<Array2<f64>>,
    singular_values: &mut Option<Array1<f64>>,
    singular_vectors_right: &mut Option<Array2<f64>>,
    u: Array2<f64>,
    s: Array1<f64>,
    vh: Array2<f64>,
    n_col: usize,
) {
    *singular_vectors_left = Some(u);
    *singular_values = Some(s);
    let k = vh.nrows();
    let mut v = Array2::<f64>::zeros((n_col, k));
    for i in 0..k {
        for j in 0..n_col {
            v[[j, i]] = vh[[i, j]];
        }
    }
    *singular_vectors_right = Some(v);
}

fn validate_svd_fit(matrix: &SVDInput, n_components: usize) -> Result<(usize, usize), SVDError> {
    let (n_row, n_col) = dims(matrix);
    if n_row == 0 || n_col == 0 {
        return Err(SVDError::EmptyMatrix);
    }
    let kmax = n_row.min(n_col).saturating_sub(1).max(1);
    if n_components == 0 || n_components > kmax {
        return Err(SVDError::InvalidComponents);
    }
    Ok((n_row, n_col))
}

/// Power-iteration normalizer for the Halko range finder (sklearn ``power_iteration_normalizer``).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PowerIterationNormalizer {
    #[default]
    /// Indicates auto.
    Auto,
    /// Indicates qr.
    Qr,
    /// Indicates lu.
    Lu,
    /// Disables the selected orthogonalization backend.
    None,
}

/// Transpose policy for randomized SVD (sklearn ``transpose``).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransposeMode {
    #[default]
    /// Indicates auto.
    Auto,
    /// Indicates yes.
    Yes,
    /// Indicates no.
    No,
}

impl TransposeMode {
    fn resolve(self, n_row: usize, n_col: usize) -> bool {
        match self {
            TransposeMode::Auto => n_row < n_col,
            TransposeMode::Yes => true,
            TransposeMode::No => false,
        }
    }
}

fn resolve_n_iter(n_iter: Option<usize>, n_components: usize, n_row: usize, n_col: usize) -> usize {
    n_iter.unwrap_or_else(|| {
        let min_dim = n_row.min(n_col);
        if n_components < ((0.1 * min_dim as f64) as usize) {
            7
        } else {
            4
        }
    })
}

fn resolve_power_normalizer(mode: PowerIterationNormalizer, n_iter: usize) -> PowerIterationNormalizer {
    match mode {
        PowerIterationNormalizer::Auto => {
            if n_iter <= 2 {
                PowerIterationNormalizer::None
            } else {
                PowerIterationNormalizer::Lu
            }
        }
        other => other,
    }
}

fn normalize_power_step(y: Array2<f64>, mode: PowerIterationNormalizer) -> Array2<f64> {
    match mode {
        PowerIterationNormalizer::None => y,
        PowerIterationNormalizer::Qr | PowerIterationNormalizer::Lu => qr_economic(y),
        PowerIterationNormalizer::Auto => y,
    }
}

/// In-place power-iteration normalizer; reuses ``q`` when shapes match (``None`` mode).
fn normalize_power_step_into(
    src: ArrayView2<f64>,
    q: &mut Array2<f64>,
    mode: PowerIterationNormalizer,
) {
    match mode {
        PowerIterationNormalizer::None => {
            if q.shape() == src.shape() {
                q.assign(&src);
            } else {
                *q = src.to_owned();
            }
        }
        PowerIterationNormalizer::Qr | PowerIterationNormalizer::Lu => {
            *q = qr_economic(src.to_owned());
        }
        PowerIterationNormalizer::Auto => q.assign(&src),
    }
}

/// Deterministic sign convention (sklearn ``svd_flip``).
fn svd_flip(u: &mut Array2<f64>, vh: &mut Array2<f64>, u_based_decision: bool) {
    let k = u.ncols();
    for c in 0..k {
        let sign = if u_based_decision {
            let mut max_abs = 0.0_f64;
            let mut max_val = 0.0_f64;
            for r in 0..u.nrows() {
                let v = u[[r, c]];
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                    max_val = v;
                }
            }
            if max_val >= 0.0 { 1.0 } else { -1.0 }
        } else {
            let mut max_abs = 0.0_f64;
            let mut max_val = 0.0_f64;
            for j in 0..vh.ncols() {
                let v = vh[[c, j]];
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                    max_val = v;
                }
            }
            if max_val >= 0.0 { 1.0 } else { -1.0 }
        };
        if sign < 0.0 {
            for r in 0..u.nrows() {
                u[[r, c]] *= -1.0;
            }
            for j in 0..vh.ncols() {
                vh[[c, j]] *= -1.0;
            }
        }
    }
}

fn randomized_range_finder_cached(
    a: &CsMat<f64>,
    applicator: &mut CachedSparseApplicator,
    n_row: usize,
    n_col: usize,
    size: usize,
    n_iter: usize,
    normalizer: PowerIterationNormalizer,
    transposed: bool,
    seed: Option<u64>,
) -> Array2<f64> {
    let n_features = if transposed { n_row } else { n_col };
    let mut q = standard_normal_matrix(n_features, size, seed);
    let mut buf_row = Array2::<f64>::zeros((n_row, size));
    let mut buf_col = Array2::<f64>::zeros((n_col, size));

    let step_norm = resolve_power_normalizer(normalizer, n_iter);
    for _ in 0..n_iter {
        if transposed {
            applicator.apply_at_mat_into(a, &q, &mut buf_col);
            let k = q.ncols();
            normalize_power_step_into(buf_col.slice(s![.., 0..k]), &mut q, step_norm);
            applicator.apply_a_mat_into(a, &q, &mut buf_row);
            let k = q.ncols();
            normalize_power_step_into(buf_row.slice(s![.., 0..k]), &mut q, step_norm);
        } else {
            applicator.apply_a_mat_into(a, &q, &mut buf_row);
            let k = q.ncols();
            normalize_power_step_into(buf_row.slice(s![.., 0..k]), &mut q, step_norm);
            applicator.apply_at_mat_into(a, &q, &mut buf_col);
            let k = q.ncols();
            normalize_power_step_into(buf_col.slice(s![.., 0..k]), &mut q, step_norm);
        }
    }

    if transposed {
        applicator.apply_at_mat_into(a, &q, &mut buf_col);
        let k = q.ncols();
        qr_economic(buf_col.slice(s![.., 0..k]).to_owned())
    } else {
        applicator.apply_a_mat_into(a, &q, &mut buf_row);
        let k = q.ncols();
        qr_economic(buf_row.slice(s![.., 0..k]).to_owned())
    }
}

fn randomized_range_finder(
    input: &SVDInput,
    size: usize,
    n_iter: usize,
    normalizer: PowerIterationNormalizer,
    transposed: bool,
    seed: Option<u64>,
) -> Array2<f64> {
    let n_features = if transposed { rows(input) } else { cols(input) };
    let mut q = standard_normal_matrix(n_features, size, seed);

    let step_norm = resolve_power_normalizer(normalizer, n_iter);
    for _ in 0..n_iter {
        q = if transposed {
            apply_at_mat(input, &q)
        } else {
            apply_a_mat(input, &q)
        };
        q = normalize_power_step(q, step_norm);
        q = if transposed {
            apply_a_mat(input, &q)
        } else {
            apply_at_mat(input, &q)
        };
        q = normalize_power_step(q, step_norm);
    }

    q = if transposed {
        apply_at_mat(input, &q)
    } else {
        apply_a_mat(input, &q)
    };
    qr_economic(q)
}

/// Halko randomized partial SVD (``sklearn.utils.extmath.randomized_svd``).
fn svds_halko(
    input: &SVDInput,
    k: usize,
    n_oversamples: usize,
    n_iter: Option<usize>,
    normalizer: PowerIterationNormalizer,
    transpose: TransposeMode,
    flip_sign: bool,
    rng_seed: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), SVDError> {
    let (n_row, n_col) = dims(input);
    let transposed = transpose.resolve(n_row, n_col);
    let n_iter = resolve_n_iter(n_iter, k, n_row, n_col);
    let n_random = k + n_oversamples;

    let (q, b) = match input {
        SVDInput::Sparse(a) => {
            let mut applicator = CachedSparseApplicator::new(a);
            let q = randomized_range_finder_cached(
                a,
                &mut applicator,
                n_row,
                n_col,
                n_random,
                n_iter,
                normalizer,
                transposed,
                rng_seed,
            );
            let out_cols = if transposed { n_row } else { n_col };
            let mut b = Array2::<f64>::zeros((n_random, out_cols));
            applicator.qt_times_a_into(a, &q, transposed, &mut b);
            (q, b)
        }
        SVDInput::SparseLR(_) => {
            let q = randomized_range_finder(
                input,
                n_random,
                n_iter,
                normalizer,
                transposed,
                rng_seed,
            );
            let b = qt_times_a(input, &q, transposed);
            (q, b)
        }
    };
    let (u_hat, s_full, vh_full) = thin_svd(&b);
    // ``thin_svd`` returns singular values in increasing order; keep the top ``k``.
    let r = s_full.len();
    let start = r.saturating_sub(k);
    let s = s_full.slice(ndarray::s![start..]).to_owned();
    let mut u_hat_top = Array2::<f64>::zeros((u_hat.nrows(), k));
    for c in 0..k {
        for row in 0..u_hat.nrows() {
            u_hat_top[[row, c]] = u_hat[[row, start + c]];
        }
    }
    let mut u = dense_matmul(&q, &u_hat_top);
    let mut vh = Array2::<f64>::zeros((k, vh_full.ncols()));
    for i in 0..k {
        for j in 0..vh.ncols() {
            vh[[i, j]] = vh_full[[start + i, j]];
        }
    }

    if flip_sign {
        if transposed {
            svd_flip(&mut u, &mut vh, false);
        } else {
            svd_flip(&mut u, &mut vh, true);
        }
    }

    let (mut u_out, mut s_out, mut vh_out) = if transposed {
        let u_t = transpose_matrix(&u);
        let vh_t = transpose_matrix(&vh);
        (vh_t, s, u_t)
    } else {
        (u, s, vh)
    };
    sort_singular_decreasing(&mut u_out, &mut s_out, &mut vh_out);
    Ok((u_out, s_out, vh_out))
}

fn rows(input: &SVDInput) -> usize {
    match input {
        SVDInput::Sparse(a) => a.rows(),
        SVDInput::SparseLR(a) => a.sparse_mat.rows(),
    }
}

fn cols(input: &SVDInput) -> usize {
    match input {
        SVDInput::Sparse(a) => a.cols(),
        SVDInput::SparseLR(a) => a.sparse_mat.cols(),
    }
}

fn dims(input: &SVDInput) -> (usize, usize) {
    (rows(input), cols(input))
}

/// Reorder singular triplets by decreasing ``σ`` (``sknetwork.linalg.LanczosSVD``).
fn sort_singular_decreasing(u: &mut Array2<f64>, s: &mut Array1<f64>, vh: &mut Array2<f64>) {
    let k = s.len();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| {
        s[b]
            .partial_cmp(&s[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let u_old = u.clone();
    let vh_old = vh.clone();
    let s_old = s.clone();
    for (new_c, &old_c) in order.iter().enumerate() {
        s[new_c] = s_old[old_c];
        for r in 0..u.nrows() {
            u[[r, new_c]] = u_old[[r, old_c]];
        }
        for c in 0..vh.ncols() {
            vh[[new_c, c]] = vh_old[[old_c, c]];
        }
    }
}

/// SciPy ``svds`` (``solver='arpack'``) on a matrix-free operator.
fn svds_arpack(
    input: &SVDInput,
    k: usize,
    tol: f64,
    init_vector: Option<Array1<f64>>,
    n_iter: Option<usize>,
    rng_seed: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), SVDError> {
    let (n_row, n_col) = dims(input);
    let transpose = n_row < n_col;
    let min_dim = n_row.min(n_col);

    // SciPy ``svds`` passes ``tol ** 2`` to ``eigsh``; when ``tol=0`` both map to machine precision.
    let eig_tol = if tol > 0.0 { tol * tol } else { 0.0 };
    let (eigvec, av) = match input {
        SVDInput::Sparse(a) => {
            let cache = Arc::new(SparseMatvecCache::new(a));
            let symmetric_adjacency = cache.is_symmetric();
            let eig_opts = EigshOptions {
                ncv: Some(choose_ncv(k, min_dim, symmetric_adjacency)),
                maxiter: n_iter,
                tol: eig_tol,
                v0: init_vector.clone(),
                rng_seed,
                orthonormalize: false,
            };
            let cache_eig = Arc::clone(&cache);
            let a_owned = a.clone();
            let mut scratch_row = Array1::<f64>::zeros(n_row);
            let mut scratch_col = Array1::<f64>::zeros(n_col);
            let transpose_eig = transpose;
            let mut gram_matvec = move |x: &Array1<f64>, out: &mut Array1<f64>| {
                cache_eig.gram_matvec_nd(
                    &a_owned,
                    transpose_eig,
                    x,
                    &mut scratch_row,
                    &mut scratch_col,
                    out,
                );
            };
            let profile = std::env::var_os("SKNETWORK_EIGSH_PROFILE").is_some();
            let eig_t0 = profile.then(Instant::now);
            let (_eigvals, eigvec) =
                eigsh(gram_matvec, min_dim, k, EigWhich::Lm, eig_opts).map_err(|_| SVDError::InvalidComponents)?;
            if let Some(t0) = eig_t0 {
                eprintln!(
                    "svds_profile: eigsh_us={}",
                    t0.elapsed().as_micros()
                );
            }
            let out_k = eigvec.ncols();
            let mut av = if transpose {
                Array2::<f64>::zeros((n_col, out_k))
            } else {
                Array2::<f64>::zeros((n_row, out_k))
            };
            let mut post_x = Array1::<f64>::zeros(eigvec.nrows());
            let mut post_y = Array1::<f64>::zeros(if transpose { n_col } else { n_row });
            let post_t0 = profile.then(Instant::now);
            if transpose {
                cache.matvec_at_matrix_into(a, &eigvec, &mut av, &mut post_x, &mut post_y);
            } else {
                cache.matvec_a_matrix_into(a, &eigvec, &mut av, &mut post_x, &mut post_y);
            }
            if let Some(t0) = post_t0 {
                eprintln!("svds_profile: post_matvec_us={}", t0.elapsed().as_micros());
            }
            Ok((eigvec, av))
        }
        SVDInput::SparseLR(_) => {
            let eig_opts = EigshOptions {
                ncv: Some(choose_ncv(k, min_dim, false)),
                maxiter: n_iter,
                tol: eig_tol,
                v0: init_vector,
                rng_seed,
                orthonormalize: false,
            };
            let mut gram = GramMatvec::new(n_row, n_col, transpose);
            let input_ref = input;
            let mut gram_matvec = move |x: &Array1<f64>, out: &mut Array1<f64>| {
                gram.apply_into(input_ref, x, out);
            };
            let (_eigvals, eigvec) =
                eigsh(gram_matvec, min_dim, k, EigWhich::Lm, eig_opts).map_err(|_| SVDError::InvalidComponents)?;
            let av = if transpose {
                let mut out = Array2::<f64>::zeros((n_col, eigvec.ncols()));
                for c in 0..eigvec.ncols() {
                    let col = eigvec.column(c).to_owned();
                    let y = apply_at(input_ref, &col);
                    for i in 0..y.len() {
                        out[[i, c]] = y[i];
                    }
                }
                out
            } else {
                apply_a_mat(input_ref, &eigvec)
            };
            Ok((eigvec, av))
        }
    }?;
    let profile = std::env::var_os("SKNETWORK_EIGSH_PROFILE").is_some();
    let pack_t0 = profile.then(Instant::now);
    let (mut u_out, mut s_out, mut vh_out) = scipy_arpack_svd_pack(&eigvec, &av, transpose);
    if let Some(t0) = pack_t0 {
        eprintln!("svds_profile: pack_svd_us={}", t0.elapsed().as_micros());
    }
    sort_singular_decreasing(&mut u_out, &mut s_out, &mut vh_out);
    Ok((u_out, s_out, vh_out))
}

/// Reverse thin-SVD triplets to decreasing ``σ`` (SciPy ``svds`` lines after ``svd(Av)``).
fn reverse_svd_triplets(u: &mut Array2<f64>, s: &mut Array1<f64>, vh: &mut Array2<f64>) {
    let k = s.len();
    for c in 0..k / 2 {
        let rev = k - 1 - c;
        s.swap(c, rev);
        for r in 0..u.nrows() {
            u.swap([r, c], [r, rev]);
        }
        for j in 0..vh.ncols() {
            vh.swap([c, j], [rev, j]);
        }
    }
}

/// SciPy ``svds`` ARPACK Rayleigh–Ritz step: ``svd(X @ eigvec)`` then combine with ``eigvec``.
fn scipy_arpack_svd_pack(
    eigvec: &Array2<f64>,
    av: &Array2<f64>,
    transpose: bool,
) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let (mut u_av, mut s, mut vh_av) = thin_svd(av);
    reverse_svd_triplets(&mut u_av, &mut s, &mut vh_av);
    if transpose {
        let u_out = dense_matmul(eigvec, &transpose_matrix(&vh_av));
        let vh_out = transpose_matrix(&u_av);
        (u_out, s, vh_out)
    } else {
        let vh_out = dense_matmul(&vh_av, &transpose_matrix(eigvec));
        (u_av, s, vh_out)
    }
}

/// Which partial SVD backend to use (``sknetwork`` ``solver`` string).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SvdSolverKind {
    #[default]
    /// Indicates lanczos.
    Lanczos,
    /// Indicates halko.
    Halko,
}

impl SvdSolverKind {
    /// Computes parse.
    pub fn parse(name: &str) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "halko" | "randomized" => Self::Halko,
            _ => Self::Lanczos,
        }
    }
}

/// Singular triplets returned by [`fit_partial_svd`].
#[derive(Debug, Clone)]
pub struct PartialSvdResult {
    /// U value.
    pub u: Array2<f64>,
    /// S value.
    pub s: Array1<f64>,
    /// V value.
    pub v: Array2<f64>,
}

/// Run a partial SVD on ``input`` with the selected backend.
pub fn fit_partial_svd(
    kind: SvdSolverKind,
    input: SVDInput,
    n_components: usize,
    random_state: Option<u64>,
) -> Result<PartialSvdResult, SVDError> {
    validate_svd_fit(&input, n_components)?;
    let n_col = cols(&input);
    match kind {
        SvdSolverKind::Lanczos => {
            let mut solver = LanczosSVD::default().with_random_state(random_state);
            solver.fit(input, n_components, None)?;
            Ok(PartialSvdResult {
                u: solver.singular_vectors_left.unwrap(),
                s: solver.singular_values.unwrap(),
                v: solver.singular_vectors_right.unwrap(),
            })
        }
        SvdSolverKind::Halko => {
            let mut solver = RandomizedSVD::default().with_random_state(random_state);
            solver.fit(input, n_components, None)?;
            Ok(PartialSvdResult {
                u: solver.singular_vectors_left.unwrap(),
                s: solver.singular_values.unwrap(),
                v: solver.singular_vectors_right.unwrap(),
            })
        }
    }
}

#[derive(Debug, Clone)]
/// SVDSolver value.
pub struct SVDSolver {
    /// Singular Vectors Left value.
    pub singular_vectors_left: Option<Array2<f64>>,
    /// Singular Vectors Right value.
    pub singular_vectors_right: Option<Array2<f64>>,
    /// Singular Values value.
    pub singular_values: Option<Array1<f64>>,
}

impl Default for SVDSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SVDSolver {
    /// Creates a new instance.
    pub fn new() -> Self {
        Self {
            singular_vectors_left: None,
            singular_vectors_right: None,
            singular_values: None,
        }
    }
}

#[derive(Debug, Clone)]
/// LanczosSVD value.
pub struct LanczosSVD {
    /// N Iter value.
    pub n_iter: Option<usize>,
    /// Tol value.
    pub tol: f64,
    /// Seed for the default ``v0`` vector (SciPy ``svds`` uses ``numpy.random.Generator``).
    pub random_state: Option<u64>,
    /// Singular Vectors Left value.
    pub singular_vectors_left: Option<Array2<f64>>,
    /// Singular Vectors Right value.
    pub singular_vectors_right: Option<Array2<f64>>,
    /// Singular Values value.
    pub singular_values: Option<Array1<f64>>,
}

impl Default for LanczosSVD {
    fn default() -> Self {
        Self::new(None, 0.0)
    }
}

impl LanczosSVD {
    /// Creates a new instance.
    pub fn new(n_iter: Option<usize>, tol: f64) -> Self {
        Self {
            n_iter,
            tol,
            random_state: None,
            singular_vectors_left: None,
            singular_vectors_right: None,
            singular_values: None,
        }
    }

    /// Computes with random state.
    pub fn with_random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Runs the fit step.
    pub fn fit(
        &mut self,
        matrix: SVDInput,
        n_components: usize,
        init_vector: Option<Array1<f64>>,
    ) -> Result<(), SVDError> {
        validate_svd_fit(&matrix, n_components)?;

        let (u, s, vh) = svds_arpack(
            &matrix,
            n_components,
            self.tol,
            init_vector,
            self.n_iter,
            self.random_state,
        )?;
        store_svd_triplets(
            &mut self.singular_vectors_left,
            &mut self.singular_values,
            &mut self.singular_vectors_right,
            u,
            s,
            vh,
            cols(&matrix),
        );
        Ok(())
    }
}

/// Approximate partial SVD via Halko's randomized range-finding algorithm.
///
/// Defaults match ``sklearn.utils.extmath.randomized_svd``:
/// ``n_oversamples=10``, ``n_iter='auto'`` (4 or 7), ``transpose='auto'``,
/// ``flip_sign=True``.
#[derive(Debug, Clone)]
pub struct RandomizedSVD {
    /// N Oversamples value.
    pub n_oversamples: usize,
    /// ``None`` selects sklearn's auto rule (4 or 7 power iterations).
    pub n_iter: Option<usize>,
    /// Power Iteration Normalizer value.
    pub power_iteration_normalizer: PowerIterationNormalizer,
    /// Transpose value.
    pub transpose: TransposeMode,
    /// Flip Sign value.
    pub flip_sign: bool,
    /// Random State value.
    pub random_state: Option<u64>,
    /// Singular Vectors Left value.
    pub singular_vectors_left: Option<Array2<f64>>,
    /// Singular Vectors Right value.
    pub singular_vectors_right: Option<Array2<f64>>,
    /// Singular Values value.
    pub singular_values: Option<Array1<f64>>,
}

impl Default for RandomizedSVD {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomizedSVD {
    /// Creates a new instance.
    pub fn new() -> Self {
        Self {
            n_oversamples: 10,
            n_iter: None,
            power_iteration_normalizer: PowerIterationNormalizer::Auto,
            transpose: TransposeMode::Auto,
            flip_sign: true,
            random_state: None,
            singular_vectors_left: None,
            singular_vectors_right: None,
            singular_values: None,
        }
    }

    /// Computes with random state.
    pub fn with_random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Runs the fit step.
    pub fn fit(
        &mut self,
        matrix: SVDInput,
        n_components: usize,
        _init_vector: Option<Array1<f64>>,
    ) -> Result<(), SVDError> {
        validate_svd_fit(&matrix, n_components)?;
        let (u, s, vh) = svds_halko(
            &matrix,
            n_components,
            self.n_oversamples,
            self.n_iter,
            self.power_iteration_normalizer,
            self.transpose,
            self.flip_sign,
            self.random_state,
        )?;
        store_svd_triplets(
            &mut self.singular_vectors_left,
            &mut self.singular_values,
            &mut self.singular_vectors_right,
            u,
            s,
            vh,
            cols(&matrix),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use sprs::TriMat;

    use super::*;
    use crate::data::test_graphs::test_bigraph;

    fn svd_err(matrix: &SVDInput, u: &Array2<f64>, v: &Array2<f64>, sigma: &Array1<f64>) -> f64 {
        let mut err = 0.0;
        for c in 0..v.ncols() {
            let vc = v.column(c).to_owned();
            let av = apply_a(matrix, &vc);
            for i in 0..av.len() {
                let d = av[i] - u[[i, c]] * sigma[c];
                err += d * d;
            }
        }
        err.sqrt()
    }

    #[test]
    fn test_lanczos() {
        let biadjacency = test_bigraph();
        let mut solver = LanczosSVD::default().with_random_state(Some(42));
        solver
            .fit(SVDInput::Sparse(biadjacency.clone()), 2, None)
            .unwrap();
        assert_eq!(solver.singular_values.as_ref().unwrap().len(), 2);
        let err = svd_err(
            &SVDInput::Sparse(biadjacency.clone()),
            solver.singular_vectors_left.as_ref().unwrap(),
            solver.singular_vectors_right.as_ref().unwrap(),
            solver.singular_values.as_ref().unwrap(),
        );
        assert!(err < 1e-3);

        let (n_row, n_col) = biadjacency.shape();
        let x = Array1::from_vec((0..n_row).map(|i| ((i + 1) as f64 * 0.13).sin()).collect());
        let y = Array1::from_vec((0..n_col).map(|j| ((j + 1) as f64 * 0.19).cos()).collect());
        let slr = SparseLR::new(&biadjacency, vec![(x, y)]).unwrap();
        solver
            .fit(SVDInput::SparseLR(slr.clone()), 2, None)
            .unwrap();
        assert_eq!(solver.singular_values.as_ref().unwrap().len(), 2);
        let err = svd_err(
            &SVDInput::SparseLR(slr),
            solver.singular_vectors_left.as_ref().unwrap(),
            solver.singular_vectors_right.as_ref().unwrap(),
            solver.singular_values.as_ref().unwrap(),
        );
        assert!(err < 1e-3);
    }

    #[test]
    fn test_invalid_components() {
        let mut tri = TriMat::<f64>::new((2, 3));
        tri.add_triplet(0, 0, 1.0);
        let a = tri.to_csr::<usize>();
        let mut solver = LanczosSVD::default();
        assert!(matches!(
            solver.fit(SVDInput::Sparse(a), 0, None),
            Err(SVDError::InvalidComponents)
        ));
    }

    /// Reference values from ``scipy.sparse.linalg.svds`` on the same CSR pattern and ``v0``.
    #[test]
    fn test_scipy_parity_fixed_v0() {
        let mut tri = TriMat::<f64>::new((10, 8));
        for i in 0..10 {
            for j in 0..8 {
                if (i + j) % 3 == 0 {
                    tri.add_triplet(i, j, 1.0);
                }
            }
        }
        let a = tri.to_csr::<usize>();
        let v0 = Array1::from_vec(vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8]);
        let mut solver = LanczosSVD::default();
        solver
            .fit(SVDInput::Sparse(a.clone()), 3, Some(v0))
            .unwrap();
        let s = solver.singular_values.as_ref().unwrap();
        let u = solver.singular_vectors_left.as_ref().unwrap();
        let v = solver.singular_vectors_right.as_ref().unwrap();
        assert!((s[0] - 3.4641016151).abs() < 1e-6);
        assert!((s[1] - 3.0).abs() < 1e-6);
        assert!((s[2] - 2.4494897428).abs() < 1e-6);
        assert!((u[[0, 0]].abs() - 0.5).abs() < 1e-5);
        assert!((u[[3, 0]].abs() - 0.5).abs() < 1e-5);
        assert!((v[[0, 0]].abs() - 0.57735026919).abs() < 1e-5);
        let err = svd_err(
            &SVDInput::Sparse(a),
            u,
            v,
            s,
        );
        assert!(err < 1e-9);
    }

    #[test]
    fn test_singular_values_descending() {
        let mut tri = TriMat::<f64>::new((5, 5));
        tri.add_triplet(0, 0, 3.0);
        tri.add_triplet(1, 1, 2.0);
        tri.add_triplet(2, 2, 1.0);
        let a = tri.to_csr::<usize>();
        let mut solver = LanczosSVD::default().with_random_state(Some(42));
        solver.fit(SVDInput::Sparse(a), 3, None).unwrap();
        let s = solver.singular_values.as_ref().unwrap();
        assert_eq!(s.len(), 3);
        assert!(s[0] >= s[1] && s[1] >= s[2]);
        assert!((s[0] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_randomized_svd() {
        let biadjacency = test_bigraph();
        let mut solver = RandomizedSVD::default().with_random_state(Some(0));
        solver
            .fit(SVDInput::Sparse(biadjacency.clone()), 2, None)
            .unwrap();
        assert_eq!(solver.singular_values.as_ref().unwrap().len(), 2);
        let err = svd_err(
            &SVDInput::Sparse(biadjacency),
            solver.singular_vectors_left.as_ref().unwrap(),
            solver.singular_vectors_right.as_ref().unwrap(),
            solver.singular_values.as_ref().unwrap(),
        );
        assert!(err < 0.5);
    }

    #[test]
    fn test_randomized_matches_lanczos_on_diagonal() {
        let mut tri = TriMat::<f64>::new((6, 6));
        for i in 0..6 {
            tri.add_triplet(i, i, (6 - i) as f64);
        }
        let a = tri.to_csr::<usize>();
        let input = SVDInput::Sparse(a);
        let mut lanczos = LanczosSVD::default().with_random_state(Some(42));
        lanczos.fit(input.clone(), 3, None).unwrap();
        let mut randomized = RandomizedSVD::default().with_random_state(Some(42));
        randomized
            .fit(input, 3, None)
            .unwrap();
        let sl = lanczos.singular_values.as_ref().unwrap();
        let sr = randomized.singular_values.as_ref().unwrap();
        for i in 0..3 {
            assert!((sl[i] - sr[i]).abs() < 0.2, "sigma[{i}] lanczos={} rand={}", sl[i], sr[i]);
        }
    }

    #[test]
    fn test_randomized_sklearn_parity_dense() {
        // Reference: sklearn randomized_svd(a, 2, random_state=0) on the docstring example.
        let mut tri = TriMat::<f64>::new((3, 4));
        let data = [
            [1.0, 2.0, 3.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0],
        ];
        for (i, row) in data.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                tri.add_triplet(i, j, v);
            }
        }
        let a = tri.to_csr::<usize>();
        let mut solver = RandomizedSVD::default().with_random_state(Some(0));
        solver.fit(SVDInput::Sparse(a.clone()), 2, None).unwrap();
        let s = solver.singular_values.as_ref().unwrap();
        // Exact ``numpy.linalg.svd`` values; randomized approximation should be close.
        assert!((s[0] - 20.35096734).abs() < 0.5);
        assert!((s[1] - 2.18854617).abs() < 0.1);
        let err = svd_err(
            &SVDInput::Sparse(a),
            solver.singular_vectors_left.as_ref().unwrap(),
            solver.singular_vectors_right.as_ref().unwrap(),
            s,
        );
        assert!(err < 0.2);
    }
}
