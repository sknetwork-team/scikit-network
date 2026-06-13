//! Cached CSR/CSC structure for repeated sparse matvecs (Lanczos Gram operator).

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use sprs::CsMat;

use crate::linalg::sparse_matvec::{
    csc_matvec_into, csr_matvec_into, csr_use_parallel, outer_ranges, row_ranges,
    PARALLEL_COL_THRESHOLD,
};
use crate::utils::check::is_symmetric;

/// Precomputed row/column slices and optional CSC ``Aᵀ`` for a fixed CSR matrix.
///
/// Used by Lanczos ``eigsh`` and Halko randomized SVD sparse matvecs.
pub struct SparseMatvecCache {
    row_ranges: Vec<(usize, usize)>,
    a_t_csc: Option<CsMat<f64>>,
    t_ranges: Vec<(usize, usize)>,
    symmetric: bool,
}

impl SparseMatvecCache {
    /// Creates a new instance.
    pub fn new(a: &CsMat<f64>) -> Self {
        let symmetric = is_symmetric(a);
        let (a_t_csc, t_ranges) = if symmetric {
            (None, Vec::new())
        } else {
            let at = a.transpose_view().to_owned();
            let tr = outer_ranges(&at);
            (Some(at), tr)
        };
        Self {
            row_ranges: row_ranges(a),
            a_t_csc,
            t_ranges,
            symmetric,
        }
    }

    /// Returns whether the input satisfies `is symmetric`.
    pub fn is_symmetric(&self) -> bool {
        self.symmetric
    }

    /// Computes matvec a into.
    pub fn matvec_a_into(&self, a: &CsMat<f64>, x: &[f64], out: &mut [f64]) {
        csr_matvec_into(a, x, out, Some(&self.row_ranges));
    }

    /// Computes matvec at into.
    pub fn matvec_at_into(&self, a: &CsMat<f64>, x: &[f64], out: &mut [f64]) {
        if self.symmetric {
            self.matvec_a_into(a, x, out);
            return;
        }
        if let Some(at) = &self.a_t_csc {
            csc_matvec_into(at, x, out, Some(&self.t_ranges));
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

    /// ``out = (Aᵀ A) x`` or ``(A Aᵀ) x`` depending on ``transpose`` (``svds`` convention).
    pub fn gram_matvec_into(
        &self,
        a: &CsMat<f64>,
        transpose: bool,
        x: &[f64],
        scratch_row: &mut [f64],
        scratch_col: &mut [f64],
        out: &mut [f64],
    ) {
        if self.symmetric {
            self.matvec_a_into(a, x, scratch_row);
            self.matvec_a_into(a, scratch_row, out);
            return;
        }
        if transpose {
            self.matvec_at_into(a, x, scratch_col);
            self.matvec_a_into(a, scratch_col, out);
        } else {
            self.matvec_a_into(a, x, scratch_row);
            self.matvec_at_into(a, scratch_row, out);
        }
    }

    /// Computes gram matvec nd.
    pub fn gram_matvec_nd(
        &self,
        a: &CsMat<f64>,
        transpose: bool,
        x: &Array1<f64>,
        scratch_row: &mut Array1<f64>,
        scratch_col: &mut Array1<f64>,
        out: &mut Array1<f64>,
    ) {
        self.gram_matvec_into(
            a,
            transpose,
            x.as_slice().unwrap(),
            scratch_row.as_slice_mut().unwrap(),
            scratch_col.as_slice_mut().unwrap(),
            out.as_slice_mut().unwrap(),
        );
    }

    /// ``out[:, c] = A @ b[:, c]`` (reuses ``scratch_x`` / ``scratch_y``, no per-column alloc).
    pub fn matvec_a_matrix_into(
        &self,
        a: &CsMat<f64>,
        b: &Array2<f64>,
        out: &mut Array2<f64>,
        scratch_x: &mut Array1<f64>,
        scratch_y: &mut Array1<f64>,
    ) {
        let k = b.ncols();
        if k >= PARALLEL_COL_THRESHOLD && csr_use_parallel(a) {
            self.matvec_a_matrix_parallel(a, b, out);
            return;
        }
        for c in 0..k {
            for r in 0..b.nrows() {
                scratch_x[r] = b[[r, c]];
            }
            self.matvec_a_into(
                a,
                scratch_x.as_slice().unwrap(),
                scratch_y.as_slice_mut().unwrap(),
            );
            for r in 0..out.nrows() {
                out[[r, c]] = scratch_y[r];
            }
        }
    }

    /// Parallel ``A @ B`` when CSR row-parallel SpMV pays off (Halko range finder).
    fn matvec_a_matrix_parallel(&self, a: &CsMat<f64>, b: &Array2<f64>, out: &mut Array2<f64>) {
        let k = b.ncols();
        let nrows = a.rows();
        let ncols = a.cols();
        let ranges = &self.row_ranges;
        let out_ptr = out.as_mut_ptr() as usize;
        (0..k).into_par_iter().for_each(|c| {
            let mut x = vec![0.0; ncols];
            let mut y = vec![0.0; nrows];
            for r in 0..b.nrows() {
                x[r] = b[[r, c]];
            }
            csr_matvec_into(a, &x, &mut y, Some(ranges));
            let base = out_ptr as *mut f64;
            for r in 0..nrows {
                unsafe {
                    *base.add(r * k + c) = y[r];
                }
            }
        });
    }

    /// ``out[:, c] = Aᵀ @ b[:, c]`` (reuses ``scratch_x`` / ``scratch_y``).
    pub fn matvec_at_matrix_into(
        &self,
        a: &CsMat<f64>,
        b: &Array2<f64>,
        out: &mut Array2<f64>,
        scratch_x: &mut Array1<f64>,
        scratch_y: &mut Array1<f64>,
    ) {
        let k = b.ncols();
        if k >= PARALLEL_COL_THRESHOLD && csr_use_parallel(a) {
            self.matvec_at_matrix_parallel(a, b, out);
            return;
        }
        for c in 0..k {
            for r in 0..b.nrows() {
                scratch_x[r] = b[[r, c]];
            }
            self.matvec_at_into(
                a,
                scratch_x.as_slice().unwrap(),
                scratch_y.as_slice_mut().unwrap(),
            );
            for r in 0..out.nrows() {
                out[[r, c]] = scratch_y[r];
            }
        }
    }

    /// Parallel ``Aᵀ @ B`` via cached CSC when the matrix is large/dense enough.
    fn matvec_at_matrix_parallel(&self, a: &CsMat<f64>, b: &Array2<f64>, out: &mut Array2<f64>) {
        if self.symmetric {
            self.matvec_a_matrix_parallel(a, b, out);
            return;
        }
        let k = b.ncols();
        let nrows = b.nrows();
        let ncols_out = out.nrows();
        let out_ptr = out.as_mut_ptr() as usize;
        if let Some(at) = &self.a_t_csc {
            let t_ranges = &self.t_ranges;
            (0..k).into_par_iter().for_each(|c| {
                let mut x = vec![0.0; nrows];
                let mut y = vec![0.0; ncols_out];
                for r in 0..nrows {
                    x[r] = b[[r, c]];
                }
                csc_matvec_into(at, &x, &mut y, Some(t_ranges));
                let base = out_ptr as *mut f64;
                for r in 0..ncols_out {
                    unsafe {
                        *base.add(r * k + c) = y[r];
                    }
                }
            });
            return;
        }
        let mut scratch_x = vec![0.0; nrows];
        let mut scratch_y = vec![0.0; ncols_out];
        for c in 0..k {
            for r in 0..nrows {
                scratch_x[r] = b[[r, c]];
            }
            self.matvec_at_into(a, &scratch_x, &mut scratch_y);
            for r in 0..ncols_out {
                out[[r, c]] = scratch_y[r];
            }
        }
    }
}

/// Fused ``Aᵀ A x`` / ``A Aᵀ x`` with scratch buffers (Lanczos ``eigsh`` matvec).
pub struct SparseGramOperator {
    cache: SparseMatvecCache,
    scratch_row: Array1<f64>,
    scratch_col: Array1<f64>,
    transpose: bool,
}

impl SparseGramOperator {
    /// Creates a new instance.
    pub fn new(a: &CsMat<f64>, transpose: bool) -> Self {
        let (n_row, n_col) = a.shape();
        Self {
            cache: SparseMatvecCache::new(a),
            scratch_row: Array1::<f64>::zeros(n_row),
            scratch_col: Array1::<f64>::zeros(n_col),
            transpose,
        }
    }

    /// Computes cache.
    pub fn cache(&self) -> &SparseMatvecCache {
        &self.cache
    }

    /// Computes apply into.
    pub fn apply_into(&mut self, a: &CsMat<f64>, x: &Array1<f64>, out: &mut Array1<f64>) {
        self.cache.gram_matvec_nd(
            a,
            self.transpose,
            x,
            &mut self.scratch_row,
            &mut self.scratch_col,
            out,
        );
    }
}

/// Reusable CSR/CSC matvec workspace for Halko multi-column ``A·Q`` / ``Aᵀ·Q``.
pub struct CachedSparseApplicator {
    cache: SparseMatvecCache,
    scratch_x_row: Array1<f64>,
    scratch_y_row: Array1<f64>,
    scratch_x_col: Array1<f64>,
    scratch_y_col: Array1<f64>,
}

impl CachedSparseApplicator {
    /// Creates a new instance.
    pub fn new(a: &CsMat<f64>) -> Self {
        let (n_row, n_col) = a.shape();
        Self {
            cache: SparseMatvecCache::new(a),
            scratch_x_row: Array1::<f64>::zeros(n_row),
            scratch_y_row: Array1::<f64>::zeros(n_row),
            scratch_x_col: Array1::<f64>::zeros(n_col),
            scratch_y_col: Array1::<f64>::zeros(n_col),
        }
    }

    /// Computes cache.
    pub fn cache(&self) -> &SparseMatvecCache {
        &self.cache
    }

    /// Computes apply a mat into.
    pub fn apply_a_mat_into(&mut self, a: &CsMat<f64>, b: &Array2<f64>, out: &mut Array2<f64>) {
        self.cache.matvec_a_matrix_into(
            a,
            b,
            out,
            &mut self.scratch_x_col,
            &mut self.scratch_y_row,
        );
    }

    /// Computes apply at mat into.
    pub fn apply_at_mat_into(&mut self, a: &CsMat<f64>, b: &Array2<f64>, out: &mut Array2<f64>) {
        self.cache.matvec_at_matrix_into(
            a,
            b,
            out,
            &mut self.scratch_x_row,
            &mut self.scratch_y_col,
        );
    }

    /// ``out[c, :] = (Aᵀ q_c)`` if ``transposed`` else ``out[c, :] = (A q_c)`` with ``q_c = Q[:, c]``.
    pub fn qt_times_a_into(
        &mut self,
        a: &CsMat<f64>,
        q: &Array2<f64>,
        transposed: bool,
        out: &mut Array2<f64>,
    ) {
        let n_random = q.ncols();
        let out_cols = out.ncols();
        if n_random >= PARALLEL_COL_THRESHOLD && csr_use_parallel(a) {
            self.qt_times_a_parallel(a, q, transposed, out);
            return;
        }
        for c in 0..n_random {
            if transposed {
                for r in 0..q.nrows() {
                    self.scratch_x_col[r] = q[[r, c]];
                }
                self.cache.matvec_a_into(
                    a,
                    self.scratch_x_col.as_slice().unwrap(),
                    self.scratch_y_row.as_slice_mut().unwrap(),
                );
                for j in 0..out_cols {
                    out[[c, j]] = self.scratch_y_row[j];
                }
            } else {
                for r in 0..q.nrows() {
                    self.scratch_x_row[r] = q[[r, c]];
                }
                self.cache.matvec_at_into(
                    a,
                    self.scratch_x_row.as_slice().unwrap(),
                    self.scratch_y_col.as_slice_mut().unwrap(),
                );
                for j in 0..out_cols {
                    out[[c, j]] = self.scratch_y_col[j];
                }
            }
        }
    }

    fn qt_times_a_parallel(
        &self,
        a: &CsMat<f64>,
        q: &Array2<f64>,
        transposed: bool,
        out: &mut Array2<f64>,
    ) {
        let n_random = q.ncols();
        let out_cols = out.ncols();
        let n_qrows = q.nrows();
        let row_ranges = &self.cache.row_ranges;
        let out_ptr = out.as_mut_ptr() as usize;
        if transposed {
            let nrows = a.rows();
            (0..n_random).into_par_iter().for_each(|c| {
                let mut x = vec![0.0; n_qrows];
                let mut y = vec![0.0; nrows];
                for r in 0..n_qrows {
                    x[r] = q[[r, c]];
                }
                csr_matvec_into(a, &x, &mut y, Some(row_ranges));
                let base = out_ptr as *mut f64;
                for j in 0..out_cols {
                    unsafe {
                        *base.add(c * out_cols + j) = y[j];
                    }
                }
            });
        } else if self.cache.symmetric {
            let ncols = a.cols();
            (0..n_random).into_par_iter().for_each(|c| {
                let mut x = vec![0.0; n_qrows];
                let mut y = vec![0.0; ncols];
                for r in 0..n_qrows {
                    x[r] = q[[r, c]];
                }
                csr_matvec_into(a, &x, &mut y, Some(row_ranges));
                let base = out_ptr as *mut f64;
                for j in 0..out_cols {
                    unsafe {
                        *base.add(c * out_cols + j) = y[j];
                    }
                }
            });
        } else if let Some(at) = &self.cache.a_t_csc {
            let t_ranges = &self.cache.t_ranges;
            let ncols = a.cols();
            (0..n_random).into_par_iter().for_each(|c| {
                let mut x = vec![0.0; n_qrows];
                let mut y = vec![0.0; ncols];
                for r in 0..n_qrows {
                    x[r] = q[[r, c]];
                }
                csc_matvec_into(at, &x, &mut y, Some(t_ranges));
                let base = out_ptr as *mut f64;
                for j in 0..out_cols {
                    unsafe {
                        *base.add(c * out_cols + j) = y[j];
                    }
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;

    #[test]
    fn test_matvec_a_matrix_parallel_matches_sequential() {
        let n = 1100usize;
        let mut tri = TriMat::<f64>::new((n, n));
        for i in 0..n {
            for j in 0..12 {
                let col = (i + j * 17) % n;
                tri.add_triplet(i, col, 1.0);
            }
        }
        let a = tri.to_csr();
        let cache = SparseMatvecCache::new(&a);
        let k = 10usize;
        let mut b = Array2::<f64>::zeros((n, k));
        for c in 0..k {
            for r in 0..n {
                b[[r, c]] = ((r + c) as f64).sin();
            }
        }
        let mut seq = Array2::<f64>::zeros((n, k));
        let mut par = Array2::<f64>::zeros((n, k));
        let mut sx = Array1::<f64>::zeros(n);
        let mut sy = Array1::<f64>::zeros(n);
        for c in 0..k {
            for r in 0..n {
                sx[r] = b[[r, c]];
            }
            cache.matvec_a_into(&a, sx.as_slice().unwrap(), sy.as_slice_mut().unwrap());
            for r in 0..n {
                seq[[r, c]] = sy[r];
            }
        }
        cache.matvec_a_matrix_into(&a, &b, &mut par, &mut sx, &mut sy);
        for r in 0..n {
            for c in 0..k {
                assert!(
                    (seq[[r, c]] - par[[r, c]]).abs() < 1e-12,
                    "mismatch at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn test_gram_matvec_matches_naive() {
        let mut tri = TriMat::<f64>::new((4, 3));
        tri.add_triplet(0, 1, 2.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(2, 2, 3.0);
        let a = tri.to_csr();
        let cache = SparseMatvecCache::new(&a);
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut out = Array1::<f64>::zeros(3);
        let mut sr = Array1::<f64>::zeros(4);
        let mut sc = Array1::<f64>::zeros(3);
        cache.gram_matvec_nd(&a, false, &x, &mut sr, &mut sc, &mut out);
        let ax = {
            let mut v = vec![0.0; 4];
            cache.matvec_a_into(&a, x.as_slice().unwrap(), &mut v);
            v
        };
        let mut ref_out = vec![0.0; 3];
        cache.matvec_at_into(&a, &ax, &mut ref_out);
        for i in 0..3 {
            assert!((out[i] - ref_out[i]).abs() < 1e-12);
        }
    }
}
