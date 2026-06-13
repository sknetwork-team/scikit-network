//! CSR sparse matrix–vector multiply (parallel over rows when worthwhile).

use rayon::prelude::*;
use sprs::CsMat;

/// Minimum row count before considering parallelism.
const PARALLEL_ROW_THRESHOLD: usize = 1024;
/// Minimum average nnz/row to amortize Rayon overhead.
const PARALLEL_AVG_NNZ_THRESHOLD: usize = 10;

/// Whether CSR row-parallel SpMV is worthwhile for this matrix.
pub fn csr_use_parallel(a: &CsMat<f64>) -> bool {
    let nrows = a.rows();
    nrows >= PARALLEL_ROW_THRESHOLD && a.nnz() / nrows.max(1) >= PARALLEL_AVG_NNZ_THRESHOLD
}

fn use_parallel(a: &CsMat<f64>) -> bool {
    csr_use_parallel(a)
}

/// Minimum column count before parallel multi-column SpMV (Halko ``A·Q``).
pub const PARALLEL_COL_THRESHOLD: usize = 8;

/// Row slice list for [`csr_matvec_into`] (build once per matrix, reuse across matvecs).
pub fn row_ranges(a: &CsMat<f64>) -> Vec<(usize, usize)> {
    outer_ranges(a)
}

/// Outer-dimension slice list (rows for CSR, columns for CSC).
pub fn outer_ranges(a: &CsMat<f64>) -> Vec<(usize, usize)> {
    a.indptr()
        .iter_outer_sz()
        .map(|r| (r.start, r.end))
        .collect()
}

/// ``y = A x`` for CSC ``A`` (column-oriented storage); writes into ``out``.
pub fn csc_matvec_into(
    a: &CsMat<f64>,
    x: &[f64],
    out: &mut [f64],
    ranges: Option<&[(usize, usize)]>,
) {
    let ncols = a.cols();
    debug_assert_eq!(x.len(), ncols);
    debug_assert_eq!(out.len(), a.rows());

    out.fill(0.0);
    let indices = a.indices();
    let data = a.data();

    if let Some(ranges) = ranges {
        for (j, &(start, end)) in ranges.iter().enumerate() {
            let xj = x[j];
            if xj == 0.0 {
                continue;
            }
            for p in start..end {
                out[indices[p]] += data[p] * xj;
            }
        }
        return;
    }

    for (j, col) in a.outer_iterator().enumerate() {
        let xj = x[j];
        if xj == 0.0 {
            continue;
        }
        for (&i, &v) in col.indices().iter().zip(col.data().iter()) {
            out[i] += v * xj;
        }
    }
}

fn csr_matvec_parallel(
    a: &CsMat<f64>,
    x: &[f64],
    out: &mut [f64],
    ranges: &[(usize, usize)],
) {
    let indices = a.indices();
    let data = a.data();
    out.par_iter_mut()
        .enumerate()
        .for_each(|(i, oi)| {
            let (start, end) = ranges[i];
            let mut s = 0.0;
            for p in start..end {
                s += data[p] * x[indices[p]];
            }
            *oi = s;
        });
}

/// ``y = A x`` for CSR ``A``; writes into ``out``.
///
/// Pass cached [`row_ranges`] when calling repeatedly on the same matrix.
pub fn csr_matvec_into(
    a: &CsMat<f64>,
    x: &[f64],
    out: &mut [f64],
    ranges: Option<&[(usize, usize)]>,
) {
    let nrows = a.rows();
    debug_assert_eq!(out.len(), nrows);
    debug_assert_eq!(x.len(), a.cols());

    if use_parallel(a) {
        if let Some(ranges) = ranges {
            csr_matvec_parallel(a, x, out, ranges);
        } else {
            let owned = row_ranges(a);
            csr_matvec_parallel(a, x, out, &owned);
        }
        return;
    }

    for (i, row) in a.outer_iterator().enumerate() {
        let mut s = 0.0;
        for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
            s += v * x[j];
        }
        out[i] = s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;

    #[test]
    fn test_csr_matvec_small() {
        let mut tri = TriMat::<f64>::new((4, 3));
        tri.add_triplet(0, 1, 2.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(2, 2, 3.0);
        let a = tri.to_csr();
        let x = [1.0, 2.0, 3.0];
        let mut out = vec![0.0; 4];
        csr_matvec_into(&a, &x, &mut out, None);
        assert!((out[0] - 4.0).abs() < 1e-12);
        assert!((out[1] - 1.0).abs() < 1e-12);
        assert!((out[2] - 9.0).abs() < 1e-12);
        assert!((out[3] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_csc_matvec_matches_transpose_csr() {
        let mut tri = TriMat::<f64>::new((4, 3));
        tri.add_triplet(0, 1, 2.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(2, 2, 3.0);
        let a_csr = tri.to_csr();
        let a_t = a_csr.transpose_view().to_owned();
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut csc_out = vec![0.0; 3];
        let mut ref_out = vec![0.0; 3];
        csc_matvec_into(&a_t, &x, &mut csc_out, None);
        for (i, row) in a_csr.outer_iterator().enumerate() {
            let xi = x[i];
            if xi == 0.0 {
                continue;
            }
            for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                ref_out[j] += v * xi;
            }
        }
        for i in 0..3 {
            assert!((csc_out[i] - ref_out[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cached_ranges_matches_uncached() {
        let mut tri = TriMat::<f64>::new((1200, 1200));
        for i in 0..1200 {
            for d in 0..12 {
                tri.add_triplet(i, (i + d * 3) % 1200, 1.0);
            }
        }
        let a = tri.to_csr();
        let x: Vec<f64> = (0..1200).map(|i| (i as f64 * 0.01).sin()).collect();
        let ranges = row_ranges(&a);
        let mut cached = vec![0.0; 1200];
        let mut uncached = vec![0.0; 1200];
        csr_matvec_into(&a, &x, &mut cached, Some(&ranges));
        csr_matvec_into(&a, &x, &mut uncached, None);
        for i in 0..1200 {
            assert!((cached[i] - uncached[i]).abs() < 1e-10);
        }
    }
}
