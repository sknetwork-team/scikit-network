use ndarray::{Array1, Array2};
use sprs::{CsMat, TriMat};

#[derive(Debug, Clone, PartialEq)]
/// Errors raised by normalizer error operations.
pub enum NormalizerError {
    /// Indicates unsupported norm.
    UnsupportedNorm,
}

/// Computes diagonal pseudo inverse.
pub fn diagonal_pseudo_inverse(weights: &Array1<f64>) -> CsMat<f64> {
    let n = weights.len();
    let mut tri = TriMat::<f64>::new((n, n));
    for (i, &w) in weights.iter().enumerate() {
        if w != 0.0 {
            tri.add_triplet(i, i, 1.0 / w);
        }
    }
    tri.to_csr::<usize>()
}

/// Returns norms dense.
///
/// # Errors
///
/// Returns [`NormalizerError`] on failure.
pub fn get_norms_dense(matrix: &Array2<f64>, p: u32) -> Result<Array1<f64>, NormalizerError> {
    let (n_row, _n_col) = matrix.dim();
    let mut norms = Array1::<f64>::zeros(n_row);
    match p {
        1 => {
            for i in 0..n_row {
                norms[i] = matrix.row(i).iter().map(|x| x.abs()).sum::<f64>();
            }
        }
        2 => {
            for i in 0..n_row {
                norms[i] = matrix.row(i).iter().map(|x| x * x).sum::<f64>().sqrt();
            }
        }
        _ => return Err(NormalizerError::UnsupportedNorm),
    }
    Ok(norms)
}

/// Returns norms sparse.
///
/// # Errors
///
/// Returns [`NormalizerError`] on failure.
pub fn get_norms_sparse(matrix: &CsMat<f64>, p: u32) -> Result<Array1<f64>, NormalizerError> {
    let mut norms = Array1::<f64>::zeros(matrix.rows());
    match p {
        1 => {
            for (i, row) in matrix.outer_iterator().enumerate() {
                norms[i] = row.data().iter().map(|x| x.abs()).sum::<f64>();
            }
        }
        2 => {
            for (i, row) in matrix.outer_iterator().enumerate() {
                norms[i] = row.data().iter().map(|x| x * x).sum::<f64>().sqrt();
            }
        }
        _ => return Err(NormalizerError::UnsupportedNorm),
    }
    Ok(norms)
}

/// Computes normalize sparse.
///
/// # Errors
///
/// Returns [`NormalizerError`] on failure.
pub fn normalize_sparse(matrix: &CsMat<f64>, p: u32) -> Result<CsMat<f64>, NormalizerError> {
    let norms = get_norms_sparse(matrix, p)?;
    let mut tri = TriMat::<f64>::new(matrix.shape());
    for (i, row) in matrix.outer_iterator().enumerate() {
        let inv = if norms[i] > 0.0 { 1.0 / norms[i] } else { 0.0 };
        if inv > 0.0 {
            for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                tri.add_triplet(i, j, inv * v);
            }
        }
    }
    Ok(tri.to_csr::<usize>())
}

/// Computes normalize dense.
///
/// # Errors
///
/// Returns [`NormalizerError`] on failure.
pub fn normalize_dense(matrix: &Array2<f64>, p: u32) -> Result<Array2<f64>, NormalizerError> {
    let norms = get_norms_dense(matrix, p)?;
    let mut out = matrix.clone();
    for i in 0..out.nrows() {
        let inv = if norms[i] > 0.0 { 1.0 / norms[i] } else { 0.0 };
        if inv > 0.0 {
            for j in 0..out.ncols() {
                out[[i, j]] *= inv;
            }
        } else {
            for j in 0..out.ncols() {
                out[[i, j]] = 0.0;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, array};

    use super::*;

    fn eye_dense(n: usize) -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            m[[i, i]] = 1.0;
        }
        m
    }

    fn sparse_vec_mul(mat: &CsMat<f64>, x: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(mat.rows());
        for (i, row) in mat.outer_iterator().enumerate() {
            let mut s = 0.0;
            for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                s += v * x[j];
            }
            out[i] = s;
        }
        out
    }

    #[test]
    fn test_formats() {
        let n = 5usize;
        let mat1 = normalize_dense(&eye_dense(n), 1).unwrap();
        let mat2 = normalize_sparse(&diagonal_pseudo_inverse(&Array1::ones(n)), 1).unwrap();

        let x = Array1::from_vec(vec![0.1, -0.4, 0.7, 1.2, -0.3]);
        let y1 = mat1.dot(&x);
        let y2 = sparse_vec_mul(&mat2, &x);
        let err1: f64 = y1
            .iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let err2: f64 = y2
            .iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(err1 < 1e-12);
        assert!(err2 < 1e-12);

        let mat1 = array![
            [0.2, 1.3, -0.5, 0.0, 1.1],
            [1.2, 0.0, 0.8, 0.1, -0.7],
            [0.0, 0.6, 0.0, 0.4, 0.9],
            [0.5, -1.5, 0.2, 0.3, 0.0],
            [0.7, 0.2, 0.9, -0.4, 0.1]
        ];
        let mut tri = TriMat::<f64>::new((n, n));
        for i in 0..n {
            for j in 0..n {
                let v = mat1[[i, j]];
                if v != 0.0 {
                    tri.add_triplet(i, j, v);
                }
            }
        }
        let mat2 = tri.to_csr::<usize>();
        let mat1 = normalize_dense(&mat1, 2).unwrap();
        let mat2 = normalize_sparse(&mat2, 2).unwrap();
        let y1 = mat1.dot(&x);
        let y2 = sparse_vec_mul(&mat2, &x);
        let err: f64 = y1
            .iter()
            .zip(y2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-12);

        assert_eq!(
            normalize_dense(&mat1, 3),
            Err(NormalizerError::UnsupportedNorm)
        );
    }
}
