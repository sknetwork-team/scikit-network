use ndarray::Array2;
use sprs::CsMat;

use crate::linalg::operators::{CoNeighbor, Laplacian, Normalizer};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by basics error operations.
pub enum BasicsError {
    /// Indicates not implemented.
    NotImplemented,
}

/// LeftOperand enum.
pub enum LeftOperand<'a> {
    /// Indicates dense.
    Dense(&'a Array2<f64>),
    /// Indicates sparse.
    Sparse(&'a CsMat<f64>),
    /// Indicates laplacian.
    Laplacian(&'a Laplacian),
    /// Indicates normalizer.
    Normalizer(&'a Normalizer),
    /// Indicates co neighbor.
    CoNeighbor(&'a CoNeighbor),
}

/// RightOperand enum.
pub enum RightOperand<'a> {
    /// Indicates dense.
    Dense(&'a Array2<f64>),
    /// Indicates sparse.
    Sparse(&'a CsMat<f64>),
}

/// DotResult enum.
pub enum DotResult {
    /// Indicates dense.
    Dense(Array2<f64>),
    /// Indicates sparse.
    Sparse(CsMat<f64>),
}

fn sparse_dense_dot(a: &CsMat<f64>, b: &Array2<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((a.rows(), b.ncols()));
    for (i, row) in a.outer_iterator().enumerate() {
        for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
            for c in 0..b.ncols() {
                out[[i, c]] += v * b[[j, c]];
            }
        }
    }
    out
}

fn dense_sparse_dot(a: &Array2<f64>, b: &CsMat<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((a.nrows(), b.cols()));
    for (i, row) in b.outer_iterator().enumerate() {
        for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
            for r in 0..a.nrows() {
                out[[r, j]] += a[[r, i]] * v;
            }
        }
    }
    out
}

fn coneighbor_dense_dot(op: &CoNeighbor, b: &Array2<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((op.backward.rows(), b.ncols()));
    for c in 0..b.ncols() {
        let x = b.column(c).to_owned();
        let y = op.dot_vec(&x);
        for i in 0..y.len() {
            out[[i, c]] = y[i];
        }
    }
    out
}

/// Computes safe sparse dot.
///
/// # Errors
///
/// Returns [`BasicsError`] on failure.
pub fn safe_sparse_dot(a: LeftOperand<'_>, b: RightOperand<'_>) -> Result<DotResult, BasicsError> {
    match (a, b) {
        // Python special-case: if a is dense, compute b.T.dot(a.T).T
        (LeftOperand::Dense(a), RightOperand::Dense(b)) => {
            Ok(DotResult::Dense(b.t().dot(&a.t()).t().to_owned()))
        }
        (LeftOperand::Dense(a), RightOperand::Sparse(b)) => {
            Ok(DotResult::Dense(dense_sparse_dot(a, b)))
        }
        (LeftOperand::Sparse(a), RightOperand::Dense(b)) => {
            Ok(DotResult::Dense(sparse_dense_dot(a, b)))
        }
        (LeftOperand::Sparse(a), RightOperand::Sparse(b)) => Ok(DotResult::Sparse(a * b)),
        (LeftOperand::Laplacian(op), RightOperand::Dense(b)) => Ok(DotResult::Dense(op.dot_mat(b))),
        (LeftOperand::Normalizer(op), RightOperand::Dense(b)) => {
            Ok(DotResult::Dense(op.dot_mat(b)))
        }
        // For operators with dedicated sparse-dot hooks in Python, only dense right-side is exposed here.
        (LeftOperand::CoNeighbor(op), RightOperand::Dense(b)) => {
            Ok(DotResult::Dense(coneighbor_dense_dot(op, b)))
        }
        _ => Err(BasicsError::NotImplemented),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;
    use crate::data::test_graphs::test_graph;
    use crate::linalg::operators::Laplacian;

    #[test]
    fn test_safe_sparse_dot_dense_and_operator() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.0, 0.0, 1.0, 3.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 1.0, 0.0, 1.0]).unwrap();
        let out = safe_sparse_dot(LeftOperand::Dense(&a), RightOperand::Dense(&b)).unwrap();
        match out {
            DotResult::Dense(c) => assert_eq!(c.dim(), (2, 2)),
            DotResult::Sparse(_) => panic!("unexpected sparse output"),
        }

        let adjacency = test_graph();
        let laplacian = Laplacian::new(&adjacency, 0.1, true);
        let x = Array2::<f64>::ones((adjacency.cols(), 3));
        let out =
            safe_sparse_dot(LeftOperand::Laplacian(&laplacian), RightOperand::Dense(&x)).unwrap();
        match out {
            DotResult::Dense(y) => assert_eq!(y.dim(), (adjacency.rows(), 3)),
            DotResult::Sparse(_) => panic!("unexpected sparse output"),
        }
    }
}
