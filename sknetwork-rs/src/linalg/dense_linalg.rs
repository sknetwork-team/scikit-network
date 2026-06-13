//! Small dense linear algebra helpers used by iterative solvers.
//!
//! Implements QR orthonormalization and thin SVD via the Gram matrix, matching
//! the post-processing steps in ``scipy.sparse.linalg.svds`` (ARPACK path).

use ndarray::{Array1, Array2, Axis};

/// Computes l2 norm.
pub fn l2_norm(x: &Array1<f64>) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Economic QR factorization (``scipy.linalg.qr(..., mode='economic')``).
///
/// For an ``m × n`` matrix, returns ``Q`` with shape ``m × min(m, n)`` and
/// orthonormal columns.
pub fn qr_economic(y: Array2<f64>) -> Array2<f64> {
    let m = y.nrows();
    let n = y.ncols();
    let k = m.min(n);
    if k == 0 {
        return Array2::zeros((m, 0));
    }
    if m <= n {
        let mut q = Array2::<f64>::zeros((m, k));
        for c in 0..k {
            for i in 0..m {
                q[[i, c]] = y[[i, c]];
            }
            for p in 0..c {
                let mut dot = 0.0;
                for i in 0..m {
                    dot += q[[i, c]] * q[[i, p]];
                }
                for i in 0..m {
                    q[[i, c]] -= dot * q[[i, p]];
                }
            }
            let mut norm = 0.0;
            for i in 0..m {
                norm += q[[i, c]] * q[[i, c]];
            }
            norm = norm.sqrt();
            if norm > 0.0 {
                for i in 0..m {
                    q[[i, c]] /= norm;
                }
            }
        }
        q
    } else {
        qr_orthonormalize(y)
    }
}

/// Modified Gram–Schmidt QR; columns of `q` are orthonormal on return.
pub fn qr_orthonormalize(mut q: Array2<f64>) -> Array2<f64> {
    let n = q.nrows();
    let k = q.ncols();
    for c in 0..k {
        for p in 0..c {
            let mut dot = 0.0;
            for i in 0..n {
                dot += q[[i, c]] * q[[i, p]];
            }
            for i in 0..n {
                q[[i, c]] -= dot * q[[i, p]];
            }
        }
        let mut norm = 0.0;
        for i in 0..n {
            norm += q[[i, c]] * q[[i, c]];
        }
        norm = norm.sqrt();
        if norm > 0.0 {
            for i in 0..n {
                q[[i, c]] /= norm;
            }
        }
    }
    q
}

/// Jacobi eigen decomposition for small dense symmetric matrices.
pub fn symmetric_eigh(mut a: Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut v = Array2::<f64>::eye(n);
    let eps = 1e-15_f64;
    let max_sweeps = 50;
    for _ in 0..max_sweeps {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[[p, q]] * a[[p, q]];
            }
        }
        if off < eps {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[[p, q]];
                if apq.abs() < eps {
                    continue;
                }
                let app = a[[p, p]];
                let aqq = a[[q, q]];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[[k, p]];
                    let akq = a[[k, q]];
                    a[[k, p]] = c * akp - s * akq;
                    a[[p, k]] = a[[k, p]];
                    a[[k, q]] = s * akp + c * akq;
                    a[[q, k]] = a[[k, q]];
                }
                a[[p, p]] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                a[[q, q]] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                a[[p, q]] = 0.0;
                a[[q, p]] = 0.0;
                for k in 0..n {
                    let vkp = v[[k, p]];
                    let vkq = v[[k, q]];
                    v[[k, p]] = c * vkp - s * vkq;
                    v[[k, q]] = s * vkp + c * vkq;
                }
            }
        }
    }
    let evals: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        evals[i]
            .partial_cmp(&evals[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut evals_sorted = Array1::<f64>::zeros(n);
    let mut vecs_sorted = Array2::<f64>::zeros((n, n));
    for (new_j, &old_j) in order.iter().enumerate() {
        evals_sorted[new_j] = evals[old_j];
        for i in 0..n {
            vecs_sorted[[i, new_j]] = v[[i, old_j]];
        }
    }
    (evals_sorted, vecs_sorted)
}

/// Thin SVD matching ``numpy.linalg.svd(..., full_matrices=False)``.
///
/// Uses a Gram matrix of size ``min(m, n)`` so wide matrices (e.g. Halko's
/// ``Qᵀ A`` with shape ``(k+p) × n_features``) stay cheap.
pub fn thin_svd(a: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let m = a.nrows();
    let n = a.ncols();
    if n == 0 {
        return (
            Array2::<f64>::zeros((m, 0)),
            Array1::<f64>::zeros(0),
            Array2::<f64>::zeros((0, 0)),
        );
    }
    if m >= n {
        thin_svd_via_ata(a)
    } else {
        thin_svd_via_aat(a)
    }
}

/// Tall matrix: ``Aᵀ A`` is ``n × n`` with ``n ≤ m``.
fn thin_svd_via_ata(a: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let m = a.nrows();
    let n = a.ncols();
    let mut gram = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut dot = 0.0;
            for r in 0..m {
                dot += a[[r, i]] * a[[r, j]];
            }
            gram[[i, j]] = dot;
            gram[[j, i]] = dot;
        }
    }
    let (evals, v) = symmetric_eigh(gram);
    let mut singular = Array1::<f64>::zeros(n);
    for i in 0..n {
        singular[i] = evals[i].max(0.0).sqrt();
    }
    let mut vh = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            vh[[i, j]] = v[[j, i]];
        }
    }
    let mut u = Array2::<f64>::zeros((m, n));
    for c in 0..n {
        let sigma = singular[c];
        if sigma > 0.0 {
            for r in 0..m {
                let mut s = 0.0;
                for j in 0..n {
                    s += a[[r, j]] * v[[j, c]];
                }
                u[[r, c]] = s / sigma;
            }
        }
    }
    (u, singular, vh)
}

/// Wide matrix: ``A Aᵀ`` is ``m × m`` with ``m < n``; ``Vh`` has shape ``m × n``.
fn thin_svd_via_aat(a: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let m = a.nrows();
    let n = a.ncols();
    let mut gram = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..=i {
            let mut dot = 0.0;
            for c in 0..n {
                dot += a[[i, c]] * a[[j, c]];
            }
            gram[[i, j]] = dot;
            gram[[j, i]] = dot;
        }
    }
    let (evals, u) = symmetric_eigh(gram);
    let mut singular = Array1::<f64>::zeros(m);
    for i in 0..m {
        singular[i] = evals[i].max(0.0).sqrt();
    }
    let mut vh = Array2::<f64>::zeros((m, n));
    for c in 0..m {
        let sigma = singular[c];
        if sigma > 0.0 {
            for j in 0..n {
                let mut s = 0.0;
                for r in 0..m {
                    s += u[[r, c]] * a[[r, j]];
                }
                vh[[c, j]] = s / sigma;
            }
        }
    }
    (u, singular, vh)
}

/// `a @ b` for dense matrices.
pub fn dense_matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let k = a.ncols();
    let m = b.ncols();
    let mut out = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0;
            for t in 0..k {
                s += a[[i, t]] * b[[t, j]];
            }
            out[[i, j]] = s;
        }
    }
    out
}

/// `a^T @ b` for dense matrices.
pub fn dense_transpose_matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let k = a.ncols();
    let m = b.ncols();
    let mut out = Array2::<f64>::zeros((k, m));
    for i in 0..k {
        for j in 0..m {
            let mut s = 0.0;
            for t in 0..n {
                s += a[[t, i]] * b[[t, j]];
            }
            out[[i, j]] = s;
        }
    }
    out
}

/// Extract column `c` from a matrix.
pub fn column(a: &Array2<f64>, c: usize) -> Array1<f64> {
    a.index_axis(Axis(1), c).to_owned()
}

/// Return the transpose of `a` as an owned matrix.
pub fn transpose_matrix(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let m = a.ncols();
    let mut out = Array2::<f64>::zeros((m, n));
    for i in 0..n {
        for j in 0..m {
            out[[j, i]] = a[[i, j]];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qr_economic_tall() {
        let y = Array2::from_shape_fn((4, 12), |(i, j)| (i + j) as f64);
        let q = qr_economic(y);
        assert_eq!(q.dim(), (4, 4));
    }

    #[test]
    fn test_thin_svd_wide_matrix() {
        // Halko-style fat ``B``: (k+p) × n with k+p << n.
        let a = Array2::from_shape_fn((3, 50), |(i, j)| ((i + 1) as f64) * 0.01 + j as f64 * 0.001);
        let (u, s, vh) = thin_svd(&a);
        assert_eq!(u.dim(), (3, 3));
        assert_eq!(s.len(), 3);
        assert_eq!(vh.dim(), (3, 50));
        let mut err = 0.0;
        for i in 0..3 {
            for j in 0..50 {
                let mut val = 0.0;
                for c in 0..3 {
                    val += u[[i, c]] * s[c] * vh[[c, j]];
                }
                let d = val - a[[i, j]];
                err += d * d;
            }
        }
        assert!(err.sqrt() < 1e-8);
    }

    #[test]
    fn test_thin_svd_matches_gram() {
        let a = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        )
        .unwrap();
        let (u, s, vh) = thin_svd(&a);
        let vh_t = transpose_matrix(&vh);
        let av = dense_matmul(&a, &vh_t);
        let us = {
            let mut out = u.clone();
            for c in 0..s.len() {
                for r in 0..out.nrows() {
                    out[[r, c]] *= s[c];
                }
            }
            out
        };
        for r in 0..av.nrows() {
            for c in 0..av.ncols() {
                assert!((av[[r, c]] - us[[r, c]]).abs() < 1e-10);
            }
        }
    }
}
