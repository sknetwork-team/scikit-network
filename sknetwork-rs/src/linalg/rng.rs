//! RNG helpers aligned with SciPy / NumPy defaults used in sparse solvers.

use ndarray::{Array1, Array2};
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Draw one standard-normal sample (Box–Muller), matching ``numpy.random.standard_normal``.
fn randn(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-300);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Starting vector for ``scipy.sparse.linalg.svds`` when ``v0`` is ``None``.
///
/// SciPy draws ``rng.standard_normal(min(A.shape))`` without normalizing; ARPACK
/// normalizes internally before the first Lanczos step.
pub fn svds_default_v0(n: usize, seed: Option<u64>) -> Array1<f64> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(rand::random()),
    };
    Array1::from_vec((0..n).map(|_| randn(&mut rng)).collect())
}

/// Default tolerance when ``tol=0`` (SciPy / ARPACK practical accuracy on ``f64``).
///
/// ARPACK advertises machine precision, but Ritz vectors from Lanczos typically
/// satisfy relative residuals around ``1e-10`` rather than ``f64::EPSILON``.
pub fn arpack_machine_tol() -> f64 {
    1e-10
}

/// Number of Lanczos vectors: ``max(2*k + 1, 20)``, clipped to ``[k+2, n]``.
///
/// Matches ``scipy.sparse.linalg.eigen.arpack.choose_ncv`` plus the symmetric
/// ARPACK constraint ``k + 1 < ncv <= n``. When ``k + 2 > n`` (near full
/// decomposition), returns ``n`` so the Krylov basis spans the whole space.
/// Standard-normal matrix of shape ``(n_rows, n_cols)`` (sklearn ``random_state.normal``).
pub fn standard_normal_matrix(n_rows: usize, n_cols: usize, seed: Option<u64>) -> Array2<f64> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(rand::random()),
    };
    Array2::from_shape_fn((n_rows, n_cols), |(_, _)| randn(&mut rng))
}

/// ``ncv`` for IRLM Lanczos / ``eigsh``.
///
/// Matches SciPy ``max(2k+1, 20)`` by default. Large-``n`` extras reduce IRLM
/// restarts; the largest bump (``+2k``) is reserved for **symmetric adjacency**
/// (``Aᵀ=A`` Gram, e.g. citation graphs). Asymmetric graphs at ``n ≥ 2048`` get a
/// modest ``+k`` only; ``n ≥ 1024`` asymmetric (e.g. polblogs) keeps ``+k`` as in
/// round 4–6 so restart count stays low.
pub fn choose_ncv(k: usize, n: usize, symmetric_adjacency: bool) -> usize {
    if k + 2 > n {
        return n;
    }
    let base = (2 * k + 1).max(20);
    let extra = if n >= 2048 {
        if symmetric_adjacency {
            (2 * k).min(30)
        } else {
            k.min(10)
        }
    } else if n >= 1024 {
        k.min(10)
    } else {
        0
    };
    (base + extra).clamp(k + 2, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_choose_ncv() {
        assert_eq!(choose_ncv(2, 100, false), 20);
        assert_eq!(choose_ncv(10, 100, false), 21);
        assert_eq!(choose_ncv(10, 3264, true), 41);
        assert_eq!(choose_ncv(10, 3264, false), 31);
        assert_eq!(choose_ncv(10, 10011, false), 31);
        assert_eq!(choose_ncv(10, 10011, true), 41);
        assert_eq!(choose_ncv(10, 1490, true), 31);
        assert_eq!(choose_ncv(10, 1490, false), 31);
        assert_eq!(choose_ncv(2, 5, false), 5);
    }

    #[test]
    fn test_v0_reproducible_with_seed() {
        let a = svds_default_v0(8, Some(42));
        let b = svds_default_v0(8, Some(42));
        assert_eq!(a, b);
        let norm: f64 = a.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm > 0.0);
    }
}
