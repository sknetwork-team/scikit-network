use ndarray::{Array1, Array2, Axis};
use sprs::CsMat;

use crate::linalg::sparse_lowrank::SparseLR;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by eig error operations.
pub enum EigError {
    /// Indicates invalid components.
    InvalidComponents,
    /// Indicates empty matrix.
    EmptyMatrix,
    /// Indicates unknown which.
    UnknownWhich,
}

#[derive(Debug, Clone)]
/// EigInput enum.
pub enum EigInput {
    /// Indicates sparse.
    Sparse(CsMat<f64>),
    /// Indicates sparse lr.
    SparseLR(SparseLR),
}

fn sparse_dot_dense(a: &CsMat<f64>, x: &Array2<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((a.rows(), x.ncols()));
    for (i, row) in a.outer_iterator().enumerate() {
        for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
            for c in 0..x.ncols() {
                out[[i, c]] += v * x[[j, c]];
            }
        }
    }
    out
}

fn apply_input(input: &EigInput, x: &Array2<f64>) -> Array2<f64> {
    match input {
        EigInput::Sparse(a) => sparse_dot_dense(a, x),
        EigInput::SparseLR(a) => a.dot_mat(x),
    }
}

fn normalize_columns(mut x: Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let k = x.ncols();
    for c in 0..k {
        for p in 0..c {
            let mut dot = 0.0;
            for i in 0..n {
                dot += x[[i, c]] * x[[i, p]];
            }
            for i in 0..n {
                x[[i, c]] -= dot * x[[i, p]];
            }
        }
        let norm = (0..n).map(|i| x[[i, c]] * x[[i, c]]).sum::<f64>().sqrt();
        if norm > 0.0 {
            for i in 0..n {
                x[[i, c]] /= norm;
            }
        }
    }
    x
}

#[derive(Debug, Clone)]
/// EigSolver value.
pub struct EigSolver {
    /// Which value.
    pub which: String,
    /// Eigenvectors value.
    pub eigenvectors: Option<Array2<f64>>,
    /// Eigenvalues value.
    pub eigenvalues: Option<Array1<f64>>,
}

impl EigSolver {
    /// Creates a new instance.
    pub fn new(which: &str) -> Self {
        Self {
            which: which.to_uppercase(),
            eigenvectors: None,
            eigenvalues: None,
        }
    }
}

#[derive(Debug, Clone)]
/// LanczosEig value.
pub struct LanczosEig {
    /// Which value.
    pub which: String,
    /// N Iter value.
    pub n_iter: Option<usize>,
    /// Tol value.
    pub tol: f64,
    /// Eigenvectors value.
    pub eigenvectors: Option<Array2<f64>>,
    /// Eigenvalues value.
    pub eigenvalues: Option<Array1<f64>>,
}

impl LanczosEig {
    /// Creates a new instance.
    pub fn new(which: &str, n_iter: Option<usize>, tol: f64) -> Self {
        Self {
            which: which.to_uppercase(),
            n_iter,
            tol,
            eigenvectors: None,
            eigenvalues: None,
        }
    }

    /// Runs the fit step.
    ///
    /// # Errors
    ///
    /// Returns [`EigError`] on failure.
    pub fn fit(&mut self, matrix: EigInput, n_components: usize) -> Result<(), EigError> {
        let n = match &matrix {
            EigInput::Sparse(a) => a.rows(),
            EigInput::SparseLR(a) => a.sparse_mat.rows(),
        };
        if n == 0 {
            return Err(EigError::EmptyMatrix);
        }
        if n_components == 0 || n_components > n {
            return Err(EigError::InvalidComponents);
        }

        if !matches!(self.which.as_str(), "LM" | "SM" | "LA" | "SA") {
            return Err(EigError::UnknownWhich);
        }

        let k = n_components;
        let mut q = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for c in 0..k {
                q[[i, c]] = ((i as f64 + 1.0) * (c as f64 + 2.0)).sin();
            }
        }
        q = normalize_columns(q);

        let max_iter = self.n_iter.unwrap_or(80);
        let tol = self.tol.max(1e-12);
        let mut last_vals = vec![0.0; k];
        for _ in 0..max_iter {
            let mut y = apply_input(&matrix, &q);
            if matches!(self.which.as_str(), "SM" | "SA") {
                // crude smallest-eigen proxy: iterate on shifted negative
                for i in 0..n {
                    for c in 0..k {
                        y[[i, c]] = -y[[i, c]];
                    }
                }
            }
            q = normalize_columns(y);
            let aq = apply_input(&matrix, &q);
            let mut vals = vec![0.0; k];
            for c in 0..k {
                let qc = q.index_axis(Axis(1), c);
                let aqc = aq.index_axis(Axis(1), c);
                let num = qc.dot(&aqc);
                let den = qc.dot(&qc);
                vals[c] = if den > 0.0 { num / den } else { 0.0 };
            }
            let diff: f64 = vals
                .iter()
                .zip(last_vals.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            last_vals = vals;
            if diff < tol {
                break;
            }
        }
        self.eigenvalues = Some(Array1::from_vec(last_vals));
        self.eigenvectors = Some(q);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use sprs::TriMat;

    use super::*;
    use crate::data::test_graphs::test_graph;

    #[test]
    fn test_lanczos() {
        let adjacency = test_graph();
        let mut solver = LanczosEig::new("LM", None, 1e-10);
        solver.fit(EigInput::Sparse(adjacency.clone()), 2).unwrap();
        assert_eq!(solver.eigenvalues.as_ref().unwrap().len(), 2);
        assert_eq!(
            solver.eigenvectors.as_ref().unwrap().dim(),
            (adjacency.rows(), 2)
        );

        let n = adjacency.rows();
        let x = Array1::from_vec((0..n).map(|i| (i as f64 + 1.0) / n as f64).collect());
        let slr = SparseLR::new(&adjacency, vec![(x.clone(), x)]).unwrap();
        let slr_n = slr.sparse_mat.rows();
        solver.fit(EigInput::SparseLR(slr), 2).unwrap();
        assert_eq!(solver.eigenvectors.as_ref().unwrap().dim(), (slr_n, 2));

        let mut tri = TriMat::<f64>::new((4, 4));
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(1, 2, 1.0);
        tri.add_triplet(2, 1, 1.0);
        tri.add_triplet(2, 3, 1.0);
        tri.add_triplet(3, 2, 1.0);
        let adjacency = tri.to_csr::<usize>();
        let mut solver = LanczosEig::new("SM", None, 1e-10);
        solver.fit(EigInput::Sparse(adjacency), 2).unwrap();
        assert_eq!(solver.eigenvalues.as_ref().unwrap().len(), 2);
        assert_eq!(solver.eigenvectors.as_ref().unwrap().dim(), (4, 2));
    }

    #[test]
    fn test_which_case_insensitive() {
        let adjacency = test_graph();
        let mut solver = LanczosEig::new("lm", None, 1e-10);
        solver.fit(EigInput::Sparse(adjacency), 2).unwrap();
        assert_eq!(solver.eigenvalues.as_ref().unwrap().len(), 2);
    }
}
