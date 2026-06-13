use ndarray::{Array1, Array2};
use sprs::CsMat;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by polynome error operations.
pub enum PolynomeError {
    /// Indicates empty coefficients.
    EmptyCoefficients,
    /// Indicates non square matrix.
    NonSquareMatrix,
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

fn sparse_mat_mul(mat: &CsMat<f64>, x: &Array2<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((mat.rows(), x.ncols()));
    for (i, row) in mat.outer_iterator().enumerate() {
        for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
            for c in 0..x.ncols() {
                out[[i, c]] += v * x[[j, c]];
            }
        }
    }
    out
}

#[derive(Debug, Clone)]
/// Polynome value.
pub struct Polynome {
    /// Matrix value.
    pub matrix: CsMat<f64>,
    /// Coeffs value.
    pub coeffs: Array1<f64>,
}

impl Polynome {
    /// Creates a new instance.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomeError`] on failure.
    pub fn new(matrix: &CsMat<f64>, coeffs: &Array1<f64>) -> Result<Self, PolynomeError> {
        if coeffs.is_empty() {
            return Err(PolynomeError::EmptyCoefficients);
        }
        if matrix.rows() != matrix.cols() {
            return Err(PolynomeError::NonSquareMatrix);
        }
        Ok(Self {
            matrix: matrix.clone(),
            coeffs: coeffs.clone(),
        })
    }

    /// Computes neg.
    pub fn neg(&self) -> Self {
        Self {
            matrix: self.matrix.clone(),
            coeffs: self.coeffs.mapv(|x| -x),
        }
    }

    /// Computes scale.
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            matrix: self.matrix.clone(),
            coeffs: self.coeffs.mapv(|x| factor * x),
        }
    }

    /// Computes dot vec.
    pub fn dot_vec(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut y = x.mapv(|v| v * self.coeffs[self.coeffs.len() - 1]);
        for &a in self.coeffs.iter().rev().skip(1) {
            y = sparse_vec_mul(&self.matrix, &y) + x.mapv(|v| a * v);
        }
        y
    }

    /// Computes dot mat.
    pub fn dot_mat(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut y = x.mapv(|v| v * self.coeffs[self.coeffs.len() - 1]);
        for &a in self.coeffs.iter().rev().skip(1) {
            y = sparse_mat_mul(&self.matrix, &y) + x.mapv(|v| a * v);
        }
        y
    }

    /// Computes transpose.
    pub fn transpose(&self) -> Self {
        Self {
            matrix: self.matrix.transpose_view().to_csr(),
            coeffs: self.coeffs.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, array};
    use sprs::TriMat;

    use super::*;
    use crate::data::test_graphs::test_graph;

    #[test]
    fn test_init() {
        let adjacency = test_graph();
        assert!(matches!(
            Polynome::new(&adjacency, &Array1::<f64>::zeros(0)),
            Err(PolynomeError::EmptyCoefficients)
        ));
    }

    #[test]
    fn test_operations() {
        let adjacency = test_graph();
        let n = adjacency.rows();
        let p = Polynome::new(&adjacency, &array![0.0, 1.0, 2.0]).unwrap();
        let x = Array1::from_vec((0..n).map(|i| (i as f64 * 0.3) - 1.0).collect());

        let y1 = p.scale(2.0).dot_vec(&x);
        let y2 = p.neg().dot_vec(&x);
        let err = (y1.mapv(|v| 0.5 * v) + y2)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-12);
    }

    #[test]
    fn test_dot() {
        let mut tri = TriMat::<f64>::new((5, 5));
        for i in 0..5 {
            tri.add_triplet(i, i, 1.0);
        }
        let adjacency = tri.to_csr::<usize>();
        let p = Polynome::new(&adjacency, &array![0.0, 1.0]).unwrap();

        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                0.1, 0.2, -0.4, 0.3, -0.1, 0.9, 1.0, 0.0, 0.5, -0.2, 0.7, 0.8, 0.6, -0.3, 0.4,
            ],
        )
        .unwrap();
        let y = p.dot_mat(&x);
        let err = (&x - &y).iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(err < 1e-12);
    }
}
