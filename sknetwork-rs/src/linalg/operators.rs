use ndarray::{Array1, Array2};
use sprs::{CsMat, TriMat};

use crate::linalg::normalizer::diagonal_pseudo_inverse;

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
/// Regularizer value.
pub struct Regularizer {
    /// Adjacency value.
    pub adjacency: CsMat<f64>,
    /// Regularization value.
    pub regularization: f64,
}

impl Regularizer {
    /// Creates a new instance.
    pub fn new(input_matrix: &CsMat<f64>, regularization: f64) -> Self {
        Self {
            adjacency: input_matrix.clone(),
            regularization,
        }
    }

    /// Computes dot vec.
    pub fn dot_vec(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut out = sparse_vec_mul(&self.adjacency, x);
        let mean = if x.is_empty() {
            0.0
        } else {
            x.sum() / x.len() as f64
        };
        for v in &mut out {
            *v += self.regularization * mean;
        }
        out
    }
}

#[derive(Debug, Clone)]
/// Normalizer value.
pub struct Normalizer {
    /// Regularization value.
    pub regularization: f64,
    adjacency: CsMat<f64>,
    norm_diag: CsMat<f64>,
}

impl Normalizer {
    /// Creates a new instance.
    pub fn new(adjacency: &CsMat<f64>, regularization: f64) -> Self {
        let n_col = adjacency.cols();
        let mut weights = Array1::<f64>::zeros(adjacency.rows());
        for (i, row) in adjacency.outer_iterator().enumerate() {
            weights[i] = row.data().iter().sum::<f64>() + regularization;
        }
        debug_assert_eq!(n_col, adjacency.cols());
        Self {
            regularization,
            adjacency: adjacency.clone(),
            norm_diag: diagonal_pseudo_inverse(&weights),
        }
    }

    /// Computes dot vec.
    pub fn dot_vec(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut prod = sparse_vec_mul(&self.adjacency, x);
        if self.regularization > 0.0 {
            let mean = if x.is_empty() {
                0.0
            } else {
                x.sum() / x.len() as f64
            };
            for v in &mut prod {
                *v += self.regularization * mean;
            }
        }
        sparse_vec_mul(&self.norm_diag, &prod)
    }

    /// Computes dot mat.
    pub fn dot_mat(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut prod = sparse_mat_mul(&self.adjacency, x);
        if self.regularization > 0.0 {
            let n_row = prod.nrows();
            for c in 0..prod.ncols() {
                let mean = x.column(c).sum() / x.nrows() as f64;
                for i in 0..n_row {
                    prod[[i, c]] += self.regularization * mean;
                }
            }
        }
        sparse_mat_mul(&self.norm_diag, &prod)
    }
}

#[derive(Debug, Clone)]
/// Laplacian value.
pub struct Laplacian {
    /// Regularization value.
    pub regularization: f64,
    /// Normalized Laplacian value.
    pub normalized_laplacian: bool,
    laplacian: CsMat<f64>,
    norm_diag: Option<CsMat<f64>>,
}

impl Laplacian {
    /// Creates a new instance.
    pub fn new(adjacency: &CsMat<f64>, regularization: f64, normalized_laplacian: bool) -> Self {
        let n = adjacency.rows();
        let mut weights = Array1::<f64>::zeros(n);
        for (i, row) in adjacency.outer_iterator().enumerate() {
            weights[i] = row.data().iter().sum::<f64>();
        }
        let mut tri = TriMat::<f64>::new((n, n));
        for i in 0..n {
            tri.add_triplet(i, i, weights[i]);
        }
        for (i, row) in adjacency.outer_iterator().enumerate() {
            for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                tri.add_triplet(i, j, -v);
            }
        }
        let lap = tri.to_csr::<usize>();
        let norm_diag = if normalized_laplacian {
            let shifted = weights.mapv(|w| (w + regularization).sqrt());
            Some(diagonal_pseudo_inverse(&shifted))
        } else {
            None
        };
        Self {
            regularization,
            normalized_laplacian,
            laplacian: lap,
            norm_diag,
        }
    }

    /// Computes dot vec.
    pub fn dot_vec(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut z = x.clone();
        if let Some(d) = &self.norm_diag {
            z = sparse_vec_mul(d, &z);
        }
        let mut prod = sparse_vec_mul(&self.laplacian, &z);
        if self.regularization > 0.0 {
            let mean = if z.is_empty() {
                0.0
            } else {
                z.sum() / z.len() as f64
            };
            for i in 0..prod.len() {
                prod[i] += self.regularization * (z[i] - mean);
            }
        }
        if let Some(d) = &self.norm_diag {
            prod = sparse_vec_mul(d, &prod);
        }
        prod
    }

    /// Computes dot mat.
    pub fn dot_mat(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut z = x.clone();
        if let Some(d) = &self.norm_diag {
            z = sparse_mat_mul(d, &z);
        }
        let mut prod = sparse_mat_mul(&self.laplacian, &z);
        if self.regularization > 0.0 {
            for c in 0..z.ncols() {
                let mean = z.column(c).sum() / z.nrows() as f64;
                for i in 0..z.nrows() {
                    prod[[i, c]] += self.regularization * (z[[i, c]] - mean);
                }
            }
        }
        if let Some(d) = &self.norm_diag {
            prod = sparse_mat_mul(d, &prod);
        }
        prod
    }
}

#[derive(Debug, Clone)]
/// CoNeighbor value.
pub struct CoNeighbor {
    /// Forward value.
    pub forward: CsMat<f64>,
    /// Backward value.
    pub backward: CsMat<f64>,
}

impl CoNeighbor {
    /// Creates a new instance.
    pub fn new(adjacency: &CsMat<f64>, normalized: bool) -> Self {
        let forward = if normalized {
            // normalize rows of adjacency.T (i.e., columns of adjacency)
            let at = adjacency.transpose_view().to_csr();
            let mut tri = TriMat::<f64>::new(at.shape());
            for (i, row) in at.outer_iterator().enumerate() {
                let s: f64 = row.data().iter().map(|x| x.abs()).sum();
                if s > 0.0 {
                    for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                        tri.add_triplet(i, j, v / s);
                    }
                }
            }
            tri.to_csr::<usize>()
        } else {
            adjacency.transpose_view().to_csr()
        };
        Self {
            forward,
            backward: adjacency.clone(),
        }
    }

    /// Computes dot vec.
    pub fn dot_vec(&self, x: &Array1<f64>) -> Array1<f64> {
        let y = sparse_vec_mul(&self.forward, x);
        sparse_vec_mul(&self.backward, &y)
    }

    /// Computes transpose.
    pub fn transpose(&self) -> Self {
        Self {
            backward: self.forward.transpose_view().to_csr(),
            forward: self.backward.transpose_view().to_csr(),
        }
    }

    /// Computes negate.
    pub fn negate(&mut self) {
        for v in self.backward.data_mut() {
            *v *= -1.0;
        }
    }

    /// Computes scale.
    pub fn scale(&mut self, factor: f64) {
        for v in self.backward.data_mut() {
            *v *= factor;
        }
    }

    /// Computes left sparse dot.
    pub fn left_sparse_dot(&mut self, matrix: &CsMat<f64>) {
        self.backward = matrix * &self.backward;
    }

    /// Computes right sparse dot.
    pub fn right_sparse_dot(&mut self, matrix: &CsMat<f64>) {
        self.forward = &self.forward * matrix;
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use sprs::CsMat;

    use super::*;
    use crate::data::test_graphs::{test_bigraph, test_disconnected_graph, test_graph};

    fn l2(v: &Array1<f64>) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    fn test_laplacian() {
        for adjacency in [test_graph(), test_disconnected_graph()] {
            let n = adjacency.cols();
            let ones = Array1::ones(n);

            let lap = Laplacian::new(&adjacency, 0.0, false);
            assert!(l2(&lap.dot_vec(&ones)) < 1e-9);

            let lap = Laplacian::new(&adjacency, 0.0, true);
            let mut weights = Array1::<f64>::zeros(n);
            for (i, row) in adjacency.outer_iterator().enumerate() {
                weights[i] = row.data().iter().sum::<f64>().sqrt();
            }
            assert!(l2(&lap.dot_vec(&weights)) < 1e-8);

            let regularization = 0.1;
            let lap = Laplacian::new(&adjacency, regularization, true);
            let mut weights = Array1::<f64>::zeros(n);
            for (i, row) in adjacency.outer_iterator().enumerate() {
                weights[i] = (row.data().iter().sum::<f64>() + regularization).sqrt();
            }
            assert!(l2(&lap.dot_vec(&weights)) < 1e-7);

            let shape = (n, 3);
            let x = Array2::<f64>::ones(shape);
            assert_eq!(lap.dot_mat(&x).dim(), shape);
        }
    }

    #[test]
    fn test_normalizer() {
        for adjacency in [test_graph(), test_disconnected_graph()] {
            let n_row = adjacency.rows();
            let n_col = adjacency.cols();
            let ones = Array1::<f64>::ones(n_col);
            let normalizer = Normalizer::new(&adjacency, 0.0);
            let y = normalizer.dot_vec(&ones);
            let non_zeros: Array1<f64> = (0..n_row)
                .map(|i| {
                    if adjacency
                        .outer_view(i)
                        .map(|r| r.data().iter().sum::<f64>())
                        .unwrap_or(0.0)
                        > 0.0
                    {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            assert!(l2(&(y - non_zeros)) < 1e-10);

            let mut tri = TriMat::<f64>::new((1, n_col));
            if let Some(row1) = adjacency.outer_view(1) {
                for (&j, &v) in row1.indices().iter().zip(row1.data().iter()) {
                    tri.add_triplet(0, j, v);
                }
            }
            let row_adj = tri.to_csr::<usize>();
            let normalizer = Normalizer::new(&row_adj, 0.0);
            assert!((normalizer.dot_vec(&ones)[0] - 1.0).abs() < 1e-12);

            let normalizer = Normalizer::new(&adjacency, 1.0);
            assert!(l2(&(normalizer.dot_vec(&ones) - Array1::<f64>::ones(n_row))) < 1e-9);
        }
    }

    #[test]
    fn test_coneighbors() {
        let biadjacency = test_bigraph();
        let mut operator = CoNeighbor::new(&biadjacency, true);
        operator.right_sparse_dot(&CsMat::eye(operator.forward.cols()));

        let mut operator1 = CoNeighbor::new(&biadjacency, false);
        let mut operator2 = CoNeighbor::new(&biadjacency, false);
        let x = Array1::from_vec(vec![0.3, -0.4, 1.2, 0.5, -1.1, 0.2, 0.0, 0.7]);
        operator1.negate();
        let x1 = operator1.dot_vec(&x);
        operator2.scale(-1.0);
        let x2 = operator2.dot_vec(&x);
        let x3 = operator1.transpose().dot_vec(&x);
        assert!(l2(&(x1 - x2.clone())) < 1e-12);
        assert!(l2(&(x2 - x3)) < 1e-12);
    }
}
