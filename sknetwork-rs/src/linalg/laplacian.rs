use ndarray::Array1;
use sprs::{CsMat, TriMat};

/// Returns laplacian.
pub fn get_laplacian(adjacency: &CsMat<f64>) -> CsMat<f64> {
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
    tri.to_csr::<usize>()
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*;
    use crate::data::test_graphs::test_graph;

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
    fn test() {
        let adjacency = test_graph();
        let laplacian = get_laplacian(&adjacency);
        let ones = Array1::<f64>::ones(adjacency.rows());
        let y = sparse_vec_mul(&laplacian, &ones);
        let norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm < 1e-12);
    }
}
