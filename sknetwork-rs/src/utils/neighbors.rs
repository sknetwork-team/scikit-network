use ndarray::Array1;
use sprs::CsMat;

/// Returns neighbors with transpose.
pub fn get_neighbors_with_transpose(
    input_matrix: &CsMat<f64>,
    transposed_matrix: Option<&CsMat<f64>>,
    node: usize,
    transpose: bool,
) -> Vec<usize> {
    if !transpose {
        return input_matrix
            .outer_view(node)
            .map(|row| row.indices().to_vec())
            .unwrap_or_default();
    }
    let fallback_transposed;
    let matrix_t = if let Some(m) = transposed_matrix {
        m
    } else {
        fallback_transposed = input_matrix.transpose_view().to_csr();
        &fallback_transposed
    };
    matrix_t
        .outer_view(node)
        .map(|row| row.indices().to_vec())
        .unwrap_or_default()
}

/// Returns neighbors.
pub fn get_neighbors(input_matrix: &CsMat<f64>, node: usize, transpose: bool) -> Vec<usize> {
    get_neighbors_with_transpose(input_matrix, None, node, transpose)
}

/// Returns degrees.
pub fn get_degrees(input_matrix: &CsMat<f64>, transpose: bool) -> Array1<usize> {
    if !transpose {
        let mut out = Array1::<usize>::zeros(input_matrix.rows());
        for i in 0..input_matrix.rows() {
            out[i] = input_matrix.outer_view(i).map(|row| row.nnz()).unwrap_or(0);
        }
        return out;
    }

    let mut out = Array1::<usize>::zeros(input_matrix.cols());
    for row in input_matrix.outer_iterator() {
        for &j in row.indices() {
            out[j] += 1;
        }
    }
    out
}

/// Returns weights.
pub fn get_weights(input_matrix: &CsMat<f64>, transpose: bool) -> Array1<f64> {
    if !transpose {
        let mut out = Array1::<f64>::zeros(input_matrix.rows());
        for i in 0..input_matrix.rows() {
            out[i] = input_matrix
                .outer_view(i)
                .map(|row| row.data().iter().sum())
                .unwrap_or(0.0);
        }
        return out;
    }

    let mut out = Array1::<f64>::zeros(input_matrix.cols());
    for row in input_matrix.outer_iterator() {
        for (j, v) in row.iter() {
            out[j] += v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use sprs::TriMat;

    use super::*;

    fn dense_to_csr(dense: &ndarray::Array2<f64>) -> CsMat<f64> {
        let (r, c) = dense.dim();
        let mut tri = TriMat::<f64>::new((r, c));
        for i in 0..r {
            for j in 0..c {
                if dense[[i, j]] != 0.0 {
                    tri.add_triplet(i, j, dense[[i, j]]);
                }
            }
        }
        tri.to_csr()
    }

    #[test]
    fn test_graph() {
        let adjacency = dense_to_csr(&array![
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 0.0]
        ]);
        let neighbors = get_neighbors(&adjacency, 0, false);
        assert_eq!(neighbors, vec![1, 4]);
        let degrees = get_degrees(&adjacency, false);
        assert_eq!(degrees[0], 2);
    }

    #[test]
    fn test_digraph() {
        let adjacency = dense_to_csr(&array![
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0]
        ]);
        let neighbors = get_neighbors(&adjacency, 0, false);
        assert_eq!(neighbors, vec![3]);
        let out_degrees = get_degrees(&adjacency, false);
        assert_eq!(out_degrees[0], 1);
        let out_weights = get_weights(&adjacency, false);
        assert_eq!(out_weights[0], 1.0);

        let predecessors = get_neighbors(&adjacency, 0, true);
        assert_eq!(predecessors, vec![1, 3]);
        let adjacency_t = adjacency.transpose_view().to_csr();
        let predecessors_cached = get_neighbors_with_transpose(&adjacency, Some(&adjacency_t), 0, true);
        assert_eq!(predecessors_cached, vec![1, 3]);
        let in_degrees = get_degrees(&adjacency, true);
        assert_eq!(in_degrees[0], 2);
    }
}
