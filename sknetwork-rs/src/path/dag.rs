use sprs::{CsMat, TriMat};

use crate::path::distances::{DistanceError, get_distances};

/// Errors raised while extracting a directed acyclic graph from distances.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DagError {
    /// Distance computation failed.
    Distance(DistanceError),
    /// The supplied node order length does not match the graph order.
    InvalidOrderLength,
    /// The adjacency matrix is not square.
    NotSquare,
}

/// Builds a shortest-path DAG from an adjacency matrix and node ordering.
///
/// Keeps edge `(i, j)` when `order[i] >= 0` and `order[i] < order[j]`. When
/// `order` is omitted and `source` is provided, distances from that source
/// define the order; otherwise the identity order is used.
///
/// # Errors
/// Returns [`DagError::NotSquare`] for non-square inputs,
/// [`DagError::InvalidOrderLength`] when a custom order has the wrong length,
/// and [`DagError::Distance`] when distance computation fails.
pub fn get_dag(
    adjacency: &CsMat<f64>,
    source: Option<usize>,
    order: Option<&[i32]>,
) -> Result<CsMat<f64>, DagError> {
    if adjacency.rows() != adjacency.cols() {
        return Err(DagError::NotSquare);
    }
    let n = adjacency.rows();
    let ord: Vec<i32> = if let Some(order) = order {
        if order.len() != n {
            return Err(DagError::InvalidOrderLength);
        }
        order.to_vec()
    } else if source.is_none() {
        (0..n as i32).collect()
    } else {
        get_distances(adjacency, source).map_err(DagError::Distance)?
    };

    let mut tri = TriMat::<f64>::new((n, n));
    for (i, row) in adjacency.outer_iterator().enumerate() {
        for (j, _) in row.iter() {
            if ord[i] >= 0 && ord[i] < ord[j] {
                tri.add_triplet(i, j, 1.0);
            }
        }
    }
    Ok(tri.to_csr::<usize>())
}

#[cfg(test)]
mod tests {
    use ndarray::array;

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
        tri.to_csr::<usize>()
    }

    #[test]
    fn test_get_dag_from_source() {
        let adjacency = dense_to_csr(&array![[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]);
        let dag = get_dag(&adjacency, Some(0), None).unwrap();
        assert_eq!(dag.nnz(), 2);
    }

    #[test]
    fn test_get_dag_with_order() {
        let adjacency = dense_to_csr(&array![
            [0., 1., 1., 0.],
            [1., 0., 1., 1.],
            [1., 1., 0., 1.],
            [0., 1., 1., 0.]
        ]);
        let dag_default = get_dag(&adjacency, None, None).unwrap();
        assert!(dag_default.nnz() > 0);
        let order = [0, 1, 0, 1];
        let dag_ordered = get_dag(&adjacency, None, Some(&order)).unwrap();
        assert!(dag_ordered.nnz() <= dag_default.nnz());
    }
}
