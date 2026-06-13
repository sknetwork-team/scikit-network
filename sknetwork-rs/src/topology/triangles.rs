use sprs::CsMat;

use crate::path::dag::{DagError, get_dag};
use crate::utils::check::{CheckError, check_square};
use crate::utils::format::directed2undirected;
use crate::utils::neighbors::get_degrees;

#[derive(Debug, Clone, PartialEq)]
/// Errors raised by triangle error operations.
pub enum TriangleError {
    /// Indicates check.
    Check(CheckError),
    /// Indicates dag.
    Dag(DagError),
    /// Indicates empty edge pairs.
    EmptyEdgePairs,
}

fn count_local_triangles_from_dag(dag: &CsMat<f64>, node: usize) -> usize {
    let Some(row_node) = dag.outer_view(node) else {
        return 0;
    };
    let neighbors_node = row_node.indices();
    let mut n_triangles = 0usize;

    for &neighbor in neighbors_node {
        let Some(row_neighbor) = dag.outer_view(neighbor) else {
            continue;
        };
        let neighbors_neighbor = row_neighbor.indices();

        let mut i = 0usize;
        let mut j = 0usize;
        while i < neighbors_node.len() && j < neighbors_neighbor.len() {
            let a = neighbors_node[i];
            let b = neighbors_neighbor[j];
            if a == b {
                n_triangles += 1;
                i += 1;
                j += 1;
            } else if a < b {
                i += 1;
            } else {
                j += 1;
            }
        }
    }
    n_triangles
}

/// Counts triangles.
///
/// # Errors
///
/// Returns [`TriangleError`] on failure.
pub fn count_triangles(adjacency: &CsMat<f64>, _parallelize: bool) -> Result<usize, TriangleError> {
    check_square(adjacency.shape()).map_err(TriangleError::Check)?;
    let undirected = directed2undirected(adjacency, true);
    let dag = get_dag(&undirected, None, None).map_err(TriangleError::Dag)?;
    let mut n_triangles = 0usize;
    for node in 0..dag.rows() {
        n_triangles += count_local_triangles_from_dag(&dag, node);
    }
    Ok(n_triangles)
}

/// Returns clustering coefficient.
pub fn get_clustering_coefficient(
    adjacency: &CsMat<f64>,
    parallelize: bool,
) -> Result<f64, TriangleError> {
    let n_triangles = count_triangles(adjacency, parallelize)?;
    let undirected = directed2undirected(adjacency, true);
    let degrees = get_degrees(&undirected, false);
    let mut n_edge_pairs = 0f64;
    for degree in degrees {
        if degree > 1 {
            let d = degree as f64;
            n_edge_pairs += d * (d - 1.0) / 2.0;
        }
    }
    if n_edge_pairs == 0.0 {
        return Err(TriangleError::EmptyEdgePairs);
    }
    Ok((3.0 * n_triangles as f64) / n_edge_pairs)
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
        tri.to_csr::<usize>()
    }

    #[test]
    fn test_empty() {
        let adjacency = dense_to_csr(&array![[0., 0.], [0., 0.]]);
        assert_eq!(count_triangles(&adjacency, false).unwrap(), 0);
    }

    #[test]
    fn test_disconnected_with_one_triangle() {
        let adjacency = dense_to_csr(&array![
            [0., 1., 1., 0., 0.],
            [1., 0., 1., 0., 0.],
            [1., 1., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [0., 0., 0., 1., 0.]
        ]);
        assert_eq!(count_triangles(&adjacency, false).unwrap(), 1);
    }

    #[test]
    fn test_cliques() {
        let n = 5;
        let mut tri = TriMat::<f64>::new((n, n));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    tri.add_triplet(i, j, 1.0);
                }
            }
        }
        let adjacency = tri.to_csr::<usize>();
        assert_eq!(count_triangles(&adjacency, false).unwrap(), 10);
    }

    #[test]
    fn test_clustering_coefficient() {
        let adjacency = dense_to_csr(&array![
            [0., 1., 1., 1.],
            [1., 0., 1., 0.],
            [1., 1., 0., 1.],
            [1., 0., 1., 0.]
        ]);
        let coefficient = get_clustering_coefficient(&adjacency, false).unwrap();
        assert!((coefficient - 0.75).abs() <= 1e-12);
    }
}
