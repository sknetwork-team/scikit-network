use sprs::CsMat;
use std::collections::VecDeque;

use crate::path::distances::{DistanceError, get_distances};

/// Runs breadth-first search from a source node.
///
/// Returns node indices in discovery order within the source's connected
/// component.
///
/// # Errors
/// Returns [`DistanceError::SourceOutOfBounds`] when `source` is out of range.
pub fn breadth_first_search(
    adjacency: &CsMat<f64>,
    source: usize,
) -> Result<Vec<usize>, DistanceError> {
    let _ = get_distances(adjacency, Some(source))?;
    let n = adjacency.rows();
    let mut seen = vec![false; n];
    let mut order = Vec::new();
    let mut q = VecDeque::new();
    seen[source] = true;
    q.push_back(source);
    while let Some(u) = q.pop_front() {
        order.push(u);
        if let Some(row) = adjacency.outer_view(u) {
            for &v in row.indices() {
                if !seen[v] {
                    seen[v] = true;
                    q.push_back(v);
                }
            }
        }
    }
    Ok(order)
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
    fn test_bfs() {
        let adjacency = dense_to_csr(&array![[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]);
        let search = breadth_first_search(&adjacency, 0).unwrap();
        assert_eq!(search, vec![0, 1, 2]);

        let empty = dense_to_csr(&array![[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]);
        let search_empty = breadth_first_search(&empty, 0).unwrap();
        assert_eq!(search_empty, vec![0]);

        let disconnected = dense_to_csr(&array![
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 0., 0.]
        ]);
        let search_disconnected = breadth_first_search(&disconnected, 2).unwrap();
        assert_eq!(search_disconnected, vec![2, 3]);
    }
}
