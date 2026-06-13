use sprs::CsMat;

use crate::path::dag::{DagError, get_dag};
use crate::path::distances::{DistanceError, DistanceResult, get_distances_full};
use crate::utils::format::bipartite2undirected;

/// Errors raised while building shortest-path DAGs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShortestPathError {
    /// No source nodes were provided.
    MissingSource,
    /// Bipartite distance output was returned for a square-graph request.
    ConflictingSources,
    /// Distance computation failed.
    Distance(DistanceError),
    /// DAG extraction failed.
    Dag(DagError),
}

/// Builds a shortest-path DAG from one or more source nodes.
///
/// Computes multi-source distances, then keeps edges that lie on at least one
/// shortest path from any source.
///
/// # Errors
/// Returns [`ShortestPathError::MissingSource`] when `sources` is empty,
/// [`ShortestPathError::ConflictingSources`] for unexpected bipartite output,
/// and wrapped [`DistanceError`] or [`DagError`] variants on failure.
pub fn get_shortest_path(
    adjacency: &CsMat<f64>,
    sources: &[usize],
) -> Result<CsMat<f64>, ShortestPathError> {
    if sources.is_empty() {
        return Err(ShortestPathError::MissingSource);
    }
    let order = match get_distances_full(adjacency, Some(sources), None, None, false, false)
        .map_err(ShortestPathError::Distance)?
    {
        DistanceResult::Single(distances) => distances,
        DistanceResult::Bipartite(_, _) => return Err(ShortestPathError::ConflictingSources),
    };
    get_dag(adjacency, None, Some(&order)).map_err(ShortestPathError::Dag)
}

/// Builds a shortest-path DAG on a bipartite graph from row and column sources.
///
/// Column sources are offset by the row count in the stacked undirected layout.
///
/// # Errors
/// Returns the same errors as [`get_shortest_path`].
pub fn get_shortest_path_bipartite(
    biadjacency: &CsMat<f64>,
    source_row: &[usize],
    source_col: &[usize],
) -> Result<CsMat<f64>, ShortestPathError> {
    if source_row.is_empty() && source_col.is_empty() {
        return Err(ShortestPathError::MissingSource);
    }
    let n_row = biadjacency.rows();
    let adjacency = bipartite2undirected(biadjacency);
    let mut sources = Vec::new();
    sources.extend_from_slice(source_row);
    sources.extend(source_col.iter().map(|c| n_row + c));
    get_shortest_path(&adjacency, &sources)
}

/// Builds a shortest-path DAG with square or bipartite source conventions.
///
/// # Arguments
/// - `source`: Unified source indices for square graphs.
/// - `source_row`: Row-node sources for bipartite inputs.
/// - `source_col`: Column-node sources for bipartite inputs.
/// - `force_bipartite`: Whether to treat the input as bipartite.
///
/// # Errors
/// Returns the same errors as [`get_shortest_path`] and
/// [`get_distances_full`].
pub fn get_shortest_path_full(
    input_matrix: &CsMat<f64>,
    source: Option<&[usize]>,
    source_row: Option<&[usize]>,
    source_col: Option<&[usize]>,
    force_bipartite: bool,
) -> Result<CsMat<f64>, ShortestPathError> {
    let distances = get_distances_full(
        input_matrix,
        source,
        source_row,
        source_col,
        false,
        force_bipartite,
    )
    .map_err(ShortestPathError::Distance)?;
    match distances {
        DistanceResult::Single(order) => {
            get_dag(input_matrix, None, Some(&order)).map_err(ShortestPathError::Dag)
        }
        DistanceResult::Bipartite(row, col) => {
            let adjacency = bipartite2undirected(input_matrix);
            let mut order = row;
            order.extend(col);
            get_dag(&adjacency, None, Some(&order)).map_err(ShortestPathError::Dag)
        }
    }
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
    fn test_shortest_path_basic() {
        let empty = dense_to_csr(&array![[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]);
        let path = get_shortest_path(&empty, &[0]).unwrap();
        assert_eq!(path.nnz(), 0);

        let adjacency = dense_to_csr(&array![
            [0., 1., 1., 0.],
            [0., 0., 1., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 0.]
        ]);
        let path = get_shortest_path(&adjacency, &[0]).unwrap();
        assert!(path.nnz() > 0);
        let path_multi = get_shortest_path(&adjacency, &[0, 3]).unwrap();
        assert!(path_multi.nnz() <= path.nnz());
        let path_full =
            get_shortest_path_full(&adjacency, Some(&[0, 3]), None, None, false).unwrap();
        assert_eq!(path_full.nnz(), path_multi.nnz());
    }

    #[test]
    fn test_shortest_path_bipartite() {
        let biadjacency = dense_to_csr(&array![[1., 0., 1.], [0., 1., 0.]]);
        let path = get_shortest_path_bipartite(&biadjacency, &[0], &[]).unwrap();
        assert_eq!(path.shape(), (5, 5));
        assert!(path.nnz() > 0);
        let path_col = get_shortest_path_bipartite(&biadjacency, &[], &[0, 1]).unwrap();
        assert_eq!(path_col.shape(), (5, 5));
        let path_full = get_shortest_path_full(&biadjacency, Some(&[0]), None, None, true).unwrap();
        assert_eq!(path_full.shape(), (5, 5));
    }
}
