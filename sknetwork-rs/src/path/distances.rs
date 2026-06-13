use std::collections::VecDeque;

use crate::utils::format::bipartite2undirected;
use sprs::CsMat;

/// Errors raised while computing graph distances.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistanceError {
    /// No source node was provided.
    MissingSource,
    /// Both unified and row-specific sources were supplied for bipartite input.
    ConflictingSources,
    /// The source list is empty.
    EmptySources,
    /// A source index exceeds the graph order.
    SourceOutOfBounds,
}

/// Distance output for square or bipartite graphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistanceResult {
    /// Distances from one or more sources in a square graph.
    Single(Vec<i32>),
    /// Row and column distances in a bipartite layout.
    Bipartite(Vec<i32>, Vec<i32>),
}

/// Computes hop distances from a single source node.
///
/// Unreachable nodes receive distance `-1`.
///
/// # Errors
/// Returns [`DistanceError::MissingSource`] when `source` is `None`.
pub fn get_distances(
    adjacency: &CsMat<f64>,
    source: Option<usize>,
) -> Result<Vec<i32>, DistanceError> {
    let source = source.ok_or(DistanceError::MissingSource)?;
    get_distances_multi(adjacency, &[source])
}

/// Computes hop distances from multiple source nodes.
///
/// Each source receives distance `0`; other nodes receive the minimum hop
/// count from any source, or `-1` when unreachable.
///
/// # Errors
/// Returns [`DistanceError::EmptySources`] for an empty source list and
/// [`DistanceError::SourceOutOfBounds`] when a source index is invalid.
pub fn get_distances_multi(
    adjacency: &CsMat<f64>,
    sources: &[usize],
) -> Result<Vec<i32>, DistanceError> {
    let n = adjacency.rows();
    if sources.is_empty() {
        return Err(DistanceError::EmptySources);
    }
    let mut distances = vec![-1; n];
    let mut queue = VecDeque::new();
    for &source in sources {
        if source >= n {
            return Err(DistanceError::SourceOutOfBounds);
        }
        if distances[source] < 0 {
            distances[source] = 0;
            queue.push_back(source);
        }
    }

    while let Some(u) = queue.pop_front() {
        let next_distance = distances[u] + 1;
        if let Some(row) = adjacency.outer_view(u) {
            for &v in row.indices() {
                if distances[v] < 0 {
                    distances[v] = next_distance;
                    queue.push_back(v);
                }
            }
        }
    }

    Ok(distances)
}

/// Computes distances with optional bipartite source splitting.
///
/// # Arguments
/// - `source`: Unified source indices for square graphs.
/// - `source_row`: Row-node sources for bipartite inputs.
/// - `source_col`: Column-node sources for bipartite inputs.
/// - `transpose`: Whether to transpose the input before computing distances.
/// - `force_bipartite`: Whether to treat the input as bipartite.
///
/// # Errors
/// Returns [`DistanceError::MissingSource`] when no sources are provided,
/// [`DistanceError::ConflictingSources`] when both unified and row sources
/// are supplied, and the same errors as [`get_distances_multi`] for invalid
/// source indices.
pub fn get_distances_full(
    input_matrix: &CsMat<f64>,
    source: Option<&[usize]>,
    source_row: Option<&[usize]>,
    source_col: Option<&[usize]>,
    transpose: bool,
    force_bipartite: bool,
) -> Result<DistanceResult, DistanceError> {
    let matrix = if transpose {
        input_matrix.transpose_view().to_csr()
    } else {
        input_matrix.to_owned()
    };
    let use_bipartite = force_bipartite || source_row.is_some() || source_col.is_some();
    if use_bipartite {
        if source.is_some() && source_row.is_some() {
            return Err(DistanceError::ConflictingSources);
        }
        let (n_row, n_col) = matrix.shape();
        let adjacency = bipartite2undirected(&matrix);
        let mut sources = Vec::new();
        if let Some(s) = source {
            sources.extend_from_slice(s);
        }
        if let Some(sr) = source_row {
            sources.extend_from_slice(sr);
        }
        if let Some(sc) = source_col {
            sources.extend(sc.iter().map(|c| n_row + c));
        }
        if sources.is_empty() {
            return Err(DistanceError::MissingSource);
        }
        let distances = get_distances_multi(&adjacency, &sources)?;
        Ok(DistanceResult::Bipartite(
            distances[..n_row].to_vec(),
            distances[n_row..n_row + n_col].to_vec(),
        ))
    } else {
        let src = source.ok_or(DistanceError::MissingSource)?;
        let distances = get_distances_multi(&matrix, src)?;
        Ok(DistanceResult::Single(distances))
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
    fn test_get_distances() {
        let adjacency = dense_to_csr(&array![[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]);
        let distances = get_distances(&adjacency, Some(0)).unwrap();
        assert_eq!(distances, vec![0, 1, 2]);
    }

    #[test]
    fn test_get_distances_disconnected() {
        let adjacency = dense_to_csr(&array![[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]]);
        let distances = get_distances(&adjacency, Some(1)).unwrap();
        assert_eq!(distances, vec![-1, 0, 1]);
    }

    #[test]
    fn test_get_distances_multi_sources() {
        let adjacency = dense_to_csr(&array![[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]);
        let distances = get_distances_multi(&adjacency, &[0, 2]).unwrap();
        assert_eq!(distances, vec![0, 1, 0]);
    }

    #[test]
    fn test_get_distances_full() {
        let adjacency = dense_to_csr(&array![[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]);
        let distances =
            get_distances_full(&adjacency, Some(&[0]), None, None, false, false).unwrap();
        match distances {
            DistanceResult::Single(d) => assert_eq!(d, vec![0, 1, 2]),
            _ => panic!("unexpected distance result"),
        }

        let biadjacency = dense_to_csr(&array![[1., 0., 1.], [0., 1., 0.]]);
        let distances =
            get_distances_full(&biadjacency, Some(&[0]), None, None, false, true).unwrap();
        match distances {
            DistanceResult::Bipartite(row, col) => {
                assert_eq!(row.len(), 2);
                assert_eq!(col.len(), 3);
                assert_eq!(row[0], 0);
            }
            _ => panic!("unexpected distance result"),
        }
    }

    #[test]
    fn test_get_distances_full_transpose() {
        let digraph = dense_to_csr(&array![
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [0., 0., 0., 0.]
        ]);
        let forward = get_distances_full(&digraph, Some(&[0]), None, None, false, false).unwrap();
        let backward = get_distances_full(&digraph, Some(&[0]), None, None, true, false).unwrap();
        match (forward, backward) {
            (DistanceResult::Single(d1), DistanceResult::Single(d2)) => {
                assert_eq!(d1, vec![0, 1, 2, 3]);
                assert_eq!(d2, vec![0, -1, -1, -1]);
            }
            _ => panic!("unexpected distance result"),
        }
    }

    #[test]
    fn test_get_distances_full_errors() {
        let biadjacency = dense_to_csr(&array![[1., 0., 1.], [0., 1., 0.]]);
        let err = get_distances_full(&biadjacency, Some(&[0]), Some(&[1]), None, false, true);
        assert_eq!(err, Err(DistanceError::ConflictingSources));

        let err = get_distances_full(&biadjacency, None, None, None, false, true);
        assert_eq!(err, Err(DistanceError::MissingSource));

        let err = get_distances_full(&biadjacency, None, None, Some(&[10]), false, true);
        assert_eq!(err, Err(DistanceError::SourceOutOfBounds));
    }
}
