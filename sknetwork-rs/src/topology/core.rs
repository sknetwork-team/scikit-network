use sprs::CsMat;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by core error operations.
pub enum CoreError {
    /// Indicates invalid node count.
    InvalidNodeCount,
}

/// Returns core decomposition.
///
/// # Errors
///
/// Returns [`CoreError`] on failure.
pub fn get_core_decomposition(adjacency: &CsMat<f64>) -> Result<Vec<usize>, CoreError> {
    let n = adjacency.rows();
    if adjacency.cols() != n {
        return Err(CoreError::InvalidNodeCount);
    }

    let mut degrees = vec![0usize; n];
    for (i, row) in adjacency.outer_iterator().enumerate() {
        degrees[i] = row.nnz();
    }

    let max_degree = degrees.iter().copied().max().unwrap_or(0);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); max_degree + 1];
    for v in 0..n {
        buckets[degrees[v]].push(v);
    }

    let mut removed = vec![false; n];
    let mut labels = vec![0usize; n];
    let mut core_value = 0usize;
    let mut removed_count = 0usize;
    let mut d_min = 0usize;
    while removed_count < n {
        while d_min <= max_degree && buckets[d_min].is_empty() {
            d_min += 1;
        }
        if d_min > max_degree {
            break;
        }
        let u = buckets[d_min].pop().unwrap_or(0);
        if removed[u] || degrees[u] != d_min {
            continue;
        }
        removed[u] = true;
        removed_count += 1;
        core_value = core_value.max(d_min);
        labels[u] = core_value;

        if let Some(row) = adjacency.outer_view(u) {
            for &v in row.indices() {
                if !removed[v] && degrees[v] > d_min {
                    let old = degrees[v];
                    degrees[v] -= 1;
                    buckets[old - 1].push(v);
                }
            }
        }
    }

    Ok(labels)
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
        let core = get_core_decomposition(&adjacency).unwrap();
        assert_eq!(*core.iter().max().unwrap_or(&0), 0);
    }

    #[test]
    fn test_clique() {
        let n = 6;
        let mut tri = TriMat::<f64>::new((n, n));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    tri.add_triplet(i, j, 1.0);
                }
            }
        }
        let adjacency = tri.to_csr::<usize>();
        let core = get_core_decomposition(&adjacency).unwrap();
        assert_eq!(*core.iter().max().unwrap_or(&0), n - 1);
    }

    #[test]
    fn test_star_graph() {
        let n = 6;
        let mut tri = TriMat::<f64>::new((n, n));
        for i in 1..n {
            tri.add_triplet(0, i, 1.0);
            tri.add_triplet(i, 0, 1.0);
        }
        let adjacency = tri.to_csr::<usize>();
        let core = get_core_decomposition(&adjacency).unwrap();
        assert!(core.iter().all(|&k| k == 1));
    }

    #[test]
    fn test_path_graph() {
        let n = 7;
        let mut tri = TriMat::<f64>::new((n, n));
        for i in 0..(n - 1) {
            tri.add_triplet(i, i + 1, 1.0);
            tri.add_triplet(i + 1, i, 1.0);
        }
        let adjacency = tri.to_csr::<usize>();
        let core = get_core_decomposition(&adjacency).unwrap();
        assert!(core.iter().all(|&k| k == 1));
    }
}
