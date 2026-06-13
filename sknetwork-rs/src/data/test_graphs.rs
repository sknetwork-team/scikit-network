use sprs::{CsMat, TriMat};

use crate::utils::format::directed2undirected;

/// Returns a 10-node weighted directed test graph.
pub fn test_digraph() -> CsMat<f64> {
    let row = [0usize, 1, 1, 3, 4, 6, 6, 6, 6, 7, 8, 8, 9];
    let col = [1usize, 4, 3, 2, 5, 4, 5, 7, 1, 9, 9, 2, 9];
    let data = [
        1.0f64, 1.0, 2.5, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.5, 2.0, 1.0, 2.0,
    ];
    let mut tri = TriMat::<f64>::new((10, 10));
    for i in 0..row.len() {
        tri.add_triplet(row[i], col[i], data[i]);
    }
    tri.to_csr::<usize>()
}

/// Returns the symmetrized version of [`test_digraph`].
pub fn test_graph() -> CsMat<f64> {
    directed2undirected(&test_digraph(), true)
}

/// Returns a 6×8 weighted bipartite test graph.
pub fn test_bigraph() -> CsMat<f64> {
    let row = [0usize, 1, 1, 2, 2, 3, 4, 5, 5];
    let col = [1usize, 2, 3, 1, 0, 4, 7, 5, 6];
    let data = [1.0f64, 2.5, 1.0, 2.0, 2.0, 1.5, 1.0, 2.0, 3.0];
    let mut tri = TriMat::<f64>::new((6, 8));
    for i in 0..row.len() {
        tri.add_triplet(row[i], col[i], data[i]);
    }
    tri.to_csr::<usize>()
}

/// Returns a 10-node undirected graph with an isolated component.
pub fn test_disconnected_graph() -> CsMat<f64> {
    let row = [1usize, 2, 3, 4, 6, 6, 6, 7, 8, 9];
    let col = [1usize, 3, 2, 5, 4, 5, 7, 9, 9, 9];
    let data = [1.0f64, 2.5, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.5, 2.0];
    let mut tri = TriMat::<f64>::new((10, 10));
    for i in 0..row.len() {
        tri.add_triplet(row[i], col[i], data[i]);
    }
    directed2undirected(&tri.to_csr::<usize>(), true)
}

/// Returns a bipartite graph with a disconnected row block.
pub fn test_bigraph_disconnect() -> CsMat<f64> {
    let row = [1usize, 1, 1, 2, 2, 3, 5, 4, 5];
    let col = [1usize, 2, 3, 1, 3, 4, 7, 7, 6];
    let data = [1.0f64, 2.5, 1.0, 2.0, 2.0, 1.5, 3.0, 0.0, 1.0];
    let mut tri = TriMat::<f64>::new((6, 8));
    for i in 0..row.len() {
        tri.add_triplet(row[i], col[i], data[i]);
    }
    tri.to_csr::<usize>()
}

/// Returns [`test_graph`] with all nonzero weights set to `1.0`.
pub fn test_graph_bool() -> CsMat<f64> {
    let mut g = test_graph();
    for x in g.data_mut() {
        *x = if *x != 0.0 { 1.0 } else { 0.0 };
    }
    g
}

/// Returns a 10-node unweighted clique adjacency matrix.
pub fn test_clique() -> CsMat<f64> {
    let n = 10usize;
    let mut tri = TriMat::<f64>::new((n, n));
    for i in 0..n {
        for j in 0..n {
            if i != j {
                tri.add_triplet(i, j, 1.0);
            }
        }
    }
    tri.to_csr::<usize>()
}

/// Returns a 10×10 zero adjacency matrix.
pub fn test_graph_empty() -> CsMat<f64> {
    CsMat::<f64>::zero((10, 10))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shapes() {
        assert_eq!(test_digraph().shape(), (10, 10));
        assert_eq!(test_graph().shape(), (10, 10));
        assert_eq!(test_bigraph().shape(), (6, 8));
        assert_eq!(test_disconnected_graph().shape(), (10, 10));
        assert_eq!(test_bigraph_disconnect().shape(), (6, 8));
        assert_eq!(test_clique().shape(), (10, 10));
        assert_eq!(test_graph_empty().shape(), (10, 10));
    }
}
