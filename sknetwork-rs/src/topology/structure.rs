use std::collections::VecDeque;

use sprs::{CsMat, TriMat};

use crate::utils::check::is_symmetric;
use crate::utils::format::{MatrixInput, get_adjacency};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by structure error operations.
pub enum StructureError {
    /// Indicates empty graph.
    EmptyGraph,
    /// Indicates invalid connection.
    InvalidConnection,
    /// Indicates not undirected.
    NotUndirected,
}

#[derive(Debug, Clone, PartialEq)]
/// BipartiteResult enum.
pub enum BipartiteResult {
    /// Indicates bool.
    Bool(bool),
    /// Returns bipartite status plus optional biadjacency and index partitions.
    Full(
        bool,
        Option<CsMat<f64>>,
        Option<Vec<usize>>,
        Option<Vec<usize>>,
    ),
}

fn weak_components(adjacency: &CsMat<f64>) -> Vec<usize> {
    let n = adjacency.rows();
    let undirected = adjacency + &adjacency.transpose_view().to_owned();
    let mut labels = vec![usize::MAX; n];
    let mut cc = 0usize;
    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }
        let mut q = VecDeque::new();
        q.push_back(start);
        labels[start] = cc;
        while let Some(u) = q.pop_front() {
            if let Some(row) = undirected.outer_view(u) {
                for &v in row.indices() {
                    if labels[v] == usize::MAX {
                        labels[v] = cc;
                        q.push_back(v);
                    }
                }
            }
        }
        cc += 1;
    }
    labels
}

fn dfs_order(graph: &CsMat<f64>, start: usize, seen: &mut [bool], order: &mut Vec<usize>) {
    let mut stack = vec![(start, false)];
    while let Some((u, processed)) = stack.pop() {
        if processed {
            order.push(u);
            continue;
        }
        if seen[u] {
            continue;
        }
        seen[u] = true;
        stack.push((u, true));
        if let Some(row) = graph.outer_view(u) {
            for &v in row.indices() {
                if !seen[v] {
                    stack.push((v, false));
                }
            }
        }
    }
}

fn strong_components(adjacency: &CsMat<f64>) -> Vec<usize> {
    let n = adjacency.rows();
    let mut seen = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for u in 0..n {
        if !seen[u] {
            dfs_order(adjacency, u, &mut seen, &mut order);
        }
    }
    let rev = adjacency.transpose_view().to_csr();
    let mut labels = vec![usize::MAX; n];
    let mut cc = 0usize;
    while let Some(start) = order.pop() {
        if labels[start] != usize::MAX {
            continue;
        }
        let mut q = VecDeque::new();
        q.push_back(start);
        labels[start] = cc;
        while let Some(u) = q.pop_front() {
            if let Some(row) = rev.outer_view(u) {
                for &v in row.indices() {
                    if labels[v] == usize::MAX {
                        labels[v] = cc;
                        q.push_back(v);
                    }
                }
            }
        }
        cc += 1;
    }
    labels
}

/// Returns connected components.
pub fn get_connected_components(
    input_matrix: &CsMat<f64>,
    connection: &str,
    force_bipartite: bool,
) -> Result<Vec<usize>, StructureError> {
    if input_matrix.rows() == 0 || input_matrix.cols() == 0 {
        return Ok(Vec::new());
    }
    let (adjacency, _) = get_adjacency(
        MatrixInput::Sparse(input_matrix.to_owned()),
        true,
        force_bipartite,
        false,
        false,
    )
    .map_err(|_| StructureError::EmptyGraph)?;
    match connection {
        "weak" => Ok(weak_components(&adjacency)),
        "strong" => Ok(strong_components(&adjacency)),
        _ => Err(StructureError::InvalidConnection),
    }
}

/// Returns whether the input satisfies `is connected`.
pub fn is_connected(
    input_matrix: &CsMat<f64>,
    connection: &str,
    force_bipartite: bool,
) -> Result<bool, StructureError> {
    let labels = get_connected_components(input_matrix, connection, force_bipartite)?;
    if labels.is_empty() {
        return Ok(true);
    }
    Ok(labels.iter().all(|&x| x == labels[0]))
}

fn submatrix_rows_cols(input: &CsMat<f64>, rows: &[usize], cols: &[usize]) -> CsMat<f64> {
    let mut col_map = vec![usize::MAX; input.cols()];
    for (j_new, &j_old) in cols.iter().enumerate() {
        col_map[j_old] = j_new;
    }
    let mut tri = TriMat::<f64>::new((rows.len(), cols.len()));
    for (i_new, &i_old) in rows.iter().enumerate() {
        if let Some(row) = input.outer_view(i_old) {
            for (j_old, v) in row.iter() {
                let mapped = col_map[j_old];
                if mapped != usize::MAX {
                    tri.add_triplet(i_new, mapped, *v);
                }
            }
        }
    }
    tri.to_csr::<usize>()
}

/// Returns largest connected component.
pub fn get_largest_connected_component(
    input_matrix: &CsMat<f64>,
    connection: &str,
    force_bipartite: bool,
    return_index: bool,
) -> Result<(CsMat<f64>, Option<Vec<usize>>), StructureError> {
    let (adjacency, bipartite) = get_adjacency(
        MatrixInput::Sparse(input_matrix.to_owned()),
        true,
        force_bipartite,
        false,
        false,
    )
    .map_err(|_| StructureError::EmptyGraph)?;
    let labels = get_connected_components(&adjacency, connection, false)?;
    let n_labels = labels.iter().copied().max().unwrap_or(0) + 1;
    let mut counts = vec![0usize; n_labels];
    for &label in &labels {
        counts[label] += 1;
    }
    let largest_label = counts
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| *c)
        .map(|(i, _)| i)
        .unwrap_or(0);

    if bipartite {
        let n_row = input_matrix.rows();
        let n_col = input_matrix.cols();
        let index_row: Vec<usize> = (0..n_row).filter(|&i| labels[i] == largest_label).collect();
        let index_col: Vec<usize> = (0..n_col)
            .filter(|&j| labels[n_row + j] == largest_label)
            .collect();
        let output = submatrix_rows_cols(input_matrix, &index_row, &index_col);
        let mut index = index_row.clone();
        index.extend(index_col.iter().copied());
        Ok((output, if return_index { Some(index) } else { None }))
    } else {
        let index: Vec<usize> = (0..input_matrix.rows())
            .filter(|&i| labels[i] == largest_label)
            .collect();
        let output = submatrix_rows_cols(input_matrix, &index, &index);
        Ok((output, if return_index { Some(index) } else { None }))
    }
}

/// Returns whether the input satisfies `is bipartite`.
pub fn is_bipartite(
    adjacency: &CsMat<f64>,
    return_biadjacency: bool,
) -> Result<BipartiteResult, StructureError> {
    if !is_symmetric(adjacency) {
        return Err(StructureError::NotUndirected);
    }
    let n = adjacency.rows();
    for i in 0..n.min(adjacency.cols()) {
        if adjacency.get(i, i).is_some() {
            return Ok(if return_biadjacency {
                BipartiteResult::Full(false, None, None, None)
            } else {
                BipartiteResult::Bool(false)
            });
        }
    }
    let mut coloring = vec![-1i8; n];
    for src in 0..n {
        if coloring[src] != -1 {
            continue;
        }
        let mut stack = vec![src];
        coloring[src] = 0;
        while let Some(node) = stack.pop() {
            if let Some(row) = adjacency.outer_view(node) {
                for &neighbor in row.indices() {
                    if coloring[neighbor] == -1 {
                        coloring[neighbor] = 1 - coloring[node];
                        stack.push(neighbor);
                    } else if coloring[neighbor] == coloring[node] {
                        return Ok(if return_biadjacency {
                            BipartiteResult::Full(false, None, None, None)
                        } else {
                            BipartiteResult::Bool(false)
                        });
                    }
                }
            }
        }
    }

    if return_biadjacency {
        let rows: Vec<usize> = (0..n).filter(|&i| coloring[i] == 0).collect();
        let cols: Vec<usize> = (0..n).filter(|&i| coloring[i] == 1).collect();
        let biadj = submatrix_rows_cols(adjacency, &rows, &cols);
        Ok(BipartiteResult::Full(
            true,
            Some(biadj),
            Some(rows),
            Some(cols),
        ))
    } else {
        Ok(BipartiteResult::Bool(true))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::utils::format::{bipartite2undirected, directed2undirected};

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
    fn test_components_and_connected() {
        let digraph = dense_to_csr(&array![[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]);
        let labels = get_connected_components(&digraph, "weak", false).unwrap();
        assert_eq!(labels.len(), 3);
        assert!(is_connected(&digraph, "weak", false).unwrap());
        assert!(!is_connected(&digraph, "strong", false).unwrap());
    }

    #[test]
    fn test_largest_component() {
        let adjacency = dense_to_csr(&array![
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 1., 0.]
        ]);
        let (largest, index) =
            get_largest_connected_component(&adjacency, "weak", false, true).unwrap();
        assert_eq!(largest.shape(), (2, 2));
        assert_eq!(index.unwrap().len(), 2);
    }

    #[test]
    fn test_is_bipartite() {
        let biadjacency = dense_to_csr(&array![[1., 0., 1.], [0., 1., 0.]]);
        let adjacency = bipartite2undirected(&biadjacency);
        let result = is_bipartite(&adjacency, false).unwrap();
        assert_eq!(result, BipartiteResult::Bool(true));

        let cyclic3 = directed2undirected(
            &dense_to_csr(&array![[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]),
            true,
        );
        let result = is_bipartite(&cyclic3, false).unwrap();
        assert_eq!(result, BipartiteResult::Bool(false));
    }
}
