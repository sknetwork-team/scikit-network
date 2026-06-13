use sprs::{CsMat, TriMat};

use crate::path::distances::{DistanceError, get_distances_multi};
use crate::topology::structure::get_connected_components;
use crate::utils::check::is_symmetric;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by cycle error operations.
pub enum CycleError {
    /// Indicates missing root.
    MissingRoot,
    /// Indicates invalid root.
    InvalidRoot,
    /// Indicates not undirected.
    NotUndirected,
    /// Indicates distance.
    Distance(DistanceError),
}

fn infer_directed(adjacency: &CsMat<f64>, directed: Option<bool>) -> Result<bool, CycleError> {
    if directed == Some(false) && !is_symmetric(adjacency) {
        return Err(CycleError::NotUndirected);
    }
    Ok(directed.unwrap_or(!is_symmetric(adjacency)))
}

fn has_self_loop(adjacency: &CsMat<f64>) -> bool {
    let n = adjacency.rows().min(adjacency.cols());
    (0..n).any(|i| adjacency.get(i, i).is_some())
}

/// Returns whether the input satisfies `is acyclic`.
///
/// # Errors
///
/// Returns [`CycleError`] on failure.
pub fn is_acyclic(adjacency: &CsMat<f64>, directed: Option<bool>) -> Result<bool, CycleError> {
    let directed = infer_directed(adjacency, directed)?;
    if has_self_loop(adjacency) {
        return Ok(false);
    }
    let labels = if directed {
        get_connected_components(adjacency, "strong", false).map_err(|_| CycleError::InvalidRoot)?
    } else {
        get_connected_components(adjacency, "weak", false).map_err(|_| CycleError::InvalidRoot)?
    };
    let n_nodes = adjacency.rows();
    let n_cc = labels.iter().max().map(|x| *x + 1).unwrap_or(0);
    if directed {
        Ok(n_cc == n_nodes)
    } else {
        let n_edges = adjacency.nnz() / 2;
        Ok(n_cc == n_nodes.saturating_sub(n_edges))
    }
}

/// Returns cycles.
pub fn get_cycles(
    adjacency: &CsMat<f64>,
    directed: Option<bool>,
) -> Result<Vec<Vec<usize>>, CycleError> {
    let directed = infer_directed(adjacency, directed)?;
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    let n = adjacency.rows();

    for i in 0..n.min(adjacency.cols()) {
        if adjacency.get(i, i).is_some() {
            cycles.push(vec![i]);
        }
    }

    if is_acyclic(adjacency, Some(directed))? {
        return Ok(cycles);
    }

    for start in 0..n {
        let mut stack: Vec<(usize, Vec<usize>)> = vec![(start, vec![start])];
        while let Some((current, path)) = stack.pop() {
            if let Some(row) = adjacency.outer_view(current) {
                for &neighbor in row.indices() {
                    if !directed && path.len() > 1 && neighbor == path[path.len() - 2] {
                        continue;
                    }
                    if let Some(pos) = path.iter().position(|&x| x == neighbor) {
                        let cycle = path[pos..].to_vec();
                        if !cycle.is_empty() {
                            cycles.push(cycle);
                        }
                    } else if path.len() <= n {
                        let mut next = path.clone();
                        next.push(neighbor);
                        stack.push((neighbor, next));
                    }
                }
            }
        }
    }

    let mut unique: Vec<Vec<usize>> = Vec::new();
    let mut seen: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();
    for cycle in cycles {
        if cycle.is_empty() {
            continue;
        }
        let mut min_pos = 0usize;
        for i in 1..cycle.len() {
            if cycle[i] < cycle[min_pos] {
                min_pos = i;
            }
        }
        let mut normalized = Vec::with_capacity(cycle.len());
        for k in 0..cycle.len() {
            normalized.push(cycle[(min_pos + k) % cycle.len()]);
        }
        let key = if directed {
            normalized.clone()
        } else {
            let mut s = normalized.clone();
            s.sort_unstable();
            s
        };
        if seen.insert(key) {
            unique.push(normalized);
        }
    }
    Ok(unique)
}

/// Computes break cycles.
pub fn break_cycles(
    adjacency: &CsMat<f64>,
    root: Option<&[usize]>,
    directed: Option<bool>,
) -> Result<CsMat<f64>, CycleError> {
    if is_acyclic(adjacency, directed)? {
        return Ok(adjacency.to_owned());
    }
    let roots = root.ok_or(CycleError::MissingRoot)?;
    if roots.is_empty() {
        return Err(CycleError::MissingRoot);
    }
    let directed = infer_directed(adjacency, directed)?;
    let n = adjacency.rows();
    if roots.iter().any(|&r| r >= n) {
        return Err(CycleError::InvalidRoot);
    }
    if roots
        .iter()
        .any(|&r| adjacency.outer_view(r).map(|row| row.nnz()).unwrap_or(0) == 0)
    {
        return Err(CycleError::InvalidRoot);
    }
    let mut to_remove = std::collections::HashSet::<(usize, usize)>::new();
    for i in 0..n.min(adjacency.cols()) {
        if adjacency.get(i, i).is_some() {
            to_remove.insert((i, i));
        }
    }
    if directed {
        let distances = get_distances_multi(adjacency, roots).map_err(CycleError::Distance)?;
        let mut cycles = get_cycles(adjacency, Some(true))?;
        cycles.sort_by_key(|c| {
            c.iter()
                .map(|&u| distances.get(u).copied().unwrap_or(i32::MAX))
                .min()
                .unwrap_or(i32::MAX)
        });
        for c in cycles {
            if c.len() >= 2 {
                let u = c[c.len() - 1];
                let v = c[0];
                to_remove.insert((u, v));
            }
        }
    } else {
        let cycles = get_cycles(adjacency, Some(false))?;
        for c in cycles {
            if c.len() >= 2 {
                let u = c[c.len() - 1];
                let v = c[0];
                to_remove.insert((u, v));
                to_remove.insert((v, u));
            }
        }
    }
    let (r, c) = adjacency.shape();
    let mut tri = TriMat::<f64>::new((r, c));
    for (i, row) in adjacency.outer_iterator().enumerate() {
        for (j, v) in row.iter() {
            if !to_remove.contains(&(i, j)) {
                tri.add_triplet(i, j, *v);
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
    fn test_is_acyclic() {
        let loops = dense_to_csr(&array![[1., 0.], [0., 1.]]);
        assert!(!is_acyclic(&loops, None).unwrap());
        let cycle = dense_to_csr(&array![[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]);
        assert!(!is_acyclic(&cycle, None).unwrap());
        let line = dense_to_csr(&array![[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]);
        assert!(is_acyclic(&line, None).unwrap());
    }

    #[test]
    fn test_get_cycles() {
        let cycle = dense_to_csr(&array![
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.]
        ]);
        let cycles = get_cycles(&cycle, Some(true)).unwrap();
        assert!(!cycles.is_empty());
        assert_eq!(cycles[0].len(), 4);
    }

    #[test]
    fn test_break_cycles() {
        let cycle = dense_to_csr(&array![
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.]
        ]);
        let dag = break_cycles(&cycle, Some(&[0]), Some(true)).unwrap();
        assert!(is_acyclic(&dag, Some(true)).unwrap());

        let undirected_cycle = dense_to_csr(&array![
            [0., 1., 0., 1.],
            [1., 0., 1., 0.],
            [0., 1., 0., 1.],
            [1., 0., 1., 0.]
        ]);
        let tree = break_cycles(&undirected_cycle, Some(&[0]), Some(false)).unwrap();
        assert!(is_acyclic(&tree, Some(false)).unwrap());
    }

    #[test]
    fn test_break_cycles_invalid_root_requires_outgoing_edge() {
        let adjacency = dense_to_csr(&array![
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.]
        ]);
        assert!(matches!(
            break_cycles(&adjacency, Some(&[5]), Some(true)),
            Err(CycleError::InvalidRoot)
        ));

        let with_isolated_root = dense_to_csr(&array![
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.]
        ]);
        assert!(matches!(
            break_cycles(&with_isolated_root, Some(&[3]), Some(true)),
            Err(CycleError::InvalidRoot)
        ));
    }
}
