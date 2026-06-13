use std::collections::{BTreeMap, HashMap};

use sprs::CsMat;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by wlerror operations.
pub enum WLError {
    /// Indicates not square.
    NotSquare,
}

fn wl_step(adjacency: &CsMat<f64>, labels: &[i32]) -> (Vec<i32>, bool) {
    let n = adjacency.rows();
    let mut signatures: Vec<(i32, Vec<i32>)> = Vec::with_capacity(n);
    for u in 0..n {
        let mut neigh_labels = adjacency
            .outer_view(u)
            .map(|row| row.indices().iter().map(|&v| labels[v]).collect::<Vec<_>>())
            .unwrap_or_default();
        neigh_labels.sort_unstable();
        signatures.push((labels[u], neigh_labels));
    }

    let mut map: BTreeMap<(i32, Vec<i32>), i32> = BTreeMap::new();
    let mut next_id = 0i32;
    let mut new_labels = vec![0i32; n];
    for (u, sig) in signatures.into_iter().enumerate() {
        let id = if let Some(v) = map.get(&sig) {
            *v
        } else {
            let v = next_id;
            map.insert(sig, v);
            next_id += 1;
            v
        };
        new_labels[u] = id;
    }
    let changed = new_labels != labels;
    (new_labels, changed)
}

/// Computes color weisfeiler lehman.
///
/// # Errors
///
/// Returns [`WLError`] on failure.
pub fn color_weisfeiler_lehman(adjacency: &CsMat<f64>, max_iter: i32) -> Result<Vec<i32>, WLError> {
    if adjacency.rows() != adjacency.cols() {
        return Err(WLError::NotSquare);
    }
    let n = adjacency.rows();
    let mut labels = vec![0i32; n];
    let iters = if max_iter < 0 || max_iter as usize > n {
        n
    } else {
        max_iter as usize
    };
    for _ in 0..iters {
        let (new_labels, changed) = wl_step(adjacency, &labels);
        labels = new_labels;
        if !changed {
            break;
        }
    }
    Ok(labels)
}

fn counts(labels: &[i32]) -> Vec<usize> {
    let mut h = HashMap::<i32, usize>::new();
    for &x in labels {
        *h.entry(x).or_insert(0) += 1;
    }
    let mut c: Vec<usize> = h.into_values().collect();
    c.sort_unstable();
    c
}

/// Computes are isomorphic.
pub fn are_isomorphic(
    adjacency1: &CsMat<f64>,
    adjacency2: &CsMat<f64>,
    max_iter: i32,
) -> Result<bool, WLError> {
    if adjacency1.rows() != adjacency1.cols() || adjacency2.rows() != adjacency2.cols() {
        return Err(WLError::NotSquare);
    }
    if adjacency1.shape() != adjacency2.shape() || adjacency1.nnz() != adjacency2.nnz() {
        return Ok(false);
    }
    let n = adjacency1.rows();
    let iters = if max_iter < 0 || max_iter as usize > n {
        n
    } else {
        max_iter as usize
    };
    let mut labels1 = vec![0i32; n];
    let mut labels2 = vec![0i32; n];
    let mut changed1 = true;
    let mut changed2 = true;
    let mut iteration = 0usize;
    while iteration < iters && (changed1 || changed2) {
        let (new1, ch1) = wl_step(adjacency1, &labels1);
        let (new2, ch2) = wl_step(adjacency2, &labels2);
        labels1 = new1;
        labels2 = new2;
        changed1 = ch1;
        changed2 = ch2;
        if counts(&labels1) != counts(&labels2) {
            return Ok(false);
        }
        iteration += 1;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;
    use std::collections::HashMap;

    fn dense_to_csr(dense: &[&[f64]]) -> CsMat<f64> {
        let r = dense.len();
        let c = dense.first().map(|x| x.len()).unwrap_or(0);
        let mut tri = TriMat::<f64>::new((r, c));
        for i in 0..r {
            for j in 0..c {
                if dense[i][j] != 0.0 {
                    tri.add_triplet(i, j, dense[i][j]);
                }
            }
        }
        tri.to_csr::<usize>()
    }

    fn permute_graph(adjacency: &CsMat<f64>, perm: &[usize]) -> CsMat<f64> {
        let n = adjacency.rows();
        let mut inv = vec![0usize; n];
        for (new_i, &old_i) in perm.iter().enumerate() {
            inv[old_i] = new_i;
        }
        let mut tri = TriMat::<f64>::new((n, n));
        for old_i in 0..n {
            if let Some(row) = adjacency.outer_view(old_i) {
                for (old_j, v) in row.iter() {
                    let new_i = inv[old_i];
                    let new_j = inv[old_j];
                    tri.add_triplet(new_i, new_j, *v);
                }
            }
        }
        tri.to_csr::<usize>()
    }

    #[test]
    fn test_empty_and_clique() {
        let empty = CsMat::<f64>::zero((10, 10));
        let labels = color_weisfeiler_lehman(&empty, -1).unwrap();
        assert_eq!(labels, vec![0; 10]);

        let mut tri = TriMat::<f64>::new((6, 6));
        for i in 0..6 {
            for j in 0..6 {
                if i != j {
                    tri.add_triplet(i, j, 1.0);
                }
            }
        }
        let clique = tri.to_csr::<usize>();
        let labels = color_weisfeiler_lehman(&clique, -1).unwrap();
        assert_eq!(labels, vec![0; 6]);
    }

    #[test]
    fn test_house_and_bow_tie() {
        let house = dense_to_csr(&[
            &[0., 1., 0., 1., 0.],
            &[1., 0., 1., 0., 1.],
            &[0., 1., 0., 1., 0.],
            &[1., 0., 1., 0., 1.],
            &[0., 1., 0., 1., 0.],
        ]);
        let labels = color_weisfeiler_lehman(&house, -1).unwrap();
        assert_eq!(labels, vec![0, 1, 0, 1, 0]);

        let bow_tie = dense_to_csr(&[
            &[0., 1., 1., 0., 0.],
            &[1., 0., 1., 0., 0.],
            &[1., 1., 0., 1., 1.],
            &[0., 0., 1., 0., 1.],
            &[0., 0., 1., 1., 0.],
        ]);
        let labels = color_weisfeiler_lehman(&bow_tie, -1).unwrap();
        let mut hist = HashMap::<i32, usize>::new();
        for label in labels {
            *hist.entry(label).or_insert(0) += 1;
        }
        let mut counts: Vec<usize> = hist.into_values().collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![1, 4]);
    }

    #[test]
    fn test_isomorphism() {
        let ref_graph = dense_to_csr(&[
            &[0., 1., 0., 1., 0.],
            &[1., 0., 1., 0., 1.],
            &[0., 1., 0., 1., 0.],
            &[1., 0., 1., 0., 1.],
            &[0., 1., 0., 1., 0.],
        ]);
        let permuted = permute_graph(&ref_graph, &[2, 4, 0, 3, 1]);
        assert!(are_isomorphic(&ref_graph, &permuted, -1).unwrap());

        let line = dense_to_csr(&[
            &[0., 1., 0., 0., 0.],
            &[1., 0., 1., 0., 0.],
            &[0., 1., 0., 1., 0.],
            &[0., 0., 1., 0., 1.],
            &[0., 0., 0., 1., 0.],
        ]);
        assert!(!are_isomorphic(&ref_graph, &line, -1).unwrap());
    }

    #[test]
    fn test_early_stop() {
        let house = dense_to_csr(&[
            &[0., 1., 0., 1., 0.],
            &[1., 0., 1., 0., 1.],
            &[0., 1., 0., 1., 0.],
            &[1., 0., 1., 0., 1.],
            &[0., 1., 0., 1., 0.],
        ]);
        let labels = color_weisfeiler_lehman(&house, 1).unwrap();
        let mut sorted = labels.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 0, 0, 1, 1]);
    }
}
