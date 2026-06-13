use ndarray::Array2;
use sprs::{CsMat, TriMat};

use crate::utils::check::{CheckError, is_square, is_symmetric};
use crate::utils::values::{ValuesInput, get_values, stack_values};

#[derive(Debug, Clone)]
/// MatrixInput enum.
pub enum MatrixInput {
    /// Indicates dense.
    Dense(Array2<f64>),
    /// Indicates sparse.
    Sparse(CsMat<f64>),
}

fn dense_to_csr(dense: &Array2<f64>) -> CsMat<f64> {
    let (r, c) = dense.dim();
    let mut tri = TriMat::<f64>::new((r, c));
    for i in 0..r {
        for j in 0..c {
            let v = dense[[i, j]];
            if v != 0.0 {
                tri.add_triplet(i, j, v);
            }
        }
    }
    tri.to_csr()
}

/// Validates format.
pub fn check_format(
    input_matrix: MatrixInput,
    allow_empty: bool,
) -> Result<CsMat<f64>, CheckError> {
    let out = match input_matrix {
        MatrixInput::Dense(dense) => dense_to_csr(&dense),
        MatrixInput::Sparse(sparse) => sparse,
    };
    if !allow_empty && out.nnz() == 0 {
        return Err(CheckError::InvalidMinNnz);
    }
    Ok(out)
}

/// Computes directed2undirected.
pub fn directed2undirected(adjacency: &CsMat<f64>, weighted: bool) -> CsMat<f64> {
    if weighted {
        adjacency + &adjacency.transpose_view().to_owned()
    } else {
        let sum = adjacency + &adjacency.transpose_view().to_owned();
        let (r, c) = sum.shape();
        let mut tri = TriMat::<f64>::new((r, c));
        for (i, row) in sum.outer_iterator().enumerate() {
            for (j, v) in row.iter() {
                if *v != 0.0 {
                    tri.add_triplet(i, j, 1.0);
                }
            }
        }
        tri.to_csr()
    }
}

/// Computes bipartite2directed.
pub fn bipartite2directed(biadjacency: &CsMat<f64>) -> CsMat<f64> {
    let (n_row, n_col) = biadjacency.shape();
    let n = n_row + n_col;
    let mut tri = TriMat::<f64>::new((n, n));
    for (i, row) in biadjacency.outer_iterator().enumerate() {
        for (j, v) in row.iter() {
            tri.add_triplet(i, n_row + j, *v);
        }
    }
    tri.to_csr()
}

/// Computes bipartite2undirected.
pub fn bipartite2undirected(biadjacency: &CsMat<f64>) -> CsMat<f64> {
    let (n_row, n_col) = biadjacency.shape();
    let n = n_row + n_col;
    let mut tri = TriMat::<f64>::new((n, n));
    for (i, row) in biadjacency.outer_iterator().enumerate() {
        for (j, v) in row.iter() {
            tri.add_triplet(i, n_row + j, *v);
            tri.add_triplet(n_row + j, i, *v);
        }
    }
    tri.to_csr()
}

/// Returns adjacency.
pub fn get_adjacency(
    input_matrix: MatrixInput,
    allow_directed: bool,
    force_bipartite: bool,
    force_directed: bool,
    allow_empty: bool,
) -> Result<(CsMat<f64>, bool), CheckError> {
    let input_matrix = check_format(input_matrix, allow_empty)?;
    let mut bipartite = false;
    if force_bipartite
        || !is_square(input_matrix.shape())
        || !(allow_directed || is_symmetric(&input_matrix))
    {
        bipartite = true;
    }
    let adjacency = if bipartite {
        if force_directed {
            bipartite2directed(&input_matrix)
        } else {
            bipartite2undirected(&input_matrix)
        }
    } else {
        input_matrix
    };
    Ok((adjacency, bipartite))
}

/// Returns adjacency values.
pub fn get_adjacency_values(
    input_matrix: MatrixInput,
    allow_directed: bool,
    force_bipartite: bool,
    force_directed: bool,
    values: Option<ValuesInput>,
    values_row: Option<ValuesInput>,
    values_col: Option<ValuesInput>,
    default_value: f64,
    which: Option<&str>,
) -> Result<(CsMat<f64>, Vec<f64>, bool), CheckError> {
    let input_csr = check_format(input_matrix, false)?;
    let shape = input_csr.shape();
    let force_bip = force_bipartite || values_row.is_some() || values_col.is_some();
    let (adjacency, bipartite) = get_adjacency(
        MatrixInput::Sparse(input_csr.clone()),
        allow_directed,
        force_bip,
        force_directed,
        false,
    )?;

    let mut out_values = if bipartite {
        if let Some(v) = values {
            stack_values(shape, Some(v), None, default_value)
                .map_err(|_| CheckError::InvalidWeights)?
        } else {
            stack_values(shape, values_row, values_col, default_value)
                .map_err(|_| CheckError::InvalidWeights)?
        }
    } else {
        get_values(
            &[shape.0],
            values.unwrap_or(ValuesInput::None),
            default_value,
        )
        .map_err(|_| CheckError::InvalidWeights)?
    };

    if let Some(mode) = which {
        if mode == "probs" {
            let s: f64 = out_values.iter().sum();
            if s > 0.0 {
                out_values.iter_mut().for_each(|x| *x /= s);
            }
        }
    }
    Ok((adjacency, out_values, bipartite))
}

#[cfg(test)]
mod tests {
    use crate::utils::check::check_square;
    use ndarray::array;

    use super::*;

    fn is_sym(m: &CsMat<f64>) -> bool {
        let t = m.transpose_view().to_owned();
        (m - &t).nnz() == 0
    }

    #[test]
    fn test_directed2undirected() {
        let dense = array![[0.0, 1.0], [0.0, 0.0]];
        let adj = dense_to_csr(&dense);
        let undirected = directed2undirected(&adj, true);
        assert_eq!(undirected.shape(), (2, 2));
        assert!(is_sym(&undirected));
    }

    #[test]
    fn test_bipartite_conversions() {
        let bi = dense_to_csr(&array![[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]);
        let d = bipartite2directed(&bi);
        let u = bipartite2undirected(&bi);
        assert_eq!(d.shape(), (5, 5));
        assert_eq!(u.shape(), (5, 5));
        assert!(is_sym(&u));
    }

    #[test]
    fn test_check_format_and_get_adjacency() {
        let bad = MatrixInput::Sparse(CsMat::zero((3, 4)));
        assert!(check_format(bad, false).is_err());

        let input = MatrixInput::Dense(array![[0.0, 1.0], [1.0, 0.0]]);
        let (adj, bip) = get_adjacency(input, true, false, false, false).unwrap();
        assert!(!bip);
        check_square(adj.shape()).unwrap();
    }
}
