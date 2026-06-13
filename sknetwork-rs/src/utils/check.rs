use ndarray::{Array1, Array2};
use sprs::{CsMat, TriMat};

use crate::utils::format::MatrixInput;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by check error operations.
pub enum CheckError {
    /// Indicates not connected.
    NotConnected,
    /// Indicates not non negative.
    NotNonNegative,
    /// Indicates not positive.
    NotPositive,
    /// Indicates not proba.
    NotProba,
    /// Indicates not square.
    NotSquare,
    /// Indicates not symmetric.
    NotSymmetric,
    /// Indicates unknown distribution.
    UnknownDistribution,
    /// Indicates wrong weights length.
    WrongWeightsLength,
    /// Indicates invalid weights.
    InvalidWeights,
    /// Indicates invalid proba range.
    InvalidProbaRange,
    /// Indicates invalid damping factor.
    InvalidDampingFactor,
    /// Indicates invalid adjacency vector length.
    InvalidAdjacencyVectorLength,
    /// Indicates invalid nclusters.
    InvalidNClusters,
    /// Indicates invalid min size.
    InvalidMinSize,
    /// Indicates invalid min nnz.
    InvalidMinNnz,
    /// Indicates invalid scaling.
    InvalidScaling,
    /// Indicates invalid vector dimension.
    InvalidVectorDimension,
    /// Indicates invalid vector length.
    InvalidVectorLength,
}

#[derive(Debug, Clone, PartialEq)]
/// WeightsInput enum.
pub enum WeightsInput {
    /// Indicates distribution.
    Distribution(String),
    /// Indicates vector.
    Vector(Array1<f64>),
}

/// Returns whether the input satisfies `has nonnegative entries sparse`.
pub fn has_nonnegative_entries_sparse(input_matrix: &CsMat<f64>) -> bool {
    input_matrix.data().iter().all(|x| *x >= 0.0)
}

/// Returns whether the input satisfies `has nonnegative entries dense`.
pub fn has_nonnegative_entries_dense(input_matrix: &Array2<f64>) -> bool {
    input_matrix.iter().all(|x| *x >= 0.0)
}

/// Returns whether the input satisfies `has positive entries`.
pub fn has_positive_entries(input_matrix: &Array1<f64>) -> bool {
    input_matrix.iter().all(|x| *x > 0.0)
}

/// Validates nonnegative sparse.
///
/// # Errors
///
/// Returns [`CheckError`] on failure.
pub fn check_nonnegative_sparse(input_matrix: &CsMat<f64>) -> Result<(), CheckError> {
    if has_nonnegative_entries_sparse(input_matrix) {
        Ok(())
    } else {
        Err(CheckError::NotNonNegative)
    }
}

/// Validates positive.
///
/// # Errors
///
/// Returns [`CheckError`] on failure.
pub fn check_positive(input_matrix: &Array1<f64>) -> Result<(), CheckError> {
    if has_positive_entries(input_matrix) {
        Ok(())
    } else {
        Err(CheckError::NotPositive)
    }
}

/// Returns whether the input satisfies `is proba array 1d`.
pub fn is_proba_array_1d(input_matrix: &Array1<f64>) -> bool {
    let sum: f64 = input_matrix.sum();
    input_matrix.iter().all(|x| *x >= 0.0) && (sum - 1.0).abs() <= 1e-12
}

/// Returns whether the input satisfies `is proba array 2d`.
pub fn is_proba_array_2d(input_matrix: &Array2<f64>) -> bool {
    for row in input_matrix.rows() {
        if row.iter().any(|x| *x < 0.0) {
            return false;
        }
        let s: f64 = row.iter().sum();
        if (s - 1.0).abs() > 1e-12 {
            return false;
        }
    }
    true
}

/// Returns whether the input satisfies `is square`.
pub fn is_square(input_shape: (usize, usize)) -> bool {
    input_shape.0 == input_shape.1
}

/// Validates square.
///
/// # Errors
///
/// Returns [`CheckError`] on failure.
pub fn check_square(input_shape: (usize, usize)) -> Result<(), CheckError> {
    if is_square(input_shape) {
        Ok(())
    } else {
        Err(CheckError::NotSquare)
    }
}

/// Returns whether the input satisfies `is symmetric`.
pub fn is_symmetric(input_matrix: &CsMat<f64>) -> bool {
    if !is_square(input_matrix.shape()) {
        return false;
    }
    let t = input_matrix.transpose_view().to_owned();
    let diff = input_matrix - &t;
    diff.nnz() == 0
}

/// Validates symmetry.
///
/// # Errors
///
/// Returns [`CheckError`] on failure.
pub fn check_symmetry(input_matrix: &CsMat<f64>) -> Result<(), CheckError> {
    if is_symmetric(input_matrix) {
        Ok(())
    } else {
        Err(CheckError::NotSymmetric)
    }
}

/// Returns whether the input satisfies `is weakly connected`.
pub fn is_weakly_connected(adjacency: &CsMat<f64>) -> bool {
    let (n, m) = adjacency.shape();
    if n == 0 || m == 0 {
        return true;
    }
    let sym = adjacency + &adjacency.transpose_view().to_owned();
    let mut seen = vec![false; n];
    let mut stack = vec![0usize];
    seen[0] = true;
    while let Some(u) = stack.pop() {
        if let Some(row) = sym.outer_view(u) {
            for v in row.indices() {
                if !seen[*v] {
                    seen[*v] = true;
                    stack.push(*v);
                }
            }
        }
    }
    seen.into_iter().all(|x| x)
}

/// Validates connected.
///
/// # Errors
///
/// Returns [`CheckError`] on failure.
pub fn check_connected(adjacency: &CsMat<f64>) -> Result<(), CheckError> {
    if is_weakly_connected(adjacency) {
        Ok(())
    } else {
        Err(CheckError::NotConnected)
    }
}

/// Computes make weights.
///
/// # Errors
///
/// Returns [`CheckError`] on failure.
pub fn make_weights(distribution: &str, adjacency: &CsMat<f64>) -> Result<Array1<f64>, CheckError> {
    let n = adjacency.rows();
    match distribution.to_lowercase().as_str() {
        "degree" => {
            let mut out = Array1::<f64>::zeros(n);
            for i in 0..n {
                let sum = adjacency
                    .outer_view(i)
                    .map(|row| row.data().iter().sum())
                    .unwrap_or(0.0);
                out[i] = sum;
            }
            Ok(out)
        }
        "uniform" => Ok(Array1::<f64>::ones(n)),
        _ => Err(CheckError::UnknownDistribution),
    }
}

/// Validates is proba.
///
/// # Errors
///
/// Returns [`CheckError`] on failure.
pub fn check_is_proba(entry: f64) -> Result<(), CheckError> {
    if (0.0..=1.0).contains(&entry) {
        Ok(())
    } else {
        Err(CheckError::InvalidProbaRange)
    }
}

/// Validates damping factor.
///
/// # Errors
///
/// Returns [`CheckError`] on failure.
pub fn check_damping_factor(damping_factor: f64) -> Result<(), CheckError> {
    if (0.0..1.0).contains(&damping_factor) {
        Ok(())
    } else {
        Err(CheckError::InvalidDampingFactor)
    }
}

/// Validates weights.
pub fn check_weights(
    weights: WeightsInput,
    adjacency: &CsMat<f64>,
    positive_entries: bool,
) -> Result<Array1<f64>, CheckError> {
    let n = adjacency.rows();
    let node_weights = match weights {
        WeightsInput::Vector(w) => {
            if w.len() != n {
                return Err(CheckError::WrongWeightsLength);
            }
            w
        }
        WeightsInput::Distribution(d) => make_weights(&d, adjacency)?,
    };

    if positive_entries {
        if !has_positive_entries(&node_weights) {
            return Err(CheckError::InvalidWeights);
        }
    } else if node_weights.iter().any(|x| *x < 0.0) || node_weights.sum() <= 0.0 {
        return Err(CheckError::InvalidWeights);
    }

    Ok(node_weights)
}

/// Returns probs.
pub fn get_probs(
    weights: WeightsInput,
    adjacency: &CsMat<f64>,
    positive_entries: bool,
) -> Result<Array1<f64>, CheckError> {
    let weights = check_weights(weights, adjacency, positive_entries)?;
    let s = weights.sum();
    Ok(weights / s)
}

/// Validates n neighbors.
pub fn check_n_neighbors(n_neighbors: usize, n_seeds: usize) -> usize {
    if n_seeds == 0 {
        0
    } else {
        n_neighbors.min(n_seeds)
    }
}

/// Validates labels.
///
/// # Errors
///
/// Returns [`usize`] on failure.
pub fn check_labels(labels: &Array1<i32>) -> Result<(Vec<i32>, usize), CheckError> {
    let mut classes: Vec<i32> = labels.iter().copied().filter(|x| *x >= 0).collect();
    classes.sort_unstable();
    classes.dedup();
    let n_classes = classes.len();
    if n_classes < 2 {
        Err(CheckError::InvalidWeights)
    } else {
        Ok((classes, n_classes))
    }
}

/// Validates n jobs.
pub fn check_n_jobs(n_jobs: Option<i32>) -> Option<i32> {
    match n_jobs {
        Some(-1) => None,
        None => Some(1),
        Some(x) => Some(x),
    }
}

/// Validates adjacency vector.
pub fn check_adjacency_vector(
    adjacency_vectors: MatrixInput,
    n: Option<usize>,
) -> Result<CsMat<f64>, CheckError> {
    let adjacency = match adjacency_vectors {
        MatrixInput::Dense(dense) => {
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
        MatrixInput::Sparse(s) => s,
    };
    if let Some(expected) = n {
        if adjacency.cols() != expected {
            return Err(CheckError::InvalidAdjacencyVectorLength);
        }
    }
    Ok(adjacency)
}

/// Validates n clusters.
///
/// # Errors
///
/// Returns [`CheckError`] on failure.
pub fn check_n_clusters(n_clusters: usize, n_row: usize, n_min: usize) -> Result<(), CheckError> {
    if n_clusters > n_row || n_clusters < n_min {
        Err(CheckError::InvalidNClusters)
    } else {
        Ok(())
    }
}

/// Validates min size.
///
/// # Errors
///
/// Returns [`CheckError`] on failure.
pub fn check_min_size(n_row: usize, n_min: usize) -> Result<(), CheckError> {
    if n_row < n_min {
        Err(CheckError::InvalidMinSize)
    } else {
        Ok(())
    }
}

/// Validates min nnz.
///
/// # Errors
///
/// Returns [`CheckError`] on failure.
pub fn check_min_nnz(nnz: usize, n_min: usize) -> Result<(), CheckError> {
    if nnz < n_min {
        Err(CheckError::InvalidMinNnz)
    } else {
        Ok(())
    }
}

/// Validates n components.
pub fn check_n_components(n_components: usize, n_min: usize) -> usize {
    if n_components > n_min {
        n_min
    } else {
        n_components
    }
}

/// Validates scaling.
pub fn check_scaling(
    scaling: f64,
    adjacency: &CsMat<f64>,
    regularize: bool,
) -> Result<(), CheckError> {
    if scaling < 0.0 {
        return Err(CheckError::InvalidScaling);
    }
    if scaling > 0.0 && !regularize && !is_weakly_connected(adjacency) {
        return Err(CheckError::InvalidScaling);
    }
    Ok(())
}

/// Validates vector format.
pub fn check_vector_format(
    vector_1: &Array1<f64>,
    vector_2: &Array1<f64>,
) -> Result<(), CheckError> {
    if vector_1.len() != vector_2.len() {
        Err(CheckError::InvalidVectorLength)
    } else {
        Ok(())
    }
}

/// Returns whether the input satisfies `has self loops`.
pub fn has_self_loops(input_matrix: &CsMat<f64>) -> bool {
    let n = input_matrix.rows().min(input_matrix.cols());
    (0..n).any(|i| input_matrix.get(i, i).is_some())
}

/// Computes add self loops.
pub fn add_self_loops(adjacency: &CsMat<f64>) -> CsMat<f64> {
    let (n_row, n_col) = adjacency.shape();
    let mut tri = TriMat::<f64>::new((n_row, n_col));
    for (i, row) in adjacency.outer_iterator().enumerate() {
        for (j, v) in row.iter() {
            tri.add_triplet(i, j, *v);
        }
    }
    for i in 0..n_row.min(n_col) {
        tri.add_triplet(i, i, 1.0);
    }
    tri.to_csr()
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};

    use super::*;

    fn csr_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
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

    #[test]
    fn test_square_connected_symmetry() {
        assert!(check_square((3, 3)).is_ok());
        assert!(check_square((3, 4)).is_err());

        let disconnected =
            csr_from_dense(&array![[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        assert!(check_connected(&disconnected).is_err());

        let symmetric = csr_from_dense(&array![[0.0, 1.0], [1.0, 0.0]]);
        assert!(check_symmetry(&symmetric).is_ok());
    }

    #[test]
    fn test_weights_and_probs() {
        let adjacency = csr_from_dense(&array![[0.0, 2.0], [1.0, 0.0]]);
        let degree = make_weights("degree", &adjacency).unwrap();
        assert_eq!(degree.to_vec(), vec![2.0, 1.0]);
        let probs = get_probs(
            WeightsInput::Distribution("uniform".to_string()),
            &adjacency,
            false,
        )
        .unwrap();
        assert!((probs.sum() - 1.0).abs() <= 1e-12);
    }

    #[test]
    fn test_misc_checks() {
        assert!(check_is_proba(0.5).is_ok());
        assert!(check_damping_factor(0.85).is_ok());
        assert_eq!(check_n_neighbors(10, 5), 5);
        assert_eq!(check_n_neighbors(1, 5), 1);
        assert_eq!(check_n_neighbors(5, 0), 0);

        let labels = Array1::from_vec(vec![0, 1, -1, 1]);
        let (_, n_classes) = check_labels(&labels).unwrap();
        assert_eq!(n_classes, 2);

        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);
        assert!(check_vector_format(&a, &b).is_ok());
    }

    #[test]
    fn test_self_loops() {
        let adjacency = csr_from_dense(&array![[0.0, 1.0], [1.0, 1.0]]);
        assert!(has_self_loops(&adjacency));
        let with_loops = add_self_loops(&adjacency);
        assert!(has_self_loops(&with_loops));
    }
}
