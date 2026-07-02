use ndarray::Array1;
use sprs::{CsMat, TriMat};

use crate::linalg::operators::Regularizer;
use crate::linalg::sparse_lowrank::SparseLR;
use crate::linalg::svd_solver::{fit_partial_svd, PartialSvdResult, SvdSolverKind, SVDInput};

/// Errors raised by SVD-family embedding estimators.
#[derive(Debug, Clone, PartialEq)]
pub enum SVDError {
    /// The input matrix is empty or SVD setup failed.
    InvalidInput,
    /// `predict` adjacency vector length mismatches column count.
    InvalidAdjacencyVectorLength,
    /// `predict` adjacency vector contains negative weights.
    NegativeAdjacencyVector,
    /// `predict` called before `fit`.
    NotFitted,
}

/// Generalized SVD bipartite embedding estimator.
#[derive(Debug, Clone)]
pub struct GSVD {
    /// Number of singular components to retain.
    pub n_components: usize,
    /// Optional diagonal regularization strength.
    pub regularization: Option<f64>,
    /// Row-degree scaling exponent.
    pub factor_row: f64,
    /// Column-degree scaling exponent.
    pub factor_col: f64,
    /// Singular-value split exponent between row and column factors.
    pub factor_singular: f64,
    /// Row-normalize embeddings when true.
    pub normalized: bool,
    /// Partial SVD backend.
    pub solver: SvdSolverKind,
    /// Optional RNG seed for randomized solvers.
    pub random_state: Option<u64>,
    /// Fitted row-node embedding (alias of row block).
    pub embedding: Vec<Vec<f64>>,
    /// Fitted row-node embedding.
    pub embedding_row: Vec<Vec<f64>>,
    /// Fitted column-node embedding.
    pub embedding_col: Vec<Vec<f64>>,
    /// Retained singular values.
    pub singular_values: Vec<f64>,
    /// Left singular vectors.
    pub singular_vectors_left: Vec<Vec<f64>>,
    /// Right singular vectors.
    pub singular_vectors_right: Vec<Vec<f64>>,
    /// Column weight vector from the last fit.
    pub weights_col: Vec<f64>,
}

impl Default for GSVD {
    fn default() -> Self {
        Self::new(2, None, 0.5, 0.5, 0.0, true, SvdSolverKind::Lanczos)
    }
}

impl GSVD {
    /// Creates a GSVD estimator with explicit hyperparameters.
    ///
    /// # Arguments
    /// - `n_components`: Number of singular components to retain.
    /// - `regularization`: Optional diagonal regularization strength.
    /// - `factor_row`: Row-degree scaling exponent.
    /// - `factor_col`: Column-degree scaling exponent.
    /// - `factor_singular`: Singular-value split exponent.
    /// - `normalized`: Row-normalize embeddings when true.
    /// - `solver`: Partial SVD backend.
    pub fn new(
        n_components: usize,
        regularization: Option<f64>,
        factor_row: f64,
        factor_col: f64,
        factor_singular: f64,
        normalized: bool,
        solver: SvdSolverKind,
    ) -> Self {
        Self {
            n_components,
            regularization,
            factor_row,
            factor_col,
            factor_singular,
            normalized,
            solver,
            random_state: None,
            embedding: Vec::new(),
            embedding_row: Vec::new(),
            embedding_col: Vec::new(),
            singular_values: Vec::new(),
            singular_vectors_left: Vec::new(),
            singular_vectors_right: Vec::new(),
            weights_col: Vec::new(),
        }
    }

    /// Sets the RNG seed for randomized solvers.
    pub fn with_random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    fn row_normalize(x: &mut [Vec<f64>]) {
        for r in x {
            let norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for v in r {
                    *v /= norm;
                }
            }
        }
    }

    fn matvec(adjacency: &CsMat<f64>, x: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(adjacency.rows());
        for (i, row) in adjacency.outer_iterator().enumerate() {
            let mut s = 0.0;
            for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                s += v * x[j];
            }
            out[i] = s;
        }
        out
    }

    fn matvec_transpose(adjacency: &CsMat<f64>, x: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(adjacency.cols());
        for (i, row) in adjacency.outer_iterator().enumerate() {
            let xi = x[i];
            if xi == 0.0 {
                continue;
            }
            for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                out[j] += v * xi;
            }
        }
        out
    }

    fn scale_adjacency(
        adjacency: &CsMat<f64>,
        row_scale: &[f64],
        col_scale: &[f64],
    ) -> CsMat<f64> {
        let (n_row, n_col) = adjacency.shape();
        let mut tri = TriMat::<f64>::new((n_row, n_col));
        for (i, row) in adjacency.outer_iterator().enumerate() {
            let rs = row_scale[i];
            for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                tri.add_triplet(i, j, rs * v * col_scale[j]);
            }
        }
        tri.to_csr::<usize>()
    }

    fn build_svd_input(
        adjacency: &CsMat<f64>,
        regularization: f64,
        factor_row: f64,
        factor_col: f64,
    ) -> Result<(SVDInput, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), SVDError> {
        let n_row = adjacency.rows();
        let n_col = adjacency.cols();
        let ones_col = Array1::from_elem(n_col, 1.0);
        let ones_row = Array1::from_elem(n_row, 1.0);

        let mut weights_row = Self::matvec(adjacency, &ones_col).to_vec();
        let mut weights_col = Self::matvec_transpose(adjacency, &ones_row).to_vec();
        if regularization > 0.0 {
            let reg = Regularizer::new(adjacency, regularization);
            weights_row = reg.dot_vec(&ones_col).to_vec();
            let add = regularization * n_row as f64 / n_col as f64;
            for w in &mut weights_col {
                *w += add;
            }
        }

        let row_scale: Vec<f64> = weights_row
            .iter()
            .map(|&w| {
                let w = if w <= 0.0 { 1.0 } else { w };
                w.powf(-factor_row)
            })
            .collect();
        let col_scale: Vec<f64> = weights_col
            .iter()
            .map(|&w| {
                let w = if w <= 0.0 { 1.0 } else { w };
                w.powf(-factor_col)
            })
            .collect();

        let scaled = Self::scale_adjacency(adjacency, &row_scale, &col_scale);
        let input = if regularization > 0.0 {
            let u_scaled = Array1::from_elem(n_row, regularization)
                .iter()
                .zip(row_scale.iter())
                .map(|(&u, &rs)| rs * u)
                .collect::<Vec<_>>();
            let v_scaled = Array1::from_elem(n_col, 1.0 / n_col as f64)
                .iter()
                .zip(col_scale.iter())
                .map(|(&v, &cs)| cs * v)
                .collect::<Vec<_>>();
            SVDInput::SparseLR(
                SparseLR::new(
                    &scaled,
                    vec![(
                        Array1::from_vec(u_scaled),
                        Array1::from_vec(v_scaled),
                    )],
                )
                .map_err(|_| SVDError::InvalidInput)?,
            )
        } else {
            SVDInput::Sparse(scaled)
        };

        Ok((input, weights_row, weights_col, row_scale, col_scale))
    }

    fn store_from_partial_svd(
        &mut self,
        n_row: usize,
        n_col: usize,
        k: usize,
        row_scale: &[f64],
        col_scale: &[f64],
        weights_col: Vec<f64>,
        result: PartialSvdResult,
    ) {
        let PartialSvdResult { u, s, v } = result;
        let k_eff = k.min(s.len()).max(1);

        let mut row = vec![vec![0.0; k_eff]; n_row];
        let mut col = vec![vec![0.0; k_eff]; n_col];
        for c in 0..k_eff {
            let s_left = s[c].powf(1.0 - self.factor_singular);
            let s_right = s[c].powf(self.factor_singular);
            for i in 0..n_row {
                row[i][c] = row_scale[i] * u[[i, c]] * s_left;
            }
            for j in 0..n_col {
                col[j][c] = col_scale[j] * v[[j, c]] * s_right;
            }
        }
        if self.normalized {
            Self::row_normalize(&mut row);
            Self::row_normalize(&mut col);
        }

        self.embedding_row = row.clone();
        self.embedding_col = col.clone();
        self.embedding = row;
        self.singular_vectors_left = (0..n_row)
            .map(|i| (0..k_eff).map(|c| u[[i, c]]).collect())
            .collect();
        self.singular_vectors_right = (0..n_col)
            .map(|j| (0..k_eff).map(|c| v[[j, c]]).collect())
            .collect();
        self.singular_values = (0..k_eff).map(|c| s[c]).collect();
        self.weights_col = weights_col;
    }

    /// Fits GSVD embeddings on a biadjacency matrix.
    ///
    /// # Errors
    /// Returns [`SVDError::InvalidInput`] for empty inputs or SVD failure.
    pub fn fit(&mut self, input_matrix: &CsMat<f64>) -> Result<(), SVDError> {
        let n_row = input_matrix.rows();
        let n_col = input_matrix.cols();
        if n_row == 0 || n_col == 0 {
            return Err(SVDError::InvalidInput);
        }
        let max_dim = n_row.min(n_col).saturating_sub(1).max(1);
        let k = self.n_components.min(max_dim);
        let regularization = self.regularization.unwrap_or(0.0).max(0.0);

        let (input, _weights_row, weights_col, row_scale, col_scale) =
            Self::build_svd_input(input_matrix, regularization, self.factor_row, self.factor_col)?;

        let result = fit_partial_svd(self.solver, input, k, self.random_state)
            .map_err(|_| SVDError::InvalidInput)?;
        self.store_from_partial_svd(n_row, n_col, k, &row_scale, &col_scale, weights_col, result);
        Ok(())
    }

    /// Fits the estimator and returns the row-node embedding.
    ///
    /// # Errors
    /// Returns the same errors as [`GSVD::fit`].
    pub fn fit_transform(&mut self, input_matrix: &CsMat<f64>) -> Result<Vec<Vec<f64>>, SVDError> {
        self.fit(input_matrix)?;
        Ok(self.embedding.clone())
    }

    /// Embeds a new column adjacency vector using fitted singular vectors.
    ///
    /// # Arguments
    /// - `adjacency_vectors`: Nonnegative column weights, length `n_col`.
    ///
    /// # Errors
    /// Returns:
    /// - [`SVDError::NotFitted`] when called before `fit`.
    /// - [`SVDError::InvalidAdjacencyVectorLength`] on length mismatch.
    /// - [`SVDError::NegativeAdjacencyVector`] when any weight is negative.
    pub fn predict(&self, adjacency_vectors: &[f64]) -> Result<Vec<f64>, SVDError> {
        if self.embedding_col.is_empty() {
            return Err(SVDError::NotFitted);
        }
        let n_col = self.embedding_col.len();
        if adjacency_vectors.len() != n_col {
            return Err(SVDError::InvalidAdjacencyVectorLength);
        }
        if adjacency_vectors.iter().any(|x| *x < 0.0) {
            return Err(SVDError::NegativeAdjacencyVector);
        }
        let k = self.embedding_col[0].len();
        let regularization = self.regularization.unwrap_or(0.0).max(0.0);
        let mean = if adjacency_vectors.is_empty() {
            0.0
        } else {
            adjacency_vectors.iter().sum::<f64>() / adjacency_vectors.len() as f64
        };
        let mut x_reg = vec![0.0; n_col];
        for j in 0..n_col {
            x_reg[j] = adjacency_vectors[j] + regularization * mean;
        }
        let weight_row = x_reg.iter().sum::<f64>();
        let row_scale = if weight_row > 0.0 {
            weight_row.powf(-self.factor_row)
        } else {
            0.0
        };
        let mut out = vec![0.0; k];
        for (j, &w) in x_reg.iter().enumerate() {
            let col_scale = self
                .weights_col
                .get(j)
                .copied()
                .unwrap_or(1.0)
                .max(1e-12)
                .powf(-self.factor_col);
            let wj = row_scale * w * col_scale;
            for c in 0..k {
                out[c] += wj * self.singular_vectors_right[j][c];
            }
        }
        for v in &mut out {
            *v *= row_scale;
        }
        for (c, v) in out.iter_mut().enumerate().take(k) {
            let s = self.singular_values.get(c).copied().unwrap_or(0.0);
            if s > 0.0 && self.factor_singular != 0.0 {
                *v /= s.powf(self.factor_singular);
            }
        }
        let norm = out.iter().map(|v| v * v).sum::<f64>().sqrt();
        if self.normalized && norm > 0.0 {
            for v in &mut out {
                *v /= norm;
            }
        }
        Ok(out)
    }
}

/// Standard SVD embedding wrapper around [`GSVD`] with zero row/column factors.
#[derive(Debug, Clone)]
pub struct SVD {
    inner: GSVD,
}

impl SVD {
    /// Creates an SVD estimator.
    ///
    /// # Arguments
    /// - `n_components`: Number of singular components to retain.
    /// - `regularization`: Optional diagonal regularization strength.
    /// - `factor_singular`: Singular-value split exponent.
    /// - `normalized`: Row-normalize embeddings when true.
    /// - `solver`: Partial SVD backend.
    pub fn new(
        n_components: usize,
        regularization: Option<f64>,
        factor_singular: f64,
        normalized: bool,
        solver: SvdSolverKind,
    ) -> Self {
        Self {
            inner: GSVD::new(
                n_components,
                regularization,
                0.0,
                0.0,
                factor_singular,
                normalized,
                solver,
            ),
        }
    }
    /// Fits SVD embeddings on a biadjacency matrix.
    ///
    /// # Errors
    /// Returns the same errors as [`GSVD::fit`].
    pub fn fit(&mut self, input_matrix: &CsMat<f64>) -> Result<(), SVDError> {
        self.inner.fit(input_matrix)
    }
    /// Returns the fitted row-node embedding.
    pub fn embedding_row(&self) -> &Vec<Vec<f64>> {
        &self.inner.embedding_row
    }
}

/// PCA embedding via centered sparse SVD.
#[derive(Debug, Clone)]
pub struct PCA {
    /// Number of principal components to retain.
    pub n_components: usize,
    /// Row-normalize embeddings when true.
    pub normalized: bool,
    /// Partial SVD backend.
    pub solver: SvdSolverKind,
    /// Optional RNG seed for randomized solvers.
    pub random_state: Option<u64>,
    /// Fitted row-node principal components.
    pub embedding_row: Vec<Vec<f64>>,
    /// Fitted column-node principal components.
    pub embedding_col: Vec<Vec<f64>>,
}

impl PCA {
    /// Creates a PCA estimator.
    ///
    /// # Arguments
    /// - `n_components`: Number of principal components to retain.
    /// - `normalized`: Row-normalize embeddings when true.
    /// - `solver`: Partial SVD backend.
    pub fn new(n_components: usize, normalized: bool, solver: SvdSolverKind) -> Self {
        Self {
            n_components,
            normalized,
            solver,
            random_state: None,
            embedding_row: Vec::new(),
            embedding_col: Vec::new(),
        }
    }

    /// Sets the RNG seed for randomized solvers.
    pub fn with_random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Fits PCA embeddings on a biadjacency matrix.
    ///
    /// # Errors
    /// Returns [`SVDError::InvalidInput`] for empty inputs or SVD failure.
    pub fn fit(&mut self, adjacency: &CsMat<f64>) -> Result<(), SVDError> {
        let (n_row, n_col) = adjacency.shape();
        if n_row == 0 || n_col == 0 {
            return Err(SVDError::InvalidInput);
        }
        let mut col_sums = vec![0.0; n_col];
        for i in 0..n_row {
            if let Some(row) = adjacency.outer_view(i) {
                for (j, v) in row.iter() {
                    col_sums[j] += *v;
                }
            }
        }
        let col_means = Array1::from_vec(
            col_sums
                .into_iter()
                .map(|s| s / n_row as f64)
                .collect::<Vec<f64>>(),
        );
        let minus_ones = Array1::from_vec(vec![-1.0; n_row]);
        let centered =
            SparseLR::new(adjacency, vec![(minus_ones, col_means)]).map_err(|_| SVDError::InvalidInput)?;
        let max_dim = n_row.min(n_col).saturating_sub(1).max(1);
        let k = self.n_components.min(max_dim);
        let result = fit_partial_svd(
            self.solver,
            SVDInput::SparseLR(centered),
            k,
            self.random_state,
        )
        .map_err(|_| SVDError::InvalidInput)?;
        let u = result.u;
        let v = result.v;
        self.embedding_row = (0..u.nrows())
            .map(|i| (0..u.ncols()).map(|c| u[[i, c]]).collect())
            .collect();
        self.embedding_col = (0..v.nrows())
            .map(|i| (0..v.ncols()).map(|c| v[[i, c]]).collect())
            .collect();
        if self.normalized {
            GSVD::row_normalize(&mut self.embedding_row);
            GSVD::row_normalize(&mut self.embedding_col);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::test_bigraph;

    #[test]
    fn test_options_contract() {
        let biadjacency = test_bigraph();
        let n_row = biadjacency.rows();
        let n_col = biadjacency.cols();
        let min_dim = n_row.min(n_col) - 1;
        let mut gsvd = GSVD::new(
            5,
            Some(0.0),
            0.5,
            0.5,
            0.0,
            true,
            SvdSolverKind::Halko,
        );
        gsvd.fit(&biadjacency).unwrap();
        assert_eq!(gsvd.embedding_row.len(), n_row);
        assert_eq!(gsvd.embedding_row[0].len(), min_dim);
        assert_eq!(gsvd.embedding_col.len(), n_col);
        assert_eq!(gsvd.embedding_col[0].len(), min_dim);

        let mut probe = vec![0.0; n_col];
        if n_col > 1 {
            probe[1] = 1.0;
        }
        if n_col > 2 {
            probe[2] = 1.0;
        }
        let embedding = gsvd.predict(&probe).unwrap();
        assert_eq!(embedding.len(), min_dim);

        let mut gsvd = GSVD::new(
            1,
            Some(0.1),
            0.5,
            0.5,
            0.0,
            true,
            SvdSolverKind::Lanczos,
        );
        gsvd.fit(&biadjacency).unwrap();
        assert_eq!(gsvd.embedding_row[0].len(), 1);

        let mut pca = PCA::new(min_dim, false, SvdSolverKind::Lanczos);
        pca.fit(&biadjacency).unwrap();
        let pca_lanczos_dim = pca.embedding_row[0].len();
        assert!(
            (1..=min_dim).contains(&pca_lanczos_dim),
            "Lanczos PCA returned invalid dimension: {pca_lanczos_dim}, expected in [1, {min_dim}]"
        );

        let mut pca = PCA::new(min_dim, false, SvdSolverKind::Halko);
        pca.fit(&biadjacency).unwrap();
        assert_eq!(pca.embedding_row[0].len(), min_dim);

        let mut svd = SVD::new(min_dim, None, 0.0, false, SvdSolverKind::Lanczos);
        svd.fit(&biadjacency).unwrap();
        let svd_lanczos_dim = svd.embedding_row()[0].len();
        assert!(
            (1..=min_dim).contains(&svd_lanczos_dim),
            "Lanczos SVD returned invalid dimension: {svd_lanczos_dim}, expected in [1, {min_dim}]"
        );
    }
}
