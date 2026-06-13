use sprs::CsMat;

use crate::linalg::normalizer::normalize_sparse;
use crate::utils::format::{MatrixInput, get_adjacency_values};
use crate::utils::values::ValuesInput;

/// Errors raised by graph regression estimators.
#[derive(Debug, Clone, PartialEq)]
pub enum RegressionError {
    /// The iteration count is not strictly positive.
    InvalidNIter,
    /// The damping factor is outside `[0, 1]`.
    InvalidDampingFactor,
    /// Adjacency or value formatting checks failed.
    InvalidInput,
    /// The estimator has not been fitted yet.
    NotFitted,
}

fn init_temperatures(seeds: &[f64], init: Option<f64>) -> (Vec<f64>, Vec<bool>) {
    let n = seeds.len();
    let border: Vec<bool> = seeds.iter().map(|&x| x >= 0.0).collect();
    let mut temperatures = vec![0.0; n];
    if let Some(v) = init {
        temperatures.fill(v);
    } else {
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for (i, &is_border) in border.iter().enumerate() {
            if is_border {
                sum += seeds[i];
                cnt += 1;
            }
        }
        let mean = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
        temperatures.fill(mean);
    }
    for i in 0..n {
        if border[i] {
            temperatures[i] = seeds[i];
        }
    }
    (temperatures, border)
}

/// Row-normalized transition matrix of ``adjacency.T`` (matches Python ``normalize(adjacency.T)``).
fn transition_matrix_transpose(adjacency: &CsMat<f64>) -> Result<CsMat<f64>, RegressionError> {
    let transposed = adjacency.transpose_view().to_csr();
    normalize_sparse(&transposed, 1).map_err(|_| RegressionError::InvalidInput)
}

/// ``true`` when row ``i`` has no outgoing transitions (dangling node).
fn dangling_rows(transition: &CsMat<f64>) -> Vec<bool> {
    (0..transition.rows())
        .map(|i| {
            transition
                .outer_view(i)
                .map(|row| row.nnz() == 0)
                .unwrap_or(true)
        })
        .collect()
}

fn transition_matvec(transition: &CsMat<f64>, x: &[f64], dangling: &[bool]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        if dangling[i] {
            out[i] = x[i];
        } else if let Some(row) = transition.outer_view(i) {
            let mut s = 0.0;
            for (j, v) in row.iter() {
                s += v * x[j];
            }
            out[i] = s;
        }
    }
    out
}

/// One step of ``v <- (1 - d) v + d P v`` without materializing the damped operator.
fn damped_diffusion_step(
    transition: &CsMat<f64>,
    v: &[f64],
    damping: f64,
    dangling: &[bool],
) -> Vec<f64> {
    let pv = transition_matvec(transition, v, dangling);
    let keep = 1.0 - damping;
    pv.into_iter()
        .zip(v.iter())
        .map(|(p, &vi)| keep * vi + damping * p)
        .collect()
}

fn sparse_vec_mul(mat: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
    let dangling = vec![false; mat.rows()];
    transition_matvec(mat, x, &dangling)
}

/// Damped diffusion regressor on sparse adjacency matrices.
#[derive(Debug, Clone)]
pub struct Diffusion {
    /// Number of diffusion iterations.
    pub n_iter: usize,
    /// Damping factor in `[0, 1]`.
    pub damping_factor: f64,
    /// Whether the last fit used a bipartite adjacency layout.
    pub bipartite: bool,
    /// Fitted row values (or full values for square graphs).
    pub values: Vec<f64>,
    /// Row-node values when the input is bipartite.
    pub values_row: Option<Vec<f64>>,
    /// Column-node values when the input is bipartite.
    pub values_col: Option<Vec<f64>>,
}

impl Default for Diffusion {
    fn default() -> Self {
        Self::new(3, 0.5).unwrap()
    }
}

impl Diffusion {
    /// Creates a diffusion regressor.
    ///
    /// # Errors
    /// Returns:
    /// - [`RegressionError::InvalidNIter`] if `n_iter <= 0`
    /// - [`RegressionError::InvalidDampingFactor`] if `damping_factor` is not
    ///   in `[0, 1]`
    pub fn new(n_iter: isize, damping_factor: f64) -> Result<Self, RegressionError> {
        if n_iter <= 0 {
            return Err(RegressionError::InvalidNIter);
        }
        if !(0.0..=1.0).contains(&damping_factor) {
            return Err(RegressionError::InvalidDampingFactor);
        }
        Ok(Self {
            n_iter: n_iter as usize,
            damping_factor,
            bipartite: false,
            values: Vec::new(),
            values_row: None,
            values_col: None,
        })
    }

    /// Fits diffusion values from optional seed values.
    ///
    /// # Errors
    /// Returns [`RegressionError::InvalidInput`] when adjacency/value
    /// formatting checks fail.
    pub fn fit(
        &mut self,
        input_matrix: &CsMat<f64>,
        values: Option<ValuesInput>,
        values_row: Option<ValuesInput>,
        values_col: Option<ValuesInput>,
        init: Option<f64>,
        force_bipartite: bool,
    ) -> Result<(), RegressionError> {
        let values = if values.is_none()
            && values_row.is_none()
            && values_col.is_none()
            && !force_bipartite
            && input_matrix.rows() == input_matrix.cols()
        {
            Some(ValuesInput::Vector(vec![1.0; input_matrix.rows()]))
        } else {
            values
        };
        let (adjacency, seeds, bip) = get_adjacency_values(
            MatrixInput::Sparse(input_matrix.clone()),
            true,
            force_bipartite,
            false,
            values,
            values_row,
            values_col,
            -1.0,
            None,
        )
        .map_err(|_| RegressionError::InvalidInput)?;
        self.bipartite = bip;
        let (mut v, _) = init_temperatures(&seeds, init);

        let transition = transition_matrix_transpose(&adjacency)?;
        let dangling = dangling_rows(&transition);
        for _ in 0..self.n_iter {
            v = damped_diffusion_step(&transition, &v, self.damping_factor, &dangling);
        }
        self.values = v;
        if self.bipartite {
            let n_row = input_matrix.rows();
            self.values_row = Some(self.values[..n_row].to_vec());
            self.values_col = Some(self.values[n_row..].to_vec());
            self.values = self.values_row.clone().unwrap_or_default();
        } else {
            self.values_row = None;
            self.values_col = None;
        }
        Ok(())
    }

    /// Fits the estimator and returns row values.
    ///
    /// # Errors
    /// Returns the same errors as [`Diffusion::fit`].
    pub fn fit_predict(
        &mut self,
        input_matrix: &CsMat<f64>,
        values: Option<ValuesInput>,
        values_row: Option<ValuesInput>,
        values_col: Option<ValuesInput>,
        init: Option<f64>,
        force_bipartite: bool,
    ) -> Result<Vec<f64>, RegressionError> {
        self.fit(
            input_matrix,
            values,
            values_row,
            values_col,
            init,
            force_bipartite,
        )?;
        Ok(self.values.clone())
    }

    /// Returns fitted values.
    ///
    /// # Errors
    /// Returns [`RegressionError::NotFitted`] when called before `fit`.
    pub fn predict(&self, columns: bool) -> Result<Vec<f64>, RegressionError> {
        if self.values.is_empty() && self.values_row.is_none() && self.values_col.is_none() {
            return Err(RegressionError::NotFitted);
        }
        if columns {
            self.values_col.clone().ok_or(RegressionError::NotFitted)
        } else {
            Ok(self.values.clone())
        }
    }
}

/// Dirichlet regressor with fixed boundary values.
#[derive(Debug, Clone)]
pub struct Dirichlet {
    /// Number of diffusion iterations.
    pub n_iter: usize,
    /// Whether the last fit used a bipartite adjacency layout.
    pub bipartite: bool,
    /// Fitted row values (or full values for square graphs).
    pub values: Vec<f64>,
    /// Row-node values when the input is bipartite.
    pub values_row: Option<Vec<f64>>,
    /// Column-node values when the input is bipartite.
    pub values_col: Option<Vec<f64>>,
}

impl Default for Dirichlet {
    fn default() -> Self {
        Self::new(10).unwrap()
    }
}

impl Dirichlet {
    /// Creates a Dirichlet regressor with the given iteration count.
    ///
    /// # Errors
    /// Returns [`RegressionError::InvalidNIter`] when `n_iter <= 0`.
    pub fn new(n_iter: isize) -> Result<Self, RegressionError> {
        if n_iter <= 0 {
            return Err(RegressionError::InvalidNIter);
        }
        Ok(Self {
            n_iter: n_iter as usize,
            bipartite: false,
            values: Vec::new(),
            values_row: None,
            values_col: None,
        })
    }

    /// Fits Dirichlet values from optional seed values.
    ///
    /// Border nodes (non-negative seeds) keep their initial temperatures at
    /// every iteration.
    ///
    /// # Errors
    /// Returns [`RegressionError::InvalidInput`] when adjacency or value
    /// formatting checks fail.
    pub fn fit(
        &mut self,
        input_matrix: &CsMat<f64>,
        values: Option<ValuesInput>,
        values_row: Option<ValuesInput>,
        values_col: Option<ValuesInput>,
        init: Option<f64>,
        force_bipartite: bool,
    ) -> Result<(), RegressionError> {
        let values = if values.is_none()
            && values_row.is_none()
            && values_col.is_none()
            && !force_bipartite
            && input_matrix.rows() == input_matrix.cols()
        {
            Some(ValuesInput::Vector(vec![1.0; input_matrix.rows()]))
        } else {
            values
        };
        let (adjacency, seeds, bip) = get_adjacency_values(
            MatrixInput::Sparse(input_matrix.clone()),
            true,
            force_bipartite,
            false,
            values,
            values_row,
            values_col,
            -1.0,
            None,
        )
        .map_err(|_| RegressionError::InvalidInput)?;
        self.bipartite = bip;
        let (temps, border) = init_temperatures(&seeds, init);
        let mut v = temps.clone();
        let diffusion = normalize_sparse(&adjacency, 1).map_err(|_| RegressionError::InvalidInput)?;
        for _ in 0..self.n_iter {
            v = sparse_vec_mul(&diffusion, &v);
            for i in 0..v.len() {
                if border[i] {
                    v[i] = temps[i];
                }
            }
        }
        self.values = v;
        if self.bipartite {
            let n_row = input_matrix.rows();
            self.values_row = Some(self.values[..n_row].to_vec());
            self.values_col = Some(self.values[n_row..].to_vec());
            self.values = self.values_row.clone().unwrap_or_default();
        } else {
            self.values_row = None;
            self.values_col = None;
        }
        Ok(())
    }

    /// Fits the estimator and returns row values.
    ///
    /// # Errors
    /// Returns the same errors as [`Dirichlet::fit`].
    pub fn fit_predict(
        &mut self,
        input_matrix: &CsMat<f64>,
        values: Option<ValuesInput>,
        values_row: Option<ValuesInput>,
        values_col: Option<ValuesInput>,
        init: Option<f64>,
        force_bipartite: bool,
    ) -> Result<Vec<f64>, RegressionError> {
        self.fit(
            input_matrix,
            values,
            values_row,
            values_col,
            init,
            force_bipartite,
        )?;
        Ok(self.values.clone())
    }

    /// Returns fitted row or column values.
    ///
    /// # Errors
    /// Returns [`RegressionError::NotFitted`] when called before `fit` or when
    /// column values are unavailable.
    pub fn predict(&self, columns: bool) -> Result<Vec<f64>, RegressionError> {
        if self.values.is_empty() && self.values_row.is_none() && self.values_col.is_none() {
            return Err(RegressionError::NotFitted);
        }
        if columns {
            self.values_col.clone().ok_or(RegressionError::NotFitted)
        } else {
            Ok(self.values.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::{test_digraph, test_graph};
    use sprs::TriMat;

    fn test_bigraph() -> CsMat<f64> {
        let mut tri = TriMat::<f64>::new((6, 8));
        for i in 0..6 {
            tri.add_triplet(i, i % 8, 1.0);
            tri.add_triplet(i, (i + 2) % 8, 1.0);
        }
        tri.to_csr::<usize>()
    }

    #[test]
    fn test_predict() {
        let adjacency = test_graph();
        let algos = vec![
            Diffusion::default()
                .fit_predict(
                    &adjacency,
                    Some(ValuesInput::Map(
                        [(0usize, 0.0), (1, 1.0), (2, 0.5)].into_iter().collect(),
                    )),
                    None,
                    None,
                    None,
                    false,
                )
                .unwrap(),
            Dirichlet::default()
                .fit_predict(
                    &adjacency,
                    Some(ValuesInput::Map(
                        [(0usize, 0.0), (1, 1.0), (2, 0.5)].into_iter().collect(),
                    )),
                    None,
                    None,
                    None,
                    false,
                )
                .unwrap(),
        ];
        assert_eq!(algos[0].len(), 10);
        assert_eq!(algos[1].len(), 10);
    }

    #[test]
    fn test_no_iter() {
        assert!(matches!(
            Diffusion::new(-1, 0.5),
            Err(RegressionError::InvalidNIter)
        ));
        assert!(matches!(
            Dirichlet::new(0),
            Err(RegressionError::InvalidNIter)
        ));
        assert!(matches!(
            Diffusion::new(3, -0.1),
            Err(RegressionError::InvalidDampingFactor)
        ));
        assert!(matches!(
            Diffusion::new(3, 1.1),
            Err(RegressionError::InvalidDampingFactor)
        ));
    }

    #[test]
    fn test_range_and_bipartite() {
        for adjacency in [test_graph(), test_digraph()] {
            let mut d = Diffusion::default();
            let values = d
                .fit_predict(
                    &adjacency,
                    Some(ValuesInput::Map(
                        [(0usize, 0.0), (1, 1.0), (2, 0.5)].into_iter().collect(),
                    )),
                    None,
                    None,
                    Some(0.3),
                    false,
                )
                .unwrap();
            assert!(values.iter().all(|x| *x <= 1.0 && *x >= 0.0));

            let mut dd = Dirichlet::default();
            let values = dd
                .fit_predict(
                    &adjacency,
                    Some(ValuesInput::Map(
                        [(0usize, 0.0), (1, 1.0), (2, 0.5)].into_iter().collect(),
                    )),
                    None,
                    None,
                    Some(0.3),
                    false,
                )
                .unwrap();
            assert!(values.iter().all(|x| *x <= 1.0 && *x >= 0.0));
        }

        let bi = test_bigraph();
        for mut algo in [Diffusion::default(), Diffusion::default()] {
            let values = algo
                .fit_predict(
                    &bi,
                    None,
                    Some(ValuesInput::Map([(0usize, 1.0)].into_iter().collect())),
                    None,
                    None,
                    true,
                )
                .unwrap();
            assert!(values.iter().all(|x| *x <= 1.0 && *x >= 0.0));
        }
    }

    #[test]
    fn test_not_fitted_predict() {
        let d = Diffusion::default();
        assert!(matches!(d.predict(false), Err(RegressionError::NotFitted)));
        assert!(matches!(d.predict(true), Err(RegressionError::NotFitted)));

        let dd = Dirichlet::default();
        assert!(matches!(dd.predict(false), Err(RegressionError::NotFitted)));
        assert!(matches!(dd.predict(true), Err(RegressionError::NotFitted)));
    }

    #[test]
    fn test_default_values_none_parity_nonzero() {
        let adjacency = test_graph();
        let mut d = Diffusion::default();
        let out = d
            .fit_predict(&adjacency, None, None, None, None, false)
            .expect("diffusion fit_predict");
        assert!(out.iter().any(|&x| x > 0.0));

        let mut dd = Dirichlet::default();
        let out = dd
            .fit_predict(&adjacency, None, None, None, None, false)
            .expect("dirichlet fit_predict");
        assert!(out.iter().any(|&x| x > 0.0));
    }
}
