use ndarray::Array1;
use sprs::{CsMat, TriMat};

use crate::linalg::polynome::Polynome;
use crate::utils::format::{MatrixInput, get_adjacency};

/// Errors raised while computing Katz centrality.
#[derive(Debug, Clone, PartialEq)]
pub enum KatzError {
    /// Adjacency formatting or polynomial construction failed.
    InvalidInput,
}

/// Katz centrality via damped walk counts on sparse adjacency matrices.
#[derive(Debug, Clone)]
pub struct Katz {
    /// Damping factor applied to walks of increasing length.
    pub damping_factor: f64,
    /// Maximum walk length included in the score polynomial.
    pub path_length: usize,
    /// Whether the last fit used a bipartite adjacency layout.
    pub bipartite: bool,
    /// Fitted row scores (or full scores for square graphs).
    pub scores: Vec<f64>,
    /// Row-node scores when the input is bipartite.
    pub scores_row: Option<Vec<f64>>,
    /// Column-node scores when the input is bipartite.
    pub scores_col: Option<Vec<f64>>,
}

impl Default for Katz {
    fn default() -> Self {
        Self {
            damping_factor: 0.5,
            path_length: 4,
            bipartite: false,
            scores: Vec::new(),
            scores_row: None,
            scores_col: None,
        }
    }
}

impl Katz {
    fn to_bool_csr(adjacency: &CsMat<f64>) -> CsMat<f64> {
        let (n_row, n_col) = adjacency.shape();
        let mut tri = TriMat::<f64>::new((n_row, n_col));
        for (i, row) in adjacency.outer_iterator().enumerate() {
            for (j, v) in row.iter() {
                if *v != 0.0 {
                    tri.add_triplet(i, j, 1.0);
                }
            }
        }
        tri.to_csr::<usize>()
    }

    /// Creates a Katz estimator with explicit damping and path length.
    pub fn new(damping_factor: f64, path_length: usize) -> Self {
        Self {
            damping_factor,
            path_length,
            ..Self::default()
        }
    }

    /// Computes Katz scores from an adjacency or biadjacency matrix.
    ///
    /// Edge weights are treated as boolean presence indicators.
    ///
    /// # Errors
    /// Returns [`KatzError::InvalidInput`] when adjacency formatting or
    /// polynomial construction fails.
    pub fn fit(&mut self, input_matrix: &CsMat<f64>) -> Result<(), KatzError> {
        let (adjacency, bipartite) = get_adjacency(
            MatrixInput::Sparse(input_matrix.to_owned()),
            true,
            false,
            false,
            false,
        )
        .map_err(|_| KatzError::InvalidInput)?;
        self.bipartite = bipartite;
        let n = adjacency.rows();
        let adjacency_bool = Self::to_bool_csr(&adjacency);
        let at = adjacency_bool.transpose_view().to_csr();
        let mut coefs = Array1::<f64>::zeros(self.path_length + 1);
        for k in 0..=self.path_length {
            coefs[k] = self.damping_factor.powi(k as i32);
        }
        coefs[0] = 0.0;
        let polynome = Polynome::new(&at, &coefs).map_err(|_| KatzError::InvalidInput)?;
        self.scores = polynome.dot_vec(&Array1::ones(n)).to_vec();
        if self.bipartite {
            let n_row = input_matrix.rows();
            self.scores_row = Some(self.scores[..n_row].to_vec());
            self.scores_col = Some(self.scores[n_row..].to_vec());
            self.scores = self.scores_row.clone().unwrap_or_default();
        }
        Ok(())
    }

    /// Fits the estimator and returns row scores.
    ///
    /// # Errors
    /// Returns the same errors as [`Katz::fit`].
    pub fn fit_predict(&mut self, input_matrix: &CsMat<f64>) -> Result<Vec<f64>, KatzError> {
        self.fit(input_matrix)?;
        Ok(self.scores.clone())
    }

    /// Returns fitted row or column scores.
    ///
    /// For non-bipartite usage, `columns = true` returns an empty vector.
    pub fn predict(&self, columns: bool) -> Vec<f64> {
        if columns {
            self.scores_col.clone().unwrap_or_default()
        } else {
            self.scores.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::{test_bigraph, test_digraph, test_graph};

    #[test]
    fn test_basic_shapes_nonnegative() {
        for adjacency in [test_graph(), test_digraph()] {
            let n = adjacency.rows();
            let mut katz = Katz::default();
            let score = katz.fit_predict(&adjacency).unwrap();
            assert_eq!(score.len(), n);
            assert!(score.iter().all(|x| *x >= 0.0));
        }
    }

    #[test]
    fn test_bipartite_split() {
        let biadjacency = test_bigraph();
        let (n_row, n_col) = biadjacency.shape();
        let mut katz = Katz::default();
        katz.fit(&biadjacency).unwrap();
        assert_eq!(katz.scores_row.clone().unwrap_or_default().len(), n_row);
        assert_eq!(katz.scores_col.clone().unwrap_or_default().len(), n_col);
    }

    #[test]
    fn test_weighted_edges_treated_as_boolean() {
        let mut tri = TriMat::<f64>::new((3, 3));
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(1, 2, 1.0);
        let a1 = tri.to_csr::<usize>();

        let mut tri = TriMat::<f64>::new((3, 3));
        tri.add_triplet(0, 1, 10.0);
        tri.add_triplet(1, 2, 5.0);
        let a2 = tri.to_csr::<usize>();

        let mut k1 = Katz::default();
        let mut k2 = Katz::default();
        let s1 = k1.fit_predict(&a1).expect("katz bool baseline");
        let s2 = k2.fit_predict(&a2).expect("katz weighted");
        assert_eq!(s1.len(), s2.len());
        for (x, y) in s1.iter().zip(s2.iter()) {
            assert!((x - y).abs() < 1e-12);
        }
    }
}
