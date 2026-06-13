use sprs::CsMat;

use crate::linalg::svd_solver::{fit_partial_svd, SvdSolverKind, SVDInput};

/// Errors raised while computing HITS hub and authority scores.
#[derive(Debug, Clone, PartialEq)]
pub enum HITSError {
    /// The adjacency matrix is empty or SVD fitting failed.
    InvalidInput,
}

/// HITS hub/authority ranking on sparse adjacency or biadjacency matrices.
#[derive(Debug, Clone)]
pub struct HITS {
    /// Partial SVD backend used to extract the leading singular vectors.
    pub solver: SvdSolverKind,
    /// Hub scores for row nodes (or full scores for square graphs).
    pub scores: Vec<f64>,
    /// Hub scores for row nodes in bipartite inputs.
    pub scores_row: Vec<f64>,
    /// Authority scores for column nodes in bipartite inputs.
    pub scores_col: Vec<f64>,
}

impl Default for HITS {
    fn default() -> Self {
        Self {
            solver: SvdSolverKind::Lanczos,
            scores: Vec::new(),
            scores_row: Vec::new(),
            scores_col: Vec::new(),
        }
    }
}

impl HITS {
    /// Creates a HITS estimator with the given partial SVD solver.
    pub fn new(solver: SvdSolverKind) -> Self {
        Self {
            solver,
            scores: Vec::new(),
            scores_row: Vec::new(),
            scores_col: Vec::new(),
        }
    }

    /// Computes hub and authority scores from an adjacency matrix.
    ///
    /// # Errors
    /// Returns [`HITSError::InvalidInput`] when the matrix is empty or SVD
    /// fitting fails.
    pub fn fit(&mut self, adjacency: &CsMat<f64>) -> Result<(), HITSError> {
        let n_row = adjacency.rows();
        let n_col = adjacency.cols();
        if n_row == 0 || n_col == 0 {
            return Err(HITSError::InvalidInput);
        }

        let result = fit_partial_svd(
            self.solver,
            SVDInput::Sparse(adjacency.clone()),
            1,
            None,
        )
        .map_err(|_| HITSError::InvalidInput)?;

        let mut hubs: Vec<f64> = (0..n_row).map(|i| result.u[[i, 0]]).collect();
        let mut auth: Vec<f64> = (0..n_col).map(|j| result.v[[j, 0]]).collect();

        let h_pos = hubs.iter().filter(|&&x| x > 0.0).count();
        let h_neg = hubs.iter().filter(|&&x| x < 0.0).count();
        let a_pos = auth.iter().filter(|&&x| x > 0.0).count();
        let a_neg = auth.iter().filter(|&&x| x < 0.0).count();

        if h_pos <= h_neg {
            for x in &mut hubs {
                *x = -*x;
            }
        }
        if a_pos <= a_neg {
            for x in &mut auth {
                *x = -*x;
            }
        }
        for x in &mut hubs {
            if *x < 0.0 {
                *x = 0.0;
            }
        }
        for x in &mut auth {
            if *x < 0.0 {
                *x = 0.0;
            }
        }

        self.scores_row = hubs.clone();
        self.scores_col = auth;
        self.scores = hubs;
        Ok(())
    }

    /// Fits the estimator and returns hub scores.
    ///
    /// # Errors
    /// Returns the same errors as [`HITS::fit`].
    pub fn fit_predict(&mut self, adjacency: &CsMat<f64>) -> Result<Vec<f64>, HITSError> {
        self.fit(adjacency)?;
        Ok(self.scores.clone())
    }

    /// Returns fitted hub or authority scores.
    ///
    /// When `columns` is true, returns authority scores for column nodes.
    pub fn predict(&self, columns: bool) -> Vec<f64> {
        if columns {
            self.scores_col.clone()
        } else {
            self.scores.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::test_bigraph;

    #[test]
    fn test_keywords() {
        let biadjacency = test_bigraph();
        let (n_row, n_col) = biadjacency.shape();
        let mut hits = HITS::default();
        hits.fit(&biadjacency).unwrap();
        assert_eq!(hits.scores_row.len(), n_row);
        assert_eq!(hits.scores_col.len(), n_col);
    }
}
