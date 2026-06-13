use rand::seq::SliceRandom;
use rand::SeedableRng;
use sprs::CsMat;

use crate::path::distances::get_distances;
use crate::utils::check::check_connected;

/// Errors raised while computing closeness centrality.
#[derive(Debug, Clone, PartialEq)]
pub enum ClosenessError {
    /// The method name is not `exact` or `approximate`.
    InvalidMethod,
    /// The graph is not connected as an undirected graph.
    NotConnected,
    /// The adjacency matrix is not square or distance computation failed.
    InvalidGraph,
}

/// Closeness centrality estimator with exact and approximate modes.
#[derive(Debug, Clone)]
pub struct Closeness {
    /// Computation mode: `exact` or `approximate`.
    pub method: String,
    /// Tolerance controlling the number of sources in approximate mode.
    pub tol: f64,
    /// Optional seed for reproducible approximate source sampling.
    pub random_state: Option<u64>,
    /// Fitted closeness scores, one per node.
    pub scores: Vec<f64>,
}

impl Default for Closeness {
    fn default() -> Self {
        Self {
            method: "exact".to_string(),
            tol: 1e-1,
            random_state: None,
            scores: Vec::new(),
        }
    }
}

impl Closeness {
    /// Creates a closeness estimator with the given method and tolerance.
    ///
    /// Method names are case-insensitive.
    pub fn new(method: &str, tol: f64) -> Self {
        Self {
            method: method.to_lowercase(),
            tol,
            ..Self::default()
        }
    }

    /// Sets the random seed used for approximate source sampling.
    pub fn with_random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Computes closeness scores from a square adjacency matrix.
    ///
    /// # Errors
    /// Returns [`ClosenessError::InvalidGraph`] when the input is not square,
    /// [`ClosenessError::NotConnected`] when the graph is disconnected, and
    /// [`ClosenessError::InvalidMethod`] for unsupported method names.
    pub fn fit(&mut self, adjacency: &CsMat<f64>) -> Result<(), ClosenessError> {
        if adjacency.rows() != adjacency.cols() {
            return Err(ClosenessError::InvalidGraph);
        }
        check_connected(adjacency).map_err(|_| ClosenessError::NotConnected)?;
        let n = adjacency.rows();

        let sources: Vec<usize> = if self.method == "exact" {
            (0..n).collect()
        } else if self.method == "approximate" {
            let target = ((n as f64).ln() / (self.tol * self.tol)).floor() as usize;
            let n_sources = target.min(n);
            let mut candidates: Vec<usize> = (0..n).collect();
            if let Some(seed) = self.random_state {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                candidates.shuffle(&mut rng);
            } else {
                let mut rng = rand::rng();
                candidates.shuffle(&mut rng);
            }
            candidates.into_iter().take(n_sources).collect()
        } else {
            return Err(ClosenessError::InvalidMethod);
        };

        let mut out = vec![0.0; n];
        for &source in &sources {
            let distances =
                get_distances(adjacency, Some(source)).map_err(|_| ClosenessError::InvalidGraph)?;
            if distances.iter().any(|&d| d < 0) {
                out[source] = 0.0;
                continue;
            }
            let mean = distances.iter().map(|&d| d as f64).sum::<f64>() / n as f64;
            out[source] = if mean > 0.0 {
                (n as f64 - 1.0) / n as f64 / mean
            } else {
                0.0
            };
        }
        if self.method == "approximate" && !sources.is_empty() {
            let mean_sampled = sources.iter().map(|&i| out[i]).sum::<f64>() / sources.len() as f64;
            let mut sampled = vec![false; n];
            for &i in &sources {
                sampled[i] = true;
            }
            for (i, score) in out.iter_mut().enumerate().take(n) {
                if !sampled[i] {
                    *score = mean_sampled;
                }
            }
        }
        self.scores = out;
        Ok(())
    }

    /// Fits the estimator and returns a copy of the closeness scores.
    ///
    /// # Errors
    /// Returns the same errors as [`Closeness::fit`].
    pub fn fit_predict(&mut self, adjacency: &CsMat<f64>) -> Result<Vec<f64>, ClosenessError> {
        self.fit(adjacency)?;
        Ok(self.scores.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::{test_disconnected_graph, test_graph};

    #[test]
    fn test_params() {
        let adjacency = test_graph();
        let mut closeness = Closeness::new("toto", 1e-1);
        assert_eq!(
            closeness.fit(&adjacency),
            Err(ClosenessError::InvalidMethod)
        );
    }

    #[test]
    fn test_parallel() {
        let adjacency = test_graph();
        let n = adjacency.rows();
        let mut closeness = Closeness::new("approximate", 1e-1);
        let scores = closeness.fit_predict(&adjacency).unwrap();
        assert_eq!(scores.len(), n);
    }

    #[test]
    fn test_approximate_seed_reproducible() {
        let adjacency = test_graph();
        let mut c1 = Closeness::new("approximate", 1e-1).with_random_state(Some(7));
        let mut c2 = Closeness::new("approximate", 1e-1).with_random_state(Some(7));
        let s1 = c1.fit_predict(&adjacency).unwrap();
        let s2 = c2.fit_predict(&adjacency).unwrap();
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_method_case_insensitive() {
        let adjacency = test_graph();
        let mut closeness = Closeness::new("ApPrOxImAtE", 1e-1).with_random_state(Some(3));
        let scores = closeness.fit_predict(&adjacency).unwrap();
        assert_eq!(scores.len(), adjacency.rows());
    }

    #[test]
    fn test_disconnected() {
        let adjacency = test_disconnected_graph();
        let mut closeness = Closeness::default();
        assert_eq!(closeness.fit(&adjacency), Err(ClosenessError::NotConnected));
    }
}
