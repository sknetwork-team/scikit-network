use std::collections::VecDeque;

use sprs::CsMat;

/// Errors raised while computing betweenness centrality.
#[derive(Debug, Clone, PartialEq)]
pub enum BetweennessError {
    /// The adjacency matrix is not square.
    InvalidGraph,
    /// The graph is not connected as an undirected graph.
    DisconnectedGraph,
}

/// Betweenness centrality estimator on connected undirected graphs.
#[derive(Debug, Clone)]
pub struct Betweenness {
    /// Whether to normalize scores to the unit interval.
    pub normalized: bool,
    /// Fitted betweenness scores, one per node.
    pub scores: Vec<f64>,
}

impl Default for Betweenness {
    fn default() -> Self {
        Self {
            normalized: false,
            scores: Vec::new(),
        }
    }
}

impl Betweenness {
    /// Creates a betweenness estimator with optional score normalization.
    pub fn new(normalized: bool) -> Self {
        Self {
            normalized,
            ..Self::default()
        }
    }

    /// Computes betweenness scores from a square adjacency matrix.
    ///
    /// # Errors
    /// Returns [`BetweennessError::InvalidGraph`] when the input is not square
    /// and [`BetweennessError::DisconnectedGraph`] when the graph is not
    /// connected.
    pub fn fit(&mut self, adjacency: &CsMat<f64>) -> Result<(), BetweennessError> {
        if adjacency.rows() != adjacency.cols() {
            return Err(BetweennessError::InvalidGraph);
        }
        if !is_connected(adjacency) {
            return Err(BetweennessError::DisconnectedGraph);
        }
        let n = adjacency.rows();
        let mut scores = vec![0.0f64; n];

        for source in 0..n {
            let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut sigma = vec![0.0f64; n];
            sigma[source] = 1.0;
            let mut dist = vec![-1i32; n];
            dist[source] = 0;

            let mut queue = VecDeque::new();
            queue.push_back(source);
            let mut stack: Vec<usize> = Vec::new();

            while let Some(v) = queue.pop_front() {
                stack.push(v);
                if let Some(row) = adjacency.outer_view(v) {
                    for &w in row.indices() {
                        if dist[w] < 0 {
                            dist[w] = dist[v] + 1;
                            queue.push_back(w);
                        }
                        if dist[w] == dist[v] + 1 {
                            sigma[w] += sigma[v];
                            preds[w].push(v);
                        }
                    }
                }
            }

            let mut delta = vec![0.0f64; n];
            while let Some(w) = stack.pop() {
                for &v in &preds[w] {
                    if sigma[w] > 0.0 {
                        delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w]);
                    }
                }
                if w != source {
                    scores[w] += delta[w];
                }
            }
        }

        for x in &mut scores {
            *x *= 0.5;
        }
        if self.normalized && n > 2 {
            let scale = 2.0 / ((n as f64 - 1.0) * (n as f64 - 2.0));
            for x in &mut scores {
                *x *= scale;
            }
        }
        self.scores = scores;
        Ok(())
    }

    /// Fits the estimator and returns a copy of the betweenness scores.
    ///
    /// # Errors
    /// Returns the same errors as [`Betweenness::fit`].
    pub fn fit_predict(&mut self, adjacency: &CsMat<f64>) -> Result<Vec<f64>, BetweennessError> {
        self.fit(adjacency)?;
        Ok(self.scores.clone())
    }
}

fn is_connected(adjacency: &CsMat<f64>) -> bool {
    let n = adjacency.rows();
    if n <= 1 {
        return true;
    }
    let mut seen = vec![false; n];
    let mut queue = VecDeque::new();
    queue.push_back(0usize);
    seen[0] = true;
    while let Some(u) = queue.pop_front() {
        if let Some(row) = adjacency.outer_view(u) {
            for &v in row.indices() {
                if !seen[v] {
                    seen[v] = true;
                    queue.push_back(v);
                }
            }
        }
    }
    seen.into_iter().all(|x| x)
}

#[cfg(test)]
mod tests {
    use sprs::TriMat;

    use super::*;
    use crate::data::test_graphs::{test_bigraph, test_disconnected_graph};

    fn test_graph() -> CsMat<f64> {
        let mut tri = TriMat::<f64>::new((6, 6));
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 4)];
        for (u, v) in edges {
            tri.add_triplet(u, v, 1.0);
            tri.add_triplet(v, u, 1.0);
        }
        tri.to_csr::<usize>()
    }

    fn bow_tie() -> CsMat<f64> {
        let mut tri = TriMat::<f64>::new((5, 5));
        let edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)];
        for (u, v) in edges {
            tri.add_triplet(u, v, 1.0);
            tri.add_triplet(v, u, 1.0);
        }
        tri.to_csr::<usize>()
    }

    #[test]
    fn test_basic() {
        let adjacency = test_graph();
        let mut b = Betweenness::default();
        let scores = b.fit_predict(&adjacency).unwrap();
        assert_eq!(scores.len(), adjacency.rows());
    }

    #[test]
    fn test_bowtie() {
        let adjacency = bow_tie();
        let mut b = Betweenness::default();
        let scores = b.fit_predict(&adjacency).unwrap();
        let positive = scores.iter().filter(|&&x| x > 0.0).count();
        assert_eq!(positive, 1);
    }

    #[test]
    fn test_disconnected() {
        let adjacency = test_disconnected_graph();
        let mut b = Betweenness::default();
        assert_eq!(b.fit(&adjacency), Err(BetweennessError::DisconnectedGraph));
    }

    #[test]
    fn test_bipartite() {
        let adjacency = test_bigraph();
        let mut b = Betweenness::default();
        assert_eq!(
            b.fit_predict(&adjacency),
            Err(BetweennessError::InvalidGraph)
        );
    }
}
