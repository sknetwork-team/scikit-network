use sprs::CsMat;

use crate::utils::check::{check_square, is_symmetric};
use crate::utils::format::directed2undirected;
use crate::utils::spatial_index::{KDTreeRef, brute_radius_into};

/// Errors raised by [`Spring`] layout.
#[derive(Debug, Clone, PartialEq)]
pub enum SpringError {
    /// `position_init` is not `random` or `spectral`.
    UnknownPositionInit,
    /// `position_init` array shape does not match the graph.
    InvalidPositionInitShape,
    /// Reserved for incompatible init array dtypes.
    InvalidPositionInitType,
    /// The adjacency matrix is not square.
    InvalidGraph,
    /// [`Spring::transform`] called before `fit`.
    NotFitted,
}

/// Spring-electrical force-directed layout estimator.
#[derive(Debug, Clone)]
pub struct Spring {
    /// Layout dimensionality.
    pub n_components: usize,
    /// Edge attraction strength (`None` uses `1/√n`).
    pub strength: Option<f64>,
    /// Maximum number of layout iterations.
    pub n_iter: usize,
    /// Mean displacement tolerance for early stopping.
    pub tol: f64,
    /// Approximate repulsion radius (`< 0` uses all-pairs repulsion).
    pub approx_radius: f64,
    /// Initial position mode: `random` or `spectral`.
    pub position_init: String,
    /// Fitted node positions after `fit`.
    pub embedding: Vec<Vec<f64>>,
}

impl Default for Spring {
    fn default() -> Self {
        Self::new(2, None, 50, 1e-4, -1.0, "random").unwrap()
    }
}

impl Spring {
    const KD_TREE_MIN_NODES: usize = 128;
    /// Creates a spring-layout estimator.
    ///
    /// # Errors
    /// Returns [`SpringError::UnknownPositionInit`] when `position_init` is not
    /// `random` or `spectral`.
    pub fn new(
        n_components: usize,
        strength: Option<f64>,
        n_iter: usize,
        tol: f64,
        approx_radius: f64,
        position_init: &str,
    ) -> Result<Self, SpringError> {
        let position_init = position_init.to_lowercase();
        if position_init != "random" && position_init != "spectral" {
            return Err(SpringError::UnknownPositionInit);
        }
        Ok(Self {
            n_components,
            strength,
            n_iter,
            tol,
            approx_radius,
            position_init,
            embedding: Vec::new(),
        })
    }

    fn deterministic_positions(n: usize, d: usize, spectral_like: bool) -> Vec<Vec<f64>> {
        let mut pos = vec![vec![0.0; d]; n];
        for i in 0..n {
            for j in 0..d {
                let x = if spectral_like {
                    ((i + 1) as f64 * (j + 2) as f64).cos()
                } else {
                    ((i + 3) as f64 * (j + 5) as f64).sin()
                };
                pos[i][j] = x;
            }
        }
        pos
    }

    /// Computes one force-accumulation step for the spring layout.
    ///
    /// Invariants:
    /// - `pos.len() == adjacency.rows()`.
    /// - Every position row has dimensionality `d`.
    /// - When `approx_radius > 0`, only neighbors within that radius are used
    ///   for repulsion.
    fn spring_step(
        adjacency: &CsMat<f64>,
        pos: &[Vec<f64>],
        strength: f64,
        approx_radius: f64,
        tree: Option<&KDTreeRef<'_>>,
    ) -> Vec<Vec<f64>> {
        let n = adjacency.rows();
        let d = if pos.is_empty() { 0 } else { pos[0].len() };
        let mut delta = vec![vec![0.0; d]; n];
        let use_radius = approx_radius > 0.0;
        let radius2 = approx_radius * approx_radius;
        let mut neighbors = Vec::<usize>::new();
        for i in 0..n {
            // attraction from neighbors
            if let Some(row) = adjacency.outer_view(i) {
                for &j in row.indices() {
                    for c in 0..d {
                        delta[i][c] += (pos[j][c] - pos[i][c]) * 0.01 / strength.max(1e-6);
                    }
                }
            }
            // weak repulsion from all nodes
            if use_radius {
                if let Some(t) = tree {
                    t.radius_query_into(&pos[i], approx_radius, &mut neighbors);
                } else {
                    brute_radius_into(pos, &pos[i], approx_radius, &mut neighbors);
                }
                for &j in &neighbors {
                    if i == j {
                        continue;
                    }
                    let mut dist2 = 0.0;
                    for c in 0..d {
                        let g = pos[i][c] - pos[j][c];
                        dist2 += g * g;
                    }
                    if dist2 > radius2 {
                        continue;
                    }
                    let dist2 = dist2.max(1e-4);
                    for c in 0..d {
                        delta[i][c] += (pos[i][c] - pos[j][c]) * strength * 0.001 / dist2;
                    }
                }
            } else {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let mut dist2 = 0.0;
                    for c in 0..d {
                        let g = pos[i][c] - pos[j][c];
                        dist2 += g * g;
                    }
                    let dist2 = dist2.max(1e-4);
                    for c in 0..d {
                        delta[i][c] += (pos[i][c] - pos[j][c]) * strength * 0.001 / dist2;
                    }
                }
            }
        }
        delta
    }

    /// Fits a 2D/ND layout from an adjacency matrix.
    ///
    /// # Errors
    /// Returns:
    /// - [`SpringError::InvalidGraph`] for non-square inputs
    /// - [`SpringError::InvalidPositionInitShape`] for incompatible init shape
    /// - [`SpringError::UnknownPositionInit`] for invalid init mode
    pub fn fit(
        &mut self,
        adjacency: &CsMat<f64>,
        position_init: Option<&Vec<Vec<f64>>>,
        n_iter: Option<usize>,
    ) -> Result<(), SpringError> {
        check_square(adjacency.shape()).map_err(|_| SpringError::InvalidGraph)?;
        let adjacency = if is_symmetric(adjacency) {
            adjacency.to_owned()
        } else {
            directed2undirected(adjacency, true)
        };
        let n = adjacency.rows();
        let d = self.n_components;

        let mut pos = if let Some(init) = position_init {
            if init.len() != n || (!init.is_empty() && init[0].len() != d) {
                return Err(SpringError::InvalidPositionInitShape);
            }
            init.clone()
        } else {
            match self.position_init.as_str() {
                "spectral" => Self::deterministic_positions(n, d, true),
                "random" => Self::deterministic_positions(n, d, false),
                _ => return Err(SpringError::UnknownPositionInit),
            }
        };

        let iters = n_iter.unwrap_or(self.n_iter);
        let strength = self.strength.unwrap_or((1.0 / n as f64).sqrt());

        for _ in 0..iters {
            let tree = if self.approx_radius > 0.0 && d <= 16 && n >= Self::KD_TREE_MIN_NODES {
                KDTreeRef::build(&pos)
            } else {
                None
            };
            let delta = Self::spring_step(&adjacency, &pos, strength, self.approx_radius, tree.as_ref());
            let mut err = 0.0;
            for i in 0..n {
                for c in 0..d {
                    pos[i][c] += delta[i][c];
                    err += delta[i][c].abs();
                }
            }
            if err / (n.max(1) as f64) < self.tol {
                break;
            }
        }

        self.embedding = pos;
        Ok(())
    }

    /// Fits the layout and returns node positions.
    ///
    /// # Errors
    /// Returns the same errors as [`Spring::fit`].
    pub fn fit_transform(
        &mut self,
        adjacency: &CsMat<f64>,
        position_init: Option<&Vec<Vec<f64>>>,
        n_iter: Option<usize>,
    ) -> Result<Vec<Vec<f64>>, SpringError> {
        self.fit(adjacency, position_init, n_iter)?;
        Ok(self.embedding.clone())
    }

    /// Returns the fitted layout.
    ///
    /// # Errors
    /// Returns [`SpringError::NotFitted`] when called before `fit`.
    pub fn transform(&self) -> Result<Vec<Vec<f64>>, SpringError> {
        if self.embedding.is_empty() {
            return Err(SpringError::NotFitted);
        }
        Ok(self.embedding.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::{test_digraph, test_graph};

    #[test]
    fn test_shape() {
        for adjacency in [test_graph(), test_digraph()] {
            let n = adjacency.rows();
            let mut spring = Spring::default();
            let layout = spring.fit_transform(&adjacency, None, None).unwrap();
            assert_eq!((layout.len(), layout[0].len()), (n, 2));

            let mut spring = Spring::new(3, None, 50, 1e-4, -1.0, "random").unwrap();
            let layout = spring.fit_transform(&adjacency, None, None).unwrap();
            assert_eq!((layout.len(), layout[0].len()), (n, 3));
        }
    }

    #[test]
    fn test_pos_init() {
        let adjacency = test_graph();
        let n = adjacency.rows();
        let mut spring = Spring::new(2, Some(0.1), 50, 1e3, -1.0, "spectral").unwrap();
        let layout = spring.fit_transform(&adjacency, None, None).unwrap();
        assert_eq!((layout.len(), layout[0].len()), (n, 2));
        let layout2 = spring
            .fit_transform(&adjacency, Some(&layout), None)
            .unwrap();
        assert_eq!((layout2.len(), layout2[0].len()), (n, 2));
    }

    #[test]
    fn test_approx_radius() {
        let adjacency = test_graph();
        let n = adjacency.rows();
        let mut spring = Spring::new(2, None, 50, 1e-4, 1.0, "random").unwrap();
        let layout = spring.fit_transform(&adjacency, None, None).unwrap();
        assert_eq!((layout.len(), layout[0].len()), (n, 2));
    }

    #[test]
    fn test_errors() {
        assert!(matches!(
            Spring::new(2, None, 50, 1e-4, -1.0, "toto"),
            Err(SpringError::UnknownPositionInit)
        ));
        let adjacency = test_graph();
        let mut spring = Spring::default();
        let bad = vec![vec![1.0, 1.0]; 2];
        assert_eq!(
            spring.fit(&adjacency, Some(&bad), None),
            Err(SpringError::InvalidPositionInitShape)
        );
        spring.position_init = "bad-init".to_string();
        assert_eq!(
            spring.fit(&adjacency, None, None),
            Err(SpringError::UnknownPositionInit)
        );
    }

    #[test]
    fn test_position_init_case_insensitive() {
        let adjacency = test_graph();
        let mut spring = Spring::new(2, None, 5, 1e-4, -1.0, "SpEcTrAl").unwrap();
        let layout = spring.fit_transform(&adjacency, None, None).unwrap();
        assert_eq!(layout.len(), adjacency.rows());
    }
}
