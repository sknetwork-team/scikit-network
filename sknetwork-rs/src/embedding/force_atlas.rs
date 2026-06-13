use sprs::CsMat;
use rand::Rng;

use crate::utils::check::{check_square, is_symmetric};
use crate::utils::format::directed2undirected;
use crate::utils::spatial_index::{KDTreeRef, brute_radius_into};

/// Errors raised by [`ForceAtlas`] layout.
#[derive(Debug, Clone, PartialEq)]
pub enum ForceAtlasError {
    /// The adjacency matrix is not square.
    InvalidGraph,
    /// Initial position dimensions do not match the graph size.
    InvalidInitDimensions,
}

/// ForceAtlas2 force-directed graph layout estimator.
#[derive(Debug, Clone)]
pub struct ForceAtlas {
    /// Layout dimensionality.
    pub n_components: usize,
    /// Maximum number of layout iterations.
    pub n_iter: usize,
    /// Approximate repulsion radius (`< 0` uses all-pairs repulsion).
    pub approx_radius: f64,
    /// Use lin-log attraction when true.
    pub lin_log: bool,
    /// Gravity pull toward the origin.
    pub gravity_factor: f64,
    /// Repulsive force scale between node pairs.
    pub repulsive_factor: f64,
    /// Global speed tolerance for adaptive iteration.
    pub tolerance: f64,
    /// Base node displacement speed.
    pub speed: f64,
    /// Maximum per-node speed cap.
    pub speed_max: f64,
    /// Fitted node positions after `fit`.
    pub embedding: Vec<Vec<f64>>,
}

impl Default for ForceAtlas {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_iter: 50,
            approx_radius: -1.0,
            lin_log: false,
            gravity_factor: 0.01,
            repulsive_factor: 0.1,
            tolerance: 0.1,
            speed: 0.1,
            speed_max: 10.0,
            embedding: Vec::new(),
        }
    }
}

impl ForceAtlas {
    const KD_TREE_MIN_NODES: usize = 128;

    /// Creates a ForceAtlas2 layout estimator with explicit hyperparameters.
    ///
    /// # Arguments
    /// - `n_components`: Layout dimensionality.
    /// - `n_iter`: Maximum number of layout iterations.
    /// - `approx_radius`: Approximate repulsion radius (`< 0` uses all pairs).
    /// - `lin_log`: Use lin-log attraction when true.
    /// - `gravity_factor`: Gravity pull toward the origin.
    /// - `repulsive_factor`: Repulsive force scale.
    /// - `tolerance`: Global speed tolerance for adaptive iteration.
    /// - `speed`: Base node displacement speed.
    /// - `speed_max`: Maximum per-node speed cap.
    pub fn new(
        n_components: usize,
        n_iter: usize,
        approx_radius: f64,
        lin_log: bool,
        gravity_factor: f64,
        repulsive_factor: f64,
        tolerance: f64,
        speed: f64,
        speed_max: f64,
    ) -> Self {
        Self {
            n_components,
            n_iter,
            approx_radius,
            lin_log,
            gravity_factor,
            repulsive_factor,
            tolerance,
            speed,
            speed_max,
            ..Self::default()
        }
    }

    fn random_normal_positions(n: usize, d: usize) -> Vec<Vec<f64>> {
        let mut rng = rand::rng();
        let mut p = vec![vec![0.0; d]; n];
        let mut spare: Option<f64> = None;
        for row in &mut p {
            for x in row.iter_mut() {
                let z = if let Some(v) = spare.take() {
                    v
                } else {
                    let mut u1 = rng.random::<f64>();
                    while u1 <= f64::MIN_POSITIVE {
                        u1 = rng.random::<f64>();
                    }
                    let u2 = rng.random::<f64>();
                    let r = (-2.0 * u1.ln()).sqrt();
                    let theta = std::f64::consts::TAU * u2;
                    let z0 = r * theta.cos();
                    let z1 = r * theta.sin();
                    spare = Some(z1);
                    z0
                };
                *x = z;
            }
        }
        p
    }

    /// Fits node positions from an adjacency matrix.
    ///
    /// # Arguments
    /// - `adjacency`: Square sparse adjacency matrix.
    /// - `pos_init`: Optional initial positions (`n × n_components`).
    /// - `n_iter`: Optional iteration override.
    ///
    /// # Errors
    /// Returns:
    /// - [`ForceAtlasError::InvalidGraph`] when the matrix is not square.
    /// - [`ForceAtlasError::InvalidInitDimensions`] when `pos_init` shape mismatches.
    pub fn fit(
        &mut self,
        adjacency: &CsMat<f64>,
        pos_init: Option<&Vec<Vec<f64>>>,
        n_iter: Option<usize>,
    ) -> Result<(), ForceAtlasError> {
        check_square(adjacency.shape()).map_err(|_| ForceAtlasError::InvalidGraph)?;
        let adjacency = if is_symmetric(adjacency) {
            adjacency.to_owned()
        } else {
            directed2undirected(adjacency, true)
        };
        let n = adjacency.rows();
        let d = self.n_components;
        let mut pos = if let Some(init) = pos_init {
            if init.len() != n || (!init.is_empty() && init[0].len() != d) {
                return Err(ForceAtlasError::InvalidInitDimensions);
            }
            init.clone()
        } else {
            Self::random_normal_positions(n, d)
        };

        let iters = n_iter.unwrap_or(self.n_iter);
        let tolerance = if n < 5000 {
            0.1
        } else if n < 50000 {
            1.0
        } else {
            10.0
        };
        let mut resultants = vec![0.0; n];
        let mut swing_vector = vec![0.0; n];
        let mut global_speed = 1.0;
        let radius2 = self.approx_radius * self.approx_radius;
        let mut degree_plus_one = vec![1.0; n];
        for (i, degree) in degree_plus_one.iter_mut().enumerate().take(n) {
            *degree = adjacency.outer_view(i).map(|r| r.nnz() as f64).unwrap_or(0.0) + 1.0;
        }

        for _ in 0..iters {
            let mut delta = vec![vec![0.0; d]; n];
            let mut global_swing = 0.0;
            let mut global_traction = 0.0;
            let tree = if self.approx_radius > 0.0 && d <= 16 && n >= Self::KD_TREE_MIN_NODES {
                KDTreeRef::build(&pos)
            } else {
                None
            };
            let mut neighbors = Vec::<usize>::new();

            for i in 0..n {
                let degree_i = degree_plus_one[i];
                // attraction
                if let Some(row) = adjacency.outer_view(i) {
                    for &j in row.indices() {
                        for c in 0..d {
                            let mut g = pos[i][c] - pos[j][c];
                            if self.lin_log {
                                g = g.signum() * (1.0 + 10.0 * g.abs()).ln();
                            }
                            delta[i][c] -= g;
                        }
                    }
                }
                // repulsion + gravity
                if self.approx_radius > 0.0 {
                    if let Some(t) = &tree {
                        t.radius_query_into(&pos[i], self.approx_radius, &mut neighbors);
                    } else {
                        brute_radius_into(&pos, &pos[i], self.approx_radius, &mut neighbors);
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
                        let dist = dist2.sqrt().max(0.01);
                        let degree_j = degree_plus_one[j];
                        for c in 0..d {
                            let g = pos[i][c] - pos[j][c];
                            delta[i][c] +=
                                self.repulsive_factor * degree_i * degree_j * g / dist.max(0.01);
                            delta[i][c] -= self.gravity_factor * degree_i * g;
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
                        let dist = dist2.sqrt().max(0.01);
                        let degree_j = degree_plus_one[j];
                        for c in 0..d {
                            let g = pos[i][c] - pos[j][c];
                            delta[i][c] +=
                                self.repulsive_factor * degree_i * degree_j * g / dist.max(0.01);
                            delta[i][c] -= self.gravity_factor * degree_i * g;
                        }
                    }
                }

                let resultant_new = delta[i].iter().map(|x| x * x).sum::<f64>().sqrt();
                let resultant_old = resultants[i];
                let swing_node = (resultant_new - resultant_old).abs();
                swing_vector[i] = swing_node;
                global_swing += (degree_i + 1.0) * swing_node;
                let traction = (resultant_new + resultants[i]).abs() / 2.0;
                global_traction += (degree_i + 1.0) * traction;
                let mut node_speed =
                    self.speed * global_speed / (1.0 + global_speed * swing_node.max(0.0).sqrt());
                if resultant_new > 0.0 {
                    node_speed = node_speed.min(self.speed_max / resultant_new);
                }
                if !node_speed.is_finite() || node_speed < 0.0 {
                    node_speed = 0.0;
                }
                for c in 0..d {
                    delta[i][c] *= node_speed;
                }
                resultants[i] = resultant_new;
                if global_swing > 0.0 {
                    global_speed = tolerance * global_traction / global_swing;
                    if !global_speed.is_finite() || global_speed <= 0.0 {
                        global_speed = 1.0;
                    }
                }
            }

            let mut all_small = true;
            for i in 0..n {
                let move_norm = delta[i].iter().map(|x| x * x).sum::<f64>().sqrt();
                if move_norm >= 1.0 {
                    all_small = false;
                }
                for c in 0..d {
                    pos[i][c] += delta[i][c];
                }
            }
            if all_small {
                break;
            }
        }

        self.embedding = pos;
        Ok(())
    }

    /// Fits the layout and returns node positions.
    ///
    /// # Errors
    /// Returns the same errors as [`ForceAtlas::fit`].
    pub fn fit_transform(
        &mut self,
        adjacency: &CsMat<f64>,
        pos_init: Option<&Vec<Vec<f64>>>,
        n_iter: Option<usize>,
    ) -> Result<Vec<Vec<f64>>, ForceAtlasError> {
        self.fit(adjacency, pos_init, n_iter)?;
        Ok(self.embedding.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::{test_digraph, test_graph};

    #[test]
    fn test_options() {
        for adjacency in [test_graph(), test_digraph()] {
            let n = adjacency.rows();
            let mut fa = ForceAtlas::default();
            let layout = fa.fit_transform(&adjacency, None, None).unwrap();
            assert_eq!((layout.len(), layout[0].len()), (n, 2));

            let mut fa = ForceAtlas::new(2, 50, -1.0, true, 0.01, 0.1, 0.1, 0.1, 10.0);
            let layout = fa.fit_transform(&adjacency, None, None).unwrap();
            assert_eq!((layout.len(), layout[0].len()), (n, 2));

            let mut fa = ForceAtlas::new(2, 50, 1.0, false, 0.01, 0.1, 0.1, 0.1, 10.0);
            let layout = fa.fit_transform(&adjacency, None, None).unwrap();
            assert_eq!((layout.len(), layout[0].len()), (n, 2));

            fa.fit(&adjacency, Some(&layout), Some(1)).unwrap();
        }
    }

    #[test]
    fn test_errors() {
        let adjacency = test_graph();
        let mut fa = ForceAtlas::default();
        let bad = vec![vec![1.0; 7]; 5];
        assert_eq!(
            fa.fit(&adjacency, Some(&bad), None),
            Err(ForceAtlasError::InvalidInitDimensions)
        );
    }
}
