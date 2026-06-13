//! K-centers clustering via PageRank-based center selection.

use ndarray::Array1;
use sprs::CsMat;

use crate::clustering::metrics::get_modularity;
use crate::ranking::pagerank::PageRank;
use crate::utils::format::{MatrixInput, directed2undirected, get_adjacency};
use crate::utils::values::ValuesInput;

/// Error type for [`KCenters`] fitting and prediction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KCentersError {
    /// `n_clusters` must be at least 2.
    InvalidNClusters,
    /// `n_init` must be at least 1.
    InvalidNInit,
    /// `center_position` is not one of `row`, `col`, or `both`.
    UnknownCenterPosition,
    /// Requested cluster count exceeds eligible center nodes.
    TooManyClusters,
    /// Input matrix formatting or PageRank computation failed.
    InvalidInput,
}

/// K-centers clustering estimator using PageRank for center updates.
#[derive(Debug, Clone)]
pub struct KCenters {
    /// Target number of clusters.
    pub n_clusters: usize,
    /// Whether to symmetrize directed inputs before clustering.
    pub directed: bool,
    /// Bipartite center mask policy (`row`, `col`, or `both`).
    pub center_position: String,
    /// Number of random restarts kept by best modularity score.
    pub n_init: usize,
    /// Maximum Lloyd-style iterations per restart.
    pub max_iter: usize,
    /// Whether the last fit used a bipartite input.
    pub bipartite: bool,
    /// Fitted cluster labels for all nodes.
    pub labels: Option<Array1<i32>>,
    /// Fitted labels for bipartite row nodes.
    pub labels_row: Option<Array1<i32>>,
    /// Fitted labels for bipartite column nodes.
    pub labels_col: Option<Array1<i32>>,
    /// Selected center node indices.
    pub centers: Option<Vec<usize>>,
    /// Center indices restricted to row nodes.
    pub centers_row: Option<Vec<usize>>,
    /// Center indices restricted to column nodes.
    pub centers_col: Option<Vec<usize>>,
}

impl KCenters {
    /// Creates a K-centers estimator with explicit hyperparameters.
    ///
    /// # Arguments
    /// - `center_position`: Bipartite center mask (`row`, `col`, or `both`).
    pub fn new(
        n_clusters: usize,
        directed: bool,
        center_position: &str,
        n_init: usize,
        max_iter: usize,
    ) -> Self {
        Self {
            n_clusters,
            directed,
            center_position: center_position.to_string(),
            n_init,
            max_iter,
            bipartite: false,
            labels: None,
            labels_row: None,
            labels_col: None,
            centers: None,
            centers_row: None,
            centers_col: None,
        }
    }

    fn compute_mask_centers(
        &self,
        input_shape: (usize, usize),
        bipartite: bool,
    ) -> Result<Vec<bool>, KCentersError> {
        let (n_row, n_col) = input_shape;
        if bipartite {
            let mut mask = vec![false; n_row + n_col];
            match self.center_position.as_str() {
                "row" => mask.iter_mut().take(n_row).for_each(|x| *x = true),
                "col" => mask.iter_mut().skip(n_row).for_each(|x| *x = true),
                "both" => mask.iter_mut().for_each(|x| *x = true),
                _ => return Err(KCentersError::UnknownCenterPosition),
            }
            Ok(mask)
        } else {
            Ok(vec![true; n_row])
        }
    }

    fn init_centers(
        &self,
        adjacency: &CsMat<f64>,
        mask: &[bool],
        n_clusters: usize,
    ) -> Result<Vec<usize>, KCentersError> {
        let candidates: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
        if candidates.len() < n_clusters {
            return Err(KCentersError::TooManyClusters);
        }
        let mut candidate_mask = mask.to_vec();
        let mut centers = vec![candidates[0]];
        candidate_mask[candidates[0]] = false;
        let mut center_weights = vec![0.0; adjacency.rows()];
        center_weights[candidates[0]] = 1.0;
        let mut pagerank = PageRank::default();

        while centers.len() < n_clusters {
            let scores = pagerank
                .fit_predict(
                    adjacency,
                    Some(ValuesInput::Vector(center_weights.clone())),
                    None,
                    None,
                    false,
                )
                .map_err(|_| KCentersError::InvalidInput)?;
            let available: Vec<usize> = (0..candidate_mask.len()).filter(|&i| candidate_mask[i]).collect();
            let zero_scores: Vec<usize> = available
                .iter()
                .copied()
                .filter(|&u| scores[u] == 0.0)
                .collect();
            let next_center = if let Some(&u) = zero_scores.first() {
                u
            } else {
                let mut best = available[0];
                let mut best_inv = f64::NEG_INFINITY;
                for &u in &available {
                    if scores[u] > 0.0 {
                        let inv = 1.0 / scores[u];
                        if inv > best_inv {
                            best_inv = inv;
                            best = u;
                        }
                    }
                }
                best
            };
            candidate_mask[next_center] = false;
            if center_weights[next_center] == 0.0 {
                center_weights[next_center] = 1.0;
            }
            centers.push(next_center);
        }
        Ok(centers)
    }

    fn assign_labels(adjacency: &CsMat<f64>, centers: &[usize]) -> Result<Vec<i32>, KCentersError> {
        let n = adjacency.rows();
        let mut labels = vec![0i32; n];
        let mut score_by_center = Vec::<Vec<f64>>::with_capacity(centers.len());
        for &center in centers {
            let mut w = vec![0.0; n];
            w[center] = 1.0;
            let mut pagerank = PageRank::default();
            let scores = pagerank
                .fit_predict(adjacency, Some(ValuesInput::Vector(w)), None, None, false)
                .map_err(|_| KCentersError::InvalidInput)?;
            score_by_center.push(scores);
        }
        for (u, lbl) in labels.iter_mut().enumerate().take(n) {
            let mut best = 0usize;
            let mut best_score = f64::NEG_INFINITY;
            for (k, scores) in score_by_center.iter().enumerate() {
                let s = scores[u];
                if s > best_score || (s == best_score && centers[k] < centers[best]) {
                    best_score = s;
                    best = k;
                }
            }
            *lbl = best as i32;
        }
        Ok(labels)
    }

    fn update_centers(
        &self,
        adjacency: &CsMat<f64>,
        labels: &[i32],
        mask: &[bool],
    ) -> Result<Vec<usize>, KCentersError> {
        let n = adjacency.rows();
        let mut centers = vec![0usize; self.n_clusters];
        let mut used = Vec::<usize>::new();
        for (k, center_slot) in centers.iter_mut().enumerate().take(self.n_clusters) {
            let members: Vec<usize> = (0..n).filter(|&u| labels[u] == k as i32).collect();
            if members.is_empty() {
                if let Some(node) = (0..n).find(|&u| mask[u] && !used.contains(&u)) {
                    *center_slot = node;
                    used.push(node);
                }
                continue;
            }
            let mut cluster_weights = vec![0.0; n];
            for &u in &members {
                if mask[u] {
                    cluster_weights[u] = 1.0;
                }
            }
            let mut pagerank = PageRank::default();
            let mut scores = pagerank
                .fit_predict(
                    adjacency,
                    Some(ValuesInput::Vector(cluster_weights)),
                    None,
                    None,
                    false,
                )
                .map_err(|_| KCentersError::InvalidInput)?;
            for u in 0..n {
                if !mask[u] {
                    scores[u] = 0.0;
                }
            }
            let mut best_node = 0usize;
            let mut best_score = f64::NEG_INFINITY;
            for (u, &score) in scores.iter().enumerate() {
                if score > best_score {
                    best_score = score;
                    best_node = u;
                }
            }
            *center_slot = best_node;
            used.push(best_node);
        }
        Ok(centers)
    }

    /// Fits the estimator and stores labels and centers.
    ///
    /// # Errors
    /// Returns [`KCentersError::InvalidNClusters`] or [`KCentersError::InvalidNInit`]
    /// for invalid hyperparameters, [`KCentersError::UnknownCenterPosition`] for
    /// unsupported bipartite masks, [`KCentersError::TooManyClusters`] when the
    /// mask cannot host enough centers, and [`KCentersError::InvalidInput`] for
    /// matrix or PageRank failures.
    pub fn fit(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<(), KCentersError> {
        if self.n_clusters < 2 {
            return Err(KCentersError::InvalidNClusters);
        }
        if self.n_init < 1 {
            return Err(KCentersError::InvalidNInit);
        }

        let input_matrix = if self.directed {
            directed2undirected(input_matrix, true)
        } else {
            input_matrix.to_owned()
        };
        let (adjacency, bipartite) = get_adjacency(
            MatrixInput::Sparse(input_matrix.to_owned()),
            true,
            force_bipartite,
            false,
            false,
        )
        .map_err(|_| KCentersError::InvalidInput)?;
        self.bipartite = bipartite;

        let n_row = input_matrix.rows();
        let mask = self.compute_mask_centers(input_matrix.shape(), bipartite)?;
        if self.n_clusters > mask.iter().filter(|&&x| x).count() {
            return Err(KCentersError::TooManyClusters);
        }

        let mut best_labels = Vec::<i32>::new();
        let mut best_centers = Vec::<usize>::new();
        let mut best_score = f64::NEG_INFINITY;

        for _ in 0..self.n_init {
            let mut centers = self.init_centers(&adjacency, &mask, self.n_clusters)?;
            let mut prev = Vec::<usize>::new();
            let mut labels = vec![0i32; adjacency.rows()];
            let mut n_iter = 0usize;
            while centers != prev && n_iter < self.max_iter {
                labels = Self::assign_labels(&adjacency, &centers)?;
                prev = centers.clone();
                centers = self.update_centers(&adjacency, &labels, &mask)?;
                n_iter += 1;
            }
            let labels_arr = Array1::from_vec(labels.clone());
            let score = if bipartite {
                let labels_row = Array1::from_vec(labels[..n_row].to_vec());
                let labels_col = Array1::from_vec(labels[n_row..].to_vec());
                get_modularity(&input_matrix, &labels_row, Some(&labels_col), "degree", 1.0)
                    .map_err(|_| KCentersError::InvalidInput)?
            } else {
                get_modularity(&adjacency, &labels_arr, None, "degree", 1.0)
                    .map_err(|_| KCentersError::InvalidInput)?
            };
            if score > best_score {
                best_score = score;
                best_labels = labels.clone();
                best_centers = centers.clone();
            }
        }

        self.labels = Some(Array1::from_vec(best_labels.clone()));
        self.centers = Some(best_centers.clone());

        if bipartite {
            self.labels_row = Some(Array1::from_vec(best_labels[..n_row].to_vec()));
            self.labels_col = Some(Array1::from_vec(best_labels[n_row..].to_vec()));
            match self.center_position.as_str() {
                "row" => {
                    self.centers_row = Some(best_centers.clone());
                    self.centers_col = None;
                }
                "col" => {
                    self.centers_row = None;
                    self.centers_col = Some(best_centers.iter().map(|c| c - n_row).collect());
                }
                "both" => {
                    self.centers_row = Some(
                        best_centers
                            .iter()
                            .copied()
                            .filter(|&c| c < n_row)
                            .collect(),
                    );
                    self.centers_col = Some(
                        best_centers
                            .iter()
                            .copied()
                            .filter(|&c| c >= n_row)
                            .map(|c| c - n_row)
                            .collect(),
                    );
                }
                _ => return Err(KCentersError::UnknownCenterPosition),
            }
        } else {
            self.labels_row = None;
            self.labels_col = None;
            self.centers_row = None;
            self.centers_col = None;
        }
        Ok(())
    }

    /// Fits the estimator and returns cluster labels.
    ///
    /// # Errors
    /// Propagates all errors from [`Self::fit`].
    pub fn fit_predict(
        &mut self,
        input_matrix: &CsMat<f64>,
        force_bipartite: bool,
    ) -> Result<Array1<i32>, KCentersError> {
        self.fit(input_matrix, force_bipartite)?;
        Ok(self.labels.clone().unwrap_or_else(|| Array1::zeros(0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::{test_bigraph, test_digraph, test_graph};

    #[test]
    fn test_kcenters() {
        let n_clusters = 2;
        let adjacency = test_graph();
        let n_row = adjacency.rows();
        let mut kcenters = KCenters::new(n_clusters, false, "row", 5, 20);
        let labels = kcenters.fit_predict(&adjacency, false).unwrap();
        assert_eq!(labels.len(), n_row);
        assert_eq!(
            labels
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            n_clusters
        );

        let n_clusters = 3;
        let adjacency = test_digraph();
        let n_row = adjacency.rows();
        let mut kcenters = KCenters::new(n_clusters, true, "row", 5, 20);
        let labels = kcenters.fit_predict(&adjacency, false).unwrap();
        assert_eq!(labels.len(), n_row);
        assert_eq!(
            labels
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            n_clusters
        );

        let n_clusters = 2;
        let biadjacency = test_bigraph();
        let (n_row, n_col) = biadjacency.shape();
        let mut kcenters = KCenters::new(n_clusters, false, "row", 5, 20);
        kcenters.fit(&biadjacency, false).unwrap();
        let labels = kcenters.labels.clone().unwrap_or_else(|| Array1::zeros(0));
        assert_eq!(
            kcenters
                .labels_row
                .clone()
                .unwrap_or_else(|| Array1::zeros(0))
                .len(),
            n_row
        );
        assert_eq!(
            kcenters
                .labels_col
                .clone()
                .unwrap_or_else(|| Array1::zeros(0))
                .len(),
            n_col
        );
        assert_eq!(
            labels
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            n_clusters
        );
    }

    #[test]
    fn test_kcenters_error() {
        let adjacency = test_graph();
        let biadjacency = test_bigraph();

        let mut kcenters = KCenters::new(1, false, "row", 5, 20);
        assert_eq!(
            kcenters.fit(&adjacency, false),
            Err(KCentersError::InvalidNClusters)
        );

        let mut kcenters = KCenters::new(2, false, "row", 0, 20);
        assert_eq!(
            kcenters.fit(&adjacency, false),
            Err(KCentersError::InvalidNInit)
        );

        let mut kcenters = KCenters::new(2, false, "other", 5, 20);
        assert_eq!(
            kcenters.fit(&biadjacency, false),
            Err(KCentersError::UnknownCenterPosition)
        );
    }

    #[test]
    fn test_center_position_variants() {
        let biadjacency = test_bigraph();
        let n_row = biadjacency.rows();
        let n_col = biadjacency.cols();

        let mut row_only = KCenters::new(2, false, "row", 3, 20);
        row_only.fit(&biadjacency, false).unwrap();
        let row_centers = row_only.centers_row.clone().unwrap_or_default();
        assert!(!row_centers.is_empty());
        assert!(row_centers.iter().all(|&c| c < n_row));
        assert!(row_only.centers_col.is_none());

        let mut col_only = KCenters::new(2, false, "col", 3, 20);
        col_only.fit(&biadjacency, false).unwrap();
        let col_centers = col_only.centers_col.clone().unwrap_or_default();
        assert!(!col_centers.is_empty());
        assert!(col_centers.iter().all(|&c| c < n_col));
        assert!(col_only.centers_row.is_none());

        let mut both = KCenters::new(2, false, "both", 3, 20);
        both.fit(&biadjacency, false).unwrap();
        let merged_len = both.centers_row.clone().unwrap_or_default().len()
            + both.centers_col.clone().unwrap_or_default().len();
        assert_eq!(merged_len, 2);
    }

    #[test]
    fn test_too_many_clusters_for_mask() {
        let biadjacency = test_bigraph();
        let n_row = biadjacency.rows();
        let n_col = biadjacency.cols();

        let mut too_many_row = KCenters::new(n_row + 1, false, "row", 2, 10);
        assert_eq!(
            too_many_row.fit(&biadjacency, false),
            Err(KCentersError::TooManyClusters)
        );

        let mut too_many_col = KCenters::new(n_col + 1, false, "col", 2, 10);
        assert_eq!(
            too_many_col.fit(&biadjacency, false),
            Err(KCentersError::TooManyClusters)
        );
    }

    #[test]
    fn test_restart_stability() {
        let adjacency = test_graph();
        let mut one = KCenters::new(2, false, "row", 1, 20);
        let mut many = KCenters::new(2, false, "row", 5, 20);
        let l1 = one.fit_predict(&adjacency, false).unwrap();
        let l2 = many.fit_predict(&adjacency, false).unwrap();
        assert_eq!(l1.to_vec(), l2.to_vec());
    }
}
