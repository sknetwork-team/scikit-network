use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{random, SeedableRng};
use sprs::CsMat;

use crate::gnn::base::{BaseGNN, BaseGNNState, GNNBaseError};
use crate::gnn::base_layer::LayerParams;
use crate::gnn::layer::Convolution;
use crate::gnn::loss::get_loss;
use crate::gnn::neighbor_sampler::UniformNeighborSampler;
use crate::gnn::optimizer::{MultiLayerAdam, MultiLayerGD};
use crate::gnn::utils::{GNNUtilsError, check_loss, get_layers};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by gnnclassifier error operations.
pub enum GNNClassifierError {
    /// Indicates invalid config.
    InvalidConfig,
    /// Indicates utils.
    Utils(GNNUtilsError),
}

impl From<GNNUtilsError> for GNNClassifierError {
    fn from(value: GNNUtilsError) -> Self {
        Self::Utils(value)
    }
}

#[derive(Debug, Clone)]
/// GNNClassifier value.
pub struct GNNClassifier {
    /// Dims value.
    pub dims: Vec<usize>,
    /// Layer Name value.
    pub layer_name: String,
    /// Sample Size value.
    pub sample_size: Option<usize>,
    /// Sample Sizes value.
    pub sample_sizes: Option<Vec<usize>>,
    /// Loss Name value.
    pub loss_name: String,
    /// Optimizer Name value.
    pub optimizer_name: String,
    /// Learning Rate value.
    pub learning_rate: f64,
    /// N Epochs value.
    pub n_epochs: usize,
    /// Random State value.
    pub random_state: Option<u64>,
    /// Layers value.
    pub layers_: Vec<Convolution>,
    /// State value.
    pub state_: BaseGNNState,
    adjacencies_: Option<Vec<CsMat<f64>>>,
}

#[derive(Debug, Clone)]
struct LayerForwardState {
    embedding: Array2<f64>,
    output: Array2<f64>,
}

impl GNNClassifier {
    /// Creates a classifier with default GCN layers and no neighbor sampling.
    pub fn new(
        dims: Vec<usize>,
        loss_name: &str,
        optimizer_name: &str,
        learning_rate: f64,
        n_epochs: usize,
    ) -> Result<Self, GNNClassifierError> {
        Self::new_with_layer(
            dims,
            "gcn",
            None,
            loss_name,
            optimizer_name,
            learning_rate,
            n_epochs,
        )
    }

    /// Creates a classifier with a single optional sampling size for all SAGE layers.
    ///
    /// For `layer_name = "sage"`, `sample_size` must be strictly positive when provided.
    /// For non-SAGE layers, any sampling configuration is rejected.
    pub fn new_with_layer(
        dims: Vec<usize>,
        layer_name: &str,
        sample_size: Option<usize>,
        loss_name: &str,
        optimizer_name: &str,
        learning_rate: f64,
        n_epochs: usize,
    ) -> Result<Self, GNNClassifierError> {
        Self::new_with_layer_sizes(
            dims,
            layer_name,
            None,
            sample_size,
            loss_name,
            optimizer_name,
            learning_rate,
            n_epochs,
        )
    }

    /// Creates a classifier with optional per-layer sampling sizes.
    ///
    /// Rules:
    /// - Sampling is supported only for `layer_name = "sage"`.
    /// - `sample_sizes`, when provided, must have exact length `dims.len() - 1`.
    /// - All provided sampling sizes must be strictly positive.
    /// - If `sample_sizes` is `None`, a global positive `sample_size` is used.
    /// - If both are `None` for SAGE, a default of `25` is used.
    pub fn new_with_layer_sizes(
        dims: Vec<usize>,
        layer_name: &str,
        sample_sizes: Option<Vec<usize>>,
        sample_size: Option<usize>,
        loss_name: &str,
        optimizer_name: &str,
        learning_rate: f64,
        n_epochs: usize,
    ) -> Result<Self, GNNClassifierError> {
        let lname = layer_name.to_lowercase();
        if lname != "sage" && (sample_sizes.is_some() || sample_size.is_some()) {
            return Err(GNNClassifierError::InvalidConfig);
        }
        let mut layers_ = get_layers(&dims, layer_name)?;
        if lname == "sage" {
            if let Some(s) = sample_size {
                if s == 0 {
                    return Err(GNNClassifierError::InvalidConfig);
                }
            }
            if let Some(sizes) = &sample_sizes {
                if sizes.iter().any(|&s| s == 0) {
                    return Err(GNNClassifierError::InvalidConfig);
                }
                if sizes.len() != layers_.len() {
                    return Err(GNNClassifierError::InvalidConfig);
                }
            }
        }
        let default_size = sample_size.unwrap_or(25);
        for (k, layer) in layers_.iter_mut().enumerate() {
            layer.layer_type = lname.clone();
            layer.sample_size = if lname == "sage" {
                let s = sample_sizes.as_ref().map(|v| v[k]).unwrap_or(default_size);
                Some(s)
            } else {
                None
            };
        }
        let last_activation = if loss_name.to_lowercase().contains("binary") {
            "sigmoid"
        } else {
            "softmax"
        };
        if let Some(last) = layers_.last_mut() {
            last.activation = last_activation.to_string();
        }
        Ok(Self {
            dims,
            layer_name: layer_name.to_string(),
            sample_size,
            sample_sizes,
            loss_name: loss_name.to_string(),
            optimizer_name: optimizer_name.to_string(),
            learning_rate,
            n_epochs,
            random_state: None,
            layers_,
            state_: BaseGNNState::default(),
            adjacencies_: None,
        })
    }

    fn initialize_weights(&mut self, in_dim: usize, rng: &mut StdRng) {
        let mut in_ch = in_dim;
        for layer in &mut self.layers_ {
            let out_ch = layer.params.weights.ncols();
            let use_bias = layer.params.bias.is_some();
            layer.params = LayerParams::he_init(in_ch, out_ch, use_bias, rng);
            in_ch = out_ch;
        }
    }

    fn sample_adjacencies_rng(
        &self,
        adjacency: &CsMat<f64>,
        rng: &mut StdRng,
    ) -> Vec<CsMat<f64>> {
        self.layers_
            .iter()
            .map(|layer| {
                if layer.layer_type.to_lowercase() == "sage" {
                    let sampler =
                        UniformNeighborSampler::new(layer.sample_size.unwrap_or(25));
                    sampler.sample_rng(adjacency, rng)
                } else {
                    adjacency.to_owned()
                }
            })
            .collect()
    }

    fn forward_layers(
        &self,
        adjacencies: &[CsMat<f64>],
        features: &Array2<f64>,
    ) -> Result<Array2<f64>, GNNBaseError> {
        Ok(self
            .forward_layers_with_states(adjacencies, features)?
            .0)
    }

    fn forward_layers_with_states(
        &self,
        adjacencies: &[CsMat<f64>],
        features: &Array2<f64>,
    ) -> Result<(Array2<f64>, Vec<LayerForwardState>), GNNBaseError> {
        let mut layer_states = Vec::with_capacity(self.layers_.len());
        let mut h = features.to_owned();
        for (k, layer) in self.layers_.iter().enumerate() {
            let (embedding, output) = layer
                .forward_with_embedding(&adjacencies[k], &h)
                .map_err(|_| GNNBaseError::LayerForward)?;
            layer_states.push(LayerForwardState { embedding, output: output.clone() });
            h = output;
        }
        Ok((h, layer_states))
    }

    fn train_indices(labels: &Array1<i32>) -> Vec<usize> {
        labels
            .iter()
            .enumerate()
            .filter_map(|(i, &lab)| if lab >= 0 { Some(i) } else { None })
            .collect()
    }

    fn gather_rows(mat: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((indices.len(), mat.ncols()));
        for (r, &i) in indices.iter().enumerate() {
            for j in 0..mat.ncols() {
                out[[r, j]] = mat[[i, j]];
            }
        }
        out
    }

    /// Cross-entropy gradient w.r.t. pre-softmax logits (Python ``CrossEntropy.loss_gradient``).
    fn cross_entropy_grad_embedding(
        embedding: &Array2<f64>,
        train_indices: &[usize],
        train_labels: &[i32],
    ) -> Array2<f64> {
        let n_train = train_indices.len();
        let n_classes = embedding.ncols();
        let mut grad = Array2::<f64>::zeros((n_train, n_classes));
        for (r, (&node, &lab)) in train_indices.iter().zip(train_labels.iter()).enumerate() {
            let mut row = vec![0.0_f64; n_classes];
            let mut max_v = f64::NEG_INFINITY;
            for j in 0..n_classes {
                let v = embedding[[node, j]];
                row[j] = v;
                if v > max_v {
                    max_v = v;
                }
            }
            let mut sum = 0.0;
            for j in 0..n_classes {
                row[j] = (row[j] - max_v).exp();
                sum += row[j];
            }
            if sum > 0.0 {
                for j in 0..n_classes {
                    row[j] /= sum;
                }
            }
            for j in 0..n_classes {
                grad[[r, j]] = row[j] - if j as i32 == lab { 1.0 } else { 0.0 };
            }
        }
        grad
    }

    fn relu_grad(embedding: &Array2<f64>, direction: &Array2<f64>) -> Array2<f64> {
        let mut out = direction.to_owned();
        for i in 0..out.nrows() {
            for j in 0..out.ncols() {
                if embedding[[i, j]] <= 0.0 {
                    out[[i, j]] = 0.0;
                }
            }
        }
        out
    }

    /// Backprop matching Python ``BaseGNN.backward`` (masked to labeled nodes).
    fn backward(
        layers: &[Convolution],
        layer_states: &[LayerForwardState],
        features: &Array2<f64>,
        labels: &Array1<i32>,
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let train_indices = Self::train_indices(labels);
        let train_labels: Vec<i32> = train_indices.iter().map(|&i| labels[i]).collect();
        let n_layers = layers.len();
        let last = n_layers - 1;

        let mut gradient = Self::cross_entropy_grad_embedding(
            &layer_states[last].embedding,
            &train_indices,
            &train_labels,
        );

        let mut deriv_w_rev = Vec::with_capacity(n_layers);
        let mut deriv_b_rev = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            let layer_idx = n_layers - 1 - i;
            let signal = if i < n_layers - 1 {
                Self::gather_rows(&layer_states[layer_idx - 1].output, &train_indices)
            } else {
                Self::gather_rows(features, &train_indices)
            };

            deriv_w_rev.push(signal.t().dot(&gradient));
            let mut mean_b = Array2::<f64>::zeros((1, gradient.ncols()));
            for j in 0..gradient.ncols() {
                let mut s = 0.0;
                for r in 0..gradient.nrows() {
                    s += gradient[[r, j]];
                }
                mean_b[[0, j]] = s / gradient.nrows().max(1) as f64;
            }
            deriv_b_rev.push(mean_b);

            if i < n_layers - 1 {
                let prev_idx = layer_idx - 1;
                let emb_masked =
                    Self::gather_rows(&layer_states[prev_idx].embedding, &train_indices);
                let w = &layers[layer_idx].params.weights;
                let direction = gradient.dot(&w.t());
                gradient = Self::relu_grad(&emb_masked, &direction);
            }
        }

        let deriv_w: Vec<Array2<f64>> = deriv_w_rev.into_iter().rev().collect();
        let deriv_b: Vec<Array2<f64>> = deriv_b_rev.into_iter().rev().collect();
        (deriv_w, deriv_b)
    }

    fn masked_cross_entropy_loss(
        labels: &Array1<i32>,
        y_true: &Array2<f64>,
        out: &Array2<f64>,
    ) -> f64 {
        let eps = 1e-15;
        let mut total = 0.0;
        let mut n_train = 0usize;
        for i in 0..out.nrows() {
            if labels[i] < 0 {
                continue;
            }
            n_train += 1;
            for j in 0..out.ncols() {
                let t = y_true[[i, j]];
                total -= t * out[[i, j]].max(eps).ln();
            }
        }
        if n_train == 0 {
            0.0
        } else {
            total / n_train as f64
        }
    }

    fn one_hot(labels: &Array1<i32>, n_classes: usize) -> Array2<f64> {
        let mut y = Array2::<f64>::zeros((labels.len(), n_classes));
        for (i, &lab) in labels.iter().enumerate() {
            if lab >= 0 {
                y[[i, lab as usize]] = 1.0;
            }
        }
        y
    }

}

impl BaseGNN for GNNClassifier {
    fn fit(
        &mut self,
        adjacency: &CsMat<f64>,
        features: &Array2<f64>,
        labels: &Array1<i32>,
    ) -> Result<(), GNNBaseError> {
        if self.layers_.is_empty() {
            self.state_.fitted = false;
            self.state_.output = None;
            return Err(GNNBaseError::InvalidConfig);
        }
        if features.nrows() != adjacency.rows() || labels.len() != adjacency.rows() {
            self.state_.fitted = false;
            self.state_.output = None;
            return Err(GNNBaseError::InvalidInput);
        }
        let final_activation = self
            .layers_
            .last()
            .map(|l| l.activation.as_str())
            .unwrap_or("softmax");
        if check_loss(final_activation, &self.loss_name).is_err() {
            self.state_.fitted = false;
            self.state_.output = None;
            return Err(GNNBaseError::InvalidConfig);
        }
        let n_classes = labels.iter().copied().max().map(|x| x as usize + 1).unwrap_or(1);
        if self
            .layers_
            .last()
            .map(|l| l.params.weights.ncols())
            .unwrap_or(0)
            != n_classes
        {
            self.state_.fitted = false;
            self.state_.output = None;
            return Err(GNNBaseError::InvalidConfig);
        }
        let y_true = Self::one_hot(labels, n_classes);
        let opt_name = self.optimizer_name.to_lowercase();
        let mut adam_opt = if opt_name == "adam" {
            Some(MultiLayerAdam::new(self.learning_rate))
        } else {
            None
        };
        let gd_opt = if opt_name != "adam" {
            Some(MultiLayerGD::new(self.learning_rate))
        } else {
            None
        };
        if opt_name != "adam" && opt_name != "gd" && opt_name != "sgd" && opt_name != "gradient" {
            self.state_.fitted = false;
            self.state_.output = None;
            return Err(GNNBaseError::UnknownOptimizer);
        }
        let _loss = match get_loss(&self.loss_name) {
            Ok(l) => l,
            Err(_) => {
                self.state_.fitted = false;
                self.state_.output = None;
                return Err(GNNBaseError::UnknownLoss);
            }
        };

        self.state_.history.clear();
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(random()),
        };
        let adjacencies = self.sample_adjacencies_rng(adjacency, &mut rng);
        self.initialize_weights(features.ncols(), &mut rng);
        self.adjacencies_ = Some(adjacencies.clone());

        for _ in 0..self.n_epochs.max(1) {
            let (out, layer_states) = self.forward_layers_with_states(&adjacencies, features)?;
            let l = Self::masked_cross_entropy_loss(labels, &y_true, &out);
            self.state_.history.push(l);

            let (deriv_w, deriv_b) =
                Self::backward(&self.layers_, &layer_states, features, labels);

            if let Some(adam) = adam_opt.as_mut() {
                adam.ensure_state(&deriv_w, &deriv_b);
                for (idx, layer) in self.layers_.iter_mut().enumerate() {
                    adam.step_layer(
                        idx,
                        &mut layer.params.weights,
                        layer.params.bias.as_mut(),
                        &deriv_w[idx],
                        &deriv_b[idx],
                    );
                }
            } else if let Some(gd) = gd_opt.as_ref() {
                for (idx, layer) in self.layers_.iter_mut().enumerate() {
                    gd.step_layer(
                        &mut layer.params.weights,
                        layer.params.bias.as_mut(),
                        &deriv_w[idx],
                        &deriv_b[idx],
                    );
                }
            }
        }

        let final_out = self.forward_layers(&adjacencies, features)?;
        self.state_.output = Some(final_out);
        self.state_.fitted = true;
        Ok(())
    }

    fn predict_proba(
        &self,
        _adjacency: &CsMat<f64>,
        _features: &Array2<f64>,
    ) -> Result<Array2<f64>, GNNBaseError> {
        self.state().check_fitted()?;
        self.state_
            .output
            .clone()
            .ok_or(GNNBaseError::NotFitted)
    }

    fn state(&self) -> &BaseGNNState {
        &self.state_
    }

    fn state_mut(&mut self) -> &mut BaseGNNState {
        &mut self.state_
    }

    fn layers(&self) -> &[Convolution] {
        &self.layers_
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use super::*;
    use crate::data::test_graphs::test_graph;
    use crate::gnn::base::BaseGNN;

    #[test]
    fn test_gnn_classifier_api() {
        let a = test_graph();
        let x = Array2::<f64>::ones((a.rows(), 4));
        let y = Array1::from_vec((0..a.rows()).map(|i| (i % 2) as i32).collect());
        let mut clf = GNNClassifier::new(vec![4, 8, 2], "cross_entropy", "adam", 1e-2, 3).expect("new");
        let pred = clf.fit_predict(&a, &x, &y).expect("fit_predict");
        assert_eq!(pred.len(), a.rows());
        let proba = clf.predict_proba(&a, &x).expect("predict_proba");
        assert_eq!(proba.nrows(), a.rows());
        assert_eq!(clf.state_.history.len(), 3);
    }

    #[test]
    fn test_gnn_classifier_new_with_layer_sage() {
        let clf = GNNClassifier::new_with_layer(
            vec![4, 8, 2],
            "sage",
            Some(3),
            "cross_entropy",
            "adam",
            1e-2,
            2,
        )
        .expect("new");
        assert_eq!(clf.layer_name, "sage");
        assert_eq!(clf.sample_size, Some(3));
        assert!(clf.layers_.iter().all(|l| l.layer_type == "sage"));
        assert!(clf.layers_.iter().all(|l| l.sample_size == Some(3)));
    }

    #[test]
    fn test_gnn_classifier_new_with_layer_sizes_sage() {
        let clf = GNNClassifier::new_with_layer_sizes(
            vec![4, 8, 6, 2],
            "sage",
            Some(vec![1, 2, 4]),
            Some(7),
            "cross_entropy",
            "adam",
            1e-2,
            2,
        )
        .expect("new");
        assert_eq!(clf.layer_name, "sage");
        assert_eq!(clf.sample_sizes, Some(vec![1, 2, 4]));
        assert_eq!(clf.layers_[0].sample_size, Some(1));
        assert_eq!(clf.layers_[1].sample_size, Some(2));
        assert_eq!(clf.layers_[2].sample_size, Some(4));
    }

    #[test]
    fn test_gnn_classifier_new_with_layer_sizes_mismatch_rejected() {
        let err = GNNClassifier::new_with_layer_sizes(
            vec![4, 8, 6, 2],
            "sage",
            Some(vec![1]),
            Some(5),
            "cross_entropy",
            "adam",
            1e-2,
            2,
        )
        .expect_err("must reject sample size vector with wrong length");
        assert_eq!(err, GNNClassifierError::InvalidConfig);
    }

    #[test]
    fn test_gnn_classifier_non_sage_rejects_sampling_config() {
        let err = GNNClassifier::new_with_layer_sizes(
            vec![4, 8, 2],
            "gcn",
            Some(vec![2, 2]),
            Some(2),
            "cross_entropy",
            "adam",
            1e-2,
            2,
        )
        .expect_err("must reject sampling config for non-sage layers");
        assert_eq!(err, GNNClassifierError::InvalidConfig);
    }

    #[test]
    fn test_gnn_classifier_rejects_zero_sample_size() {
        let err = GNNClassifier::new_with_layer_sizes(
            vec![4, 8, 2],
            "sage",
            None,
            Some(0),
            "cross_entropy",
            "adam",
            1e-2,
            2,
        )
        .expect_err("must reject zero global sample size");
        assert_eq!(err, GNNClassifierError::InvalidConfig);
    }

    #[test]
    fn test_gnn_classifier_rejects_zero_in_sample_sizes() {
        let err = GNNClassifier::new_with_layer_sizes(
            vec![4, 8, 6, 2],
            "sage",
            Some(vec![2, 0, 3]),
            None,
            "cross_entropy",
            "adam",
            1e-2,
            2,
        )
        .expect_err("must reject zero in per-layer sample sizes");
        assert_eq!(err, GNNClassifierError::InvalidConfig);
    }

    #[test]
    fn test_gnn_classifier_invalid_config_paths() {
        let a = test_graph();
        let x = Array2::<f64>::ones((a.rows(), 4));
        let y = Array1::from_vec((0..a.rows()).map(|i| (i % 2) as i32).collect());

        // output dim mismatch: last dim != n_classes
        let mut clf = GNNClassifier::new(vec![4, 8, 3], "cross_entropy", "adam", 1e-2, 2).expect("new");
        let out = clf.fit(&a, &x, &y);
        assert!(matches!(out, Err(GNNBaseError::InvalidConfig)));
        assert!(!clf.state_.fitted);
        assert!(clf.state_.output.is_none());

        // shape mismatch
        let mut clf2 = GNNClassifier::new(vec![4, 8, 2], "cross_entropy", "adam", 1e-2, 2).expect("new");
        let bad_x = Array2::<f64>::ones((a.rows() - 1, 4));
        let out = clf2.fit(&a, &bad_x, &y);
        assert!(matches!(out, Err(GNNBaseError::InvalidInput)));
        assert!(!clf2.state_.fitted);
    }

    #[test]
    fn test_gnn_classifier_sage_sampling_branch() {
        let a = test_graph();
        let x = Array2::<f64>::ones((a.rows(), 4));
        let y = Array1::from_vec((0..a.rows()).map(|i| (i % 2) as i32).collect());
        let mut clf =
            GNNClassifier::new_with_layer(vec![4, 8, 2], "sage", Some(1), "cross_entropy", "adam", 1e-2, 2)
                .expect("new");
        let _ = clf.fit_predict(&a, &x, &y).expect("fit_predict");
        assert!(clf.state_.fitted);
        assert_eq!(clf.state_.history.len(), 2);
    }

    #[test]
    fn test_not_fitted_inference_gated() {
        let a = test_graph();
        let x = Array2::<f64>::ones((a.rows(), 4));
        let clf = GNNClassifier::new(vec![4, 8, 2], "cross_entropy", "adam", 1e-2, 2).expect("new");
        assert!(matches!(clf.predict(&a, &x), Err(GNNBaseError::NotFitted)));
        assert!(matches!(
            clf.predict_proba(&a, &x),
            Err(GNNBaseError::NotFitted)
        ));
        assert!(matches!(clf.transform(&a, &x), Err(GNNBaseError::NotFitted)));
    }
}
