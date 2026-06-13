use ndarray::{Array1, Array2};
use sprs::CsMat;

use crate::gnn::layer::Convolution;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by gnnbase error operations.
pub enum GNNBaseError {
    /// Indicates not fitted.
    NotFitted,
    /// Indicates invalid input.
    InvalidInput,
    /// Indicates invalid config.
    InvalidConfig,
    /// Indicates unknown loss.
    UnknownLoss,
    /// Indicates unknown optimizer.
    UnknownOptimizer,
    /// Indicates layer forward.
    LayerForward,
}

#[derive(Debug, Clone, Default)]
/// BaseGNNState value.
pub struct BaseGNNState {
    /// Fitted value.
    pub fitted: bool,
    /// History value.
    pub history: Vec<f64>,
    /// Output value.
    pub output: Option<Array2<f64>>,
}

impl BaseGNNState {
    /// Validates fitted.
    ///
    /// # Errors
    ///
    /// Returns [`GNNBaseError`] on failure.
    pub fn check_fitted(&self) -> Result<(), GNNBaseError> {
        if self.fitted {
            Ok(())
        } else {
            Err(GNNBaseError::NotFitted)
        }
    }
}

/// Common interface for base gnn.
pub trait BaseGNN {
    /// Runs the fit step.
    fn fit(
        &mut self,
        adjacency: &CsMat<f64>,
        features: &Array2<f64>,
        labels: &Array1<i32>,
    ) -> Result<(), GNNBaseError>;
    /// Computes state.
    fn state(&self) -> &BaseGNNState;
    /// Computes state mut.
    fn state_mut(&mut self) -> &mut BaseGNNState;
    /// Computes layers.
    fn layers(&self) -> &[Convolution];

    /// Runs the fit-predict step.
    fn fit_predict(
        &mut self,
        adjacency: &CsMat<f64>,
        features: &Array2<f64>,
        labels: &Array1<i32>,
    ) -> Result<Array1<i32>, GNNBaseError> {
        self.fit(adjacency, features, labels)?;
        self.predict(adjacency, features)
    }

    /// Runs the fit-transform step.
    fn fit_transform(
        &mut self,
        adjacency: &CsMat<f64>,
        features: &Array2<f64>,
        labels: &Array1<i32>,
    ) -> Result<Array2<f64>, GNNBaseError> {
        self.fit(adjacency, features, labels)?;
        self.transform(adjacency, features)
    }

    /// Runs the predict step.
    fn predict(
        &self,
        adjacency: &CsMat<f64>,
        features: &Array2<f64>,
    ) -> Result<Array1<i32>, GNNBaseError> {
        self.state().check_fitted()?;
        let out = self.predict_proba(adjacency, features)?;
        let mut pred = Array1::<i32>::zeros(out.nrows());
        for i in 0..out.nrows() {
            let mut best = 0usize;
            let mut best_v = f64::NEG_INFINITY;
            for j in 0..out.ncols() {
                if out[[i, j]] > best_v {
                    best_v = out[[i, j]];
                    best = j;
                }
            }
            pred[i] = best as i32;
        }
        Ok(pred)
    }

    /// Runs the predict-proba step.
    fn predict_proba(
        &self,
        adjacency: &CsMat<f64>,
        features: &Array2<f64>,
    ) -> Result<Array2<f64>, GNNBaseError> {
        self.state().check_fitted()?;
        let h = self.transform(adjacency, features)?;
        if h.nrows() == 0 {
            return Ok(Array2::<f64>::zeros((0, 0)));
        }
        Ok(h)
    }

    /// Runs the transform step.
    fn transform(
        &self,
        adjacency: &CsMat<f64>,
        features: &Array2<f64>,
    ) -> Result<Array2<f64>, GNNBaseError> {
        self.state().check_fitted()?;
        let mut h = features.to_owned();
        for layer in self.layers() {
            let next = crate::gnn::base_layer::BaseLayer::forward(layer, adjacency, &h)
                .map_err(|_| GNNBaseError::LayerForward)?;
            h = next;
        }
        Ok(h)
    }
}
