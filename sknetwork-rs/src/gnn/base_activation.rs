use ndarray::Array2;

/// Common interface for base activation.
pub trait BaseActivation {
    /// Computes output.
    fn output(&self, x: &Array2<f64>) -> Array2<f64>;
    /// Computes gradient.
    fn gradient(&self, x: &Array2<f64>) -> Array2<f64>;
}

/// Common interface for base loss.
pub trait BaseLoss {
    /// Computes loss.
    fn loss(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64;
    /// Computes loss gradient.
    fn loss_gradient(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> Array2<f64>;
}
