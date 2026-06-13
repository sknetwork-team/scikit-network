use ndarray::Array2;

use crate::gnn::base_activation::BaseLoss;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by loss error operations.
pub enum LossError {
    /// Indicates unknown loss.
    UnknownLoss,
    /// Indicates shape mismatch.
    ShapeMismatch,
}

#[derive(Debug, Clone, Default)]
/// CrossEntropy value.
pub struct CrossEntropy;

impl BaseLoss for CrossEntropy {
    fn loss(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
        let eps = 1e-15;
        let mut total = 0.0;
        let n = y_true.nrows().max(1) as f64;
        for (t, p) in y_true.iter().zip(y_pred.iter()) {
            total -= *t * (*p).max(eps).ln();
        }
        total / n
    }

    fn loss_gradient(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> Array2<f64> {
        y_pred - y_true
    }
}

#[derive(Debug, Clone, Default)]
/// BinaryCrossEntropy value.
pub struct BinaryCrossEntropy;

impl BaseLoss for BinaryCrossEntropy {
    fn loss(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
        let eps = 1e-15;
        let n = y_true.len().max(1) as f64;
        let mut total = 0.0;
        for (t, p) in y_true.iter().zip(y_pred.iter()) {
            let pp = (*p).clamp(eps, 1.0 - eps);
            total += -(*t * pp.ln() + (1.0 - *t) * (1.0 - pp).ln());
        }
        total / n
    }

    fn loss_gradient(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> Array2<f64> {
        y_pred - y_true
    }
}

/// Returns loss.
///
/// # Errors
///
/// Returns [`LossError`] on failure.
pub fn get_loss(name: &str) -> Result<Box<dyn BaseLoss>, LossError> {
    match name.to_lowercase().as_str() {
        "crossentropy" | "cross_entropy" | "ce" => Ok(Box::new(CrossEntropy)),
        "binarycrossentropy" | "binary_cross_entropy" | "bce" => Ok(Box::new(BinaryCrossEntropy)),
        _ => Err(LossError::UnknownLoss),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_loss_factory_and_values() {
        let y_true = array![[1.0, 0.0], [0.0, 1.0]];
        let y_pred = array![[0.8, 0.2], [0.3, 0.7]];
        let ce = get_loss("ce").expect("ce");
        assert!(ce.loss(&y_true, &y_pred) > 0.0);
        assert_eq!(ce.loss_gradient(&y_true, &y_pred).shape(), &[2, 2]);

        let bce = get_loss("bce").expect("bce");
        assert!(bce.loss(&y_true, &y_pred) > 0.0);
        assert!(matches!(get_loss("bad"), Err(LossError::UnknownLoss)));
    }
}
