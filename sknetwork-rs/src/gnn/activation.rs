use ndarray::{Array2, Axis};

use crate::gnn::base_activation::BaseActivation;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by activation error operations.
pub enum ActivationError {
    /// Indicates unknown activation.
    UnknownActivation,
}

#[derive(Debug, Clone, Default)]
/// ReLu value.
pub struct ReLu;

impl BaseActivation for ReLu {
    fn output(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0.0 { v } else { 0.0 })
    }

    fn gradient(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }
}

#[derive(Debug, Clone, Default)]
/// Sigmoid value.
pub struct Sigmoid;

impl BaseActivation for Sigmoid {
    fn output(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn gradient(&self, x: &Array2<f64>) -> Array2<f64> {
        let y = self.output(x);
        y.mapv(|v| v * (1.0 - v))
    }
}

#[derive(Debug, Clone, Default)]
/// Softmax value.
pub struct Softmax;

impl BaseActivation for Softmax {
    fn output(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut out = x.to_owned();
        for mut row in out.axis_iter_mut(Axis(0)) {
            let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut s = 0.0;
            for v in &mut row {
                *v = (*v - m).exp();
                s += *v;
            }
            if s > 0.0 {
                for v in &mut row {
                    *v /= s;
                }
            }
        }
        out
    }

    fn gradient(&self, x: &Array2<f64>) -> Array2<f64> {
        // Diagonal approximation used in training loops.
        let y = self.output(x);
        y.mapv(|v| v * (1.0 - v))
    }
}

/// Returns activation.
///
/// # Errors
///
/// Returns [`ActivationError`] on failure.
pub fn get_activation(name: &str) -> Result<Box<dyn BaseActivation>, ActivationError> {
    match name.to_lowercase().as_str() {
        "relu" => Ok(Box::new(ReLu)),
        "sigmoid" => Ok(Box::new(Sigmoid)),
        "softmax" => Ok(Box::new(Softmax)),
        _ => Err(ActivationError::UnknownActivation),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_activation_factory_and_shapes() {
        let x = array![[-1.0, 0.0, 2.0]];
        let relu = get_activation("relu").expect("relu");
        assert_eq!(relu.output(&x).shape(), &[1, 3]);
        let sig = get_activation("sigmoid").expect("sigmoid");
        assert_eq!(sig.output(&x).shape(), &[1, 3]);
        let sm = get_activation("softmax").expect("softmax");
        let y = sm.output(&x);
        assert!((y.sum() - 1.0).abs() < 1e-12);
        assert!(matches!(
            get_activation("bad"),
            Err(ActivationError::UnknownActivation)
        ));
    }
}
