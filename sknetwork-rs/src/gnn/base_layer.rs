use ndarray::Array2;
use rand::Rng;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by layer error operations.
pub enum LayerError {
    /// Indicates invalid shape.
    InvalidShape,
    /// Indicates not implemented.
    NotImplemented,
}

/// Common interface for base layer.
pub trait BaseLayer {
    /// Computes forward.
    ///
    /// # Errors
    ///
    /// Returns [`LayerError`] on failure.
    fn forward(&self, adjacency: &sprs::CsMat<f64>, features: &Array2<f64>) -> Result<Array2<f64>, LayerError>;
}

#[derive(Debug, Clone)]
/// LayerParams value.
pub struct LayerParams {
    /// Weights value.
    pub weights: Array2<f64>,
    /// Bias value.
    pub bias: Option<Array2<f64>>,
}

fn normal_sample(rng: &mut impl Rng) -> f64 {
    loop {
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        if u1 > f64::EPSILON {
            return (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        }
    }
}

impl LayerParams {
    /// Creates a new instance.
    pub fn new(in_dim: usize, out_dim: usize, use_bias: bool) -> Self {
        Self::he_init(in_dim, out_dim, use_bias, &mut rand::rng())
    }

    /// Computes he init.
    pub fn he_init(in_dim: usize, out_dim: usize, use_bias: bool, rng: &mut impl Rng) -> Self {
        let scale = (2.0_f64 / out_dim.max(1) as f64).sqrt();
        let weights = Array2::from_shape_fn((in_dim, out_dim), |(_, _)| normal_sample(rng) * scale);
        let bias = if use_bias {
            Some(Array2::zeros((1, out_dim)))
        } else {
            None
        };
        Self { weights, bias }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_params_shape() {
        let p = LayerParams::new(4, 3, true);
        assert_eq!(p.weights.shape(), &[4, 3]);
        assert_eq!(p.bias.expect("bias").shape(), &[1, 3]);
    }
}
