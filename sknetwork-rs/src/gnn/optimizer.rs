use ndarray::Array2;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by optimizer error operations.
pub enum OptimizerError {
    /// Indicates unknown optimizer.
    UnknownOptimizer,
}

/// Common interface for base optimizer.
pub trait BaseOptimizer {
    /// Computes step.
    fn step(&mut self, weights: &mut Array2<f64>, grad: &Array2<f64>);
}

#[derive(Debug, Clone)]
/// GD value.
pub struct GD {
    /// Learning Rate value.
    pub learning_rate: f64,
}

impl BaseOptimizer for GD {
    fn step(&mut self, weights: &mut Array2<f64>, grad: &Array2<f64>) {
        *weights = weights.to_owned() - &(grad * self.learning_rate);
    }
}

#[derive(Debug, Clone)]
/// ADAM value.
pub struct ADAM {
    /// Learning Rate value.
    pub learning_rate: f64,
    /// Beta1 value.
    pub beta1: f64,
    /// Beta2 value.
    pub beta2: f64,
    /// Eps value.
    pub eps: f64,
    /// T value.
    pub t: usize,
    /// M value.
    pub m: Option<Array2<f64>>,
    /// V value.
    pub v: Option<Array2<f64>>,
}

impl ADAM {
    /// Creates a new instance.
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: None,
            v: None,
        }
    }
}

impl BaseOptimizer for ADAM {
    fn step(&mut self, weights: &mut Array2<f64>, grad: &Array2<f64>) {
        self.t += 1;
        let m = self.m.get_or_insert_with(|| Array2::zeros(grad.raw_dim()));
        let v = self.v.get_or_insert_with(|| Array2::zeros(grad.raw_dim()));
        *m = m.mapv(|x| x * self.beta1) + &(grad * (1.0 - self.beta1));
        *v = v.mapv(|x| x * self.beta2) + &(grad.mapv(|g| g * g) * (1.0 - self.beta2));
        let m_hat = m.mapv(|x| x / (1.0 - self.beta1.powi(self.t as i32)));
        let v_hat = v.mapv(|x| x / (1.0 - self.beta2.powi(self.t as i32)));
        let upd = m_hat / v_hat.mapv(|x| x.sqrt() + self.eps);
        *weights = weights.to_owned() - &(upd * self.learning_rate);
    }
}

/// Returns optimizer.
///
/// # Errors
///
/// Returns [`OptimizerError`] on failure.
pub fn get_optimizer(name: &str, learning_rate: f64) -> Result<Box<dyn BaseOptimizer>, OptimizerError> {
    match name.to_lowercase().as_str() {
        "gd" | "sgd" | "gradient" => Ok(Box::new(GD { learning_rate })),
        "adam" => Ok(Box::new(ADAM::new(learning_rate))),
        _ => Err(OptimizerError::UnknownOptimizer),
    }
}

/// Per-layer Adam state matching Python ``sknetwork.gnn.optimizer.ADAM`` (one ``t`` step per layer).
#[derive(Debug, Clone)]
pub struct MultiLayerAdam {
    /// Learning Rate value.
    pub learning_rate: f64,
    /// Beta1 value.
    pub beta1: f64,
    /// Beta2 value.
    pub beta2: f64,
    /// Eps value.
    pub eps: f64,
    /// T value.
    pub t: usize,
    m_weights: Vec<Array2<f64>>,
    v_weights: Vec<Array2<f64>>,
    m_bias: Vec<Array2<f64>>,
    v_bias: Vec<Array2<f64>>,
}

impl MultiLayerAdam {
    /// Creates a new instance.
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m_weights: Vec::new(),
            v_weights: Vec::new(),
            m_bias: Vec::new(),
            v_bias: Vec::new(),
        }
    }

    /// Computes ensure state.
    pub fn ensure_state(&mut self, deriv_w: &[Array2<f64>], deriv_b: &[Array2<f64>]) {
        if self.t == 0 {
            self.m_weights = deriv_w.iter().map(|g| Array2::zeros(g.raw_dim())).collect();
            self.v_weights = deriv_w.iter().map(|g| Array2::zeros(g.raw_dim())).collect();
            self.m_bias = deriv_b.iter().map(|g| Array2::zeros(g.raw_dim())).collect();
            self.v_bias = deriv_b.iter().map(|g| Array2::zeros(g.raw_dim())).collect();
        }
    }

    /// Computes step layer.
    pub fn step_layer(
        &mut self,
        idx: usize,
        weights: &mut Array2<f64>,
        bias: Option<&mut Array2<f64>>,
        grad_w: &Array2<f64>,
        grad_b: &Array2<f64>,
    ) {
        self.t += 1;
        let denom_1 = 1.0 - self.beta1.powi(self.t as i32);
        let denom_2 = 1.0 - self.beta2.powi(self.t as i32);

        {
            let m = &mut self.m_weights[idx];
            let v = &mut self.v_weights[idx];
            *m = m.mapv(|x| x * self.beta1) + &(grad_w * (1.0 - self.beta1));
            *v = v.mapv(|x| x * self.beta2) + &(grad_w.mapv(|g| g * g) * (1.0 - self.beta2));
            let m_hat = m.mapv(|x| x / denom_1);
            let v_hat = v.mapv(|x| x / denom_2);
            let upd = m_hat / v_hat.mapv(|x| x.sqrt() + self.eps);
            *weights = weights.to_owned() - &(upd * self.learning_rate);
        }

        if let Some(bias) = bias {
            let m = &mut self.m_bias[idx];
            let v = &mut self.v_bias[idx];
            *m = m.mapv(|x| x * self.beta1) + &(grad_b * (1.0 - self.beta1));
            *v = v.mapv(|x| x * self.beta2) + &(grad_b.mapv(|g| g * g) * (1.0 - self.beta2));
            let m_hat = m.mapv(|x| x / denom_1);
            let v_hat = v.mapv(|x| x / denom_2);
            let upd = m_hat / v_hat.mapv(|x| x.sqrt() + self.eps);
            *bias = bias.to_owned() - &(upd * self.learning_rate);
        }
    }
}

/// Per-layer GD matching Python ``sknetwork.gnn.optimizer.GD``.
#[derive(Debug, Clone)]
pub struct MultiLayerGD {
    /// Learning Rate value.
    pub learning_rate: f64,
}

impl MultiLayerGD {
    /// Creates a new instance.
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    /// Computes step layer.
    pub fn step_layer(
        &self,
        weights: &mut Array2<f64>,
        bias: Option<&mut Array2<f64>>,
        grad_w: &Array2<f64>,
        grad_b: &Array2<f64>,
    ) {
        *weights = weights.to_owned() - &(grad_w * self.learning_rate);
        if let Some(bias) = bias {
            *bias = bias.to_owned() - &(grad_b * self.learning_rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_optimizers_update() {
        let mut w = array![[1.0, 2.0]];
        let g = array![[0.5, -0.5]];
        let mut gd = get_optimizer("gd", 0.1).expect("gd");
        gd.step(&mut w, &g);
        assert_ne!(w[[0, 0]], 1.0);

        let mut w2 = array![[1.0, 2.0]];
        let mut adam = get_optimizer("adam", 0.01).expect("adam");
        adam.step(&mut w2, &g);
        assert_ne!(w2[[0, 0]], 1.0);
        assert!(matches!(
            get_optimizer("bad", 0.1),
            Err(OptimizerError::UnknownOptimizer)
        ));
    }
}
