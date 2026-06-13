use crate::gnn::base_layer::LayerError;
use crate::gnn::layer::{Convolution, get_layer};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by gnnutils error operations.
pub enum GNNUtilsError {
    /// Indicates invalid parameter.
    InvalidParameter,
    /// Indicates invalid loss config.
    InvalidLossConfig,
    /// Indicates layer.
    Layer(LayerError),
}

impl From<LayerError> for GNNUtilsError {
    fn from(value: LayerError) -> Self {
        Self::Layer(value)
    }
}

/// Validates early stopping.
///
/// # Errors
///
/// Returns [`GNNUtilsError`] on failure.
pub fn check_early_stopping(patience: isize) -> Result<(), GNNUtilsError> {
    if patience < 0 {
        Err(GNNUtilsError::InvalidParameter)
    } else {
        Ok(())
    }
}

/// Validates normalizations.
///
/// # Errors
///
/// Returns [`GNNUtilsError`] on failure.
pub fn check_normalizations(norm: &str) -> Result<(), GNNUtilsError> {
    match norm.to_lowercase().as_str() {
        "left" | "right" | "both" | "none" => Ok(()),
        _ => Err(GNNUtilsError::InvalidParameter),
    }
}

/// Validates loss.
///
/// # Errors
///
/// Returns [`GNNUtilsError`] on failure.
pub fn check_loss(final_activation: &str, loss_name: &str) -> Result<(), GNNUtilsError> {
    let fa = final_activation.to_lowercase();
    let ln = loss_name.to_lowercase();
    if (ln.contains("cross") && fa == "softmax") || (ln.contains("binary") && fa == "sigmoid") {
        Ok(())
    } else {
        Err(GNNUtilsError::InvalidLossConfig)
    }
}

/// Returns layers.
///
/// # Errors
///
/// Returns [`GNNUtilsError`] on failure.
pub fn get_layers(dims: &[usize], layer_name: &str) -> Result<Vec<Convolution>, GNNUtilsError> {
    if dims.len() < 2 {
        return Err(GNNUtilsError::InvalidParameter);
    }
    let mut out = Vec::with_capacity(dims.len() - 1);
    for w in dims.windows(2) {
        out.push(get_layer(layer_name, w[0], w[1])?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utils_checks() {
        assert!(check_early_stopping(0).is_ok());
        assert!(check_early_stopping(-1).is_err());
        assert!(check_normalizations("both").is_ok());
        assert!(check_normalizations("bad").is_err());
        assert!(check_loss("softmax", "cross_entropy").is_ok());
        assert!(check_loss("relu", "cross_entropy").is_err());
        assert_eq!(get_layers(&[4, 3, 2], "gcn").expect("layers").len(), 2);
    }
}
