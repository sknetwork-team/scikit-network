use ndarray::Array1;
use ndarray::Array2;
use sprs::{CsMat, TriMat};

use crate::gnn::activation::get_activation;
use crate::gnn::base_layer::{BaseLayer, LayerError, LayerParams};

#[derive(Debug, Clone)]
/// Convolution value.
pub struct Convolution {
    /// Params value.
    pub params: LayerParams,
    /// Activation value.
    pub activation: String,
    /// Self Embeddings value.
    pub self_embeddings: bool,
    /// Normalization value.
    pub normalization: String,
    /// Layer Type value.
    pub layer_type: String,
    /// Sample Size value.
    pub sample_size: Option<usize>,
}

impl Convolution {
    /// Creates a new instance.
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        activation: &str,
        use_bias: bool,
        layer_type: &str,
        sample_size: Option<usize>,
        normalization: &str,
        self_embeddings: bool,
    ) -> Self {
        Self {
            params: LayerParams::new(in_dim, out_dim, use_bias),
            activation: activation.to_string(),
            self_embeddings,
            normalization: normalization.to_lowercase(),
            layer_type: layer_type.to_string(),
            sample_size,
        }
    }
}

fn sparse_dense_mul(a: &CsMat<f64>, x: &Array2<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((a.rows(), x.ncols()));
    for i in 0..a.rows() {
        if let Some(row) = a.outer_view(i) {
            for (j, v) in row.iter() {
                for c in 0..x.ncols() {
                    out[[i, c]] += *v * x[[j, c]];
                }
            }
        }
    }
    out
}

fn row_degrees(adjacency: &CsMat<f64>) -> Array1<f64> {
    let mut weights = Array1::<f64>::zeros(adjacency.rows());
    for (i, row) in adjacency.outer_iterator().enumerate() {
        weights[i] = row.data().iter().sum::<f64>();
    }
    weights
}

fn scale_rows(mat: &CsMat<f64>, scale: &Array1<f64>) -> CsMat<f64> {
    let mut tri = TriMat::<f64>::new(mat.shape());
    for (i, row) in mat.outer_iterator().enumerate() {
        let s = scale[i];
        for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
            tri.add_triplet(i, j, s * v);
        }
    }
    tri.to_csr::<usize>()
}

fn scale_cols(mat: &CsMat<f64>, scale: &Array1<f64>) -> CsMat<f64> {
    let mut tri = TriMat::<f64>::new(mat.shape());
    for (i, row) in mat.outer_iterator().enumerate() {
        for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
            tri.add_triplet(i, j, scale[j] * v);
        }
    }
    tri.to_csr::<usize>()
}

fn normalize_adjacency(adjacency: &CsMat<f64>, normalization: &str) -> CsMat<f64> {
    match normalization {
        "none" => adjacency.to_owned(),
        "left" => {
            let inv = row_degrees(adjacency).mapv(|w| if w != 0.0 { 1.0 / w } else { 0.0 });
            scale_rows(adjacency, &inv)
        }
        "right" => {
            let inv = row_degrees(adjacency).mapv(|w| if w != 0.0 { 1.0 / w } else { 0.0 });
            scale_cols(adjacency, &inv)
        }
        "both" => {
            let inv = row_degrees(adjacency).mapv(|w| if w > 0.0 { 1.0 / w.sqrt() } else { 0.0 });
            let left = scale_rows(adjacency, &inv);
            scale_cols(&left, &inv)
        }
        _ => adjacency.to_owned(),
    }
}

fn add_self_loops(adjacency: &CsMat<f64>) -> CsMat<f64> {
    let (r, c) = adjacency.shape();
    let mut tri = TriMat::<f64>::new((r, c));
    for (i, row) in adjacency.outer_iterator().enumerate() {
        for (j, v) in row.iter() {
            tri.add_triplet(i, j, *v);
        }
    }
    for i in 0..r.min(c) {
        tri.add_triplet(i, i, 1.0);
    }
    tri.to_csr::<usize>()
}

impl BaseLayer for Convolution {
    fn forward(&self, adjacency: &CsMat<f64>, features: &Array2<f64>) -> Result<Array2<f64>, LayerError> {
        self.forward_with_embedding(adjacency, features)
            .map(|(_, output)| output)
    }
}

impl Convolution {
    /// Forward pass returning pre-activation embedding and post-activation output.
    pub fn forward_with_embedding(
        &self,
        adjacency: &CsMat<f64>,
        features: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), LayerError> {
        if features.ncols() != self.params.weights.nrows() || adjacency.cols() != features.nrows() {
            return Err(LayerError::InvalidShape);
        }
        let mut effective = normalize_adjacency(adjacency, &self.normalization);
        if self.self_embeddings {
            effective = add_self_loops(&effective);
        }
        let ax = sparse_dense_mul(&effective, features);
        let mut z = ax.dot(&self.params.weights);
        if let Some(bias) = &self.params.bias {
            for i in 0..z.nrows() {
                for j in 0..z.ncols() {
                    z[[i, j]] += bias[[0, j]];
                }
            }
        }
        let act = get_activation(&self.activation).map_err(|_| LayerError::NotImplemented)?;
        Ok((z.clone(), act.output(&z)))
    }
}

/// Returns layer.
///
/// # Errors
///
/// Returns [`LayerError`] on failure.
pub fn get_layer(name: &str, in_dim: usize, out_dim: usize) -> Result<Convolution, LayerError> {
    match name.to_lowercase().as_str() {
        "convolution" | "gcn" | "conv" => Ok(Convolution::new(
            in_dim,
            out_dim,
            "relu",
            true,
            "gcn",
            None,
            "both",
            true,
        )),
        "sage" => Ok(Convolution::new(
            in_dim,
            out_dim,
            "relu",
            true,
            "sage",
            Some(25),
            "both",
            true,
        )),
        _ => Err(LayerError::NotImplemented),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use sprs::TriMat;

    use super::*;

    #[test]
    fn test_convolution_forward_shape() {
        let mut tri = TriMat::<f64>::new((3, 3));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(1, 1, 1.0);
        tri.add_triplet(2, 2, 1.0);
        let a = tri.to_csr::<usize>();
        let x = Array2::<f64>::ones((3, 4));
        let layer = Convolution::new(4, 2, "relu", true, "gcn", None, "both", true);
        let y = layer.forward(&a, &x).expect("forward");
        assert_eq!(y.shape(), &[3, 2]);
    }
}
