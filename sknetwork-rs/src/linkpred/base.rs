use sprs::CsMat;

/// Fitted link-prediction output held by linker estimators.
#[derive(Debug, Clone, Default)]
pub struct BaseLinkerState {
    /// Predicted link scores as a sparse matrix, when fitted.
    pub links: Option<CsMat<f64>>,
}

/// Errors raised by shared link-prediction state accessors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaseLinkerError {
    /// [`BaseLinkerState::predict`] called before links are stored.
    NotFitted,
}

impl BaseLinkerState {
    /// Creates an empty linker state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the fitted link-score matrix.
    ///
    /// # Errors
    /// Returns [`BaseLinkerError::NotFitted`] when called before `fit`.
    pub fn predict(&self) -> Result<CsMat<f64>, BaseLinkerError> {
        self.links.clone().ok_or(BaseLinkerError::NotFitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;

    #[test]
    fn test_predict() {
        let mut state = BaseLinkerState::new();
        assert!(state.predict().is_err());

        let mut tri = TriMat::<f64>::new((3, 3));
        tri.add_triplet(0, 1, 1.0);
        state.links = Some(tri.to_csr::<usize>());
        let links = state.predict().expect("links");
        assert_eq!(links.shape(), (3, 3));
        assert_eq!(links.nnz(), 1);
    }
}
