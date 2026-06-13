/// Fitted regression values with optional bipartite splits.
#[derive(Debug, Clone, Default)]
pub struct BaseRegressorState {
    /// Row values (or full values for square graphs).
    pub values: Vec<f64>,
    /// Row-node values when the input is bipartite.
    pub values_row: Option<Vec<f64>>,
    /// Column-node values when the input is bipartite.
    pub values_col: Option<Vec<f64>>,
}

/// Errors raised while reading fitted regression state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaseRegressorError {
    /// The estimator has not been fitted yet.
    NotFitted,
}

impl BaseRegressorState {
    /// Creates an empty regression state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns fitted row or column values.
    ///
    /// # Errors
    /// Returns [`BaseRegressorError::NotFitted`] when called before values
    /// are stored or when column values are unavailable.
    pub fn predict(&self, columns: bool) -> Result<Vec<f64>, BaseRegressorError> {
        if self.values.is_empty() && self.values_row.is_none() && self.values_col.is_none() {
            return Err(BaseRegressorError::NotFitted);
        }
        if columns {
            self.values_col
                .clone()
                .ok_or(BaseRegressorError::NotFitted)
        } else {
            Ok(self.values.clone())
        }
    }

    /// Splits unified values into row and column partitions.
    ///
    /// When `values.len() < n_row`, stores all values as row values and leaves
    /// column values empty.
    pub fn split_vars(&mut self, n_row: usize) {
        if self.values.len() < n_row {
            self.values_row = Some(self.values.clone());
            self.values_col = Some(Vec::new());
            return;
        }
        self.values_row = Some(self.values[..n_row].to_vec());
        self.values_col = Some(self.values[n_row..].to_vec());
        self.values = self.values_row.clone().unwrap_or_default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict() {
        let mut state = BaseRegressorState::new();
        state.values = vec![0.1, 0.2, 0.3];
        assert_eq!(
            state.predict(false).expect("row prediction"),
            vec![0.1, 0.2, 0.3]
        );
        assert!(state.predict(true).is_err());
    }

    #[test]
    fn test_split_vars() {
        let mut state = BaseRegressorState::new();
        state.values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        state.split_vars(2);
        assert_eq!(state.values_row.clone().unwrap_or_default(), vec![1.0, 2.0]);
        assert_eq!(
            state.values_col.clone().unwrap_or_default(),
            vec![3.0, 4.0, 5.0]
        );
        assert_eq!(state.values, vec![1.0, 2.0]);
    }
}
