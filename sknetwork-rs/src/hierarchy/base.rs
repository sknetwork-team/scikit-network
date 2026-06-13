use crate::hierarchy::postprocess::{Dendrogram, split_dendrogram};

#[derive(Debug, Clone, Default)]
/// BaseHierarchyState value.
pub struct BaseHierarchyState {
    /// Dendrogram value.
    pub dendrogram: Option<Dendrogram>,
    /// Dendrogram Row value.
    pub dendrogram_row: Option<Dendrogram>,
    /// Dendrogram Col value.
    pub dendrogram_col: Option<Dendrogram>,
    /// Dendrogram Full value.
    pub dendrogram_full: Option<Dendrogram>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by base hierarchy error operations.
pub enum BaseHierarchyError {
    /// Indicates not fitted.
    NotFitted,
}

impl BaseHierarchyState {
    /// Creates a new instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes init vars.
    pub fn init_vars(&mut self) {
        self.dendrogram = None;
        self.dendrogram_row = None;
        self.dendrogram_col = None;
        self.dendrogram_full = None;
    }

    /// Runs the predict step.
    ///
    /// # Errors
    ///
    /// Returns [`BaseHierarchyError`] on failure.
    pub fn predict(&self, columns: bool) -> Result<Dendrogram, BaseHierarchyError> {
        if columns {
            self.dendrogram_col
                .clone()
                .ok_or(BaseHierarchyError::NotFitted)
        } else {
            self.dendrogram.clone().ok_or(BaseHierarchyError::NotFitted)
        }
    }

    /// Runs the transform step.
    ///
    /// # Errors
    ///
    /// Returns [`BaseHierarchyError`] on failure.
    pub fn transform(&self) -> Result<Dendrogram, BaseHierarchyError> {
        self.dendrogram.clone().ok_or(BaseHierarchyError::NotFitted)
    }

    /// Computes split vars.
    pub fn split_vars(&mut self, shape: (usize, usize)) -> &mut Self {
        if let Some(d) = self.dendrogram.clone() {
            let (dr, dc) = split_dendrogram(&d, shape);
            self.dendrogram_full = Some(d.clone());
            self.dendrogram = Some(dr.clone());
            self.dendrogram_row = Some(dr);
            self.dendrogram_col = Some(dc);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_hierarchy_state() {
        let mut s = BaseHierarchyState::new();
        s.dendrogram = Some(vec![
            [0.0, 1.0, 0.0, 2.0],
            [2.0, 3.0, 0.0, 2.0],
            [4.0, 5.0, 1.0, 4.0],
        ]);
        s.split_vars((2, 2));
        assert_eq!(s.predict(false).expect("row dendrogram").len(), 1);
        assert_eq!(s.predict(true).expect("col dendrogram").len(), 1);
        assert_eq!(s.transform().expect("transform").len(), 1);
        assert_eq!(s.dendrogram_full.clone().unwrap_or_default().len(), 3);
    }
}
