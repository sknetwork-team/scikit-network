/// Compatibility trait for path-level parity with Python's `ranking.base`.
///
/// This keeps a stable surface for future shared ranking abstractions without
/// forcing existing ranking implementations to conform immediately.
pub trait BaseRanking {
    /// Fits the ranking estimator in place.
    ///
    /// # Errors
    /// Returns a human-readable message when fitting fails.
    fn fit(&mut self) -> Result<(), String>;

    /// Returns the fitted score vector.
    fn scores(&self) -> &[f64];
}
