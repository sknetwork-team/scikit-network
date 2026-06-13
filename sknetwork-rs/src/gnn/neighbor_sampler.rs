use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::{random, SeedableRng};
use sprs::{CsMat, TriMat};

#[derive(Debug, Clone)]
/// UniformNeighborSampler value.
pub struct UniformNeighborSampler {
    /// Sample Size value.
    pub sample_size: usize,
    /// Random State value.
    pub random_state: Option<u64>,
}

impl UniformNeighborSampler {
    /// Creates a new instance.
    pub fn new(sample_size: usize) -> Self {
        Self {
            sample_size,
            random_state: None,
        }
    }

    /// Computes with random state.
    pub fn with_random_state(sample_size: usize, random_state: Option<u64>) -> Self {
        Self {
            sample_size,
            random_state,
        }
    }

    /// Computes sample.
    pub fn sample(&self, adjacency: &CsMat<f64>) -> CsMat<f64> {
        let mut rng = StdRng::seed_from_u64(self.random_state.unwrap_or_else(random));
        self.sample_rng(adjacency, &mut rng)
    }

    /// Computes sample rng.
    pub fn sample_rng(&self, adjacency: &CsMat<f64>, rng: &mut StdRng) -> CsMat<f64> {
        let (r, c) = adjacency.shape();
        let mut tri = TriMat::<f64>::new((r, c));
        for i in 0..r {
            if let Some(row) = adjacency.outer_view(i) {
                let indices = row.indices();
                let k = self.sample_size.min(indices.len());
                if k == indices.len() {
                    for &j in indices {
                        // Python-style sampled adjacencies are binary regardless of source weights.
                        tri.add_triplet(i, j, 1.0);
                    }
                    continue;
                }
                for local_idx in sample(rng, indices.len(), k) {
                    let j = indices[local_idx];
                    tri.add_triplet(i, j, 1.0);
                }
            }
        }
        tri.to_csr::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use sprs::TriMat;

    use super::*;

    #[test]
    fn test_sampler_degree_bound() {
        let mut tri = TriMat::<f64>::new((3, 3));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(0, 2, 1.0);
        let a = tri.to_csr::<usize>();
        let s = UniformNeighborSampler::new(1);
        let out = s.sample(&a);
        let deg0 = out.outer_view(0).map(|r| r.nnz()).unwrap_or(0);
        assert!(deg0 <= 1);
    }

    #[test]
    fn test_sampler_binarizes_weights() {
        let mut tri = TriMat::<f64>::new((1, 3));
        tri.add_triplet(0, 0, 3.5);
        tri.add_triplet(0, 1, 2.2);
        tri.add_triplet(0, 2, 9.9);
        let a = tri.to_csr::<usize>();
        let s = UniformNeighborSampler::new(2);
        let out = s.sample(&a);
        let row = out.outer_view(0).unwrap();
        assert!(row.data().iter().all(|v| (*v - 1.0).abs() < 1e-12));
    }

    #[test]
    fn test_sampler_seed_reproducible() {
        let mut tri = TriMat::<f64>::new((3, 4));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(0, 2, 1.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(1, 1, 1.0);
        tri.add_triplet(1, 3, 1.0);
        tri.add_triplet(2, 1, 1.0);
        tri.add_triplet(2, 2, 1.0);
        tri.add_triplet(2, 3, 1.0);
        let a = tri.to_csr::<usize>();

        let s1 = UniformNeighborSampler::with_random_state(2, Some(42));
        let s2 = UniformNeighborSampler::with_random_state(2, Some(42));
        let out1 = s1.sample(&a);
        let out2 = s2.sample(&a);
        assert_eq!(out1.nnz(), out2.nnz());
        for i in 0..out1.rows() {
            let r1 = out1.outer_view(i).unwrap();
            let r2 = out2.outer_view(i).unwrap();
            assert_eq!(r1.indices(), r2.indices());
            assert_eq!(r1.data(), r2.data());
        }
    }
}
