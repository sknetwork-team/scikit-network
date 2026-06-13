use ndarray::{Array1, Array2};
use sprs::CsMat;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by sparse lrerror operations.
pub enum SparseLRError {
    /// Indicates invalid low rank shape.
    InvalidLowRankShape,
}

#[derive(Debug, Clone)]
/// SumResult enum.
pub enum SumResult {
    /// Indicates total.
    Total(f64),
    /// Indicates axis.
    Axis(Array1<f64>),
}

#[derive(Debug, Clone)]
/// SparseLR value.
pub struct SparseLR {
    /// Sparse Mat value.
    pub sparse_mat: CsMat<f64>,
    /// Low Rank Tuples value.
    pub low_rank_tuples: Vec<(Array1<f64>, Array1<f64>)>,
}

impl SparseLR {
    /// Creates a new instance.
    pub fn new(
        sparse_mat: &CsMat<f64>,
        low_rank_tuples: Vec<(Array1<f64>, Array1<f64>)>,
    ) -> Result<Self, SparseLRError> {
        let (n_row, n_col) = sparse_mat.shape();
        for (x, y) in &low_rank_tuples {
            if x.len() != n_row || y.len() != n_col {
                return Err(SparseLRError::InvalidLowRankShape);
            }
        }
        Ok(Self {
            sparse_mat: sparse_mat.clone(),
            low_rank_tuples,
        })
    }

    /// Computes neg.
    pub fn neg(&self) -> Self {
        let mut sparse = self.sparse_mat.clone();
        for v in sparse.data_mut() {
            *v *= -1.0;
        }
        let low_rank_tuples = self
            .low_rank_tuples
            .iter()
            .map(|(x, y)| (x.mapv(|v| -v), y.clone()))
            .collect();
        Self {
            sparse_mat: sparse,
            low_rank_tuples,
        }
    }

    /// Computes add sparse.
    pub fn add_sparse(&self, other: &CsMat<f64>) -> Self {
        Self {
            sparse_mat: &self.sparse_mat + other,
            low_rank_tuples: self.low_rank_tuples.clone(),
        }
    }

    /// Computes add slr.
    pub fn add_slr(&self, other: &SparseLR) -> Self {
        let mut low_rank_tuples = self.low_rank_tuples.clone();
        low_rank_tuples.extend(other.low_rank_tuples.clone());
        Self {
            sparse_mat: &self.sparse_mat + &other.sparse_mat,
            low_rank_tuples,
        }
    }

    /// Computes sub slr.
    pub fn sub_slr(&self, other: &SparseLR) -> Self {
        self.add_slr(&other.neg())
    }

    /// Computes scale.
    pub fn scale(&self, factor: f64) -> Self {
        let mut sparse = self.sparse_mat.clone();
        for v in sparse.data_mut() {
            *v *= factor;
        }
        let low_rank_tuples = self
            .low_rank_tuples
            .iter()
            .map(|(x, y)| (x.mapv(|v| factor * v), y.clone()))
            .collect();
        Self {
            sparse_mat: sparse,
            low_rank_tuples,
        }
    }

    /// Computes dot vec.
    pub fn dot_vec(&self, matrix: &Array1<f64>) -> Array1<f64> {
        let mut prod = Array1::<f64>::zeros(self.sparse_mat.rows());
        for (i, row) in self.sparse_mat.outer_iterator().enumerate() {
            let mut s = 0.0;
            for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                s += v * matrix[j];
            }
            prod[i] = s;
        }
        for (x, y) in &self.low_rank_tuples {
            let alpha = matrix.dot(y);
            for i in 0..x.len() {
                prod[i] += x[i] * alpha;
            }
        }
        prod
    }

    /// Computes dot mat.
    pub fn dot_mat(&self, matrix: &Array2<f64>) -> Array2<f64> {
        let mut prod = Array2::<f64>::zeros((self.sparse_mat.rows(), matrix.ncols()));
        for (i, row) in self.sparse_mat.outer_iterator().enumerate() {
            for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                for c in 0..matrix.ncols() {
                    prod[[i, c]] += v * matrix[[j, c]];
                }
            }
        }
        for (x, y) in &self.low_rank_tuples {
            for c in 0..matrix.ncols() {
                let alpha = matrix.column(c).dot(y);
                for i in 0..x.len() {
                    prod[[i, c]] += x[i] * alpha;
                }
            }
        }
        prod
    }

    /// Computes transpose.
    pub fn transpose(&self) -> Self {
        let tuples = self
            .low_rank_tuples
            .iter()
            .map(|(x, y)| (y.clone(), x.clone()))
            .collect();
        Self {
            sparse_mat: self.sparse_mat.transpose_view().to_csr(),
            low_rank_tuples: tuples,
        }
    }

    /// Computes left sparse dot.
    pub fn left_sparse_dot(&self, matrix: &CsMat<f64>) -> Self {
        let tuples = self
            .low_rank_tuples
            .iter()
            .map(|(x, y)| {
                let mut mx = Array1::<f64>::zeros(matrix.rows());
                for (i, row) in matrix.outer_iterator().enumerate() {
                    let mut s = 0.0;
                    for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                        s += v * x[j];
                    }
                    mx[i] = s;
                }
                (mx, y.clone())
            })
            .collect();
        Self {
            sparse_mat: matrix * &self.sparse_mat,
            low_rank_tuples: tuples,
        }
    }

    /// Computes right sparse dot.
    pub fn right_sparse_dot(&self, matrix: &CsMat<f64>) -> Self {
        let tuples = self
            .low_rank_tuples
            .iter()
            .map(|(x, y)| {
                let mut mty = Array1::<f64>::zeros(matrix.cols());
                let mt = matrix.transpose_view().to_csr();
                for (i, row) in mt.outer_iterator().enumerate() {
                    let mut s = 0.0;
                    for (&j, &v) in row.indices().iter().zip(row.data().iter()) {
                        s += v * y[j];
                    }
                    mty[i] = s;
                }
                (x.clone(), mty)
            })
            .collect();
        Self {
            sparse_mat: &self.sparse_mat * matrix,
            low_rank_tuples: tuples,
        }
    }

    /// Computes sum.
    pub fn sum(&self, axis: Option<usize>) -> SumResult {
        match axis {
            Some(0) => {
                let ones = Array1::<f64>::ones(self.sparse_mat.rows());
                SumResult::Axis(self.transpose().dot_vec(&ones))
            }
            Some(1) => {
                let ones = Array1::<f64>::ones(self.sparse_mat.cols());
                SumResult::Axis(self.dot_vec(&ones))
            }
            _ => {
                let ones = Array1::<f64>::ones(self.sparse_mat.cols());
                SumResult::Total(self.dot_vec(&ones).sum())
            }
        }
    }

    /// Computes astype.
    pub fn astype(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};
    use sprs::TriMat;

    use super::*;

    fn house() -> CsMat<f64> {
        let mut tri = TriMat::<f64>::new((5, 5));
        let edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (2, 4)];
        for (u, v) in edges {
            tri.add_triplet(u, v, 1.0);
            tri.add_triplet(v, u, 1.0);
        }
        tri.to_csr::<usize>()
    }

    fn star_wars() -> CsMat<f64> {
        // Row sums [2, 1, 3, 2]
        let mut tri = TriMat::<f64>::new((4, 3));
        let edges = [
            (0, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
            (3, 2),
        ];
        for (u, v) in edges {
            tri.add_triplet(u, v, 1.0);
        }
        tri.to_csr::<usize>()
    }

    #[test]
    fn test_init() {
        assert!(matches!(
            SparseLR::new(&house(), vec![(Array1::ones(5), Array1::ones(4))]),
            Err(SparseLRError::InvalidLowRankShape)
        ));
        assert!(matches!(
            SparseLR::new(&house(), vec![(Array1::ones(4), Array1::ones(5))]),
            Err(SparseLRError::InvalidLowRankShape)
        ));
    }

    #[test]
    fn test_addition() {
        let undirected = SparseLR::new(&house(), vec![(Array1::ones(5), Array1::ones(5))]).unwrap();
        let addition = undirected.add_slr(&undirected);
        let expected = SparseLR::new(
            &(&house() * 2.0),
            vec![(Array1::ones(5), Array1::ones(5) * 2.0)],
        )
        .unwrap();
        assert_eq!((&addition.sparse_mat - &expected.sparse_mat).nnz(), 0);
        let x = array![0.2, 0.5, 0.1, 0.9, 0.3];
        let err = (&addition.dot_vec(&x) - &expected.dot_vec(&x))
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-12);
    }

    #[test]
    fn test_operations_and_product_transpose_sum() {
        let undirected = SparseLR::new(&house(), vec![(Array1::ones(5), Array1::ones(5))]).unwrap();
        let bipartite =
            SparseLR::new(&star_wars(), vec![(Array1::ones(4), Array1::ones(3))]).unwrap();

        let adjacency = undirected.sparse_mat.clone();
        let mut slr = undirected.neg();
        slr = slr.add_sparse(&adjacency);
        slr = slr.sub_slr(&SparseLR::new(&adjacency, vec![]).unwrap());
        slr = slr.left_sparse_dot(&adjacency);
        slr = slr.right_sparse_dot(&adjacency);
        let _ = slr.astype();

        let prod = undirected.dot_vec(&Array1::ones(5));
        assert_eq!(prod.len(), 5);
        let prod = bipartite.dot_vec(&Array1::ones(3));
        let ref1 = array![5., 4., 6., 5.];
        assert!((&prod - &ref1).iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-12);
        let prod = bipartite.dot_vec(&(Array1::ones(3) * 0.5));
        let ref2 = array![2.5, 2., 3., 2.5];
        assert!((&prod - &ref2).iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-12);
        let prod = bipartite.scale(2.0).dot_vec(&(Array1::ones(3) * 0.5));
        assert!(
            (&prod - &(ref2 * 2.0))
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
                < 1e-12
        );

        let transposed = undirected.transpose();
        assert_eq!((&undirected.sparse_mat - &transposed.sparse_mat).nnz(), 0);
        let transposed = bipartite.transpose();
        let (x, y) = &transposed.low_rank_tuples[0];
        assert_eq!(x.to_vec(), vec![1.0; 3]);
        assert_eq!(y.to_vec(), vec![1.0; 4]);

        match undirected.sum(Some(0)) {
            SumResult::Axis(v) => assert_eq!(v.len(), 5),
            _ => panic!("unexpected sum type"),
        }
        match undirected.sum(Some(1)) {
            SumResult::Axis(v) => assert_eq!(v.len(), 5),
            _ => panic!("unexpected sum type"),
        }
        match undirected.sum(None) {
            SumResult::Total(v) => assert!(v > 0.0),
            _ => panic!("unexpected sum type"),
        }
    }
}
