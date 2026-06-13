use ndarray::Array1;
use sprs::{CsMat, TriMat};

use crate::utils::neighbors::get_degrees;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors raised by membership error operations.
pub enum MembershipError {
    /// Indicates empty labels.
    EmptyLabels,
    /// Indicates negative label in labeled set.
    NegativeLabelInLabeledSet,
    /// Indicates label out of range.
    LabelOutOfRange,
}

/// Returns membership.
pub fn get_membership(
    labels: &Array1<i32>,
    n_labels: Option<usize>,
) -> Result<CsMat<f64>, MembershipError> {
    let n = labels.len();
    let n_cols = match n_labels {
        Some(k) => k,
        None => {
            let mut max_label: Option<i32> = None;
            for &label in labels {
                if label >= 0 {
                    max_label = Some(max_label.map_or(label, |m| m.max(label)));
                }
            }
            match max_label {
                Some(m) => (m as usize) + 1,
                None => return Err(MembershipError::EmptyLabels),
            }
        }
    };

    let mut tri = TriMat::<f64>::new((n, n_cols));
    for (i, &label) in labels.iter().enumerate() {
        if label < 0 {
            continue;
        }
        let col = usize::try_from(label).map_err(|_| MembershipError::NegativeLabelInLabeledSet)?;
        if col >= n_cols {
            return Err(MembershipError::LabelOutOfRange);
        }
        tri.add_triplet(i, col, 1.0);
    }
    Ok(tri.to_csr())
}

/// Computes from membership.
pub fn from_membership(membership: &CsMat<f64>) -> Array1<i32> {
    let degrees = get_degrees(membership, false);
    let n = membership.rows();
    let mut labels = Array1::<i32>::from_elem(n, -1);
    for i in 0..n {
        if degrees[i] > 0 {
            let label = membership
                .outer_view(i)
                .and_then(|row| row.indices().first().copied())
                .map(|x| x as i32)
                .unwrap_or(-1);
            labels[i] = label;
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_membership_roundtrip() {
        let labels = Array1::from_vec(vec![0, 0, 1, 2, 1, 1]);
        let membership = get_membership(&labels, None).unwrap();
        assert_eq!(membership.nnz(), 6);
        let restored = from_membership(&membership);
        assert_eq!(labels.to_vec(), restored.to_vec());
    }

    #[test]
    fn test_membership_with_unlabeled() {
        let labels = Array1::from_vec(vec![0, 0, 1, 2, 1, -1]);
        let membership = get_membership(&labels, None).unwrap();
        assert_eq!(membership.nnz(), 5);
        let restored = from_membership(&membership);
        assert_eq!(labels.to_vec(), restored.to_vec());
    }
}
