//! Classification evaluation metrics.

use std::collections::BTreeSet;

use ndarray::Array1;
use sprs::{CsMat, TriMat};

/// Error type for classification metric computations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassificationMetricError {
    /// Label vectors have different lengths.
    DimensionMismatch,
    /// No samples with non-negative labels in both vectors.
    NoValidSamples,
    /// Binary F1 requires labels in `{0, 1}` only.
    NonBinaryLabels,
    /// `average` is not one of `micro`, `macro`, or `weighted`.
    UnknownAverage,
}

fn check_vector_format(
    labels_true: &Array1<i32>,
    labels_pred: &Array1<i32>,
) -> Result<(), ClassificationMetricError> {
    if labels_true.len() != labels_pred.len() {
        Err(ClassificationMetricError::DimensionMismatch)
    } else {
        Ok(())
    }
}

/// Computes the fraction of correctly classified valid samples.
///
/// Samples with negative labels in either vector are ignored.
///
/// # Errors
/// Returns [`ClassificationMetricError::DimensionMismatch`] when vector lengths
/// differ, or [`ClassificationMetricError::NoValidSamples`] when no valid pairs
/// remain after filtering.
pub fn get_accuracy_score(
    labels_true: &Array1<i32>,
    labels_pred: &Array1<i32>,
) -> Result<f64, ClassificationMetricError> {
    check_vector_format(labels_true, labels_pred)?;
    let valid: Vec<usize> = (0..labels_true.len())
        .filter(|&i| labels_true[i] >= 0 && labels_pred[i] >= 0)
        .collect();
    if valid.is_empty() {
        return Err(ClassificationMetricError::NoValidSamples);
    }
    let correct = valid
        .iter()
        .filter(|&&i| labels_true[i] == labels_pred[i])
        .count();
    Ok(correct as f64 / valid.len() as f64)
}

/// Builds a confusion matrix from true and predicted labels.
///
/// Negative labels are ignored. Matrix size follows the maximum observed label.
///
/// # Errors
/// Returns [`ClassificationMetricError::DimensionMismatch`] when vector lengths
/// differ, or [`ClassificationMetricError::NoValidSamples`] when no valid pairs
/// remain after filtering.
pub fn get_confusion_matrix(
    labels_true: &Array1<i32>,
    labels_pred: &Array1<i32>,
) -> Result<CsMat<usize>, ClassificationMetricError> {
    check_vector_format(labels_true, labels_pred)?;
    let valid: Vec<usize> = (0..labels_true.len())
        .filter(|&i| labels_true[i] >= 0 && labels_pred[i] >= 0)
        .collect();
    if valid.is_empty() {
        return Err(ClassificationMetricError::NoValidSamples);
    }
    let mut max_label = 0usize;
    for &i in &valid {
        let t = labels_true[i] as usize;
        let p = labels_pred[i] as usize;
        max_label = max_label.max(t.max(p));
    }
    let n_labels = max_label + 1;
    let mut tri = TriMat::<usize>::new((n_labels, n_labels));
    for &i in &valid {
        tri.add_triplet(labels_true[i] as usize, labels_pred[i] as usize, 1usize);
    }
    Ok(tri.to_csr::<usize>())
}

/// Computes per-class F1 scores with optional precision and recall vectors.
///
/// # Arguments
/// - `return_precision_recall`: Whether to include precision and recall arrays.
///
/// # Errors
/// Returns [`ClassificationMetricError::DimensionMismatch`] or
/// [`ClassificationMetricError::NoValidSamples`] via the confusion matrix step.
pub fn get_f1_scores(
    labels_true: &Array1<i32>,
    labels_pred: &Array1<i32>,
    return_precision_recall: bool,
) -> Result<(Array1<f64>, Option<Array1<f64>>, Option<Array1<f64>>), ClassificationMetricError> {
    let confusion = get_confusion_matrix(labels_true, labels_pred)?;
    let n_labels = confusion.rows();
    let mut counts_correct = vec![0.0; n_labels];
    let mut counts_true = vec![0.0; n_labels];
    let mut counts_pred = vec![0.0; n_labels];
    for i in 0..n_labels {
        counts_correct[i] = confusion.get(i, i).copied().unwrap_or(0) as f64;
        if let Some(row) = confusion.outer_view(i) {
            counts_true[i] = row.data().iter().map(|x| *x as f64).sum();
        }
    }
    let ct = confusion.transpose_view().to_csr();
    for i in 0..n_labels {
        if let Some(row) = ct.outer_view(i) {
            counts_pred[i] = row.data().iter().map(|x| *x as f64).sum();
        }
    }
    let mut recalls = vec![0.0; n_labels];
    let mut precisions = vec![0.0; n_labels];
    let mut f1 = vec![0.0; n_labels];
    for i in 0..n_labels {
        if counts_true[i] > 0.0 {
            recalls[i] = counts_correct[i] / counts_true[i];
        }
        if counts_pred[i] > 0.0 {
            precisions[i] = counts_correct[i] / counts_pred[i];
        }
        if recalls[i] > 0.0 && precisions[i] > 0.0 {
            f1[i] = 2.0 / (1.0 / precisions[i] + 1.0 / recalls[i]);
        }
    }
    if return_precision_recall {
        Ok((
            Array1::from_vec(f1),
            Some(Array1::from_vec(precisions)),
            Some(Array1::from_vec(recalls)),
        ))
    } else {
        Ok((Array1::from_vec(f1), None, None))
    }
}

/// Computes the binary F1 score for the positive class (`1`).
///
/// # Arguments
/// - `return_precision_recall`: Whether to include precision and recall values.
///
/// # Errors
/// Returns [`ClassificationMetricError::NonBinaryLabels`] when labels are not
/// restricted to `{0, 1}`, or errors propagated from [`get_f1_scores`].
pub fn get_f1_score(
    labels_true: &Array1<i32>,
    labels_pred: &Array1<i32>,
    return_precision_recall: bool,
) -> Result<(f64, Option<f64>, Option<f64>), ClassificationMetricError> {
    let values: BTreeSet<i32> = labels_true
        .iter()
        .chain(labels_pred.iter())
        .copied()
        .filter(|x| *x >= 0)
        .collect();
    if values != BTreeSet::from([0, 1]) {
        return Err(ClassificationMetricError::NonBinaryLabels);
    }
    let (f1, p, r) = get_f1_scores(labels_true, labels_pred, true)?;
    if return_precision_recall {
        Ok((
            f1[1],
            Some(p.unwrap_or_else(|| Array1::zeros(0))[1]),
            Some(r.unwrap_or_else(|| Array1::zeros(0))[1]),
        ))
    } else {
        Ok((f1[1], None, None))
    }
}

/// Computes an averaged F1 score across classes.
///
/// # Arguments
/// - `average`: One of `micro`, `macro`, or `weighted`.
///
/// # Errors
/// Returns [`ClassificationMetricError::UnknownAverage`] for unsupported modes,
/// or errors propagated from underlying metric helpers.
pub fn get_average_f1_score(
    labels_true: &Array1<i32>,
    labels_pred: &Array1<i32>,
    average: &str,
) -> Result<f64, ClassificationMetricError> {
    match average {
        "micro" => get_accuracy_score(labels_true, labels_pred),
        "macro" => {
            let (f1, _, _) = get_f1_scores(labels_true, labels_pred, false)?;
            Ok(f1.sum() / f1.len() as f64)
        }
        "weighted" => {
            let (f1, _, _) = get_f1_scores(labels_true, labels_pred, false)?;
            let mut counts = std::collections::HashMap::<usize, usize>::new();
            for &x in labels_true {
                if x >= 0 {
                    *counts.entry(x as usize).or_insert(0) += 1;
                }
            }
            let tot: usize = counts.values().sum();
            if tot == 0 {
                return Err(ClassificationMetricError::NoValidSamples);
            }
            let mut s = 0.0;
            for (label, c) in counts {
                s += f1[label] * c as f64;
            }
            Ok(s / tot as f64)
        }
        _ => Err(ClassificationMetricError::UnknownAverage),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_accuracy() {
        let labels_true = array![0, 1, 1, 2, 2, -1];
        let labels_pred1 = array![0, -1, 1, 2, 0, 0];
        let labels_pred2 = array![-1, -1, -1, -1, -1, 0];
        assert!((get_accuracy_score(&labels_true, &labels_pred1).unwrap() - 0.75).abs() < 1e-12);
        assert_eq!(
            get_accuracy_score(&labels_true, &labels_pred2),
            Err(ClassificationMetricError::NoValidSamples)
        );
    }

    #[test]
    fn test_confusion() {
        let labels_true = array![0, 1, 1, 2, 2, -1];
        let labels_pred1 = array![0, -1, 1, 2, 0, 0];
        let labels_pred2 = array![-1, -1, -1, -1, -1, 0];
        let confusion = get_confusion_matrix(&labels_true, &labels_pred1).unwrap();
        let data_sum: usize = confusion.data().iter().sum();
        assert_eq!(data_sum, 4);
        let diag_sum = (0..confusion.rows())
            .map(|i| confusion.get(i, i).copied().unwrap_or(0))
            .sum::<usize>();
        assert_eq!(diag_sum, 3);
        assert_eq!(
            get_accuracy_score(&labels_true, &labels_pred2),
            Err(ClassificationMetricError::NoValidSamples)
        );
    }

    #[test]
    fn test_confusion_ignores_invalid_label_domain_outliers() {
        let labels_true = array![0, 1, 2, 1_000_000];
        let labels_pred = array![0, 1, 2, -1];
        let confusion = get_confusion_matrix(&labels_true, &labels_pred).unwrap();
        assert_eq!(confusion.shape(), (3, 3));
        let data_sum: usize = confusion.data().iter().sum();
        assert_eq!(data_sum, 3);
    }

    #[test]
    fn test_f1_score() {
        let f1 = get_f1_score(&array![0, 0, 1], &array![0, 1, 1], false)
            .unwrap()
            .0;
        assert!((f1 - 0.666666666).abs() < 1e-2);
        let labels_true = array![0, 1, 1, 2, 2, -1];
        let labels_pred1 = array![0, -1, 1, 2, 0, 0];
        assert_eq!(
            get_f1_score(&labels_true, &labels_pred1, false),
            Err(ClassificationMetricError::NonBinaryLabels)
        );
    }

    #[test]
    fn test_f1_scores() {
        let labels_true = array![0, 1, 1, 2, 2, -1];
        let labels_pred1 = array![0, -1, 1, 2, 0, 0];
        let labels_pred2 = array![-1, -1, -1, -1, -1, 0];
        let (f1_scores, _, _) = get_f1_scores(&labels_true, &labels_pred1, false).unwrap();
        assert!(
            (f1_scores.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 0.666666666).abs() < 1e-2
        );
        let (f1_scores, precisions, recalls) =
            get_f1_scores(&labels_true, &labels_pred1, true).unwrap();
        let precisions = precisions.unwrap_or_else(|| Array1::zeros(0));
        let recalls = recalls.unwrap_or_else(|| Array1::zeros(0));
        assert!(
            (f1_scores.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 0.666666666).abs() < 1e-2
        );
        assert!((precisions[0] - 0.5).abs() < 1e-12 || (precisions[2] - 0.5).abs() < 1e-12);
        assert!((recalls[0] - 0.5).abs() < 1e-12 || (recalls[2] - 0.5).abs() < 1e-12);
        assert_eq!(
            get_f1_scores(&labels_true, &labels_pred2, false),
            Err(ClassificationMetricError::NoValidSamples)
        );
    }

    #[test]
    fn test_average_f1_score() {
        let labels_true = array![0, 1, 1, 2, 2, -1];
        let labels_pred1 = array![0, -1, 1, 2, 0, 0];
        let labels_pred2 = array![-1, -1, -1, -1, -1, 0];
        let f1 = get_average_f1_score(&labels_true, &labels_pred1, "macro").unwrap();
        assert!((f1 - 0.7777777).abs() < 1e-2);
        let f1 = get_average_f1_score(&labels_true, &labels_pred1, "micro").unwrap();
        assert!((f1 - 0.75).abs() < 1e-12);
        let f1 = get_average_f1_score(&labels_true, &labels_pred1, "weighted").unwrap();
        assert!((f1 - 0.8).abs() < 1e-12);
        assert_eq!(
            get_average_f1_score(&labels_true, &labels_pred2, "toto"),
            Err(ClassificationMetricError::UnknownAverage)
        );
    }
}
