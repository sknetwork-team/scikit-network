#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in July 2020
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Thomas Bonald <thomas.bonald@telecom-paris.fr>
"""
from typing import Union, Tuple

import numpy as np
from scipy import sparse

from sknetwork.utils.check import check_vector_format


def get_accuracy_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Return the proportion of correctly labeled samples.
    Negative labels ignored.

    Parameters
    ----------
    labels_true : np.ndarray
        True labels.
    labels_pred : np.ndarray
        Predicted labels

    Returns
    -------
    accuracy : float
        A score between 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> labels_true = np.array([0, 0, 1, 1])
    >>> labels_pred = np.array([0, 0, 0, 1])
    >>> get_accuracy_score(labels_true, labels_pred)
    0.75
    """
    check_vector_format(labels_true, labels_pred)
    mask = (labels_true >= 0) & (labels_pred >= 0)
    if np.sum(mask):
        return np.mean(labels_true[mask] == labels_pred[mask])
    else:
        raise ValueError('No sample with both true non-negative label and predicted non-negative label.')


def get_confusion_matrix(labels_true: np.ndarray, labels_pred: np.ndarray) -> sparse.csr_matrix:
    """Return the confusion matrix in sparse format (true labels on rows, predicted labels on columns).
    Negative labels ignored.

    Parameters
    ----------
    labels_true : np.ndarray
        True labels.
    labels_pred : np.ndarray
        Predicted labels

    Returns
    -------
    confusion matrix : sparse.csr_matrix
        Confusion matrix.

    Examples
    --------
    >>> import numpy as np
    >>> labels_true = np.array([0, 0, 1, 1])
    >>> labels_pred = np.array([0, 0, 0, 1])
    >>> get_confusion_matrix(labels_true, labels_pred).toarray()
    array([[2, 0],
           [1, 1]])
    """
    check_vector_format(labels_true, labels_pred)
    mask = (labels_true >= 0) & (labels_pred >= 0)
    if np.sum(mask):
        n_labels = max(max(labels_true), max(labels_pred)) + 1
        row = labels_true[mask]
        col = labels_pred[mask]
        data = np.ones(np.sum(mask), dtype=int)
        return sparse.csr_matrix((data, (row, col)), shape=(n_labels, n_labels))
    else:
        raise ValueError('No sample with both true non-negative label and predicted non-negative label.')


def get_f1_scores(labels_true: np.ndarray, labels_pred: np.ndarray, return_precision_recall: bool = False) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return the f1 scores of classification, per label.
    Negative labels ignored.

    Parameters
    ----------
    labels_true : np.ndarray
        True labels.
    labels_pred : np.ndarray
        Predicted labels
    return_precision_recall : bool
        If ``True``, also return precisions and recalls.

    Returns
    -------
    scores, [precisions, recalls] : np.ndarray
        F1 scores (between 0 and 1). Optionally, F1 scores, precisions, recalls.
    Examples
    --------
    >>> import numpy as np
    >>> labels_true = np.array([0, 0, 1, 1])
    >>> labels_pred = np.array([0, 0, 0, 1])
    >>> np.round(get_f1_scores(labels_true, labels_pred), 2)
    array([0.8 , 0.67])
    """
    confusion = get_confusion_matrix(labels_true, labels_pred)
    n_labels = confusion.shape[0]
    counts_correct = confusion.diagonal()
    counts_true = confusion.dot(np.ones(n_labels))
    counts_pred = confusion.T.dot(np.ones(n_labels))
    mask = counts_true > 0
    recalls = np.zeros(n_labels)
    recalls[mask] = counts_correct[mask] / counts_true[mask]
    precisions = np.zeros(n_labels)
    mask = counts_pred > 0
    precisions[mask] = counts_correct[mask] / counts_pred[mask]
    f1_scores = np.zeros(n_labels)
    mask = (counts_true > 0) & (counts_pred > 0)
    f1_scores[mask] = 2 / (1 / precisions[mask] + 1 / recalls[mask])
    if return_precision_recall:
        return f1_scores, precisions, recalls
    else:
        return f1_scores
