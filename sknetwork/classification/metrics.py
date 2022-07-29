#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in July 2020
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Thomas Bonald <thomas.bonald@telecom-paris.fr>
"""
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
