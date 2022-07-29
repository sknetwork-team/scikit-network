#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in July 2020
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Thomas Bonald <thomas.bonald@telecom-paris.fr>
"""
import numpy as np

from sknetwork.utils.check import check_vector_format


def get_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the proportion of correctly labeled samples.
    Negative values ignored.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    precision : float
        A score between 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 0, 0, 1])
    >>> get_accuracy_score(y_true, y_pred)
    0.75
    """
    check_vector_format(y_true, y_pred)
    mask = (y_true >= 0) & (y_pred >= 0)
    if np.sum(mask):
        return np.mean(y_true[mask] == y_pred[mask])
    else:
        raise ValueError('Negative ')
