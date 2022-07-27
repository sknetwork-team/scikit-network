#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in July 2020
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the proportion of correctly labeled samples.

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
    >>> accuracy_score(y_true, y_pred)
    0.75
    """
    return (y_true == y_pred).mean()
