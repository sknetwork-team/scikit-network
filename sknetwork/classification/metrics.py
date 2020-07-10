#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
import numpy as np


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision

    :math:`P = \\dfrac{TP}{TP + FP}`.

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
    >>> precision_score(y_true, y_pred)
    0.75
    """
    return (y_true == y_pred).mean()
