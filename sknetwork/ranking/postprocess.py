#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
import numpy as np


def top_k(scores: np.ndarray, k: int = 1):
    """Return the indices of the k elements of highest values.

    Parameters
    ----------
    scores : np.ndarray
        Array of values.
    k : int
        Number of elements to return.

    Examples
    --------
    >>> top_k([1, 3, 2], k=2)
    array([1, 2])
    """
    return np.argpartition(-np.array(scores), k)[:k]
