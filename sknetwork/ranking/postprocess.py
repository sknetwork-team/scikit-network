#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 31 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
import numpy as np


def top_k(scores: np.ndarray, k: int = 1):
    """Index of the k elements with highest value.

    Parameters
    ----------
    scores : np.ndarray
        Array of values.
    k : int
        Number of elements to return.

    Examples
    --------
    >>> scores = np.array([0, 1, 0, 0.5])
    >>> top_k(scores, k=2)
    array([1, 3])

    Notes
    -----
    This is a basic implementation that sorts the entire array to find its top k elements.
    """
    return np.argsort(-scores)[:k]
