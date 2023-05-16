#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
import numpy as np


def top_k(scores: np.ndarray, k: int = 1, sort: bool = True):
    """Return the indices of the k elements of highest values.

    Parameters
    ----------
    scores : np.ndarray
        Array of values.
    k : int
        Number of elements to return.
    sort : bool
        If ``True``, sort the indices in decreasing order of value (element of highest value first).

    Examples
    --------
    >>> top_k([1, 3, 2], k=2)
    array([1, 2])
    """
    scores = np.array(scores)
    if k >= len(scores):
        if sort:
            index = np.argsort(-scores)
        else:
            index = np.arange(scores)
    else:
        index = np.argpartition(-scores, k)[:k]
        if sort:
            index = index[np.argsort(-scores[index])]
    return index
