#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from typing import Optional, Union

import numpy as np

from sknetwork.utils.check import check_seeds


def stack_seeds(n_row: int, n_col: int, seeds_row: Optional[Union[np.ndarray, dict]],
                seeds_col: Optional[Union[np.ndarray, dict]] = None, default_value: float = -1) -> np.ndarray:
    """Process seeds for rows and columns and stack the results into a single vector."""
    if seeds_row is None and seeds_col is None:
        seeds_row = np.ones(n_row)
        seeds_col = default_value * np.ones(n_col)
    elif seeds_row is None:
        seeds_row = default_value * np.ones(n_row)
    elif seeds_col is None:
        seeds_col = default_value * np.ones(n_col)
    seeds_row = check_seeds(seeds_row, n_row)
    seeds_col = check_seeds(seeds_col, n_col)
    return np.hstack((seeds_row, seeds_col))


def seeds2probs(n: int, seeds: Union[dict, np.ndarray] = None) -> np.ndarray:
    """Transform seeds into probability vector.

    Parameters
    ----------
    n : int
        Total number of samples.
    seeds :
        If ``None``, the uniform distribution is used.
        Otherwise, a non-negative, non-zero vector or a dictionary must be provided.

    Returns
    -------
    probs: np.ndarray
        A probability vector.
    """
    if seeds is None:
        return np.ones(n) / n
    else:
        seeds = check_seeds(seeds, n)
        probs = np.zeros_like(seeds, dtype=float)
        ix = (seeds > 0)
        probs[ix] = seeds[ix]
        w: float = probs.sum()
        if w > 0:
            return probs / w
        else:
            raise ValueError('At least one seeds must have a positive probability.')
