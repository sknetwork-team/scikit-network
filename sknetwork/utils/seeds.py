#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
import warnings
from typing import Optional, Union

import numpy as np


def get_seeds(shape: tuple, seeds: Union[np.ndarray, dict], default_value: float = -1) -> np.ndarray:
    """Get seeds as array."""
    n = shape[0]
    if isinstance(seeds, np.ndarray):
        if len(seeds) != n:
            raise ValueError('Dimensions mismatch between adjacency and seeds vector.')
        else:
            seeds = seeds.astype(float)
    elif isinstance(seeds, dict):
        keys, values = np.array(list(seeds.keys())), np.array(list(seeds.values()))
        if np.min(values) < 0:
            warnings.warn(Warning("Negative values will not be taken into account."))
        seeds = default_value * np.ones(n)
        seeds[keys] = values
    else:
        seeds = np.ones(n)
    return seeds


def stack_seeds(shape: tuple, seeds_row: Optional[Union[np.ndarray, dict]],
                seeds_col: Optional[Union[np.ndarray, dict]] = None, default_value: float = -1) -> np.ndarray:
    """Process seeds for rows and columns and stack the results into a single vector."""
    n_row, n_col = shape
    if seeds_row is None and seeds_col is None:
        seeds_row = np.ones(n_row)
        seeds_col = default_value * np.ones(n_col)
    elif seeds_row is None:
        seeds_row = default_value * np.ones(n_row)
    elif seeds_col is None:
        seeds_col = default_value * np.ones(n_col)
    seeds_row = get_seeds(shape, seeds_row, default_value)
    seeds_col = get_seeds((n_col,), seeds_col, default_value)
    return np.hstack((seeds_row, seeds_col))


def seeds2probs(n: int, seeds: np.ndarray = None) -> np.ndarray:
    """Transform seed values into probability vector.

    Parameters
    ----------
    n : int
        Number of nodes.
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
        seeds = get_seeds((n,), seeds)
        probs = np.zeros_like(seeds, dtype=float)
        ix = (seeds > 0)
        probs[ix] = seeds[ix]
        w: float = probs.sum()
        if w > 0:
            return probs / w
        else:
            raise ValueError('At least one seed must have a positive value.')
