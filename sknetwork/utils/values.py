#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in April 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
import warnings
from typing import Optional, Union

import numpy as np


def get_values(shape: tuple, values: Union[np.ndarray, list, dict], default_value: float = -1) -> np.ndarray:
    """Get values as array."""
    n = shape[0]
    if isinstance(values, list):
        values = np.array(values)
    if isinstance(values, np.ndarray):
        if len(values) != n:
            raise ValueError('Dimensions mismatch between adjacency and values.')
        else:
            values = values.astype(float)
    elif isinstance(values, dict):
        keys, values_ = np.array(list(values.keys())), np.array(list(values.values()))
        if np.min(values_) < 0:
            warnings.warn(Warning("Negative values will not be taken into account."))
        values = default_value * np.ones(n)
        values[keys] = values_
    else:
        values = np.ones(n)
    return values


def stack_values(shape: tuple, values_row: Optional[Union[np.ndarray, list, dict]],
                 values_col: Optional[Union[np.ndarray, list, dict]] = None, default_value: float = -1) -> np.ndarray:
    """Process values for rows and columns and stack the results into a single vector."""
    n_row, n_col = shape
    if values_row is None and values_col is None:
        values_row = np.ones(n_row)
        values_col = default_value * np.ones(n_col)
    elif values_row is None:
        values_row = default_value * np.ones(n_row)
    elif values_col is None:
        values_col = default_value * np.ones(n_col)
    values_row = get_values(shape, values_row, default_value)
    values_col = get_values((n_col,), values_col, default_value)
    return np.hstack((values_row, values_col))


def values2prob(n: int, values: np.ndarray = None) -> np.ndarray:
    """Transform seed values into probability vector.

    Parameters
    ----------
    n : int
        Number of nodes.
    values :
        If ``None``, the uniform distribution is used.
        Otherwise, a non-negative, non-zero vector or a dictionary must be provided.

    Returns
    -------
    probs: np.ndarray
        A probability vector.
    """
    if values is None:
        return np.ones(n) / n
    else:
        values = get_values((n,), values)
        probs = np.zeros_like(values, dtype=float)
        ix = (values > 0)
        probs[ix] = values[ix]
        if probs.sum() > 0:
            return probs / probs.sum()
        else:
            raise ValueError('At least one value must be positive.')
