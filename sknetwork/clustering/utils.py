#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 4, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import sparse
from typing import Union


def check_engine(engine):
    try:
        from numba import njit, prange
        is_numba_available = True
    except ImportError:
        is_numba_available = False

    if engine == 'default':
        if is_numba_available:
            engine = 'numba'
        else:
            engine = 'python'
    elif engine == 'numba':
        if is_numba_available:
            engine = 'numba'
        else:
            raise ValueError('Numba is not available')
    elif engine == 'python':
        engine = 'python'
    else:
        raise ValueError('Engine must be default, python or numba.')
    return engine


def check_weights(weights: Union['str', np.ndarray],
                  adjacency: Union[sparse.csr_matrix, sparse.csc_matrix]) -> np.ndarray:
    """Checks whether the weights are a valid distribution for the graph and returns a probability vector.

    Parameters
    ----------
    weights:
        Probabilities for node sampling in the null model. ``'degree'``, ``'uniform'`` or custom weights.
    adjacency:
        The adjacency matrix of the graph

    Returns
    -------
        props: np.ndarray
            probability vector for node sampling.

    """
    n_weights = adjacency.shape[0]
    if type(weights) == np.ndarray:
        if len(weights) != n_weights:
            raise ValueError('The number of node weights must match the number of nodes.')
        else:
            node_weights_vec = weights
    elif type(weights) == str:
        if weights == 'degree':
            node_weights_vec = adjacency.dot(np.ones(adjacency.shape[1]))
        elif weights == 'uniform':
            node_weights_vec = np.ones(n_weights)
        else:
            raise ValueError('Unknown distribution of node weights.')
    else:
        raise TypeError(
            'Node weights must be a known distribution ("degree" or "uniform" string) or a custom NumPy array.')

    if np.any(node_weights_vec < 0) or node_weights_vec.sum() <= 0:
        raise ValueError('Node weights must be non-negative with positive sum.')
    else:
        return node_weights_vec / np.sum(node_weights_vec)
