#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 4, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import sparse
from typing import Union, Optional
from sknetwork import is_numba_available


def has_nonnegative_entries(entry: Union[sparse.csr_matrix, np.ndarray]) -> bool:
    """Checks whether the array has non negative entries.
    """
    if type(entry) == sparse.csr_matrix:
        return np.all(entry.data >= 0)
    else:
        return np.all(entry >= 0)


def has_positive_entries(entry: Union[sparse.csr_matrix, np.ndarray]) -> bool:
    """Checks whether the array has positive entries.
    """
    if type(entry) == sparse.csr_matrix:
        return np.all(entry.data > 0)
    else:
        return np.all(entry > 0)


def is_proba_array(entry: np.ndarray) -> bool:
    """Checks whether each line of the array has non negative entries which sum to 1.
    """
    if len(entry.shape) == 1:
        return has_nonnegative_entries(entry) and np.isclose(entry.sum(), 1)
    elif len(entry.shape) == 2:
        n_samples, n_features = entry.shape
        err = entry.dot(np.ones(n_features)) - np.ones(n_samples)
        return has_nonnegative_entries(entry) and np.isclose(np.linalg.norm(err), 0)
    else:
        raise TypeError('entry must be one or two-dimensional array.')


def is_square(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> bool:
    """Checks whether the matrix is square.
    """
    return adjacency.shape[0] == adjacency.shape[1]


def is_symmetric(adjacency: Union[sparse.csr_matrix, np.ndarray], tol: float = 1e-10) -> bool:
    """Checks whether the matrix is symmetric.
    """
    sym_error = adjacency - adjacency.T
    return np.all(np.abs(sym_error.data) <= tol)


def make_weights(distribution: str, adjacency: sparse.csr_matrix) -> np.ndarray:
    """Returns an array of weights from a matrix and a desired distribution.

       Parameters
       ----------
       distribution:
           Distribution for node sampling. Only ``'degree'`` or ``'uniform'`` are accepted.
       adjacency:
           The adjacency matrix of the graph.

       Returns
       -------
       node_weights: np.ndarray
           Valid weights of nodes.

    """
    n_weights = adjacency.shape[0]
    if distribution == 'degree':
        node_weights_vec = adjacency.dot(np.ones(adjacency.shape[1]))
    elif distribution == 'uniform':
        node_weights_vec = np.ones(n_weights)
    else:
        raise ValueError('Unknown distribution of node weights.')
    return node_weights_vec


def check_engine(engine: str) -> str:
    """Checks if the desired engine is available and returns Numba whenever possible rather than Python if asked for
    the ``'default'`` engine.
    """
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


def check_format(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> sparse.csr_matrix:
    """Checks whether the matrix is an instance of a supported type (NumPy array or Scipy CSR matrix) and returns
    the corresponding Scipy CSR matrix.
    """
    if type(adjacency) not in {sparse.csr_matrix, np.ndarray}:
        raise TypeError('Adjacency must be in Scipy CSR format or Numpy ndarray format.')
    else:
        return sparse.csr_matrix(adjacency)


def check_is_proba(entry: Union[float, int]):
    """Checks whether the number is non-negative and less than or equal to 1.
    """
    if type(entry) not in [float, int]:
        raise TypeError('Probabilities must be floats (or ints if 0 or 1).')
    if entry < 0 or entry > 1:
        raise ValueError('Probabilities must have value between 0 and 1.')
    return entry


def check_weights(weights: Union['str', np.ndarray], adjacency: Union[sparse.csr_matrix, sparse.csc_matrix],
                  positive_entries: bool = False) -> np.ndarray:
    """Checks whether the weights are a valid distribution for the graph and returns a probability vector.

    Parameters
    ----------
    weights:
        Probabilities for node sampling in the null model. ``'degree'``, ``'uniform'`` or custom weights.
    adjacency:
        The adjacency matrix of the graph.
    positive_entries:
        If true, the weights must all be positive, if False, the weights must be nonnegative.

    Returns
    -------
    node_weights: np.ndarray
        Valid weights of nodes.

    """
    n_weights = adjacency.shape[0]
    if type(weights) == np.ndarray:
        if len(weights) != n_weights:
            raise ValueError('The number of node weights must match the number of nodes.')
        else:
            node_weights_vec = weights
    elif type(weights) == str:
        node_weights_vec = make_weights(weights, adjacency)
    else:
        raise TypeError(
            'Node weights must be a known distribution ("degree" or "uniform" string) or a custom NumPy array.')

    if positive_entries and not has_positive_entries(node_weights_vec):
        raise ValueError('Some of the weights are not positive.')
    else:
        if np.any(node_weights_vec < 0) or node_weights_vec.sum() <= 0:
            raise ValueError('Node weights must be non-negative with positive sum.')

    return node_weights_vec


def check_probs(weights: Union['str', np.ndarray], adjacency: Union[sparse.csr_matrix, sparse.csc_matrix],
                positive_entries: bool = False) -> np.ndarray:
    """Checks whether the weights are a valid distribution for the graph and returns a normalized probability vector.
    """
    weights = check_weights(weights, adjacency, positive_entries)
    return weights / np.sum(weights)


def check_random_state(random_state: Optional[Union[np.random.RandomState, int]]):
    """Checks whether the argument is a seed or a NumPy random state. If None, numpy.random is used by default.
    """
    if random_state is None or random_state is np.random:
        return np.random
    elif type(random_state) == int:
        return np.random.RandomState(random_state)
    elif type(random_state) == np.random.RandomState:
        return random_state
    else:
        raise TypeError('To specify a random state, pass the seed (as an int) or a NumPy random state object.')
