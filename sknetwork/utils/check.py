#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 4, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
import warnings
from typing import Union, Optional

import numpy as np
from scipy import sparse


def has_nonnegative_entries(entry: Union[sparse.csr_matrix, np.ndarray]) -> bool:
    """True if the array has non negative entries."""
    if type(entry) == sparse.csr_matrix:
        return np.all(entry.data >= 0)
    else:
        return np.all(entry >= 0)


def check_nonnegative(entry: Union[sparse.csr_matrix, np.ndarray]):
    """Check whether the array has non negative entries."""
    if not has_nonnegative_entries(entry):
        raise ValueError('Only nonnegative values are expected.')
    else:
        return


def has_positive_entries(entry: np.ndarray) -> bool:
    """True if the array has positive entries."""
    if type(entry) != np.ndarray:
        raise TypeError('Entry must be a dense NumPy array.')
    else:
        return np.all(entry > 0)


def check_positive(entry: Union[sparse.csr_matrix, np.ndarray]):
    """Check whether the array has positive entries."""
    if not has_positive_entries(entry):
        raise ValueError('Only positive values are expected.')
    else:
        return


def is_proba_array(entry: np.ndarray) -> bool:
    """True if each line of the array has non negative entries which sum to 1."""
    if len(entry.shape) == 1:
        return has_nonnegative_entries(entry) and np.isclose(entry.sum(), 1)
    elif len(entry.shape) == 2:
        n_row, n_col = entry.shape
        err = entry.dot(np.ones(n_col)) - np.ones(n_row)
        return has_nonnegative_entries(entry) and np.isclose(np.linalg.norm(err), 0)
    else:
        raise TypeError('Entry must be one or two-dimensional array.')


def is_square(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> bool:
    """True if the matrix is square."""
    return adjacency.shape[0] == adjacency.shape[1]


def check_square(adjacency: Union[sparse.csr_matrix, np.ndarray]):
    """Check whether a matrix is square and return an error otherwise."""
    if is_square(adjacency):
        return
    else:
        raise ValueError('The adjacency is expected to be square.')


def is_symmetric(adjacency: sparse.csr_matrix, tol: float = 1e-10) -> bool:
    """True if the matrix is symmetric."""
    sym_error = adjacency - adjacency.T
    return np.all(np.abs(sym_error.data) <= tol * np.abs(adjacency.data.max()))


def check_symmetry(adjacency: sparse.csr_matrix, tol: float = 1e-10):
    """Check whether a matrix is symmetric and return an error otherwise."""
    if is_symmetric(adjacency, tol):
        return
    else:
        raise ValueError('The adjacency is expected to be symmetric.')


def is_connected(adjacency: sparse.csr_matrix) -> bool:
    """Check whether a graph is weakly connected.

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph.
    """
    n_cc = sparse.csgraph.connected_components(adjacency, (not is_symmetric(adjacency)), 'weak', False)
    return n_cc == 1


def check_connected(adjacency: sparse.csr_matrix):
    """Check is a graph is connected and return an error otherwise."""
    if is_connected(adjacency):
        return
    else:
        raise ValueError('The adjacency is expected to be connected.')


def make_weights(distribution: str, adjacency: sparse.csr_matrix) -> np.ndarray:
    """Array of weights from a matrix and a desired distribution.

   Parameters
   ----------
   distribution:
       Distribution for node sampling. Only ``'degree'`` or ``'uniform'`` are accepted.
   adjacency:
       The adjacency matrix of the neighbors.

   Returns
   -------
   node_weights: np.ndarray
       Weights of nodes.
    """
    n = adjacency.shape[0]
    distribution = distribution.lower()
    if distribution == 'degree':
        node_weights_vec = adjacency.dot(np.ones(adjacency.shape[1]))
    elif distribution == 'uniform':
        node_weights_vec = np.ones(n)
    else:
        raise ValueError('Unknown distribution of node weights.')
    return node_weights_vec


def check_format(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> sparse.csr_matrix:
    """Check whether the matrix is a NumPy array or a Scipy CSR matrix and return
    the corresponding Scipy CSR matrix.
    """
    if type(adjacency) not in {sparse.csr_matrix, np.ndarray}:
        raise TypeError('Adjacency must be in Scipy CSR format or Numpy ndarray format.')
    else:
        return sparse.csr_matrix(adjacency)


def check_is_proba(entry: Union[float, int], name: str = None):
    """Check whether the number is non-negative and less than or equal to 1."""
    if name is None:
        name = 'Probabilities'
    if type(entry) not in [float, int]:
        raise TypeError('{} must be floats (or ints if 0 or 1).'.format(name))
    if entry < 0 or entry > 1:
        raise ValueError('{} must have value between 0 and 1.'.format(name))


def check_damping_factor(damping_factor: float):
    """Check if the damping factor has a valid value."""
    if damping_factor < 0 or damping_factor >= 1:
        raise ValueError('A damping factor must have a value in [0, 1[.')


def check_weights(weights: Union['str', np.ndarray], adjacency: Union[sparse.csr_matrix, sparse.csc_matrix],
                  positive_entries: bool = False) -> np.ndarray:
    """Check whether the weights are a valid distribution for the adjacency and return a probability vector.

    Parameters
    ----------
    weights:
        Probabilities for node sampling in the null model. ``'degree'``, ``'uniform'`` or custom weights.
    adjacency:
        The adjacency matrix of the graph.
    positive_entries:
        If true, the weights must all be positive, if False, the weights must be non-negative.

    Returns
    -------
    node_weights: np.ndarray
        Valid weights of nodes.
    """
    n = adjacency.shape[0]
    if type(weights) == np.ndarray:
        if len(weights) != n:
            raise ValueError('The number of node weights must match the number of nodes.')
        else:
            node_weights_vec = weights
    elif type(weights) == str:
        node_weights_vec = make_weights(weights, adjacency)
    else:
        raise TypeError(
            'Node weights must be a known distribution ("degree" or "uniform" string) or a custom NumPy array.')

    if positive_entries and not has_positive_entries(node_weights_vec):
        raise ValueError('All weights must be positive.')
    else:
        if np.any(node_weights_vec < 0) or node_weights_vec.sum() <= 0:
            raise ValueError('Node weights must be non-negative with positive sum.')

    return node_weights_vec


def check_probs(weights: Union['str', np.ndarray], adjacency: Union[sparse.csr_matrix, sparse.csc_matrix],
                positive_entries: bool = False) -> np.ndarray:
    """Check whether the weights are a valid distribution for the adjacency
    and return a normalized probability vector.
    """
    weights = check_weights(weights, adjacency, positive_entries)
    return weights / np.sum(weights)


def check_random_state(random_state: Optional[Union[np.random.RandomState, int]]):
    """Check whether the argument is a seed or a NumPy random state. If None, numpy.random is used by default."""
    if random_state is None or random_state is np.random:
        return np.random
    elif type(random_state) == int:
        return np.random.RandomState(random_state)
    elif type(random_state) == np.random.RandomState:
        return random_state
    else:
        raise TypeError('To specify a random state, pass the seed (as an int) or a NumPy random state object.')


def check_seeds(seeds: Union[np.ndarray, dict], n: int) -> np.ndarray:
    """Check the format of seeds for semi-supervised algorithms."""

    if isinstance(seeds, np.ndarray):
        if len(seeds) != n:
            raise ValueError('Dimensions mismatch between adjacency and seeds vector.')
    elif isinstance(seeds, dict):
        keys, values = np.array(list(seeds.keys())), np.array(list(seeds.values()))
        if min(values) < 0:
            warnings.warn(Warning("Negative values will not be taken into account."))
        seeds = -np.ones(n)
        seeds[keys] = values
    else:
        raise TypeError('"seeds" must be a dictionary or a one-dimensional array.')
    return seeds


def check_n_neighbors(n_neighbors: int, n_seeds: int):
    """Set the number of neighbors so that it does not exceed the number of labeled samples."""
    if n_neighbors > n_seeds:
        warnings.warn(Warning("The number of neighbors cannot exceed the number of seeds. Changed accordingly."))
        n_neighbors = n_seeds
    return n_neighbors


def check_labels(labels: np.ndarray):
    """Check labels of the seeds for semi-supervised algorithms."""

    classes: np.ndarray = np.unique(labels[labels >= 0])
    n_classes: int = len(classes)

    if n_classes < 2:
        raise ValueError('There must be at least two distinct labels.')
    else:
        return classes, n_classes


def check_n_jobs(n_jobs: Optional[int] = None):
    """Parse the ``n_jobs`` parameter for multiprocessing."""
    if n_jobs == -1:
        return None
    elif n_jobs is None:
        return 1
    else:
        return n_jobs


def check_adjacency_vector(adjacency_vectors: Union[sparse.csr_matrix, np.ndarray],
                           n: Optional[int] = None) -> sparse.csr_matrix:
    """Check format of new samples for predict methods"""
    adjacency_vectors = check_format(adjacency_vectors)

    if n is not None:
        if adjacency_vectors.shape[1] != n:
            raise ValueError('The adjacency vector must be of length equal to the number nodes in the initial graph.')

    return adjacency_vectors


def check_n_clusters(n_clusters: int, n_row: int, n_min: int = 0):
    """Check that the number of clusters"""
    if n_clusters > n_row:
        raise ValueError('The number of clusters exceeds the number of rows.')
    if n_clusters < n_min:
        raise ValueError('The number of clusters must be at least {}.'.format(n_min))
    else:
        return


def check_min_size(n_row, n_min):
    """Check that an adjacency has the required number of rows and returns an error otherwise."""
    if n_row < n_min:
        raise ValueError('The graph must contain at least {} nodes.'.format(n_min))
    else:
        return


def check_dendrogram(dendrogram):
    """Check the shape of a dendrogram."""
    if dendrogram.ndim != 2 or dendrogram.shape[1] != 4:
        raise ValueError("Dendrogram has incorrect shape.")
    else:
        return


def check_min_nnz(nnz, n_min):
    """Check that an adjacency has the required number of edges and returns an error otherwise."""
    if nnz < n_min:
        raise ValueError('The graph must contain at least {} edge(s).'.format(n_min))
    else:
        return


def check_n_components(n_components, n_min) -> int:
    """Check the number of components"""
    if n_components > n_min:
        warnings.warn(Warning("The dimension of the embedding cannot exceed {}."
                              "Changed accordingly.".format(n_min)))
        return n_min
    else:
        return n_components


def check_scaling(scaling: float, adjacency: sparse.csr_matrix, regularize: bool):
    """Check the scaling factor"""
    if scaling < 0:
        raise ValueError("The 'scaling' parameter must be non-negative.")

    if scaling and (not regularize) and not is_connected(adjacency):
        raise ValueError("The option 'equalize' is valid only if the graph is connected or with regularization."
                         "Call 'fit' either with 'equalize' = False or positive 'regularization'.")
