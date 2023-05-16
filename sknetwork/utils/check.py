#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in April 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
import warnings
from typing import Union, Optional

import numpy as np
from scipy import sparse


def has_nonnegative_entries(input_matrix: Union[sparse.csr_matrix, np.ndarray]) -> bool:
    """True if the array has non-negative entries."""
    if type(input_matrix) == sparse.csr_matrix:
        return np.all(input_matrix.data >= 0)
    else:
        return np.all(input_matrix >= 0)


def is_weakly_connected(adjacency: sparse.csr_matrix) -> bool:
    """Check whether a graph is weakly connected.
    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph.
    """
    n_cc = sparse.csgraph.connected_components(adjacency, (not is_symmetric(adjacency)), 'weak', False)
    return n_cc == 1


def check_connected(adjacency: sparse.csr_matrix):
    """Check is a graph is weakly connected and return an error otherwise."""
    if is_weakly_connected(adjacency):
        return
    else:
        raise ValueError('The graph is expected to be connected.')


def check_nonnegative(input_matrix: Union[sparse.csr_matrix, np.ndarray]):
    """Check whether the array has non-negative entries."""
    if not has_nonnegative_entries(input_matrix):
        raise ValueError('Only nonnegative values are expected.')


def has_positive_entries(input_matrix: np.ndarray) -> bool:
    """True if the array has positive entries."""
    if type(input_matrix) != np.ndarray:
        raise TypeError('Entry must be a dense NumPy array.')
    else:
        return np.all(input_matrix > 0)


def check_positive(input_matrix: Union[sparse.csr_matrix, np.ndarray]):
    """Check whether the array has positive entries."""
    if not has_positive_entries(input_matrix):
        raise ValueError('Only positive values are expected.')


def is_proba_array(input_matrix: np.ndarray) -> bool:
    """True if each line of the array has non-negative entries which sum to 1."""
    if len(input_matrix.shape) == 1:
        return has_nonnegative_entries(input_matrix) and np.isclose(input_matrix.sum(), 1)
    elif len(input_matrix.shape) == 2:
        n_row, n_col = input_matrix.shape
        err = input_matrix.dot(np.ones(n_col)) - np.ones(n_row)
        return has_nonnegative_entries(input_matrix) and np.isclose(np.linalg.norm(err), 0)
    else:
        raise TypeError('Entry must be one or two-dimensional array.')


def is_square(input_matrix: Union[sparse.csr_matrix, np.ndarray]) -> bool:
    """True if the matrix is square."""
    return input_matrix.shape[0] == input_matrix.shape[1]


def check_square(input_matrix: Union[sparse.csr_matrix, np.ndarray]):
    """Check whether a matrix is square and return an error otherwise."""
    if is_square(input_matrix):
        return
    else:
        raise ValueError('The adjacency matrix is expected to be square.')


def is_symmetric(input_matrix: sparse.csr_matrix) -> bool:
    """True if the matrix is symmetric."""
    return sparse.csr_matrix(input_matrix - input_matrix.T).nnz == 0


def check_symmetry(input_matrix: sparse.csr_matrix):
    """Check whether a matrix is symmetric and return an error otherwise."""
    if not is_symmetric(input_matrix):
        raise ValueError('The input matrix is expected to be symmetric.')


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


def check_format(input_matrix: Union[sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix, sparse.lil_matrix,
                                     np.ndarray], allow_empty: bool = False) -> sparse.csr_matrix:
    """Check whether the matrix is a NumPy array or a Scipy sparse matrix and return
    the corresponding Scipy CSR matrix.
    """
    formats = {sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix, sparse.lil_matrix, np.ndarray}
    if type(input_matrix) not in formats:
        raise TypeError('The input matrix must be in Scipy sparse format or Numpy ndarray format.')
    input_matrix = sparse.csr_matrix(input_matrix)
    if not allow_empty and input_matrix.nnz == 0:
        raise ValueError('The input matrix is empty.')
    return input_matrix


def check_is_proba(entry: Union[float, int], name: str = None):
    """Check whether the number is non-negative and less than or equal to 1."""
    if name is None:
        name = 'Probabilities'
    if type(entry) not in [float, int]:
        raise TypeError('{} must be floats (or ints if 0 or 1).'.format(name))
    if entry < 0 or entry > 1:
        raise ValueError('{} must be between 0 and 1.'.format(name))


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


def get_probs(weights: Union['str', np.ndarray], adjacency: Union[sparse.csr_matrix, sparse.csc_matrix],
              positive_entries: bool = False) -> np.ndarray:
    """Check whether the weights are a valid distribution for the adjacency
    and return a normalized probability vector.
    """
    weights = check_weights(weights, adjacency, positive_entries)
    return weights / np.sum(weights)


def check_random_state(random_state: Optional[Union[np.random.RandomState, int]]):
    """Check whether the argument is a seed or a NumPy random state. If None, 'numpy.random' is used by default."""
    if random_state is None:
        return np.random.RandomState()
    elif type(random_state) == int:
        return np.random.RandomState(random_state)
    elif type(random_state) == np.random.RandomState:
        return random_state
    else:
        raise TypeError('To specify a random state, pass the seed (as an int) or a NumPy random state object.')


def check_n_neighbors(n_neighbors: int, n_seeds: int):
    """Set the number of neighbors so that it is less than the number of labeled samples."""
    if n_neighbors >= n_seeds:
        warnings.warn(Warning("The number of neighbors must be lower than the number of nodes with known labels. "
                              "Changed accordingly."))
        n_neighbors = n_seeds - 1
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
    if n is not None and adjacency_vectors.shape[1] != n:
        raise ValueError('The adjacency vector must be of length equal to the number nodes in the graph.')
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
        warnings.warn(Warning("The dimension of the embedding cannot exceed {}. Changed accordingly.".format(n_min)))
        return n_min
    else:
        return n_components


def check_scaling(scaling: float, adjacency: sparse.csr_matrix, regularize: bool):
    """Check the scaling factor"""
    if scaling < 0:
        raise ValueError("The 'scaling' parameter must be non-negative.")

    if scaling and (not regularize) and not is_weakly_connected(adjacency):
        raise ValueError("Positive 'scaling' is valid only if the graph is connected or with regularization."
                         "Call 'fit' either with 'scaling' = 0 or positive 'regularization'.")


def has_boolean_entries(input_matrix: np.ndarray) -> bool:
    """True if the array has boolean entries."""
    if type(input_matrix) != np.ndarray:
        raise TypeError('Entry must be a dense NumPy array.')
    else:
        return input_matrix.dtype == 'bool'


def check_boolean(input_matrix: np.ndarray):
    """Check whether the array has positive entries."""
    if not has_boolean_entries(input_matrix):
        raise ValueError('Only boolean values are expected.')


def check_vector_format(vector_1: np.ndarray, vector_2: np.ndarray):
    """Check whether the inputs are vectors of same length."""
    if len(vector_1.shape) > 1 or len(vector_2.shape) > 1:
        raise ValueError('The arrays must be 1-dimensional.')
    if vector_1.shape[0] != vector_2.shape[0]:
        raise ValueError('The arrays do not have the same length.')


def has_self_loops(input_matrix: sparse.csr_matrix) -> bool:
    """True if each node has a self loop."""
    return all(input_matrix.diagonal().astype(bool))


def add_self_loops(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """Add self loops to adjacency matrix.

    Parameters
    ----------
    adjacency : sparse.csr_matrix
        Adjacency matrix of the graph.

    Returns
    -------
    sparse.csr_matrix
        Adjacency matrix of the graph with self loops.
    """
    n_row, n_col = adjacency.shape

    if is_square(adjacency):
        adjacency = sparse.diags(np.ones(n_col), format='csr') + adjacency
    else:
        tmp = sparse.eye(n_row)
        tmp.resize(n_row, n_col)
        adjacency += tmp

    return adjacency
