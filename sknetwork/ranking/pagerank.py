#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 31 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from sknetwork.utils.checks import check_format, is_proba_array, is_square
from sknetwork.utils.randomized_matrix_factorization import SparseLR, randomized_eig
from scipy import sparse
from typing import Union


def pagerank(adjacency: Union[sparse.csr_matrix, np.ndarray], damping_factor: float = 0.85,
             personalization: Union[None, np.ndarray] = None) -> np.ndarray:
    """Standard pagerank via matrix factorization.

    Parameters
    ----------
    adjacency
        Adjacency matrix of the graph.
    damping_factor
        Probability to jump according to the graph transition matrix in the random walk.
    personalization
        If ``None``, the uniform probability distribution over the nodes is used.
        Otherwise, the user must provide a valid probability vector.

    Returns
    -------
    pagerank: np.ndarray
        The ranking score of each node.
    """
    adjacency = check_format(adjacency)
    if not is_square(adjacency):
        raise ValueError('Adjacency must be square.')
    else:
        n_nodes: int = adjacency.shape[0]
    if damping_factor < 0. or damping_factor > 1.:
        raise ValueError('Damping factor must be between 0 and 1.')

    # pseudo inverse square-root out-degree matrix
    diag_out: sparse.csr_matrix = sparse.diags(adjacency.dot(np.ones(n_nodes)), shape=(n_nodes, n_nodes), format='csr')
    diag_out.data = 1 / diag_out.data

    transition_matrix = diag_out.dot(adjacency)

    if personalization is None:
        root: np.ndarray = np.ones(n_nodes) / n_nodes
    else:
        if is_proba_array(personalization) and len(personalization) == n_nodes:
            root = personalization
        else:
            raise ValueError('Personalization must be None or a valid probability array.')

    weight_matrix = sparse.eye(n_nodes, format='csr') - damping_factor * diag_out.astype(bool)
    root = weight_matrix.dot(root)

    pagerank_matrix = SparseLR(damping_factor * transition_matrix, [(root, np.ones(n_nodes))])
    _, eigenvector = randomized_eig(pagerank_matrix, 1)

    return abs(eigenvector.real) / abs(eigenvector.real).sum()
