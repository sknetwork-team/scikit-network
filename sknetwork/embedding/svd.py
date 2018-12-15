#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:16:22 2018

@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np

from sknetwork.embedding.randomized_matrix_factorization import randomized_svd
from scipy import sparse, errstate, sqrt, isinf, linalg


def normalized_adjacency(adjacency_matrix):
    """Normalized adjacency matrix :math:`(D_{in}^{+})^{1/2} A (D_{out}^{+})^{1/2}`.
        Parameters
        ----------
        adjacency_matrix: sparse.csr_matrix or np.ndarray

        Returns
        -------
        The normalized adjacency matrix.
    """

    if type(adjacency_matrix) == sparse.csr_matrix:
        adj_matrix = adjacency_matrix
    elif type(adjacency_matrix) == np.ndarray:
        adj_matrix = sparse.csr_matrix(adjacency_matrix)
    else:
        raise TypeError(
            "The argument must be a NumPy array or a SciPy Compressed Sparse Row matrix.")
    n_nodes, m_nodes = adj_matrix.shape

    # out-degree vector
    dou = adj_matrix.dot(np.ones(n_nodes))
    # in-degree vector
    din = adj_matrix.T.dot(np.ones(m_nodes))

    with errstate(divide='ignore'):
        dou_sqrt = 1.0 / sqrt(dou)
        din_sqrt = 1.0 / sqrt(din)
    dou_sqrt[isinf(dou_sqrt)] = 0
    din_sqrt[isinf(din_sqrt)] = 0
    # pseudo inverse square-root out-degree matrix
    dhou = sparse.spdiags(dou_sqrt, [0], n_nodes, n_nodes, format='csr')
    # pseudo inverse square-root in-degree matrix
    dhin = sparse.spdiags(din_sqrt, [0], m_nodes, m_nodes, format='csr')

    return dhou.dot(adj_matrix.dot(dhin))


class ForwardBackwardEmbedding:
    """Forward and Backward embeddings for non-linear dimensionality reduction.

    Parameters
    -----------
    embedding_dimension: int, optional
        The dimension of the projected subspace (default=2).

    Attributes
    ----------
    embedding_ : array, shape = (n_samples, embedding_dimension)
        Forward embedding of the training matrix.
    backward_embedding_ : array, shape = (n_samples, embedding_dimension)
        Backward embedding of the training matrix.
    singular_values_ : array, shape = (embedding_dimension)
        Singular values of the training matrix

    References
    ----------
    - Bonald, De Lara. "The Forward-Backward Embedding of Directed Graphs."
    """

    def __init__(self, embedding_dimension=2):
        self.embedding_dimension = embedding_dimension
        self.embedding_ = None
        self.backward_embedding_ = None
        self.singular_values_ = None

    def fit(self, adjacency_matrix, randomized_decomposition: bool = True, tol=1e-6, n_iter='auto',
            power_iteration_normalizer='auto', random_state=None):
        """Fits the model from data in adjacency_matrix.

        Parameters
        ----------
        adjacency_matrix: array-like, shape = (n, m)
            Adjacency matrix, where n = m = |V| for a standard graph,
            n = |V1|, m = |V2| for a bipartite graph.
        randomized_decomposition: whether to use a randomized (and faster) svd method or
            the standard scipy one.
        tol: float, optional
            Tolerance for pseudo-inverse of singular values (default=1e-6).
        n_iter: int or 'auto' (default is 'auto')
            Number of power iterations. It can be used to deal with very noisy
            problems. When 'auto', it is set to 4, unless `embedding_dimension` is small
            (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
            This improves precision with few components.
        power_iteration_normalizer: 'auto' (default), 'QR', 'LU', 'none'
            Whether the power iterations are normalized with step-by-step
            QR factorization (the slowest but most accurate), 'none'
            (the fastest but numerically unstable when `n_iter` is large, e.g.
            typically 5 or larger), or 'LU' factorization (numerically stable
            but can lose slightly in accuracy). The 'auto' mode applies no
            normalization if `n_iter`<=2 and switches to LU otherwise.
        random_state: int, RandomState instance or None, optional (default=None)
            The seed of the pseudo random number generator to use when shuffling
            the data.  If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random number
            generator; If None, the random number generator is the RandomState
            instance used by `np.random`.

        Returns
        -------
        self

        """
        if type(adjacency_matrix) == sparse.csr_matrix:
            adj_matrix = adjacency_matrix
        elif type(adjacency_matrix) == np.ndarray:
            adj_matrix = sparse.csr_matrix(adjacency_matrix)
        else:
            raise TypeError(
                "The argument must be a NumPy array or a SciPy Compressed Sparse Row matrix.")
        n_nodes, m_nodes = adj_matrix.shape

        # out-degree vector
        dou = adj_matrix.dot(np.ones(n_nodes))
        # in-degree vector
        din = adj_matrix.T.dot(np.ones(m_nodes))

        with errstate(divide='ignore'):
            dou_sqrt = 1.0 / sqrt(dou)
            din_sqrt = 1.0 / sqrt(din)
        dou_sqrt[isinf(dou_sqrt)] = 0
        din_sqrt[isinf(din_sqrt)] = 0
        # pseudo inverse square-root out-degree matrix
        dhou = sparse.spdiags(dou_sqrt, [0], n_nodes, n_nodes, format='csr')
        # pseudo inverse square-root in-degree matrix
        dhin = sparse.spdiags(din_sqrt, [0], m_nodes, m_nodes, format='csr')

        laplacian = dhou.dot(adj_matrix.dot(dhin))

        if randomized_decomposition:
            u, sigma, vt = randomized_svd(laplacian, self.embedding_dimension,
                                          n_iter=n_iter,
                                          power_iteration_normalizer=power_iteration_normalizer,
                                          random_state=random_state)
        else:
            u, sigma, vt = linalg.svds(laplacian, self.embedding_dimension)

        self.singular_values_ = sigma

        gamma = 1 - sigma ** 2
        gamma_sqrt = np.diag(np.piecewise(gamma, [gamma > tol, gamma <= tol], [lambda x: 1 / np.sqrt(x), 0]))
        self.embedding_ = dhou.dot(u).dot(gamma_sqrt)
        self.backward_embedding_ = dhin.dot(vt.T).dot(gamma_sqrt)

        return self
