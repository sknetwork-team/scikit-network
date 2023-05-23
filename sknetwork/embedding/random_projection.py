#!/usr/bin/env python3
# coding: utf-8
"""
Created in January 2021
@author: Thomas Bonald <bonald@enst.fr>
"""
from abc import ABC
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.embedding.base import BaseEmbedding
from sknetwork.linalg import Regularizer, Normalizer, normalize
from sknetwork.utils.check import check_format, check_random_state
from sknetwork.utils.format import get_adjacency


class RandomProjection(BaseEmbedding, ABC):
    """Embedding of graphs based the random projection of the adjacency matrix:

    :math:`(I + \\alpha A +... + (\\alpha A)^K)G`

    where :math:`A` is the adjacency matrix, :math:`G` is a random Gaussian matrix,
    :math:`\\alpha` is some smoothing factor and :math:`K` some non-negative integer.

    Parameters
    ----------
    n_components : int (default = 2)
        Dimension of the embedding space.
    alpha : float (default = 0.5)
        Smoothing parameter.
    n_iter : int (default = 3)
        Number of power iterations of the adjacency matrix.
    random_walk : bool (default = ``False``)
        If ``True``, use the transition matrix of the random walk, :math:`P = D^{-1}A`, instead of the adjacency matrix.
    regularization : float (default = ``-1``)
        Regularization factor :math:`\\alpha` so that the matrix is :math:`A + \\alpha \\frac{11^T}{n}`.
        If negative, regularization is applied only if the graph is disconnected (and then equal to the absolute value
        of the parameter).
    normalized : bool (default = ``True``)
        If ``True``, normalize the embedding so that each vector has norm 1 in the embedding space, i.e.,
        each vector lies on the unit sphere.
    random_state : int, optional
        Seed used by the random number generator.

    Attributes
    ----------
    embedding_ : array, shape = (n, n_components)
        Embedding of the nodes.
    embedding_row_ : array, shape = (n_row, n_components)
        Embedding of the rows, for bipartite graphs.
    embedding_col_ : array, shape = (n_col, n_components)
        Embedding of the columns, for bipartite graphs.

    Example
    -------
    >>> from sknetwork.embedding import RandomProjection
    >>> from sknetwork.data import karate_club
    >>> projection = RandomProjection()
    >>> adjacency = karate_club()
    >>> embedding = projection.fit_transform(adjacency)
    >>> embedding.shape
    (34, 2)

    References
    ----------
    Zhang, Z., Cui, P., Li, H., Wang, X., & Zhu, W. (2018).
    Billion-scale network embedding with iterative random projection, ICDM.
    """
    def __init__(self, n_components: int = 2, alpha: float = 0.5, n_iter: int = 3, random_walk: bool = False,
                 regularization: float = -1, normalized: bool = True, random_state: int = None):
        super(RandomProjection, self).__init__()

        self.embedding_ = None
        self.n_components = n_components
        self.alpha = alpha
        self.n_iter = n_iter
        self.random_walk = random_walk
        self.regularization = regularization
        self.normalized = normalized
        self.random_state = random_state
        self.bipartite = None
        self.regularized = None

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], force_bipartite: bool = False) \
            -> 'RandomProjection':
        """Compute the graph embedding.

        Parameters
        ----------
        input_matrix : sparse.csr_matrix, np.ndarray
              Adjacency matrix or biadjacency matrix of the graph.
        force_bipartite : bool (default = ``False``)
            If ``True``, force the input matrix to be considered as a biadjacency matrix.
        Returns
        -------
        self: :class:`RandomProjection`
        """
        # input
        input_matrix = check_format(input_matrix)
        adjacency, self.bipartite = get_adjacency(input_matrix, force_bipartite=force_bipartite)
        n = adjacency.shape[0]

        # regularization
        regularization = self._get_regularization(self.regularization, adjacency)
        self.regularized = regularization > 0

        # multiplier
        if self.random_walk:
            multiplier = Normalizer(adjacency, regularization)
        else:
            multiplier = Regularizer(adjacency, regularization)

        # random matrix
        random_generator = check_random_state(self.random_state)
        random_matrix = random_generator.normal(size=(n, self.n_components))
        random_matrix, _ = np.linalg.qr(random_matrix)

        # random projection
        factor = random_matrix
        embedding = factor.copy()
        for t in range(self.n_iter):
            factor = self.alpha * multiplier.dot(factor)
            embedding += factor

        # normalization
        if self.normalized:
            embedding = normalize(embedding, p=2)

        # output
        self.embedding_ = embedding
        if self.bipartite:
            self._split_vars(input_matrix.shape)
        return self
