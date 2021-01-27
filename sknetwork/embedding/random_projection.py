#!/usr/bin/env python3
# coding: utf-8
"""
Created on January, 15 2021
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.embedding.base import BaseEmbedding, BaseBiEmbedding
from sknetwork.linalg import normalize
from sknetwork.utils.check import check_random_state, check_format, check_square
from sknetwork.utils.format import bipartite2undirected


class RandomProjection(BaseEmbedding):
    """Embedding of graphs based the random projection of the adjacency matrix:

    :math:`(I + \\alpha A +... + (\\alpha A)^K)G`

    where :math:`A` is the adjacency matrix, :math:`G` is a random Gaussian matrix,
    :math:`\\alpha` is some smoothing factor and :math:`K` non-negative integer.

    * Graphs
    * Digraphs

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
    normalized : bool (default = ``True``)
        If ``True``, normalize the embedding so that each vector has norm 1 in the embedding space, i.e.,
        each vector lies on the unit sphere.
    random_state : int, optional
        Seed used by the random number generator.

    Attributes
    ----------
    embedding_ : array, shape = (n, n_components)
        Embedding of the nodes.

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
    def __init__(self, n_components: int = 2, alpha: float = 0.5, n_iter: int = 3, random_walk: bool = False ,
                 normalized: bool = True, random_state: int = None):
        super(RandomProjection, self).__init__()

        self.n_components = n_components
        self.alpha = alpha
        self.n_iter = n_iter
        self.random_walk = random_walk
        self.normalized = normalized
        self.random_state = random_state

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'RandomProjection':
        """Compute the graph embedding.

        Parameters
        ----------
        adjacency :
              Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`RandomProjection`
        """
        adjacency = check_format(adjacency).asfptype()
        check_square(adjacency)
        n = adjacency.shape[0]

        random_generator = check_random_state(self.random_state)
        random_matrix = random_generator.normal(size=(n, self.n_components))

        # make the matrix orthogonal
        random_matrix, _ = np.linalg.qr(random_matrix)

        factor = random_matrix
        embedding = factor.copy()

        if self.random_walk:
            transition = normalize(adjacency)
        else:
            transition = adjacency

        for t in range(self.n_iter):
            factor = self.alpha * transition.dot(factor)
            embedding += factor

        if self.normalized:
            embedding = normalize(embedding, p=2)

        self.embedding_ = embedding

        return self


class BiRandomProjection(RandomProjection, BaseBiEmbedding):
    """Embedding of bipartite graphs, based the random projection of the corresponding adjacency matrix:

        :math:`A  = \\begin{bmatrix} 0 & B \\\\ B^T & 0 \\end{bmatrix}`

    where :math:`B` is the biadjacency matrix of the bipartite graph.

    * Bigraphs

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
    normalized : bool (default = ``True``)
        If ``True``, normalized the embedding so that each vector has norm 1 in the embedding space, i.e.,
        each vector lies on the unit sphere.

    Attributes
    ----------
    embedding_ : array, shape = (n_row, n_components)
        Embedding of the rows.
    embedding_row_ : array, shape = (n_row, n_components)
        Embedding of the rows (copy of **embedding_**).
    embedding_col_ : array, shape = (n_col, n_components)
        Embedding of the columns.

    Example
    -------
    >>> from sknetwork.embedding import BiRandomProjection
    >>> from sknetwork.data import movie_actor
    >>> biprojection = BiRandomProjection()
    >>> biadjacency = movie_actor()
    >>> embedding = biprojection.fit_transform(biadjacency)
    >>> embedding.shape
    (15, 2)

    References
    ----------
    Zhang, Z., Cui, P., Li, H., Wang, X., & Zhu, W. (2018).
    Billion-scale network embedding with iterative random projection, ICDM.
    """
    def __init__(self, n_components: int = 2, alpha: float = 0.5, n_iter: int = 3, random_walk: bool = False,
                 normalized: bool = True):
        super(BiRandomProjection, self).__init__(n_components, alpha, n_iter, random_walk, normalized)

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'BiRandomProjection':
        """Compute the embedding.

        Parameters
        ----------
        biadjacency:
            Biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`BiRandomProjection`
        """
        biadjacency = check_format(biadjacency)
        n_row, _ = biadjacency.shape
        RandomProjection.fit(self, bipartite2undirected(biadjacency))
        self._split_vars(n_row)

        return self
