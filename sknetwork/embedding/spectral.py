#!/usr/bin/env python3
# coding: utf-8
"""
Created on September 2018
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.embedding.base import BaseEmbedding
from sknetwork.linalg import LanczosEig, Laplacian, Normalizer, normalize
from sknetwork.utils.format import get_adjacency
from sknetwork.utils.check import check_format, check_adjacency_vector, check_nonnegative, check_n_components


class Spectral(BaseEmbedding):
    """Spectral embedding of graphs, based the spectral decomposition of the Laplacian matrix :math:`L = D - A`
    or the transition matrix of the random walk :math:`P = D^{-1}A` (default), where :math:`D` is the
    diagonal matrix of degrees.

    Eigenvectors are considered in increasing order (for the Laplacian matrix :math:`L`) or decreasing order
    (for the transition matrix of the random walk :math:`P`) of eigenvalues, skipping the first.

    Parameters
    ----------
    n_components : int (default = ``2``)
        Dimension of the embedding space.
    decomposition : str (``laplacian`` or ``rw``, default = ``rw``)
        Matrix used for the spectral decomposition.
    regularization : float (default = ``-1``)
        Regularization factor :math:`\\alpha` so that the adjacency matrix is :math:`A + \\alpha \\frac{11^T}{n}`.
        If negative, regularization is applied only if the graph is disconnected; the regularization factor
        :math:`\\alpha` is then set to the absolute value of the parameter.
    normalized : bool (default = ``True``)
        If ``True``, normalized the embedding so that each vector has norm 1 in the embedding space, i.e.,
        each vector lies on the unit sphere.
    Attributes
    ----------
    embedding_ : array, shape = (n, n_components)
        Embedding of the nodes.
    embedding_row_ : array, shape = (n_row, n_components)
        Embedding of the rows, for bipartite graphs.
    embedding_col_ : array, shape = (n_col, n_components)
        Embedding of the columns, for bipartite graphs.
    eigenvalues_ : array, shape = (n_components)
        Eigenvalues.
    eigenvectors_ : array, shape = (n, n_components)
        Eigenvectors.

    Example
    -------
    >>> from sknetwork.embedding import Spectral
    >>> from sknetwork.data import karate_club
    >>> spectral = Spectral(n_components=3)
    >>> adjacency = karate_club()
    >>> embedding = spectral.fit_transform(adjacency)
    >>> embedding.shape
    (34, 3)

    References
    ----------
    Belkin, M. & Niyogi, P. (2003). Laplacian Eigenmaps for Dimensionality Reduction and Data Representation,
    Neural computation.
    """
    def __init__(self, n_components: int = 2, decomposition: str = 'rw', regularization: float = -1,
                 normalized: bool = True):
        super(Spectral, self).__init__()

        self.embedding_ = None
        self.n_components = n_components
        self.decomposition = decomposition
        self.regularization = regularization
        self.normalized = normalized
        self.bipartite = None
        self.regularized = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], force_bipartite: bool = False) -> 'Spectral':
        """Compute the graph embedding.

        If the input matrix :math:`B` is not square (e.g., biadjacency matrix of a bipartite graph) or not symmetric
        (e.g., adjacency matrix of a directed graph), use the adjacency matrix

        :math:`A  = \\begin{bmatrix} 0 & B \\\\ B^T & 0 \\end{bmatrix}`

        and return the embedding for both rows and columns of the input matrix :math:`B`.

        Parameters
        ----------
        input_matrix :
              Adjacency matrix or biadjacency matrix of the graph.
        force_bipartite : bool (default = ``False``)
            If ``True``, force the input matrix to be considered as a biadjacency matrix.

        Returns
        -------
        self: :class:`Spectral`
        """
        # input
        input_matrix = check_format(input_matrix)
        adjacency, self.bipartite = get_adjacency(input_matrix, allow_directed=False, force_bipartite=force_bipartite)
        n = adjacency.shape[0]

        # regularization
        regularization = self._get_regularization(self.regularization, adjacency)
        self.regularized = regularization > 0

        # laplacian
        normalized_laplacian = self.decomposition == 'rw'
        laplacian = Laplacian(adjacency, regularization, normalized_laplacian)

        # spectral decomposition
        n_components = check_n_components(self.n_components, n - 2) + 1
        solver = LanczosEig(which='SM')
        solver.fit(matrix=laplacian, n_components=n_components)
        index = np.argsort(solver.eigenvalues_)[1:]  # increasing order, skip first

        eigenvalues = solver.eigenvalues_[index]
        eigenvectors = solver.eigenvectors_[:, index]

        if normalized_laplacian:
            eigenvectors = laplacian.norm_diag.dot(eigenvectors)
            eigenvalues = 1 - eigenvalues

        # embedding
        embedding = eigenvectors.copy()
        if self.normalized:
            embedding = normalize(embedding, p=2)

        # output
        self.embedding_ = embedding
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        if self.bipartite:
            self._split_vars(input_matrix.shape)

        return self
