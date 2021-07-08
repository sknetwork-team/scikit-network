#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Sep 13 2018
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.embedding.base import BaseEmbedding
from sknetwork.linalg import LanczosEig, Laplacian, Normalizer
from sknetwork.utils.check import is_symmetric, is_connected
from sknetwork.utils.check import check_format, check_adjacency_vector, check_nonnegative, check_n_components
from sknetwork.utils.format import bipartite2undirected


class Spectral(BaseEmbedding):
    """Spectral embedding of graphs, based the spectral decomposition of the Laplacian matrix :math:`L = D - A`
    or the normalized Laplacian matrix :math:`L = I - D^{-1/2}AD^{-1/2}` (default).
    Eigenvectors are considered in increasing order of eigenvalues, skipping the first.

    Parameters
    ----------
    n_components : int (default = ``2``)
        Dimension of the embedding space.
    normalized_laplacian : bool (default = ``True``)
        If ``True`` (default), use the normalized Laplacian matrix :math:`L = I - D^{-1/2}AD^{-1/2}`.
        This is equivalent to the spectral decomposition of the transition matrix of the random walk,
        :math:`P = D^{-1}A`.
        If ``False``, use the regular Laplacian matrix :math:`L = D - A`
    regularization : float (default = ``-1``)
        Regularization factor. If negative, regularization is applied only if the graph is disconnected.

    Attributes
    ----------
    embedding_ : array, shape = (n, n_components)
        Embedding of the nodes.
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
    def __init__(self, n_components: int = 2, normalized_laplacian: bool = True, regularization: float = -1,
                 normalized_embedding: bool = False):
        super(Spectral, self).__init__()

        self.n_components = n_components
        self.normalized_laplacian = normalized_laplacian
        self.regularization = regularization
        self.bipartite = None
        self.regularized = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray]) -> 'Spectral':
        """Compute the graph embedding.

        If the input matrix :math:`B` is not square (e.g., biadjacency matrix of a bipartite graph) or not symmetric
        (e.g., adjacency matrix of a directed graph), use the adjacency matrix
         :math:`A  = \\begin{bmatrix} 0 & B \\\\ B^T & 0 \\end{bmatrix}` and return the embedding for both rows and
         columns of the input matrix :math:`B`.

        Parameters
        ----------
        input_matrix :
              Adjacency matrix or biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`Spectral`
        """
        # input
        adjacency = check_format(input_matrix)
        n_row, n_col = adjacency.shape
        self.bipartite = False
        if n_row != n_col or not is_symmetric(adjacency):
            self.bipartite = True
            adjacency = bipartite2undirected(adjacency)
        n = adjacency.shape[0]

        # regularization
        if self.regularization < 0:
            if is_connected(adjacency):
                regularization = 0
            else:
                regularization = np.abs(self.regularization)
        else:
            regularization = self.regularization
        self.regularized = regularization > 0

        # laplacian
        laplacian = Laplacian(adjacency, regularization, self.normalized_laplacian)

        # spectral decomposition
        n_components = check_n_components(self.n_components, n - 2) + 1
        solver = LanczosEig(which='SM')
        solver.fit(matrix=laplacian, n_components=n_components)
        index = np.argsort(solver.eigenvalues_)[1:]  # increasing order, skip first

        eigenvalues = solver.eigenvalues_[index]
        eigenvectors = solver.eigenvectors_[:, index]

        # embedding
        if self.normalized_laplacian:
            embedding = laplacian.norm_diag.dot(eigenvectors)
        else:
            embedding = eigenvectors.copy()

        # output
        self.embedding_ = embedding
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        if self.bipartite:
            self._split_vars(n_row)

        return self

    def predict(self, adjacency_vectors: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Predict the embedding of new nodes, when possible (otherwise return 0).

        Each new node is defined by its adjacency vector (i.e., neighbors in the graph).

        Parameters
        ----------
        adjacency_vectors :
            Adjacency vectors of nodes.
            Array of shape (n_col,) (single vector) or (n_vectors, n_col)

        Returns
        -------
        embedding_vectors : np.ndarray
            Embedding of the nodes.

        Example
        -------
        >>> from sknetwork.embedding import Spectral
        >>> from sknetwork.data import karate_club
        >>> spectral = Spectral(n_components=3)
        >>> adjacency = karate_club()
        >>> adjacency_vector = np.arange(34) < 5
        >>> _ = spectral.fit(adjacency)
        >>> len(spectral.predict(adjacency_vector))
        3
        """
        self._check_fitted()

        # input
        if self.bipartite:
            n = len(self.embedding_col_)
        else:
            n = len(self.embedding_)
        adjacency_vectors = check_adjacency_vector(adjacency_vectors, n)
        check_nonnegative(adjacency_vectors)

        if self.bipartite:
            shape = (adjacency_vectors.shape[0], self.embedding_row_.shape[0])
            adjacency_vectors = sparse.csr_matrix(adjacency_vectors)
            adjacency_vectors = sparse.hstack([sparse.csr_matrix(shape), adjacency_vectors], format='csr')
            embedding = np.vstack([self.embedding_row_, self.embedding_col_])
        else:
            embedding = self.embedding_
        eigenvalues = self.eigenvalues_

        # regularization
        if self.regularized:
            regularization = np.abs(self.regularization)
        else:
            regularization = 0
        normalizer = Normalizer(adjacency_vectors, regularization)

        # prediction
        embedding_vectors = normalizer.dot(embedding)
        if self.normalized_laplacian:
            norm_vec = 1 - eigenvalues
            norm_vec[norm_vec > 0] = 1 / norm_vec[norm_vec > 0]
            embedding_vectors *= norm_vec
        else:
            norm_matrix = sparse.csr_matrix(1 - np.outer(normalizer.norm_diag.data, eigenvalues))
            norm_matrix.data = 1 / norm_matrix.data
            embedding_vectors *= norm_matrix.toarray()

        # shape
        if len(embedding_vectors) == 1:
            embedding_vectors = embedding_vectors.ravel()

        return embedding_vectors
