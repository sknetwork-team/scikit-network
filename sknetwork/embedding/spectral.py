#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Sep 13 2018
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

import warnings
from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from sknetwork.basics.structure import is_connected
from sknetwork.embedding.base import BaseEmbedding
from sknetwork.linalg import EigSolver, HalkoEig, LanczosEig, auto_solver, diag_pinv
from sknetwork.utils.checks import check_format, is_square, is_symmetric


class LaplacianOperator(LinearOperator):
    """
    Regularized Laplacian matrix as a scipy LinearOperator.
    """

    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], regularization: float = 0.):
        LinearOperator.__init__(self, dtype=float, shape=adjacency.shape)
        self.regularization = regularization
        self.weights = adjacency.dot(np.ones(adjacency.shape[1]))
        self.laplacian = sparse.diags(self.weights, format='csr') - adjacency

    def _matvec(self, matrix: np.ndarray):
        prod = self.laplacian.dot(matrix)
        prod += self.shape[0] * self.regularization * matrix
        if len(matrix.shape) == 2:
            prod -= self.regularization * np.tile(matrix.sum(axis=0), (self.shape[0], 1))
        else:
            prod -= self.regularization * matrix.sum()

        return prod

    def astype(self, dtype: Union[str, np.dtype]):
        """Change dtype of the object.

        Parameters
        ----------
        dtype

        Returns
        -------
        self

        """
        self.dtype = np.dtype(dtype)
        self.laplacian = self.laplacian.astype(self.dtype)
        self.weights = self.weights.astype(self.dtype)

        return self


class NormalizedAdjacencyOperator(LinearOperator):
    """
    Regularized normalized adjacency matrix as a scipy LinearOperator.
    """

    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], regularization: float = 0.):
        LinearOperator.__init__(self, dtype=float, shape=adjacency.shape)
        self.adjacency = adjacency
        self.regularization = regularization

        n = self.adjacency.shape[0]
        self.weights_sqrt = np.sqrt(self.adjacency.dot(np.ones(n)) + self.regularization * n)

    def _matvec(self, matrix: np.ndarray):
        matrix = (matrix.T / self.weights_sqrt).T
        prod = self.adjacency.dot(matrix)
        if len(matrix.shape) == 2:
            prod += self.regularization * np.tile(matrix.sum(axis=0), (self.shape[0], 1))
        else:
            prod += self.regularization * matrix.sum()
        return (prod.T / self.weights_sqrt).T

    def _transpose(self):
        return self

    def astype(self, dtype: Union[str, np.dtype]):
        """Change dtype of the object.

        Parameters
        ----------
        dtype

        Returns
        -------
        self

        """
        self.dtype = np.dtype(dtype)
        self.adjacency = self.adjacency.astype(self.dtype)
        self.weights_sqrt = self.weights_sqrt.astype(self.dtype)

        return self


class Spectral(BaseEmbedding):
    """
    Spectral embedding of the graph, using the spectral decomposition of the Laplacian matrix :math:`L = D - A`.
    Eigenvectors are considered in increasing order of eigenvalues, skipping the first eigenvector.
    The graph must be undirected (see BiSpectral for directed graphs and bipartite graphs).

    Parameters
    ----------
    n_components : int (default = 2)
        Dimension of the embedding space.
    normalized_laplacian : bool (default = ``True``)

        * If ``True``, solves the eigenvalue problem :math:`LU = DU \\Lambda`.
        * If ``False``, solves the eigenvalue problem :math:`LU = U \\Lambda`.

    regularization : ``None`` or float (default = ``0.01``)
        Adds edges of given weight between all pairs of nodes.
    relative_regularization : bool (default = ``True``)
        If ``True``, considers the regularization as relative to the total weight of the graph.
    equalize : bool (default = ``False``)
        If ``True``, equalizes the energy levels of the corresponding physical system, i.e., uses
        :math:`U \\Lambda^{- \\frac 1 2}`. Requires regularization if the graph is not connected.
    barycenter : bool (default = ``True``)
        If ``True``, uses the barycenter of neighboring nodes for the embedding, i.e., :math:`PU`
        with :math:`P = D^{-1}A`.
    normalize : bool (default = ``True``)
        If ``True``, normalizes the embedding so that each vector has norm 1 in the embedding space, i.e.,
        each vector lies on the unit sphere.
    solver : ``'auto'``, ``'halko'``, ``'lanczos'`` or :class:`EigSolver` (default = ``'auto'``)
        Which eigenvalue solver to use.

        * ``'auto'`` call the auto_solver function.
        * ``'halko'``: randomized method, fast but less accurate than ``'lanczos'`` for ill-conditioned matrices.
        * ``'lanczos'``: power-iteration based method.
        * :class:`EigSolver`: custom solver.

    Attributes
    ----------
    embedding_ : array, shape = (n, n_components)
        Embedding of the nodes.
    eigenvalues_ : array, shape = (n_components)
        Eigenvalues in increasing order (first eigenvalue ignored).
    eigenvectors_ : array, shape = (n, n_components)
        Corresponding eigenvectors.
    regularization_ : ``None`` or float
        Regularization factor added to all pairs of nodes.

    Example
    -------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> spectral = Spectral()
    >>> embedding = spectral.fit_transform(adjacency)
    >>> embedding.shape
    (5, 2)

    References
    ----------
    Belkin, M. & Niyogi, P. (2003). Laplacian Eigenmaps for Dimensionality Reduction and Data Representation,
    Neural computation.
    """

    def __init__(self, n_components: int = 2, normalized_laplacian=True,
                 regularization: Union[None, float] = 0.01, relative_regularization: bool = True,
                 equalize: bool = False, barycenter: bool = True, normalize: bool = True,
                 solver: Union[str, EigSolver] = 'auto'):
        super(Spectral, self).__init__()

        self.n_components = n_components
        self.normalized_laplacian = normalized_laplacian

        if regularization == 0:
            self.regularization = None
        else:
            self.regularization = regularization
        self.relative_regularization = relative_regularization

        self.equalize = equalize
        self.barycenter = barycenter
        self.normalize = normalize

        if solver == 'halko':
            self.solver: EigSolver = HalkoEig(which='SM')
        elif solver == 'lanczos':
            self.solver: EigSolver = LanczosEig(which='SM')
        else:
            self.solver = solver

        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.regularization_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Spectral':
        """Computes the graph embedding.

        Parameters
        ----------
        adjacency :
              Adjacency matrix of the graph (symmetric matrix).

        Returns
        -------
        self: :class:`Spectral`
        """

        adjacency = check_format(adjacency).asfptype()

        if not is_square(adjacency):
            raise ValueError('The adjacency matrix is not square. See BiSpectral for biadjacency matrices.')

        if not is_symmetric(adjacency):
            raise ValueError('The adjacency matrix is not symmetric.'
                             'Either convert it to a symmetric matrix or use BiSpectral.')

        n = adjacency.shape[0]

        if self.solver == 'auto':
            solver = auto_solver(adjacency.nnz)
            if solver == 'lanczos':
                self.solver: EigSolver = LanczosEig()
            else:
                self.solver: EigSolver = HalkoEig()

        if self.n_components > n - 2:
            warnings.warn(Warning("The dimension of the embedding must be less than the number of nodes - 1."))
            n_components = n - 2
        else:
            n_components = self.n_components + 1

        if self.equalize and (self.regularization is None or self.regularization == 0.) and not is_connected(adjacency):
            raise ValueError("The option 'equalize' is valid only if the graph is connected or with regularization."
                             "Call 'fit' either with 'equalize' = False or positive 'regularization'.")

        if isinstance(self.solver, HalkoEig) and not self.normalized_laplacian:
            raise NotImplementedError("Halko solver is not yet compatible with regular Laplacian."
                                      "Call 'fit' with 'normalized_laplacian' = True or force lanczos solver.")

        weights = adjacency.dot(np.ones(n))
        regularization = self.regularization
        if regularization:
            if self.relative_regularization:
                regularization = regularization * weights.sum() / n ** 2
            weights += regularization * n

        if self.normalized_laplacian:
            # Finding the largest eigenvalues of the normalized adjacency is easier for the solver than finding the
            # smallest eigenvalues of the normalized laplacian.
            weights_inv_sqrt_diag = diag_pinv(np.sqrt(weights))

            if regularization:
                norm_adjacency = NormalizedAdjacencyOperator(adjacency, regularization)
            else:
                norm_adjacency = weights_inv_sqrt_diag.dot(adjacency.dot(weights_inv_sqrt_diag))

            self.solver.which = 'LA'
            self.solver.fit(matrix=norm_adjacency, n_components=n_components)
            eigenvalues = 1 - self.solver.eigenvalues_
            # eigenvalues of the Laplacian in increasing order
            index = np.argsort(eigenvalues)[1:]
            # skip first eigenvalue
            eigenvalues = eigenvalues[index]
            # eigenvectors of the Laplacian, skip first eigenvector
            eigenvectors = np.array(weights_inv_sqrt_diag.dot(self.solver.eigenvectors_[:, index]))

        else:
            if regularization:
                laplacian = LaplacianOperator(adjacency, regularization)
            else:
                weight_diag = sparse.diags(weights, format='csr')
                laplacian = weight_diag - adjacency

            self.solver.which = 'SM'
            self.solver.fit(matrix=laplacian, n_components=n_components)
            eigenvalues = self.solver.eigenvalues_[1:]
            eigenvectors = self.solver.eigenvectors_[:, 1:]

        embedding = eigenvectors.copy()

        if self.equalize:
            eigenvalues_sqrt_inv_diag = diag_pinv(np.sqrt(eigenvalues))
            embedding = eigenvalues_sqrt_inv_diag.dot(embedding.T).T

        if self.barycenter:
            eigenvalues_diag = sparse.diags(eigenvalues)
            subtract = eigenvalues_diag.dot(embedding.T).T
            if not self.normalized_laplacian:
                weights_inv_diag = diag_pinv(weights)
                subtract = weights_inv_diag.dot(subtract)
            embedding -= subtract

        if self.normalize:
            norms_inv_diag = diag_pinv(np.linalg.norm(embedding, axis=1))
            embedding = norms_inv_diag.dot(embedding)

        self.embedding_ = embedding
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        self.regularization_ = regularization

        return self

    def predict(self, adjacency_vector: np.ndarray) -> np.ndarray:
        """Predicts the embedding of a new node, defined by its adjacency vector.

        Parameters
        ----------
        adjacency_vector : array, shape (n,)
            Adjacency vector of a node.

        Returns
        -------
        embedding_vector : array, shape (n_components,)
            Embedding of the node.
        """
        eigenvectors = self.eigenvectors_
        eigenvalues = self.eigenvalues_

        if eigenvectors is None:
            raise ValueError("This instance of Spectral embedding is not fitted yet."
                             " Call 'fit' with appropriate arguments before using this method.")
        else:
            n = eigenvectors.shape[0]

        if adjacency_vector.shape[0] != n:
            raise ValueError('The adjacency vector must be of length equal to the number of nodes.')
        elif not np.all(adjacency_vector >= 0):
            raise ValueError('The adjacency vector must be non-negative.')

        # regularization
        adjacency_vector_reg = adjacency_vector.astype(float)
        if self.regularization_:
            adjacency_vector_reg += self.regularization_

        # projection in the embedding space
        average_vector = adjacency_vector_reg / np.sum(adjacency_vector_reg)
        embedding_vector = average_vector.reshape(1, -1).dot(eigenvectors).ravel()

        if not self.barycenter:
            if self.normalized_laplacian:
                factors = 1 - eigenvalues
            else:
                factors = 1 - eigenvalues / np.sum(adjacency_vector_reg)
            factors_inv_diag = diag_pinv(factors)
            embedding_vector = factors_inv_diag.dot(embedding_vector)

        if self.equalize:
            eigenvalues_sqrt_inv_diag = diag_pinv(np.sqrt(eigenvalues))
            embedding_vector = eigenvalues_sqrt_inv_diag.dot(embedding_vector)

        if self.normalize:
            norm = np.linalg.norm(embedding_vector)
            if norm > 0:
                embedding_vector /= norm

        return embedding_vector
